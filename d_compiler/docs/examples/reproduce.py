"""Reproduce the small documentation examples; never compile a full Llama.

Run from any directory with the pinned npu-tvm Python. Uses the public build
entry and records actual selected scheduled calls, not a second scheduler.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

COMPILER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMPILER))

import numpy as np
import tvm
from tvm import relax
from npu_compiler import npu_generate, npu_memplan, npu_target, tvm_pipeline
from npu_compiler.device import PROFILES
from npu_compiler.npu_compilation import compilation, reachable_module
from npu_compiler.npu_numeric import tile_order_reference
from npu_compiler.npu_provenance import describe
from npu_compiler.v09_stats import analyze, check_counters, packets
from npu_compiler.isa_v09 import OP_GLOAD, OP_GSTORE, decode_dma


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def graph(shapes, operation, parameter_boundary=None):
    args = [relax.Var(name, relax.TensorStructInfo(shape, "float16"))
            for name, shape in shapes.items()]
    bb = relax.BlockBuilder()
    attrs = {} if parameter_boundary is None else {"num_input": parameter_boundary}
    with bb.function("prefill", args, attrs=attrs):
        with bb.dataflow():
            result = bb.emit_output(operation(*args))
        bb.emit_func_output(result)
    return bb.finalize()


def transform(mod, weights):
    host = npu_generate._host_transform_module(mod, "prefill")
    vm = relax.VirtualMachine(relax.build(host, "llvm"), tvm.cpu())
    return [a.numpy() for a in vm["prefill_transform_params"](weights)]


def pipeline(out):
    out.mkdir(parents=True, exist_ok=True)
    shapes = {"x": (64, 128), "weight": (64, 128), "bias": (64,)}
    mod = graph(shapes, lambda x, w, b: relax.op.nn.silu(relax.op.add(
        relax.op.matmul(x, relax.op.permute_dims(w, [1, 0])), b)), 1)
    def dump(name, ir):
        (out / name).write_text(ir.script() + "\n")
    dump("01_authored_relax.py", mod)
    dump("02_graph_prefix.py", tvm_pipeline.graph_prefix()(mod))
    records, executables, contexts = {}, {}, {}
    for mode in ("off", "auto"):
        start = time.perf_counter()
        with compilation() as context:
            executables[mode] = exe = npu_target.build(mod, fusion=mode)
        contexts[mode] = context
        records[mode] = {"compile_seconds": time.perf_counter()-start,
                         "statistics": analyze(exe.words, exe.plan),
                         "kernel_count": exe.kernel_count}
    plain, chosen = executables["off"], executables["auto"]
    dump("03_unfused_semantic_tir.py", plain.module)
    dump("04_selected_semantic_tir.py", chosen.module)
    dump("05_host_transform.py", npu_generate._host_transform_module(chosen.module, "prefill"))
    planned, _ = npu_memplan.assign_addresses(chosen.module, "prefill")
    dump("06_initial_dram_plan.py", planned)
    rng = np.random.default_rng(914)
    x = rng.normal(0, .2, shapes["x"]).astype("float16")
    w = rng.normal(0, .2, shapes["weight"]).astype("float16")
    b = rng.normal(0, .1, shapes["bias"]).astype("float16")
    weights = [tvm.nd.array(w), tvm.nd.array(b)]
    np.savez(out / "inputs.npz", x=x, weight=w, bias=b)
    outputs, bound = {}, {}
    for mode, exe in executables.items():
        values = npu_generate._bind_inputs(exe.func, {"x": x}, transform(exe.module, weights))
        bound[mode] = values
        actual, counters = exe.run(values)
        check_counters(exe.words, counters)
        outputs[mode] = actual
        records[mode]["counters"] = counters
        records[mode]["word_sha256"] = hashlib.sha256(np.asarray(exe.words,dtype="<u4").tobytes()).hexdigest()
    np.testing.assert_array_equal(outputs["off"].view("uint16"), outputs["auto"].view("uint16"))
    ref_ir = tile_order_reference(reachable_module(plain.module, ["prefill"]))
    dump("08_llvm_reference.py", ref_ir)
    vm = relax.VirtualMachine(relax.build(ref_ir, "llvm"), tvm.cpu())
    reference = vm["prefill"](*[tvm.nd.array(bound["off"][name]) for name in plain.param_names]).numpy()
    np.testing.assert_allclose(outputs["auto"], reference, rtol=.003, atol=5e-5)
    np.savez(out / "outputs.npz", npu=outputs["auto"], unfused=outputs["off"], llvm=reference)
    chosen.export_bundle(out / "bundle", bound["auto"])
    # export_bundle saves joint-memory choices. The single fused kernel uses
    # the ordinary recipe path; retrieve its actual cached result, without
    # scheduling it again or changing the compiler's plan.
    recipe_exports = []
    for name, diagnostics in chosen.plan.schedules.items():
        if any(isinstance(f, tvm.tir.PrimFunc) and str(g.name_hint) == name
               for g, f in chosen.module.functions.items()):
            source = chosen.module[name].without_attr('global_symbol').without_attr('op_pattern')
            matches = [result[0] for bucket in contexts['auto'].recipes.values()
                       for original, result, detail in bucket
                       if tvm.ir.structural_equal(original, source) and detail == diagnostics]
            if matches:
                assert all(tvm.ir.structural_equal(matches[0], m) for m in matches)
                filename = f'07_scheduled_{name}.py'
                dump(filename, matches[0])
                recipe_exports.append(filename)
    assert recipe_exports or list((out / 'bundle').glob('scheduled_*.py'))
    decoded=[]
    for offset, opcode, packet in packets(chosen.words):
        item=dict(word_offset=offset, opcode=f"0x{opcode:02x}", words=[f"0x{int(w):08x}" for w in packet])
        if opcode in (OP_GLOAD,OP_GSTORE):item["dma"]=decode_dma(packet)
        decoded.append(item)
    write_json(out/"09_packets.json",decoded)
    summary=dict(example="SiLU(X[64,128] @ W[64,128].T + bias[64])",seed=914,
        scope="fresh small-example execution, not a full-model benchmark",records=records,
        actual_recipe_exports=recipe_exports,
        runtime_parameters=[dict(name=n,shape=list(a.shape),dtype=str(a.dtype)) for n,a in bound["auto"].items()],
        validation=dict(fused_unfused_bit_exact=True,cmodel_counters_match=True,
            llvm_max_abs=float(np.max(np.abs(outputs["auto"].astype("float32")-reference.astype("float32")))),
            llvm_rtol=.003,llvm_atol=5e-5),provenance=describe())
    write_json(out/"stage_summary.json",summary)
    return summary["validation"]


def memory(out):
    out.mkdir(parents=True,exist_ok=True)
    mod=graph({"x":(64,64),"w":(64,128),"v":(128,64)},
        lambda x,w,v:relax.op.matmul(relax.op.matmul(x,w),v))
    rng=np.random.default_rng(928)
    values={n:rng.normal(0,.1,s).astype("float16") for n,s in
            {"x":(64,64),"w":(64,128),"v":(128,64)}.items()}
    records=[];baseline=None
    for capacity,joint in ((32*1024,False),(32*1024,True),(44*1024,False),(44*1024,True),(64*1024,True)):
        start=time.perf_counter()
        exe=npu_target.build(mod,fusion="off",profile=replace(PROFILES["v09"],
            sram_bytes=capacity,joint_memory=joint))
        elapsed=time.perf_counter()-start
        actual,counters=exe.run(values);check_counters(exe.words,counters)
        if baseline is None:baseline=actual
        np.testing.assert_array_equal(baseline.view("uint16"),actual.view("uint16"))
        record=dict(sram_capacity_bytes=capacity,joint_memory=joint,compile_seconds=elapsed,
            statistics=analyze(exe.words,exe.plan),bit_exact_to_baseline=True,
            counters_match=True,retention=exe.plan.joint_memory,
            dram_compaction=exe.plan.dram_compaction,
            word_sha256=hashlib.sha256(np.asarray(exe.words,dtype="<u4").tobytes()).hexdigest())
        records.append(record)
        print('memory',capacity,joint,record['statistics']['words'],flush=True)
    write_json(out/"measurements.json",dict(scope="fresh controlled capacity/retention experiment",
        seed=928,shape="(64,64) @ (64,128) @ (128,64)",fusion="off",records=records,provenance=describe()))
    return len(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=COMPILER/'build/docs_examples')
    parser.add_argument('--case',choices=('all','pipeline','memory'),default='all')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();done={}
    if args.case in ('all','pipeline'):
        print('pipeline: compiling two small configurations',flush=True)
        done['pipeline']=pipeline(args.out/'pipeline')
    if args.case in ('all','memory'):
        print('memory: compiling five small configurations',flush=True)
        done['memory']=memory(args.out/'memory')
    write_json(args.out/'reproduction.json',dict(status='passed',cases=done,
        seconds=time.perf_counter()-start,full_model_compiled=False))
    print(json.dumps(done),flush=True)


if __name__=='__main__':main()
