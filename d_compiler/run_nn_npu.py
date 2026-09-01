#!/usr/bin/env python3
"""S7 gate: a real checkpoint, compiled through the standard pipeline, linked
to v09 instructions, and executed on the C-model.

The same lowered module is also built for llvm, so the run is judged twice:
against the CPU build of the identical IR (numerical agreement) and against
the known first generated token (end-to-end agreement with HF).  Note that
the CPU build accumulates float16 matmuls in float16 while the machine
accumulates in float32, so a cosine below 1 there is expected -- see
run_real_layer_npu.py, which scores both against float32.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler import npu_legalize, npu_link, npu_memplan
from npu_compiler import tvm_pipeline as pipeline

os.environ.setdefault("NPU_V09_TMPDIR", "/data2/chokwans99/npu_tmp")


REFERENCES = {
    "llama": "v3_reference_generate_hello_3.npz",
    "qwen3": "qwen3_reference_generate_hello_3.npz",
    "gemma": "gemma4_reference_generate_hello_3.npz",
    "hf": "v3_reference_generate_hello_3.npz",     # same Llama checkpoint
}
GOLDEN_TOKEN = {"llama": 358, "qwen3": 358, "gemma": 108, "hf": 358}


def load_family(name):
    """-> (frontend module, checkpoint assets) for one model family."""
    if name == "llama":
        from npu_compiler.nn_models import llama
        from npu_compiler.v3_model import Llama32Assets
        return llama, Llama32Assets()
    if name == "qwen3":
        from npu_compiler.nn_models import qwen3
        from npu_compiler.qwen3_model import Qwen3Assets
        return qwen3, Qwen3Assets()
    if name == "gemma":
        from npu_compiler.nn_models import gemma
        from npu_compiler.gemma4_model import Gemma4Assets
        return gemma, Gemma4Assets()
    if name == "hf":
        # the HF checkpoint itself is the frontend: transformers builds the
        # model, torch.export traces it, TVM's torch importer converts it
        from npu_compiler.nn_models import hf
        from npu_compiler.v3_model import Llama32Assets
        return hf, Llama32Assets()
    raise SystemExit(f"unknown model family {name!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama",
                        choices=("llama", "qwen3", "gemma", "hf"))
    parser.add_argument("--prompt", default="Hello, NPU compiler!")
    parser.add_argument("--layers", type=int, default=0,
                        help="0 = all layers; smaller values truncate for a fast check")
    parser.add_argument("--expect", type=int, default=None,
                        help="known first generated token id for the default "
                             "prompt (defaults to this family's golden)")
    parser.add_argument("--reference", default=None,
                        help="npz of HF logits to score against (defaults to "
                             "this family's recorded generation reference)")
    parser.add_argument("--params-cache", default=None,
                        help="npz path: save the transformed (e.g. quantized) "
                             "weights on first run, reuse them afterwards -- "
                             "the offline half of quantization")
    parser.add_argument("--skip-llvm", action="store_true",
                        help="skip the CPU build (saves memory on the full model)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.expect is None:
        args.expect = GOLDEN_TOKEN[args.model]
    family, assets = load_family(args.model)
    config = family.model_config(assets, args.layers)
    input_ids = np.asarray(
        assets.tokenizer(args.prompt, return_tensors="np")["input_ids"][0],
        dtype=np.int64)
    seq = int(input_ids.size)
    mod, params, cfg = family.build_prefill(config, seq)
    print(f"prompt {args.prompt!r} -> {seq} tokens, "
          f"{cfg.num_layers} layers", flush=True)

    # the HF-traced frontend bakes its weights in as constants, so there is
    # nothing to lift and no host-side transform to run
    lowered = pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=bool(params))(mod)

    started = time.perf_counter()
    weights = [tvm.nd.array(value)
               for value in family.load_params(assets, params, cfg)]
    print(f"  weights loaded: {time.perf_counter() - started:.1f}s", flush=True)
    entry = "prefill" if params else "main"

    graph_inputs = family.runtime_inputs(assets, cfg, input_ids)
    # take the order from the function signature, not the dict
    order = [param.name_hint for param in lowered[entry].params
             if param.name_hint in graph_inputs]
    if len(order) != len(graph_inputs):
        raise SystemExit(f"inputs {sorted(graph_inputs)} do not match "
                         f"{[p.name_hint for p in lowered[entry].params]}")

    started = time.perf_counter()
    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    cache_path = Path(args.params_cache) if args.params_cache else None
    if cache_path is not None and cache_path.exists():
        loaded = np.load(cache_path)
        transformed = [tvm.nd.array(loaded[key]) for key in loaded.files]
        print(f"  transform_params: reused {len(transformed)} tensors from "
              f"{cache_path}", flush=True)
    else:
        transformed = (vm["prefill_transform_params"]([weights])
                       if params else [])
        if cache_path is not None and transformed:
            np.savez(cache_path, *[t.numpy() for t in transformed])
            print(f"  transform_params: saved to {cache_path}", flush=True)
        print(f"  transform_params: {time.perf_counter() - started:.1f}s",
              flush=True)

    reference = None
    if not args.skip_llvm:
        started = time.perf_counter()
        reference = vm[entry](
            *[tvm.nd.array(graph_inputs[name]) for name in order],
            *transformed).numpy().reshape(seq, cfg.vocab_size)
        print(f"  llvm prefill: {time.perf_counter() - started:.1f}s "
              f"-> token {int(np.argmax(reference[-1]))}", flush=True)

    started = time.perf_counter()
    asm, plan = npu_link.compile_program(lowered, entry)
    print(f"  link: {len(asm.words):,} words, {asm.kernel_count} kernels, "
          f"image {plan.top * 2 / 2**20:.1f} MiB "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    planned, _ = npu_memplan.assign_addresses(lowered, entry)
    func = planned[entry]
    values = dict(zip([p.name_hint for p in func.params],
                      [graph_inputs[name] for name in order]
                      + [t.numpy() for t in transformed]))
    started = time.perf_counter()
    logits, counters = npu_link.run_program(
        asm, plan, func, values, (seq, cfg.vocab_size))
    logits = logits.reshape(seq, cfg.vocab_size)
    print(f"  c-model run: {time.perf_counter() - started:.0f}s", flush=True)

    token = int(np.argmax(logits[-1]))
    result = {"model": args.model, "first_token": token,
              "decoded": assets.tokenizer.decode([token]),
              "words": len(asm.words), "kernels": asm.kernel_count}
    if reference is not None:
        a = logits.astype(np.float64).ravel()
        b = reference.astype(np.float64).ravel()
        result["cosine_vs_llvm"] = round(
            float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 6)
        result["max_abs_diff"] = round(float(np.abs(a - b).max()), 5)
        result["llvm_token"] = int(np.argmax(reference[-1]))
    if args.layers == 0:
        result["expected"] = args.expect
        result["match"] = token == args.expect
        # HF logits are the gate; score them off the NPU result, not the CPU
        # build, which accumulates float16 matmuls in float16
        path = Path(args.reference or
                    Path(__file__).resolve().parent / "build"
                    / REFERENCES[args.model])
        if path.exists():
            hf = np.load(path)["logits"][0].astype(np.float64)
            ours = logits[-1].astype(np.float64)
            result["hf_logits_cosine"] = round(float(
                ours @ hf / (np.linalg.norm(ours) * np.linalg.norm(hf))), 7)
    print(json.dumps(result))
    if isinstance(counters, dict):
        print("counters:", json.dumps(counters))
    if args.layers == 0 and token != args.expect:
        raise SystemExit(f"token {token} != expected {args.expect}")


if __name__ == "__main__":
    main()
