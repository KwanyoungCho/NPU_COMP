#!/usr/bin/env python3
"""The HF model as input, at prompt lengths past the gates' seq=7.

Every full-depth gate runs at seq 7, and the traffic sweep uses the
hand-written frontend, so the HF path -- torch.export of HF's own forward,
converted by TVM's torch frontend -- has only ever been compiled short.  This
links it at a range of lengths and reports what the memory planning does with
it: words, global traffic split by kind, and how much of the 8 MiB each
kernel ends up using.

Static: one link per point, no C-model run.  Correctness at these lengths is
tests/test_long_prompt.py's job.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler import npu_legalize, npu_link
from npu_compiler import tvm_pipeline as pipeline
from npu_compiler.isa_v09 import DMA_WORDS, OP_GLOAD, OP_GSTORE
from npu_compiler.nn_models import hf

CELL = 4


def _ranges(func, plan):
    """[(start, end, kind)] per placed tensor.

    The HF path binds weights as graph constants rather than passing them as
    parameters, so they are not in ``plan.address`` at all -- classifying by
    parameter name alone files a gigabyte of weight reads under "other".
    ``plan.const_data`` is where they are.
    """
    parameters = {p.name_hint for p in func.params}
    spans = []
    for name, address in plan.address.items():
        nbytes = plan.nbytes.get(name)
        if not nbytes:
            continue
        start = address * plan.unit_bytes
        spans.append((start, start + nbytes,
                      "weight" if name in parameters else "activation"))
    for address, array in plan.const_data:
        start = address * plan.unit_bytes
        spans.append((start, start + array.size * array.dtype.itemsize,
                      "weight"))
    spans.sort()
    return spans


def _classify(spans, address):
    low, high = 0, len(spans) - 1
    while low <= high:
        mid = (low + high) // 2
        start, end, kind = spans[mid]
        if address < start:
            high = mid - 1
        elif address >= end:
            low = mid + 1
        else:
            return kind
    return "other"


def measure(path, seq, layers):
    started = time.perf_counter()
    mod, _, _ = hf.build_prefill({"path": path, "layers": layers}, seq)
    traced = time.perf_counter() - started
    lowered = pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=False)(mod)
    started = time.perf_counter()
    asm, plan = npu_link.compile_program(lowered, "main", peephole=False)
    linked = time.perf_counter() - started

    planned, _ = npu_link.npu_memplan.assign_addresses(lowered, "main")
    spans = _ranges(planned["main"], plan)
    totals, index = {}, 0
    while index < len(asm.words):
        opcode = asm.words[index] & 0xFF
        if opcode in (OP_GLOAD, OP_GSTORE):
            address = int(asm.words[index + 1]) * CELL
            rows = int(asm.words[index + 3]) >> 16
            cols = int(asm.words[index + 3]) & 0xFFFF
            kind = ("store " if opcode == OP_GSTORE else "load ") + \
                _classify(spans, address)
            totals[kind] = totals.get(kind, 0) + rows * cols * CELL
            index += DMA_WORDS
        else:
            index += 1
    peak = max(plan.sram_peak.values()) if plan.sram_peak else 0
    return {
        "seq": seq, "layers": layers, "words": len(asm.words),
        "trace_seconds": round(traced, 1), "link_seconds": round(linked, 1),
        "traffic": totals, "total_bytes": sum(totals.values()),
        "sram_peak_bytes": peak // 2,
        "sram_peak_kernel": max(plan.sram_peak, key=plan.sram_peak.get)
        if plan.sram_peak else None,
        "sram_by_kernel": {k: v // 2 for k, v in plan.sram_peak.items()},
    }


def measure_handwritten(seq, layers):
    """The same shapes through the frontend we wrote by hand, for comparison.

    Same vocabulary as the HF model so the two totals mean the same thing --
    the traffic sweep shrinks the vocabulary to 256 and would not be
    comparable.
    """
    from npu_compiler import model
    from npu_compiler.nn_models import llama
    cfg = model.LLAMA_3_2_3B
    config = dict(hidden_size=cfg.D, intermediate_size=cfg.F,
                  num_hidden_layers=layers, num_attention_heads=cfg.H,
                  num_key_value_heads=cfg.KV, head_dim=cfg.HD,
                  vocab_size=128256, rms_norm_eps=1e-5, rope_theta=500000.0)
    mod, _, _ = llama.build_prefill(config, seq=seq)
    lowered = pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=True)(mod)
    asm, plan = npu_link.compile_program(lowered, "prefill", peephole=False)
    planned, _ = npu_link.npu_memplan.assign_addresses(lowered, "prefill")
    spans = _ranges(planned["prefill"], plan)
    totals, index = {}, 0
    while index < len(asm.words):
        opcode = asm.words[index] & 0xFF
        if opcode in (OP_GLOAD, OP_GSTORE):
            address = int(asm.words[index + 1]) * CELL
            rows = int(asm.words[index + 3]) >> 16
            cols = int(asm.words[index + 3]) & 0xFFFF
            kind = ("store " if opcode == OP_GSTORE else "load ") + \
                _classify(spans, address)
            totals[kind] = totals.get(kind, 0) + rows * cols * CELL
            index += DMA_WORDS
        else:
            index += 1
    peak = max(plan.sram_peak.values()) if plan.sram_peak else 0
    return {"words": len(asm.words), "traffic": totals,
            "total_bytes": sum(totals.values()),
            "sram_peak_bytes": peak // 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None,
                        help="checkpoint directory (default: the Llama assets)")
    parser.add_argument("--seq", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--compare", action="store_true",
                        help="also link the hand-written frontend at the same "
                             "shapes, so the two paths can be read side by side")
    parser.add_argument("--output", default=str(
        Path(__file__).resolve().parents[1] / "report" / "figs" / "0908" /
        "hf_long.json"))
    args = parser.parse_args()
    path = args.path
    if path is None:
        from npu_compiler.v3_model import Llama32Assets
        path = str(Llama32Assets().path)

    results = []
    for seq in args.seq:
        try:
            result = measure(path, seq, args.layers)
        except Exception as error:
            print(f"  seq {seq:5d}: FAILED -- {type(error).__name__}: "
                  f"{str(error).splitlines()[-1][:90]}", flush=True)
            results.append({"seq": seq, "layers": args.layers,
                            "error": str(error)})
            continue
        print(f"  seq {seq:5d}: {result['words']:>12,} words | "
              f"{result['total_bytes']/1e6:8.1f} MB | "
              f"{result['total_bytes']/1e6/seq:5.2f} MB/token | "
              f"SRAM {result['sram_peak_bytes']/2**20:.2f} MiB "
              f"({result['sram_peak_kernel']}) | trace "
              f"{result['trace_seconds']}s link {result['link_seconds']}s",
              flush=True)
        results.append(result)
        if args.compare:
            other = measure_handwritten(seq, args.layers)
            print(f"  {'':5s}  (hand-written: {other['words']:>12,} words | "
                  f"{other['total_bytes']/1e6:8.1f} MB | SRAM "
                  f"{other['sram_peak_bytes']/2**20:.2f} MiB)", flush=True)
            result["handwritten"] = other
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
