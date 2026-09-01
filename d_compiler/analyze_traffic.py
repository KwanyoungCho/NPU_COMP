#!/usr/bin/env python3
"""Global<->SRAM traffic as a function of prompt length.

Every gate we run is seq=7, where the machine reads each weight exactly once
and activation round trips are ~1.6% of traffic.  Neither of the two SRAM
questions -- do we re-read weights, and do activations bounce through global
between kernels -- is visible at that point, because both scale with the
prompt and one of them is at its floor.  This sweeps the prompt length and
separates the two terms so the design can be argued from numbers.

Traffic is counted from the linked program itself, not from a run: every DMA
instruction carries the rows and columns it moves, so summing them is exact
and costs one link instead of a C-model execution.  ``--verify`` links the
whole model and checks the total against the C-model's own counter.

Each transfer is attributed by its global address: parameters are weights,
everything else is an activation the previous kernel wrote and this one is
reading back.

Writes JSON for report/figs/0901/plot_traffic.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler import model, npu_legalize, npu_link
from npu_compiler import tvm_pipeline as pipeline
from npu_compiler.isa_v09 import DMA_WORDS, OP_GLOAD, OP_GSTORE
from npu_compiler.nn_models import llama

CELL = 4                     # a DMA moves whole 32-bit cells


def layer_config(layers=1, vocab=256):
    """Real Llama 3.2 3B dimensions; the vocabulary is shrunk unless asked for
    so that lm_head does not swamp the per-layer picture."""
    cfg = model.LLAMA_3_2_3B
    return dict(hidden_size=cfg.D, intermediate_size=cfg.F,
                num_hidden_layers=layers, num_attention_heads=cfg.H,
                num_key_value_heads=cfg.KV, head_dim=cfg.HD,
                vocab_size=vocab, rms_norm_eps=1e-5, rope_theta=500000.0)


def _ranges(func, plan):
    """[(start byte, end byte, kind)] for every placed tensor."""
    parameters = {param.name_hint for param in func.params}
    spans = []
    for name, address in plan.address.items():
        nbytes = plan.nbytes.get(name)
        if not nbytes:
            continue
        start = address * plan.unit_bytes
        kind = "weight" if name in parameters else "activation"
        spans.append((start, start + nbytes, kind))
    spans.sort()
    return spans


def _classify(spans, address):
    """Which tensor kind lives at this global byte address."""
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


def traffic(words, spans):
    """Sum the bytes every DMA instruction in the stream moves."""
    totals = {"load": {}, "store": {}}
    counts = {"load": 0, "store": 0}
    index, total = 0, len(words)
    while index < total:
        opcode = words[index] & 0xFF
        if opcode not in (OP_GLOAD, OP_GSTORE):
            index += 1
            continue
        direction = "load" if opcode == OP_GLOAD else "store"
        address = int(words[index + 1]) * CELL
        rows = int(words[index + 3]) >> 16
        cols = int(words[index + 3]) & 0xFFFF
        kind = _classify(spans, address)
        totals[direction][kind] = totals[direction].get(kind, 0) + \
            rows * cols * CELL
        counts[direction] += 1
        index += DMA_WORDS
    return totals, counts


def measure(seq, layers=1, vocab=256):
    mod, _, _ = llama.build_prefill(layer_config(layers, vocab), seq=seq)
    lowered = pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=True)(mod)
    started = time.perf_counter()
    try:
        asm, plan = npu_link.compile_program(lowered, "prefill")
    except npu_link.LinkError as error:
        # not a measurement failure: the current schedule stages whole padded
        # operands, so past some prompt length a kernel simply does not fit,
        # which is itself the result
        return {"seq": seq, "layers": layers, "vocab": vocab,
                "fits": False, "error": str(error),
                "link_seconds": round(time.perf_counter() - started, 1)}
    seconds = time.perf_counter() - started
    spans = _ranges(npu_link.npu_memplan.assign_addresses(lowered, "prefill")[0]
                    ["prefill"], plan)
    totals, counts = traffic(asm.words, spans)
    peak = max(plan.sram_peak.values()) if plan.sram_peak else 0
    return {
        "seq": seq, "layers": layers, "vocab": vocab, "fits": True,
        "words": len(asm.words), "link_seconds": round(seconds, 1),
        "load": totals["load"], "store": totals["store"],
        "transfers": counts,
        "sram_peak_bytes": peak // 2,
        "sram_peak_kernel": max(plan.sram_peak, key=plan.sram_peak.get)
        if plan.sram_peak else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, nargs="+",
                        default=[7, 32, 64, 128, 256, 512])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] /
                    "report" / "figs" / "0901" / "traffic.json"))
    parser.add_argument("--verify", action="store_true",
                        help="also link the whole 28-layer model and print the "
                             "total, to check against the C-model counter")
    args = parser.parse_args()

    results = []
    for seq in args.seq:
        result = measure(seq, args.layers)
        if not result["fits"]:
            print(f"  seq {seq:4d}: DOES NOT FIT -- {result['error']}",
                  flush=True)
            results.append(result)
            continue
        loaded = sum(result["load"].values())
        stored = sum(result["store"].values())
        weight = result["load"].get("weight", 0)
        print(f"  seq {seq:4d}: {result['words']:>12,} words | "
              f"load {loaded/1e6:8.1f} MB (weight {weight/1e6:7.1f}) | "
              f"store {stored/1e6:6.1f} MB | "
              f"SRAM peak {result['sram_peak_bytes']/1024/1024:.2f} MiB | "
              f"link {result['link_seconds']}s", flush=True)
        results.append(result)

    if args.verify:
        full = measure(7, layers=28, vocab=128256)
        loaded = sum(full["load"].values())
        print(f"  full 28 layers, seq 7: {full['words']:,} words, "
              f"{loaded/1e9:.3f} GB loaded "
              f"({loaded/CELL:,.0f} cells) -- the C-model counted "
              f"1,619,976,421 cells on this program")
        results.append(dict(full, label="verify"))

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
