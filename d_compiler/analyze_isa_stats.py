#!/usr/bin/env python3
"""Compile the canonical program set for all three families and report
instruction-word counts (total and per role), for before/after comparison
of compiler pipeline changes.

Usage:
  analyze_isa_stats.py --output build/isa_stats_<tag>.json
  analyze_isa_stats.py --compare build/isa_stats_a.json build/isa_stats_b.json
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler import driver, model
from npu_compiler.gemma4_graph import (
    build_gemma4_decode_layer_module,
    build_gemma4_final_norm_module,
    build_gemma4_prefill_layer_module,
)
from npu_compiler.model_spec import gemma4_e2b_spec, qwen3_spec
from npu_compiler.qwen3_graph import (
    build_qwen3_decode_layer_module,
    build_qwen3_final_norm_module,
    build_qwen3_prefill_layer_module,
)

S = 7          # the validated prompt length
CTX = 8        # first decode context

QWEN3_4B = {
    "hidden_size": 2560, "intermediate_size": 9728, "num_hidden_layers": 36,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
    "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
    "hidden_act": "silu", "tie_word_embeddings": True,
}


def programs():
    cfg = model.LLAMA_3_2_3B
    gemma = gemma4_e2b_spec()
    qwen = qwen3_spec(QWEN3_4B)
    return [
        ("llama/prefill_layer", lambda: model.build_v3_prefill_layer_module(cfg, S, return_cache=True)),
        ("llama/decode_ctx8", lambda: model.build_v3_decode_fused_layer_module(cfg, CTX)),
        ("llama/final_norm", lambda: model.build_v3_final_norm_module(cfg, 1)),
        ("gemma/prefill_sliding_owner", lambda: build_gemma4_prefill_layer_module(gemma, 0, S)),
        ("gemma/prefill_full_shared", lambda: build_gemma4_prefill_layer_module(gemma, 19, S)),
        ("gemma/decode_sliding_owner_ctx8", lambda: build_gemma4_decode_layer_module(gemma, 0, CTX)),
        ("gemma/final_norm", lambda: build_gemma4_final_norm_module(gemma, 1)),
        ("qwen/prefill_layer", lambda: build_qwen3_prefill_layer_module(qwen, 0, S)),
        ("qwen/decode_ctx8", lambda: build_qwen3_decode_layer_module(qwen, 0, CTX)),
        ("qwen/final_norm", lambda: build_qwen3_final_norm_module(qwen, 1)),
    ]


BACKENDS = {
    "0818": dict(backend="source-0818"),
    "v09": dict(backend="v09"),
    "v09-w8a16": dict(backend="v09", quant_int8=None),      # filled in measure()
    "v09-w8a8": dict(backend="v09", quant_int8=None, quant_act=True),
}


def dma_stats(words):
    """Count DMA instructions and the bytes they move (4-word form)."""
    count = moved = 0
    index = 0
    while index < len(words):
        opcode = words[index] & 0xFF
        if opcode in (0xA0, 0xA8) and index + 3 < len(words):
            rows, cols = words[index + 3] >> 16, words[index + 3] & 0xFFFF
            count += 1
            moved += rows * cols * 4
            index += 4
            continue
        index += 1
    return count, moved


def measure(target="0818"):
    from npu_compiler.quantize import QUANT_PARAM_NAMES
    options = dict(BACKENDS[target])
    if target.startswith("v09-w8a"):
        options["quant_int8"] = QUANT_PARAM_NAMES
    stats = {}
    for name, build in programs():
        started = time.perf_counter()
        asm, mp = driver.compile_module(build(), reuse=True, **options)
        roles = collections.Counter(tag or "control" for tag in asm.tags)
        dma_count, dma_bytes = dma_stats(asm.words)
        stats[name] = {
            "words": len(asm.words),
            "gbuffer_entries": int(mp.top),
            "dma_instructions": dma_count,
            "dma_bytes": dma_bytes,
            "sram_peak_bytes": int(getattr(asm, "sram_peak_nibbles", 0)) // 2,
            "compile_seconds": round(time.perf_counter() - started, 1),
            "roles": dict(sorted(roles.items(), key=lambda kv: -kv[1])),
        }
        row = stats[name]
        print(f"  {name:32s} {row['words']:9,d} words  "
              f"gbuf {row['gbuffer_entries']:12,d}  "
              f"DMA {row['dma_bytes'] / 2**20:8.1f} MiB  "
              f"SRAM {row['sram_peak_bytes'] / 2**10:8.1f} KiB  "
              f"({row['compile_seconds']}s)", flush=True)
    return stats


def compare(path_a, path_b):
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    print(f"{'program':34s} {'before':>10s} {'after':>10s} {'delta':>9s} "
          f"{'DMA MiB':>10s} {'SRAM KiB':>10s}")
    total_a = total_b = 0
    for name in a:
        wa, wb = a[name]["words"], b.get(name, {}).get("words", 0)
        total_a += wa; total_b += wb
        row = b.get(name, {})
        print(f"{name:34s} {wa:10,d} {wb:10,d} {100.0*(wb-wa)/wa:+8.2f}% "
              f"{row.get('dma_bytes', 0)/2**20:10.1f} "
              f"{row.get('sram_peak_bytes', 0)/2**10:10.1f}")
    print(f"{'TOTAL':34s} {total_a:10,d} {total_b:10,d} "
          f"{100.0*(total_b-total_a)/total_a:+8.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--target", default="0818", choices=sorted(BACKENDS))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    stats = measure(args.target)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
