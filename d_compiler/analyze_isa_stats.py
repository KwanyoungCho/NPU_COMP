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


def measure():
    stats = {}
    for name, build in programs():
        started = time.perf_counter()
        asm, mp = driver.compile_module(build(), backend="source-0818", reuse=True)
        roles = collections.Counter(tag or "control" for tag in asm.tags)
        stats[name] = {
            "words": len(asm.words),
            "gbuffer_entries": int(mp.top),
            "compile_seconds": round(time.perf_counter() - started, 1),
            "roles": dict(sorted(roles.items(), key=lambda kv: -kv[1])),
        }
        print(f"  {name:32s} {stats[name]['words']:9,d} words  "
              f"gbuf {stats[name]['gbuffer_entries']:12,d}  "
              f"({stats[name]['compile_seconds']}s)", flush=True)
    return stats


def compare(path_a, path_b):
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    print(f"{'program':34s} {'before':>10s} {'after':>10s} {'delta':>9s}")
    total_a = total_b = 0
    for name in a:
        wa, wb = a[name]["words"], b.get(name, {}).get("words", 0)
        total_a += wa; total_b += wb
        print(f"{name:34s} {wa:10,d} {wb:10,d} {100.0*(wb-wa)/wa:+8.2f}%")
    print(f"{'TOTAL':34s} {total_a:10,d} {total_b:10,d} "
          f"{100.0*(total_b-total_a)/total_a:+8.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    stats = measure()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
