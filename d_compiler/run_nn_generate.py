#!/usr/bin/env python3
"""Autoregressive generation on the standard path, against the recorded
golden tokens.

Decode programs are compiled once at fixed capacities. Optional cache buckets
reduce the static context scanned by early tokens.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler import npu_generate
from npu_compiler.npu_compilation import compilation

os.environ.setdefault("NPU_V09_TMPDIR", "/data2/chokwans99/npu_tmp")

# The tokens HF generated for the default prompt, read from the recorded
# reference rather than repeated here: the literal that used to sit in this
# file was Qwen3's, so the Llama gate compared against another model's answer
# and reported a failure the compiler never had.
REFERENCES = {"llama": "v3_reference_generate_hello_3.npz"}


def golden(model):
    """-> the recorded HF tokens, or None when there is no reference."""
    name = REFERENCES.get(model)
    if name is None:
        return None
    path = Path(__file__).resolve().parent / "build" / name
    if not path.exists():
        return None
    import numpy as np

    return [int(t) for t in np.load(path)["generated_ids"]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama", choices=("llama",))
    parser.add_argument("--prompt", default="Hello, NPU compiler!")
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=0,
                        help="decode cache capacity (default: prompt+tokens-1;"
                             " a deployment would pick e.g. 512 once)")
    parser.add_argument("--llvm", action="store_true",
                        help="run the identical programs on the llvm build "
                             "instead of the C-model")
    parser.add_argument("--cache-buckets", type=int, nargs="+",
                        help="increasing decode capacities; last must equal --capacity when supplied")
    parser.add_argument("--fusion", choices=("auto", "off", "on"), default="auto")
    parser.add_argument("--params-cache", default=None,
                        help="directory holding this checkpoint's transformed "
                             "weights; computed on first use and reused after")
    return parser.parse_args()


def main():
    args = parse_args()
    from npu_compiler.nn_models import llama as family
    from npu_compiler.assets.llama import Llama32Assets

    assets = Llama32Assets()
    config = family.model_config(assets, args.layers)
    input_ids = [int(i) for i in assets.tokenizer(
        args.prompt, return_tensors="np")["input_ids"][0]]
    print(f"prompt {args.prompt!r} -> {len(input_ids)} tokens, "
          f"{config['num_hidden_layers']} layers, +{args.tokens}", flush=True)

    def report(stage, position, token, words):
        print(f"  {stage} @{position}: token {token} "
              f"({words:,} words)", flush=True)

    started = time.perf_counter()
    runner = npu_generate.llvm_runner if args.llvm else None
    diagnostics = {}
    last_compile_update = 0.
    def compile_progress(event):
        nonlocal last_compile_update
        kind = event['event']
        if kind in ('candidate_begin','candidate_end','entry_compiled','entry_reused'):
            print(f"  compile {kind}: {event.get('entry', event.get('label', ''))} "
                  f"{event.get('status', '')} ({event['elapsed_seconds']:.1f}s)",flush=True)
        elif event['elapsed_seconds']-last_compile_update >= 20:
            last_compile_update = event['elapsed_seconds']
            print(f"  compile {event.get('entry', '')} {event.get('kernel', '')}: "
                  f"{event.get('phase', '')} ({event['elapsed_seconds']:.1f}s)",flush=True)
    with compilation(progress=compile_progress):
        generated = npu_generate.generate(family, config, assets, input_ids,
                                      args.tokens,
                                      capacity=args.capacity or None,
                                      runner=runner, progress=report,
                                      params_cache=args.params_cache,
                                      fusion=args.fusion, cache_buckets=args.cache_buckets,
                                      diagnostics=diagnostics)
    result = {"model": args.model, "backend": "llvm" if args.llvm else "npu",
              "generated": generated,
              "decoded": assets.tokenizer.decode(generated),
              "seconds": round(time.perf_counter() - started, 1)}
    result["execution"] = diagnostics
    expected = golden(args.model) if args.layers == 0 else None
    if expected is not None:
        # The reference holds however many tokens were recorded, which is
        # fewer than --tokens can ask for.  Compare the overlap and say how
        # long it was; comparing the whole list against a shorter reference
        # reports a mismatch for tokens the reference never had an opinion on.
        checked = min(len(expected), len(generated))
        expected = expected[:checked]
        result["expected"] = expected
        result["checked_tokens"] = checked
        result["match"] = generated[:checked] == expected
    print(json.dumps(result))
    if args.layers == 0 and not result.get("match", True):
        raise SystemExit(f"generated {generated[:len(expected)]} != golden "
                         f"{expected}")


if __name__ == "__main__":
    main()
