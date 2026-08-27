#!/usr/bin/env python3
"""S0 gate: the standard-frontend Llama definition, built for llvm, must
reproduce the known first generated token on the validated prompt.

Validates the frontend + standard pass pipeline independently of any NPU
codegen: nn.Module -> export_tvm -> relax.build(target="llvm") -> VM.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler.nn_models import llama
from npu_compiler.v3_model import Llama32Assets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, NPU compiler!")
    parser.add_argument("--layers", type=int, default=0,
                        help="0 = all layers; smaller values truncate for a fast check")
    parser.add_argument("--expect", type=int, default=358,
                        help="known first generated token id for the default prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    assets = Llama32Assets()
    config = dict(assets.config)
    if args.layers:
        config["num_hidden_layers"] = args.layers
    input_ids = np.asarray(
        assets.tokenizer(args.prompt, return_tensors="np")["input_ids"][0],
        dtype=np.int64)
    seq = int(input_ids.size)
    print(f"prompt {args.prompt!r} -> {seq} tokens, "
          f"{config['num_hidden_layers']} layers", flush=True)

    started = time.perf_counter()
    mod, params, cfg = llama.build_prefill(config, seq)
    print(f"  export_tvm: {len(params)} params "
          f"({time.perf_counter() - started:.1f}s)", flush=True)

    started = time.perf_counter()
    ex = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    print(f"  relax.build(llvm): {time.perf_counter() - started:.1f}s", flush=True)

    started = time.perf_counter()
    plist = []
    for name, param in params:
        key = llama.hf_param_map(name, cfg.num_layers)
        if key == "lm_head.weight" and key not in assets.weight_map:
            key = "model.embed_tokens.weight"       # tied embeddings
        value = assets._slice(key, (slice(None),) * len(param.shape))
        if list(value.shape) != list(param.shape):
            raise SystemExit(f"{name}: checkpoint {value.shape} != spec {param.shape}")
        plist.append(tvm.nd.array(np.ascontiguousarray(value, dtype=np.float16)))
    print(f"  weights loaded: {time.perf_counter() - started:.1f}s", flush=True)

    embeds = assets.embedding([int(i) for i in input_ids]).astype(np.float16)
    cos, sin = llama.rope_inputs(cfg, np.arange(seq))
    mask = llama.causal_mask(cfg.num_heads, seq)

    started = time.perf_counter()
    logits = vm["prefill"](*[tvm.nd.array(v) for v in (embeds, cos, sin, mask)],
                           plist).numpy()
    print(f"  prefill: {time.perf_counter() - started:.1f}s", flush=True)

    token = int(np.argmax(logits[0]))
    text = assets.tokenizer.decode([token])
    ok = args.layers == 0 and token == args.expect
    print(json.dumps({"first_token": token, "decoded": text,
                      "expected": args.expect if args.layers == 0 else None,
                      "match": ok if args.layers == 0 else None}))
    if args.layers == 0 and not ok:
        raise SystemExit(f"token {token} != expected {args.expect}")


if __name__ == "__main__":
    main()
