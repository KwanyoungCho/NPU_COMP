#!/usr/bin/env python3
"""S7 gate (frontend half): the standard-frontend Qwen3 definition, built for
llvm, must reproduce the known first generated token.

Validates the model definition and the standard passes independently of any
NPU codegen, the same way run_nn_llama_cpu.py does for Llama.
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

from npu_compiler.nn_models import qwen3
from npu_compiler.qwen3_model import Qwen3Assets

REFERENCE = Path(__file__).resolve().parent / "build" \
    / "qwen3_reference_generate_hello_3.npz"


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
    assets = Qwen3Assets()
    config = dict(assets.config)
    if args.layers:
        config["num_hidden_layers"] = args.layers
    input_ids = np.asarray(
        assets.tokenizer(args.prompt, return_tensors="np")["input_ids"][0],
        dtype=np.int64)
    seq = int(input_ids.size)
    print(f"prompt {args.prompt!r} -> {seq} tokens, "
          f"{config['num_hidden_layers']} layers", flush=True)

    mod, params, cfg = qwen3.build_prefill(config, seq)
    started = time.perf_counter()
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    print(f"  relax.build(llvm): {time.perf_counter() - started:.1f}s", flush=True)

    started = time.perf_counter()
    plist = []
    for name, param in params:
        key = qwen3.hf_param_map(name, cfg.num_layers)
        if key == "lm_head.weight" and key not in assets.weight_map:
            key = "model.embed_tokens.weight"       # tied embeddings
        value = assets._slice(key, (slice(None),) * len(param.shape))
        if list(value.shape) != list(param.shape):
            raise SystemExit(f"{name}: checkpoint {value.shape} != spec {param.shape}")
        plist.append(tvm.nd.array(np.ascontiguousarray(value, np.float16)))
    print(f"  weights loaded: {time.perf_counter() - started:.1f}s", flush=True)

    embeds = assets.embedding([int(i) for i in input_ids]).astype(np.float16)
    cos, sin = qwen3.rope_inputs(cfg, np.arange(seq))
    mask = qwen3.causal_mask(cfg.num_heads, seq)

    started = time.perf_counter()
    logits = vm["prefill"](*[tvm.nd.array(v) for v in (embeds, cos, sin, mask)],
                           plist).numpy()
    print(f"  prefill: {time.perf_counter() - started:.1f}s", flush=True)

    token = int(np.argmax(logits[-1]))
    result = {"first_token": token, "decoded": assets.tokenizer.decode([token])}
    if args.layers == 0:
        result["expected"] = args.expect
        result["match"] = token == args.expect
        if REFERENCE.exists():
            reference = np.load(REFERENCE)["logits"][0].astype(np.float64)
            ours = logits[-1].astype(np.float64)
            result["logits_cosine"] = round(float(
                ours @ reference
                / (np.linalg.norm(ours) * np.linalg.norm(reference))), 7)
    print(json.dumps(result))
    if args.layers == 0 and token != args.expect:
        raise SystemExit(f"token {token} != expected {args.expect}")


if __name__ == "__main__":
    main()
