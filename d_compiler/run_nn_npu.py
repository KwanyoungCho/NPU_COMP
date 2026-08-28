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
    raise SystemExit(f"unknown model family {name!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama", choices=("llama", "qwen3"))
    parser.add_argument("--prompt", default="Hello, NPU compiler!")
    parser.add_argument("--layers", type=int, default=0,
                        help="0 = all layers; smaller values truncate for a fast check")
    parser.add_argument("--expect", type=int, default=358,
                        help="known first generated token id for the default prompt")
    parser.add_argument("--skip-llvm", action="store_true",
                        help="skip the CPU build (saves memory on the full model)")
    return parser.parse_args()


def main():
    args = parse_args()
    family, assets = load_family(args.model)
    config = dict(assets.config)
    if args.layers:
        config["num_hidden_layers"] = args.layers
    input_ids = np.asarray(
        assets.tokenizer(args.prompt, return_tensors="np")["input_ids"][0],
        dtype=np.int64)
    seq = int(input_ids.size)
    print(f"prompt {args.prompt!r} -> {seq} tokens, "
          f"{config['num_hidden_layers']} layers", flush=True)

    mod, params, cfg = family.build_prefill(config, seq)
    lowered = pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=True)(mod)

    started = time.perf_counter()
    weights = []
    for name, param in params:
        key = family.hf_param_map(name, cfg.num_layers)
        if key == "lm_head.weight" and key not in assets.weight_map:
            key = "model.embed_tokens.weight"       # tied embeddings
        value = assets._slice(key, (slice(None),) * len(param.shape))
        weights.append(tvm.nd.array(np.ascontiguousarray(value, np.float16)))
    print(f"  weights loaded: {time.perf_counter() - started:.1f}s", flush=True)

    embeds = assets.embedding([int(i) for i in input_ids]).astype(np.float16)
    cos, sin = family.rope_inputs(cfg, np.arange(seq))
    mask = family.causal_mask(cfg.num_heads, seq)

    started = time.perf_counter()
    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    transformed = vm["prefill_transform_params"]([weights])
    print(f"  transform_params: {time.perf_counter() - started:.1f}s", flush=True)

    reference = None
    if not args.skip_llvm:
        started = time.perf_counter()
        reference = vm["prefill"](*[tvm.nd.array(v)
                                    for v in (embeds, cos, sin, mask)],
                                  *transformed).numpy()
        print(f"  llvm prefill: {time.perf_counter() - started:.1f}s "
              f"-> token {int(np.argmax(reference[-1]))}", flush=True)

    started = time.perf_counter()
    asm, plan = npu_link.compile_program(lowered)
    print(f"  link: {len(asm.words):,} words, {asm.kernel_count} kernels, "
          f"image {plan.top * 2 / 2**20:.1f} MiB "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    planned, _ = npu_memplan.assign_addresses(lowered)
    func = planned["prefill"]
    values = dict(zip([p.name_hint for p in func.params],
                      [embeds, cos, sin, mask]
                      + [t.numpy() for t in transformed]))
    started = time.perf_counter()
    logits, counters = npu_link.run_program(
        asm, plan, func, values, (seq, cfg.vocab_size))
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
    print(json.dumps(result))
    if isinstance(counters, dict):
        print("counters:", json.dumps(counters))
    if args.layers == 0 and token != args.expect:
        raise SystemExit(f"token {token} != expected {args.expect}")


if __name__ == "__main__":
    main()
