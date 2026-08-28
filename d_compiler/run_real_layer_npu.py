#!/usr/bin/env python3
"""A decoder layer at real model dimensions, on the C-model, judged against a
float32 reference.

The layer shape is Llama 3.2 3B's, but the weights are random, so this needs
no checkpoint and no tokenizer -- it exercises exactly the sizes that the
tile-scale tests in tests/test_npu_link.py cannot.

Both builds are scored against a plain float32 numpy forward pass, because
they disagree for a reason worth seeing: TVM's float16 matmul accumulates in
float16, while the machine accumulates in float32 internally, so where the
two differ the C-model is the accurate one.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from npu_compiler import npu_legalize, npu_link, npu_memplan
from npu_compiler import tvm_pipeline as pipeline
from npu_compiler.nn_models import llama

os.environ.setdefault("NPU_V09_TMPDIR", "/data2/chokwans99/npu_tmp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=7)
    parser.add_argument("--vocab", type=int, default=256,
                        help="a small vocabulary keeps lm_head cheap; the real "
                             "one is exercised by run_nn_npu.py")
    return parser.parse_args()


def score(name, value, exact):
    a, b = np.asarray(value, np.float64).ravel(), exact.astype(np.float64).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"  {name:5s} vs float32: cosine {cosine:.6f}  "
          f"max|diff| {float(np.abs(a - b).max()):.5f}  "
          f"argmax(last row) {int(np.argmax(np.asarray(value)[-1]))}")
    return cosine


def main():
    args = parse_args()
    cfg = dict(hidden_size=3072, intermediate_size=8192, num_hidden_layers=1,
               num_attention_heads=24, num_key_value_heads=8, head_dim=128,
               vocab_size=args.vocab, rms_norm_eps=1e-5, rope_theta=500000.0)
    mod, params, config = llama.build_prefill(cfg, seq=args.seq)
    rng = np.random.default_rng(5)
    weights = {name: rng.normal(0, 0.02, p.shape).astype(np.float16)
               for name, p in params}
    x = rng.normal(0, 0.5, (args.seq, config.hidden_size)).astype(np.float16)
    cos, sin = llama.rope_inputs(config, np.arange(args.seq))
    mask = llama.causal_mask(config.num_heads, args.seq)

    from test_nn_frontend import _reference
    exact = _reference(config, weights, x, cos, sin, mask)

    lowered = pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=True)(mod)
    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    order = [tvm.nd.array(weights[name]) for name, _ in params]
    transformed = vm["prefill_transform_params"]([order])
    reference = vm["prefill"](*[tvm.nd.array(v) for v in (x, cos, sin, mask)],
                              *transformed).numpy()

    started = time.perf_counter()
    asm, plan = npu_link.compile_program(lowered)
    print(f"linked {len(asm.words):,} words, {asm.kernel_count} kernels, "
          f"image {plan.top * 2 / 2**20:.1f} MiB "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    planned, _ = npu_memplan.assign_addresses(lowered)
    func = planned["prefill"]
    values = dict(zip([p.name_hint for p in func.params],
                      [x, cos, sin, mask] + [t.numpy() for t in transformed]))
    started = time.perf_counter()
    got, _ = npu_link.run_program(asm, plan, func, values, reference.shape)
    print(f"c-model run {time.perf_counter() - started:.0f}s", flush=True)

    npu = score("npu", got, exact)
    score("llvm", reference, exact)
    if npu < 0.9999:
        raise SystemExit(f"npu cosine {npu:.6f} below tolerance")


if __name__ == "__main__":
    main()
