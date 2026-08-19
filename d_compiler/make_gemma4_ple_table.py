#!/usr/bin/env python3
"""Precompute the Gemma 4 E2B per-layer-input (PLE) table for all vocab tokens.

The table stores exactly what the NPU PLE graph (gemma4_graph.build_gemma4_ple_module)
would compute, using the same FP16 G-buffer step boundaries: FP16 storage after
every op, float32 arithmetic inside each op.  Sampled rows are proven bit-exact
against the actual source C-model program by tests/test_gemma4_ple.py.

Output: [vocab_size, num_layers*ple_dim] FP16 row-major, written next to the
real checkpoint file (ple_table_f16.bin), ~4.7GB.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_compiler.gemma4_model import Gemma4Assets


def f16(value):
    return np.asarray(value, dtype=np.float16)


def cmodel_matmul(a, b, tile=64):
    """Matmul with the source C-model's exact accumulation order.

    Products of FP16 values are exact in float32; only addition order matters.
    The C-model sums k sequentially inside each 64-wide K tile into a local
    float32, then MACs tile sums into the accumulator.  Every step here is an
    elementwise op over [rows, cols], so SIMD/threading cannot reorder the
    per-element sequence; torch is used only for multithreaded elementwise.
    """
    import torch
    a32 = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    b32 = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
    rows, inner = a32.shape
    cols = b32.shape[1]
    acc = torch.zeros((rows, cols), dtype=torch.float32)
    product = torch.empty((rows, cols), dtype=torch.float32)
    for start in range(0, inner, tile):
        local = torch.zeros((rows, cols), dtype=torch.float32)
        for k in range(start, min(start + tile, inner)):
            torch.mul(a32[:, k:k + 1], b32[k:k + 1, :], out=product)
            local += product
        acc += local
    return f16(acc.numpy())


def cmodel_rowsum(x):
    """Sequential last-axis float32 sum (vector reduce order), FP16 result."""
    x32 = np.asarray(x, dtype=np.float32)
    acc = np.zeros(x32.shape[:-1] + (1,), dtype=np.float32)
    for index in range(x32.shape[-1]):
        acc = acc + x32[..., index:index + 1]
    return f16(acc)


def compute_rows(spec, scaled_embed, tok_rows, projection, norm_weight):
    """FP16 step emulation of the PLE graph for a batch of token rows."""
    D = spec.hidden_size
    Pd = spec.layers[0].ple_dim
    eps = np.float16(spec.rms_norm_eps)
    rows = scaled_embed.shape[0]
    proj = cmodel_matmul(scaled_embed, projection)
    proj = f16(proj * np.float16(1.0 / np.sqrt(float(D))))
    tok = f16(tok_rows * np.float16(np.sqrt(float(Pd))))
    out = np.empty((rows, proj.shape[1]), dtype=np.float16)
    inv_sqrt_p = np.float16(1.0 / np.sqrt(float(Pd)))
    inv_sqrt_2 = np.float16(1.0 / np.sqrt(2.0))
    for index in range(spec.num_layers):
        begin, end = index * Pd, (index + 1) * Pd
        x = proj[:, begin:end]
        scaled = f16(x * inv_sqrt_p)
        sq = f16(scaled.astype(np.float32) * scaled.astype(np.float32))
        mean = cmodel_rowsum(sq)
        rms = f16(np.sqrt(f16(mean + eps).astype(np.float32)))
        inv = f16(1.0 / rms.astype(np.float32))
        xn = f16(x.astype(np.float32) * inv.astype(np.float32))
        normed = f16(xn.astype(np.float32) * norm_weight.astype(np.float32))
        combined = f16(normed.astype(np.float32) + tok[:, begin:end].astype(np.float32))
        out[:, begin:end] = f16(combined.astype(np.float32) * inv_sqrt_2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--limit", type=int, help="only this many tokens (debug)")
    args = parser.parse_args()

    assets = Gemma4Assets()
    spec = assets.spec
    vocab = spec.vocab_size if args.limit is None else int(args.limit)
    width = spec.num_layers * spec.layers[0].ple_dim
    output = assets.ple_table_path()
    projection = assets.per_layer_model_projection()
    norm_weight = assets.per_layer_projection_norm()
    embed_scale = np.float16(np.sqrt(float(spec.hidden_size)))

    started = time.perf_counter()
    with open(output, "wb") as file:
        for begin in range(0, vocab, args.chunk):
            end = min(vocab, begin + args.chunk)
            span = (slice(begin, end), slice(None))
            scaled_embed = f16(assets._slice("embed_tokens.weight", span) * embed_scale)
            tok_rows = assets._slice("embed_tokens_per_layer.weight", span)
            rows = compute_rows(spec, scaled_embed, tok_rows, projection, norm_weight)
            if not np.isfinite(rows.astype(np.float32)).all():
                raise RuntimeError(f"non-finite PLE rows in [{begin},{end})")
            file.write(rows.tobytes())
            print(f"  rows {end}/{vocab}  {time.perf_counter() - started:.1f}s",
                  flush=True)
    size = output.stat().st_size
    print(f"DONE {output} bytes={size} expected={vocab * width * 2}")


if __name__ == "__main__":
    main()
