"""Gemma 4 PLE per-layer-input math with the C-model's exact FP16 semantics.

Used by the offline table generator and as an equivalent fallback when the
precomputed table is not present.  ``compute_rows`` is proven bit-exact
against the source C-model PLE program (tests/test_gemma4_ple.py): every step
stores FP16 like the G-buffer, and the two reductions replicate the C-model's
accumulation order — sequential k inside each 64-wide K tile with float32
tile-MAC, and sequential row reduction.  All replicated steps are elementwise,
so SIMD/threading cannot reorder the per-element sequence.
"""
from __future__ import annotations

import numpy as np


def f16(value):
    return np.asarray(value, dtype=np.float16)


def cmodel_matmul(a, b, tile=64):
    """Matmul with the source C-model's exact accumulation order."""
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
    proj = cmodel_matmul(scaled_embed, projection)
    proj = f16(proj * np.float16(1.0 / np.sqrt(float(D))))
    tok = f16(tok_rows * np.float16(np.sqrt(float(Pd))))
    out = np.empty((scaled_embed.shape[0], proj.shape[1]), dtype=np.float16)
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
