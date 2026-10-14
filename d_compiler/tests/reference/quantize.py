"""Independent NumPy oracle for dynamic W8A8 quantization and tile accumulation."""
from __future__ import annotations

import numpy as np


def quantize_rows_int8(rows_fp16):
    """Device-side per-row dynamic quantization, replicated exactly.

    On device: |x| = max(x, -x) stored FP16 (exact for FP16 inputs) ->
    seeded reduce-max -> scale = FP32(absmax)/FP32(127) stored FP32 ->
    VQUANT with RNE and +-127 saturation.
    """
    x = np.asarray(rows_fp16, dtype=np.float16)
    absmax = np.abs(x).max(axis=1).astype(np.float32)
    scale = absmax / np.float32(127)
    scale[scale == 0] = np.float32(1)
    q = np.clip(np.rint(x.astype(np.float32) / scale[:, None]), -127, 127)
    return q.astype(np.int8), scale


def w8a8_reference(a_fp16, q_w, w_scale, tile=64):
    """Mirror of the simulator's W8A8 order: per K-tile exact integer dot ->
    x w_scale[col] x a_scale[row] -> FP32 MAC accumulation -> FP16 at save."""
    q_a, a_scale = quantize_rows_int8(a_fp16)
    qa = q_a.astype(np.int64)
    qw = np.asarray(q_w, dtype=np.int64)
    m, k = qa.shape
    n = qw.shape[1]
    out = np.zeros((m, n), dtype=np.float32)
    for k0 in range(0, k, tile):
        part = (qa[:, k0:k0 + tile] @ qw[k0:k0 + tile]).astype(np.float32)
        out = out + part * w_scale[None, :] * a_scale[:, None]
    return out.astype(np.float16), q_a, a_scale
