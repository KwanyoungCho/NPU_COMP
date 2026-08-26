"""Host-side weight quantization for the v09 W8A16 path.

Format (per-output-channel symmetric INT8, spec section 9):
  * a [K, N] FP16 weight quantizes column-wise: scale[n] = absmax(W[:, n])/127
    (FP32), q[k, n] = clamp(round-nearest-even(W[k, n]/scale[n]), -127, 127).
  * Storage inside the weight's existing memplan region (FP16-element units):
    packed int8 bytes occupy the first K*N/2 elements; the FP32 scale vector
    follows at the next even element (so its SRAM nibble address is 8-aligned).
    Total K*N/2 + pad + 2N elements always fits the original K*N allocation.
"""
from __future__ import annotations

import numpy as np

QUANT_PARAM_NAMES = frozenset(["Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd"])


def packed_layout(offset, k, n):
    """(data_elem, scale_elem) element offsets inside the weight's region."""
    data_elem = int(offset)
    scale_elem = data_elem + (k * n) // 2
    if scale_elem % 2:
        scale_elem += 1
    if scale_elem + 2 * n > data_elem + k * n:
        raise ValueError(f"packed layout overflows the FP16 region for [{k},{n}]")
    return data_elem, scale_elem


def quantize_per_col_int8(weight):
    """FP16 [K, N] -> (q int8 [K, N], scale float32 [N])."""
    w = np.asarray(weight, dtype=np.float16).astype(np.float32)
    if w.ndim != 2:
        raise ValueError(f"per-column quantization expects rank 2, got {w.shape}")
    absmax = np.abs(w).max(axis=0)
    scale = (absmax / np.float32(127)).astype(np.float32)
    scale[scale == 0] = np.float32(1)          # all-zero column: any scale works
    q = np.rint(w / scale[None, :])            # round half to even, as the HW
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def write_packed(raw_bytes, offset, q, scale):
    """Place packed data + scales into a byte view of the FP16 global buffer."""
    k, n = q.shape
    data_elem, scale_elem = packed_layout(offset, k, n)
    raw_bytes[2 * data_elem:2 * data_elem + k * n] = \
        q.reshape(-1).view(np.uint8)
    raw_bytes[2 * scale_elem:2 * scale_elem + 4 * n] = \
        np.ascontiguousarray(scale, dtype="<f4").view(np.uint8)


def w8a16_reference(a_fp16, q, scale, tile=64):
    """Mirror of the simulator's W8A16 arithmetic order, for bit-exact tests.

    Per K-tile: flat-FP32 dot of the tile, times scale[n], accumulated in
    FP32 (the MAC accumulator); FP16 rounding once at save.
    """
    a = np.asarray(a_fp16, dtype=np.float16).astype(np.float32)
    qf = np.asarray(q, dtype=np.float32)
    m, k = a.shape
    n = qf.shape[1]
    out = np.zeros((m, n), dtype=np.float32)
    for k0 in range(0, k, tile):
        part = np.zeros((m, n), dtype=np.float32)
        for row in range(m):
            for col in range(n):
                acc = np.float32(0)
                for kk in range(k0, min(k0 + tile, k)):
                    acc = np.float32(acc + np.float32(a[row, kk] * qf[kk, col]))
                part[row, col] = acc
        out = out + part * scale[None, :]
    return out.astype(np.float16)
