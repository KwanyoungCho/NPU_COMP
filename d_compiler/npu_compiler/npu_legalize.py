"""NPU-specific legalization overrides for ``relax.transform.LegalizeOps``.

TVM's default lowering of some ops is correct but assumes a machine we do not
have: RMSNorm, for instance, casts to float32 and keeps float32 intermediates,
which our vector unit cannot hold (it stores FP16 and accumulates in FP32
internally).  Registering a replacement here is the standard extension point --
the pipeline stage stays TVM's, only the per-op recipe is ours.
"""
from __future__ import annotations

import math

from tvm import te, tir


def _rms_norm_te(data, weight, epsilon):
    """RMSNorm without float32 intermediates.

    Scaling by 1/sqrt(D) before squaring keeps the sum of squares inside the
    FP16 range for the activation magnitudes these models reach, which is why
    the validated hand-written path does the same (issue V3-020).
    """
    shape = data.shape
    width = int(shape[-1])
    inv_sqrt = tir.const(1.0 / math.sqrt(width), data.dtype)
    scaled = te.compute(shape, lambda *idx: data[idx] * inv_sqrt,
                        name="rms_scaled")
    axis = te.reduce_axis((0, width), name="k")
    squares = te.compute(
        shape[:-1],
        lambda *idx: te.sum(scaled[idx + (axis,)] * scaled[idx + (axis,)],
                            axis=axis),
        name="rms_square_sum")
    eps = tir.const(float(epsilon), data.dtype)
    inv = te.compute(squares.shape,
                     lambda *idx: tir.const(1.0, data.dtype)
                     / te.sqrt(squares[idx] + eps),
                     name="rms_inv")
    normalized = te.compute(shape, lambda *idx: data[idx] * inv[idx[:-1]],
                            name="rms_normalized")
    return te.compute(shape, lambda *idx: normalized[idx] * weight[idx[-1]],
                      name="rms_norm")


def _legalize_rms_norm(bb, call):
    data, weight = call.args[0], call.args[1]
    epsilon = float(call.attrs.epsilon)
    return bb.call_te(_rms_norm_te, data, weight, epsilon,
                      primfunc_name_hint="npu_rms_norm")


def legalize_map():
    """Per-op overrides handed to ``LegalizeOps``."""
    return {"relax.nn.rms_norm": _legalize_rms_norm}
