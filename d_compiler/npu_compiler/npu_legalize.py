"""NPU-specific legalization overrides for ``relax.transform.LegalizeOps``.

TVM's default lowering of some ops is correct but assumes a machine we do not
have: RMSNorm, for instance, casts to float32 and keeps float32 intermediates,
which our vector unit cannot hold (it stores FP16 and accumulates in FP32
internally).  Registering a replacement here is the standard extension point --
the pipeline stage stays TVM's, only the per-op recipe is ours.
"""
from __future__ import annotations

import math

from tvm import te, tir, topi
from .v09_contract import NUMERIC_POLICY


def _rms_norm_te(data, weight, epsilon):
    """Scaled RMSNorm with FP16 storage and FP32 hardware reductions.

    a=max(max(abs(x)), sqrt(eps)), y=x/a gives
    y/sqrt(mean(y*y)+(sqrt(eps)/a)**2). Both ratios are bounded by one.
    Scaling before the square also bounds the stored sum independently of D.
    This is an explicit approximation, including each FP16 store boundary.
    """
    shape = data.shape
    width = int(shape[-1])
    sqrt_eps = tir.const(math.sqrt(epsilon), data.dtype)
    reduce_max = te.reduce_axis((0, width), name="km")
    absmax = te.compute(shape[:-1], lambda *idx: te.max(
        te.abs(data[idx + (reduce_max,)]), axis=reduce_max), name="rms_absmax")
    divisor = te.compute(shape[:-1], lambda *idx: te.max(absmax[idx], sqrt_eps),
                         name="rms_divisor")
    unit = te.compute(shape, lambda *idx: data[idx] / divisor[idx[:-1]],
                      name="rms_unit")
    inv_sqrt = tir.const(1.0 / math.sqrt(width), data.dtype)
    scaled = te.compute(shape, lambda *idx: unit[idx] * inv_sqrt,
                        name="rms_scaled")
    axis = te.reduce_axis((0, width), name="k")
    squared = te.compute(shape, lambda *idx: scaled[idx] * scaled[idx], name="rms_squared")
    sum32 = te.compute(
        shape[:-1],
        lambda *idx: te.sum(squared[idx + (axis,)].astype("float32"),
                            axis=axis),
        name="rms_sum32")
    squares = te.compute(shape[:-1], lambda *idx: sum32[idx].astype("float16"),
                         name="rms_square_sum")
    ratio = te.compute(squares.shape, lambda *idx: sqrt_eps / divisor[idx],
                       name="rms_eps_ratio")
    inv = te.compute(squares.shape,
                     lambda *idx: tir.const(1.0, data.dtype)
                     / te.sqrt(squares[idx] + ratio[idx] * ratio[idx]),
                     name="rms_inv")
    normalized = te.compute(shape, lambda *idx: unit[idx] * inv[idx[:-1]],
                            name="rms_normalized")
    return te.compute(shape, lambda *idx: normalized[idx] * weight[idx[-1]],
                      name="rms_norm")


def _legalize_rms_norm(bb, call):
    data, weight = call.args[0], call.args[1]
    info = data.struct_info
    axes = [int(axis) for axis in call.attrs.axes]
    if (info.ndim < 1 or len(axes) != 1
            or axes[0] not in (-1, info.ndim - 1)
            or weight.struct_info.ndim != 1):
        raise ValueError("v09 RMSNorm legalization supports only the last axis "
                         "with a one-dimensional weight")
    if info.dtype != "float16" or weight.struct_info.dtype != "float16":
        raise ValueError("v09 RMSNorm legalization requires FP16 data and weight")
    epsilon = float(call.attrs.epsilon)
    if not math.isfinite(epsilon) or not 2.0**-48 <= epsilon <= 65504.0**2:
        raise ValueError("v09 RMSNorm epsilon must have a finite positive FP16 "
                         "square root (2^-48 <= epsilon <= 65504^2)")
    if (info.shape is None or not isinstance(info.shape[-1], tir.IntImm)
            or int(info.shape[-1]) <= 0):
        raise ValueError("v09 RMSNorm requires a static positive last dimension")
    return bb.call_te(_rms_norm_te, data, weight, epsilon,
                      primfunc_name_hint="npu_rms_norm",
                      primfunc_attrs={"npu.numeric_policy": NUMERIC_POLICY,
                                      "npu.rms_algorithm": "scaled_absmax_v1"})


def _legalize_softmax(bb, call):
    """TOPI stable softmax within the FP16 sum's representable range.

    exp(x-max(x)) <= 1 for finite logits. A last-axis size <= 65504
    therefore bounds the stored sum. Longer rows require a different
    reduction/storage contract, even though vlen itself allows 65535.
    """
    data = call.args[0]
    info = data.struct_info
    axis = int(call.attrs.axis)
    if info.dtype != "float16":
        raise ValueError("v09 softmax requires FP16 storage")
    if info.ndim < 1 or axis not in (-1, info.ndim - 1):
        raise ValueError("v09 softmax supports only the last axis")
    if (info.shape is None or not isinstance(info.shape[-1], tir.IntImm)
            or not 0 < int(info.shape[-1]) <= 65504):
        raise ValueError("v09 softmax requires static axis length 1..65504 "
                         "to bound its FP16 sum")
    return bb.call_te(_softmax_te, data, primfunc_name_hint="softmax")


def _softmax_te(data):
    shape = data.shape
    k = te.reduce_axis((0, shape[-1]), name="k")
    maximum = te.compute(shape[:-1], lambda *i: te.max(data[i + (k,)], axis=k), name="T_softmax_maxelem")
    exponent = te.compute(shape, lambda *i: te.exp(data[i] - maximum[i[:-1]]), name="T_softmax_exp")
    total32 = te.compute(shape[:-1], lambda *i: te.sum(exponent[i + (k,)].astype("float32"), axis=k), name="softmax_sum32")
    total = te.compute(shape[:-1], lambda *i: total32[i].astype("float16"), name="T_softmax_expsum")
    return te.compute(shape, lambda *i: exponent[i] / total[i[:-1]], name="T_softmax_norm")


def _matmul_te(a, b):
    import numpy as np
    ashape, bshape = tuple(map(int, a.shape)), tuple(map(int, b.shape))
    if len(ashape) < 2 or len(bshape) < 2:
        raise ValueError("v09 matmul requires rank >= 2; reshape vectors to matrices explicitly")
    batch = np.broadcast_shapes(ashape[:-2], bshape[:-2])
    shape = batch + (ashape[-2], bshape[-1])
    axis = te.reduce_axis((0, ashape[-1]), name="k")
    def indices(idx, operand):
        leading = idx[len(batch) - len(operand[:-2]):len(batch)]
        return tuple(0 if size == 1 else i for i, size in zip(leading, operand[:-2]))
    result = te.compute(shape, lambda *idx: te.sum(
        a[indices(idx, ashape) + (idx[-2], axis)].astype("float32")
        * b[indices(idx, bshape) + (axis, idx[-1])].astype("float32"), axis=axis), name="matmul")
    return topi.cast(result, "float16")


def _legalize_matmul(bb, call):
    if (call.args[0].struct_info.dtype == call.args[1].struct_info.dtype == "float16"
            and str(call.attrs.out_dtype) in ("", "float16")):
        return bb.call_te(_matmul_te, *call.args, primfunc_name_hint="matmul",
                          primfunc_attrs={"npu.numeric_policy": NUMERIC_POLICY})
    from tvm.relax.transform.legalize_ops.linear_algebra import _matmul
    return _matmul(bb, call)


def _legalize_sum(bb, call):
    data = call.args[0]
    info = data.struct_info
    axes = list(call.attrs.axis) if call.attrs.axis is not None else list(range(info.ndim))
    if info.dtype != "float16":
        from tvm.relax.transform.legalize_ops.statistical import _sum
        return _sum(bb, call)
    return bb.call_te(_sum_te, data, axes, bool(call.attrs.keepdims), primfunc_name_hint="sum")


def _sum_te(x, axes, keep):
    rank = len(x.shape)
    normalized = sorted({int(a) % rank for a in axes})
    reduced = {a: te.reduce_axis((0, x.shape[a]), name=f"k{a}") for a in normalized}
    shape = tuple(1 if a in reduced else size for a, size in enumerate(x.shape)) if keep else tuple(
        size for a, size in enumerate(x.shape) if a not in reduced)
    def value(*idx):
        remaining = iter(idx)
        source = []
        for a in range(rank):
            item = next(remaining) if keep or a not in reduced else None
            source.append(reduced[a] if a in reduced else item)
        return te.sum(x[tuple(source)].astype("float32"), axis=list(reduced.values()))
    result = te.compute(shape, value, name="sum32")
    return topi.cast(result, "float16")


def _legalize_mean(bb, call):
    data = call.args[0]
    if data.struct_info.dtype != "float16":
        from tvm.relax.transform.legalize_ops.statistical import _mean
        return _mean(bb, call)
    axes = list(call.attrs.axis) if call.attrs.axis is not None else list(range(data.struct_info.ndim))
    def compute(x):
        axes_normalized = {int(a) % len(x.shape) for a in axes}
        count = math.prod(int(x.shape[a]) for a in axes_normalized)
        summed = _sum_te(x, axes, bool(call.attrs.keepdims))
        return topi.multiply(summed, tir.const(1.0 / count, "float16"))
    return bb.call_te(compute, data, primfunc_name_hint="mean")


def legalize_map(numeric_policy=NUMERIC_POLICY):
    """Per-op overrides handed to ``LegalizeOps``."""
    if numeric_policy != NUMERIC_POLICY:
        raise ValueError(f"unsupported v09 numeric policy {numeric_policy!r}; "
                         f"available: {NUMERIC_POLICY}")
    from .npu_views import legalize_split
    return {"relax.matmul": _legalize_matmul, "relax.split": legalize_split,
            "relax.sum": _legalize_sum, "relax.mean": _legalize_mean,
            "relax.nn.rms_norm": _legalize_rms_norm,
            "relax.nn.softmax": _legalize_softmax}
