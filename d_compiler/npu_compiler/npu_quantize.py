"""W8A16 as a Relax pass.

Each ``matmul(x, W)`` whose right operand comes from the parameter tuple is
rewritten into three ops: derive a per-output-channel scale from W, quantize W
to INT8 against it, and multiply against the quantized weight with the scale
applied to the accumulated sum.

The rewrite is placed *before* ``LiftTransformParams``, which is what makes it
cheap: the scale and the quantized weight depend only on parameters, so the
standard pass hoists both into the host-side transform function on its own and
nothing extra runs per token.

Granularity is per output channel because the scale has to be constant along
the summation axis -- it factors out of the dot product, which is what lets the
machine apply it as the partial sums enter the accumulator.
"""
from __future__ import annotations

import numpy as np
import tvm
from tvm import relax, te, tir
from tvm.relax.expr_functor import PyExprMutator, mutator

INT8_MAX = 127.0
TINY = 1e-8            # keeps an all-zero channel from dividing by zero


def _scale_te(weight):
    """FP16 [K, N] -> FP32 [N], the per-output-channel step size."""
    k, n = weight.shape
    axis = te.reduce_axis((0, k), name="k")
    absmax = te.compute(
        (n,), lambda j: te.max(te.abs(weight[axis, j].astype("float32")),
                               axis=axis), name="absmax")
    return te.compute(
        (n,), lambda j: te.max(absmax[j], tir.const(TINY, "float32"))
        / tir.const(INT8_MAX, "float32"), name="w_scale")


def _quantize_te(weight, scale):
    """FP16 [K, N] and FP32 [N] -> INT8 [K, N], round-half-to-even."""
    return te.compute(
        weight.shape,
        lambda i, j: te.max(
            te.min(te.round(weight[i, j].astype("float32") / scale[j]),
                   tir.const(INT8_MAX, "float32")),
            tir.const(-INT8_MAX, "float32")).astype("int8"),
        name="w_quant")


def _scale16_te(scale):
    """FP32 [N] -> FP16 [N], the dequant multiplier the device keeps."""
    return te.compute(scale.shape, lambda j: scale[j].astype("float16"),
                      name="w_scale16")


def _qmatmul_te(x, weight, scale16):
    """FP16 [M, K] against INT8 [K, N] with FP16 [N] scales -> FP16 [M, N].

    The weight is dequantized to FP16 rows first and the matmul is the
    ordinary FP16 one -- which is what the machine does: VDEQUANT expands an
    INT8 row, one vector multiply applies the per-channel scale, and the
    validated FP16 gemm chain runs unchanged.  Both steps live in this one
    PrimFunc so LiftTransformParams cannot hoist the dequant and put an FP16
    weight back in global memory, which would undo the halved DMA traffic.
    """
    m = x.shape[0]
    k = te.reduce_axis((0, x.shape[1]), name="k")
    n = weight.shape[1]
    dequantized = te.compute(
        weight.shape,
        lambda i, j: weight[i, j].astype("float16") * scale16[j],
        name="w_dequant")
    # written like the standard FP16 matmul lowering -- fp16 accumulation in
    # the TIR -- so the existing tensorized schedule applies; the machine's
    # gemm supplies the FP32 internal accumulation either way
    return te.compute(
        (m, n),
        lambda i, j: te.sum(x[i, k] * dequantized[k, j], axis=k),
        name="matmul")


def _weight_source(value, lookup, weights):
    """True if a matmul's right operand comes from a parameter.

    Only the shapes nn.Linear produces are followed -- the parameter itself,
    an element of the packed parameter tuple, or either transposed -- so
    anything else keeps the dense path.  ``weights`` holds the function's
    parameters past ``num_input``, which is how a weight is told from a
    runtime input.
    """
    while True:
        if isinstance(value, relax.Var):
            if value in weights:
                return True
            bound = lookup(value)
            if bound is None:
                return False
            value = bound
            continue
        if isinstance(value, relax.TupleGetItem):
            value = value.tuple_value
            continue
        if isinstance(value, relax.Call) and value.op == tvm.ir.Op.get(
                "relax.permute_dims"):
            value = value.args[0]
            continue
        return False


@mutator
class _Quantizer(PyExprMutator):
    def __init__(self, module, min_channels):
        super().__init__(module)
        self.min_channels = min_channels
        self.weights = set()
        self.count = 0

    def _lookup(self, var):
        try:
            return self.lookup_binding(var)
        except Exception:
            return None

    act = False                    # True: W8A8 (per-row activation quant)

    def _quantizable(self, call):
        """The struct info of a weight worth quantizing, or None.

        Read off the original call, whose struct info is populated; the
        rebuilt one the mutator returns has not been normalized yet.
        """
        if call.op != tvm.ir.Op.get("relax.matmul"):
            return None
        weight = call.args[1]
        if not _weight_source(weight, self._lookup, self.weights):
            return None
        info = weight.struct_info
        if (not isinstance(info, relax.TensorStructInfo)
                or info.ndim != 2 or info.dtype != "float16"):
            return None
        if int(info.shape[1]) < self.min_channels:
            return None            # too narrow to pay for the scale vector
        if self.act:
            left = call.args[0].struct_info
            if (not isinstance(left, relax.TensorStructInfo)
                    or left.ndim != 2
                    or int(info.shape[0]) % 64 or int(info.shape[1]) % 64):
                return None        # the INT8 gemm's K/N contract
        return info

    def visit_call_(self, call):
        quantizable = self._quantizable(call)
        call = super().visit_call_(call)
        if quantizable is None:
            return call
        block = self.builder_
        # call_te returns an un-emitted call, so each intermediate is emitted
        # before the next one takes it as an argument
        left, right = block.normalize(call.args[0]), block.normalize(call.args[1])
        scale = block.emit(
            block.call_te(_scale_te, right, primfunc_name_hint="npu_w_scale"))
        quantized = block.emit(
            block.call_te(_quantize_te, right, scale,
                          primfunc_name_hint="npu_w_quantize"))
        if self.act:
            self.count += 1
            return block.call_te(_qmatmul_a8_te, left, quantized, scale,
                                 primfunc_name_hint="npu_qmatmul_a8")
        scale16 = block.emit(
            block.call_te(_scale16_te, scale, primfunc_name_hint="npu_w_scale16"))
        self.count += 1
        return block.call_te(_qmatmul_te, left, quantized, scale16,
                             primfunc_name_hint="npu_qmatmul")


@tvm.transform.module_pass(opt_level=0, name="QuantizeWeightsW8A16")
class QuantizeWeightsW8A16:
    """Rewrite parameter-weight matmuls to INT8 weights with FP32 scales.

    ``min_channels`` leaves narrow matmuls dense: the scale vector costs one
    FP32 per output channel, which is not worth it when there are few.
    """

    def __init__(self, min_channels=64):
        self.min_channels = min_channels

    def transform_module(self, module, context):
        quantizer = _Quantizer(module, self.min_channels)
        for global_var, function in list(module.functions.items()):
            if not isinstance(function, relax.Function):
                continue
            runtime = function.attrs and function.attrs.get("num_input")
            # parameters past num_input are weights; without the attribute the
            # function takes no weights and there is nothing to quantize
            quantizer.weights = (set(function.params[int(runtime):])
                                 if runtime is not None else set())
            if quantizer.weights:
                quantizer.builder_.update_func(
                    global_var, quantizer.visit_expr(function))
        # the builder holds the PrimFuncs the rewrite created, so the module
        # has to come from it rather than being edited in place
        module = quantizer.builder_.get()
        return relax.transform.DeadCodeElimination()(module)


def _qmatmul_a8_te(x, weight, scale):
    """FP16 x against INT8 weights with FP32 scales, activations quantized
    per row on the fly -- the W8A8 kernel.

    Written to match the machine's arithmetic (see quantize.w8a8_reference):
    the row scale is this row's absmax over 127 in fp32, VQUANT rounds
    half-to-even with +-127 saturation, and both scales multiply the
    integer dot product.  The block names are the linker's cue to emit this
    kernel as the validated W8A8 instruction sequence rather than walk it.
    """
    m, k_size = x.shape
    n = weight.shape[1]
    reduce_abs = te.reduce_axis((0, k_size), name="ka")
    absmax = te.compute(
        (m,), lambda i: te.max(te.abs(x[i, reduce_abs].astype("float32")),
                               axis=reduce_abs), name="a_absmax")
    a_scale = te.compute(
        (m,), lambda i: tir.if_then_else(
            absmax[i] == tir.const(0, "float32"),
            tir.const(1, "float32"),
            absmax[i] / tir.const(INT8_MAX, "float32")), name="a_scale")
    a_quant = te.compute(
        x.shape,
        lambda i, j: te.max(
            te.min(te.round(x[i, j].astype("float32") / a_scale[i]),
                   tir.const(INT8_MAX, "float32")),
            tir.const(-INT8_MAX, "float32")).astype("int8"), name="a_quant")
    k = te.reduce_axis((0, k_size), name="k")
    product = te.compute(
        (m, n),
        lambda i, j: te.sum(a_quant[i, k].astype("float32")
                            * weight[k, j].astype("float32"), axis=k),
        name="matmul")
    return te.compute(
        (m, n),
        lambda i, j: (product[i, j] * scale[j] * a_scale[i]).astype("float16"),
        name="dequant")


@tvm.transform.module_pass(opt_level=0, name="QuantizeW8A8")
class QuantizeW8A8:
    """W8A16 plus per-row dynamic activation quantization on the device.

    Weights quantize offline exactly as in W8A16; the FP32 scale reaches the
    device (the matmul's scale registers read FP32), and each rewritten
    matmul becomes the single W8A8 kernel.  The 64-multiple K/N contract of
    the INT8 gemm gates which matmuls are rewritten.
    """

    def __init__(self, min_channels=64):
        self.min_channels = min_channels

    def transform_module(self, module, context):
        quantizer = _Quantizer(module, self.min_channels)
        quantizer.act = True
        for global_var, function in list(module.functions.items()):
            if not isinstance(function, relax.Function):
                continue
            runtime = function.attrs and function.attrs.get("num_input")
            quantizer.weights = (set(function.params[int(runtime):])
                                 if runtime is not None else set())
            if quantizer.weights:
                quantizer.builder_.update_func(
                    global_var, quantizer.visit_expr(function))
        module = quantizer.builder_.get()
        return relax.transform.DeadCodeElimination()(module)


def reference_w8a16(x, weight):
    """What the pass computes, in numpy, for checking it."""
    w = np.asarray(weight, np.float16).astype(np.float32)
    scale = np.maximum(np.abs(w).max(axis=0), TINY) / INT8_MAX
    quantized = np.clip(np.rint(w / scale[None, :]), -INT8_MAX, INT8_MAX)
    dequantized = (quantized * scale[None, :].astype(np.float16)
                   .astype(np.float32)).astype(np.float16)
    product = (np.asarray(x, np.float16).astype(np.float32)
               @ dequantized.astype(np.float32))
    return product.astype(np.float16)
