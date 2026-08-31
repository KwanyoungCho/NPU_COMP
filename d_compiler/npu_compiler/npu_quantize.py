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


def _qmatmul_te(x, weight, scale):
    """FP16 [M, K] against INT8 [K, N] with FP32 [N] scales -> FP16 [M, N].

    The scale multiplies the accumulated sum, which is where the machine
    applies it -- as each K-tile's product enters the FP32 accumulator.  It is
    constant along K, so the two orders agree.
    """
    m = x.shape[0]
    k = te.reduce_axis((0, x.shape[1]), name="k")
    n = weight.shape[1]
    product = te.compute(
        (m, n),
        lambda i, j: te.sum(x[i, k].astype("float32")
                            * weight[k, j].astype("float32"), axis=k),
        name="matmul")
    return te.compute((m, n),
                      lambda i, j: (product[i, j] * scale[j]).astype(x.dtype),
                      name="dequant")


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
        self.count += 1
        return block.call_te(_qmatmul_te, left, quantized, scale,
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


def reference_w8a16(x, weight):
    """What the pass computes, in numpy, for checking it."""
    w = np.asarray(weight, np.float16).astype(np.float32)
    scale = np.maximum(np.abs(w).max(axis=0), TINY) / INT8_MAX
    quantized = np.clip(np.rint(w / scale[None, :]), -INT8_MAX, INT8_MAX)
    product = np.asarray(x, np.float16).astype(np.float32) @ quantized
    return (product * scale[None, :]).astype(np.float16)
