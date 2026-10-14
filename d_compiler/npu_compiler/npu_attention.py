"""Expose high-level BSNH attention as canonical Relax matrix operations."""
import numpy as np
import tvm
from tvm import relax, tir
from tvm.relax.expr_functor import PyExprMutator, mutator
from .npu_errors import LinkError


@mutator
class _Attention(PyExprMutator):
    def visit_call_(self, call):
        call = super().visit_call_(call)
        name = getattr(call.op, "name", "")
        if name not in ("relax.nn.attention", "relax.nn.attention_bias"):
            return call
        q, k, v = call.args[:3]
        if any(x.struct_info.dtype != "float16" or x.struct_info.ndim != 4
               or x.struct_info.shape is None
               or not all(isinstance(d, tir.IntImm) for d in x.struct_info.shape)
               for x in (q, k, v)):
            raise LinkError("v09 attention requires static FP16 BSNH inputs")
        b, s, h, d = map(int, q.struct_info.shape)
        bk, t, hk, dk = map(int, k.struct_info.shape)
        bv, tv, hv, dv = map(int, v.struct_info.shape)
        if (bk, hk, dk) != (b, h, d) or (bv, tv, hv) != (b, t, h):
            raise LinkError("attention batch/head/key dimensions disagree")
        bb, op = self.builder_, relax.op
        emit = bb.emit
        def reshape(x, shape):
            return emit(op.reshape(x, shape))
        q = reshape(emit(op.permute_dims(q, [0, 2, 1, 3])), [b*h, s, d])
        k = reshape(emit(op.permute_dims(k, [0, 2, 3, 1])), [b*h, d, t])
        v = reshape(emit(op.permute_dims(v, [0, 2, 1, 3])), [b*h, t, dv])
        scores = emit(op.matmul(q, k))
        if call.attrs.scale is None:
            # Preserve the FP16 sqrt then divide boundary of TVM 0.19.
            scores = emit(op.divide(scores, relax.const(np.sqrt(np.float16(d)), "float16")))
        else:
            scores = emit(op.multiply(scores, relax.const(float(call.attrs.scale), "float16")))
        if len(call.args) > 3:
            scores = reshape(emit(op.add(reshape(scores, [b,h,s,t]), call.args[3])), [b*h,s,t])
        causal = call.attrs.causal_mask
        window = call.attrs.window_size
        if window is not None and causal is None:
            raise LinkError("v09 sliding-window attention requires an explicit causal mask")
        if causal is None and window is None:
            probs = emit(op.nn.softmax(scores, axis=-1))
        else:
            if causal not in (None, "TopLeft", "BottomRight"):
                raise LinkError(f"unsupported attention causal mask {causal}")
            offset = abs(s-t) if causal == "BottomRight" else 0
            if window is not None and int(window) <= 0:
                raise LinkError("attention window_size must be positive")
            # trilu lowers index predicates, with exact zero masked exponent.
            # It reproduces TVM's causal decomposition without vector boolean
            # loads or an approximation using a finite negative mask.
            masked = emit(op.tril(scores, offset)) if causal else scores
            if window is not None:
                masked = emit(op.triu(masked, offset-int(window)+1))
            maximum = emit(op.max(masked, axis=-1, keepdims=True))
            exps = emit(op.exp(emit(op.subtract(masked, maximum))))
            if causal:
                exps = emit(op.tril(exps, offset))
            if window is not None:
                exps = emit(op.triu(exps, offset-int(window)+1))
            probs = emit(op.divide(exps, emit(op.sum(exps, axis=-1, keepdims=True))))
        result = reshape(emit(op.matmul(probs, v)), [b,h,s,dv])
        return bb.normalize(op.permute_dims(result, [0,2,1,3]))


@tvm.transform.module_pass(opt_level=0, name="CanonicalizeNpuAttention")
class CanonicalizeAttention:
    def transform_module(self, mod, ctx):
        rewrite = _Attention(mod)
        for gv, fn in mod.functions.items():
            if isinstance(fn, relax.Function):
                rewrite.builder_.update_func(gv, rewrite.visit_expr(fn))
        return rewrite.builder_.get()
