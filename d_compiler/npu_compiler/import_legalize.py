"""Legalize an imported Relax graph into our NPU primitive op set.

TVM's torch frontend emits high-level ops (relax.nn.silu, softmax, ...) that our
codegen doesn't know. This Relax->Relax pass rewrites them into the primitives we
support: matmul, add/subtract/multiply/divide, sqrt, exp, permute_dims.

Extensible: add a handler per op. Currently:
  - relax.nn.silu  ->  z / (1 + exp(-z))      (= z * sigmoid(z))

TODO for a full Llama block: reduce/mean (rms,softmax) via ones-matmul, rsqrt,
softmax (no max-sub), reshape/strided_slice/concat (head split), RoPE.
"""
import numpy as np
import tvm
from tvm import relax


def _op(name):
    return tvm.ir.Op.get(name)


@relax.expr_functor.mutator
class _Legalizer(relax.PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)

    def visit_call_(self, call):
        call = super().visit_call_(call)
        op = call.op
        if op == _op("relax.nn.silu"):
            return self._silu(call.args[0], call.struct_info)
        if op == _op("relax.negative"):
            return self._negative(call.args[0], call.struct_info)
        if op == _op("relax.power"):
            return self._power(call.args[0], call.args[1])
        if op == _op("relax.rsqrt"):
            return self._rsqrt(call.args[0], call.struct_info)
        if op == _op("relax.mean"):
            return self._mean(call.args[0], call.attrs, call.struct_info)
        if op == _op("relax.nn.softmax"):
            return self._softmax(call.args[0], call.attrs, call.struct_info)
        return call

    # ---- helpers ----
    @staticmethod
    def _shp(sinfo):
        return [int(d) for d in sinfo.shape], sinfo.dtype

    # SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))
    def _silu(self, z, sinfo):
        shp, dt = self._shp(sinfo)
        ones = relax.const(np.ones(shp, dt))
        b = self.builder_
        neg = b.emit(relax.op.subtract(relax.const(np.zeros(shp, dt)), z))
        den = b.emit(relax.op.add(b.emit(relax.op.exp(neg)), ones))
        sig = b.emit(relax.op.divide(ones, den))
        return b.emit(relax.op.multiply(z, sig))

    # -x = 0 - x
    def _negative(self, x, sinfo):
        shp, dt = self._shp(sinfo)
        return self.builder_.emit(relax.op.subtract(relax.const(np.zeros(shp, dt)), x))

    # x ** 2 = x * x  (only integer exponent 2 supported)
    def _power(self, x, exp_const):
        e = float(exp_const.data.numpy()) if isinstance(exp_const, relax.Constant) else None
        assert e == 2.0, f"power exponent {e} unsupported (only 2)"
        return self.builder_.emit(relax.op.multiply(x, x))

    # rsqrt(x) = 1 / sqrt(x)
    def _rsqrt(self, x, sinfo):
        shp, dt = self._shp(sinfo)
        s = self.builder_.emit(relax.op.sqrt(x))
        return self.builder_.emit(relax.op.divide(relax.const(np.ones(shp, dt)), s))

    # mean over last dim of [R,C] (keepdims) = (x @ ones[C,1]) * (1/C)  -> [R,1]
    def _mean(self, x, attrs, sinfo):
        xshp, dt = self._shp(x.struct_info)
        axis = [int(a) % len(xshp) for a in attrs.axis]      # normalize -1 -> last
        assert len(xshp) == 2 and axis == [len(xshp) - 1] and int(attrs.keepdims) == 1, \
            f"mean: only last-axis keepdims 2D (axis={axis})"
        R, C = xshp
        b = self.builder_
        ssum = b.emit(relax.op.matmul(x, relax.const(np.ones((C, 1), dt))))   # [R,1]
        return b.emit(relax.op.multiply(ssum, relax.const(np.full((R, 1), 1.0 / C, dt))))

    # softmax over last dim of [R,C] (no max-subtraction): exp / rowsum-broadcast
    def _softmax(self, x, attrs, sinfo):
        shp, dt = self._shp(sinfo)
        assert len(shp) == 2 and int(attrs.axis) in (-1, 1), f"softmax axis {attrs.axis}"
        R, C = shp
        b = self.builder_
        e = b.emit(relax.op.exp(x))                                          # [R,C]
        ssum = b.emit(relax.op.matmul(e, relax.const(np.ones((C, 1), dt))))  # [R,1]
        denom = b.emit(relax.op.matmul(ssum, relax.const(np.ones((1, C), dt))))  # [R,C]
        return b.emit(relax.op.divide(e, denom))


def legalize(mod, func_name="main"):
    """Rewrite imported high-level ops into our primitive set. Returns new IRModule."""
    mut = _Legalizer(mod)
    gv = mod.get_global_var(func_name)
    new_func = mut.visit_expr(mod[func_name])
    mut.builder_.update_func(gv, new_func)
    return mut.builder_.get()
