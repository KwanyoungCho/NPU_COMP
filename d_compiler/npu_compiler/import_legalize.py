"""Legalize an imported Relax graph into our NPU op set.

TVM's torch frontend emits high-level ops (relax.nn.silu, softmax, power, ...).
This Relax->Relax pass rewrites the ones our codegen can't take directly, and —
where a native 0710 op or a shared builder exists — routes to the SAME lowering
the manual model path uses (npu_compiler.legalize), so import and manual layers
are byte-for-byte the same graph:
  - relax.nn.silu    -> legalize.silu           (native HW activation, kept as relax.nn.silu)
  - relax.nn.softmax -> legalize.softmax_lastdim (STABLE: max-subtraction, report.md §6.1)
  - relax.negative   -> kept as-is              (native sign-inversion 0x16; shared RoPE path)
  - relax.power/rsqrt/mean -> primitive multiply/sqrt/divide/sum (RMSNorm internals)

Reductions (mean/softmax rowsum) and broadcasts are dedicated relax.sum /
relax.broadcast_to ops (NOT ones-matmul), so relax.matmul stays true GEMM
(-> TIR-only backend); codegen lowers sum/broadcast on the matrix engine.
"""
import numpy as np
import tvm
from tvm import relax

from . import legalize as _lg     # module (aliased: this file also defines a `legalize` fn)


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

    # SiLU(z) = z * sigmoid(z): keep the native op (0710 HW activation) via the shared
    # legalize builder — codegen lowers relax.nn.silu to one m_add(+0, act=SiLU) per chunk.
    def _silu(self, z, sinfo):
        shp, _ = self._shp(sinfo)
        return _lg.silu(self.builder_, z, shp[0], shp[1])

    # relax.negative is kept as-is (codegen lowers it to native sign-inversion 0x16,
    # shared with the manual RoPE rotate_half path) — no decomposition needed.

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

    # mean over last dim of [R,C] (keepdims) = sum(x,-1) * (1/C)  -> [R,1]
    # reduce is a dedicated op (relax.sum), NOT a matmul: codegen lowers it
    # efficiently on the matrix engine, keeping matmul-the-op TIR-only.
    def _mean(self, x, attrs, sinfo):
        xshp, dt = self._shp(x.struct_info)
        axis = [int(a) % len(xshp) for a in attrs.axis]      # normalize -1 -> last
        assert len(xshp) == 2 and axis == [len(xshp) - 1] and int(attrs.keepdims) == 1, \
            f"mean: only last-axis keepdims 2D (axis={axis})"
        R, C = xshp
        b = self.builder_
        ssum = b.emit(relax.op.sum(x, axis=[len(xshp) - 1], keepdims=True))   # [R,1]
        return b.emit(relax.op.multiply(ssum, relax.const(np.full((R, 1), 1.0 / C, dt))))

    # softmax over last dim of [R,C]: delegate to the shared STABLE builder
    # (max-subtraction, then exp / rowsum-broadcast / divide) so the import path
    # matches the manual path (report.md §6.1). reduce/broadcast stay dedicated ops.
    def _softmax(self, x, attrs, sinfo):
        shp, _ = self._shp(sinfo)
        assert len(shp) == 2 and int(attrs.axis) in (-1, 1), f"softmax axis {attrs.axis}"
        return _lg.softmax_lastdim(self.builder_, x, shp[0], shp[1], stable=True)


def legalize(mod, func_name="main"):
    """Rewrite imported high-level ops into our primitive set. Returns new IRModule."""
    mut = _Legalizer(mod)
    gv = mod.get_global_var(func_name)
    new_func = mut.visit_expr(mod[func_name])
    mut.builder_.update_func(gv, new_func)
    return mut.builder_.get()
