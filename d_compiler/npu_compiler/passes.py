"""Relax pass pipeline for the ver.08 (0818) backends.

Family graph builders emit high-level ops (``nn.rms_norm``, ``nn.softmax``,
``nn.gelu_tanh``); ``LowerToNPUPrimitives`` expands them into the NPU primitive
mix using the single-source decompositions in :mod:`legalize`, and the standard
TVM cleanup passes then tidy/fold the result before planning and codegen.
Applied by ``driver.compile_module`` for the 0818/source-0818 backends only;
legacy (0710/hybrid/tir) paths are untouched.
"""
from __future__ import annotations

import numpy as np
import tvm
from tvm import relax

from . import legalize


def _dims(expr):
    return [int(dim) for dim in expr.struct_info.shape]


def _is_ones_constant(expr):
    return isinstance(expr, relax.Constant) and bool(
        np.all(expr.data.numpy() == 1))


def _is_last_axis(axes, ndim):
    return [int(axis) % ndim for axis in axes] == [ndim - 1]


@relax.expr_functor.mutator
class _LowerNPU(relax.PyExprMutator):
    """Expand high-level nn ops into the legalize primitive decompositions.

    The NPU cannot use its native reduce-max (zero-seed bug, V3-003) or native
    GELU (vendor formula, V3-004), and RMSNorm must scale before squaring
    (FP16 overflow, V3-020) — all of that lives in :mod:`legalize` and this
    pass is the one place that applies it.  An all-ones RMSNorm weight (the
    weight-less Gemma V-norm) skips the trailing multiply entirely.
    """

    def visit_call_(self, call):
        call = super().visit_call_(call)
        if not isinstance(call.op, tvm.ir.Op):
            return call
        name = call.op.name
        if name == "relax.nn.rms_norm":
            data, weight = call.args[0], call.args[1]
            rows, cols = _dims(data)
            if not _is_last_axis(call.attrs.axes, 2):
                raise ValueError(f"rms_norm axes {call.attrs.axes} unsupported")
            weight = None if _is_ones_constant(weight) else weight
            return legalize.rms_norm(self.builder_, data, weight, rows, cols,
                                     eps=float(call.attrs.epsilon))
        if name == "relax.nn.softmax":
            data = call.args[0]
            rows, cols = _dims(data)
            if int(call.attrs.axis) % 2 != 1:
                raise ValueError(f"softmax axis {call.attrs.axis} unsupported")
            return legalize.softmax_lastdim(self.builder_, data, rows, cols)
        if name == "relax.nn.gelu_tanh":
            data = call.args[0]
            rows, cols = _dims(data)
            return legalize.gelu_tanh(self.builder_, data, rows, cols)
        return call


@tvm.transform.module_pass(opt_level=0, name="LowerToNPUPrimitives")
class LowerToNPUPrimitives:
    def transform_module(self, mod, _context):
        lowerer = _LowerNPU(mod)
        for gvar, func in mod.functions.items():
            if isinstance(func, relax.Function):
                lowerer.builder_.update_func(gvar, lowerer.visit_expr(func))
        return lowerer.builder_.get()


def npu_pipeline():
    """Lower high-level ops, then run the standard Relax cleanup.

    FoldConstant bakes constant-only subgraphs (e.g. prefill RoPE cos/sin from
    the static position ramp) into initial G-buffer constants, removing their
    on-device instructions; CSE/DCE/canonicalization tidy the binding list the
    backend walks.  Verified byte-exact against the unoptimized programs on the
    family layer graphs.
    """
    return tvm.transform.Sequential([
        LowerToNPUPrimitives(),
        relax.transform.CanonicalizeBindings(),
        relax.transform.EliminateCommonSubexpr(),
        relax.transform.FoldConstant(),
        relax.transform.DeadCodeElimination(),
    ], name="npu0818_pipeline")
