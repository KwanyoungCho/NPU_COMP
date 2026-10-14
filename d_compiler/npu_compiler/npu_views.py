"""Use TVM's reshape proof for the device's alias-capable output ABI."""
import tvm
from tvm import relax, tir
from tvm.relax.expr_functor import mutator, PyExprMutator


@mutator
class _Views(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)
        self.mod = mod

    def visit_call_(self, call):
        call = super().visit_call_(call)
        if call.op != tvm.ir.Op.get("relax.call_tir"):
            return call
        fn = self.mod[call.args[0]]
        args = call.args[1].fields
        if not isinstance(fn, tir.PrimFunc) or len(args) != 1:
            return call
        src, dst = args[0].struct_info, call.struct_info
        if not (isinstance(dst, relax.TensorStructInfo) and src.shape is not None
                and dst.shape is not None and src.dtype == dst.dtype):
            return call
        analyzer = tvm.arith.Analyzer()
        def numel(shape):
            result = 1
            for dim in shape:
                result = result * dim
            return result
        if (analyzer.can_prove_equal(numel(src.shape), numel(dst.shape))
                and relax.analysis.has_reshape_pattern(fn)):
            return self.builder_.normalize(relax.op.reshape(args[0], dst.shape))
        return call


@tvm.transform.module_pass(opt_level=0, name="NpuOutputViews")
class OutputViews:
    def transform_module(self, mod, ctx):
        rewrite = _Views(mod)
        for gv, fn in mod.functions.items():
            if isinstance(fn, relax.Function):
                rewrite.builder_.update_func(gv, rewrite.visit_expr(fn))
        return rewrite.builder_.get()


def legalize_split(bb, call):
    """Only axis-zero contiguous partitions become byte-offset TVM views."""
    import numpy as np
    source = call.args[0]
    info = source.struct_info
    axis = int(call.attrs.axis) % info.ndim
    if axis != 0 or not all(isinstance(d, tir.IntImm) for d in info.shape):
        # Returning the unchanged call makes LegalizeOps use its default.
        from tvm.relax.transform.legalize_ops.manipulate import _split
        return _split(bb, call)
    outputs, offset = [], 0
    for field in call.struct_info.fields:
        shape = list(field.shape)
        view = bb.emit(relax.op.memory.view(source, shape=shape, relative_byte_offset=offset))
        # TVM's packed PrimFuncs expect byte_offset=0. The VM materializes
        # this when needed; the NPU's absolute-address ABI already presents
        # each view as an independent base, so its planner needs no copy.
        outputs.append(bb.emit(relax.op.memory.ensure_zero_offset(view)))
        offset += int(np.prod([int(d) for d in shape])) * np.dtype(info.dtype).itemsize
    return relax.Tuple(outputs)
