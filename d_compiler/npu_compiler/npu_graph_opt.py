"""Exact Relax index rewrites for the target's single-row matmul path."""
import tvm
from tvm import relax, tir
from tvm.relax.expr_functor import PyExprMutator, mutator


@mutator
class _RepeatedGemv(PyExprMutator):
    def __init__(self, module):
        super().__init__(module)
        self.count = 0

    def visit_call_(self, call):
        call = super().visit_call_(call)
        if call.op != tvm.ir.Op.get('relax.matmul'):
            return call
        lhs, rhs = call.args
        repeated = self.lookup_binding(rhs) if isinstance(rhs, relax.Var) else rhs
        if not (isinstance(repeated, relax.Call) and repeated.op == tvm.ir.Op.get('relax.repeat')
                and repeated.attrs.axis is not None and int(repeated.attrs.axis) in (0,-3)):
            return call
        original = repeated.args[0]
        left, right = lhs.struct_info, original.struct_info
        if not (isinstance(left, relax.TensorStructInfo) and isinstance(right, relax.TensorStructInfo)
                and left.ndim == right.ndim == 3 and left.dtype == right.dtype == 'float16'
                and left.shape is not None and right.shape is not None
                and str(call.attrs.out_dtype) in ('','float16')):
            return call
        if not all(isinstance(d,tir.IntImm) for d in (*left.shape,*right.shape)):
            return call
        h,m,k = map(int,left.shape);kv,depth,n = map(int,right.shape)
        group = int(repeated.attrs.repeats)
        # The compound attention schedule operates on multi-row rank-3 inputs.
        # This pass covers the separate GEMV path and preserves that boundary.
        if m != 1 or group <= 1 or min(h,kv,k,n) <= 0 or h != kv*group or k != depth:
            return call
        bb = self.builder_
        a = bb.emit(relax.op.reshape(lhs,(kv,group,1,k)))
        b = bb.emit(relax.op.expand_dims(original,1))
        value = bb.emit(relax.Call(call.op,[a,b],call.attrs))
        self.count += 1
        return bb.normalize(relax.op.reshape(value,(h,1,n)))


@tvm.transform.module_pass(opt_level=0,name='FoldRepeatedGemv')
class FoldRepeatedGemv:
    """matmul(A[KV*G,1,K], repeat(B[KV,K,N],G,axis=0))

    becomes broadcast matmul(A.reshape(KV,G,1,K), B.expand_dims(1)).
    Head h reads B[h//G] in both forms. No arithmetic/rounding is changed;
    standard TVM legalization, scheduling and codegen handle the new graph.
    """
    def transform_module(self, module, context):
        rewrite = _RepeatedGemv(module)
        for gv,fn in module.functions.items():
            if isinstance(fn,relax.Function):
                rewrite.builder_.update_func(gv,rewrite.visit_expr(fn))
        if not rewrite.count:
            return module
        result = rewrite.builder_.get().with_attr('npu.repeated_gemv',rewrite.count)
        return relax.transform.DeadCodeElimination()(result)
