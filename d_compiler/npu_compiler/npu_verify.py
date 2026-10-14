"""Semantic checks before scheduling a device PrimFunc."""
import tvm
from tvm import tir, te
from .npu_errors import LinkError


def validate_w8a8(prim):
    from .npu_quantize import _qmatmul_a8_te
    inputs = [te.placeholder(prim.buffer_map[p].shape, str(prim.buffer_map[p].dtype), name=f"p{i}")
              for i,p in enumerate(prim.params[:-1])]
    if len(inputs) != 3:
        raise LinkError("W8A8 requires three inputs and one output")
    expected = te.create_prim_func([*inputs, _qmatmul_a8_te(*inputs)], index_dtype_override="int64")
    def normalize(fn):
        def block(node):
            if isinstance(node,tir.Block):
                return tir.Block(node.iter_vars,node.reads,node.writes,"block",node.body,
                    node.init,node.alloc_buffers,node.match_buffers,node.annotations)
        body=tir.stmt_functor.ir_transform(fn.body,None,block,["tir.Block"])
        return tir.PrimFunc(fn.params,body,fn.ret_type,fn.buffer_map)
    if not tvm.ir.structural_equal(normalize(prim),normalize(expected),map_free_vars=True):
        raise LinkError("W8A8 semantic body does not match its declared quantization contract")


def validate_primitive(prim, name):
    if prim.attrs and prim.attrs.get("npu.kernel") == "w8a8":
        validate_w8a8(prim)
        return
    def check(block):
        if not isinstance(block,tir.Block) or block.init is None:
            return
        init,store=block.init,block.body
        if not isinstance(init,tir.BufferStore) or not isinstance(store,tir.BufferStore):
            raise LinkError(f"{name}/{block.name_hint}: unsupported reduction body")
        value=store.value
        if isinstance(value,tir.Add):
            if str(store.buffer.dtype) != "float32":
                raise LinkError(f"{name}/{block.name_hint}: sum reduction must explicitly use FP32 accumulation")
            if not isinstance(init.value,(tir.FloatImm,tir.IntImm)) or init.value.value != 0:
                raise LinkError(f"{name}/{block.name_hint}: sum reduction requires zero initialization")
        elif isinstance(value,tir.Max):
            if not isinstance(init.value,tir.FloatImm) or init.value.value != -65504:
                raise LinkError(f"{name}/{block.name_hint}: max reduction requires the FP16 minimum identity")
    tir.stmt_functor.post_order_visit(prim.body,check)
