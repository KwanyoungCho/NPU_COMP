"""Lower internal FP32 accumulators to v09 FP16 result-store slots.

Semantic TIR and TensorIntrin descriptions keep FP32 arithmetic. This pass
runs AFTER tensorization: matrix updates are ISA calls, and vector reductions
become explicit FP32-reduction markers. A buffer may use the narrow physical
slot only when every non-accumulating consumer is an explicit FP16 cast.
"""
import tvm
from tvm import tir
from .npu_errors import LinkError
from .v09_contract import CONTRACT


def forward_output_tiles(prim):
    """Store a completed matrix tile directly into its sole SRAM output view.

    Match-buffer regions carry the destination stride/offset through the
    existing TensorIntrin ABI. The source allocation must only be written by
    matrix intrinsics and read by one full-tile identity copy. This does not
    alter accumulation, casting, or a live intermediate's other consumers.
    """
    analyzer = tvm.arith.Analyzer()
    outputs = {b.data for b in prim.buffer_map.values() if b.scope() == 'global.sram'}
    if not outputs:
        return prim
    allocations, candidates = {}, []
    def collect(node):
        if isinstance(node,tir.Block):
            allocations.update({b.data:b for b in node.alloc_buffers})
        if isinstance(node,tir.For):
            loops, body = [], node
            while isinstance(body,tir.For):
                loops.append(body); body = body.body
            if (len(loops)==2 and isinstance(body,tir.BlockRealize)
                    and not body.block.iter_vars and not body.block.init
                    and analyzer.can_prove(body.predicate)
                    and isinstance(body.block.body,tir.BufferStore)):
                store = body.block.body
                if isinstance(store.value,tir.BufferLoad) and store.buffer.data in outputs:
                    candidates.append((loops,body,store))
    tir.stmt_functor.post_order_visit(prim.body,collect)
    result = prim
    forwarded = 0
    for loops,copy,store in candidates:
        source, target = store.value.buffer, store.buffer
        if (source.data not in allocations or len(source.shape)!=2 or len(target.shape)!=2
                or source.dtype != 'float16' or target.dtype != 'float16'
                or source.strides or target.strides
                or any(not analyzer.can_prove_equal(d,64) for d in source.shape)
                or any(n.kind != tir.ForKind.SERIAL or not analyzer.can_prove_equal(n.extent,64)
                       or not analyzer.can_prove_equal(n.min,0) for n in loops)):
            continue
        if not all(analyzer.can_prove_equal(index,node.loop_var)
                   for index,node in zip(store.value.indices,loops)):
            continue
        offsets = [analyzer.simplify(index-node.loop_var) for index,node in zip(store.indices,loops)]
        variables = {n.loop_var for n in loops}
        invalid = []
        for offset in offsets:
            tir.stmt_functor.post_order_visit(offset,lambda n:
                invalid.append(n) if isinstance(n,tir.Var) and n in variables else None)
        views = set()
        def inspect(node):
            if isinstance(node,tir.Block):
                for match in node.match_buffers:
                    if match.source.buffer.data.same_as(source.data): views.add(match.buffer.data)
            if isinstance(node,tir.BufferLoad) and node.buffer.data.same_as(source.data):
                if not node.same_as(store.value): invalid.append(node)
            if isinstance(node,tir.BufferStore) and node.buffer.data.same_as(source.data): invalid.append(node)
        tir.stmt_functor.post_order_visit(result.body,inspect)
        destinations = 0
        def calls(node):
            nonlocal destinations
            if isinstance(node,tir.Call) and node.op.name == 'tir.call_extern':
                pointers = [arg for arg in node.args if isinstance(arg,tir.Call)
                            and arg.op.name == 'tir.tvm_access_ptr' and arg.args[1] in views]
                if pointers:
                    if (node.args[0].value not in ('npu_fill_zero','npu_gemm_acc')
                            or len(pointers)!=1 or not pointers[0].same_as(node.args[1])):
                        invalid.append(node)
                    else: destinations += 1
        tir.stmt_functor.post_order_visit(result.body,calls)
        if invalid or destinations != 2 or not views:
            continue
        def region(r):
            if r.buffer.data.same_as(source.data):
                return tir.BufferRegion(target,[tvm.ir.Range.from_min_extent(
                    analyzer.simplify(v.min+offset),v.extent) for v,offset in zip(r.region,offsets)])
            return r
        def before(node):
            return tir.Evaluate(0) if node.same_as(copy) else None
        def rewrite(node):
            if isinstance(node,tir.Block):
                return tir.Block(node.iter_vars,[region(r) for r in node.reads],
                    [region(r) for r in node.writes],node.name_hint,node.body,node.init,
                    [b for b in node.alloc_buffers if not b.data.same_as(source.data)],
                    [tir.MatchBufferRegion(m.buffer,region(m.source)) for m in node.match_buffers],
                    node.annotations)
        result = result.with_body(tir.stmt_functor.ir_transform(result.body,before,rewrite))
        forwarded += 1
    if not forwarded:
        return prim
    result = tir.transform.RemoveNoOp()(tvm.IRModule({'main':result}))['main']
    return result.with_attr('npu.forwarded_output_tiles',forwarded)


def lower_stores(prim):
    buffers = {}
    def collect(node):
        if isinstance(node, tir.Block):
            for b in node.alloc_buffers:
                if b.dtype == "float32" and b.scope() != "local.accumulator":
                    buffers[b.data] = b
            for match in node.match_buffers:
                b = match.buffer
                if b.dtype == "float32" and b.scope() != "local.accumulator":
                    buffers[b.data] = b
    tir.stmt_functor.post_order_visit(prim.body, collect)
    if not buffers:
        return prim
    allowed, loads = set(), set()
    reductions = set()
    def validate(node):
        if isinstance(node, tir.BufferLoad) and node.buffer.data in buffers:
            loads.add(node)
        if isinstance(node, tir.Cast) and node.dtype == "float16" and isinstance(node.value, tir.BufferLoad):
            allowed.add(node.value)
        if not isinstance(node, tir.BufferStore) or node.buffer.data not in buffers:
            return
        value = node.value
        if isinstance(value, tir.BufferLoad) and value.buffer.data in buffers:
            allowed.add(value)
        elif isinstance(value, tir.FloatImm) and float(value.value) == 0:
            pass
        elif (isinstance(value, tir.Add) and isinstance(value.a, tir.BufferLoad)
              and value.a.buffer.same_as(node.buffer) and isinstance(value.b, tir.Cast)
              and value.b.dtype == "float32" and value.b.value.dtype == "float16"):
            allowed.add(value.a)
            reductions.add(node)
        else:
            raise LinkError(f"{node.buffer.name}: v09 requires FP16 storage; unsupported FP32 accumulator use")
    tir.stmt_functor.post_order_visit(prim.body, validate)
    if loads - allowed:
        raise LinkError("FP32 accumulator escapes without an explicit FP16 store")
    variables = {v: tir.Var(v.name, tvm.ir.PointerType(tvm.ir.PrimType("float16"), b.scope()))
                 for v, b in buffers.items()}
    replacements = {}
    def buffer(b):
        if b.data not in buffers:
            return b
        if b not in replacements:
            replacements[b] = tir.decl_buffer(b.shape, "float16", b.name, data=variables[b.data],
                strides=b.strides, elem_offset=b.elem_offset, scope=b.scope(),
                data_alignment=b.data_alignment, offset_factor=b.offset_factor)
        return replacements[b]
    def region(r):
        return tir.BufferRegion(buffer(r.buffer), r.region)
    def before(node):
        if isinstance(node, tir.Cast) and node.dtype == "float16" and isinstance(node.value, tir.BufferLoad):
            load = node.value
            if load.buffer.data in buffers:
                return tir.BufferLoad(buffer(load.buffer), load.indices)
        if isinstance(node, tir.BufferStore) and node.buffer.data in buffers:
            value = node.value
            if isinstance(value, tir.Add):
                acc = tir.BufferLoad(buffer(value.a.buffer), value.a.indices)
                value = tir.call_pure_extern("float16", CONTRACT["reductions"]["sum_update"], acc, value.b.value)
            elif isinstance(value, tir.FloatImm):
                value = tir.const(value.value, "float16")
            elif isinstance(value, tir.BufferLoad):
                value = tir.BufferLoad(buffer(value.buffer), value.indices)
            return tir.BufferStore(buffer(node.buffer), value, node.indices)
        return None
    def rewrite(node):
        if isinstance(node, tir.BufferLoad) and node.buffer.data in buffers:
            return tir.BufferLoad(buffer(node.buffer), node.indices)
        if isinstance(node, tir.Cast) and node.dtype == "float16" and node.value.dtype == "float16":
            return node.value
        if isinstance(node, tir.BufferStore) and node.buffer.data in buffers:
            value = node.value
            if isinstance(value, tir.Add):
                # The source was verified above. Keep the accumulation width
                # explicit in the final instruction-level TIR.
                term = value.b.value if isinstance(value.b, tir.Cast) else value.b
                value = tir.call_pure_extern("float16", CONTRACT["reductions"]["sum_update"], value.a, term)
            elif isinstance(value, tir.FloatImm):
                value = tir.const(value.value, "float16")
            return tir.BufferStore(buffer(node.buffer), value, node.indices)
        if isinstance(node, tir.Block):
            return tir.Block(node.iter_vars, [region(r) for r in node.reads], [region(r) for r in node.writes],
                node.name_hint, node.body, node.init, [buffer(b) for b in node.alloc_buffers],
                [tir.MatchBufferRegion(buffer(m.buffer), region(m.source)) for m in node.match_buffers],
                node.annotations)
        if isinstance(node, tir.Call) and node.op.name == "tir.tvm_access_ptr" and node.args[1] in buffers:
            args = list(node.args)
            args[0] = tir.type_annotation("float16")
            return tir.Call(node.dtype, node.op, args)
        return None
    body = tir.stmt_functor.ir_transform(prim.body, before, rewrite)
    body = tir.stmt_functor.substitute(body, variables)
    return prim.with_body(body).with_attr("npu.accumulator_store_lowered", 1)
