"""Structural contracts for kernels accepted by the NPU schedules.

Block names are handles for TVM's Schedule API, not operation identifiers.
This analysis deliberately recognizes the canonical batched M/N/K dot
reduction that our tensor intrinsic implements.  Other reductions remain
generic; a name such as 'matmul' cannot grant them matrix-unit semantics.
"""
from dataclasses import dataclass

import tvm
from tvm import tir


@dataclass(frozen=True)
class MatmulBlock:
    name: str
    extents: tuple[int, int, int]  # M, N, K (batch loops are outside)
    dtypes: tuple[str, str, str]  # A, B, C


def matmul_blocks(prim):
    found = []

    def equal(a, b):
        return tvm.ir.structural_equal(a, b)

    def load(expr):
        while isinstance(expr, tir.Cast):
            expr = expr.value
        return expr if isinstance(expr, tir.BufferLoad) else None

    def visit(block):
        if not isinstance(block, tir.Block) or len(block.iter_vars) < 3:
            return
        axes = block.iter_vars
        if ([int(v.iter_type) for v in axes]
                != [tir.IterVar.DataPar] * (len(axes) - 1)
                   + [tir.IterVar.CommReduce]):
            return
        init, store = block.init, block.body
        if not (isinstance(init, tir.BufferStore)
                and isinstance(store, tir.BufferStore)
                and init.buffer.same_as(store.buffer)
                and isinstance(init.value, (tir.FloatImm, tir.IntImm))
                and init.value.value == 0
                and equal(init.indices, store.indices)
                and isinstance(store.value, tir.Add)):
            return
        acc, product = store.value.a, store.value.b
        if not (isinstance(acc, tir.BufferLoad)
                and acc.buffer.same_as(store.buffer)
                and equal(acc.indices, store.indices)
                and isinstance(product, tir.Mul)):
            return
        a, b = load(product.a), load(product.b)
        if a is None or b is None or min(len(a.indices), len(b.indices), len(store.indices)) < 2:
            return
        i, j, k = [v.var for v in axes[-3:]]
        if not all(equal(x, y) for x, y in zip(
                [a.indices[-2], a.indices[-1], b.indices[-2], b.indices[-1],
                 store.indices[-2], store.indices[-1]], [i, k, k, j, i, j])):
            return
        if not all(isinstance(v.dom.extent, tir.IntImm) for v in axes[-3:]):
            return
        found.append(MatmulBlock(
            block.name_hint, tuple(int(v.dom.extent) for v in axes[-3:]),
            (str(a.buffer.dtype), str(b.buffer.dtype), str(store.buffer.dtype))))

    tir.stmt_functor.post_order_visit(prim.body, visit)
    return tuple(found)


def single_matmul(prim):
    blocks = matmul_blocks(prim)
    return blocks[0] if len(blocks) == 1 else None


@dataclass(frozen=True)
class AttentionKernel:
    score: str
    output: str
    rows: int
    store: str


def attention_kernel(prim):
    """Recognize canonical rank-3 QK -> last-axis softmax -> PV dataflow.

    Names are returned only as Schedule handles.  A renamed graph has the
    same contract; an unrelated pair of dots plus a block named softmax does
    not. Extra independent outputs are deliberately outside this schedule.
    """
    dots = matmul_blocks(prim)
    if len(dots) != 2 or any(d.dtypes != ("float16", "float16", "float32") for d in dots):
        return None
    blocks = []
    tir.stmt_functor.post_order_visit(
        prim.body, lambda node: blocks.append(node)
        if isinstance(node, tir.Block) and isinstance(node.body, tir.BufferStore)
        else None)
    by_name = {b.name_hint: b for b in blocks}
    writers = {b.body.buffer.data: b for b in blocks}
    if len(writers) != len(blocks):
        return None
    params = {b.data for b in prim.buffer_map.values()}
    stores = {}
    for d in dots:
        for block in blocks:
            value = block.body.value
            if (block.body.buffer.data in params and isinstance(value, tir.Cast)
                    and value.dtype == "float16" and isinstance(value.value, tir.BufferLoad)
                    and value.value.buffer.same_as(by_name[d.name].body.buffer)):
                stores[d.name] = block
    outputs = [d for d in dots if d.name in stores]
    if len(outputs) != 1:
        return None
    final = outputs[0]
    first = next(d for d in dots if d != final)
    output, score = by_name[final.name], by_name[first.name]
    if len(output.iter_vars) != 4 or len(score.iter_vars) != 4:
        return None
    if first.extents[0] != final.extents[0] or first.extents[1] != final.extents[2]:
        return None

    def producer(load):
        while isinstance(load, tir.Cast):
            load = load.value
        return writers.get(load.buffer.data) if isinstance(load, tir.BufferLoad) else None

    def eq(a, b):
        return tvm.ir.structural_equal(a, b)

    norm = producer(output.body.value.b.a)
    if norm is None or not isinstance(norm.body.value, tir.Div):
        return None
    exp_load, sum_load = norm.body.value.a, norm.body.value.b
    exp, total = producer(exp_load), producer(sum_load)
    if total is not None and isinstance(total.body.value, tir.Cast):
        total = producer(total.body.value)
    if exp is None or total is None or len(norm.iter_vars) != 3:
        return None
    indices = [v.var for v in norm.iter_vars]
    if not (eq(exp_load.indices, indices) and eq(sum_load.indices, indices[:-1])):
        return None
    call = exp.body.value
    if not (isinstance(call, tir.Call) and call.op.name == "tir.exp"
            and isinstance(call.args[0], tir.Sub)):
        return None
    source, max_load = call.args[0].a, call.args[0].b
    maximum = producer(max_load)
    if maximum is None or not isinstance(source, tir.BufferLoad):
        return None
    indices = [v.var for v in exp.iter_vars]
    if not (eq(source.indices, indices) and eq(max_load.indices, indices[:-1])):
        return None
    for reduction, op, input_buffer, init in (
            (total, tir.Add, exp.body.buffer, 0),
            (maximum, tir.Max, source.buffer, -65504)):
        if (len(reduction.iter_vars) != 3
                or [int(v.iter_type) for v in reduction.iter_vars] != [0, 0, 2]
                or not isinstance(reduction.init, tir.BufferStore)
                or not isinstance(reduction.init.value, tir.FloatImm)
                or float(reduction.init.value.value) != init
                or not isinstance(reduction.body.value, op)):
            return None
        acc, term = reduction.body.value.a, reduction.body.value.b
        if isinstance(term, tir.Cast) and term.dtype == "float32":
            term = term.value
        if not (isinstance(acc, tir.BufferLoad) and isinstance(term, tir.BufferLoad)
                and acc.buffer.same_as(reduction.body.buffer)
                and term.buffer.same_as(input_buffer)
                and eq(acc.indices, [v.var for v in reduction.iter_vars[:-1]])
                and eq(term.indices, [v.var for v in reduction.iter_vars])):
            return None

    # All producer blocks must belong to this output, and QK must feed softmax.
    def ancestors(block, seen):
        if block.name_hint in seen:
            return
        seen.add(block.name_hint)
        for region in block.reads:
            parent = writers.get(region.buffer.data)
            if parent is not None and parent.name_hint != block.name_hint:
                ancestors(parent, seen)

    seen, score_path = set(), set()
    ancestors(stores[final.name], seen)
    ancestors(exp, score_path)
    if seen != set(by_name) or first.name not in score_path:
        return None
    # compute_at shares the first two axes across every block.
    shape = output.body.buffer.shape
    for b in blocks:
        if (len(b.iter_vars) < 2 or str(b.body.buffer.dtype) not in ("float16", "float32")
                or not all(eq(b.iter_vars[i].dom.extent, shape[i]) for i in (0, 1))):
            return None
    return AttentionKernel(first.name, final.name, final.extents[0], stores[final.name].name_hint)
