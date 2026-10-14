"""TVM schedules and matrix TensorIntrin definitions for the v09 NPU.

Vector expressions remain in TIR and are emitted by the common Python/native
walkers. Matmul uses explicit FP32 accumulation and tile64 tensorization.
"""
from __future__ import annotations

import tvm
import tvm.tir as tir
from tvm.script import tir as T

from .npu_kernel import single_matmul

REGISTERED = []


def register_all():
    """Register the v09 matrix intrinsics (idempotent across re-imports)."""
    if REGISTERED:
        return REGISTERED
    _register("npu_gemm_acc_sram", _sram_gemm_desc, _sram_gemm_impl)
    _register("npu_fill_zero_sram", _sram_fill_desc, _sram_fill_impl)
    return REGISTERED


def _register(name, desc, impl):
    from tvm.ir import structural_equal
    existing = tir.TensorIntrin.get(name, allow_missing=True)
    if existing is None:
        tir.TensorIntrin.register(name, desc, impl)
    elif not (structural_equal(existing.desc, desc) and structural_equal(existing.impl, impl)):
        raise ValueError(f"conflicting NPU TensorIntrin registration: {name}")
    REGISTERED.append(name)


SRAM_SCOPE = "global.sram"
TILE = 64


@T.prim_func
def _sram_fill_desc(c: T.handle):
    sc = T.int32()
    C = T.match_buffer(c, (TILE, TILE), "float32", strides=[sc, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    with T.block("root"):
        T.reads()
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.block("fill"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.float32(0)


@T.prim_func
def _sram_fill_impl(c: T.handle):
    sc = T.int32()
    C = T.match_buffer(c, (TILE, TILE), "float32", strides=[sc, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    with T.block("root"):
        T.reads()
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("int32", "npu_fill_zero",
                                 C.access_ptr("w"), sc))


@T.prim_func
def _sram_gemm_desc(a: T.handle, b: T.handle, c: T.handle):
    sa = T.int32(); sb = T.int32(); sc = T.int32()
    A = T.match_buffer(a, (TILE, TILE), "float16", strides=[sa, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    B = T.match_buffer(b, (TILE, TILE), "float16", strides=[sb, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    C = T.match_buffer(c, (TILE, TILE), "float32", strides=[sc, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    with T.block("root"):
        T.reads(C[0:TILE, 0:TILE], A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j, k in T.grid(TILE, TILE, TILE):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + T.Cast("float32", A[vi, vk]) * T.Cast("float32", B[vk, vj])


@T.prim_func
def _sram_gemm_impl(a: T.handle, b: T.handle, c: T.handle):
    sa = T.int32(); sb = T.int32(); sc = T.int32()
    A = T.match_buffer(a, (TILE, TILE), "float16", strides=[sa, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    B = T.match_buffer(b, (TILE, TILE), "float16", strides=[sb, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    C = T.match_buffer(c, (TILE, TILE), "float32", strides=[sc, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    with T.block("root"):
        T.reads(C[0:TILE, 0:TILE], A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("int32", "npu_gemm_acc",
                                 C.access_ptr("rw"), sc,
                                 A.access_ptr("r"), sa,
                                 B.access_ptr("r"), sb))


def matmul_extents(mod, func_name):
    """(M, N, K) of a structurally recognized single dot reduction."""
    info = single_matmul(mod[func_name])
    return info.extents if info else None


def schedule_matmul_sram(mod, func_name, tile=64, hold=None, panel=None,
                         tile_pad=False, config=None, *, return_schedule=False, schedule=None):
    """Tile a legalized matmul and stage its operands into SRAM.

    The DMA that the hand-written backend inserts by hand is expressed here as
    ``cache_read`` into an SRAM-scoped buffer, placed by ``compute_at`` at the
    K-tile loop so each tile is staged just before it is consumed.
    """
    sch = schedule if schedule is not None else tir.Schedule(mod)
    info = single_matmul(mod[func_name])
    if info is None:
        raise ValueError("expected one canonical M/N/K dot reduction")
    if info.dtypes != ("float16", "float16", "float32"):
        raise ValueError(f"matrix schedule requires FP16 A/B/C, got {info.dtypes}; "
                         "INT8 requires the W8A8 scale/output contract")
    block_name = info.name
    block = sch.get_block(block_name, func_name=func_name)
    loops = sch.get_loops(block)
    if len(loops) < 3:
        raise ValueError("matmul needs at least three loops")
    # a batched matmul keeps its batch loops outside the tiled M/N/K nest
    i, j, k = loops[-3:]
    extents = [int(sch.get(loop).extent) for loop in (i, j, k)]
    unpad = None
    output = sch.get(block).writes[0].buffer
    consumers = list(sch.get_consumers(block))
    if len(consumers) != 1:
        raise ValueError("FP32 matrix accumulator requires one explicit FP16 store")
    final_store = consumers[0]
    final_value = sch.get(final_store).body.value
    if not (isinstance(final_value, tir.Cast) and final_value.dtype == "float16"
            and isinstance(final_value.value, tir.BufferLoad)):
        raise ValueError("matrix accumulator consumer must be an FP16 cast")
    pending, reachable = [final_store], set()
    while pending:
        rv = pending.pop()
        next_block = sch.get(rv)
        if next_block.name_hint in reachable:
            continue
        if (next_block.init is not None or len(next_block.writes) != 1
                or next_block.writes[0].buffer.dtype != "float16"
                or not all(v.iter_type == tir.IterVar.DataPar for v in next_block.iter_vars)
                or not tvm.ir.structural_equal(next_block.writes[0].buffer.shape, output.shape)):
            raise ValueError("matrix epilogue must preserve the output shape and FP16 storage")
        reachable.add(next_block.name_hint)
        pending.extend(sch.get_consumers(rv))
    epilogue = [rv for rv in sch.get_child_blocks(sch.get_block("root",func_name=func_name))
                if sch.get(rv).name_hint in reachable]
    epilogue_names = {sch.get(rv).name_hint for rv in epilogue}
    # a producer block ahead of the matmul (pad_einsum's fills, or the
    # w_dequant of a quantized weight) computes in SRAM: stage its inputs and
    # re-scope its output, and the matmul then reads that operand in place
    root = sch.get_block("root", func_name=func_name)
    for child in sch.get_child_blocks(root):
        info = sch.get(child)
        if (info.name_hint == block_name or info.name_hint in epilogue_names
                or info.writes[0].buffer.same_as(output)):
            continue
        for index in range(len(info.reads)):
            try:
                sch.cache_read(child, index, SRAM_SCOPE)
            except tir.ScheduleError:
                pass
        try:
            sch.set_scope(child, 0, SRAM_SCOPE)
        except tir.ScheduleError:
            pass
    if any(extent % tile for extent in extents):
        # pad_einsum is the standard primitive for exactly this: grow the
        # iteration domain to the intrinsic's factor and insert the producer /
        # consumer padding blocks that keep accesses in bounds
        batch = len(loops) - 3
        old_buffers = {sch.get(rv).writes[0].buffer.data for rv in sch.get_child_blocks(root)
                       if sch.get(rv).writes}
        sch.pad_einsum(block, [1] * batch + [tile, tile, tile])
        # pad_einsum has just added its own producer and consumer blocks; the
        # producers get the same staging treatment as above
        for child in sch.get_child_blocks(root):
            info = sch.get(child)
            if info.writes[0].buffer.same_as(output):
                unpad = child          # copies the padded result back out
                continue
            if info.writes[0].buffer.data in old_buffers:
                continue
            for index in range(len(info.reads)):
                try:
                    sch.cache_read(child, index, SRAM_SCOPE)
                except tir.ScheduleError:
                    pass
            # the padded operand itself already lives in SRAM, so the matmul
            # must not stage it again (an SRAM-to-SRAM copy would take the PE
            # output register and break the MAC chain).  The consumer block
            # writes the function output, which cannot be re-scoped.
            try:
                sch.set_scope(child, 0, SRAM_SCOPE)
            except tir.ScheduleError:
                pass
        # the fill stays a conditional in the producer blocks; the codegen
        # splits each row into its in-bounds and padded parts (decompose_padding
        # would give the buffer two writers, which blocks cache_read)
        loops = sch.get_loops(block)
        i, j, k = loops[-3:]
    # Where each operand is staged is the whole of the intra-op decision.  The
    # PE tile is 64x64 and there is one accumulator, so k must stay innermost
    # to keep a C tile's MAC chain unbroken; that leaves only the order of the
    # two outer loops, and with it which operand is invariant inside.  Staging
    # that one at the outer loop reads it once and streams the other past it.
    if config is not None:
        from .npu_schedule_space import loop_spec
        mp, mt = sch.split(i, [None, config['m_panel']])
        mi, ii = sch.split(mt, [None, tile])
        np, nt = sch.split(j, [None, config['n_panel']])
        ni, ji = sch.split(nt, [None, tile])
        kg, kt = sch.split(k, [None, config['k_chunk']])
        ko, ki = sch.split(kt, [None, tile])
        by_name = dict(m_panel=mp, n_panel=np, m_tile=mi, n_tile=ni,
                       k_group=kg, k_tile=ko)
        order = [name for name, _ in loop_spec(tuple(extents), config, tile)]
        sch.reorder(*[by_name[name] for name in order], ii, ji, ki)
        staged_at = (by_name[config['a_at']], by_name[config['b_at']])
        enclosing = by_name[order[3]]
        # Initialization must precede the entire K chain, including K groups.
        k_o = kg
    elif hold is None:
        # both operands staged at the reduction loop: the smallest footprint
        # and the most re-reads, and what every kernel got before a budget was
        # available to say otherwise
        i_o, i_i = sch.split(i, [None, tile])
        j_o, j_i = sch.split(j, [None, tile])
        k_o, k_i = sch.split(k, [None, tile])
        sch.reorder(i_o, j_o, k_o, i_i, j_i, k_i)
        staged_at, enclosing = (k_o, k_o), j_o
    elif hold == "A":
        # input-stationary: A[panel, K] is read once, B[K, tile] once per row
        # block
        m_b, i_r = sch.split(i, [None, panel])
        i_o, i_i = sch.split(i_r, [None, tile])
        j_o, j_i = sch.split(j, [None, tile])
        k_o, k_i = sch.split(k, [None, tile])
        sch.reorder(m_b, j_o, i_o, k_o, i_i, j_i, k_i)
        staged_at, enclosing = (m_b, j_o), i_o
    else:
        # weight-stationary: the mirror image
        n_b, j_r = sch.split(j, [None, panel])
        j_o, j_i = sch.split(j_r, [None, tile])
        i_o, i_i = sch.split(i, [None, tile])
        k_o, k_i = sch.split(k, [None, tile])
        sch.reorder(n_b, i_o, j_o, k_o, i_i, j_i, k_i)
        # Streaming A at K preserves the FP32 chain: this is direct DMA,
        # while the computed/dequantized B panel stays outside that chain.
        staged_at, enclosing = (k_o if hold == "B_stream" else i_o, n_b), j_o
    write_back = sch.cache_write(block, 0, SRAM_SCOPE)
    sch.set_scope(write_back, 0, SRAM_SCOPE)
    sch.reverse_compute_at(write_back, enclosing, preserve_unit_loops=True)
    if unpad is not None:
        # otherwise the padded result stays live across the whole N axis, and
        # a wide output (lm_head is [64, 128256] once M is padded) does not fit
        sch.reverse_compute_at(unpad, enclosing)
        sch.set_scope(unpad, 0, SRAM_SCOPE)
    sch.reverse_compute_at(final_store, enclosing, preserve_unit_loops=True)
    if len(epilogue) > 1:
        sch.set_scope(final_store,0,SRAM_SCOPE)
        params = {b.data for b in mod[func_name].buffer_map.values()}
        for rv in epilogue[1:]:
            # Preserve every FP16 boundary while keeping the intermediate on
            # chip. Standard reverse_compute_at determines the tile domain.
            sch.reverse_compute_at(rv,enclosing,preserve_unit_loops=True)
            if sch.get(rv).writes[0].buffer.data in params:
                if sch.get(rv).writes[0].buffer.scope() != SRAM_SCOPE:
                    copy = sch.cache_write(rv,0,SRAM_SCOPE)
                    sch.reverse_compute_at(copy,enclosing,preserve_unit_loops=True)
            else:
                sch.set_scope(rv,0,SRAM_SCOPE)
            for index, region in enumerate(sch.get(rv).reads):
                if region.buffer.data in params and region.buffer.scope() != SRAM_SCOPE:
                    stage=sch.cache_read(rv,index,SRAM_SCOPE,consumer_blocks=[rv])
                    sch.compute_at(stage,enclosing,preserve_unit_loops=True)
    if unpad is not None:
        sch.compute_inline(unpad)
    sch.compute_inline(write_back)
    for index in (0, 1):
        if sch.get(block).reads[index].buffer.scope() == SRAM_SCOPE:
            # a padded operand is already in SRAM, whole.  Holding it whole is
            # what we want when it is small -- it is then read exactly once,
            # which is why a padded prefill is accidentally cheaper than an
            # unpadded one.  It stops being what we want when the padded thing
            # is the KV cache: decode has M=1 always, so this branch is always
            # taken, and V as [heads, capacity+1, head_dim] does not fit past
            # capacity 256.  Tiling the producer into the staging loop trades
            # that back -- re-filled per tile, sized by CompactBufferAllocation.
            if not tile_pad:
                continue
            padded = sch.get(block).reads[index].buffer
            for producer in sch.get_producers(block):
                if not sch.get(producer).writes[0].buffer.same_as(padded):
                    continue
                behind = list(sch.get_producers(producer))
                sch.compute_at(producer, staged_at[index])
                for stage in behind:
                    sch.compute_at(stage, staged_at[index])
            continue
        stage = sch.cache_read(block, index, SRAM_SCOPE)
        sch.compute_at(stage, staged_at[index])
    # a dequantized weight is a computed producer: without this it holds the
    # WHOLE weight in SRAM (18 MiB at real widths).  Tiling it into the
    # K-loop makes it -- and its int8/scale stages behind it -- one tile at a
    # time, which CompactBufferAllocation then shrinks to tile extents.  The
    # cost is re-dequantizing per row tile, the same trade B1 already makes
    # for re-staging.
    # Follow the RHS producer graph by buffer identity. Names are solely
    # Schedule handles; renaming a weight decoder must not allocate all W.
    rhs = sch.get(block).reads[1].buffer
    def place_producer(rv):
        producers = list(sch.get_producers(rv))
        if not any(sch.get(loop).same_as(sch.get(staged_at[1])) for loop in sch.get_loops(rv)):
            sch.compute_at(rv, staged_at[1], preserve_unit_loops=True)
        for producer in producers:
            place_producer(producer)
    for producer in sch.get_producers(block):
        producer_info = sch.get(producer)
        direct_stage = (isinstance(producer_info.body, tir.BufferStore)
                        and isinstance(producer_info.body.value, tir.BufferLoad)
                        and producer_info.body.value.buffer.data in {
                            b.data for b in mod[func_name].buffer_map.values()})
        if (producer_info.writes and producer_info.writes[0].buffer.same_as(rhs)
                and not direct_stage
                and (tile_pad or not any(extent % tile for extent in extents))):
            place_producer(producer)
    init = sch.decompose_reduction(block, k_o)
    sch.tensorize(sch.get_loops(block)[-3], "npu_gemm_acc_sram")
    sch.tensorize(sch.get_loops(init)[-2], "npu_fill_zero_sram")
    return sch if return_schedule else sch.mod


def attention_rows(mod, func_name):
    from .npu_kernel import attention_kernel
    kernel = attention_kernel(mod[func_name])
    return kernel.rows if kernel else None


def schedule_attention_blocked(mod, func_name, block, tile=64):
    """Block a whole-attention kernel over ``block`` query rows at a time.

    The score matrix is [heads, seq, seq] and never fits; blocking the query
    axis makes every intermediate [1, block, seq], which does.  Because a
    block holds a query row's WHOLE row of scores, the softmax over it is
    exact -- there is no running maximum to carry and no accumulated output to
    rescale, so unlike FlashAttention this needs no online softmax and leaves
    the matrix unit's single accumulator alone.  That is a consequence of
    having 8 MiB rather than a GPU's shared memory.

    Every block of the kernel shares (head, row) as its first two loops; the
    matmuls only add a reduction axis after them.  So one split of the anchor's
    row axis and a compute_at of everything else is the whole transform.  The
    row block is put OUTSIDE the head loop, which is what lets the mask rows --
    shared by every head -- be staged once per block instead of once per
    (block, head).
    """
    from .npu_kernel import attention_kernel
    kernel = attention_kernel(mod[func_name])
    if kernel is None:
        return None
    sch = tir.Schedule(mod)
    root = sch.get_block("root", func_name=func_name)
    children = list(sch.get_child_blocks(root))
    prim = sch.mod[func_name]
    params = {prim.buffer_map[p].data for p in prim.params}
    anchor = sch.get_block(kernel.store, func_name=func_name)
    if sch.get(anchor).writes[0].buffer.data not in params:
        return None                     # not the result: not the shape we know

    head, row = sch.get_loops(anchor)[:2]
    row_o, _ = sch.split(row, [None, block])
    inner = sch.get_loops(anchor)[3:]   # row_i is index 2 after the split
    sch.reorder(row_o, head, *sch.get_loops(anchor)[2:3], *inner)
    for block_rv in reversed([rv for rv in children
                              if sch.get(rv).name_hint != kernel.store]):
        try:
            sch.compute_at(block_rv, head)
        except tir.ScheduleError:
            return None                 # a block that will not go inside

    def blocks():
        return sch.get_child_blocks(sch.get_block("root", func_name=func_name))

    for block_rv in blocks():
        info = sch.get(block_rv)
        if info.writes and info.writes[0].buffer.data not in params:
            try:
                sch.set_scope(block_rv, 0, SRAM_SCOPE)
            except tir.ScheduleError:
                pass
    sch.reverse_compute_at(sch.cache_write(anchor, 0, SRAM_SCOPE), head)
    for block_rv in blocks():
        for index, region in enumerate(sch.get(block_rv).reads):
            buffer = region.buffer
            if buffer.data not in params:
                continue
            # An operand whose head axis is 1 is broadcast over the heads --
            # the mask is the one -- so it belongs OUTSIDE the head loop and
            # is then read once per row block instead of once per (block,
            # head).  Staged inside, the mask was 201 MB of the 428 MB this
            # kernel moved at seq 2048, which is the whole reason the row
            # block is outside the head loop in the first place.
            try:
                broadcast = int(buffer.shape[0]) == 1
            except (TypeError, ValueError):
                broadcast = False
            try:
                stage = sch.cache_read(block_rv, index, SRAM_SCOPE,
                                       consumer_blocks=[block_rv])
                sch.compute_at(stage, row_o if broadcast else head)
            except tir.ScheduleError:
                pass

    # both matmuls go to the matrix unit; same steps as schedule_matmul_sram,
    # applied inside the block rather than over the whole tensor
    for name in (kernel.score, kernel.output):
        target = sch.get_block(name, func_name=func_name)
        loops = sch.get_loops(target)
        if any(int(sch.get(loop).extent) % tile for loop in loops[-3:]):
            # pad_einsum wants one entry per block iter var.  After compute_at
            # get_loops also returns the enclosing loops, which are not iters
            # of this block, so the count has to come from the block itself.
            iters = len(sch.get(target).iter_vars)
            sch.pad_einsum(target, [1] * (iters - 3) + [tile] * 3)
            loops = sch.get_loops(target)
        # capture the three handles before splitting any of them: get_loops
        # after a split returns the new loops and the tail no longer names
        # what it did
        m, n, k = loops[-3:]
        m_o, m_i = sch.split(m, [None, tile])
        n_o, n_i = sch.split(n, [None, tile])
        k_o, k_i = sch.split(k, [None, tile])
        sch.reorder(m_o, n_o, k_o, m_i, n_i, k_i)
        init = sch.decompose_reduction(target, k_o)
        sch.tensorize(sch.get_loops(target)[-3], "npu_gemm_acc_sram")
        sch.tensorize(sch.get_loops(init)[-2], "npu_fill_zero_sram")
    return sch.mod


register_all()


def generic_stripes(mod, func_name):
    """Descending stripes whose fused row range remains rectangular.

    cache_write/reverse_compute_at copies a rectangular buffer region. A
    flattened stripe must not wrap only part of a leading axis: [3,2,16]
    rows admit 32-row stripes but not 48-row stripes, which would write back
    uncomputed holes. Divisors of each axis times its trailing row volume
    enumerate exactly the aligned rectangles for this schedule.
    """
    from math import isqrt
    sch = tir.Schedule(mod)
    root = sch.get_block("root", func_name=func_name)
    prim = sch.mod[func_name]
    params = {prim.buffer_map[p].data for p in prim.params}
    common = None
    for block_rv in sch.get_child_blocks(root):
        block = sch.get(block_rv)
        if block.writes and block.writes[0].buffer.data in params:
            loops = sch.get_loops(block_rv)
            if len(loops) < 2:
                return []
            choices, suffix = set(), 1
            for loop in reversed(loops[:-1]):
                extent = int(sch.get(loop).extent)
                for factor in range(1, isqrt(extent) + 1):
                    if extent % factor == 0:
                        choices.update((suffix * factor, suffix * (extent // factor)))
                suffix *= extent
            common = choices if common is None else common & choices
    return sorted(common or (), reverse=True)


def _block_work_bounds(sch, block_rv):
    """Conservative execution bounds, using TVM's loop-domain analyzer.

    A lower bound is useful only when the block predicate is always true.
    In particular, do not count padded/tail iterations as executed work.
    """
    analyzer = tvm.arith.Analyzer()
    lower, upper = 1, 1
    loops = sch.get_loops(block_rv)
    if not loops:
        return 0, 1
    for loop in loops:
        node = sch.get(loop)
        bound = analyzer.const_int_bound(node.extent)
        lower *= max(0, bound.min_value)
        upper *= max(0, bound.max_value)
        analyzer.bind(node.loop_var, tvm.ir.Range(node.min, node.extent))
    block = sch.get(block_rv)
    predicates = []
    tir.stmt_functor.post_order_visit(sch.get(loops[-1]).body, lambda node:
        predicates.append(node.predicate) if isinstance(node, tir.BlockRealize)
        and node.block.same_as(block) else None)
    if len(predicates) != 1 or not analyzer.can_prove(predicates[0]):
        lower = 0
    return lower, upper


def schedule_generic_striped(mod, func_name, stripe):
    """Stage a non-matmul kernel ``stripe`` rows at a time.

    ``schedule_generic_sram`` stages whole tensors, which stops fitting once
    the prompt grows: at 512 tokens a normalisation kernel alone asks for
    9 MiB of the 8 MiB SRAM.  These kernels all iterate their result's leading
    axes and then one inner axis, so fusing the leading axes gives one row
    loop that every block of the kernel shares.  Splitting it and pulling the
    rest of the kernel inside makes every staged buffer ``stripe`` rows tall,
    which is what CompactBufferAllocation then sizes them to.

    Returns None when the kernel does not have that shape, so the caller can
    keep the whole-tensor schedule rather than get a wrong one.
    """
    if stripe not in generic_stripes(mod, func_name):
        from .npu_errors import LinkError
        raise LinkError(f"{func_name}: stripe {stripe} crosses a rectangular output boundary")
    sch = tir.Schedule(mod)
    root = sch.get_block("root", func_name=func_name)
    children = sch.get_child_blocks(root)
    prim = sch.mod[func_name]
    params = {prim.buffer_map[p].data for p in prim.params}

    # a kernel result is a parameter some block writes; there can be more than
    # one (a split writes two), and then the blocks are siblings rather than a
    # chain, so each result gets its own stripe loop
    anchors = [rv for rv in children
               if sch.get(rv).writes
               and any(sch.get(rv).writes[0].buffer.data.same_as(p)
                       for p in params)]
    if not anchors:
        return None
    if len(anchors) > 1 and len(anchors) != len(children):
        return None                # siblings mixed with producers: not handled
    names = [sch.get(rv).name_hint for rv in anchors]
    chain = len(anchors) == 1

    outers = {}
    for anchor, name in zip(anchors, names):
        loops = sch.get_loops(anchor)
        if len(loops) < 2:
            return None
        outers[name], _ = sch.split(sch.fuse(*loops[:-1]), [None, stripe])

    if chain:
        # every other block produces something the result needs, so pulling
        # them under the stripe loop is what shrinks their buffers too;
        # nearest producer first, which is the order compute_at can verify
        outer = outers[names[0]]
        for block_rv in reversed(children):
            if sch.get(block_rv).name_hint == names[0]:
                continue
            _, original_work = _block_work_bounds(sch, block_rv)
            try:
                sch.compute_at(block_rv, outer)
            except tir.ScheduleError:
                return None
            repeated_work, _ = _block_work_bounds(sch, block_rv)
            if repeated_work > original_work:
                # compute_at infers a rectangular region. A conditional
                # multi-input consumer can therefore repeat *every* producer
                # for each output slice, even though only one is needed.
                # This generic schedule supports partitioning, not producer
                # recomputation. Reject before staging/codegen; FuseOps region
                # selection retains the legal unfused reference.
                from .npu_errors import LinkError
                raise LinkError(
                    f"generic stripe repeats producer {sch.get(block_rv).name_hint}: "
                    f"at least {repeated_work} iterations versus at most "
                    f"{original_work} before compute_at; requires a different placement")

    def blocks():
        return sch.get_child_blocks(sch.get_block("root", func_name=func_name))

    def outer_for(block_rv):
        """The stripe loop this block's staging belongs to."""
        if chain:
            return outers[names[0]]
        return outers.get(sch.get(block_rv).name_hint)

    # from here the three steps are schedule_generic_sram's, with the staging
    # placed at the stripe loop instead of the whole loop nest
    for block_rv in blocks():
        block = sch.get(block_rv)
        if not block.writes:
            continue
        target = block.writes[0].buffer
        if not any(target.data.same_as(p) for p in params):
            try:
                sch.set_scope(block_rv, 0, SRAM_SCOPE)
            except tir.ScheduleError:
                pass

    for block_rv in blocks():
        name = sch.get(block_rv).name_hint
        if name in names:
            sch.reverse_compute_at(sch.cache_write(block_rv, 0, SRAM_SCOPE),
                                   outers[name])

    for block_rv in blocks():
        block = sch.get(block_rv)
        outer = outer_for(block_rv)
        if outer is None:
            continue
        for index, region in enumerate(block.reads):
            if region.buffer.data in params and region.buffer.scope() != SRAM_SCOPE:
                try:
                    # Sibling results (a split's two halves) read the same
                    # input but live under DIFFERENT stripe loops, and one
                    # shared cache cannot sit in both -- so those get a cache
                    # per consumer.  A chain's blocks all live under the one
                    # stripe loop and can share, and must: silu reads its
                    # input twice (sigmoid, then multiply) and was staging it
                    # twice, 1.50x its floor.
                    stage = sch.cache_read(
                        block_rv, index, SRAM_SCOPE,
                        consumer_blocks=None if chain else [block_rv])
                    sch.compute_at(stage, outer)
                except tir.ScheduleError:
                    pass
    return sch.mod


def schedule_generic_sram(mod, func_name):
    """Stage a non-matmul kernel entirely in SRAM.

    Compute units address SRAM only, so a kernel's parameters are cache_read
    in, its result cache_written out, and every intermediate the kernel
    allocates internally is re-scoped to SRAM.  Whole tensors are staged here;
    tiling them is a scheduling refinement, not a correctness requirement.
    """
    sch = tir.Schedule(mod)
    root = sch.get_block("root", func_name=func_name)
    children = sch.get_child_blocks(root)
    prim = sch.mod[func_name]
    params = {prim.buffer_map[p].data for p in prim.params}

    # intermediates the kernel allocates itself
    for block_rv in children:
        block = sch.get(block_rv)
        if not block.writes:
            continue
        target = block.writes[0].buffer
        if not any(target.data.same_as(p) for p in params):
            sch.set_scope(block_rv, 0, SRAM_SCOPE)

    # the producer of the result writes SRAM, then one stage copies it out
    for block_rv in children:
        block = sch.get(block_rv)
        for index, region in enumerate(block.writes):
            if region.buffer.data in params and region.buffer.scope() != SRAM_SCOPE:
                sch.cache_write(block_rv, index, SRAM_SCOPE)

    # every read of a parameter comes from SRAM
    for block_rv in sch.get_child_blocks(sch.get_block("root", func_name=func_name)):
        block = sch.get(block_rv)
        for index, region in enumerate(block.reads):
            if region.buffer.data in params and region.buffer.scope() != SRAM_SCOPE:
                try:
                    sch.cache_read(block_rv, index, SRAM_SCOPE)
                except tir.ScheduleError:
                    pass
    return sch.mod
