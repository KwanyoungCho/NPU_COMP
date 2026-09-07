"""Tensor intrinsics for the v09 vector unit, and the schedules that match them.

Mirrors the compiler-v2 approach (desc = what a lane block computes, impl =
a ``call_extern`` marker the codegen recognises) but sized and named for v09.
Only the marker changes between ops, so one pair of templates covers the whole
elementwise family.

The vector unit processes ``VLEN`` elements per instruction; a legalized
elementwise loop is split by that and the inner run tensorized.
"""
from __future__ import annotations

import tvm.tir as tir
from tvm.script import tir as T

from . import tir_backend        # registers npu_gemm_acc / npu_fill_zero

VLEN = 64          # elements per tensorized vector block

BINARY_OPS = {
    "add": lambda x, y: x + y,
    "subtract": lambda x, y: x - y,
    "multiply": lambda x, y: x * y,
    "divide": lambda x, y: x / y,
}
UNARY_OPS = ("sqrt", "exp", "negative", "cos", "sin")


def _binary_pair(name, compute):
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(a, (VLEN,), "float16", offset_factor=1)
        B = T.match_buffer(b, (VLEN,), "float16", offset_factor=1)
        C = T.match_buffer(c, (VLEN,), "float16", offset_factor=1)
        with T.block("root"):
            T.reads(A[0:VLEN], B[0:VLEN])
            T.writes(C[0:VLEN])
            for i in range(VLEN):
                with T.block("u"):
                    vi = T.axis.remap("S", [i])
                    C[vi] = compute(A[vi], B[vi])

    marker = f"npu_ew2_{name}"

    @T.prim_func
    def impl(a: T.handle, b: T.handle, c: T.handle):
        sa = T.int32(); sb = T.int32(); sc = T.int32()
        A = T.match_buffer(a, (VLEN,), "float16", strides=[sa], offset_factor=1)
        B = T.match_buffer(b, (VLEN,), "float16", strides=[sb], offset_factor=1)
        C = T.match_buffer(c, (VLEN,), "float16", strides=[sc], offset_factor=1)
        with T.block("root"):
            T.reads(A[0:VLEN], B[0:VLEN])
            T.writes(C[0:VLEN])
            T.evaluate(T.call_extern("int32", marker, C.access_ptr("w"),
                                     A.access_ptr("r"), B.access_ptr("r"), VLEN))

    return desc, impl


def _unary_pair(name, compute):
    @T.prim_func
    def desc(a: T.handle, c: T.handle):
        A = T.match_buffer(a, (VLEN,), "float16", offset_factor=1)
        C = T.match_buffer(c, (VLEN,), "float16", offset_factor=1)
        with T.block("root"):
            T.reads(A[0:VLEN])
            T.writes(C[0:VLEN])
            for i in range(VLEN):
                with T.block("u"):
                    vi = T.axis.remap("S", [i])
                    C[vi] = compute(A[vi])

    marker = f"npu_ew1_{name}"

    @T.prim_func
    def impl(a: T.handle, c: T.handle):
        sa = T.int32(); sc = T.int32()
        A = T.match_buffer(a, (VLEN,), "float16", strides=[sa], offset_factor=1)
        C = T.match_buffer(c, (VLEN,), "float16", strides=[sc], offset_factor=1)
        with T.block("root"):
            T.reads(A[0:VLEN])
            T.writes(C[0:VLEN])
            T.evaluate(T.call_extern("int32", marker, C.access_ptr("w"),
                                     A.access_ptr("r"), VLEN))

    return desc, impl


_UNARY_COMPUTE = {
    "sqrt": T.sqrt,
    "exp": T.exp,
    "negative": lambda x: -x,
    "cos": T.cos,
    "sin": T.sin,
}

REGISTERED = []


def register_all():
    """Register every v09 vector intrinsic (idempotent across re-imports)."""
    if REGISTERED:
        return REGISTERED
    for name, compute in BINARY_OPS.items():
        desc, impl = _binary_pair(name, compute)
        _register(f"npu_v09_ew2_{name}", desc, impl)
    for name in UNARY_OPS:
        desc, impl = _unary_pair(name, _UNARY_COMPUTE[name])
        _register(f"npu_v09_ew1_{name}", desc, impl)
    _register("npu_gemm_acc_sram", _sram_gemm_desc, _sram_gemm_impl)
    _register("npu_fill_zero_sram", _sram_fill_desc, _sram_fill_impl)
    return REGISTERED


def _register(name, desc, impl):
    try:
        tir.TensorIntrin.register(name, desc, impl)
    except Exception:                      # already registered (re-import)
        pass
    REGISTERED.append(name)


SRAM_SCOPE = "global.sram"
TILE = 64


@T.prim_func
def _sram_fill_desc(c: T.handle):
    sc = T.int32()
    C = T.match_buffer(c, (TILE, TILE), "float16", strides=[sc, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    with T.block("root"):
        T.reads()
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.block("fill"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.float16(0)


@T.prim_func
def _sram_fill_impl(c: T.handle):
    sc = T.int32()
    C = T.match_buffer(c, (TILE, TILE), "float16", strides=[sc, 1],
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
    C = T.match_buffer(c, (TILE, TILE), "float16", strides=[sc, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    with T.block("root"):
        T.reads(C[0:TILE, 0:TILE], A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j, k in T.grid(TILE, TILE, TILE):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def _sram_gemm_impl(a: T.handle, b: T.handle, c: T.handle):
    sa = T.int32(); sb = T.int32(); sc = T.int32()
    A = T.match_buffer(a, (TILE, TILE), "float16", strides=[sa, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    B = T.match_buffer(b, (TILE, TILE), "float16", strides=[sb, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    C = T.match_buffer(c, (TILE, TILE), "float16", strides=[sc, 1],
                       offset_factor=1, scope=SRAM_SCOPE)
    with T.block("root"):
        T.reads(C[0:TILE, 0:TILE], A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("int32", "npu_gemm_acc",
                                 C.access_ptr("rw"), sc,
                                 A.access_ptr("r"), sa,
                                 B.access_ptr("r"), sb))


def matmul_extents(mod, func_name):
    """(M, N, K) of the kernel's matmul, or None if it has no three-loop nest."""
    sch = tir.Schedule(mod)
    try:
        block = sch.get_block("matmul", func_name=func_name)
    except Exception:
        return None
    loops = sch.get_loops(block)
    if len(loops) < 3:
        return None
    return tuple(int(sch.get(loop).extent) for loop in loops[-3:])


def schedule_matmul_sram(mod, func_name, tile=64, hold=None, panel=None):
    """Tile a legalized matmul and stage its operands into SRAM.

    The DMA that the hand-written backend inserts by hand is expressed here as
    ``cache_read`` into an SRAM-scoped buffer, placed by ``compute_at`` at the
    K-tile loop so each tile is staged just before it is consumed.
    """
    sch = tir.Schedule(mod)
    block = sch.get_block("matmul", func_name=func_name)
    loops = sch.get_loops(block)
    if len(loops) < 3:
        raise ValueError("matmul needs at least three loops")
    # a batched matmul keeps its batch loops outside the tiled M/N/K nest
    i, j, k = loops[-3:]
    extents = [int(sch.get(loop).extent) for loop in (i, j, k)]
    unpad = None
    output = sch.get(block).writes[0].buffer
    # a producer block ahead of the matmul (pad_einsum's fills, or the
    # w_dequant of a quantized weight) computes in SRAM: stage its inputs and
    # re-scope its output, and the matmul then reads that operand in place
    root = sch.get_block("root", func_name=func_name)
    for child in sch.get_child_blocks(root):
        info = sch.get(child)
        if info.name_hint == "matmul" or info.writes[0].buffer.same_as(output):
            continue
        for index in range(len(info.reads)):
            try:
                sch.cache_read(child, index, SRAM_SCOPE)
            except Exception:
                pass
        try:
            sch.set_scope(child, 0, SRAM_SCOPE)
        except Exception:
            pass
    if any(extent % tile for extent in extents):
        # pad_einsum is the standard primitive for exactly this: grow the
        # iteration domain to the intrinsic's factor and insert the producer /
        # consumer padding blocks that keep accesses in bounds
        batch = len(loops) - 3
        sch.pad_einsum(block, [1] * batch + [tile, tile, tile])
        # pad_einsum has just added its own producer and consumer blocks; the
        # producers get the same staging treatment as above
        for child in sch.get_child_blocks(root):
            info = sch.get(child)
            if info.writes[0].buffer.same_as(output):
                unpad = child          # copies the padded result back out
                continue
            if not info.name_hint.endswith("_pad"):
                continue
            for index in range(len(info.reads)):
                try:
                    sch.cache_read(child, index, SRAM_SCOPE)
                except Exception:
                    pass
            # the padded operand itself already lives in SRAM, so the matmul
            # must not stage it again (an SRAM-to-SRAM copy would take the PE
            # output register and break the MAC chain).  The consumer block
            # writes the function output, which cannot be re-scoped.
            try:
                sch.set_scope(child, 0, SRAM_SCOPE)
            except Exception:
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
    if hold is None:
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
        staged_at, enclosing = (i_o, n_b), j_o
    write_back = sch.cache_write(block, 0, SRAM_SCOPE)
    sch.reverse_compute_at(write_back, enclosing)
    if unpad is not None:
        # otherwise the padded result stays live across the whole N axis, and
        # a wide output (lm_head is [64, 128256] once M is padded) does not fit
        sch.reverse_compute_at(unpad, enclosing)
    for index in (0, 1):
        if sch.get(block).reads[index].buffer.scope() == SRAM_SCOPE:
            continue                       # already staged (padded operand)
        stage = sch.cache_read(block, index, SRAM_SCOPE)
        sch.compute_at(stage, staged_at[index])
    # a dequantized weight is a computed producer: without this it holds the
    # WHOLE weight in SRAM (18 MiB at real widths).  Tiling it into the
    # K-loop makes it -- and its int8/scale stages behind it -- one tile at a
    # time, which CompactBufferAllocation then shrinks to tile extents.  The
    # cost is re-dequantizing per row tile, the same trade B1 already makes
    # for re-staging.
    try:
        dequant = sch.get_block("w_dequant", func_name=func_name)
    except Exception:
        dequant = None
    if dequant is not None:
        producers = list(sch.get_producers(dequant))
        sch.compute_at(dequant, staged_at[1])
        for producer in producers:
            sch.compute_at(producer, staged_at[1])
    init = sch.decompose_reduction(block, k_o)
    sch.tensorize(sch.get_loops(block)[-3], "npu_gemm_acc_sram")
    sch.tensorize(sch.get_loops(init)[-2], "npu_fill_zero_sram")
    return sch.mod


def schedule_elementwise(mod, func_name, block_name, intrin):
    """Split the legalized elementwise loop by VLEN and tensorize the inner run."""
    sch = tir.Schedule(mod)
    block = sch.get_block(block_name, func_name=func_name)
    loops = sch.get_loops(block)
    fused = sch.fuse(*loops) if len(loops) > 1 else loops[0]
    _, inner = sch.split(fused, [None, VLEN])
    sch.tensorize(inner, intrin)
    return sch.mod


register_all()


def generic_row_count(mod, func_name):
    """Rows the kernel's result iterates, or None if it has no row axis.

    The stripe is measured in these rows; a kernel whose output is a single
    loop (a per-row scalar, say) has nothing to stripe.
    """
    sch = tir.Schedule(mod)
    root = sch.get_block("root", func_name=func_name)
    prim = sch.mod[func_name]
    output = prim.buffer_map[prim.params[-1]].data
    for block_rv in sch.get_child_blocks(root):
        block = sch.get(block_rv)
        if block.writes and block.writes[0].buffer.data.same_as(output):
            loops = sch.get_loops(block_rv)
            if len(loops) < 2:
                return None
            rows = 1
            for loop in loops[:-1]:
                rows *= int(sch.get(loop).extent)
            return rows
    return None


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
    sch = tir.Schedule(mod)
    root = sch.get_block("root", func_name=func_name)
    children = sch.get_child_blocks(root)
    prim = sch.mod[func_name]
    params = {prim.buffer_map[p].data for p in prim.params}
    output = prim.buffer_map[prim.params[-1]].data

    # the block writing the kernel's result anchors the stripe loop
    anchor = None
    for block_rv in children:
        block = sch.get(block_rv)
        if block.writes and block.writes[0].buffer.data.same_as(output):
            anchor = block_rv
    if anchor is None:
        return None
    anchor_name = sch.get(anchor).name_hint
    loops = sch.get_loops(anchor)
    if len(loops) < 2:
        return None
    outer, _ = sch.split(sch.fuse(*loops[:-1]), [None, stripe])

    # every other block produces something the result needs, so pulling them
    # under the stripe loop is what shrinks their buffers too; nearest
    # producer first, which is the order compute_at can verify
    for block_rv in reversed(children):
        if sch.get(block_rv).name_hint == anchor_name:
            continue
        try:
            sch.compute_at(block_rv, outer)
        except Exception:
            return None

    def blocks():
        return sch.get_child_blocks(sch.get_block("root", func_name=func_name))

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
            except Exception:
                pass

    for block_rv in blocks():
        if sch.get(block_rv).name_hint == anchor_name:
            sch.reverse_compute_at(sch.cache_write(block_rv, 0, SRAM_SCOPE),
                                   outer)
            break

    for block_rv in blocks():
        block = sch.get(block_rv)
        for index, region in enumerate(block.reads):
            if region.buffer.data in params:
                try:
                    sch.compute_at(sch.cache_read(block_rv, index, SRAM_SCOPE),
                                   outer)
                except Exception:
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
    output = prim.buffer_map[prim.params[-1]].data
    is_output = lambda data: data.same_as(output)

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
        if block.writes and is_output(block.writes[0].buffer.data):
            sch.cache_write(block_rv, 0, SRAM_SCOPE)
            break

    # every read of a parameter comes from SRAM
    for block_rv in sch.get_child_blocks(sch.get_block("root", func_name=func_name)):
        block = sch.get(block_rv)
        for index, region in enumerate(block.reads):
            if region.buffer.data in params:
                try:
                    sch.cache_read(block_rv, index, SRAM_SCOPE)
                except Exception:
                    pass
    return sch.mod
