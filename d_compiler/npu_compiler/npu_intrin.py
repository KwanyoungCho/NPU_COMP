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


def schedule_matmul_sram(mod, func_name, tile=64):
    """Tile a legalized matmul and stage its operands into SRAM.

    The DMA that the hand-written backend inserts by hand is expressed here as
    ``cache_read`` into an SRAM-scoped buffer, placed by ``compute_at`` at the
    K-tile loop so each tile is staged just before it is consumed.
    """
    sch = tir.Schedule(mod)
    block = sch.get_block("matmul", func_name=func_name)
    i, j, k = sch.get_loops(block)
    i_o, i_i = sch.split(i, [None, tile])
    j_o, j_i = sch.split(j, [None, tile])
    k_o, k_i = sch.split(k, [None, tile])
    sch.reorder(i_o, j_o, k_o, i_i, j_i, k_i)
    write_back = sch.cache_write(block, 0, SRAM_SCOPE)
    sch.reverse_compute_at(write_back, j_o)
    for index in (0, 1):
        stage = sch.cache_read(block, index, SRAM_SCOPE)
        sch.compute_at(stage, k_o)
    init = sch.decompose_reduction(block, k_o)
    sch.tensorize(sch.get_loops(block)[3], "npu_gemm_acc_sram")
    sch.tensorize(sch.get_loops(init)[2], "npu_fill_zero_sram")
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
