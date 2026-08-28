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
    return REGISTERED


def _register(name, desc, impl):
    try:
        tir.TensorIntrin.register(name, desc, impl)
    except Exception:                      # already registered (re-import)
        pass
    REGISTERED.append(name)


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
