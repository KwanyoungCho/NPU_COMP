"""S3 gate: scheduled TIR -> v09 ISA must be bit-exact with the Relax backend.

Both paths compile the same matmul for the same memory layout; the TIR path
goes Relax -> LegalizeOps -> tir.Schedule(split/reorder/tensorize) -> walker,
the reference path is the existing backend_v09 emitter.
"""
import sys
from pathlib import Path

import numpy as np
import tvm
from tvm import relax, tir

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import backend_v09, driver, tir_backend
from npu_compiler.backend_v09 import V09Asm, _Stager
from npu_compiler.isa_0818 import DST, SRC1, SRC2
from npu_compiler.tir_codegen_v09 import Walker


def _matmul_module(m, k, n):
    bb = relax.BlockBuilder()
    lhs = relax.Var("lhs", relax.TensorStructInfo([m, k], "float16"))
    rhs = relax.Var("rhs", relax.TensorStructInfo([k, n], "float16"))
    with bb.function("main", [lhs, rhs]):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(relax.op.matmul(lhs, rhs)))
        bb.emit_func_output(out)
    return bb.finalize()


class _SramView:
    """Descriptor emitter over a whole-tensor SRAM residency (S3 scope).

    S4 replaces this with cache_read/cache_write-driven staging; here every
    operand is staged whole so the comparison isolates the kernel emission.
    """

    def __init__(self, asm, stager, bases):
        self.asm = asm
        self.stager = stager
        self.nib = bases          # element offset -> SRAM nibble of that tensor base

    def vector(self, operand, elem_off):
        """Address a contiguous vector run (no shape words needed)."""
        base_elem, base_nib = self._owner(elem_off)
        self.asm.addr(operand, base_nib + (elem_off - base_elem) * 4, 1)

    def broadcast(self, elem_off):
        base_elem, base_nib = self._owner(elem_off)
        self.asm.v_broadcast_addr(base_nib + (elem_off - base_elem) * 4)

    def strided(self, operand, elem_off, stride, count):
        """A column of `count` elements spaced `stride` apart: MAIN carries the
        row width, PARTIAL selects a single column."""
        base_elem, base_nib = self._owner(elem_off)
        nib = base_nib + (elem_off - base_elem) * 4
        self.asm.addr(operand, nib, 0)
        self.asm.shape(operand, count, stride, 0)
        self.asm.addr(operand, nib, 1)
        self.asm.shape(operand, count, 1, 1)

    def region(self, operand, elem_off, stride, rows, cols):
        base_elem, base_nib = self._owner(elem_off)
        nib = base_nib + (elem_off - base_elem) * 4
        self.asm.addr(operand, nib - (elem_off - base_elem) * 4, 0)
        self.asm.shape(operand, rows, stride, 0)
        self.asm.addr(operand, nib, 1)
        self.asm.shape(operand, rows, cols, 1)

    def _owner(self, elem_off):
        best = None
        for base_elem, (base_nib, size) in self.nib.items():
            if base_elem <= elem_off < base_elem + size:
                best = (base_elem, base_nib)
        if best is None:
            raise KeyError(elem_off)
        return best


def _tir_program(mod, mp, m, k, n):
    """Compile the matmul through LegalizeOps + tensorize + the v09 walker."""
    legalized = relax.transform.LegalizeOps()(mod)
    func_name = None
    for gv, fn in legalized.functions.items():
        if isinstance(fn, tir.PrimFunc):
            func_name = gv.name_hint
    scheduled = tir_backend.schedule_matmul(legalized, func_name)
    prim = scheduled[func_name]

    asm = V09Asm()
    stager = _Stager(asm)
    off = mp.offset
    params = list(mp.params)
    lhs_off, rhs_off = off[params[0]], off[params[1]]
    dst_off = off[mp.output]
    sizes = {lhs_off: m * k, rhs_off: k * n, dst_off: m * n}
    nibs = {}
    for elem, count in sizes.items():
        nibs[elem] = (stager.range_in(elem, count), count)
    view = _SramView(asm, stager, nibs)

    bases = {}
    for buf_var, elem in zip([b.data for b in prim.buffer_map.values()],
                             [lhs_off, rhs_off, dst_off]):
        bases[buf_var] = elem
    Walker(asm, {}, view).run(prim, bases)

    # write the destination back out
    stager.range_out(dst_off, m * n, nibs[dst_off][0])
    asm.halt()
    return asm


def test_tir_matmul_bit_exact_with_relax_backend():
    m = k = n = 64
    mod = _matmul_module(m, k, n)
    rng = np.random.default_rng(3)
    inputs = {"lhs": rng.normal(0, 0.3, (m, k)).astype(np.float16),
              "rhs": rng.normal(0, 0.3, (k, n)).astype(np.float16)}

    from npu_compiler import passes
    lowered = passes.npu_pipeline()(mod)
    func = lowered["main"]
    mp = backend_v09.plan(func)
    reference = np.asarray(
        driver.run_compiled(backend_v09.compile_func(func, mp), mp, inputs),
        np.float16)

    asm = _tir_program(lowered, mp, m, k, n)
    got = np.asarray(driver.run_compiled(asm, mp, inputs), np.float16)
    np.testing.assert_array_equal(reference.view(np.uint16), got.view(np.uint16))
    print(f"  [PASS] TIR path bit-exact with backend_v09 "
          f"({len(asm.words)} words vs reference)")


def test_tir_matmul_multi_tile():
    """Two K tiles and a 2x2 output grid exercise the MAC chain and tile walk."""
    m, k, n = 128, 128, 128
    mod = _matmul_module(m, k, n)
    rng = np.random.default_rng(4)
    inputs = {"lhs": rng.normal(0, 0.2, (m, k)).astype(np.float16),
              "rhs": rng.normal(0, 0.2, (k, n)).astype(np.float16)}
    from npu_compiler import passes
    lowered = passes.npu_pipeline()(mod)
    func = lowered["main"]
    mp = backend_v09.plan(func)
    reference = np.asarray(
        driver.run_compiled(backend_v09.compile_func(func, mp), mp, inputs),
        np.float16)
    asm = _tir_program(lowered, mp, m, k, n)
    got = np.asarray(driver.run_compiled(asm, mp, inputs), np.float16)
    np.testing.assert_array_equal(reference.view(np.uint16), got.view(np.uint16))
    print(f"  [PASS] multi-tile TIR matmul bit-exact ({len(asm.words)} words)")


def _elementwise_module(shape, make):
    bb = relax.BlockBuilder()
    sinfo = relax.TensorStructInfo(list(shape), "float16")
    x = relax.Var("x", sinfo)
    y = relax.Var("y", sinfo)
    with bb.function("main", [x, y]):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(make(x, y)))
        bb.emit_func_output(out)
    return bb.finalize()


def _tir_elementwise(mod, mp, shape, block, intrin, unary):
    from npu_compiler import npu_intrin
    legalized = relax.transform.LegalizeOps()(mod)
    func_name = next(gv.name_hint for gv, fn in legalized.functions.items()
                     if isinstance(fn, tir.PrimFunc))
    scheduled = npu_intrin.schedule_elementwise(legalized, func_name, block, intrin)
    prim = scheduled[func_name]

    count = int(np.prod(shape))
    asm = V09Asm()
    stager = _Stager(asm)
    params = list(mp.params)
    offs = [mp.offset[p] for p in params] + [mp.offset[mp.output]]
    if unary:
        offs = [mp.offset[params[0]], mp.offset[mp.output]]
    nibs = {off: (stager.range_in(off, count), count) for off in offs}
    view = _SramView(asm, stager, nibs)
    bases = {b.data: off for b, off in
             zip(prim.buffer_map.values(), offs)}
    Walker(asm, {}, view).run(prim, bases)
    stager.range_out(offs[-1], count, nibs[offs[-1]][0])
    asm.halt()
    return asm


def _check_elementwise(shape, make, block, intrin, unary=False):
    from npu_compiler import passes
    mod = _elementwise_module(shape, make)
    rng = np.random.default_rng(9)
    inputs = {"x": (rng.random(shape).astype(np.float16) + np.float16(0.5)),
              "y": (rng.random(shape).astype(np.float16) + np.float16(0.5))}
    lowered = passes.npu_pipeline()(mod)
    func = lowered["main"]
    mp = backend_v09.plan(func)
    reference = np.asarray(
        driver.run_compiled(backend_v09.compile_func(func, mp), mp, inputs),
        np.float16)
    asm = _tir_elementwise(lowered, mp, shape, block, intrin, unary)
    got = np.asarray(driver.run_compiled(asm, mp, inputs), np.float16)
    np.testing.assert_array_equal(reference.view(np.uint16), got.view(np.uint16))
    return len(asm.words)


def test_tir_elementwise_binary_bit_exact():
    import tvm.relax.op as R
    cases = [("add", R.add, "T_add"), ("subtract", R.subtract, "T_subtract"),
             ("multiply", R.multiply, "T_multiply"), ("divide", R.divide, "T_divide")]
    for name, op, block in cases:
        words = _check_elementwise((4, 64), op, block, f"npu_v09_ew2_{name}")
        print(f"  [PASS] {name:9s} bit-exact ({words} words)")


def test_tir_elementwise_unary_bit_exact():
    import tvm.relax.op as R
    cases = [("sqrt", R.sqrt, "compute"), ("exp", R.exp, "compute"),
             ("negative", R.negative, "compute")]
    for name, op, block in cases:
        words = _check_elementwise((4, 64), lambda a, _b, f=op: f(a), block,
                                   f"npu_v09_ew1_{name}", unary=True)
        print(f"  [PASS] {name:9s} bit-exact ({words} words)")


def _tir_single_op(mod, mp, offs, count_out):
    """Compile a single-op module through LegalizeOps + the pattern walker."""
    legalized = relax.transform.LegalizeOps()(mod)
    func_name = next(gv.name_hint for gv, fn in legalized.functions.items()
                     if isinstance(fn, tir.PrimFunc))
    prim = legalized[func_name]
    asm = V09Asm()
    stager = _Stager(asm)
    nibs = {off: (stager.range_in(off, size), size) for off, size in offs}
    view = _SramView(asm, stager, nibs)
    bases = {b.data: off for b, (off, _) in
             zip(prim.buffer_map.values(), [(o, s) for o, s in offs])}
    Walker(asm, {}, view).run(prim, bases)
    out_off = offs[-1][0]
    stager.range_out(out_off, count_out, nibs[out_off][0])
    asm.halt()
    return asm


def _check_single(shapes, make, inputs):
    from npu_compiler import passes
    bb = relax.BlockBuilder()
    vars = [relax.Var(f"a{i}", relax.TensorStructInfo(list(s), "float16"))
            for i, s in enumerate(shapes)]
    with bb.function("main", vars):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(make(*vars)))
        bb.emit_func_output(out)
    mod = passes.npu_pipeline()(bb.finalize())
    func = mod["main"]
    mp = backend_v09.plan(func)
    reference = np.asarray(
        driver.run_compiled(backend_v09.compile_func(func, mp), mp, inputs),
        np.float16)
    sizes = [(mp.offset[p], int(np.prod(mp.shape[p]))) for p in mp.params]
    sizes.append((mp.offset[mp.output], int(np.prod(mp.shape[mp.output]))))
    asm = _tir_single_op(mod, mp, sizes, sizes[-1][1])
    got = np.asarray(driver.run_compiled(asm, mp, inputs), np.float16)
    np.testing.assert_array_equal(reference.view(np.uint16), got.view(np.uint16))
    return len(asm.words)


def test_tir_reduction_bit_exact():
    import tvm.relax.op as R
    rng = np.random.default_rng(12)
    x = rng.standard_normal((4, 64)).astype(np.float16)
    for name, op in (("sum", lambda a: R.sum(a, axis=[1], keepdims=True)),
                     ("max", lambda a: R.max(a, axis=[1], keepdims=True))):
        words = _check_single([(4, 64)], op, {"a0": x})
        print(f"  [PASS] {name:9s} bit-exact ({words} words)")


def test_tir_movement_bit_exact():
    import tvm.relax.op as R
    rng = np.random.default_rng(13)
    cases = [
        ("broadcast", [(4, 1)], lambda a: R.broadcast_to(a, relax.ShapeExpr([4, 64]))),
        ("transpose", [(4, 64)], lambda a: R.permute_dims(a)),
        ("slice", [(4, 64)],
         lambda a: R.strided_slice(a, axes=[1], begin=[16], end=[48])),
    ]
    for name, shapes, op in cases:
        inputs = {f"a{i}": rng.standard_normal(s).astype(np.float16)
                  for i, s in enumerate(shapes)}
        words = _check_single(shapes, op, inputs)
        print(f"  [PASS] {name:9s} bit-exact ({words} words)")


if __name__ == "__main__":
    test_tir_matmul_bit_exact_with_relax_backend()
    test_tir_matmul_multi_tile()
    test_tir_elementwise_binary_bit_exact()
    test_tir_elementwise_unary_bit_exact()
    test_tir_reduction_bit_exact()
    test_tir_movement_bit_exact()
    print("ALL TIR->v09 CODEGEN (S3) TESTS PASSED")
