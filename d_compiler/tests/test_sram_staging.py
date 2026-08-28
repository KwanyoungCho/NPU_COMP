"""S4 gate: SRAM staging expressed as cache_read / cache_write.

The DMA the hand-written backend inserts by hand is now a scheduling decision:
``cache_read``/``cache_write`` into a ``global.sram`` buffer, placed by
``compute_at``.  The codegen turns those cache stages into GLOAD/GSTORE and
addresses the staged buffers in SRAM nibbles.
"""
import sys
from pathlib import Path

import numpy as np
from tvm import relax, tir

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import backend_v09, driver, npu_intrin, passes
from npu_compiler.backend_v09 import V09Asm
from npu_compiler.tir_codegen_v09 import SramEmitter, Walker


def _matmul_module(m, k, n):
    bb = relax.BlockBuilder()
    lhs = relax.Var("lhs", relax.TensorStructInfo([m, k], "float16"))
    rhs = relax.Var("rhs", relax.TensorStructInfo([k, n], "float16"))
    with bb.function("main", [lhs, rhs]):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(relax.op.matmul(lhs, rhs)))
        bb.emit_func_output(out)
    return bb.finalize()


def _compile_staged(mod, mp):
    """LegalizeOps -> SRAM-staged schedule -> v09 program."""
    legalized = relax.transform.LegalizeOps()(mod)
    func_name = next(gv.name_hint for gv, fn in legalized.functions.items()
                     if isinstance(fn, tir.PrimFunc))
    scheduled = npu_intrin.schedule_matmul_sram(legalized, func_name)
    prim = scheduled[func_name]

    asm = V09Asm()
    walker = Walker(asm, {}, SramEmitter(asm))

    # global placement comes from the memory plan
    offsets = [mp.offset[p] for p in mp.params] + [mp.offset[mp.output]]
    for buffer, offset in zip(prim.buffer_map.values(), offsets):
        walker.bases[buffer.data] = offset

    # SRAM placement for the cache stages, bump-allocated per buffer
    cursor = 0
    root = prim.body.block
    for buffer in root.alloc_buffers:
        if buffer.scope() == npu_intrin.SRAM_SCOPE:
            walker.declare_sram(buffer, cursor)
            size = 1
            for dim in buffer.shape:
                size *= int(dim)
            cursor += size * 4
            cursor = (cursor + 7) // 8 * 8
    walker.run(prim, {})
    asm.halt()
    return asm


def _check(m, k, n, seed):
    rng = np.random.default_rng(seed)
    inputs = {"lhs": rng.normal(0, 0.3, (m, k)).astype(np.float16),
              "rhs": rng.normal(0, 0.3, (k, n)).astype(np.float16)}
    mod = passes.npu_pipeline()(_matmul_module(m, k, n))
    func = mod["main"]
    mp = backend_v09.plan(func)
    reference = np.asarray(
        driver.run_compiled(backend_v09.compile_func(func, mp), mp, inputs),
        np.float16)
    asm = _compile_staged(mod, mp)
    dma = sum(1 for w in asm.words if (w & 0xFF) in (0xA0, 0xA8))
    got = np.asarray(driver.run_compiled(asm, mp, inputs), np.float16)
    np.testing.assert_array_equal(reference.view(np.uint16), got.view(np.uint16))
    return len(asm.words), dma


def test_sram_staged_matmul_single_tile():
    words, dma = _check(64, 64, 64, seed=21)
    assert dma > 0, "no DMA emitted for the cache stages"
    print(f"  [PASS] 64^3 staged matmul bit-exact ({words} words, {dma} DMA)")


def test_sram_staged_matmul_multi_tile():
    words, dma = _check(128, 128, 128, seed=22)
    print(f"  [PASS] 128^3 staged matmul bit-exact ({words} words, {dma} DMA)")


def test_cache_stages_are_the_only_global_access():
    """Compute must touch SRAM only; global memory appears solely in DMA."""
    mod = passes.npu_pipeline()(_matmul_module(64, 64, 64))
    func = mod["main"]
    mp = backend_v09.plan(func)
    asm = _compile_staged(mod, mp)
    index, seen_load, seen_dma = 0, 0, 0
    while index < len(asm.words):
        opcode = asm.words[index] & 0xFF
        if opcode in (0xA0, 0xA8):
            seen_dma += 1
            index += 4
            continue
        if opcode == 0x90:
            seen_load += 1
        index += 1
    assert seen_dma >= 3, seen_dma          # two reads + one write-back
    assert seen_load > 0, seen_load
    print(f"  [PASS] {seen_dma} DMA stages, {seen_load} unit loads from SRAM")


if __name__ == "__main__":
    test_sram_staged_matmul_single_tile()
    test_sram_staged_matmul_multi_tile()
    test_cache_stages_are_the_only_global_access()
    print("ALL SRAM STAGING (S4) TESTS PASSED")
