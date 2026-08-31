"""Backlog A1: dropping redundant descriptor writes must not change results.

The straight-line stream makes every descriptor register's value decidable, so
a write of the value already held is provably dead.  These tests check that
claim end to end: the same program with and without the pass must produce
identical memory, and the pass must actually remove a large share of the words.
"""
import sys
from pathlib import Path

import numpy as np
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import npu_legalize, npu_link, npu_memplan as M
from npu_compiler import tvm_pipeline as P
from npu_compiler.peephole import eliminate_dead_stores


def _lower(shapes, make):
    bb = relax.BlockBuilder()
    args = [relax.Var(f"a{i}", relax.TensorStructInfo(list(s), "float16"))
            for i, s in enumerate(shapes)]
    with bb.function("prefill", args):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(make(*args)))
        bb.emit_func_output(out)
    return P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                            fuse=False, lift_params=False)(bb.finalize())


def _run(lowered, values, out_shape, peephole):
    asm, plan = npu_link.compile_program(lowered, peephole=peephole)
    planned, _ = M.assign_addresses(lowered)
    func = planned["prefill"]
    got, _ = npu_link.run_program(
        asm, plan, func,
        dict(zip([p.name_hint for p in func.params], values)), out_shape)
    return got, len(asm.words)


def test_results_are_identical_with_and_without():
    rng = np.random.default_rng(4)
    cases = [
        ("matmul [64,128]x[128,192]", [(64, 128), (128, 192)],
         relax.op.matmul, (64, 192)),
        ("matmul padded [7,128]x[128,96]", [(7, 128), (128, 96)],
         relax.op.matmul, (7, 96)),
        ("softmax [8,64,64]", [(8, 64, 64)],
         lambda a: relax.op.nn.softmax(a, axis=-1), (8, 64, 64)),
        ("silu [6,9728]", [(6, 9728)], relax.op.nn.silu, (6, 9728)),
        ("rms_norm [4,2560]", [(4, 2560), (2560,)],
         lambda a, b: relax.op.nn.rms_norm(a, b, axes=[-1]), (4, 2560)),
    ]
    for name, shapes, make, out_shape in cases:
        values = [rng.normal(0, 0.4, s).astype(np.float16) for s in shapes]
        lowered = _lower(shapes, make)
        plain, before = _run(lowered, values, out_shape, peephole=False)
        lean, after = _run(lowered, values, out_shape, peephole=True)
        assert np.array_equal(plain.view(np.uint16), lean.view(np.uint16)), name
        assert after < before, (name, before, after)
        print(f"  [PASS] {name:32s} bit-exact, {before:,} -> {after:,} words "
              f"(-{100 * (before - after) / before:.1f}%)")


def test_a_write_of_the_same_value_is_dropped():
    """Directly: two identical descriptor writes collapse, a differing one
    survives, and a DMA's payload words are never read as opcodes."""
    from npu_compiler.isa_0818 import SRC1
    from npu_compiler.isa_v09 import enc_addr_lo, enc_gload, enc_vlen

    same = [enc_vlen(64), enc_vlen(64), enc_vlen(64)]
    assert eliminate_dead_stores(same) == same[:1]

    differ = [enc_vlen(64), enc_vlen(65), enc_vlen(64)]
    assert eliminate_dead_stores(differ) == differ

    # a DMA word whose payload happens to look like a vlen opcode
    payload = list(enc_gload(0x82, 0x82, 0, 1, 1))
    stream = [enc_vlen(64)] + payload + [enc_vlen(64)]
    assert eliminate_dead_stores(stream) == stream[:-1]

    # addresses are tracked per half, so the halves do not shadow each other
    halves = [enc_addr_lo(SRC1, 16), enc_addr_lo(SRC1, 16)]
    assert eliminate_dead_stores(halves) == halves[:1]
    print("  [PASS] same-value writes drop, differing writes survive, "
          "DMA payload is skipped")


if __name__ == "__main__":
    test_a_write_of_the_same_value_is_dropped()
    test_results_are_identical_with_and_without()
    print("ALL PEEPHOLE (A1) TESTS PASSED")
