"""The native codegen must encode exactly what the Python encoders encode.

The C++ side exists only to be faster, so the only thing that makes it usable
is that it is indistinguishable.  This runs the identical script of operations
through both and compares word for word -- a single differing bit here would
be a wrong program on the machine, with no other symptom than bad numbers.

Skips (rather than fails) when the library has not been built, so the suite
still runs on a checkout that has not compiled it.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import native
from npu_compiler.backend_v09 import V09Asm
from npu_compiler.isa_0818 import DST, IMM, MAIN, PARTIAL, SCALAR, SRC1, SRC2, VECTOR
from npu_compiler.isa_v09 import (ACT_SILU, DT_FP16, DT_FP32, DT_INT8)
from npu_compiler.tir_codegen_v09 import SramEmitter


def _python_selftest():
    """The mirror of EncodeSelfTest() in npu_codegen/codegen_v09.cc."""
    a = V09Asm()
    stage = SramEmitter(a)

    a.nop()
    a.vlen(0xFFFF)
    a.addr(SRC1, 0x12345678, PARTIAL)
    a.addr(SRC2, 0x0000ABCD, MAIN)
    a.addr(DST, 0, PARTIAL)
    a.shape_dt(SRC1, 64, 3072, MAIN, DT_FP16)
    a.shape_dt(SRC2, 7, 64, PARTIAL, DT_INT8)
    a.shape_dt(DST, 1, 9728, PARTIAL, DT_FP32)

    a.load(0, SRC1, 0, 0, 0)
    a.load(1, SRC2, 1, 63, 5)
    a.save(0, 0, 0, 0)
    a.save(1, 1, 64, 7)

    a.v_add(VECTOR)
    a.v_sub(IMM, 0x1234)
    a.v_mul(SCALAR, 7)
    a.v_div(IMM, 127)
    a.v_max(VECTOR)
    a.v_min(IMM, 3)
    a.v_sqrt()
    a.v_exp()
    a.v_cos()
    a.v_sin()
    a.v_sign_inv()
    a.v_copy()
    a.v_reduce_sum()
    a.v_reduce_max()
    a.v_broadcast_addr(0xDEADBEEF)

    a.m_mul(VECTOR, mac=False)
    a.m_mul(VECTOR, mac=True)
    a.m_mul(IMM, 9, ACT_SILU, True)

    a.vquant()
    a.vdequant()
    a.ascale(0x1FFFF8)
    a.wscale(8)

    stage.vector(SRC1, 4096)
    stage.broadcast(2048)
    stage.region(SRC2, 8192, 3072, 64, 64)
    stage.strided(DST, 512, 128, 64)
    stage.dma_in(0, 0, 4096)
    stage.dma_out(4 * 1024, 8 * 1024, 260)
    stage.dma_2d(64, 6144, 16384, 64, 128, True)
    stage.dma_2d(0, 8, 0, 70000, 4, False)
    a.halt()
    return np.asarray(a.words, dtype=np.uint32)


def test_encoders_are_bit_exact():
    if not native.load():
        print(f"  [SKIP] native codegen not built ({native.library_path()});"
              f" run npu_codegen/build.sh")
        return
    expected = _python_selftest()
    got = native.encode_selftest()
    assert got.shape == expected.shape, \
        f"word count {got.shape} != python {expected.shape}"
    bad = np.nonzero(got != expected)[0]
    assert bad.size == 0, (
        f"{bad.size} words differ, first at {bad[0]}: "
        f"native {got[bad[0]]:#010x} != python {expected[bad[0]]:#010x}")
    print(f"  [PASS] {got.size} words bit-exact across every encoder, "
          f"descriptor and DMA shape")


TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0)


def _lower(mod):
    from npu_compiler import npu_legalize
    from npu_compiler import tvm_pipeline as pipeline

    return pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=True)(mod)


def _layer_module(seq=4):
    """A small but complete Llama layer -- every kernel class the walker has
    to handle: staged matmul, RMSNorm's reduction, softmax, RoPE, concat."""
    from npu_compiler.nn_models import llama

    mod, _, _ = llama.build_prefill(TINY, seq=seq)
    return _lower(mod)


def _graphs():
    """Graphs whose kernel shapes differ enough that a walker can cover one
    and miss another: Qwen3 adds its per-head norms, and decode's attention
    runs over a fixed cache with a length mask.

    Gemma is not here because its builder needs a real checkpoint spec; it is
    covered instead by test_nn_gemma, which links and runs a full graph with
    the native walker on by default.
    """
    from npu_compiler.nn_models import llama, qwen3

    generate = _lower(llama.build_generate(TINY, 4, 5)[0])
    return [
        ("llama prefill", "prefill", _layer_module()),
        ("qwen3 prefill", "prefill",
         _lower(qwen3.build_prefill(dict(TINY), seq=4)[0])),
        ("llama prefill_cache", "prefill_cache", generate),
        ("llama decode", "decode", generate),
    ]


def test_native_matches_python_on_every_family():
    """Every kernel the native walker covers must emit the same words.

    ``native_mode="compare"`` runs the Python walker as usual and raises on
    the first differing word, so this passes only if the two are identical
    everywhere the native one claims coverage.
    """
    if not native.load():
        print("  [SKIP] native codegen not built")
        return
    from npu_compiler import npu_link

    for label, entry, lowered in _graphs():
        asm, plan = npu_link.compile_program(lowered, entry, peephole=False,
                                             native_mode="compare")
        covered = sum(1 for ok in plan.native.values() if ok)
        total = len(plan.native)
        assert total, f"{label}: no kernels were checked"
        print(f"  [PASS] {label:22s} {covered:3d}/{total:<3d} kernel types "
              f"identical ({len(asm.words):,} words)")
        for name, ok in sorted(plan.native.items()):
            if not ok:
                print(f"         (Python emitted it: {name})")


def test_native_program_is_identical_end_to_end():
    """Linking with the native walker must give the same program as without.

    This is the gate that matters: coverage is a speed question only if the
    mixed program is word-for-word what Python alone produces.
    """
    if not native.load():
        print("  [SKIP] native codegen not built")
        return
    from npu_compiler import npu_link

    lowered = _layer_module()
    python_asm, _ = npu_link.compile_program(lowered, "prefill",
                                             native_mode="python")
    native_asm, plan = npu_link.compile_program(lowered, "prefill",
                                                native_mode="use")
    python_words = np.asarray(python_asm.words, dtype=np.uint32)
    native_words = np.asarray(native_asm.words, dtype=np.uint32)
    assert native_words.shape == python_words.shape, \
        f"native program has {native_words.size} words, python {python_words.size}"
    bad = np.nonzero(native_words != python_words)[0]
    assert bad.size == 0, (
        f"{bad.size} words differ, first at {bad[0]}: "
        f"native {native_words[bad[0]]:#010x} != "
        f"python {python_words[bad[0]]:#010x}")
    covered = sum(1 for ok in plan.native.values() if ok)
    print(f"  [PASS] whole program identical: {python_words.size:,} words, "
          f"{covered}/{len(plan.native)} kernel types native")


def test_an_uncovered_kernel_falls_back():
    """The native walker refusing a kernel must be a fall back, not a crash.

    Coverage is currently complete, so nothing in the suite exercises the
    fall back naturally -- and an unhandled exception type here would turn
    the first genuinely unsupported kernel into a hard failure instead of a
    slow one.  A deliberately malformed call stands in for that kernel.
    """
    if not native.load():
        print("  [SKIP] native codegen not built")
        return
    from npu_compiler import npu_link, npu_memplan
    from tvm import relax, tir

    lowered = _layer_module()
    planned, _ = npu_memplan.assign_addresses(lowered, "prefill")
    prim = next(fn for _, fn in planned.functions.items()
                if isinstance(fn, tir.PrimFunc))
    # too few addresses for the signature: the native walker raises, and the
    # wrapper has to turn that into "not covered"
    assert native.codegen_kernel(prim, [], {}, (), {}, 0) is None
    print("  [PASS] a kernel the native walker refuses returns None")


if __name__ == "__main__":
    test_encoders_are_bit_exact()
    test_an_uncovered_kernel_falls_back()
    test_native_matches_python_on_every_family()
    test_native_program_is_identical_end_to_end()
    print("ALL NATIVE CODEGEN TESTS PASSED")
