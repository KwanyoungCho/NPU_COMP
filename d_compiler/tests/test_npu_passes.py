"""Relax pass pipeline regression for the 0818 backends.

Checks that (a) the standard cleanup pipeline preserves program semantics
byte-exactly, and (b) LowerToNPUPrimitives expands high-level nn ops into
exactly the same primitive mix as the hand-built legalize graphs, including
the all-ones-weight fast path (no trailing multiply).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

import tvm
from tvm import relax

from npu_compiler import driver, legalize, passes


def _run(module, inputs):
    compiled = driver.compile_module(module, backend="source-0818")
    return np.asarray(driver.run_compiled(*compiled, inputs), dtype=np.float16), compiled


def _build_highlevel(rows, cols, ones_weight=False):
    """norm -> softmax -> gelu chain via high-level nn ops."""
    x = relax.Var("x", relax.TensorStructInfo([rows, cols], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([cols], "float16"))
    weight = relax.const(np.ones(cols, dtype="float16")) if ones_weight else w
    params = [x] if ones_weight else [x, w]
    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            n = bb.emit(relax.op.nn.rms_norm(x, weight, axes=[-1], epsilon=1e-6))
            s = bb.emit(relax.op.nn.softmax(n, axis=-1))
            g = bb.emit(relax.op.nn.gelu_tanh(s))
            gv = bb.emit_output(g)
        bb.emit_func_output(gv)
    return bb.finalize()


def _build_legalized(rows, cols, ones_weight=False):
    """The same chain built directly from legalize primitives."""
    x = relax.Var("x", relax.TensorStructInfo([rows, cols], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([cols], "float16"))
    params = [x] if ones_weight else [x, w]
    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            n = legalize.rms_norm(bb, x, None if ones_weight else w,
                                  rows, cols, eps=1e-6)
            s = legalize.softmax_lastdim(bb, n, rows, cols)
            g = legalize.gelu_tanh(bb, s, rows, cols)
            gv = bb.emit_output(g)
        bb.emit_func_output(gv)
    return bb.finalize()


def test_lowering_matches_legalize():
    rows, cols = 5, 96
    rng = np.random.default_rng(3)
    inputs = {"x": np.asarray(rng.standard_normal((rows, cols)) * 2, np.float16),
              "w": np.asarray(rng.uniform(0.8, 1.2, (cols,)), np.float16)}
    high, high_c = _run(_build_highlevel(rows, cols), inputs)
    hand, hand_c = _run(_build_legalized(rows, cols), inputs)
    assert np.array_equal(high, hand), "lowered high-level ops differ from legalize"
    assert len(high_c[0].words) == len(hand_c[0].words), (
        len(high_c[0].words), len(hand_c[0].words))
    print(f"  [PASS] high-level == legalize (byte-exact, "
          f"{len(high_c[0].words)} words both)")


def test_ones_weight_skips_multiply():
    rows, cols = 5, 96
    rng = np.random.default_rng(4)
    inputs = {"x": np.asarray(rng.standard_normal((rows, cols)) * 2, np.float16)}
    high, high_c = _run(_build_highlevel(rows, cols, ones_weight=True), inputs)
    hand, hand_c = _run(_build_legalized(rows, cols, ones_weight=True), inputs)
    assert np.array_equal(high, hand)
    assert len(high_c[0].words) == len(hand_c[0].words), (
        len(high_c[0].words), len(hand_c[0].words))
    print(f"  [PASS] all-ones weight lowers without the weight multiply "
          f"({len(high_c[0].words)} words both)")


def test_no_highlevel_ops_survive():
    lowered = passes.npu_pipeline()(_build_highlevel(4, 32))
    names = set()
    for block in lowered["main"].body.blocks:
        for binding in block.bindings:
            if isinstance(binding.value, relax.Call) and isinstance(
                    binding.value.op, tvm.ir.Op):
                names.add(binding.value.op.name)
    banned = {"relax.nn.rms_norm", "relax.nn.softmax", "relax.nn.gelu_tanh"}
    assert not (names & banned), names & banned
    print(f"  [PASS] no high-level ops survive lowering ({len(names)} op kinds)")


def test_pipeline_preserves_primitive_graphs():
    """Legalize-built graphs must stay byte-exact through the pipeline."""
    rows, cols = 5, 96
    module = _build_legalized(rows, cols)
    rng = np.random.default_rng(5)
    inputs = {"x": np.asarray(rng.standard_normal((rows, cols)) * 2, np.float16),
              "w": np.asarray(rng.uniform(0.8, 1.2, (cols,)), np.float16)}
    out, _ = _run(module, inputs)  # driver applies the pipeline
    from npu_compiler import backend_0818
    raw_asm, raw_mp = backend_0818.compile_module(module, validate=False)
    raw_asm.execution_target = "source-0818"
    raw = np.asarray(driver.run_compiled(raw_asm, raw_mp, inputs), np.float16)
    assert np.array_equal(out, raw), "pipeline changed primitive-graph results"
    print("  [PASS] pipeline is value-preserving on primitive graphs")


if __name__ == "__main__":
    test_pipeline_preserves_primitive_graphs()
    test_lowering_matches_legalize()
    test_ones_weight_skips_multiply()
    test_no_highlevel_ops_survive()
    print("ALL NPU PASS PIPELINE TESTS PASSED")
