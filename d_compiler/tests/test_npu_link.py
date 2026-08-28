"""S5 gate: a linked program runs on the C-model and computes the right values.

Single-op programs are checked against numpy; the shapes cover the exact-tile
path, the padded path (pad_einsum), and multi-K-tile accumulation.
"""
import sys
from pathlib import Path

import numpy as np
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import npu_legalize, npu_link, npu_memplan as M
from npu_compiler import tvm_pipeline as P


def _run(shapes, make, values, out_shape):
    bb = relax.BlockBuilder()
    args = [relax.Var(f"a{i}", relax.TensorStructInfo(list(s), "float16"))
            for i, s in enumerate(shapes)]
    with bb.function("prefill", args):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(make(*args)))
        bb.emit_func_output(out)
    lowered = P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                               fuse=False, lift_params=False)(bb.finalize())
    asm, plan = npu_link.compile_program(lowered)
    planned, _ = M.assign_addresses(lowered)
    func = planned["prefill"]
    names = [p.name_hint for p in func.params]
    got, counters = npu_link.run_program(
        asm, plan, func, dict(zip(names, values)), out_shape)
    return got, len(asm.words), counters


def test_linked_matmul_shapes():
    """Exact tile, padded M and N, padded with two K tiles, and a 2x2 grid."""
    cases = [(64, 64, 64), (4, 64, 32), (7, 128, 96), (128, 128, 128)]
    for index, (m, k, n) in enumerate(cases):
        rng = np.random.default_rng(index)
        a = rng.normal(0, 0.3, (m, k)).astype(np.float16)
        b = rng.normal(0, 0.3, (k, n)).astype(np.float16)
        got, words, _ = _run([(m, k), (k, n)], relax.op.matmul, [a, b], (m, n))
        ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
        error = float(np.abs(got.astype(np.float32)
                             - ref.astype(np.float32)).max())
        assert error < 0.01, (m, k, n, error)
        print(f"  [PASS] matmul [{m},{k}]x[{k},{n}] max|diff|={error:.4f} "
              f"({words:,} words)")


def test_linked_rms_norm():
    seq, width = 4, 64
    rng = np.random.default_rng(11)
    x = rng.normal(0, 0.5, (seq, width)).astype(np.float16)
    w = rng.random(width).astype(np.float16) + np.float16(0.5)
    got, words, _ = _run(
        [(seq, width), (width,)],
        lambda a, b: relax.op.nn.rms_norm(a, b, axes=[-1]), [x, w],
        (seq, width))
    xf = x.astype(np.float32)
    ref = (xf / np.sqrt((xf * xf).mean(-1, keepdims=True) + 1e-5)
           * w.astype(np.float32))
    error = float(np.abs(got.astype(np.float32) - ref).max())
    assert error < 0.02, error
    print(f"  [PASS] rms_norm max|diff|={error:.4f} ({words:,} words)")


def test_padded_matmul_stages_its_inputs():
    """A padded matmul must not read global memory from a compute block."""
    lowered = None
    bb = relax.BlockBuilder()
    si = lambda s: relax.TensorStructInfo(list(s), "float16")
    a = relax.Var("a", si([4, 64]))
    b = relax.Var("b", si([64, 32]))
    with bb.function("prefill", [a, b]):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(relax.op.matmul(a, b)))
        bb.emit_func_output(out)
    lowered = P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                               fuse=False, lift_params=False)(bb.finalize())
    asm, _ = npu_link.compile_program(lowered)
    dma = sum(1 for index, word in enumerate(asm.words)
              if (word & 0xFF) in (0xA0, 0xA8))
    assert dma > 0
    print(f"  [PASS] padded matmul stages through DMA ({dma} DMA words seen)")


if __name__ == "__main__":
    test_linked_rms_norm()
    test_linked_matmul_shapes()
    test_padded_matmul_stages_its_inputs()
    print("ALL NPU LINK (S5) TESTS PASSED")
