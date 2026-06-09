"""M2: first TVM Relax -> NPU ISA -> mysim end-to-end.

Build a single-matmul Relax module, compile it to NPU instructions with our
codegen, run on the GIVEN mysim, and compare to a FP16 reference.
B0 logical: dims <= 255, no 64x64 tiling.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

import tvm
from tvm import relax
from npu_compiler import driver


def _fp16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def make_matmul_mod(M, K, N):
    """Parametric single-matmul Relax module (BlockBuilder; TVMScript can't capture
    closure dims in shape annotations)."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([M, K], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([K, N], "float16"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x, w))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def test_matmul_8x64_64x64():
    M, K, N = 8, 64, 64
    rng = np.random.default_rng(0)
    x = _fp16(rng.standard_normal((M, K)) * 0.5)
    w = _fp16(rng.standard_normal((K, N)) * 0.1)

    mod = make_matmul_mod(M, K, N)
    got = driver.run_module(mod, {"x": x, "w": w})

    ref = _fp16(x @ w)                      # FP16-input reference
    maxerr = float(np.max(np.abs(got - ref)))
    rel = maxerr / (float(np.max(np.abs(ref))) + 1e-6)
    assert got.shape == (M, N)
    assert rel < 0.02, f"rel={rel} maxerr={maxerr}"
    return rel, maxerr


def test_matmul_various():
    out = {}
    for (M, K, N) in [(1, 16, 16), (4, 32, 8), (8, 64, 64), (16, 16, 255)]:
        rng = np.random.default_rng(M * 100 + K + N)
        x = _fp16(rng.standard_normal((M, K)) * 0.5)
        w = _fp16(rng.standard_normal((K, N)) * 0.1)
        got = driver.run_module(make_matmul_mod(M, K, N), {"x": x, "w": w})
        ref = _fp16(x @ w)
        rel = float(np.max(np.abs(got - ref))) / (float(np.max(np.abs(ref))) + 1e-6)
        assert rel < 0.05, f"{M}x{K}x{N}: rel={rel}"
        out[f"{M}x{K}@{K}x{N}"] = round(rel, 4)
    return out


if __name__ == "__main__":
    rel, maxerr = test_matmul_8x64_64x64()
    print(f"[PASS] matmul 8x64@64x64 e2e: rel={rel:.4g} maxerr={maxerr:.4g}")
    out = test_matmul_various()
    print(f"[PASS] matmul various: {out}")
    print("ALL M2 TESTS PASSED")
