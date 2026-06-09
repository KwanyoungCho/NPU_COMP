"""M4: RMSNorm through legalize (reduce/broadcast via ones-matmul) -> ISA -> mysim.

First use of legalize.py + constant (ones) placement in the G-buffer.
Compared to a float RMSNorm reference (FP16 tolerance).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from tvm import relax
from npu_compiler import driver, legalize


def _fp16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def make_rmsnorm_mod(seq, d):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([seq, d], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([1, d], "float16"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            y = legalize.rms_norm(bb, x, w, seq, d)
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def rmsnorm_ref(x, w, d):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms)) * w


def test_rmsnorm_8x64():
    seq, d = 8, 64
    rng = np.random.default_rng(7)
    x = _fp16(rng.standard_normal((seq, d)))
    w = _fp16(rng.uniform(0.8, 1.2, (1, d)))

    got = driver.run_module(make_rmsnorm_mod(seq, d), {"x": x, "w": w})
    exp = rmsnorm_ref(x, w, d)

    maxabs = float(np.max(np.abs(exp)))
    maxerr = float(np.max(np.abs(got - exp)))
    rel = maxerr / (maxabs + 1e-6)
    nmis = int(np.sum(np.abs(got - exp) > 1e-3))
    assert maxerr < 0.05 * maxabs + 0.02, f"maxerr={maxerr} maxabs={maxabs}"
    return rel, maxerr, nmis, got.size


if __name__ == "__main__":
    rel, maxerr, nmis, n = test_rmsnorm_8x64()
    print(f"[PASS] RMSNorm 8x64 e2e: rel={rel:.4g} maxerr={maxerr:.4g} mism(>1e-3)={nmis}/{n}")
    print("ALL M4 TESTS PASSED")
