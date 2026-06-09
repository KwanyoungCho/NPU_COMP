"""M6: SwiGLU FFN (SiLU built from exp/add/div/mul) -> ISA -> mysim.

Exercises the silu legalization. Compared to a float SwiGLU reference.
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


def make_swiglu_mod(seq, d, f):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([seq, d], "float16"))
    Wg = relax.Var("Wg", relax.TensorStructInfo([d, f], "float16"))
    Wu = relax.Var("Wu", relax.TensorStructInfo([d, f], "float16"))
    Wd = relax.Var("Wd", relax.TensorStructInfo([f, d], "float16"))
    with bb.function("main", [x, Wg, Wu, Wd]):
        with bb.dataflow():
            y = legalize.swiglu(bb, x, Wg, Wu, Wd, seq, d, f)
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def swiglu_ref(x, Wg, Wu, Wd):
    gate = x @ Wg
    up = x @ Wu
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up) @ Wd


def test_swiglu_8x64_128():
    seq, d, f = 8, 64, 128
    rng = np.random.default_rng(11)
    ws = 0.2
    x = _fp16(rng.standard_normal((seq, d)) * 0.5)
    Wg = _fp16(rng.uniform(-ws, ws, (d, f)))
    Wu = _fp16(rng.uniform(-ws, ws, (d, f)))
    Wd = _fp16(rng.uniform(-ws, ws, (f, d)))

    got = driver.run_module(make_swiglu_mod(seq, d, f), {"x": x, "Wg": Wg, "Wu": Wu, "Wd": Wd})
    exp = swiglu_ref(x.astype(np.float64), Wg.astype(np.float64), Wu.astype(np.float64), Wd.astype(np.float64))

    maxabs = float(np.max(np.abs(exp)))
    maxerr = float(np.max(np.abs(got - exp)))
    rel = maxerr / (maxabs + 1e-6)
    assert maxerr < 0.05 * maxabs + 0.02, f"maxerr={maxerr} maxabs={maxabs}"
    return rel, maxerr, got.size


if __name__ == "__main__":
    rel, maxerr, n = test_swiglu_8x64_128()
    print(f"[PASS] SwiGLU [8,64]->128 e2e: rel={rel:.4g} maxerr={maxerr:.4g} n={n}")
    print("ALL M6 TESTS PASSED")
