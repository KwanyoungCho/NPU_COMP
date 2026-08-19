"""Standard tanh-GELU lowering micro-test (Gemma correctness path, V3-004/G4-001).

Runs the primitive-sequence gelu_tanh on the vendor 0818 executable (authority),
checks source C-model parity, an FP16 step-emulated reference bit-exactly, and
the float64 formula within FP16 tolerance, including saturation extremes.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from tvm import relax

from npu_compiler import driver, legalize


def build_module(rows, cols):
    x = relax.Var("x", relax.TensorStructInfo([rows, cols], "float16"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            gv = bb.emit_output(legalize.gelu_tanh(bb, x, rows, cols))
        bb.emit_func_output(gv)
    return bb.finalize()


def f16(value):
    return np.asarray(value, dtype=np.float16).astype(np.float32)


def emulated_reference(x):
    """Replicate the lowering op-by-op with an FP16 G-buffer save after each."""
    c2 = float(2.0 * np.sqrt(2.0 / np.pi))
    x = f16(x)
    xx = f16(x * x)
    poly = f16(xx * f16(np.full_like(x, c2 * 0.044715)))
    poly = f16(poly + f16(np.full_like(x, c2)))
    t2 = f16(x * poly)
    e = f16(np.exp(f16(-t2)))
    den = f16(e + 1.0)
    with np.errstate(divide="ignore"):
        sig = f16(1.0 / den)
    return f16(x * sig)


def exact_reference(x):
    x = np.asarray(x, dtype=np.float64)
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def run_case(x, backend):
    module = build_module(*x.shape)
    compiled = driver.compile_module(module, backend=backend)
    return np.asarray(driver.run_compiled(*compiled, {"x": x}), dtype=np.float32)


def test_gelu_tanh():
    cases = {
        "dense": np.linspace(-8.0, 8.0, 64, dtype=np.float16).reshape(1, 64),
        "near_zero": np.linspace(-0.25, 0.25, 64, dtype=np.float16).reshape(1, 64),
        "extreme": np.asarray([[
            -30000.0, -1000.0, -100.0, -40.0, -20.0, -12.0, -9.0, -6.0,
            -4.0, -2.0, -1.0, -0.5, -0.0625, 0.0, 0.0625, 0.5,
            1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 20.0, 40.0,
            100.0, 1000.0, 30000.0, 55.0, -55.0, 3.14159, -3.14159, 0.001,
        ]], dtype=np.float16),
    }
    for name, x in cases.items():
        vendor = run_case(x, "0818")
        source = run_case(x, "source-0818")
        emulated = emulated_reference(x)
        exact = exact_reference(x.astype(np.float64))
        assert np.array_equal(vendor, source), f"{name}: vendor/source mismatch"
        assert np.array_equal(vendor, emulated), f"{name}: FP16 emulation mismatch"
        assert np.isfinite(vendor).all(), f"{name}: non-finite gelu output"
        error = np.max(np.abs(vendor.astype(np.float64) - exact))
        # FP16 argument/step rounding bounds the error; saturated inputs are exact.
        tolerance = 1e-3 + 1e-3 * np.max(np.abs(exact))
        assert error <= tolerance, f"{name}: max abs {error} > {tolerance}"
        print(f"  [PASS] {name}: vendor==source==fp16-emulation, "
              f"max abs vs float64 {error:.3e}")

    # Saturation contract: large positive passes through, large negative is zero.
    x = np.asarray([[300.0, 3000.0, -300.0, -3000.0]], dtype=np.float16)
    vendor = run_case(x, "0818")
    assert np.array_equal(vendor[0, :2], f16([300.0, 3000.0]))
    assert np.array_equal(vendor[0, 2:], np.zeros(2, dtype=np.float32))

    # The lowering must differ from the native vendor GELU where the two
    # formulas disagree (x*sigmoid(2x) vs standard tanh-GELU).
    probe = np.asarray([[-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]], dtype=np.float16)
    vendor_native = probe.astype(np.float64) / (
        1.0 + np.exp(-2.0 * probe.astype(np.float64)))
    ours = run_case(probe, "0818").astype(np.float64)
    assert np.max(np.abs(ours - vendor_native)) > 1e-3


if __name__ == "__main__":
    test_gelu_tanh()
    print("ALL TANH-GELU LOWERING TESTS PASSED")
