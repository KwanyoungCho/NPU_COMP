"""M3: elementwise vector ops through the full pipeline (Relax -> ISA -> mysim).

Binary (add/sub/mul/div) + unary (sqrt/exp), same-shape (no broadcast yet),
plus a small chain to exercise intermediate-buffer allocation in memplan.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from tvm import relax
from npu_compiler import driver


def _fp16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def _sinfo(shape):
    return relax.TensorStructInfo(list(shape), "float16")


def make_binary(opfn, shape):
    bb = relax.BlockBuilder()
    a = relax.Var("a", _sinfo(shape)); b = relax.Var("b", _sinfo(shape))
    with bb.function("main", [a, b]):
        with bb.dataflow():
            y = bb.emit(opfn(a, b)); gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def make_unary(opfn, shape):
    bb = relax.BlockBuilder()
    a = relax.Var("a", _sinfo(shape))
    with bb.function("main", [a]):
        with bb.dataflow():
            y = bb.emit(opfn(a)); gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def make_chain(shape):
    """y = (a + b) * c  -> two ops, one intermediate buffer."""
    bb = relax.BlockBuilder()
    a = relax.Var("a", _sinfo(shape)); b = relax.Var("b", _sinfo(shape)); c = relax.Var("c", _sinfo(shape))
    with bb.function("main", [a, b, c]):
        with bb.dataflow():
            s = bb.emit(relax.op.add(a, b))
            y = bb.emit(relax.op.multiply(s, c))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def test_binary_ops():
    shape = (8, 16)
    rng = np.random.default_rng(1)
    res = {}
    cases = [("add", relax.op.add, lambda a, b: a + b),
             ("sub", relax.op.subtract, lambda a, b: a - b),
             ("mul", relax.op.multiply, lambda a, b: a * b),
             ("div", relax.op.divide, lambda a, b: a / b)]
    for name, opfn, ref in cases:
        a = _fp16(rng.standard_normal(shape))
        b = _fp16(rng.standard_normal(shape) + 1.5)        # away from 0 for div
        got = driver.run_module(make_binary(opfn, shape), {"a": a, "b": b})
        exp = _fp16(ref(a, b))
        rel = float(np.max(np.abs(got - exp))) / (float(np.max(np.abs(exp))) + 1e-6)
        assert rel < 0.02, f"{name}: rel={rel}"
        res[name] = round(rel, 4)
    return res


def test_unary_ops():
    shape = (8, 16)
    rng = np.random.default_rng(2)
    res = {}
    a_pos = _fp16(np.abs(rng.standard_normal(shape)) + 0.1)
    got = driver.run_module(make_unary(relax.op.sqrt, shape), {"a": a_pos})
    res["sqrt"] = round(float(np.max(np.abs(got - _fp16(np.sqrt(a_pos))))) / (float(np.max(np.abs(_fp16(np.sqrt(a_pos))))) + 1e-6), 4)
    assert res["sqrt"] < 0.02
    a_small = _fp16(rng.standard_normal(shape) * 0.8)
    got = driver.run_module(make_unary(relax.op.exp, shape), {"a": a_small})
    exp = _fp16(np.exp(a_small))
    res["exp"] = round(float(np.max(np.abs(got - exp))) / (float(np.max(np.abs(exp))) + 1e-6), 4)
    assert res["exp"] < 0.02
    return res


def test_chain():
    shape = (8, 16)
    rng = np.random.default_rng(3)
    a = _fp16(rng.standard_normal(shape)); b = _fp16(rng.standard_normal(shape)); c = _fp16(rng.standard_normal(shape))
    got = driver.run_module(make_chain(shape), {"a": a, "b": b, "c": c})
    exp = _fp16(_fp16(a + b) * c)                          # intermediate is FP16-rounded by NPU save
    rel = float(np.max(np.abs(got - exp))) / (float(np.max(np.abs(exp))) + 1e-6)
    assert rel < 0.02, f"chain rel={rel}"
    return round(rel, 4)


if __name__ == "__main__":
    print("[PASS] binary:", test_binary_ops())
    print("[PASS] unary :", test_unary_ops())
    print("[PASS] chain (a+b)*c rel:", test_chain())
    print("ALL M3 TESTS PASSED")
