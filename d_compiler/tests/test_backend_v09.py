"""N4 gate: backend_v09 (SRAM staging codegen) bit-exact vs the 0818 oracle.

Every case compiles one Relax module twice -- backend="source-0818" (frozen
oracle) and backend="v09" -- and requires bitwise-identical FP16 results.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from tvm import relax

from npu_compiler import backend_v09, driver, passes


def _sinfo(shape):
    return relax.TensorStructInfo(list(shape), "float16")


def _assert_match(mod, inputs):
    oracle = driver.run_module(mod, inputs, backend="source-0818")
    ours = driver.run_module(mod, inputs, backend="v09")
    a = np.asarray(oracle, np.float16)
    b = np.asarray(ours, np.float16)
    assert a.shape == b.shape
    np.testing.assert_array_equal(a.view(np.uint16), b.view(np.uint16))
    return b


def _proxy_layer_module(s=7, d=96, hd=64):
    """Odd-sized attention + MLP fragment covering every backend kernel."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", _sinfo([s, d]))
    wq = relax.Var("wq", _sinfo([d, hd]))
    wk = relax.Var("wk", _sinfo([d, hd]))
    wv = relax.Var("wv", _sinfo([d, hd]))
    w2 = relax.Var("w2", _sinfo([hd, d]))
    with bb.function("main", [x, wq, wk, wv, w2]):
        with bb.dataflow():
            q = bb.emit(relax.op.matmul(x, wq))
            k = bb.emit(relax.op.matmul(x, wk))
            kT = bb.emit(relax.op.permute_dims(k))
            scores = bb.emit(relax.op.matmul(q, kT))           # [s, s] odd
            mx = bb.emit(relax.op.max(scores, axis=[1], keepdims=True))
            mb = bb.emit(relax.op.broadcast_to(mx, relax.ShapeExpr([s, s])))
            shifted = bb.emit(relax.op.subtract(scores, mb))
            e = bb.emit(relax.op.exp(shifted))
            sm = bb.emit(relax.op.sum(e, axis=[1], keepdims=True))
            sb = bb.emit(relax.op.broadcast_to(sm, relax.ShapeExpr([s, s])))
            probs = bb.emit(relax.op.divide(e, sb))
            v = bb.emit(relax.op.matmul(x, wv))
            o = bb.emit(relax.op.matmul(probs, v))
            g = bb.emit(relax.op.nn.silu(o))
            cat = bb.emit(relax.op.concat([o, g], axis=1))     # [s, 2*hd]
            sl = bb.emit(relax.op.strided_slice(
                cat, axes=[1], begin=[hd // 2], end=[hd // 2 + hd]))
            y = bb.emit(relax.op.matmul(sl, w2))               # [s, d]
            out = bb.emit(relax.op.add(x, y))
            output = bb.emit_output(out)
        bb.emit_func_output(output)
    return bb.finalize()


def test_proxy_layer_bit_exact():
    s, d, hd = 7, 96, 64
    rng = np.random.default_rng(409)
    inputs = {
        "x": rng.normal(0, 0.5, (s, d)).astype(np.float16),
        "wq": rng.normal(0, 0.2, (d, hd)).astype(np.float16),
        "wk": rng.normal(0, 0.2, (d, hd)).astype(np.float16),
        "wv": rng.normal(0, 0.2, (d, hd)).astype(np.float16),
        "w2": rng.normal(0, 0.2, (hd, d)).astype(np.float16),
    }
    _assert_match(_proxy_layer_module(s, d, hd), inputs)


def test_long_row_reduce_chunking():
    """cols=600 rows exercise the 256-lane carry chain vs one flat reduce."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", _sinfo([3, 600]))
    with bb.function("main", [x]):
        with bb.dataflow():
            s = bb.emit(relax.op.sum(x, axis=[1], keepdims=True))
            m = bb.emit(relax.op.max(x, axis=[1], keepdims=True))
            out = bb.emit(relax.op.add(s, m))
            output = bb.emit_output(out)
        bb.emit_func_output(output)
    mod = bb.finalize()
    rng = np.random.default_rng(11)
    values = -np.abs(rng.normal(0, 1, (3, 600))).astype(np.float16)  # all < 0
    _assert_match(mod, {"x": values})


def test_long_elementwise_chunking():
    bb = relax.BlockBuilder()
    x = relax.Var("x", _sinfo([4, 300]))
    y = relax.Var("y", _sinfo([4, 300]))
    with bb.function("main", [x, y]):
        with bb.dataflow():
            p = bb.emit(relax.op.multiply(x, y))
            q = bb.emit(relax.op.exp(p))
            output = bb.emit_output(bb.emit(relax.op.add(q, x)))
        bb.emit_func_output(output)
    mod = bb.finalize()
    rng = np.random.default_rng(21)
    inputs = {"x": rng.normal(0, 0.3, (4, 300)).astype(np.float16),
              "y": rng.normal(0, 0.3, (4, 300)).astype(np.float16)}
    _assert_match(mod, inputs)


def test_streaming_matmul_paths():
    """A tiny whole_limit forces the lhs-panel / rhs-tile / dst-tile streams."""
    m, k, n = 65, 96, 64
    bb = relax.BlockBuilder()
    lhs = relax.Var("lhs", _sinfo([m, k]))
    rhs = relax.Var("rhs", _sinfo([k, n]))
    with bb.function("main", [lhs, rhs]):
        with bb.dataflow():
            output = bb.emit_output(bb.emit(relax.op.matmul(lhs, rhs)))
        bb.emit_func_output(output)
    mod = bb.finalize()
    rng = np.random.default_rng(31)
    inputs = {"lhs": rng.normal(0, 0.2, (m, k)).astype(np.float16),
              "rhs": rng.normal(0, 0.2, (k, n)).astype(np.float16)}
    oracle = driver.run_module(mod, inputs, backend="source-0818")

    lowered = passes.npu_pipeline()(mod)
    func = lowered["main"]
    mp = backend_v09.plan(func)
    asm = backend_v09.compile_func(func, mp, whole_limit=512)
    assert any((word & 0xFF) == 0xA0 for word in asm.words)
    got = driver.run_compiled(asm, mp, inputs)
    np.testing.assert_array_equal(
        np.asarray(oracle, np.float16).view(np.uint16),
        np.asarray(got, np.float16).view(np.uint16))


def test_scalar_broadcast_path():
    bb = relax.BlockBuilder()
    x = relax.Var("x", _sinfo([1, 1]))
    with bb.function("main", [x]):
        with bb.dataflow():
            b = bb.emit(relax.op.broadcast_to(x, relax.ShapeExpr([5, 9])))
            output = bb.emit_output(bb.emit(relax.op.sqrt(b)))
        bb.emit_func_output(output)
    mod = bb.finalize()
    _assert_match(mod, {"x": np.asarray([[2.0]], np.float16)})


def test_packed_rhs_gemm_matches_0818():
    """The v09 wide-vocab panel GEMM is bit-exact with the ver.08 lowering."""
    from npu_compiler.source_gemm_0818 import PackedRhsGemm
    from npu_compiler.source_gemm_v09 import V09PackedRhsGemm
    inner, columns = 96, 192          # 3 panels, 2 K-tiles with MAC
    rng = np.random.default_rng(77)
    lhs = rng.normal(0, 0.3, inner).astype(np.float16)
    packed = rng.normal(0, 0.3, inner * columns).astype(np.float16)
    oracle = PackedRhsGemm(inner, columns).run(lhs, packed)
    ours = V09PackedRhsGemm(inner, columns).run(lhs, packed)
    np.testing.assert_array_equal(oracle.view(np.uint16), ours.view(np.uint16))


if __name__ == "__main__":
    test_proxy_layer_bit_exact()
    test_long_row_reduce_chunking()
    test_long_elementwise_chunking()
    test_streaming_matmul_paths()
    test_scalar_broadcast_path()
    test_packed_rhs_gemm_matches_0818()
    print("ALL V09 BACKEND N4 TESTS PASSED")
