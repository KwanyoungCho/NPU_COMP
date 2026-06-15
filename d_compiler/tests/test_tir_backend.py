"""T0: TIR+tensorize backend (Relax -> LegalizeOps -> TIR -> schedule/tensorize
-> walker -> ISA) must produce the SAME numeric results as the existing direct
codegen (the oracle), and stay hardware-legal.

Checks:
  (1) TIR-path result == direct(tile=64) result, byte-exact (same FP16 rounding order)
  (2) TIR-path result == tiled_fp16_reference, byte-exact
  (3) every m_mul tile <= 64x64
  (4) multi-matmul module (chained) also works through the TIR path
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from tvm import relax
from npu_compiler import driver
from npu_compiler import tir_backend


def _fp16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def make_matmul_mod(M, K, N):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([M, K], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([K, N], "float16"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x, w)); gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def make_chain_mod(M, K, N, P):
    """y = (x @ w1) @ w2 — two call_tirs through the TIR path."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([M, K], "float16"))
    w1 = relax.Var("w1", relax.TensorStructInfo([K, N], "float16"))
    w2 = relax.Var("w2", relax.TensorStructInfo([N, P], "float16"))
    with bb.function("main", [x, w1, w2]):
        with bb.dataflow():
            t = bb.emit(relax.op.matmul(x, w1))
            y = bb.emit(relax.op.matmul(t, w2))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def tiled_fp16_ref(A, B, T=64):
    K = A.shape[1]
    C = None
    for kk in range(0, K, T):
        part = _fp16(A[:, kk:kk + T].astype(np.float32) @ B[kk:kk + T, :].astype(np.float32))
        C = part if C is None else _fp16(C + part)
    return C


def test_vs_direct_and_ref():
    out = {}
    for (M, K, N) in [(64, 64, 64), (128, 192, 128), (64, 256, 128), (192, 64, 64)]:
        rng = np.random.default_rng(M + K + N)
        x = _fp16(rng.standard_normal((M, K)) * 0.3)
        w = _fp16(rng.standard_normal((K, N)) * 0.1)
        mod = make_matmul_mod(M, K, N)
        got_tir = driver.run_module(mod, {"x": x, "w": w}, backend="tir")
        got_direct = driver.run_module(mod, {"x": x, "w": w}, tile=64)
        ref = tiled_fp16_ref(x, w)
        assert np.array_equal(got_tir, got_direct), f"{M}x{K}x{N}: TIR != direct"
        assert np.array_equal(got_tir, ref), f"{M}x{K}x{N}: TIR != tiled_fp16_ref"
        out[f"{M}x{K}@{K}x{N}"] = "byte-exact"
    return out


def test_legality_and_count():
    M, K, N = 128, 192, 128
    asm, _ = tir_backend.compile_func(make_matmul_mod(M, K, N))
    bad = []
    n_mm = 0
    for w in asm.words:
        op = w & 0xFF
        if op == 0x88:
            d1 = (w >> 8) & 0xFF; d2 = (w >> 16) & 0xFF
            if d1 > 64 or d2 > 64:
                bad.append((d1, d2))
        if op == 0x42 and (w >> 30) & 3 == 2:
            n_mm += 1
    assert not bad, f"illegal tiles: {bad}"
    expect_mm = (M // 64) * (N // 64) * (K // 64)
    assert n_mm == expect_mm, f"matmul count {n_mm} != {expect_mm}"
    return len(asm.words), n_mm


def make_mm_ew_graph(M, K, N):
    """y = (x @ w) * s + b — matmul + elementwise, to test the hybrid path."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([M, K], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([K, N], "float16"))
    s = relax.Var("s", relax.TensorStructInfo([M, N], "float16"))
    b = relax.Var("b", relax.TensorStructInfo([M, N], "float16"))
    with bb.function("main", [x, w, s, b]):
        with bb.dataflow():
            t = bb.emit(relax.op.matmul(x, w))
            u = bb.emit(relax.op.multiply(t, s))
            y = bb.emit(relax.op.add(u, b))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def test_hybrid_vs_direct():
    """Whole graph (matmul + elementwise): hybrid (matmul->TIR, rest->direct)
    must equal the all-direct result, byte-exact."""
    M, K, N = 128, 128, 128
    rng = np.random.default_rng(0)
    ins = {"x": _fp16(rng.standard_normal((M, K)) * 0.3),
           "w": _fp16(rng.standard_normal((K, N)) * 0.1),
           "s": _fp16(rng.standard_normal((M, N)) * 0.5),
           "b": _fp16(rng.standard_normal((M, N)) * 0.2)}
    mod = make_mm_ew_graph(M, K, N)
    gd = driver.run_module(mod, ins, tile=64)
    gh = driver.run_module(mod, ins, backend="hybrid")
    assert np.array_equal(gd, gh), "hybrid != direct"
    return "byte-exact"


def test_chain():
    M, K, N, P = 64, 128, 64, 64
    rng = np.random.default_rng(5)
    x = _fp16(rng.standard_normal((M, K)) * 0.3)
    w1 = _fp16(rng.standard_normal((K, N)) * 0.1)
    w2 = _fp16(rng.standard_normal((N, P)) * 0.1)
    mod = make_chain_mod(M, K, N, P)
    got = driver.run_module(mod, {"x": x, "w1": w1, "w2": w2}, backend="tir")
    ref = tiled_fp16_ref(tiled_fp16_ref(x, w1), w2)
    assert np.array_equal(got, ref), "chained matmul mismatch"
    return "byte-exact"


if __name__ == "__main__":
    print("[PASS] TIR vs direct vs ref:", test_vs_direct_and_ref())
    ninstr, nmm = test_legality_and_count()
    print(f"[PASS] legality: all tiles <=64; 128x192@192x128 -> {ninstr} instrs, {nmm} matmul tiles")
    print("[PASS] chained matmuls:", test_chain())
    print("[PASS] hybrid (matmul->TIR, elementwise->direct):", test_hybrid_vs_direct())
    print("ALL TIR-BACKEND (T0/T1/T2) TESTS PASSED")
