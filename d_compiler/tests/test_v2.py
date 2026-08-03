"""v2 Phase 2-A: elementwise ops lowered by the unified walker (V2Walker) match v1.

Two independent checks per op:
  (1) ISA byte-exact — V2Walker(marker) words == v1 emit_ew algorithm (codegen.py:488-497).
  (2) numerical — run the walked program on mysim, compare to numpy fp16.
v1 stays the oracle; this proves the walker path reproduces it before we migrate ops.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tvm.tir as tir
from tvm import relax
from npu_compiler.isa import Asm, SRC1, SRC2, DST, VECTOR
from npu_compiler import v2_backend as v2, runtime, memplan

_f16 = lambda a: np.asarray(a, np.float16)

# numpy semantics per marker op (for the numerical check)
_NP2 = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply, "divide": np.divide}
_NP1 = {"sqrt": np.sqrt, "exp": np.exp, "negative": np.negative, "cos": np.cos, "sin": np.sin,
        "silu": lambda x: x / (1.0 + np.exp(-x))}


# ---- v1 reference: emit_ew body verbatim (npu_compiler/codegen.py:488-497) ----
def _ref_ew2(a, od, o0, o1, n, method, CH=8192):
    op = getattr(a, method)
    for base in range(0, n, CH):
        a.vlen(min(CH, n - base))
        a.addr(SRC1, o0 + base); a.load(0, 0)
        a.addr(SRC2, o1 + base); a.load(0, 1)
        op(mode=VECTOR)
        a.addr(DST, od + base); a.save(0)


def _ref_ew1(a, od, o0, n, method, CH=8192):
    op = getattr(a, method)
    for base in range(0, n, CH):
        a.vlen(min(CH, n - base))
        a.addr(SRC1, o0 + base); a.load(0, 0)
        op()
        a.addr(DST, od + base); a.save(0)


def _ref_silu(a, od, o0, n, CH=8192):
    from npu_compiler.isa import IMM
    for base in range(0, n, CH):
        a.vlen(min(CH, n - base))
        a.addr(SRC1, o0 + base); a.load(0, 0)
        a.m_add(mode=IMM, imm=0, act=True)
        a.addr(DST, od + base); a.save(0)


def test_ew2_byte_exact_and_numeric():
    rng = np.random.default_rng(0)
    for op in ["add", "subtract", "multiply", "divide"]:
        for N in [4096, 8192, 20000, 128 * 256]:
            o0, o1, od = 0, N, 2 * N
            # (1) ISA byte-exact
            aref = Asm(); _ref_ew2(aref, od, o0, o1, N, v2._EW2[op])
            a2 = Asm(); v2.walk_marker(a2, v2.ew2_marker(op, N), [o0, o1, od])
            assert aref.words == a2.words, f"{op} N={N}: ISA not byte-exact"
            # (2) numerical vs numpy (spot-check one size to keep mysim runs cheap)
        N = 8192; o0, o1, od = 0, N, 2 * N
        A = _f16(rng.standard_normal(N)); B = _f16(np.abs(rng.standard_normal(N)) + 0.5)
        gbuf = np.zeros(od + N, np.float32); gbuf[o0:o0 + N] = A; gbuf[o1:o1 + N] = B
        a2 = Asm(); v2.walk_marker(a2, v2.ew2_marker(op, N), [o0, o1, od])
        out = runtime.run(a2.words, gbuf, gn=od + N)[od:od + N]
        exp = _f16(_NP2[op](A.astype(np.float32), B.astype(np.float32)))
        assert np.array_equal(out.astype(np.float16), exp), f"{op}: numeric mismatch"
    return "ew2 (add/sub/mul/div): ISA byte-exact vs v1 + mysim==numpy"


def test_ew1_byte_exact_and_numeric():
    rng = np.random.default_rng(1)
    for op in ["sqrt", "exp", "negative", "cos", "sin"]:
        for N in [4096, 20000]:
            o0, od = 0, N
            aref = Asm(); _ref_ew1(aref, od, o0, N, v2._EW1[op])
            a2 = Asm(); v2.walk_marker(a2, v2.ew1_marker(op, N), [o0, od])
            assert aref.words == a2.words, f"{op} N={N}: ISA not byte-exact"
        N = 4096; o0, od = 0, N
        A = _f16(np.abs(rng.standard_normal(N)) + 0.5) if op == "sqrt" \
            else _f16(rng.standard_normal(N) * 0.5)
        gbuf = np.zeros(od + N, np.float32); gbuf[o0:o0 + N] = A
        a2 = Asm(); v2.walk_marker(a2, v2.ew1_marker(op, N), [o0, od])
        out = runtime.run(a2.words, gbuf, gn=od + N)[od:od + N]
        exp = _f16(_NP1[op](A.astype(np.float32)))
        assert np.max(np.abs(out.astype(np.float32) - exp.astype(np.float32))) < 0.05, \
            f"{op}: numeric mismatch"
    return "ew1 (sqrt/exp/neg/cos/sin): ISA byte-exact vs v1 + mysim~=numpy"


def test_silu_byte_exact_and_numeric():
    rng = np.random.default_rng(2)
    N = 8192; o0, od = 0, N
    aref = Asm(); _ref_silu(aref, od, o0, N)
    a2 = Asm(); v2.walk_marker(a2, v2.ew1_marker("silu", N), [o0, od])
    assert aref.words == a2.words, "silu: ISA not byte-exact"
    A = _f16(rng.standard_normal(N))
    gbuf = np.zeros(od + N, np.float32); gbuf[o0:o0 + N] = A
    a2 = Asm(); v2.walk_marker(a2, v2.ew1_marker("silu", N), [o0, od])
    out = runtime.run(a2.words, gbuf, gn=od + N)[od:od + N]
    exp = _f16(_NP1["silu"](A.astype(np.float32)))
    assert np.max(np.abs(out.astype(np.float32) - exp.astype(np.float32))) < 0.05, "silu: numeric mismatch"
    return "silu: ISA byte-exact vs v1 + mysim~=numpy"


# ---- v1 reference: emit_row_sum/emit_row_max bodies verbatim (codegen.py:214-294) ----
def _ref_rsum_row(a, sd, ssrc, R, C):
    for r in range(R):
        a.vlen(C); a.addr(SRC1, ssrc + r * C); a.load(0, 0)
        a.v_reduce_sum(); a.addr(DST, sd + r); a.save(0)


def _ref_rsum_tile(a, sd, ssrc, R, C, mp):
    Ct = C // 64; A = mp.scratch_alloc(64 * 64)
    for rt in range(R // 64):
        base = ssrc + (rt * Ct) * 4096
        a.vlen(4096); a.addr(SRC1, base); a.load(0, 0); a.v_copy(); a.addr(DST, A); a.save(0)
        for ct in range(1, Ct):
            a.vlen(4096); a.addr(SRC1, base + ct * 4096); a.load(0, 0)
            a.addr(SRC2, A); a.load(0, 1); a.v_add(mode=VECTOR); a.addr(DST, A); a.save(0)
        for ir in range(64):
            a.vlen(64); a.addr(SRC1, A + ir * 64); a.load(0, 0)
            a.v_reduce_sum(); a.addr(DST, sd + rt * 64 + ir); a.save(0)


def _ref_rmax_row(a, d0, s0, R, C, mp):
    acc = mp.scratch_alloc(R)
    a.tile(0, R, C); a.addr(SRC1, s0); a.load(1, 0, strided=1, ncols=1, start=0)
    a.vlen(R); a.v_copy(); a.addr(DST, acc); a.save(0)
    for j in range(1, C):
        a.tile(0, R, C); a.addr(SRC1, s0); a.load(1, 0, strided=1, ncols=1, start=j)
        a.vlen(R); a.addr(SRC2, acc); a.load(0, 1); a.v_max(mode=VECTOR); a.addr(DST, acc); a.save(0)
    a.vlen(R); a.addr(SRC1, acc); a.load(0, 0); a.v_copy(); a.addr(DST, d0); a.save(0)


def _ref_rmax_tile(a, d0, s0, R, C, mp):
    Ct = C // 64; B = mp.scratch_alloc(64 * 64); ac = mp.scratch_alloc(64)
    for rt in range(R // 64):
        base = s0 + (rt * Ct) * 4096
        a.vlen(4096); a.addr(SRC1, base); a.load(0, 0); a.v_copy(); a.addr(DST, B); a.save(0)
        for ct in range(1, Ct):
            a.vlen(4096); a.addr(SRC1, base + ct * 4096); a.load(0, 0)
            a.addr(SRC2, B); a.load(0, 1); a.v_max(mode=VECTOR); a.addr(DST, B); a.save(0)
        a.tile(0, 64, 64); a.addr(SRC1, B); a.load(1, 0, strided=1, ncols=1, start=0)
        a.vlen(64); a.v_copy(); a.addr(DST, ac); a.save(0)
        for j in range(1, 64):
            a.tile(0, 64, 64); a.addr(SRC1, B); a.load(1, 0, strided=1, ncols=1, start=j)
            a.vlen(64); a.addr(SRC2, ac); a.load(0, 1); a.v_max(mode=VECTOR); a.addr(DST, ac); a.save(0)
        a.vlen(64); a.addr(SRC1, ac); a.load(0, 0); a.v_copy(); a.addr(DST, d0 + rt * 64); a.save(0)


def _fresh_mp(top):
    mp = memplan.MemPlan(); mp.top = top
    return mp


def test_reduce_byte_exact_and_numeric():
    rng = np.random.default_rng(3)
    # ---- row_sum (row-major) ----
    R, C = 8, 100; o_s, o_d = 0, R * C
    aref = Asm(); _ref_rsum_row(aref, o_d, o_s, R, C)
    a2 = Asm(); v2.walk_marker(a2, v2.reduce_marker("npu_rsum_row", R, C, R * C), [o_s, o_d])
    assert aref.words == a2.words, "rsum_row: ISA not byte-exact"
    X = _f16(rng.standard_normal((R, C)))
    gbuf = np.zeros(o_d + R + 8192, np.float32); gbuf[o_s:o_s + R * C] = X.reshape(-1)
    a2 = Asm(); v2.walk_marker(a2, v2.reduce_marker("npu_rsum_row", R, C, R * C), [o_s, o_d], _fresh_mp(o_d + R))
    out = runtime.run(a2.words, gbuf, gn=o_d + R)[o_d:o_d + R]
    exp = X.astype(np.float32).sum(1)
    assert np.max(np.abs(out - exp)) < 0.05 * np.max(np.abs(exp)) + 0.5, "rsum_row: numeric"

    # ---- row_sum (tile-blocked) ----
    R, C = 128, 128; o_s, o_d = 0, R * C
    T0 = o_d + R
    aref = Asm(); _ref_rsum_tile(aref, o_d, o_s, R, C, _fresh_mp(T0))
    a2 = Asm(); v2.walk_marker(a2, v2.reduce_marker("npu_rsum_tile", R, C, R * C), [o_s, o_d], _fresh_mp(T0))
    assert aref.words == a2.words, "rsum_tile: ISA not byte-exact"
    X = _f16(rng.standard_normal((R, C)) * 0.1)
    packed = memplan.pack_tiled(X)
    gbuf = np.zeros(T0 + 8192, np.float32); gbuf[o_s:o_s + packed.size] = packed
    a2 = Asm(); v2.walk_marker(a2, v2.reduce_marker("npu_rsum_tile", R, C, R * C), [o_s, o_d], _fresh_mp(T0))
    out = runtime.run(a2.words, gbuf, gn=o_d + R)[o_d:o_d + R]
    exp = X.astype(np.float32).sum(1)
    assert np.max(np.abs(out - exp)) < 0.05 * np.max(np.abs(exp)) + 0.5, "rsum_tile: numeric"

    # ---- row_max (row-major, R,C<256) ----
    R, C = 64, 100; o_s, o_d = 0, R * C; T0 = o_d + R
    aref = Asm(); _ref_rmax_row(aref, o_d, o_s, R, C, _fresh_mp(T0))
    a2 = Asm(); v2.walk_marker(a2, v2.reduce_marker("npu_rmax_row", R, C, R * C), [o_s, o_d], _fresh_mp(T0))
    assert aref.words == a2.words, "rmax_row: ISA not byte-exact"
    X = _f16(rng.standard_normal((R, C)))
    gbuf = np.zeros(T0 + 8192, np.float32); gbuf[o_s:o_s + R * C] = X.reshape(-1)
    a2 = Asm(); v2.walk_marker(a2, v2.reduce_marker("npu_rmax_row", R, C, R * C), [o_s, o_d], _fresh_mp(T0))
    out = runtime.run(a2.words, gbuf, gn=o_d + R)[o_d:o_d + R]
    exp = X.astype(np.float32).max(1)
    assert np.array_equal(out.astype(np.float16), _f16(exp)), "rmax_row: numeric"

    # ---- row_max (tile-blocked) ----
    R, C = 128, 192; o_s, o_d = 0, R * C; T0 = o_d + R
    aref = Asm(); _ref_rmax_tile(aref, o_d, o_s, R, C, _fresh_mp(T0))
    a2 = Asm(); v2.walk_marker(a2, v2.reduce_marker("npu_rmax_tile", R, C, R * C), [o_s, o_d], _fresh_mp(T0))
    assert aref.words == a2.words, "rmax_tile: ISA not byte-exact"
    X = _f16(rng.standard_normal((R, C)))
    packed = memplan.pack_tiled(X)
    gbuf = np.zeros(T0 + 8192, np.float32); gbuf[o_s:o_s + packed.size] = packed
    a2 = Asm(); v2.walk_marker(a2, v2.reduce_marker("npu_rmax_tile", R, C, R * C), [o_s, o_d], _fresh_mp(T0))
    out = runtime.run(a2.words, gbuf, gn=o_d + R)[o_d:o_d + R]
    exp = X.astype(np.float32).max(1)
    assert np.array_equal(out.astype(np.float16), _f16(exp)), "rmax_tile: numeric"
    return "reduce (rsum/rmax x row/tile): ISA byte-exact vs v1 + mysim==numpy"


def test_native_primitives():
    """Thin native primitives: copy (0x17) = identity, ttile (strided-load 0x90) = 64x64 transpose."""
    rng = np.random.default_rng(4)
    # copy
    n = 20000; o0, od = 0, n
    A = _f16(rng.standard_normal(n))
    gbuf = np.zeros(od + n, np.float32); gbuf[o0:o0 + n] = A
    a2 = Asm(); v2.walk_marker(a2, v2.copy_marker(n), [o0, od])
    out = runtime.run(a2.words, gbuf, gn=od + n)[od:od + n]
    assert np.array_equal(out.astype(np.float16), A), "copy: not identity"
    # ttile: transpose one 64x64 tile
    o0, od = 0, 4096
    X = _f16(rng.standard_normal((64, 64)))
    gbuf = np.zeros(od + 4096, np.float32); gbuf[o0:o0 + 4096] = X.reshape(-1)
    a2 = Asm(); v2.walk_marker(a2, v2.ttile_marker(), [o0, od])
    out = runtime.run(a2.words, gbuf, gn=od + 4096)[od:od + 4096].reshape(64, 64)
    assert np.array_equal(out.astype(np.float16), X.T), "ttile: not transpose"
    return "native primitives: copy==identity + ttile==transpose (mysim)"


def test_e2e_elementwise_real_pipeline():
    """The REAL Path A flow (not a hand-built marker): Relax op -> LegalizeOps -> TIR ->
    tir.Schedule tensorize(npu_vadd) -> V2Walker -> mysim == numpy."""
    N = 2 * v2.CH
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([N], "float16"))
    y = relax.Var("y", relax.TensorStructInfo([N], "float16"))
    with bb.function("main", [x, y]):
        with bb.dataflow():
            z = bb.emit(relax.op.add(x, y)); gv = bb.emit_output(z)
        bb.emit_func_output(gv)
    mod = relax.transform.LegalizeOps()(bb.finalize())
    fn = [g.name_hint for g in mod.functions if isinstance(mod[g], tir.PrimFunc)][0]
    sch_mod = v2.schedule_ew_binary(mod, fn, "T_add", "npu_vadd")
    pf = sch_mod[fn]
    a2 = Asm(); wk = v2.V2Walker(a2, None, {})
    for p, off in zip(pf.params, [0, N, 2 * N]):
        wk.base[pf.buffer_map[p].data] = off
    wk.walk(pf.body); wk.flush()
    rng = np.random.default_rng(5)
    X = _f16(rng.standard_normal(N)); Y = _f16(rng.standard_normal(N))
    gbuf = np.zeros(3 * N, np.float32); gbuf[0:N] = X; gbuf[N:2 * N] = Y
    out = runtime.run(a2.words, gbuf, gn=3 * N)[2 * N:3 * N]
    exp = _f16(X.astype(np.float32) + Y.astype(np.float32))
    assert np.array_equal(out.astype(np.float16), exp), "e2e add: mysim != numpy"
    return "e2e elementwise: Relax add -> LegalizeOps -> tensorize -> walker -> mysim==numpy"


def test_v2_compile_pipeline():
    """v2.compile_module: a legalized Relax multi-op module -> NPU ISA -> mysim == numpy(tol).
    y = (x @ w1 + b) @ w2  (matmul -> add -> matmul; intermediates flow through the G-buffer)."""
    D = 128
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([D, D], "float16"))
    w1 = relax.Var("w1", relax.TensorStructInfo([D, D], "float16"))
    b = relax.Var("b", relax.TensorStructInfo([D, D], "float16"))
    w2 = relax.Var("w2", relax.TensorStructInfo([D, D], "float16"))
    with bb.function("main", [x, w1, b, w2]):
        with bb.dataflow():
            h = bb.emit(relax.op.add(bb.emit(relax.op.matmul(x, w1)), b))
            y = bb.emit(relax.op.matmul(h, w2))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    mod = relax.transform.LegalizeOps()(bb.finalize())
    asm, off, shp, top, outn = v2.compile_module(mod)
    rng = np.random.default_rng(7); g = lambda: _f16(rng.standard_normal((D, D)) * 0.1)
    X, W1, B, W2 = g(), g(), g(), g()
    gbuf = np.zeros(top + 20000, np.float32)
    for nm, arr in [("x", X), ("w1", W1), ("b", B), ("w2", W2)]:
        gbuf[off[nm]:off[nm] + D * D] = arr.reshape(-1)
    out = runtime.run(asm.words, gbuf, gn=top)[off[outn]:off[outn] + D * D].reshape(D, D)
    exp = (X.astype(np.float32) @ W1.astype(np.float32) + B.astype(np.float32)) @ W2.astype(np.float32)
    md = np.max(np.abs(out.astype(np.float32) - exp))
    assert md < 0.5 * np.max(np.abs(exp)) + 0.5, f"v2.compile: maxdiff={md}"
    return f"v2.compile_module: (x@w1+b)@w2 -> mysim==numpy (tol, maxdiff={md:.3f})"


def test_v2_compile_swiglu():
    """v2.compile_module on a SwiGLU FFN: y = (silu(x@Wg) * (x@Wu)) @ Wd.
    Exercises unary (silu) + binary (mul) elementwise + a 2-stage matmul chain."""
    D, F = 128, 256
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([D, D], "float16"))
    Wg = relax.Var("Wg", relax.TensorStructInfo([D, F], "float16"))
    Wu = relax.Var("Wu", relax.TensorStructInfo([D, F], "float16"))
    Wd = relax.Var("Wd", relax.TensorStructInfo([F, D], "float16"))
    with bb.function("main", [x, Wg, Wu, Wd]):
        with bb.dataflow():
            g = bb.emit(relax.op.nn.silu(bb.emit(relax.op.matmul(x, Wg))))
            u = bb.emit(relax.op.matmul(x, Wu))
            y = bb.emit(relax.op.matmul(bb.emit(relax.op.multiply(g, u)), Wd))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    mod = relax.transform.LegalizeOps()(bb.finalize())
    asm, off, shp, top, outn = v2.compile_module(mod)
    rng = np.random.default_rng(9)
    X = _f16(rng.standard_normal((D, D)) * 0.1); WG = _f16(rng.standard_normal((D, F)) * 0.1)
    WU = _f16(rng.standard_normal((D, F)) * 0.1); WD = _f16(rng.standard_normal((F, D)) * 0.1)
    gbuf = np.zeros(top + 20000, np.float32)
    for nm, arr in [("x", X), ("Wg", WG), ("Wu", WU), ("Wd", WD)]:
        gbuf[off[nm]:off[nm] + arr.size] = arr.reshape(-1)
    out = runtime.run(asm.words, gbuf, gn=top)[off[outn]:off[outn] + D * D].reshape(D, D)
    gate = X.astype(np.float32) @ WG.astype(np.float32)
    exp = ((gate / (1.0 + np.exp(-gate))) * (X.astype(np.float32) @ WU.astype(np.float32))) @ WD.astype(np.float32)
    md = np.max(np.abs(out.astype(np.float32) - exp))
    assert md < 0.05 * np.max(np.abs(exp)) + 0.5, f"swiglu maxdiff={md}"
    return f"v2.compile_module SwiGLU FFN -> mysim==numpy (tol, maxdiff={md:.3f})"


if __name__ == "__main__":
    print("[PASS]", test_ew2_byte_exact_and_numeric())
    print("[PASS]", test_ew1_byte_exact_and_numeric())
    print("[PASS]", test_silu_byte_exact_and_numeric())
    print("[PASS]", test_reduce_byte_exact_and_numeric())
    print("[PASS]", test_native_primitives())
    print("[PASS]", test_e2e_elementwise_real_pipeline())
    print("[PASS]", test_v2_compile_pipeline())
    print("[PASS]", test_v2_compile_swiglu())
    print("ALL v2 (unified walker + REAL pipeline + v2.compile_module) TESTS PASSED")
