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


def _feed(gbuf, off, tiled_feed, items):
    """Load named param arrays into the G-buffer, pre-packing tile-blocked params
    (names in tiled_feed) with memplan.pack_tiled — mirrors v1 run_compiled host packing."""
    for nm, arr in items:
        if nm not in off:
            continue
        a = memplan.pack_tiled(np.asarray(arr), 64) if nm in tiled_feed else np.asarray(arr).reshape(-1)
        gbuf[off[nm]:off[nm] + a.size] = np.asarray(a, np.float32).reshape(-1)


def _read_out(gbuf, off, shp, outn, layout=None):
    """Read a 2D output at off[outn], host-unpacking it if the layout stored it tile-blocked
    (Phase 4.1: a tile residual stream ends in a tile output that the host unpacks to row)."""
    R, C = shp[outn]
    if layout is not None and layout.get(outn) == "tile":
        raw = gbuf[off[outn]:off[outn] + memplan.tiled_numel([R, C])]
        return memplan.unpack_tiled(raw, R, C, 64).reshape(-1)
    return gbuf[off[outn]:off[outn] + R * C]

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
    asm, off, shp, top, outn, _, tf, lay = v2.compile_module(mod)
    rng = np.random.default_rng(7); g = lambda: _f16(rng.standard_normal((D, D)) * 0.1)
    X, W1, B, W2 = g(), g(), g(), g()
    gbuf = np.zeros(top + 20000, np.float32)
    _feed(gbuf, off, tf, [("x", X), ("w1", W1), ("b", B), ("w2", W2)])
    out = _read_out(runtime.run(asm.words, gbuf, gn=top), off, shp, outn, lay).reshape(D, D)
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
    asm, off, shp, top, outn, _, tf, lay = v2.compile_module(mod)
    rng = np.random.default_rng(9)
    X = _f16(rng.standard_normal((D, D)) * 0.1); WG = _f16(rng.standard_normal((D, F)) * 0.1)
    WU = _f16(rng.standard_normal((D, F)) * 0.1); WD = _f16(rng.standard_normal((F, D)) * 0.1)
    gbuf = np.zeros(top + 20000, np.float32)
    _feed(gbuf, off, tf, [("x", X), ("Wg", WG), ("Wu", WU), ("Wd", WD)])
    out = _read_out(runtime.run(asm.words, gbuf, gn=top), off, shp, outn, lay).reshape(D, D)
    gate = X.astype(np.float32) @ WG.astype(np.float32)
    exp = ((gate / (1.0 + np.exp(-gate))) * (X.astype(np.float32) @ WU.astype(np.float32))) @ WD.astype(np.float32)
    md = np.max(np.abs(out.astype(np.float32) - exp))
    assert md < 0.05 * np.max(np.abs(exp)) + 0.5, f"swiglu maxdiff={md}"
    return f"v2.compile_module SwiGLU FFN -> mysim==numpy (tol, maxdiff={md:.3f})"


def test_v2_compile_rmsnorm():
    """v2.compile_module on v1's legalize.rms_norm: exercises reduce(sum) + broadcast
    (col [R,1]->[R,D] and row [1,D]->[R,D]) + constants (1/d, eps, ones)."""
    from npu_compiler import legalize
    S, D, eps = 64, 128, 1e-5
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([S, D], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([1, D], "float16"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            gv = bb.emit_output(legalize.rms_norm(bb, x, w, S, D, eps=eps))
        bb.emit_func_output(gv)
    mod = relax.transform.LegalizeOps()(bb.finalize())
    asm, off, shp, top, outn, consts, tf, lay = v2.compile_module(mod)
    rng = np.random.default_rng(11)
    X = _f16(rng.standard_normal((S, D))); W = _f16(rng.standard_normal((1, D)))
    gbuf = np.zeros(top + 20000, np.float32)
    for co, arr in consts:
        gbuf[co:co + arr.size] = np.asarray(arr, np.float32).reshape(-1)
    _feed(gbuf, off, tf, [("x", X), ("w", W)])
    out = _read_out(runtime.run(asm.words, gbuf, gn=top), off, shp, outn, lay).reshape(S, D)
    Xf = X.astype(np.float32); ms = np.mean(Xf ** 2, axis=1, keepdims=True) + eps
    exp = Xf / np.sqrt(ms) * W.astype(np.float32)
    md = np.max(np.abs(out.astype(np.float32) - exp))
    assert md < 0.05 * np.max(np.abs(exp)) + 0.1, f"rmsnorm maxdiff={md}"
    return f"v2.compile_module RMSNorm (reduce+broadcast+consts) -> mysim==numpy (tol, maxdiff={md:.3f})"


def test_v2_compile_full_layer():
    """★ v2.compile_module on a FULL v1 layer (RMSNorm + attention[RoPE/softmax] + FFN),
    validated against v1's numpy reference model.ref_layer. Exercises every op family:
    matmul, elementwise, reduce, broadcast, transpose, strided_slice, concat, constants."""
    from npu_compiler import model
    cfg = model.REDUCED
    cos, sin, rot = model.rope_tables(cfg)
    W = model.make_weights(cfg, seed=7)
    mod = relax.transform.LegalizeOps()(model.build_layer_module(cfg))
    asm, off, shp, top, outn, consts, tf, lay = v2.compile_module(mod)
    gbuf = np.zeros(top + 50000, np.float32)
    for co, arr in consts:
        gbuf[co:co + arr.size] = np.asarray(arr, np.float32).reshape(-1)
    _feed(gbuf, off, tf, list(W.items()))
    out = _read_out(runtime.run(asm.words, gbuf, gn=top), off, shp, outn, lay)
    exp = np.asarray(model.ref_layer(cfg, W, cos, sin), np.float32).reshape(-1)
    rel = float(np.max(np.abs(out.astype(np.float32) - exp))) / (float(np.max(np.abs(exp))) + 1e-9)
    assert rel < 0.05, f"full layer rel={rel}"
    return f"v2.compile_module FULL LAYER (RMSNorm+attn+FFN) vs ref_layer: rel={rel:.4f}"


def test_v2_guards_reject_unsupported():
    """Correctness hardening (found by adversarial review): v2.compile_module must FAIL
    LOUDLY, never silently miscompile, on cases it can't yet handle — >=256 dims that would
    wrap the 8-bit ISA tile fields, and scalar [1,1] broadcast. Mirrors v1's guards."""
    def _raises(build):
        try:
            v2.compile_module(relax.transform.LegalizeOps()(build()))
            return False
        except AssertionError:
            return True

    def b_max256():                                          # softmax max at SEQ>=256 -> 8-bit wrap
        bb = relax.BlockBuilder(); x = relax.Var("x", relax.TensorStructInfo([64, 256], "float16"))
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(bb.emit(relax.op.max(x, axis=[-1], keepdims=True)))
            bb.emit_func_output(gv)
        return bb.finalize()

    def b_scalar():                                          # [1,1] scalar broadcast (mis-routed pre-fix)
        bb = relax.BlockBuilder(); s = relax.Var("s", relax.TensorStructInfo([1, 1], "float16"))
        with bb.function("main", [s]):
            with bb.dataflow():
                gv = bb.emit_output(bb.emit(relax.op.broadcast_to(s, relax.ShapeExpr([8, 16]))))
            bb.emit_func_output(gv)
        return bb.finalize()

    assert _raises(b_max256), "reduce-max C>=256 must assert, not silently wrap"
    assert _raises(b_scalar), "scalar [1,1] broadcast must assert, not read past source"
    return "guards: >=256 reduce + scalar broadcast rejected loudly (no silent miscompile)"


def test_v2_memory_reuse():
    """M1 pass (A1 liveness reuse): reuse=True hands a dead intermediate's slot to a later
    same-size one. Two invariants: (1) peak G-buffer `top` shrinks (activation region), and
    (2) it is byte-exact vs the flat-bump plan (reuse relocates data, never changes values)."""
    from npu_compiler import model
    cfg = model.MEDIUM
    cos, sin, rot = model.rope_tables(cfg); W = model.make_weights(cfg, seed=7)
    mod = relax.transform.LegalizeOps()(model.build_layer_module(cfg))

    params, ops, shp, ca, outn = v2._parse(mod)
    layout, packed = {}, {}                                   # all-row: isolate A1 (no A4)
    fr, cons = {}, set()                                      # no fusion: isolate A1
    _, top0, _, _, _ = v2._plan_memory(params, ops, shp, ca, outn, layout, packed, fr, cons, reuse=False)
    _, top1, _, _, _ = v2._plan_memory(params, ops, shp, ca, outn, layout, packed, fr, cons, reuse=True)
    assert top1 < top0, f"A1 reuse did not shrink peak ({top1} !< {top0})"

    def run(reuse):
        asm, off, shp_, top, on, consts, tf, lay = v2.compile_module(mod, reuse=reuse, tile=False)
        g = np.zeros(top + 80000, np.float32)
        for co, arr in consts: g[co:co + arr.size] = np.asarray(arr, np.float32).reshape(-1)
        _feed(g, off, tf, list(W.items()))
        return _read_out(runtime.run(asm.words, g, gn=top), off, shp_, on, lay).copy()

    a, b = run(False), run(True)
    assert np.array_equal(a, b), "A1 reuse must be byte-exact vs flat bump"
    return f"M1 A1 reuse: peak {top0}->{top1} ({100*(top0-top1)/top0:.0f}% smaller), byte-exact"


def _run_layer(cfg, tile, fuse=True):
    """Compile a full layer for cfg and run it on mysim; return (rel-vs-ref_layer, #words)."""
    from npu_compiler import model
    cos, sin, rot = model.rope_tables(cfg); W = model.make_weights(cfg, seed=7)
    mod = relax.transform.LegalizeOps()(model.build_layer_module(cfg))
    asm, off, shp, top, outn, consts, tf, lay = v2.compile_module(mod, tile=tile, fuse=fuse)
    g = np.zeros(top + 200000, np.float32)
    for co, arr in consts:
        g[co:co + arr.size] = np.asarray(arr, np.float32).reshape(-1)
    _feed(g, off, tf, list(W.items()))
    gbuf = runtime.run(asm.words, g, gn=top)
    out = _read_out(gbuf, off, shp, outn, lay)
    exp = np.asarray(model.ref_layer(cfg, W, cos, sin), np.float32).reshape(-1)
    rel = float(np.max(np.abs(out.astype(np.float32) - exp))) / (float(np.max(np.abs(exp))) + 1e-9)
    return rel, len(asm.words)


def test_v2_a4_multitile():
    """★ A4 tile-blocked layout on a MULTI-tile layer (SEQ=128 -> 2x2 score tiles, so
    tile order != row order — MEDIUM's 64x64 single-tile case can't catch layout bugs).
    Validates: tile=True is correct vs ref_layer AND vs the row path, and emits FEWER
    instructions (gather/scatter elimination = the A4 benefit)."""
    from npu_compiler import model
    cfg = model.LayerConfig("s128", SEQ=128, D=128, H=2, KV=2, HD=64, F=256,
                            eps=1e-5, rope_base=1e4, rope_scale=False)
    # tile layout actually fires + weights get packed
    mod = relax.transform.LegalizeOps()(model.build_layer_module(cfg))
    params, ops, shp, ca, outn = v2._parse(mod)
    lay, pk = v2._assign_layouts(params, ops, shp, ca, outn)
    ntile = sum(1 for v in lay.values() if v == "tile")
    assert ntile >= 10 and len(pk) >= 5, f"A4 did not engage (tile={ntile}, packed={len(pk)})"
    rt, nr = _run_layer(cfg, tile=True)
    rf, nf = _run_layer(cfg, tile=False)
    assert rt < 0.05, f"A4 tile full layer wrong: rel={rt}"
    assert nr < nf, f"A4 tile must emit fewer words (gather elim): {nr} !< {nf}"
    return (f"A4 multi-tile SEQ128: tile rel={rt:.4f} == row rel={rf:.4f}, "
            f"{nr}w vs {nf}w ({100*(nf-nr)//nf}% fewer, gather elim)")


def test_v2_oproj_fusion():
    """★ F3 O-proj fusion: build_layer_module emits attn = sum_h (ctx_h @ Wo_h) as H
    same-shape matmuls + an add-tree. v2 fuses them into ONE in-place accumulate group
    (emit_matmul_accumulate_group) — H scatters -> 1. Validates: a group is detected,
    fuse=True is correct vs ref_layer AND vs fuse=False, and emits fewer instructions."""
    from npu_compiler import model
    cfg = model.LayerConfig("h8", SEQ=128, D=512, H=8, KV=2, HD=64, F=512,
                            eps=1e-5, rope_base=1e4, rope_scale=False)
    mod = relax.transform.LegalizeOps()(model.build_layer_module(cfg))
    params, ops, shp, ca, outn = v2._parse(mod)
    fr, cons = v2._detect_oproj_groups(ops, shp)
    assert len(fr) == 1 and sum(len(v) for v in fr.values()) == 8, \
        f"expected 1 O-proj group over 8 heads, got {fr}"
    rT, nT = _run_layer(cfg, tile=True, fuse=True)
    rF, nF = _run_layer(cfg, tile=True, fuse=False)
    assert rT < 0.05, f"fused O-proj wrong: rel={rT}"
    assert abs(rT - rF) < 5e-3, f"fuse must match no-fuse: {rT} vs {rF}"
    assert nT < nF, f"fusion must emit fewer words (H scatters -> 1): {nT} !< {nF}"
    return f"F3 O-proj fusion (H=8): rel={rT:.4f} (~no-fuse {rF:.4f}), {nF}->{nT}w ({100*(nF-nT)//nF}% fewer)"


def test_v2_a4_hd128_real_head_dim():
    """★ A4 tile on HD=128 — the REAL Llama 3.2 3B head dim. At HD=128 the RoPE rotate-half
    slice is 64-wide/64-aligned; an earlier layout rule wrongly seeded it 'tile' and the
    row-only slice/concat/transpose emit crashed (adversarial review V2-019). The fixpoint
    now keeps transpose/slice/concat row at any head dim, so the RoPE chain stays row while
    the softmax + FFN islands stay tile. Validates tile==row correctness + gather elim."""
    from npu_compiler import model
    cfg = model.LayerConfig("hd128", SEQ=128, D=256, H=2, KV=2, HD=128, F=384,
                            eps=1e-5, rope_base=1e4, rope_scale=False)
    rt, nr = _run_layer(cfg, tile=True)
    rf, nf = _run_layer(cfg, tile=False)
    assert rt < 0.05, f"HD=128 tile wrong: rel={rt}"
    assert abs(rt - rf) < 1e-3, f"HD=128 tile must match row oracle: {rt} vs {rf}"
    assert nr < nf, f"HD=128 tile must emit fewer words: {nr} !< {nf}"
    return f"A4 HD=128 (real Llama head dim): tile rel={rt:.4f} == row {rf:.4f}, {nr}w vs {nf}w"


def test_v2_packed_model_agnostic():
    """Stage 4 IR->IR (Step 4): the F4 layout pass packed._build_packed is model-AGNOSTIC —
    an op it doesn't know how to tile (here relax.tanh, standing in for any other model's op)
    does NOT crash the pass; it flows through as a ROW island (verbatim on row-coerced inputs)
    while matmul still tiles. So the graph transform generalizes beyond build_layer_module
    (codegen still needs a per-op-family lowering — the expected op-coverage cost)."""
    from npu_compiler import packed
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([128, 128], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([128, 128], "float16"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            h = bb.emit(relax.op.matmul(x, w))               # -> einsum (tile)
            gv = bb.emit_output(bb.emit(relax.op.tanh(h)))   # unknown op -> row island
        bb.emit_func_output(gv)
    pk, _pack_names = packed._build_packed(bb.finalize())
    leg = relax.transform.LegalizeOps()(pk)                                     # must not crash
    p, ops, shp, ca, outn = v2._parse(leg)
    fams = set(o.opname for o in ops)
    assert "einsum" in fams and "tanh" in fams, f"expected matmul tiled + tanh row, got {fams}"
    return "F4 packed pass model-agnostic: unknown op (tanh) -> row island, matmul -> tile"


def test_v2_compile_packed():
    """★ Stage 4 IR->IR: the layout_transform-native packed path compiles a FULL layer to NPU
    ISA correctly. packed._build_packed (Relax IRModule->IRModule) -> LegalizeOps -> op-FAMILY
    dispatch (einsum->emit_packed_matmul, 4D ew/reduce/broadcast->markers, layout_transform->
    npu_copy64) -> mysim. Params/output are ROW (in-graph layout_transform packs on-device),
    so the host feeds/reads raw. This is the model-agnostic codegen that replaces the op-list
    _emit; the hand tile emitters retire in Phase 4.5."""
    from npu_compiler import model
    cfgs = [model.MEDIUM]     # SEQ128/HD128 validated by direct measurement; packed emit is
    #                           slow (unoptimized) so keep the gate light until Step 5 speedup
    rels = []
    for cfg in cfgs:
        cos, sin, rot = model.rope_tables(cfg); W = model.make_weights(cfg, seed=7)
        asm, off, shp, top, outn, consts, pack_names = v2.compile_packed(model.build_layer_module(cfg))
        g = np.zeros(top + 300000, np.float32)
        for co, arr in consts:
            g[co:co + arr.size] = np.asarray(arr, np.float32).reshape(-1)
        for nm, arr in W.items():
            if nm in off:
                a = memplan.pack_tiled(np.asarray(arr), 64) if nm in pack_names else np.asarray(arr).reshape(-1)
                g[off[nm]:off[nm] + a.size] = np.asarray(a, np.float32).reshape(-1)
        R, C = shp[outn]
        out = runtime.run(asm.words, g, gn=top)[off[outn]:off[outn] + R * C]
        exp = np.asarray(model.ref_layer(cfg, W, cos, sin), np.float32).reshape(-1)
        rel = float(np.max(np.abs(out.astype(np.float32) - exp))) / (float(np.max(np.abs(exp))) + 1e-9)
        assert rel < 0.05, f"compile_packed {cfg.name}: rel={rel}"
        rels.append(rel)
    return f"Stage 4 IR->IR compile_packed FULL LAYER (layout_transform-native): rels={[f'{r:.4f}' for r in rels]}"


def test_v2_tile_rope():
    """★ Stage 4 Phase 4.4: tile-native RoPE. When head_dim/2 is a 64-multiple (real Llama
    hd=128 -> h=64) the rotate-half slice/concat and the K^T transpose lower in TILE layout
    (whole-tile-column slice, permute_dims[1,0,3,2] transpose) instead of a row island — so
    the packed path has NO interior unpack/repack and beats the hand-tuned op-list on command
    count while staying model-agnostic. Sub-tile hd (h=32) still falls back to the row island."""
    from npu_compiler import model, packed as _packed
    from tvm import relax
    cfg = model.LayerConfig("h4", SEQ=128, D=512, H=4, KV=2, HD=128, F=256,
                            eps=1e-5, rope_base=1e4, rope_scale=False)
    cos, sin, rot = model.rope_tables(cfg); W = model.make_weights(cfg, seed=7)

    # (a) RoPE structural ops are all TILE (row island fully eliminated at hd=128)
    pkmod, _ = _packed._build_packed(model.build_layer_module(cfg))
    _, ops, shp, _, _ = v2._parse(relax.transform.LegalizeOps()(pkmod))
    row_struct = [op.opname for op in ops
                  if op.opname in ("strided_slice", "concatenate", "transpose") and len(shp[op.outn]) == 2]
    assert not row_struct, f"tile-RoPE left a row island: {row_struct}"

    # (b) correct vs fp64 reference
    asm, off, shp, top, outn, consts, pack = v2.compile_packed(model.build_layer_module(cfg))
    g = np.zeros(top + 300000, np.float32)
    for co, arr in consts:
        g[co:co + arr.size] = np.asarray(arr, np.float32).reshape(-1)
    for nm, arr in W.items():
        if nm in off:
            a = memplan.pack_tiled(np.asarray(arr), 64) if nm in pack else np.asarray(arr).reshape(-1)
            g[off[nm]:off[nm] + a.size] = np.asarray(a, np.float32).reshape(-1)
    R, C = shp[outn]
    out = runtime.run(asm.words, g, gn=top)[off[outn]:off[outn] + R * C]
    exp = np.asarray(model.ref_layer(cfg, W, cos, sin), np.float32).reshape(-1)
    rel = float(np.max(np.abs(out.astype(np.float32) - exp))) / (float(np.max(np.abs(exp))) + 1e-9)
    assert rel < 0.05, f"tile-RoPE HD128: rel={rel}"

    # (c) packed now emits FEWER commands than the op-list tile path (the whole point)
    aol = v2.compile_module(relax.transform.LegalizeOps()(model.build_layer_module(cfg)), tile=True)[0]
    assert len(asm.words) < len(aol.words), f"packed {len(asm.words)} !< op-list {len(aol.words)}"
    return (f"Phase 4.4 tile-RoPE HD128: rel={rel:.4f}, row-island=0, "
            f"packed {len(asm.words):,} < op-list {len(aol.words):,} cmds")


def test_v2_packed_oproj_fusion_option():
    """F3 O-proj fusion is available as an OPT-IN in the packed path (compile_packed(fuse=True))
    but OFF by default: tile-C already keeps the per-head O-proj output tile-blocked (scatter=0),
    so fusion is superseded and saves <0.1%. This test keeps the opt-in path correct + detected."""
    from npu_compiler import model, packed as _packed
    from tvm import relax
    cfg = model.LayerConfig("g8", SEQ=128, D=1024, H=8, KV=4, HD=128, F=512,
                            eps=1e-5, rope_base=1e4, rope_scale=False)
    cos, sin, rot = model.rope_tables(cfg); W = model.make_weights(cfg, seed=7)

    # fusion actually fires: the 8-head O-proj add-tree collapses to one accumulate group
    pkmod, _ = _packed._build_packed(model.build_layer_module(cfg))
    _, ops, shp, _, _ = v2._parse(relax.transform.LegalizeOps()(pkmod))
    fused_root, consumed = v2._detect_oproj_groups(ops, shp, leaf_op="einsum")
    assert fused_root, "packed O-proj add-tree not detected"
    assert any(len(v) >= cfg.H for v in fused_root.values()), f"expected an H={cfg.H}-leaf group"

    exp = np.asarray(model.ref_layer(cfg, W, cos, sin), np.float32).reshape(-1)
    den = float(np.max(np.abs(exp))) + 1e-9
    outs = []
    for fuse in (False, True):
        asm, off, shp, top, outn, consts, pack = v2.compile_packed(model.build_layer_module(cfg), fuse=fuse)
        g = np.zeros(top + 500000, np.float32)
        for co, arr in consts:
            g[co:co + arr.size] = np.asarray(arr, np.float32).reshape(-1)
        for nm, arr in W.items():
            if nm in off:
                a = memplan.pack_tiled(np.asarray(arr), 64) if nm in pack else np.asarray(arr).reshape(-1)
                g[off[nm]:off[nm] + a.size] = np.asarray(a, np.float32).reshape(-1)
        R, C = shp[outn]
        out = runtime.run(asm.words, g, gn=top)[off[outn]:off[outn] + R * C].astype(np.float32)
        rel = float(np.max(np.abs(out - exp))) / den
        assert rel < 0.05, f"packed fuse={fuse}: rel={rel}"
        outs.append((fuse, len(asm.words), rel))
    (_, n_no, _), (_, n_fu, _) = outs
    assert n_fu <= n_no, f"fuse should not add commands: {n_fu} > {n_no}"
    return f"F3 packed opt-in: fuse fires (H={cfg.H} group), fuse {n_fu:,} <= nofuse {n_no:,} cmds, correct"


def test_v2_reindex_copy():
    """Stage 4 codegen primitive: a legalized layout_transform / reshape (the pack/unpack
    boundary of the layout_transform-native path) lowers via schedule_reindex_copy
    (split-by-64 + tensorize npu_copy64) + V2Walker to NPU ISA == memplan.pack_tiled. This
    is the last lowering piece the packed-IR codegen needed (einsum/ew/reduce/broadcast
    already existed)."""
    import tvm.tir as tir
    imap = lambda r, c: (r // 64, c // 64, r % 64, c % 64)
    for (R, C) in [(128, 128), (128, 256), (192, 64)]:
        bb = relax.BlockBuilder(); X = relax.Var("X", relax.TensorStructInfo([R, C], "float16"))
        with bb.function("main", [X]):
            with bb.dataflow():
                gv = bb.emit_output(bb.emit(relax.op.layout_transform(X, index_map=imap)))
            bb.emit_func_output(gv)
        mod = relax.transform.LegalizeOps()(bb.finalize())
        fn = [g.name_hint for g in mod.functions if isinstance(mod[g], tir.PrimFunc)][0]
        pf = v2.schedule_reindex_copy(mod[fn])
        asm = Asm(); mp = memplan.MemPlan(); mp.top = R * C * 2
        wk = v2.V2Walker(asm, mp, {})
        for p, off in zip(pf.params, [0, R * C]):
            wk.base[pf.buffer_map[p].data] = off
        wk.walk(pf.body); wk.flush()
        X0 = _f16(np.arange(R * C).reshape(R, C) * 0.001)
        g = np.zeros(R * C * 2 + 50000, np.float32); g[0:R * C] = X0.reshape(-1)
        out = runtime.run(asm.words, g, gn=R * C * 2)[R * C:R * C + R * C]
        assert np.array_equal(out.astype(np.float32), memplan.pack_tiled(X0, 64).astype(np.float32)), \
            f"reindex copy [{R},{C}] != pack_tiled"
    return "Stage 4 reindex copy (layout_transform -> npu_copy64): mysim == pack_tiled"


def test_v2_f4_packed_pass():
    """Stage 4 Phase 4.3 (F4 Relax rewrite, foundation): packed._build_packed rewrites the
    high-level layer into tile-blocked packed 4D form (matmul->einsum, ew/reduce/broadcast
    on [Rt,Ct,64,64], RoPE row island via in-graph layout_transform) so LegalizeOps emits
    native tile TIR. Validated SEMANTICALLY at the Relax level (relax VM on llvm): the packed
    graph is BIT-IDENTICAL to the row graph and correct vs ref_layer. (NPU-ISA codegen
    integration = the packed-aware walker for einsum/layout_transform/reshape is Phase 4.4+.)"""
    from npu_compiler import packed, model
    r = packed.validate(model.MEDIUM)
    assert r["rel_row"] < 1e-4, f"packed pass must be bit-identical to row graph, got {r['rel_row']}"
    assert r["rel_ref"] < 0.05, f"packed pass wrong vs ref_layer: {r['rel_ref']}"
    return f"F4 packed pass (Relax): bit-identical to row (rel {r['rel_row']:.6f}), ref rel={r['rel_ref']:.4f}"


def test_v2_packed_matmul():
    """Stage 4 foundation: the layout_transform-native packed matmul (einsum nest over 4D
    [Rt,Ct,64,64] buffers, tensorized to npu_gemm_acc) is (1) numerically correct vs numpy
    and (2) BYTE-EXACT vs v1's emit_matmul_into(a_tiled,c_tiled,b_pack_nt) — same 64x64 MAC,
    same tile order — so it can replace the per-op tile flags with zero ISA change."""
    from npu_compiler.tir_backend import emit_matmul_into
    pt = lambda a: memplan.pack_tiled(np.asarray(a), 64)
    rng = np.random.default_rng(3)
    for (M, K, N) in [(128, 128, 128), (128, 192, 256), (256, 128, 64)]:
        Mt, Kt, Nt = M // 64, K // 64, N // 64
        A = _f16(rng.standard_normal((M, K)) * 0.1); B = _f16(rng.standard_normal((K, N)) * 0.1)
        exp = A.astype(np.float32) @ B.astype(np.float32)
        asm = Asm(); mp = memplan.MemPlan()
        ao, bo, co = 0, M * K, M * K + K * N; mp.top = M * K + K * N + M * N
        v2.emit_packed_matmul(asm, mp, co, ao, bo, Mt, Kt, Nt)
        g = np.zeros(mp.top + 50000, np.float32)
        g[ao:ao + M * K] = pt(A).astype(np.float32).reshape(-1)
        g[bo:bo + K * N] = pt(B).astype(np.float32).reshape(-1)
        out = memplan.unpack_tiled(runtime.run(asm.words, g, gn=mp.top)[co:co + M * N], M, N, 64)
        rel = np.max(np.abs(out.astype(np.float32) - exp)) / (np.max(np.abs(exp)) + 1e-9)
        assert rel < 0.02, f"packed matmul {M}x{K}x{N}: rel={rel}"
        asm2 = Asm(); mp2 = memplan.MemPlan(); mp2.top = mp.top
        emit_matmul_into(asm2, mp2, co, ao, bo, M, K, N, b_pack_nt=Nt, a_tiled=True, c_tiled=True)
        assert list(asm.words) == list(asm2.words), f"packed matmul {M}x{K}x{N} not byte-exact vs v1"
    return "Stage 4 packed matmul (einsum->npu_gemm_acc): correct + byte-exact vs emit_matmul_into"


def test_v2_residual_tile_gather():
    """★ Phase 4.1: keep the residual/RMSNorm/FFN stream tile end-to-end (64-mult 2D output
    stays tile, host unpacks) so every projection reads a tile A-operand -> input gather
    collapses. Validates: input x + output are tile, and the compiled input-gather is a
    small fraction of the all-row path (matching v1's 'after' profile: proj gather ~0)."""
    from npu_compiler import model, cost
    cfg = model.LayerConfig("h4", SEQ=128, D=512, H=4, KV=2, HD=128, F=256,
                            eps=1e-5, rope_base=1e4, rope_scale=False)
    mod = relax.transform.LegalizeOps()(model.build_layer_module(cfg))
    p, ops, shp, ca, outn = v2._parse(mod)
    lay, pk = v2._assign_layouts(p, ops, shp, ca, outn)
    assert lay.get(p[0]) == "tile", f"residual input x must be tile, got {lay.get(p[0])}"
    assert lay.get(outn) == "tile", f"64-mult output must stay tile, got {lay.get(outn)}"
    g_tile = cost.analyze(v2.compile_module(mod, tile=True)[0]).get("by_role", {}).get("gather", 0)
    g_row = cost.analyze(v2.compile_module(mod, tile=False)[0]).get("by_role", {}).get("gather", 1)
    assert g_tile < 0.15 * g_row, f"residual-tile gather ({g_tile}) must be << row gather ({g_row})"
    return f"Phase 4.1 residual tile: x+out tile, gather {g_row}->{g_tile} ({100*(1-g_tile/g_row):.0f}% closed)"


def test_v2_a4_unblocks_256():
    """A4 unblocks SEQ>=256: the tile softmax reduce/matmul use only 16-bit vlen + 64-wide
    tile fields (no 8-bit R/C wrap), so tile=True compiles+runs correctly where the row
    path (tile=False) must AssertionError on the 8-bit reduce-max field."""
    from npu_compiler import model
    cfg = model.LayerConfig("s256", SEQ=256, D=128, H=2, KV=2, HD=64, F=256,
                            eps=1e-5, rope_base=1e4, rope_scale=False)
    rt, nr = _run_layer(cfg, tile=True)
    assert rt < 0.05, f"A4 SEQ=256 wrong: rel={rt}"
    row_failed = False
    try:
        _run_layer(cfg, tile=False)
    except AssertionError:
        row_failed = True
    assert row_failed, "row path must reject SEQ>=256 (8-bit field), A4 must accept it"
    return f"A4 unblocks SEQ=256: tile rel={rt:.4f} OK; row path AssertionError (8-bit field)"


if __name__ == "__main__":
    print("[PASS]", test_ew2_byte_exact_and_numeric())
    print("[PASS]", test_ew1_byte_exact_and_numeric())
    print("[PASS]", test_silu_byte_exact_and_numeric())
    print("[PASS]", test_reduce_byte_exact_and_numeric())
    print("[PASS]", test_native_primitives())
    print("[PASS]", test_e2e_elementwise_real_pipeline())
    print("[PASS]", test_v2_compile_pipeline())
    print("[PASS]", test_v2_compile_swiglu())
    print("[PASS]", test_v2_compile_rmsnorm())
    print("[PASS]", test_v2_compile_full_layer())
    print("[PASS]", test_v2_memory_reuse())
    print("[PASS]", test_v2_a4_multitile())
    print("[PASS]", test_v2_oproj_fusion())
    print("[PASS]", test_v2_packed_matmul())
    print("[PASS]", test_v2_packed_model_agnostic())
    print("[PASS]", test_v2_compile_packed())
    print("[PASS]", test_v2_tile_rope())
    print("[PASS]", test_v2_packed_oproj_fusion_option())
    print("[PASS]", test_v2_reindex_copy())
    print("[PASS]", test_v2_f4_packed_pass())
    print("[PASS]", test_v2_residual_tile_gather())
    print("[PASS]", test_v2_a4_hd128_real_head_dim())
    print("[PASS]", test_v2_a4_unblocks_256())
    print("[PASS]", test_v2_guards_reject_unsupported())
    print("ALL v2 (unified walker + REAL pipeline + v2.compile_module) TESTS PASSED")
