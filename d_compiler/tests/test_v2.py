"""v2 Phase 2-A: elementwise ops lowered by the unified walker (V2Walker) match v1.

Two independent checks per op:
  (1) ISA byte-exact — V2Walker(marker) words == v1 emit_ew algorithm (codegen.py:488-497).
  (2) numerical — run the walked program on mysim, compare to numpy fp16.
v1 stays the oracle; this proves the walker path reproduces it before we migrate ops.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from npu_compiler.isa import Asm, SRC1, SRC2, DST, VECTOR
from npu_compiler import v2_backend as v2, runtime

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


if __name__ == "__main__":
    print("[PASS]", test_ew2_byte_exact_and_numeric())
    print("[PASS]", test_ew1_byte_exact_and_numeric())
    print("[PASS]", test_silu_byte_exact_and_numeric())
    print("ALL v2 Phase 2-A (elementwise via unified walker) TESTS PASSED")
