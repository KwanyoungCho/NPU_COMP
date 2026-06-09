"""Smoke test: encode a program with isa.py, run it on the GIVEN mysim, check math.

Replicates inst_0001 (vector_add_immediate) semantics end-to-end:
  src1 @ addr 5, vlen 8, load, add immediate 3, save @ addr 32.
With a ramp input G[i]=i, G[32:40] should become [5..12]+3 = [8..15].
This exercises isa.py (encoder) + runtime.py (mysim run) together.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))
from npu_compiler.isa import Asm, SRC1, SRC2, DST, IMM
from npu_compiler import runtime


def test_vector_add_immediate_end_to_end():
    N = 64
    gbuf = np.arange(N, dtype=np.float32)               # ramp, like the original input

    a = Asm()
    a.addr(SRC1, 5).vlen(8).load(0, 0)                  # load G[5:13]
    a.v_add(IMM, 3)                                     # +3
    a.addr(DST, 32).save(0)                             # store to G[32:40]
    a.halt()

    out = runtime.run(a, gbuf, gn=N)
    got = out[32:40]
    exp = np.arange(5, 13, dtype=np.float32) + 3        # [8..15]
    assert np.array_equal(got, exp), f"got {got} exp {exp}"
    return got


def test_matmul_2x3_3x2():
    """Small real-matmul (mode=vector) through mysim: C[2x2] = A[2x3] @ B[3x2]."""
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    # G-buffer layout: A at 0 (6), B at 6 (6), C at 12 (4)
    gbuf = np.zeros(32, dtype=np.float32)
    gbuf[0:6] = A.reshape(-1)
    gbuf[6:12] = B.reshape(-1)

    a = Asm()
    a.tile(0, 2, 3)         # A: d1=2(rows) d2=3(cols)
    a.tile(1, 3, 2)         # B: d1=3 d2=2
    a.addr(SRC1, 0).load(1, 0)   # load A as matrix operand1
    a.addr(SRC2, 6).load(1, 1)   # load B as matrix operand2
    a.m_mul(mode=2)              # matrix multiply (vector/matrix mode)
    a.addr(DST, 12).save(1)     # store C (matrix)
    a.halt()

    out = runtime.run(a, gbuf, gn=32)
    got = out[12:16].reshape(2, 2)
    exp = A @ B
    assert np.allclose(got, exp, atol=1e-2), f"got\n{got}\nexp\n{exp}"
    return got


if __name__ == "__main__":
    print("building/using mysim:", runtime.build_mysim())
    g = test_vector_add_immediate_end_to_end()
    print("[PASS] vector add immediate e2e ->", g)
    m = test_matmul_2x3_3x2()
    print("[PASS] matmul 2x3 @ 3x2 e2e ->\n", m)
    print("ALL RUNTIME TESTS PASSED")
