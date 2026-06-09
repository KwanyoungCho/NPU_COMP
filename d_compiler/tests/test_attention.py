"""M5: single-head causal attention -> ISA -> mysim.

Exercises: element-copy transpose (k^T), constant scale, causal mask,
softmax WITHOUT max-subtraction. Compared to a float reference.
Also reports the transpose instruction overhead (reviewer's analysis ask).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from tvm import relax
from npu_compiler import driver, legalize
from npu_compiler.driver import compile_func


def _fp16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def make_attn_mod(seq, hd):
    bb = relax.BlockBuilder()
    q = relax.Var("q", relax.TensorStructInfo([seq, hd], "float16"))
    k = relax.Var("k", relax.TensorStructInfo([seq, hd], "float16"))
    v = relax.Var("v", relax.TensorStructInfo([seq, hd], "float16"))
    with bb.function("main", [q, k, v]):
        with bb.dataflow():
            y = legalize.attention_singlehead_causal(bb, q, k, v, seq, hd)
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def attn_ref(q, k, v, seq, hd):
    s = (q @ k.T) / np.sqrt(hd)
    for i in range(seq):
        for j in range(seq):
            if j > i:
                s[i, j] = -np.inf
    s = s - s.max(axis=1, keepdims=True)        # stable softmax (shift-invariant)
    e = np.exp(s)
    p = e / e.sum(axis=1, keepdims=True)
    return p @ v


def test_attention_seq8_hd16():
    seq, hd = 8, 16
    rng = np.random.default_rng(13)
    q = _fp16(rng.standard_normal((seq, hd)) * 0.2)
    k = _fp16(rng.standard_normal((seq, hd)) * 0.2)
    v = _fp16(rng.standard_normal((seq, hd)) * 0.5)

    mod = make_attn_mod(seq, hd)
    got = driver.run_module(mod, {"q": q, "k": k, "v": v})
    exp = attn_ref(q.astype(np.float64), k.astype(np.float64), v.astype(np.float64), seq, hd)

    maxabs = float(np.max(np.abs(exp)))
    maxerr = float(np.max(np.abs(got - exp)))
    assert maxerr < 0.05 * maxabs + 0.02, f"maxerr={maxerr} maxabs={maxabs}"

    # transpose overhead analysis
    asm, _ = compile_func(mod["main"])
    total = len(asm.words)
    transpose_instr = seq * hd * 6                       # vlen+addr*2(=4)+load+add+save per elem
    return maxerr / (maxabs + 1e-6), maxerr, total, transpose_instr


if __name__ == "__main__":
    rel, maxerr, total, tr = test_attention_seq8_hd16()
    print(f"[PASS] attention seq8 hd16 e2e: rel={rel:.4g} maxerr={maxerr:.4g}")
    print(f"   instrs total={total}, transpose(k^T)~={tr} ({100*tr//total}% of program) "
          f"-> element-copy transpose overhead")
    print("ALL M5 TESTS PASSED")
