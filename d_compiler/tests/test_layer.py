"""M7: full Llama-style transformer layer (reduced proxy) -> ISA -> mysim.

  h = x + Attn(RMSNorm(x))     # GQA + RoPE + causal softmax (no max-sub)
  y = h + FFN(RMSNorm(h))      # SwiGLU

Assembles every B0 building block. Validated against an independent numpy
reference (FP16 tolerance, like b_program_examples/llama_layer.py: rel ~0.1%).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from tvm import relax
from npu_compiler import driver, legalize
from npu_compiler.driver import compile_func

# reduced proxy config (Llama 3.2 3B *structure*, small dims; all <=255)
SEQ, D, H, KV, HD, F = 8, 64, 4, 2, 16, 128
GPK = H // KV


def _fp16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def _const(a):
    return relax.const(np.asarray(a, dtype="float16"))


def make_weights(seed=21):
    rng = np.random.default_rng(seed)
    ws = 0.2
    W = {
        "x":   _fp16(rng.standard_normal((SEQ, D))),
        "Wn1": _fp16(rng.uniform(0.8, 1.2, (1, D))),
        "Wn2": _fp16(rng.uniform(0.8, 1.2, (1, D))),
        "Wg":  _fp16(rng.uniform(-ws, ws, (D, F))),
        "Wu":  _fp16(rng.uniform(-ws, ws, (D, F))),
        "Wd":  _fp16(rng.uniform(-ws, ws, (F, D))),
    }
    for h in range(H):
        W[f"Wq{h}"] = _fp16(rng.uniform(-ws, ws, (D, HD)))
        W[f"Wo{h}"] = _fp16(rng.uniform(-ws, ws, (HD, D)))
    for k in range(KV):
        W[f"Wk{k}"] = _fp16(rng.uniform(-ws, ws, (D, HD)))
        W[f"Wv{k}"] = _fp16(rng.uniform(-ws, ws, (D, HD)))
    return W


def build_layer_mod(cos, sin, rot):
    bb = relax.BlockBuilder()
    # param vars (names must match the weights dict)
    def P(name, shape):
        return relax.Var(name, relax.TensorStructInfo(list(shape), "float16"))
    x = P("x", (SEQ, D))
    Wn1 = P("Wn1", (1, D)); Wn2 = P("Wn2", (1, D))
    Wq = [P(f"Wq{h}", (D, HD)) for h in range(H)]
    Wo = [P(f"Wo{h}", (HD, D)) for h in range(H)]
    Wk = [P(f"Wk{k}", (D, HD)) for k in range(KV)]
    Wv = [P(f"Wv{k}", (D, HD)) for k in range(KV)]
    Wg = P("Wg", (D, F)); Wu = P("Wu", (D, F)); Wd = P("Wd", (F, D))
    params = [x, Wn1, Wn2] + Wq + Wo + Wk + Wv + [Wg, Wu, Wd]

    cos_c, sin_c, rot_c = _const(cos), _const(sin), _const(rot)
    inv = 1.0 / float(np.sqrt(HD))
    scale_c = _const(np.full((SEQ, SEQ), inv))
    mask_c = _const(legalize.causal_mask(SEQ))

    op = relax.op
    with bb.function("main", params):
        with bb.dataflow():
            xn = legalize.rms_norm(bb, x, Wn1, SEQ, D)
            # kv projections (per kv-head): K^T and V
            Kt, V = [], []
            for k in range(KV):
                Kk = bb.emit(op.matmul(xn, Wk[k]))             # [SEQ,HD]
                Kk = legalize.rope(bb, Kk, cos_c, sin_c, rot_c)
                Kt.append(bb.emit(op.permute_dims(Kk, axes=[1, 0])))  # [HD,SEQ]
                V.append(bb.emit(op.matmul(xn, Wv[k])))        # [SEQ,HD]
            attn = None
            for h in range(H):
                kv = h // GPK
                Q = bb.emit(op.matmul(xn, Wq[h]))              # [SEQ,HD]
                Q = legalize.rope(bb, Q, cos_c, sin_c, rot_c)
                S = bb.emit(op.matmul(Q, Kt[kv]))              # [SEQ,SEQ]
                S = bb.emit(op.multiply(S, scale_c))
                S = bb.emit(op.add(S, mask_c))
                Pr = legalize.softmax_lastdim(bb, S, SEQ, SEQ)
                ctx = bb.emit(op.matmul(Pr, V[kv]))            # [SEQ,HD]
                part = bb.emit(op.matmul(ctx, Wo[h]))          # [SEQ,D]
                attn = part if attn is None else bb.emit(op.add(attn, part))
            hres = bb.emit(op.add(x, attn))                   # residual 1
            hn = legalize.rms_norm(bb, hres, Wn2, SEQ, D)
            ffn = legalize.swiglu(bb, hn, Wg, Wu, Wd, SEQ, D, F)
            y = bb.emit(op.add(hres, ffn))                    # residual 2
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def ref_layer(W, cos, sin):
    x = W["x"].astype(np.float64)
    def rms(X, w):
        ms = np.mean(X ** 2, axis=1, keepdims=True)
        return X / np.sqrt(ms) * w
    half = HD // 2
    def rope_apply(M):
        out = np.zeros_like(M)
        for p in range(SEQ):
            for d in range(HD):
                rh = -M[p, d + half] if d < half else M[p, d - half]
                out[p, d] = M[p, d] * cos[p, d] + rh * sin[p, d]
        return out
    Wn1 = W["Wn1"].astype(np.float64); Wn2 = W["Wn2"].astype(np.float64)
    Xn = rms(x, Wn1)
    Kt = {}; Vh = {}
    for k in range(KV):
        Kk = rope_apply(Xn @ W[f"Wk{k}"].astype(np.float64))
        Kt[k] = Kk.T
        Vh[k] = Xn @ W[f"Wv{k}"].astype(np.float64)
    attn = np.zeros((SEQ, D))
    for h in range(H):
        kv = h // GPK
        Q = rope_apply(Xn @ W[f"Wq{h}"].astype(np.float64))
        S = (Q @ Kt[kv]) / np.sqrt(HD)
        for i in range(SEQ):
            for j in range(SEQ):
                if j > i:
                    S[i, j] = -np.inf
        S = S - S.max(axis=1, keepdims=True)
        e = np.exp(S); Pr = e / e.sum(axis=1, keepdims=True)
        attn += (Pr @ Vh[kv]) @ W[f"Wo{h}"].astype(np.float64)
    hres = x + attn
    Hn = rms(hres, Wn2)
    gate = Hn @ W["Wg"].astype(np.float64); up = Hn @ W["Wu"].astype(np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    ffn = (silu * up) @ W["Wd"].astype(np.float64)
    return hres + ffn


def test_full_layer():
    cos, sin, rot = legalize.rope_tables(SEQ, HD)
    W = make_weights()
    mod = build_layer_mod(cos, sin, rot)
    got = driver.run_module(mod, W)
    exp = ref_layer(W, cos, sin)

    maxabs = float(np.max(np.abs(exp)))
    maxerr = float(np.max(np.abs(got - exp)))
    rel = maxerr / (maxabs + 1e-6)
    asm, mp = compile_func(mod["main"])
    assert maxerr < 0.05 * maxabs + 0.05, f"maxerr={maxerr} maxabs={maxabs}"
    return rel, maxerr, maxabs, len(asm.words), mp.top


if __name__ == "__main__":
    rel, maxerr, maxabs, instr, gsize = test_full_layer()
    print(f"[PASS] FULL layer [SEQ={SEQ},D={D},H={H},KV={KV},HD={HD},F={F}]")
    print(f"   rel={rel:.4g}  maxerr={maxerr:.4g}  max|out|={maxabs:.4g}")
    print(f"   instructions={instr}  G-buffer={gsize} FP16")
    print("ALL M7 TESTS PASSED")
