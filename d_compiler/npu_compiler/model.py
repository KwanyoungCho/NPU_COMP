"""Parametric Llama transformer layer (prefill) as a Relax graph + numpy reference.

One builder, three configs: reduced proxy / medium (64-multiple, runnable) /
real Llama 3.2 3B. Built from legalize primitives so the hybrid backend can
compile it (matmul -> TIR, elementwise -> direct).

Flow:  h = x + Attn(RMSNorm(x))   (GQA + RoPE + causal softmax)
       y = h + FFN(RMSNorm(h))    (SwiGLU)
"""
from collections import namedtuple
import numpy as np
from tvm import relax

from . import legalize

LayerConfig = namedtuple(
    "LayerConfig",
    "name SEQ D H KV HD F eps rope_base rope_scale")

# Real Llama 3.2 3B (prefill SEQ padded to a 64-multiple, e.g. 128)
LLAMA_3_2_3B = LayerConfig("llama-3.2-3B", SEQ=128, D=3072, H=24, KV=8, HD=128,
                           F=8192, eps=1e-5, rope_base=500000.0, rope_scale=True)
# Minimal all-64-multiple layer: exercises the TIR path, small enough to run in mysim
MEDIUM = LayerConfig("medium-64x", SEQ=64, D=64, H=1, KV=1, HD=64,
                     F=64, eps=1e-5, rope_base=10000.0, rope_scale=False)
# Reduced proxy (non-64 dims -> mostly direct path), matches the original llama_layer
REDUCED = LayerConfig("reduced", SEQ=8, D=64, H=4, KV=2, HD=16,
                      F=128, eps=0.0, rope_base=10000.0, rope_scale=False)


def _f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def _const(a):
    return relax.const(np.asarray(a, dtype="float16"))


def rope_tables(cfg):
    return legalize.rope_tables(cfg.SEQ, cfg.HD, base=cfg.rope_base,
                                llama3_scaling=cfg.rope_scale)


def make_weights(cfg, seed=0):
    rng = np.random.default_rng(seed)
    ws = 0.2
    W = {
        "x":   _f16(rng.standard_normal((cfg.SEQ, cfg.D))),
        "Wn1": _f16(rng.uniform(0.8, 1.2, (1, cfg.D))),
        "Wn2": _f16(rng.uniform(0.8, 1.2, (1, cfg.D))),
        "Wg":  _f16(rng.uniform(-ws, ws, (cfg.D, cfg.F))),
        "Wu":  _f16(rng.uniform(-ws, ws, (cfg.D, cfg.F))),
        "Wd":  _f16(rng.uniform(-ws, ws, (cfg.F, cfg.D))),
    }
    for h in range(cfg.H):
        W[f"Wq{h}"] = _f16(rng.uniform(-ws, ws, (cfg.D, cfg.HD)))
        W[f"Wo{h}"] = _f16(rng.uniform(-ws, ws, (cfg.HD, cfg.D)))
    for k in range(cfg.KV):
        W[f"Wk{k}"] = _f16(rng.uniform(-ws, ws, (cfg.D, cfg.HD)))
        W[f"Wv{k}"] = _f16(rng.uniform(-ws, ws, (cfg.D, cfg.HD)))
    return W


def build_layer_module(cfg, cos, sin, rot):
    SEQ, D, H, KV, HD, F = cfg.SEQ, cfg.D, cfg.H, cfg.KV, cfg.HD, cfg.F
    GPK = H // KV
    bb = relax.BlockBuilder()

    def P(name, shape):
        return relax.Var(name, relax.TensorStructInfo(list(shape), "float16"))

    x = P("x", (SEQ, D)); Wn1 = P("Wn1", (1, D)); Wn2 = P("Wn2", (1, D))
    Wq = [P(f"Wq{h}", (D, HD)) for h in range(H)]
    Wo = [P(f"Wo{h}", (HD, D)) for h in range(H)]
    Wk = [P(f"Wk{k}", (D, HD)) for k in range(KV)]
    Wv = [P(f"Wv{k}", (D, HD)) for k in range(KV)]
    Wg = P("Wg", (D, F)); Wu = P("Wu", (D, F)); Wd = P("Wd", (F, D))
    params = [x, Wn1, Wn2] + Wq + Wo + Wk + Wv + [Wg, Wu, Wd]

    cos_c, sin_c, rot_c = _const(cos), _const(sin), _const(rot)
    scale_c = _const(np.full((SEQ, SEQ), 1.0 / float(np.sqrt(HD))))
    mask_c = _const(legalize.causal_mask(SEQ))
    op = relax.op

    with bb.function("main", params):
        with bb.dataflow():
            xn = legalize.rms_norm(bb, x, Wn1, SEQ, D, eps=cfg.eps)
            Kt, V = [], []
            for k in range(KV):
                Kk = bb.emit(op.matmul(xn, Wk[k]))
                Kk = legalize.rope(bb, Kk, cos_c, sin_c, rot_c)
                Kt.append(bb.emit(op.permute_dims(Kk, axes=[1, 0])))
                V.append(bb.emit(op.matmul(xn, Wv[k])))
            attn = None
            for h in range(H):
                kv = h // GPK
                Q = bb.emit(op.matmul(xn, Wq[h]))
                Q = legalize.rope(bb, Q, cos_c, sin_c, rot_c)
                S = bb.emit(op.matmul(Q, Kt[kv]))
                S = bb.emit(op.multiply(S, scale_c))
                S = bb.emit(op.add(S, mask_c))
                Pr = legalize.softmax_lastdim(bb, S, SEQ, SEQ)
                ctx = bb.emit(op.matmul(Pr, V[kv]))
                part = bb.emit(op.matmul(ctx, Wo[h]))
                attn = part if attn is None else bb.emit(op.add(attn, part))
            hres = bb.emit(op.add(x, attn))
            hn = legalize.rms_norm(bb, hres, Wn2, SEQ, D, eps=cfg.eps)
            ffn = legalize.swiglu(bb, hn, Wg, Wu, Wd, SEQ, D, F)
            y = bb.emit(op.add(hres, ffn))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def ref_layer(cfg, W, cos, sin):
    SEQ, D, H, KV, HD, F = cfg.SEQ, cfg.D, cfg.H, cfg.KV, cfg.HD, cfg.F
    GPK = H // KV
    half = HD // 2
    x = W["x"].astype(np.float64)

    def rms(X, w):
        ms = np.mean(X ** 2, axis=1, keepdims=True) + cfg.eps
        return X / np.sqrt(ms) * w

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
