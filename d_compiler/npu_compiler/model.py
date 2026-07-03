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


def _P(name, shape):
    return relax.Var(name, relax.TensorStructInfo(list(shape), "float16"))


def _attn_head(bb, xn, Wq_h, Wo_h, Kt_kv, V_kv, scale, mask, cos, sin, rot, qrows, kcols):
    """One attention head, shared by prefill and decode builders:
    Q = rope(xn @ Wq); ctx = softmax(Q @ Kt * scale + mask) @ V; out = ctx @ Wo.
    Kt_kv/V_kv are supplied by the caller (prefill computes them from xn; decode
    passes the KV-cache params), so the per-head body is identical for both."""
    op = relax.op
    Q = legalize.rope(bb, bb.emit(op.matmul(xn, Wq_h)), cos, sin, rot)
    S = bb.emit(op.matmul(Q, Kt_kv))
    S = bb.emit(op.multiply(S, scale))
    S = bb.emit(op.add(S, mask))
    Pr = legalize.softmax_lastdim(bb, S, qrows, kcols)
    ctx = bb.emit(op.matmul(Pr, V_kv))
    return bb.emit(op.matmul(ctx, Wo_h))


def _residual_ffn(bb, x, attn, Wn2, Wg, Wu, Wd, seq, d, f, eps):
    """hres = x + attn ; y = hres + SwiGLU(RMSNorm(hres)). Shared tail."""
    op = relax.op
    hres = bb.emit(op.add(x, attn))
    hn = rms_norm_(bb, hres, Wn2, seq, d, eps)
    ffn = legalize.swiglu(bb, hn, Wg, Wu, Wd, seq, d, f)
    return bb.emit(op.add(hres, ffn))


def rms_norm_(bb, x, w, seq, d, eps):      # thin alias to keep call sites short
    return legalize.rms_norm(bb, x, w, seq, d, eps=eps)


def rope_tables(cfg):
    return legalize.rope_tables(cfg.SEQ, cfg.HD, base=cfg.rope_base,
                                llama3_scaling=cfg.rope_scale)


def make_weights(cfg, seed=0, ws=0.2):
    rng = np.random.default_rng(seed)
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
                part = _attn_head(bb, xn, Wq[h], Wo[h], Kt[h // GPK], V[h // GPK],
                                  scale_c, mask_c, cos_c, sin_c, rot_c, SEQ, SEQ)
                attn = part if attn is None else bb.emit(op.add(attn, part))
            y = _residual_ffn(bb, x, attn, Wn2, Wg, Wu, Wd, SEQ, D, F, cfg.eps)
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


# ==================== decode-step graph builders (KV cache) ====================
# Decode is split into two static-shape kernels; the host appends the new token's
# K/V into the cache between them (dynamic offset -> host, not NPU). See driver.generate.

def build_kv_proj_module(cfg):
    """Decode kernel 1: x_new[1,D] -> concat([K0,V0,K1,V1,...])[1,2*KV*HD] (K roped).
    Host slices the output and writes K into cache column pos, V into row pos."""
    D, KV, HD = cfg.D, cfg.KV, cfg.HD
    rot_c = _const(legalize.rope_tables(1, HD, base=cfg.rope_base, llama3_scaling=cfg.rope_scale)[2])
    op = relax.op
    x = _P("x", (1, D)); Wn1 = _P("Wn1", (1, D))
    Wk = [_P(f"Wk{k}", (D, HD)) for k in range(KV)]
    Wv = [_P(f"Wv{k}", (D, HD)) for k in range(KV)]
    cos = _P("cos", (1, HD)); sin = _P("sin", (1, HD))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, Wn1] + Wk + Wv + [cos, sin]):
        with bb.dataflow():
            xn = rms_norm_(bb, x, Wn1, 1, D, cfg.eps)
            outs = []
            for k in range(KV):
                Kk = legalize.rope(bb, bb.emit(op.matmul(xn, Wk[k])), cos, sin, rot_c)
                outs += [Kk, bb.emit(op.matmul(xn, Wv[k]))]
            gv = bb.emit_output(bb.emit(op.concat(outs, axis=1)))
        bb.emit_func_output(gv)
    return bb.finalize()


def build_kv_proj_batched(cfg, S):
    """Batched-prefill K/V: x[S,D] -> concat([K0,V0,K1,V1,...])[S, 2*KV*HD] (K roped
    per position). Host writes K_h.T into cache columns 0..S-1 and V_h into rows
    0..S-1 — the SAME layout decode reads, so a batched prefill can seed the cache
    that the decode loop continues from. cos/sin are [S,HD] (positions 0..S-1)."""
    D, KV, HD = cfg.D, cfg.KV, cfg.HD
    rot_c = _const(legalize.rope_tables(1, HD, base=cfg.rope_base, llama3_scaling=cfg.rope_scale)[2])
    op = relax.op
    x = _P("x", (S, D)); Wn1 = _P("Wn1", (1, D))
    Wk = [_P(f"Wk{k}", (D, HD)) for k in range(KV)]
    Wv = [_P(f"Wv{k}", (D, HD)) for k in range(KV)]
    cos = _P("cos", (S, HD)); sin = _P("sin", (S, HD))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, Wn1] + Wk + Wv + [cos, sin]):
        with bb.dataflow():
            xn = rms_norm_(bb, x, Wn1, S, D, cfg.eps)
            outs = []
            for k in range(KV):
                Kk = legalize.rope(bb, bb.emit(op.matmul(xn, Wk[k])), cos, sin, rot_c)
                outs += [Kk, bb.emit(op.matmul(xn, Wv[k]))]
            gv = bb.emit_output(bb.emit(op.concat(outs, axis=1)))
        bb.emit_func_output(gv)
    return bb.finalize()


def build_prefill_layer_module(cfg, S):
    """Batched prefill of one layer: x[S,D] -> concat([y, K0,V0,K1,V1,...]) where
    y[S,D] is the layer output (feeds the next layer) and Kk[S,HD](roped)/Vk[S,HD]
    seed the cache — host writes Kk.T into columns 0..S-1, Vk into rows 0..S-1 (the
    same layout decode reads). Reuses _attn_head/_residual_ffn (same math as decode,
    just M=S with a causal mask). cos/sin are [S,HD] (positions 0..S-1)."""
    D, H, KV, HD, F = cfg.D, cfg.H, cfg.KV, cfg.HD, cfg.F
    GPK = H // KV
    rot_c = _const(legalize.rope_tables(1, HD, base=cfg.rope_base, llama3_scaling=cfg.rope_scale)[2])
    scale_c = _const(np.full((S, S), 1.0 / float(np.sqrt(HD))))
    mask_c = _const(legalize.causal_mask(S))
    op = relax.op
    x = _P("x", (S, D)); Wn1 = _P("Wn1", (1, D)); Wn2 = _P("Wn2", (1, D))
    Wq = [_P(f"Wq{h}", (D, HD)) for h in range(H)]
    Wk = [_P(f"Wk{k}", (D, HD)) for k in range(KV)]
    Wv = [_P(f"Wv{k}", (D, HD)) for k in range(KV)]
    Wo = [_P(f"Wo{h}", (HD, D)) for h in range(H)]
    cos = _P("cos", (S, HD)); sin = _P("sin", (S, HD))
    Wg = _P("Wg", (D, F)); Wu = _P("Wu", (D, F)); Wd = _P("Wd", (F, D))
    params = [x, Wn1, Wn2] + Wq + Wk + Wv + Wo + [cos, sin, Wg, Wu, Wd]
    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            xn = rms_norm_(bb, x, Wn1, S, D, cfg.eps)
            Kk_list, Kt, V = [], [], []
            for k in range(KV):
                Kk = legalize.rope(bb, bb.emit(op.matmul(xn, Wk[k])), cos, sin, rot_c)  # [S,HD]
                Kk_list.append(Kk)
                Kt.append(bb.emit(op.permute_dims(Kk, axes=[1, 0])))                    # [HD,S]
                V.append(bb.emit(op.matmul(xn, Wv[k])))                                 # [S,HD]
            attn = None
            for h in range(H):
                part = _attn_head(bb, xn, Wq[h], Wo[h], Kt[h // GPK], V[h // GPK],
                                  scale_c, mask_c, cos, sin, rot_c, S, S)
                attn = part if attn is None else bb.emit(op.add(attn, part))
            y = _residual_ffn(bb, x, attn, Wn2, Wg, Wu, Wd, S, D, F, cfg.eps)
            outs = [y]
            for k in range(KV):
                outs += [Kk_list[k], V[k]]
            gv = bb.emit_output(bb.emit(op.concat(outs, axis=1)))       # [S, D + 2*KV*HD]
        bb.emit_func_output(gv)
    return bb.finalize()


def build_attn_ffn_module(cfg, MAX):
    """Decode kernel 2: x_new[1,D] + Kt[HD,MAX]/Vc[MAX,HD] (+ runtime mask) -> y[1,D].
    Attention runs over the whole MAX cache; `mask` zeros slots j>pos."""
    D, H, KV, HD, F = cfg.D, cfg.H, cfg.KV, cfg.HD, cfg.F
    GPK = H // KV
    rot_c = _const(legalize.rope_tables(1, HD, base=cfg.rope_base, llama3_scaling=cfg.rope_scale)[2])
    scale_c = _const(np.full((1, MAX), 1.0 / float(np.sqrt(HD))))
    op = relax.op
    x = _P("x", (1, D)); Wn1 = _P("Wn1", (1, D)); Wn2 = _P("Wn2", (1, D))
    Wq = [_P(f"Wq{h}", (D, HD)) for h in range(H)]
    Wo = [_P(f"Wo{h}", (HD, D)) for h in range(H)]
    Kt = [_P(f"Kt{k}", (HD, MAX)) for k in range(KV)]
    Vc = [_P(f"Vc{k}", (MAX, HD)) for k in range(KV)]
    cos = _P("cos", (1, HD)); sin = _P("sin", (1, HD)); mask = _P("mask", (1, MAX))
    Wg = _P("Wg", (D, F)); Wu = _P("Wu", (D, F)); Wd = _P("Wd", (F, D))
    params = [x, Wn1, Wn2] + Wq + Wo + Kt + Vc + [cos, sin, mask, Wg, Wu, Wd]
    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            xn = rms_norm_(bb, x, Wn1, 1, D, cfg.eps)
            attn = None
            for h in range(H):
                part = _attn_head(bb, xn, Wq[h], Wo[h], Kt[h // GPK], Vc[h // GPK],
                                  scale_c, mask, cos, sin, rot_c, 1, MAX)
                attn = part if attn is None else bb.emit(op.add(attn, part))
            gv = bb.emit_output(_residual_ffn(bb, x, attn, Wn2, Wg, Wu, Wd, 1, D, F, cfg.eps))
        bb.emit_func_output(gv)
    return bb.finalize()


# ==================== decode numpy reference (KV cache) ====================
# float64 ground truth, split into the same two pieces the kernels emit so each
# can be validated 1:1 (tests/test_decode.py). K stored transposed [HD,L].

def new_cache(cfg):
    return {"Kt": [np.zeros((cfg.HD, 0)) for _ in range(cfg.KV)],
            "V":  [np.zeros((0, cfg.HD)) for _ in range(cfg.KV)]}


def _rms_np(X, w, eps):
    ms = np.mean(X ** 2, axis=1, keepdims=True) + eps
    return X / np.sqrt(ms) * w


def _rope_np(M, cos, sin, rot, positions):
    return M * cos[positions] + (M @ rot) * sin[positions]


def ref_kv_proj(cfg, W, x_query, positions, cos, sin, rot):
    """New tokens' K (roped) / V per KV head."""
    xn = _rms_np(x_query.astype(np.float64), W["Wn1"].astype(np.float64), cfg.eps)
    Ks = [_rope_np(xn @ W[f"Wk{k}"].astype(np.float64), cos, sin, rot, positions) for k in range(cfg.KV)]
    Vs = [xn @ W[f"Wv{k}"].astype(np.float64) for k in range(cfg.KV)]
    return Ks, Vs


def cache_append(cache, Ks, Vs):
    for k in range(len(Ks)):
        cache["Kt"][k] = np.concatenate([cache["Kt"][k], Ks[k].T], axis=1)   # K -> columns
        cache["V"][k] = np.concatenate([cache["V"][k], Vs[k]], axis=0)       # V -> rows
    return cache


def ref_attn_ffn(cfg, W, x_query, cache, positions, cos, sin, rot, stable_softmax=False):
    """Attention over existing cache + FFN (no append)."""
    D, H, HD, GPK = cfg.D, cfg.H, cfg.HD, cfg.H // cfg.KV
    M = x_query.shape[0]
    xn = _rms_np(x_query.astype(np.float64), W["Wn1"].astype(np.float64), cfg.eps)
    attn = np.zeros((M, D))
    for h in range(H):
        kv = h // GPK
        Q = _rope_np(xn @ W[f"Wq{h}"].astype(np.float64), cos, sin, rot, positions)
        Sc = (Q @ cache["Kt"][kv]) / np.sqrt(HD)
        for i in range(M):
            Sc[i, positions[i] + 1:] = -np.inf
        if stable_softmax:
            Sc = Sc - Sc.max(axis=1, keepdims=True)
        e = np.exp(Sc); Pr = e / e.sum(axis=1, keepdims=True)
        attn += (Pr @ cache["V"][kv]) @ W[f"Wo{h}"].astype(np.float64)
    hres = x_query.astype(np.float64) + attn
    hn = _rms_np(hres, W["Wn2"].astype(np.float64), cfg.eps)
    gate = hn @ W["Wg"].astype(np.float64); up = hn @ W["Wu"].astype(np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    return hres + (silu * up) @ W["Wd"].astype(np.float64)


def ref_layer_cache(cfg, W, x_query, cache, base_len, cos, sin, rot, stable_softmax=False):
    positions = np.arange(base_len, base_len + x_query.shape[0])
    Ks, Vs = ref_kv_proj(cfg, W, x_query, positions, cos, sin, rot)
    cache_append(cache, Ks, Vs)
    return ref_attn_ffn(cfg, W, x_query, cache, positions, cos, sin, rot, stable_softmax), cache


def ref_generate(cfg, W, X_full, n_prompt, cos, sin, rot):
    cache = new_cache(cfg)
    _, cache = ref_layer_cache(cfg, W, X_full[:n_prompt], cache, 0, cos, sin, rot)
    outs = []
    for t in range(n_prompt, X_full.shape[0]):
        y, cache = ref_layer_cache(cfg, W, X_full[t:t + 1], cache, t, cos, sin, rot)
        outs.append(y)
    return np.concatenate(outs, axis=0)


def decode_self_consistency(cfg, n_prompt, n_gen, seed=0):
    """prefill-all[n_prompt:] == prefill(prompt)+decode(rest)  (float64)."""
    N = n_prompt + n_gen
    cos, sin, rot = legalize.rope_tables(N, cfg.HD, base=cfg.rope_base, llama3_scaling=cfg.rope_scale)
    W = make_weights(cfg, seed=seed)
    rng = np.random.default_rng(seed + 1)
    X = np.asarray(rng.standard_normal((N, cfg.D)), dtype=np.float16).astype(np.float64)
    cache = new_cache(cfg)
    Y_all, _ = ref_layer_cache(cfg, W, X, cache, 0, cos, sin, rot)
    Y_dec = ref_generate(cfg, W, X, n_prompt, cos, sin, rot)
    err = float(np.max(np.abs(Y_all[n_prompt:] - Y_dec)))
    return err, err / float(np.max(np.abs(Y_all[n_prompt:])) + 1e-12)


# ==================== full LM: embedding + N layers + lm_head ====================
# CPU: embedding lookup (dynamic gather) + argmax. NPU: the transformer matmuls,
# incl. lm_head (hidden -> logits). See driver.generate_tokens.

def make_gen_weights(cfg, n_layers, vocab, seed=0, ws=0.2):
    """Per-layer weights + top-level {Wemb[vocab,D], Wn_f[1,D], Wlm[D,vocab]}.
    ws scales the projection weights: at large dims (3B) use a small ws so attention
    scores stay in the fp16-safe range (our softmax omits max-subtraction)."""
    layer_Ws = [make_weights(cfg, seed=seed + 100 * (l + 1), ws=ws) for l in range(n_layers)]
    rng = np.random.default_rng(seed + 7)
    top = {"Wemb": _f16(rng.standard_normal((vocab, cfg.D)) * 0.5),
           "Wn_f": _f16(rng.uniform(0.8, 1.2, (1, cfg.D))),
           "Wlm":  _f16(rng.uniform(-ws, ws, (cfg.D, vocab)))}
    return layer_Ws, top


def build_lm_head_module(cfg, vocab):
    """h[1,D] -> logits[1,vocab] = RMSNorm(h, Wn_f) @ Wlm  (final norm + lm_head)."""
    D = cfg.D
    op = relax.op
    h = _P("h", (1, D)); Wn_f = _P("Wn_f", (1, D)); Wlm = _P("Wlm", (D, vocab))
    bb = relax.BlockBuilder()
    with bb.function("main", [h, Wn_f, Wlm]):
        with bb.dataflow():
            hn = rms_norm_(bb, h, Wn_f, 1, D, cfg.eps)
            gv = bb.emit_output(bb.emit(op.matmul(hn, Wlm)))
        bb.emit_func_output(gv)
    return bb.finalize()


def ref_logits(cfg, top, h):
    hn = _rms_np(h.astype(np.float64), top["Wn_f"].astype(np.float64), cfg.eps)
    return hn @ top["Wlm"].astype(np.float64)          # [1, vocab]


def ref_generate_tokens(cfg, layer_Ws, top, prompt_ids, n_gen, cos, sin, rot):
    """Greedy autoregressive generation (float64 ref). Returns generated ids."""
    caches = [new_cache(cfg) for _ in layer_Ws]
    emb = top["Wemb"].astype(np.float64)
    tokens = list(prompt_ids); P = len(prompt_ids); p = 0
    while len(tokens) < P + n_gen:
        x = emb[tokens[p]:tokens[p] + 1]                # [1,D]
        for l, W in enumerate(layer_Ws):
            x, caches[l] = ref_layer_cache(cfg, W, x, caches[l], p, cos, sin, rot)
        if p >= P - 1:
            tokens.append(int(np.argmax(ref_logits(cfg, top, x)[0])))
        p += 1
    return tokens[P:]
