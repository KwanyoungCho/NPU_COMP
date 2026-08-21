"""Gemma 4 E2B text graphs as Relax IR over the shared legalize primitives.

Same construction route as the Llama V3 graphs: relax.BlockBuilder emitting
legalize builders, compiled by the common 0818 backend.  No reshape is needed
for the packed PLE dimension: [S, 35*256] group operations are expressed with
last-axis slice / concat, which the backend executes as strided copies.
"""
from __future__ import annotations

import numpy as np
from tvm import relax

from . import legalize


def _c(value):
    return relax.const(np.asarray(value, dtype="float16"))


def _P(name, shape):
    return relax.Var(name, relax.TensorStructInfo(list(shape), "float16"))


def gemma_freqs_row(attention):
    """Half-duplicated RoPE frequency row [1,HD] for one attention spec.

    Proportional RoPE keeps the full-head-dim exponent but only the first
    ``partial_rotary_factor*HD/2`` angles are non-zero; zero frequency gives
    cos=1/sin=0, so the standard full-dim rotate-half passes those dims
    through unchanged and no slice/concat variant of rope is needed.
    """
    hd = attention.head_dim
    half = hd // 2
    angles = int(attention.partial_rotary_factor * hd) // 2
    freqs = np.zeros(half, dtype=np.float64)
    freqs[:angles] = attention.rope_theta ** (-2.0 * np.arange(angles) / hd)
    return np.concatenate([freqs, freqs])[None, :]


def banded_causal_mask(S, window=None):
    """[S,S] additive mask: causal; sliding also drops i-j >= window."""
    mask = np.zeros((S, S), dtype="float32")
    for i in range(S):
        for j in range(S):
            if j > i or (window is not None and i - j >= window):
                mask[i, j] = -30000.0
    return mask


def _layer_tail(bb, x, attention_out, p, S, D, F, Pd, eps):
    """Shared post-attention chain: post-attn norm/residual, gated tanh-GELU
    MLP with post-FFN norm/residual, PLE injection, and layer scalar."""
    op = relax.op
    h1 = bb.emit(op.add(
        x, legalize.rms_norm(bb, attention_out, p["Wn2"], S, D, eps=eps)))
    f = legalize.rms_norm(bb, h1, p["Wn3"], S, D, eps=eps)
    gate = legalize.gelu_tanh(bb, bb.emit(op.matmul(f, p["Wg"])), S, F)
    up = bb.emit(op.matmul(f, p["Wu"]))
    ffn = bb.emit(op.matmul(bb.emit(op.multiply(gate, up)), p["Wd"]))
    h2 = bb.emit(op.add(
        h1, legalize.rms_norm(bb, ffn, p["Wn4"], S, D, eps=eps)))
    gate2 = legalize.gelu_tanh(bb, bb.emit(op.matmul(h2, p["Wpg"])), S, Pd)
    gated = bb.emit(op.multiply(gate2, p["pli"]))
    projected = bb.emit(op.matmul(gated, p["Wpp"]))
    h3 = bb.emit(op.add(
        h2, legalize.rms_norm(bb, projected, p["Wn5"], S, D, eps=eps)))
    scale = bb.emit(op.broadcast_to(p["ls"], relax.ShapeExpr([S, D])))
    return bb.emit(op.multiply(h3, scale))


def build_gemma4_prefill_layer_module(spec, layer_index, S, return_cache=True):
    """One official Gemma 4 decoder layer over S prompt tokens.

    Owner layers project/norm/rope K and norm V in-program and (optionally)
    return them concatenated after the hidden state for cache seeding; shared
    layers instead take the owner's Kt/V as inputs.  All arithmetic — QK-Norm,
    weight-less V-norm, scale-1 attention, tanh-GELU gated MLP, PLE injection,
    layer scalar — runs on the NPU; ``pli`` rows come from the precomputed
    PLE table.
    """
    layer = spec.layers[layer_index]
    attention = layer.attention
    if attention.num_kv_heads != 1:
        raise ValueError("Gemma 4 graph assumes a single KV head")
    D = spec.hidden_size
    HD = attention.head_dim
    H = attention.num_query_heads
    F = layer.ffn_hidden
    Pd = layer.ple_dim
    eps = spec.rms_norm_eps
    owner = layer.owns_cache
    op = relax.op

    x = _P("x", (S, D))
    pli = _P("pli", (S, Pd))
    Wn1 = _P("Wn1", (1, D)); Wn2 = _P("Wn2", (1, D)); Wn3 = _P("Wn3", (1, D))
    Wn4 = _P("Wn4", (1, D)); Wn5 = _P("Wn5", (1, D))
    Wq = _P("Wq", (D, H * HD)); Wqn = _P("Wqn", (1, HD))
    Wo = _P("Wo", (H * HD, D))
    Wg = _P("Wg", (D, F)); Wu = _P("Wu", (D, F)); Wd = _P("Wd", (F, D))
    Wpg = _P("Wpg", (D, Pd)); Wpp = _P("Wpp", (Pd, D))
    ls = _P("ls", (1, 1))
    params = [x, pli, Wn1, Wn2, Wn3, Wn4, Wn5, Wq, Wqn]
    if owner:
        Wk = _P("Wk", (D, HD)); Wkn = _P("Wkn", (1, HD)); Wv = _P("Wv", (D, HD))
        params += [Wk, Wkn, Wv]
    else:
        Kt = _P("Kt", (HD, S)); Vc = _P("Vc", (S, HD))
        params += [Kt, Vc]
    params += [Wo, Wg, Wu, Wd, Wpg, Wpp, ls]

    pos_c = _c(np.arange(S, dtype="float32").reshape(S, 1))
    freqs_c = _c(gemma_freqs_row(attention))
    window = attention.window if attention.kind == "sliding" else None
    mask_c = _c(banded_causal_mask(S, window))

    def head_slice(bb, tensor, index):
        return bb.emit(op.strided_slice(
            tensor, axes=[1], begin=[index * HD], end=[(index + 1) * HD]))

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            cos, sin = legalize.rope_cos_sin(bb, pos_c, freqs_c, S, HD)
            xn = legalize.rms_norm(bb, x, Wn1, S, D, eps=eps)
            q_all = bb.emit(op.matmul(xn, Wq))
            if owner:
                key = bb.emit(op.matmul(xn, Wk))
                key = legalize.rms_norm(bb, key, Wkn, S, HD, eps=eps)
                key = legalize.rope(bb, key, cos, sin)
                value = bb.emit(op.matmul(xn, Wv))
                value = legalize.rms_norm(bb, value, None, S, HD, eps=eps)
                kt = bb.emit(op.permute_dims(key, axes=[1, 0]))
            else:
                kt, value = Kt, Vc
            contexts = []
            for head in range(H):
                query = head_slice(bb, q_all, head)
                query = legalize.rms_norm(bb, query, Wqn, S, HD, eps=eps)
                query = legalize.rope(bb, query, cos, sin)
                score = bb.emit(op.matmul(query, kt))          # scale is 1.0
                score = bb.emit(op.add(score, mask_c))
                probability = legalize.softmax_lastdim(bb, score, S, S)
                contexts.append(bb.emit(op.matmul(probability, value)))
            attention_out = bb.emit(op.matmul(
                bb.emit(op.concat(contexts, axis=1)), Wo))
            tail = {"Wn2": Wn2, "Wn3": Wn3, "Wn4": Wn4, "Wn5": Wn5,
                    "Wg": Wg, "Wu": Wu, "Wd": Wd, "Wpg": Wpg, "Wpp": Wpp,
                    "pli": pli, "ls": ls}
            y = _layer_tail(bb, x, attention_out, tail, S, D, F, Pd, eps)
            if owner and return_cache:
                y = bb.emit(op.concat([y, key, value], axis=1))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def build_gemma4_decode_layer_module(spec, layer_index, context):
    """One Gemma decode layer as a single program over an exact context.

    Owner layers take the previous ``context-1`` cache, project/norm/rope this
    token's K/V in-program, append on-device, and return ``[y, K_new, V_new]``
    so the host can extend the slot cache (shared layers later in the same
    step read the extended cache).  Shared layers take the full ``context``
    cache.  Contexts within the sliding window need no mask; longer sliding
    contexts are the streaming-attention follow-up (G4.7).
    """
    layer = spec.layers[layer_index]
    attention = layer.attention
    if attention.num_kv_heads != 1:
        raise ValueError("Gemma 4 graph assumes a single KV head")
    context = int(context)
    if context < 2:
        raise ValueError("fused decode needs at least one cached position")
    if attention.kind == "sliding" and context > attention.window:
        raise NotImplementedError(
            "sliding decode beyond the window needs the streaming lowering")
    D = spec.hidden_size
    HD = attention.head_dim
    H = attention.num_query_heads
    F = layer.ffn_hidden
    Pd = layer.ple_dim
    eps = spec.rms_norm_eps
    owner = layer.owns_cache
    op = relax.op

    x = _P("x", (1, D))
    pli = _P("pli", (1, Pd))
    Wn1 = _P("Wn1", (1, D)); Wn2 = _P("Wn2", (1, D)); Wn3 = _P("Wn3", (1, D))
    Wn4 = _P("Wn4", (1, D)); Wn5 = _P("Wn5", (1, D))
    Wq = _P("Wq", (D, H * HD)); Wqn = _P("Wqn", (1, HD))
    Wo = _P("Wo", (H * HD, D))
    Wg = _P("Wg", (D, F)); Wu = _P("Wu", (D, F)); Wd = _P("Wd", (F, D))
    Wpg = _P("Wpg", (D, Pd)); Wpp = _P("Wpp", (Pd, D))
    ls = _P("ls", (1, 1))
    pos = _P("pos", (1, 1))
    params = [x, pli, Wn1, Wn2, Wn3, Wn4, Wn5, Wq, Wqn]
    if owner:
        Wk = _P("Wk", (D, HD)); Wkn = _P("Wkn", (1, HD)); Wv = _P("Wv", (D, HD))
        Kt = _P("Kt", (HD, context - 1)); Vc = _P("Vc", (context - 1, HD))
        params += [Wk, Wkn, Wv, Kt, Vc]
    else:
        Kt = _P("Kt", (HD, context)); Vc = _P("Vc", (context, HD))
        params += [Kt, Vc]
    params += [pos, Wo, Wg, Wu, Wd, Wpg, Wpp, ls]
    freqs_c = _c(gemma_freqs_row(attention))

    def head_slice(bb, tensor, index):
        return bb.emit(op.strided_slice(
            tensor, axes=[1], begin=[index * HD], end=[(index + 1) * HD]))

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            cos, sin = legalize.rope_cos_sin(bb, pos, freqs_c, 1, HD)
            xn = legalize.rms_norm(bb, x, Wn1, 1, D, eps=eps)
            q_all = bb.emit(op.matmul(xn, Wq))
            if owner:
                key = bb.emit(op.matmul(xn, Wk))
                key = legalize.rms_norm(bb, key, Wkn, 1, HD, eps=eps)
                key = legalize.rope(bb, key, cos, sin)                  # [1,HD]
                value = bb.emit(op.matmul(xn, Wv))
                value = legalize.rms_norm(bb, value, None, 1, HD, eps=eps)
                key_col = bb.emit(op.permute_dims(key, axes=[1, 0]))    # [HD,1]
                kt_full = bb.emit(op.concat([Kt, key_col], axis=1))     # [HD,ctx]
                vt_old = bb.emit(op.permute_dims(Vc, axes=[1, 0]))
                value_col = bb.emit(op.permute_dims(value, axes=[1, 0]))
                vt_full = bb.emit(op.concat([vt_old, value_col], axis=1))
                v_full = bb.emit(op.permute_dims(vt_full, axes=[1, 0]))  # [ctx,HD]
            else:
                kt_full, v_full = Kt, Vc
            contexts = []
            for head in range(H):
                query = head_slice(bb, q_all, head)
                query = legalize.rms_norm(bb, query, Wqn, 1, HD, eps=eps)
                query = legalize.rope(bb, query, cos, sin)
                score = bb.emit(op.matmul(query, kt_full))              # scale 1.0
                probability = legalize.softmax_lastdim(bb, score, 1, context)
                contexts.append(bb.emit(op.matmul(probability, v_full)))
            attention_out = bb.emit(op.matmul(
                bb.emit(op.concat(contexts, axis=1)), Wo))
            tail = {"Wn2": Wn2, "Wn3": Wn3, "Wn4": Wn4, "Wn5": Wn5,
                    "Wg": Wg, "Wu": Wu, "Wd": Wd, "Wpg": Wpg, "Wpp": Wpp,
                    "pli": pli, "ls": ls}
            y = _layer_tail(bb, x, attention_out, tail, 1, D, F, Pd, eps)
            if owner:
                y = bb.emit(op.concat([y, key, value], axis=1))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def build_gemma4_final_norm_module(spec, S):
    """Final model RMSNorm over S positions."""
    x = _P("x", (S, spec.hidden_size))
    weight = _P("weight", (1, spec.hidden_size))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, weight]):
        with bb.dataflow():
            gv = bb.emit_output(legalize.rms_norm(
                bb, x, weight, S, spec.hidden_size, eps=spec.rms_norm_eps))
        bb.emit_func_output(gv)
    return bb.finalize()


def build_gemma4_ple_module(spec, S):
    """per_layer_input for S tokens: [S, num_layers*ple_dim].

    Inputs: ``se`` host-scaled embeddings [S,D], raw PLE token rows ``tok``
    [S, L*P], projection ``Wproj`` [D, L*P], group norm weight ``Wn`` [1,P].
    Computes (RMSNorm(se @ Wproj * D^-0.5) + tok * sqrt(P)) * 2^-0.5 with the
    RMSNorm applied per ple_dim group via slice/concat.
    """
    D = spec.hidden_size
    L = spec.num_layers
    Pd = spec.layers[0].ple_dim
    width = L * Pd
    op = relax.op
    se = _P("se", (S, D))
    tok = _P("tok", (S, width))
    Wproj = _P("Wproj", (D, width))
    Wn = _P("Wn", (1, Pd))
    bb = relax.BlockBuilder()
    with bb.function("main", [se, tok, Wproj, Wn]):
        with bb.dataflow():
            proj = bb.emit(op.matmul(se, Wproj))
            proj = bb.emit(op.multiply(
                proj, _c(np.full((S, width), 1.0 / np.sqrt(float(D))))))
            tok_scaled = bb.emit(op.multiply(
                tok, _c(np.full((S, width), np.sqrt(float(Pd))))))
            groups = []
            for index in range(L):
                begin, end = index * Pd, (index + 1) * Pd
                part = bb.emit(op.strided_slice(
                    proj, axes=[1], begin=[begin], end=[end]))
                normed = legalize.rms_norm(
                    bb, part, Wn, S, Pd, eps=spec.rms_norm_eps)
                token_part = bb.emit(op.strided_slice(
                    tok_scaled, axes=[1], begin=[begin], end=[end]))
                combined = bb.emit(op.add(normed, token_part))
                groups.append(bb.emit(op.multiply(
                    combined, _c(np.full((S, Pd), 1.0 / np.sqrt(2.0))))))
            gv = bb.emit_output(bb.emit(op.concat(groups, axis=1)))
        bb.emit_func_output(gv)
    return bb.finalize()
