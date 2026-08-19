"""Qwen3 decoder graphs as Relax IR over the shared legalize primitives.

Llama-shaped flow (pre-norm attention + SwiGLU, per-layer GQA KV cache) with
Gemma-style per-head QK-Norm before RoPE.  The query width H*head_dim differs
from hidden_size, the score scale is the default 1/sqrt(head_dim), and V is
not normalized.  Cache layout matches the Llama compiler: prefill returns
``[y, K0,V0,...]`` per KV head, decode appends in-program and returns the new
K/V columns for the host cache.
"""
from __future__ import annotations

import numpy as np
from tvm import relax

from . import legalize
from .gemma4_graph import banded_causal_mask, gemma_freqs_row


def _c(value):
    return relax.const(np.asarray(value, dtype="float16"))


def _P(name, shape):
    return relax.Var(name, relax.TensorStructInfo(list(shape), "float16"))


def _score_scale(attention):
    if attention.score_scale is None:
        return 1.0 / float(np.sqrt(attention.head_dim))
    return float(attention.score_scale)


def _common_params(spec, layer_index, S):
    layer = spec.layers[layer_index]
    attention = layer.attention
    D = spec.hidden_size
    HD = attention.head_dim
    return layer, attention, D, HD, {
        "x": _P("x", (S, D)),
        "Wn1": _P("Wn1", (1, D)),
        "Wn2": _P("Wn2", (1, D)),
        "Wq": _P("Wq", (D, attention.num_query_heads * HD)),
        "Wqn": _P("Wqn", (1, HD)),
        "Wk": _P("Wk", (D, attention.num_kv_heads * HD)),
        "Wkn": _P("Wkn", (1, HD)),
        "Wv": _P("Wv", (D, attention.num_kv_heads * HD)),
        "Wo": _P("Wo", (attention.num_query_heads * HD, D)),
        "Wg": _P("Wg", (D, layer.ffn_hidden)),
        "Wu": _P("Wu", (D, layer.ffn_hidden)),
        "Wd": _P("Wd", (layer.ffn_hidden, D)),
    }


def build_qwen3_prefill_layer_module(spec, layer_index, S, return_cache=True):
    """One Qwen3 decoder layer over S prompt tokens, optionally returning the
    roped/normed K and V per KV head for cache seeding."""
    layer, attention, D, HD, p = _common_params(spec, layer_index, S)
    H = attention.num_query_heads
    KV = attention.num_kv_heads
    GPK = H // KV
    F = layer.ffn_hidden
    eps = spec.rms_norm_eps
    op = relax.op
    params = [p[name] for name in
              ("x", "Wn1", "Wn2", "Wq", "Wqn", "Wk", "Wkn", "Wv", "Wo",
               "Wg", "Wu", "Wd")]
    pos_c = _c(np.arange(S, dtype="float32").reshape(S, 1))
    freqs_c = _c(gemma_freqs_row(attention))
    scale_c = _c(np.full((S, S), _score_scale(attention)))
    mask_c = _c(banded_causal_mask(S))

    def head_slice(bb, tensor, index):
        return bb.emit(op.strided_slice(
            tensor, axes=[1], begin=[index * HD], end=[(index + 1) * HD]))

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            cos, sin = legalize.rope_cos_sin(bb, pos_c, freqs_c, S, HD)
            xn = legalize.rms_norm(bb, p["x"], p["Wn1"], S, D, eps=eps)
            q_all = bb.emit(op.matmul(xn, p["Wq"]))
            k_all = bb.emit(op.matmul(xn, p["Wk"]))
            v_all = bb.emit(op.matmul(xn, p["Wv"]))
            keys, kt, values = [], [], []
            for kv in range(KV):
                key = head_slice(bb, k_all, kv)
                key = legalize.rms_norm(bb, key, p["Wkn"], S, HD, eps=eps)
                key = legalize.rope(bb, key, cos, sin)
                keys.append(key)
                kt.append(bb.emit(op.permute_dims(key, axes=[1, 0])))
                values.append(head_slice(bb, v_all, kv))
            contexts = []
            for head in range(H):
                query = head_slice(bb, q_all, head)
                query = legalize.rms_norm(bb, query, p["Wqn"], S, HD, eps=eps)
                query = legalize.rope(bb, query, cos, sin)
                score = bb.emit(op.matmul(query, kt[head // GPK]))
                score = bb.emit(op.multiply(score, scale_c))
                score = bb.emit(op.add(score, mask_c))
                probability = legalize.softmax_lastdim(bb, score, S, S)
                contexts.append(bb.emit(op.matmul(probability, values[head // GPK])))
            attention_out = bb.emit(op.matmul(
                bb.emit(op.concat(contexts, axis=1)), p["Wo"]))
            h1 = bb.emit(op.add(p["x"], attention_out))
            hn = legalize.rms_norm(bb, h1, p["Wn2"], S, D, eps=eps)
            ffn = legalize.swiglu(bb, hn, p["Wg"], p["Wu"], p["Wd"], S, D, F)
            y = bb.emit(op.add(h1, ffn))
            if return_cache:
                outputs = [y]
                for kv in range(KV):
                    outputs.extend((keys[kv], values[kv]))
                y = bb.emit(op.concat(outputs, axis=1))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def build_qwen3_decode_layer_module(spec, layer_index, context):
    """One Qwen3 decode layer as a single program over an exact context:
    projects/norms/ropes this token's K/V, appends on-device, attends, and
    returns ``[y, K0,V0,...]`` so the host extends the cache once per step."""
    context = int(context)
    if context < 2:
        raise ValueError("fused decode needs at least one cached position")
    layer, attention, D, HD, p = _common_params(spec, layer_index, 1)
    H = attention.num_query_heads
    KV = attention.num_kv_heads
    GPK = H // KV
    F = layer.ffn_hidden
    eps = spec.rms_norm_eps
    op = relax.op
    kt_old = [_P(f"Kt{kv}", (HD, context - 1)) for kv in range(KV)]
    vc_old = [_P(f"Vc{kv}", (context - 1, HD)) for kv in range(KV)]
    pos = _P("pos", (1, 1))
    params = [p[name] for name in
              ("x", "Wn1", "Wn2", "Wq", "Wqn", "Wk", "Wkn", "Wv", "Wo")]
    params += kt_old + vc_old + [pos, p["Wg"], p["Wu"], p["Wd"]]
    freqs_c = _c(gemma_freqs_row(attention))
    scale_c = _c(np.full((1, context), _score_scale(attention)))

    def head_slice(bb, tensor, index):
        return bb.emit(op.strided_slice(
            tensor, axes=[1], begin=[index * HD], end=[(index + 1) * HD]))

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            cos, sin = legalize.rope_cos_sin(bb, pos, freqs_c, 1, HD)
            xn = legalize.rms_norm(bb, p["x"], p["Wn1"], 1, D, eps=eps)
            q_all = bb.emit(op.matmul(xn, p["Wq"]))
            k_all = bb.emit(op.matmul(xn, p["Wk"]))
            v_all = bb.emit(op.matmul(xn, p["Wv"]))
            new_kv, kt_full, v_full = [], [], []
            for kv in range(KV):
                key = head_slice(bb, k_all, kv)
                key = legalize.rms_norm(bb, key, p["Wkn"], 1, HD, eps=eps)
                key = legalize.rope(bb, key, cos, sin)                  # [1,HD]
                value = head_slice(bb, v_all, kv)                        # [1,HD]
                key_col = bb.emit(op.permute_dims(key, axes=[1, 0]))
                kt_full.append(bb.emit(op.concat([kt_old[kv], key_col], axis=1)))
                vt_old = bb.emit(op.permute_dims(vc_old[kv], axes=[1, 0]))
                value_col = bb.emit(op.permute_dims(value, axes=[1, 0]))
                vt_new = bb.emit(op.concat([vt_old, value_col], axis=1))
                v_full.append(bb.emit(op.permute_dims(vt_new, axes=[1, 0])))
                new_kv.extend((key, value))
            contexts = []
            for head in range(H):
                query = head_slice(bb, q_all, head)
                query = legalize.rms_norm(bb, query, p["Wqn"], 1, HD, eps=eps)
                query = legalize.rope(bb, query, cos, sin)
                score = bb.emit(op.matmul(query, kt_full[head // GPK]))
                score = bb.emit(op.multiply(score, scale_c))
                probability = legalize.softmax_lastdim(bb, score, 1, context)
                contexts.append(bb.emit(op.matmul(probability, v_full[head // GPK])))
            attention_out = bb.emit(op.matmul(
                bb.emit(op.concat(contexts, axis=1)), p["Wo"]))
            h1 = bb.emit(op.add(p["x"], attention_out))
            hn = legalize.rms_norm(bb, h1, p["Wn2"], 1, D, eps=eps)
            ffn = legalize.swiglu(bb, hn, p["Wg"], p["Wu"], p["Wd"], 1, D, F)
            y = bb.emit(op.add(h1, ffn))
            gv = bb.emit_output(bb.emit(op.concat([y] + new_kv, axis=1)))
        bb.emit_func_output(gv)
    return bb.finalize()


def build_qwen3_final_norm_module(spec, S):
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
