"""Llama 3.2 3B defined with the standard TVM frontend (relax.frontend.nn).

Structure only; weights come from the HF checkpoint and are bound by parameter
name (see :func:`hf_param_map`).  RoPE follows the same convention as the
existing hand-written path: half-duplicated frequencies and rotate_half, with
Llama-3 frequency scaling, so numerics are comparable.

Two entry points are exported:
  ``prefill(input_embeds[S, D], positions[S])`` -> logits[1, vocab]
  ``decode(input_embeds[1, D], positions[1], k_cache/v_cache)`` — added later.

S0 scope: prefill only, validated on llvm against HF.
"""
from __future__ import annotations

import numpy as np
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op


def llama3_scale_freqs(freqs, factor=32.0, low=1.0, high=4.0, old_ctx=8192):
    """Llama-3 RoPE frequency scaling (same routine as legalize.py)."""
    out = np.empty_like(freqs)
    low_wl, high_wl = old_ctx / low, old_ctx / high
    for i, f in enumerate(freqs):
        wl = 2.0 * np.pi / f
        if wl > low_wl:
            out[i] = f / factor
        elif wl < high_wl:
            out[i] = f
        else:
            s = (old_ctx / wl - low) / (high - low)
            out[i] = (1 - s) * f / factor + s * f
    return out


def rope_freqs(head_dim, base, llama3_scaling):
    """Half-duplicated frequency row [head_dim] shared by cos and sin."""
    half = head_dim // 2
    freqs = base ** (-2.0 * np.arange(half) / head_dim)
    if llama3_scaling:
        freqs = llama3_scale_freqs(freqs)
    return np.concatenate([freqs, freqs])


class LlamaConfig:
    def __init__(self, hf_config):
        self.hidden_size = int(hf_config["hidden_size"])
        self.intermediate_size = int(hf_config["intermediate_size"])
        self.num_layers = int(hf_config["num_hidden_layers"])
        self.num_heads = int(hf_config["num_attention_heads"])
        self.num_kv_heads = int(hf_config["num_key_value_heads"])
        self.head_dim = int(hf_config.get(
            "head_dim", self.hidden_size // self.num_heads))
        self.vocab_size = int(hf_config["vocab_size"])
        self.rms_eps = float(hf_config.get("rms_norm_eps", 1e-5))
        self.rope_theta = float(hf_config.get("rope_theta", 500000.0))
        self.rope_scaling = bool(hf_config.get("rope_scaling"))
        self.dtype = "float16"


def _rotate_half(x, head_dim):
    """[-x2, x1] on the last axis (the Llama convention)."""
    half = head_dim // 2
    x1, x2 = op.split(x, [half], axis=-1)
    return op.concat([op.negative(x2), x1], dim=-1)


def _apply_rope(x, cos, sin, head_dim):
    return op.add(op.multiply(x, cos), op.multiply(_rotate_half(x, head_dim), sin))


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size,
                                config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size,
                                config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size,
                                config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim,
                                config.hidden_size, bias=False)

    def forward(self, hidden, cos, sin, mask):
        seq = hidden.shape[0]
        h, kv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q = op.reshape(self.q_proj(hidden), [seq, h, hd])
        k = op.reshape(self.k_proj(hidden), [seq, kv, hd])
        v = op.reshape(self.v_proj(hidden), [seq, kv, hd])
        # cos/sin are [seq, 1, hd] and broadcast over heads
        q = _apply_rope(q, cos, sin, hd)
        k = _apply_rope(k, cos, sin, hd)
        # grouped-query attention: repeat each kv head h//kv times
        group = h // kv
        k = op.reshape(op.repeat(k, group, axis=1), [seq, h, hd])
        v = op.reshape(op.repeat(v, group, axis=1), [seq, h, hd])
        # [h, seq, hd] for batched attention
        q = op.permute_dims(q, [1, 0, 2])
        k = op.permute_dims(k, [1, 0, 2])
        v = op.permute_dims(v, [1, 0, 2])
        scores = op.matmul(q, op.permute_dims(k, [0, 2, 1]))
        scores = op.multiply(scores, nn.Tensor.from_scalar(
            1.0 / np.sqrt(hd), dtype=scores.dtype))
        scores = op.add(scores, mask)
        # softmax stays in the tensor dtype: the NPU's vector unit already
        # accumulates in FP32 internally, so an explicit float32 round trip
        # would only add casts the machine cannot express
        probs = op.softmax(scores, axis=-1)
        out = op.matmul(probs, v)                       # [h, seq, hd]
        out = op.reshape(op.permute_dims(out, [1, 0, 2]), [seq, h * hd])
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size,
                                   bias=False)

    def forward(self, hidden):
        return self.down_proj(op.multiply(op.silu(self.gate_proj(hidden)),
                                          self.up_proj(hidden)))


class DecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1,
                                          config.rms_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1,
                                                   config.rms_eps, bias=False)

    def forward(self, hidden, cos, sin, mask):
        hidden = op.add(hidden, self.self_attn(
            self.input_layernorm(hidden), cos, sin, mask))
        return op.add(hidden, self.mlp(self.post_attention_layernorm(hidden)))


class LlamaModel(nn.Module):
    """Text-only Llama; the embedding lookup stays on the host (token ids in,
    embeddings out), matching the existing execution model."""

    def __init__(self, config: LlamaConfig):
        self.config = config
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_eps, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def prefill(self, input_embeds: nn.Tensor, cos: nn.Tensor, sin: nn.Tensor,
                mask: nn.Tensor):
        hidden = input_embeds
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, mask)
        hidden = self.norm(hidden)
        last = op.reshape(
            op.take(hidden, nn.Tensor.from_const(
                np.asarray([hidden.shape[0] - 1], dtype="int32")), axis=0),
            [1, self.config.hidden_size])
        return self.lm_head(last)

    def get_default_spec(self, seq):
        hd = self.config.head_dim
        d = self.config.hidden_size
        h = self.config.num_heads
        return nn.spec.ModuleSpec.from_raw({
            "prefill": {
                "input_embeds": nn.spec.Tensor([seq, d], self.config.dtype),
                "cos": nn.spec.Tensor([seq, 1, hd], self.config.dtype),
                "sin": nn.spec.Tensor([seq, 1, hd], self.config.dtype),
                "mask": nn.spec.Tensor([h, seq, seq], self.config.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            }
        }, self)


def build_prefill(hf_config, seq):
    """-> (IRModule, [(param_name, Parameter)])."""
    config = LlamaConfig(hf_config)
    model = LlamaModel(config)
    model.to(config.dtype)
    mod, params = model.export_tvm(spec=model.get_default_spec(seq))
    return mod, params, config


def rope_inputs(config, positions):
    """Host-side cos/sin tables [S, 1, head_dim] for the given positions."""
    freqs = rope_freqs(config.head_dim, config.rope_theta, config.rope_scaling)
    angle = np.asarray(positions, dtype=np.float64)[:, None] * freqs[None, :]
    cos = np.cos(angle).astype(np.float16)[:, None, :]
    sin = np.sin(angle).astype(np.float16)[:, None, :]
    return cos, sin


def causal_mask(num_heads, seq, dtype=np.float16):
    m = np.zeros((seq, seq), dtype=np.float32)
    m[np.triu_indices(seq, k=1)] = -1e4
    return np.broadcast_to(m, (num_heads, seq, seq)).astype(dtype).copy()


def hf_param_map(name, num_layers):
    """nn.Module parameter name -> HF checkpoint tensor name."""
    if name == "norm.weight":
        return "model.norm.weight"
    if name == "lm_head.weight":
        return "lm_head.weight"          # tied to embed_tokens for Llama 3.2
    parts = name.split(".")
    if parts[0] == "layers":
        return "model." + name
    raise KeyError(name)
