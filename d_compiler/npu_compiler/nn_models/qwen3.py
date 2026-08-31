"""Qwen3 defined with the standard TVM frontend (relax.frontend.nn).

The graph is Llama's with one addition: Qwen3 normalizes each attention head
of Q and K over the head dimension before RoPE, so the two families share
everything else and this module reuses it.
"""
from __future__ import annotations

import numpy as np
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op

from .llama import MLP, _apply_rope, causal_mask, rope_freqs, rope_inputs


class Qwen3Config:
    def __init__(self, hf_config):
        self.hidden_size = int(hf_config["hidden_size"])
        self.intermediate_size = int(hf_config["intermediate_size"])
        self.num_layers = int(hf_config["num_hidden_layers"])
        self.num_heads = int(hf_config["num_attention_heads"])
        self.num_kv_heads = int(hf_config["num_key_value_heads"])
        self.head_dim = int(hf_config.get(
            "head_dim", self.hidden_size // self.num_heads))
        self.vocab_size = int(hf_config["vocab_size"])
        self.rms_eps = float(hf_config.get("rms_norm_eps", 1e-6))
        self.rope_theta = float(hf_config.get("rope_theta", 1000000.0))
        self.rope_scaling = bool(hf_config.get("rope_scaling"))
        self.tie_embeddings = bool(hf_config.get("tie_word_embeddings", False))
        self.dtype = "float16"


class Attention(nn.Module):
    def __init__(self, config: Qwen3Config):
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
        self.q_norm = nn.RMSNorm(config.head_dim, -1, config.rms_eps, bias=False)
        self.k_norm = nn.RMSNorm(config.head_dim, -1, config.rms_eps, bias=False)

    def forward(self, hidden, cos, sin, mask):
        seq = hidden.shape[0]
        h, kv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q = self.q_norm(op.reshape(self.q_proj(hidden), [seq, h, hd]))
        k = self.k_norm(op.reshape(self.k_proj(hidden), [seq, kv, hd]))
        v = op.reshape(self.v_proj(hidden), [seq, kv, hd])
        q = _apply_rope(q, cos, sin, hd)
        k = _apply_rope(k, cos, sin, hd)
        group = h // kv
        k = op.reshape(op.repeat(k, group, axis=1), [seq, h, hd])
        v = op.reshape(op.repeat(v, group, axis=1), [seq, h, hd])
        q = op.permute_dims(q, [1, 0, 2])
        k = op.permute_dims(k, [1, 0, 2])
        v = op.permute_dims(v, [1, 0, 2])
        scores = op.matmul(q, op.permute_dims(k, [0, 2, 1]))
        scores = op.multiply(scores, nn.Tensor.from_scalar(
            1.0 / np.sqrt(hd), dtype=scores.dtype))
        scores = op.add(scores, mask)
        out = op.matmul(op.softmax(scores, axis=-1), v)
        out = op.reshape(op.permute_dims(out, [1, 0, 2]), [seq, h * hd])
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
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


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
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
        return self.lm_head(self.norm(hidden))

    def get_default_spec(self, seq):
        hd, d, h = (self.config.head_dim, self.config.hidden_size,
                    self.config.num_heads)
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
    """-> (IRModule, [(param_name, Parameter)], config)."""
    config = Qwen3Config(hf_config)
    model = Qwen3Model(config)
    model.to(config.dtype)
    mod, params = model.export_tvm(spec=model.get_default_spec(seq))
    return mod, params, config


def model_config(assets, layers=0):
    """The config ``build_prefill`` wants, optionally truncated in depth."""
    config = dict(assets.config)
    if layers:
        config["num_hidden_layers"] = layers
    return config


def runtime_inputs(assets, config, token_ids):
    """The non-parameter inputs of ``prefill``, by name."""
    seq = len(token_ids)
    cos, sin = rope_inputs(config, np.arange(seq))
    return {
        "input_embeds": assets.embedding(
            [int(i) for i in token_ids]).astype(np.float16),
        "cos": cos, "sin": sin,
        "mask": causal_mask(config.num_heads, seq),
    }


def load_params(assets, params, config):
    """Checkpoint arrays in the module's parameter order."""
    values = []
    for name, parameter in params:
        key = hf_param_map(name, config.num_layers)
        if key == "lm_head.weight" and key not in assets.weight_map:
            key = "model.embed_tokens.weight"          # tied embeddings
        value = assets._slice(key, (slice(None),) * len(parameter.shape))
        shape = tuple(int(d) for d in parameter.shape)
        if value.shape != shape:
            raise ValueError(f"{name}: checkpoint {value.shape} != {shape}")
        values.append(np.ascontiguousarray(value, np.float16))
    return values


def hf_param_map(name, num_layers):
    """nn.Module parameter name -> HF checkpoint tensor name."""
    if name == "norm.weight":
        return "model.norm.weight"
    if name == "lm_head.weight":
        return "lm_head.weight"       # tied to embed_tokens on the small models
    if name.startswith("layers."):
        return "model." + name
    raise KeyError(name)


__all__ = ["Qwen3Config", "Qwen3Model", "build_prefill", "hf_param_map",
           "rope_freqs", "rope_inputs", "causal_mask"]
