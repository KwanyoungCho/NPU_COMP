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

    def forward(self, hidden, cos, sin, mask, with_cache=False):
        seq = hidden.shape[0]
        h, kv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q = op.reshape(self.q_proj(hidden), [seq, h, hd])
        k = op.reshape(self.k_proj(hidden), [seq, kv, hd])
        v = op.reshape(self.v_proj(hidden), [seq, kv, hd])
        # cos/sin are [seq, 1, hd] and broadcast over heads
        q = _apply_rope(q, cos, sin, hd)
        k = _apply_rope(k, cos, sin, hd)
        cache = (k, v) if with_cache else None
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
        out = self.o_proj(out)
        return (out, cache) if with_cache else out

    def decode(self, hidden, cos, sin, k_prev, v_prev, mask):
        """One token against a fixed-capacity cache.

        K arrives transposed [kv, hd, C] and V as [kv, C, hd] where C is the
        cache capacity, not the current length; the additive ``mask``
        [1, 1, C+1] carries the length (zero over the valid slots and this
        token, the fp16 floor elsewhere).  This is what makes ONE compiled
        decode program serve every step: the shapes never change, only the
        mask and the cache contents do.  This token's K/V are appended
        in-graph for the attention and also returned so the host can write
        them into the cache slot for the next step.
        """
        h, kv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q = op.reshape(self.q_proj(hidden), [1, h, hd])
        k = op.reshape(self.k_proj(hidden), [1, kv, hd])
        v = op.reshape(self.v_proj(hidden), [1, kv, hd])
        q = _apply_rope(q, cos, sin, hd)
        k = _apply_rope(k, cos, sin, hd)
        k_new = op.permute_dims(k, [1, 2, 0])           # [kv, hd, 1]
        v_new = op.permute_dims(v, [1, 0, 2])           # [kv, 1, hd]
        keys = op.concat([k_prev, k_new], dim=2)        # [kv, hd, C+1]
        values = op.concat([v_prev, v_new], dim=1)      # [kv, C+1, hd]
        group = h // kv
        keys = op.repeat(keys, group, axis=0)           # [h, hd, C+1]
        values = op.repeat(values, group, axis=0)       # [h, C+1, hd]
        q = op.permute_dims(q, [1, 0, 2])               # [h, 1, hd]
        scores = op.matmul(q, keys)                     # [h, 1, C+1]
        scores = op.multiply(scores, nn.Tensor.from_scalar(
            1.0 / np.sqrt(hd), dtype=scores.dtype))
        scores = op.add(scores, mask)                   # kills empty slots
        out = op.matmul(op.softmax(scores, axis=-1), values)   # [h, 1, hd]
        out = op.reshape(op.permute_dims(out, [1, 0, 2]), [1, h * hd])
        return self.o_proj(out), k_new, v_new


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

    def forward(self, hidden, cos, sin, mask, with_cache=False):
        attended = self.self_attn(self.input_layernorm(hidden), cos, sin,
                                  mask, with_cache)
        cache = None
        if with_cache:
            attended, cache = attended
        hidden = op.add(hidden, attended)
        hidden = op.add(hidden, self.mlp(self.post_attention_layernorm(hidden)))
        return (hidden, cache) if with_cache else hidden

    def decode(self, hidden, cos, sin, k_prev, v_prev, mask):
        attended, k_new, v_new = self.self_attn.decode(
            self.input_layernorm(hidden), cos, sin, k_prev, v_prev, mask)
        hidden = op.add(hidden, attended)
        hidden = op.add(hidden, self.mlp(self.post_attention_layernorm(hidden)))
        return hidden, k_new, v_new


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
        # logits for every position; the host reads the last row.  Selecting it
        # here would need a gather, whose index the static codegen cannot
        # evaluate -- see the optimization backlog (constant-index take ->
        # slice) for the pass that would remove the extra work.
        return self.lm_head(hidden)

    def prefill_cache(self, input_embeds: nn.Tensor, cos: nn.Tensor,
                      sin: nn.Tensor, mask: nn.Tensor):
        """Prefill that also returns the stacked KV cache: the keys already
        transposed to [L, kv, hd, S] (what the decode score matmul reads) and
        the values as [L, kv, S, hd]."""
        hidden = input_embeds
        keys, values = [], []
        for layer in self.layers:
            hidden, (k, v) = layer(hidden, cos, sin, mask, with_cache=True)
            keys.append(op.unsqueeze(op.permute_dims(k, [1, 2, 0]), 0))
            values.append(op.unsqueeze(op.permute_dims(v, [1, 0, 2]), 0))
        logits = self.lm_head(self.norm(hidden))
        return logits, op.concat(keys, dim=0), op.concat(values, dim=0)

    def decode(self, input_embeds: nn.Tensor, cos: nn.Tensor, sin: nn.Tensor,
               k_cache: nn.Tensor, v_cache: nn.Tensor, mask: nn.Tensor):
        """One token against the fixed-capacity cache; -> (logits, this
        step's K rows [L,kv,hd,1] and V rows [L,kv,1,hd]) for the host to
        write into the next free slot."""
        layers = len(self.layers)
        kv = self.config.num_kv_heads
        hd = self.config.head_dim
        prev = k_cache.shape[3]           # the capacity C
        k_split = op.split(k_cache, layers, axis=0) if layers > 1 else [k_cache]
        v_split = op.split(v_cache, layers, axis=0) if layers > 1 else [v_cache]
        hidden = input_embeds
        new_keys, new_values = [], []
        for index, layer in enumerate(self.layers):
            k_prev = op.reshape(k_split[index], [kv, hd, prev])
            v_prev = op.reshape(v_split[index], [kv, prev, hd])
            hidden, k_new, v_new = layer.decode(hidden, cos, sin,
                                                k_prev, v_prev, mask)
            new_keys.append(op.unsqueeze(k_new, 0))
            new_values.append(op.unsqueeze(v_new, 0))
        logits = self.lm_head(self.norm(hidden))
        return (logits, op.concat(new_keys, dim=0),
                op.concat(new_values, dim=0))

    def get_default_spec(self, seq, decode_context=0):
        hd = self.config.head_dim
        d = self.config.hidden_size
        h = self.config.num_heads
        kv = self.config.num_kv_heads
        layers = self.config.num_layers
        dtype = self.config.dtype
        spec = {
            "prefill": {
                "input_embeds": nn.spec.Tensor([seq, d], dtype),
                "cos": nn.spec.Tensor([seq, 1, hd], dtype),
                "sin": nn.spec.Tensor([seq, 1, hd], dtype),
                "mask": nn.spec.Tensor([h, seq, seq], dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
        }
        if decode_context:
            capacity = decode_context
            spec["prefill_cache"] = dict(spec["prefill"])
            spec["decode"] = {
                "input_embeds": nn.spec.Tensor([1, d], dtype),
                "cos": nn.spec.Tensor([1, 1, hd], dtype),
                "sin": nn.spec.Tensor([1, 1, hd], dtype),
                "k_cache": nn.spec.Tensor([layers, kv, hd, capacity], dtype),
                "v_cache": nn.spec.Tensor([layers, kv, capacity, hd], dtype),
                "mask": nn.spec.Tensor([1, 1, capacity + 1], dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            }
        return nn.spec.ModuleSpec.from_raw(spec, self)


def build_prefill(hf_config, seq):
    """-> (IRModule, [(param_name, Parameter)])."""
    config = LlamaConfig(hf_config)
    model = LlamaModel(config)
    model.to(config.dtype)
    mod, params = model.export_tvm(spec=model.get_default_spec(seq))
    return mod, params, config


def build_generate(hf_config, seq, capacity):
    """The two programs of a deployed model: prefill_cache at the prompt
    length and ONE decode at a fixed cache ``capacity``.

    Static shapes cannot vary per step, so the decode program always attends
    over ``capacity + 1`` slots and an additive mask input carries the
    current length -- one compiled program serves the whole generation.
    """
    config = LlamaConfig(hf_config)
    model = LlamaModel(config)
    model.to(config.dtype)
    mod, params = model.export_tvm(
        spec=model.get_default_spec(seq, decode_context=capacity))
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
        return "lm_head.weight"          # tied to embed_tokens for Llama 3.2
    parts = name.split(".")
    if parts[0] == "layers":
        return "model." + name
    raise KeyError(name)
