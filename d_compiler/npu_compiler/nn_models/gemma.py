"""Gemma 4 E2B (text) defined with the standard TVM frontend.

This family departs from Llama in more ways than Qwen3 does, and each of them
is visible below:

* **two attention kinds.** Layers are sliding-window or full, with different
  head dimensions (256 vs 512), different RoPE thetas, and -- for the full
  ones -- proportional RoPE, where only the first ``partial_rotary_factor``
  of the angles are non-zero.  Zero frequency gives cos=1/sin=0, so the plain
  full-width rotate-half passes those dimensions through untouched and no
  second RoPE variant is needed.
* **shared KV.** The last layers do not project K and V at all; they attend
  with the K/V of an earlier owner layer of the same kind.
* **per-layer embeddings.** Each layer injects a row from a table the host
  precomputes, through a gate and a projection.
* **five norms per layer**, a weight-less V-norm, attention at scale 1.0, a
  tanh-GELU MLP, and a per-layer output scalar.

The final logit softcap is deliberately not in the graph: it is monotonic, so
argmax is unaffected, and the validated path applies it on the comparison side
when scoring logits against HF.  ``softcap`` below does that for callers who
need the scored values.
"""
from __future__ import annotations

import numpy as np
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op

from .llama import _rotate_half


def rope_freqs(head_dim, theta, partial_rotary_factor=1.0):
    """Half-duplicated frequency row [head_dim] for one attention kind.

    Proportional RoPE keeps the full head-dim exponent but zeroes every angle
    past ``partial_rotary_factor``; those dimensions then pass through.
    """
    half = head_dim // 2
    angles = int(partial_rotary_factor * head_dim) // 2
    freqs = np.zeros(half, dtype=np.float64)
    freqs[:angles] = theta ** (-2.0 * np.arange(angles) / head_dim)
    return np.concatenate([freqs, freqs])


def rope_inputs(head_dim, theta, positions, partial_rotary_factor=1.0):
    """Host-side cos/sin tables [S, 1, head_dim] for one attention kind."""
    freqs = rope_freqs(head_dim, theta, partial_rotary_factor)
    angle = np.asarray(positions, dtype=np.float64)[:, None] * freqs[None, :]
    return (np.cos(angle).astype(np.float16)[:, None, :],
            np.sin(angle).astype(np.float16)[:, None, :])


def banded_mask(num_heads, seq, window=None, dtype=np.float16):
    """Additive mask: causal, and for sliding attention also i-j >= window."""
    mask = np.zeros((seq, seq), dtype=np.float32)
    for i in range(seq):
        for j in range(seq):
            if j > i or (window is not None and i - j >= window):
                mask[i, j] = -30000.0
    return np.broadcast_to(mask, (num_heads, seq, seq)).astype(dtype).copy()


def softcap(logits, cap):
    """The final logit softcap, applied outside the graph (monotonic)."""
    if not cap:
        return logits
    return cap * np.tanh(np.asarray(logits, np.float32) / cap)


class Attention(nn.Module):
    """One layer's attention; owner layers project K/V, shared layers reuse."""

    def __init__(self, spec, hidden_size, eps, owns_cache):
        self.heads = spec.num_query_heads
        self.kv_heads = spec.num_kv_heads
        self.head_dim = spec.head_dim
        self.eps = eps
        self.owns_cache = owns_cache
        width = self.heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, width, bias=False)
        self.o_proj = nn.Linear(width, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, -1, eps, bias=False)
        if owns_cache:
            kv_width = self.kv_heads * self.head_dim
            self.k_proj = nn.Linear(hidden_size, kv_width, bias=False)
            self.v_proj = nn.Linear(hidden_size, kv_width, bias=False)
            self.k_norm = nn.RMSNorm(self.head_dim, -1, eps, bias=False)

    def project_kv(self, hidden, cos, sin):
        """-> (k, v) for this layer's cache slot, normalized and roped."""
        seq = hidden.shape[0]
        kv, hd = self.kv_heads, self.head_dim
        k = self.k_norm(op.reshape(self.k_proj(hidden), [seq, kv, hd]))
        k = op.add(op.multiply(k, cos),
                   op.multiply(_rotate_half(k, hd), sin))
        v = op.reshape(self.v_proj(hidden), [seq, kv, hd])
        # the V norm has no weight of its own; a constant one leaves the
        # standard rms_norm lowering in place instead of a second recipe
        ones = nn.Tensor.from_const(np.ones(hd, "float16"))
        v = op.rms_norm(v, ones, axes=[-1], epsilon=self.eps)
        return k, v

    def forward(self, hidden, cos, sin, mask, k, v):
        seq = hidden.shape[0]
        h, kv, hd = self.heads, self.kv_heads, self.head_dim
        q = self.q_norm(op.reshape(self.q_proj(hidden), [seq, h, hd]))
        q = op.add(op.multiply(q, cos),
                   op.multiply(_rotate_half(q, hd), sin))
        group = h // kv
        k = op.reshape(op.repeat(k, group, axis=1), [seq, h, hd])
        v = op.reshape(op.repeat(v, group, axis=1), [seq, h, hd])
        q = op.permute_dims(q, [1, 0, 2])
        k = op.permute_dims(k, [1, 0, 2])
        v = op.permute_dims(v, [1, 0, 2])
        # Gemma attends at scale 1.0; the query norm has already set the scale
        scores = op.add(op.matmul(q, op.permute_dims(k, [0, 2, 1])), mask)
        out = op.matmul(op.softmax(scores, axis=-1), v)
        out = op.reshape(op.permute_dims(out, [1, 0, 2]), [seq, h * hd])
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, spec, hidden_size, eps):
        attention = spec.attention
        self.eps = eps
        self.owns_cache = spec.owns_cache
        self.kv_owner = spec.kv_owner
        self.self_attn = Attention(attention, hidden_size, eps, spec.owns_cache)
        self.mlp_gate = nn.Linear(hidden_size, spec.ffn_hidden, bias=False)
        self.mlp_up = nn.Linear(hidden_size, spec.ffn_hidden, bias=False)
        self.mlp_down = nn.Linear(spec.ffn_hidden, hidden_size, bias=False)
        self.per_layer_input_gate = nn.Linear(hidden_size, spec.ple_dim,
                                              bias=False)
        self.per_layer_projection = nn.Linear(spec.ple_dim, hidden_size,
                                              bias=False)
        self.input_layernorm = nn.RMSNorm(hidden_size, -1, eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, -1, eps,
                                                   bias=False)
        self.pre_feedforward_layernorm = nn.RMSNorm(hidden_size, -1, eps,
                                                    bias=False)
        self.post_feedforward_layernorm = nn.RMSNorm(hidden_size, -1, eps,
                                                     bias=False)
        self.post_per_layer_input_norm = nn.RMSNorm(hidden_size, -1, eps,
                                                    bias=False)
        self.layer_scale = nn.Parameter((1, 1), dtype="float16")

    def forward(self, hidden, ple, cos, sin, mask, cached):
        xn = self.input_layernorm(hidden)
        # K and V come from the same normalized input the query uses, so an
        # owner layer projects them here and shared layers take the owner's
        kv = self.self_attn.project_kv(xn, cos, sin) if self.owns_cache else cached
        attention = self.self_attn(xn, cos, sin, mask, *kv)
        h1 = op.add(hidden, self.post_attention_layernorm(attention))
        f = self.pre_feedforward_layernorm(h1)
        gate = op.gelu(self.mlp_gate(f), approximate="tanh")
        ffn = self.mlp_down(op.multiply(gate, self.mlp_up(f)))
        h2 = op.add(h1, self.post_feedforward_layernorm(ffn))
        gated = op.multiply(op.gelu(self.per_layer_input_gate(h2),
                                    approximate="tanh"), ple)
        h3 = op.add(h2, self.post_per_layer_input_norm(
            self.per_layer_projection(gated)))
        return op.multiply(h3, self.layer_scale), kv


class GemmaModel(nn.Module):
    """Text-only Gemma 4 E2B; embeddings and the PLE table stay on the host."""

    def __init__(self, spec):
        self.spec = spec
        self.dtype = "float16"
        self.layers = nn.ModuleList(
            [DecoderLayer(layer, spec.hidden_size, spec.rms_norm_eps)
             for layer in spec.layers])
        self.norm = nn.RMSNorm(spec.hidden_size, -1, spec.rms_norm_eps,
                               bias=False)
        self.lm_head = nn.Linear(spec.hidden_size, spec.vocab_size, bias=False)

    def prefill(self, input_embeds: nn.Tensor, ple: nn.Tensor,
                cos_sliding: nn.Tensor, sin_sliding: nn.Tensor,
                cos_full: nn.Tensor, sin_full: nn.Tensor,
                mask_sliding: nn.Tensor, mask_full: nn.Tensor):
        return self.lm_head(self.norm(self._layers(
            input_embeds, ple, cos_sliding, sin_sliding, cos_full, sin_full,
            mask_sliding, mask_full)))

    def hidden(self, input_embeds: nn.Tensor, ple: nn.Tensor,
               cos_sliding: nn.Tensor, sin_sliding: nn.Tensor,
               cos_full: nn.Tensor, sin_full: nn.Tensor,
               mask_sliding: nn.Tensor, mask_full: nn.Tensor):
        """The hidden state after the layers -- what the recorded HF reference
        stores per layer, so a truncated model can be checked against it."""
        return self._layers(input_embeds, ple, cos_sliding, sin_sliding,
                            cos_full, sin_full, mask_sliding, mask_full)

    def _layers(self, input_embeds, ple, cos_sliding, sin_sliding,
                cos_full, sin_full, mask_sliding, mask_full):
        spec = self.spec
        # split returns a bare tensor rather than a tuple for one piece
        rows = (op.split(ple, len(spec.layers), axis=0)
                if len(spec.layers) > 1 else [ple])
        hidden = input_embeds
        cache = {}
        for index, layer in enumerate(self.layers):
            sliding = spec.layers[index].attention.kind == "sliding"
            row = op.reshape(rows[index],
                             [input_embeds.shape[0], spec.layers[index].ple_dim])
            hidden, kv = layer(
                hidden, row,
                cos_sliding if sliding else cos_full,
                sin_sliding if sliding else sin_full,
                mask_sliding if sliding else mask_full,
                cache.get(layer.kv_owner))
            if layer.owns_cache:
                cache[index] = kv
        return hidden

    def get_default_spec(self, seq):
        spec = self.spec
        heads = spec.layers[0].attention.num_query_heads
        sliding = next(l.attention for l in spec.layers
                       if l.attention.kind == "sliding")
        full = next((l.attention for l in spec.layers
                     if l.attention.kind == "full"), sliding)
        tensor = nn.spec.Tensor
        inputs = {
                "input_embeds": tensor([seq, spec.hidden_size], self.dtype),
                "ple": tensor([len(spec.layers), seq,
                               spec.layers[0].ple_dim], self.dtype),
                "cos_sliding": tensor([seq, 1, sliding.head_dim], self.dtype),
                "sin_sliding": tensor([seq, 1, sliding.head_dim], self.dtype),
                "cos_full": tensor([seq, 1, full.head_dim], self.dtype),
                "sin_full": tensor([seq, 1, full.head_dim], self.dtype),
                "mask_sliding": tensor([heads, seq, seq], self.dtype),
                "mask_full": tensor([heads, seq, seq], self.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
        }
        return nn.spec.ModuleSpec.from_raw(
            {"prefill": inputs, "hidden": dict(inputs)}, self)


def build_prefill(spec, seq):
    """-> (IRModule, [(param_name, Parameter)], spec)."""
    model = GemmaModel(spec)
    model.to("float16")
    mod, params = model.export_tvm(spec=model.get_default_spec(seq))
    return mod, params, spec


def host_inputs(spec, positions):
    """cos/sin and masks for both attention kinds, for the given positions."""
    seq = len(positions)
    heads = spec.layers[0].attention.num_query_heads
    sliding = next(l.attention for l in spec.layers
                   if l.attention.kind == "sliding")
    full = next((l.attention for l in spec.layers
                 if l.attention.kind == "full"), sliding)
    cos_s, sin_s = rope_inputs(sliding.head_dim, sliding.rope_theta, positions,
                               sliding.partial_rotary_factor)
    cos_f, sin_f = rope_inputs(full.head_dim, full.rope_theta, positions,
                               full.partial_rotary_factor)
    return {
        "cos_sliding": cos_s, "sin_sliding": sin_s,
        "cos_full": cos_f, "sin_full": sin_f,
        "mask_sliding": banded_mask(heads, seq, sliding.window),
        "mask_full": banded_mask(heads, seq, full.window),
    }


_LINEARS = {
    "self_attn.q_proj": "self_attn.q_proj", "self_attn.o_proj": "self_attn.o_proj",
    "self_attn.k_proj": "self_attn.k_proj", "self_attn.v_proj": "self_attn.v_proj",
    "mlp_gate": "mlp.gate_proj", "mlp_up": "mlp.up_proj",
    "mlp_down": "mlp.down_proj",
    "per_layer_input_gate": "per_layer_input_gate",
    "per_layer_projection": "per_layer_projection",
}


def model_config(assets, layers=0):
    """The spec ``build_prefill`` wants, optionally truncated in depth."""
    import dataclasses

    spec = assets.spec
    if layers:
        spec = dataclasses.replace(spec, layers=spec.layers[:layers])
    return spec


def runtime_inputs(assets, spec, token_ids):
    """The non-parameter inputs of ``prefill``, by name.

    The per-layer embedding table is a host input like the RoPE tables: it is
    a gather from a precomputed table, which the static codegen cannot do.
    """
    ids = [int(i) for i in token_ids]
    ple_dim = spec.layers[0].ple_dim
    rows = assets.ple_table()[ids] if hasattr(assets, "ple_table") else None
    rows = np.asarray(rows, np.float16).reshape(len(ids), -1, ple_dim)
    inputs = {"input_embeds": assets.embedding(ids).astype(np.float16),
              "ple": np.ascontiguousarray(
                  rows[:, :len(spec.layers)].transpose(1, 0, 2))}
    inputs.update(host_inputs(spec, np.arange(len(ids))))
    return inputs


def load_params(assets, params, spec=None):
    """Checkpoint arrays in the module's parameter order.

    Goes through the asset accessors rather than rebuilding checkpoint names:
    they already know this checkpoint's prefix and layout, and nn.Linear wants
    exactly the [out, in] the checkpoint stores.
    """
    values = []
    for name, parameter in params:
        values.append(_load_one(assets, name))
        shape = tuple(int(d) for d in parameter.shape)
        if values[-1].shape != shape:
            raise ValueError(f"{name}: checkpoint {values[-1].shape} != {shape}")
    return values


def _load_one(assets, name):
    if name == "norm.weight":
        return assets.final_norm().reshape(-1)
    if name == "lm_head.weight":                     # tied to the embeddings
        return assets._slice("embed_tokens.weight", (slice(None), slice(None)))
    parts = name.split(".")
    if parts[0] != "layers":
        raise KeyError(name)
    layer, rest = int(parts[1]), ".".join(parts[2:])
    if rest == "layer_scale":
        return assets.layer_scalar(layer)
    module, field = rest.rsplit(".", 1)
    if field != "weight":
        raise KeyError(name)
    if module in ("self_attn.q_norm", "self_attn.k_norm"):
        return assets.qk_norm(layer, module.split(".")[-1]).reshape(-1)
    if module.endswith("layernorm") or module == "post_per_layer_input_norm":
        return assets.norm(layer, module).reshape(-1)
    if module in _LINEARS:
        return assets._slice(f"layers.{layer}.{_LINEARS[module]}.weight",
                             (slice(None), slice(None)))
    raise KeyError(name)
