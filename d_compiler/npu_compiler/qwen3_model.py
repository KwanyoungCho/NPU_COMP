"""Streaming access to the official Qwen3-4B checkpoint.

Same discipline as the Llama/Gemma loaders: the sharded BF16 safetensors stay
untouched, every access reads only the needed rows/columns and converts that
view to FP16.  Layer geometry comes from ``model_spec.qwen3_spec``.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from .model_spec import qwen3_spec

MODEL_ID = "Qwen/Qwen3-4B"

LAYER_NORMS = ("input_layernorm", "post_attention_layernorm")


def default_model_path():
    configured = os.environ.get("NPU_QWEN3_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "build" / "qwen3_4b_hf"


class ModelAssetError(RuntimeError):
    pass


class Qwen3Assets:
    """Validated, slice-oriented view of the official Qwen3-4B checkpoint."""

    def __init__(self, path=None):
        self.path = Path(path or default_model_path()).resolve()
        config_path = self.path / "config.json"
        index_path = self.path / "model.safetensors.index.json"
        if not config_path.exists() or not index_path.exists():
            raise ModelAssetError(
                f"Qwen3-4B assets not found in {self.path}; set NPU_QWEN3_PATH")
        self.config = json.loads(config_path.read_text())
        self.spec = qwen3_spec(self.config)
        index = json.loads(index_path.read_text())
        self.weight_map = index["weight_map"]
        self.shards = {name: self.path / name for name in set(self.weight_map.values())}
        missing = [str(item) for item in self.shards.values() if not item.exists()]
        if missing:
            raise ModelAssetError(f"missing checkpoint shard(s): {missing}")
        self._tokenizer = None
        self._files = {}
        self._lock = threading.Lock()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.path, local_files_only=True, clean_up_tokenization_spaces=False)
        return self._tokenizer

    def _slice(self, key, selection):
        """One BF16 checkpoint slice as a contiguous FP16 NumPy array."""
        if key not in self.weight_map:
            raise KeyError(key)
        from safetensors import safe_open
        shard = self.shards[self.weight_map[key]]
        with self._lock:
            if shard not in self._files:
                self._files[shard] = safe_open(shard, framework="pt", device="cpu")
            tensor = self._files[shard].get_slice(key)[selection]
        return np.ascontiguousarray(
            tensor.to(dtype=__import__("torch").float16).numpy())

    def embedding(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("empty token id list")
        rows = [self._slice("model.embed_tokens.weight",
                            (slice(int(i), int(i) + 1), slice(None)))
                for i in ids]
        return np.concatenate(rows, axis=0)

    def module_shape(self, layer, module):
        """Logical row-major B[K,N] shape of one layer linear."""
        spec = self.spec.layers[layer]
        attention = spec.attention
        d = self.spec.hidden_size
        q_width = attention.num_query_heads * attention.head_dim
        kv_width = attention.num_kv_heads * attention.head_dim
        shapes = {
            "self_attn.q_proj": (d, q_width),
            "self_attn.k_proj": (d, kv_width),
            "self_attn.v_proj": (d, kv_width),
            "self_attn.o_proj": (q_width, d),
            "mlp.gate_proj": (d, spec.ffn_hidden),
            "mlp.up_proj": (d, spec.ffn_hidden),
            "mlp.down_proj": (spec.ffn_hidden, d),
        }
        if module not in shapes:
            raise KeyError(f"layer {layer} has no module {module}")
        return shapes[module]

    def linear(self, layer, module):
        shape = self.module_shape(layer, module)
        matrix_nk = self._slice(
            f"model.layers.{int(layer)}.{module}.weight", (slice(None), slice(None)))
        if matrix_nk.shape != (shape[1], shape[0]):
            raise ModelAssetError(
                f"layer {layer} {module} has checkpoint shape {matrix_nk.shape}, "
                f"expected {(shape[1], shape[0])}")
        return np.ascontiguousarray(matrix_nk.T)

    def norm(self, layer, name):
        if name not in LAYER_NORMS:
            raise KeyError(name)
        return self._slice(
            f"model.layers.{int(layer)}.{name}.weight", slice(None)).reshape(1, -1)

    def qk_norm(self, layer, which):
        if which not in ("q_norm", "k_norm"):
            raise KeyError(which)
        return self._slice(
            f"model.layers.{int(layer)}.self_attn.{which}.weight", slice(None)).reshape(1, -1)

    def final_norm(self):
        return self._slice("model.norm.weight", slice(None)).reshape(1, -1)

    def lm_head_packed(self, panel=64):
        """LM head as consecutive B[K,panel] vocabulary column panels
        (tied embedding, or the separate lm_head weight when untied)."""
        panel = int(panel)
        vocab = self.spec.vocab_size
        hidden = self.spec.hidden_size
        if panel < 1 or vocab % panel:
            raise ValueError(f"vocab {vocab} must be divisible by panel {panel}")
        key = ("model.embed_tokens.weight" if self.spec.tie_word_embeddings
               else "lm_head.weight")
        matrix_vk = self._slice(key, (slice(None), slice(None)))
        panels = matrix_vk.reshape(vocab // panel, panel, hidden).transpose(0, 2, 1)
        return np.ascontiguousarray(panels).reshape(-1)

    def validate_keyset(self):
        required = {"model.embed_tokens.weight", "model.norm.weight"}
        suffixes = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
        for layer in range(self.spec.num_layers):
            required.update(f"model.layers.{layer}.{suffix}" for suffix in suffixes)
        if not self.spec.tie_word_embeddings:
            required.add("lm_head.weight")
        missing = sorted(required - set(self.weight_map))
        if missing:
            raise ModelAssetError(f"checkpoint keyset is incomplete: {missing[:8]}")
        return len(required)
