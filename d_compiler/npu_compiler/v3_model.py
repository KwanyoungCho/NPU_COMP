"""Streaming access to the official Llama 3.2 3B checkpoint.

The checkpoint stays in its two original BF16 safetensors shards.  Compiler V3
reads only the rows/columns needed by the current vendor tile and converts that
small view to FP16; it never materializes another full-model copy.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from dataclasses import dataclass

import numpy as np


MODEL_REVISION = "13afe5124825b4f3751f836b40dafda64c1ed062"
MODEL_ID = "meta-llama/Llama-3.2-3B"


def default_model_path():
    configured = os.environ.get("NPU_LLAMA32_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "build" / "llama32_3b_hf"


class ModelAssetError(RuntimeError):
    pass


@dataclass(frozen=True)
class LinearTileSource:
    """Lazy B[K,N] operand consumed directly by the V3 GEMM scheduler."""

    shape: tuple
    loader: object
    name: str

    def load_tile(self, k_slice, n_slice):
        return self.loader(k_slice, n_slice)


class Llama32Assets:
    """Validated, slice-oriented view of the official two-shard checkpoint."""

    EXPECTED = {
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "tie_word_embeddings": True,
    }

    def __init__(self, path=None):
        self.path = Path(path or default_model_path()).resolve()
        config_path = self.path / "config.json"
        index_path = self.path / "model.safetensors.index.json"
        if not config_path.exists() or not index_path.exists():
            raise ModelAssetError(
                f"Llama 3.2 3B assets not found in {self.path}; set NPU_LLAMA32_PATH"
            )
        self.config = json.loads(config_path.read_text())
        mismatch = {
            key: (self.config.get(key), expected)
            for key, expected in self.EXPECTED.items()
            if self.config.get(key) != expected
        }
        if mismatch:
            raise ModelAssetError(f"unexpected Llama 3.2 3B config: {mismatch}")
        index = json.loads(index_path.read_text())
        self.weight_map = index["weight_map"]
        self.shards = {name: self.path / name for name in set(self.weight_map.values())}
        missing = [str(path) for path in self.shards.values() if not path.exists()]
        if missing:
            raise ModelAssetError(f"missing checkpoint shard(s): {missing}")
        self._tokenizer = None
        self._files = {}
        self._files_lock = threading.Lock()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.path, local_files_only=True, clean_up_tokenization_spaces=False)
        return self._tokenizer

    def _slice(self, key, selection):
        """Return one BF16 checkpoint slice as a contiguous FP16 NumPy array."""
        if key not in self.weight_map:
            raise KeyError(key)
        from safetensors import safe_open
        shard = self.shards[self.weight_map[key]]
        # safe_open handles are persistent but their slice API is serialized;
        # vendor processes still run concurrently after each small tile is read.
        with self._files_lock:
            if shard not in self._files:
                self._files[shard] = safe_open(shard, framework="pt", device="cpu")
            tensor = self._files[shard].get_slice(key)[selection]
        return np.ascontiguousarray(tensor.to(dtype=__import__("torch").float16).numpy())

    def embedding(self, token_ids):
        """Embedding lookup without loading the full 128256x3072 table."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.config["hidden_size"]), dtype=np.float16)
        # Safetensors slicing does not provide arbitrary gather.  Read one row at
        # a time; prompt token counts are small and repeated ids remain exact.
        rows = [self._slice("model.embed_tokens.weight", (slice(int(i), int(i) + 1), slice(None)))
                for i in ids]
        return np.concatenate(rows, axis=0)

    def linear_tile(self, layer, module, k_slice, n_slice):
        """Return B[K,N] for a checkpoint linear weight stored as [N,K]."""
        if not 0 <= layer < self.config["num_hidden_layers"]:
            raise IndexError(layer)
        key = f"model.layers.{layer}.{module}.weight"
        tile_nk = self._slice(key, (n_slice, k_slice))
        return np.ascontiguousarray(tile_nk.T)

    def linear_source(self, layer, module, shape):
        shape = tuple(shape)
        return LinearTileSource(
            shape,
            lambda k_slice, n_slice: self.linear_tile(layer, module, k_slice, n_slice),
            f"layer{layer}.{module}",
        )

    def linear(self, layer, module, shape):
        """Materialize one official linear weight as row-major B[K,N]."""
        shape = tuple(int(value) for value in shape)
        key = f"model.layers.{int(layer)}.{module}.weight"
        matrix_nk = self._slice(key, (slice(None), slice(None)))
        if matrix_nk.shape != (shape[1], shape[0]):
            raise ModelAssetError(
                f"{key} has checkpoint shape {matrix_nk.shape}, expected {(shape[1], shape[0])}")
        return np.ascontiguousarray(matrix_nk.T)

    def lm_head(self):
        """Materialize the tied embedding as row-major LM-head B[D,V]."""
        matrix_vk = self._slice(
            "model.embed_tokens.weight", (slice(None), slice(None)))
        return np.ascontiguousarray(matrix_vk.T)

    def norm(self, layer, post_attention=False):
        name = "post_attention_layernorm" if post_attention else "input_layernorm"
        return self._slice(f"model.layers.{layer}.{name}.weight", slice(None))

    def final_norm(self):
        return self._slice("model.norm.weight", slice(None))

    def lm_head_tile(self, k_slice, n_slice):
        """Return tied LM-head B[K,V] from embedding rows [V,K]."""
        tile_vk = self._slice("model.embed_tokens.weight", (n_slice, k_slice))
        return np.ascontiguousarray(tile_vk.T)

    def lm_head_source(self):
        shape = (self.config["hidden_size"], self.config["vocab_size"])
        return LinearTileSource(shape, self.lm_head_tile, "tied_lm_head")

    def validate_keyset(self):
        required = {"model.embed_tokens.weight", "model.norm.weight"}
        suffixes = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
        for layer in range(self.config["num_hidden_layers"]):
            required.update(f"model.layers.{layer}.{suffix}" for suffix in suffixes)
        missing = sorted(required - set(self.weight_map))
        if missing:
            raise ModelAssetError(f"checkpoint keyset is incomplete: {missing[:8]}")
        return len(required)
