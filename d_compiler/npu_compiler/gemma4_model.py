"""Streaming access to the official Gemma 4 E2B checkpoint (text-only scope).

Mirrors ``Llama32Assets``: the checkpoint stays in its original monolithic BF16
safetensors file and every access reads only the rows/columns needed right now,
converting that view to FP16.  Text weights live under the multimodal prefix
``model.language_model.``; the vision/audio towers are never touched.

Per-layer geometry (full/sliding head dims, double-wide MLP, shared-KV layers
without k/v weights) comes from ``model_spec.gemma4_e2b_spec``.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from .model_spec import gemma4_e2b_spec

MODEL_ID = "google/gemma-4-E2B"
MODEL_REVISION = "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
PREFIX = "model.language_model."

LAYER_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "post_per_layer_input_norm",
)


def default_model_path():
    configured = os.environ.get("NPU_GEMMA4_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "build" / "gemma4_e2b_hf"


class ModelAssetError(RuntimeError):
    pass


class Gemma4Assets:
    """Validated, slice-oriented view of the official Gemma 4 E2B checkpoint."""

    def __init__(self, path=None):
        self.path = Path(path or default_model_path()).resolve()
        config_path = self.path / "config.json"
        self.shard = self.path / "model.safetensors"
        if not config_path.exists() or not self.shard.exists():
            raise ModelAssetError(
                f"Gemma 4 E2B assets not found in {self.path}; set NPU_GEMMA4_PATH")
        config = json.loads(config_path.read_text())
        self.text_config = config["text_config"]
        # gemma4_e2b_spec validates the structural fields it consumes.
        self.spec = gemma4_e2b_spec(self.text_config)
        self._tokenizer = None
        self._file = None
        self._keys = None
        self._lock = threading.Lock()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.path, local_files_only=True, clean_up_tokenization_spaces=False)
        return self._tokenizer

    def _open(self):
        if self._file is None:
            from safetensors import safe_open
            self._file = safe_open(self.shard, framework="pt", device="cpu")
            self._keys = set(self._file.keys())
        return self._file

    def keys(self):
        self._open()
        return self._keys

    def _slice(self, name, selection):
        """One BF16 slice of ``model.language_model.<name>`` as FP16 NumPy."""
        key = PREFIX + name
        with self._lock:
            handle = self._open()
            if key not in self._keys:
                raise KeyError(key)
            tensor = handle.get_slice(key)[selection]
        return np.ascontiguousarray(
            tensor.to(dtype=__import__("torch").float16).numpy())

    def _rows(self, name, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("empty token id list")
        rows = [self._slice(name, (slice(int(i), int(i) + 1), slice(None)))
                for i in ids]
        return np.concatenate(rows, axis=0)

    def embedding(self, token_ids, *, scaled=True):
        """Token embedding rows; ``scaled`` applies the sqrt(D) embed scale in
        FP16 (the table itself stays raw because the LM head ties to it)."""
        rows = self._rows("embed_tokens.weight", token_ids)
        if scaled:
            rows = rows * np.float16(np.sqrt(float(self.spec.hidden_size)))
        return rows.astype(np.float16)

    def ple_rows(self, token_ids):
        """Raw PLE token rows [n, num_layers*ple_dim]; scaling happens in the
        precomputed-table pipeline."""
        return self._rows("embed_tokens_per_layer.weight", token_ids)

    def module_shape(self, layer, module):
        """Logical row-major B[K,N] shape of one layer linear."""
        spec = self.spec.layers[layer]
        attention = spec.attention
        d = self.spec.hidden_size
        q_width = attention.num_query_heads * attention.head_dim
        kv_width = attention.num_kv_heads * attention.head_dim
        shapes = {
            "self_attn.q_proj": (d, q_width),
            "self_attn.o_proj": (q_width, d),
            "mlp.gate_proj": (d, spec.ffn_hidden),
            "mlp.up_proj": (d, spec.ffn_hidden),
            "mlp.down_proj": (spec.ffn_hidden, d),
            "per_layer_input_gate": (d, spec.ple_dim),
            "per_layer_projection": (spec.ple_dim, d),
        }
        if spec.owns_cache:
            shapes["self_attn.k_proj"] = (d, kv_width)
            shapes["self_attn.v_proj"] = (d, kv_width)
        if module not in shapes:
            raise KeyError(f"layer {layer} has no module {module}")
        return shapes[module]

    def linear(self, layer, module):
        """Materialize one layer linear as row-major B[K,N] (checkpoint is [N,K])."""
        shape = self.module_shape(layer, module)
        matrix_nk = self._slice(
            f"layers.{int(layer)}.{module}.weight", (slice(None), slice(None)))
        if matrix_nk.shape != (shape[1], shape[0]):
            raise ModelAssetError(
                f"layer {layer} {module} has checkpoint shape {matrix_nk.shape}, "
                f"expected {(shape[1], shape[0])}")
        return np.ascontiguousarray(matrix_nk.T)

    def norm(self, layer, name):
        """Layer RMSNorm weight [1,D] (see LAYER_NORMS for valid names)."""
        if name not in LAYER_NORMS:
            raise KeyError(name)
        return self._slice(f"layers.{int(layer)}.{name}.weight", slice(None)).reshape(1, -1)

    def qk_norm(self, layer, which):
        """Q/K head RMSNorm weight [1,head_dim]; K exists on owner layers only."""
        if which not in ("q_norm", "k_norm"):
            raise KeyError(which)
        return self._slice(
            f"layers.{int(layer)}.self_attn.{which}.weight", slice(None)).reshape(1, -1)

    def layer_scalar(self, layer):
        return self._slice(f"layers.{int(layer)}.layer_scalar", slice(None)).reshape(1, 1)

    def final_norm(self):
        return self._slice("norm.weight", slice(None)).reshape(1, -1)

    def per_layer_model_projection(self):
        """B[D, num_layers*ple_dim] for the PLE context projection."""
        matrix_nk = self._slice(
            "per_layer_model_projection.weight", (slice(None), slice(None)))
        return np.ascontiguousarray(matrix_nk.T)

    def per_layer_projection_norm(self):
        return self._slice("per_layer_projection_norm.weight", slice(None)).reshape(1, -1)

    def lm_head_packed(self, panel=64):
        """Tied LM head as consecutive B[K,panel] vocabulary column panels."""
        panel = int(panel)
        vocab = self.spec.vocab_size
        hidden = self.spec.hidden_size
        if panel < 1 or vocab % panel:
            raise ValueError(f"vocab {vocab} must be divisible by panel {panel}")
        matrix_vk = self._slice("embed_tokens.weight", (slice(None), slice(None)))
        panels = matrix_vk.reshape(vocab // panel, panel, hidden).transpose(0, 2, 1)
        return np.ascontiguousarray(panels).reshape(-1)

    def ple_table_path(self):
        """Precomputed per-layer-input table next to the real checkpoint file."""
        return self.shard.resolve().parent / "ple_table_f16.bin"

    def ple_table(self):
        """Memory-mapped [vocab, num_layers*ple_dim] FP16 per-layer-input table."""
        path = self.ple_table_path()
        width = self.spec.num_layers * self.spec.layers[0].ple_dim
        expected = self.spec.vocab_size * width * 2
        if not path.exists() or path.stat().st_size != expected:
            raise ModelAssetError(
                f"PLE table missing or wrong size at {path}; "
                "run d_compiler/make_gemma4_ple_table.py")
        return np.memmap(path, dtype=np.float16, mode="r",
                         shape=(self.spec.vocab_size, width))

    def validate_keyset(self):
        """Presence of every required text key; absence of k/v on shared layers."""
        keys = self.keys()
        required = {
            PREFIX + "embed_tokens.weight",
            PREFIX + "embed_tokens_per_layer.weight",
            PREFIX + "per_layer_model_projection.weight",
            PREFIX + "per_layer_projection_norm.weight",
            PREFIX + "norm.weight",
        }
        for spec in self.spec.layers:
            base = f"{PREFIX}layers.{spec.index}."
            required.update(base + name + ".weight" for name in LAYER_NORMS)
            required.add(base + "layer_scalar")
            required.add(base + "self_attn.q_proj.weight")
            required.add(base + "self_attn.q_norm.weight")
            required.add(base + "self_attn.o_proj.weight")
            for name in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                         "per_layer_input_gate", "per_layer_projection"):
                required.add(base + name + ".weight")
            # Shared layers also carry k/v tensors in the checkpoint file, but
            # the official model ignores them (_keys_to_ignore_on_load_unexpected);
            # they are required only where the graph actually consumes them.
            if spec.owns_cache:
                for name in ("self_attn.k_proj.weight", "self_attn.k_norm.weight",
                             "self_attn.v_proj.weight"):
                    required.add(base + name)
        missing = sorted(required - keys)
        if missing:
            raise ModelAssetError(f"checkpoint keyset is incomplete: {missing[:8]}")
        return len(required)
