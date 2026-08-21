"""Official Qwen3-4B prefill and decode on the extended source C-model.

All 36 layers share one prefill program and one decode program per context
(shapes are identical across layers; only weight inputs differ).  Cache layout
matches the Llama compiler: keys[layer][kv] transposed [HD, ctx], values
[ctx, HD]; the fused decode program appends this token's K/V on-device and
the host extends the caches once per step.
"""
from __future__ import annotations

import time

import numpy as np

from . import driver
from .generation import SourceGenerationSession, SourcePrefillResult
from .model_spec import build_cache_plan
from .qwen3_graph import (
    build_qwen3_decode_layer_module,
    build_qwen3_final_norm_module,
    build_qwen3_prefill_layer_module,
)
from .qwen3_model import Qwen3Assets
from .source_gemm_0818 import PackedRhsGemm


class Qwen3SourceCompiler(SourceGenerationSession):
    """Compile static-shape layer programs and run official Qwen3-4B."""

    def __init__(self, sequence, assets=None, *, cache_weights=True):
        self.assets = assets or Qwen3Assets()
        super().__init__(self.assets.spec, sequence)
        self.cache_plan = build_cache_plan(self.spec)
        self.cache_weights = bool(cache_weights)
        self._weights = {}
        self._lm_weight = None
        self.prefill_program = driver.compile_module(
            build_qwen3_prefill_layer_module(self.spec, 0, self.sequence),
            backend="source-0818", reuse=True)
        self.norm_program = driver.compile_module(
            build_qwen3_final_norm_module(self.spec, 1),
            backend="source-0818", reuse=True)
        self.lm_gemm = PackedRhsGemm(self.spec.hidden_size, self.spec.vocab_size)
        self.decode_programs = {}

    def compile_stats(self):
        return {
            "prefill": self._program_stats(self.prefill_program),
            "final_norm": self._program_stats(self.norm_program),
            "lm_head": {
                "program_words": len(self.lm_gemm),
                "gbuffer_entries": self.lm_gemm.gbuffer_entries,
                "rhs_layout": "Kx64 column panels",
            },
            "decode": {
                str(context): self._program_stats(program)
                for context, program in sorted(self.decode_programs.items())
            },
        }

    def _decode_program(self, context):
        context = int(context)
        if context not in self.decode_programs:
            self.decode_programs[context] = driver.compile_module(
                build_qwen3_decode_layer_module(self.spec, 0, context),
                backend="source-0818", reuse=True)
        return self.decode_programs[context]

    def _cached(self, key, loader):
        if key in self._weights:
            return self._weights[key]
        value = loader()
        if self.cache_weights:
            self._weights[key] = value
        return value

    def _layer_inputs(self, layer, hidden):
        cached = self._cached
        return {
            "x": hidden,
            "Wn1": cached((layer, "n1"), lambda: self.assets.norm(layer, "input_layernorm")),
            "Wn2": cached((layer, "n2"), lambda: self.assets.norm(layer, "post_attention_layernorm")),
            "Wq": cached((layer, "q"), lambda: self.assets.linear(layer, "self_attn.q_proj")),
            "Wqn": cached((layer, "qn"), lambda: self.assets.qk_norm(layer, "q_norm")),
            "Wk": cached((layer, "k"), lambda: self.assets.linear(layer, "self_attn.k_proj")),
            "Wkn": cached((layer, "kn"), lambda: self.assets.qk_norm(layer, "k_norm")),
            "Wv": cached((layer, "v"), lambda: self.assets.linear(layer, "self_attn.v_proj")),
            "Wo": cached((layer, "o"), lambda: self.assets.linear(layer, "self_attn.o_proj")),
            "Wg": cached((layer, "g"), lambda: self.assets.linear(layer, "mlp.gate_proj")),
            "Wu": cached((layer, "u"), lambda: self.assets.linear(layer, "mlp.up_proj")),
            "Wd": cached((layer, "d"), lambda: self.assets.linear(layer, "mlp.down_proj")),
        }

    def _logits(self, hidden):
        normalized = self._run(self.norm_program, {
            "x": hidden,
            "weight": self._cached(("final", "norm"), self.assets.final_norm),
        })
        if self._lm_weight is None:
            self._lm_weight = self.assets.lm_head_packed(panel=self.lm_gemm.panel)
        started = time.perf_counter()
        logits = self.lm_gemm.run(normalized, self._lm_weight).reshape(-1)
        self.elapsed_seconds += time.perf_counter() - started
        self.invocations += 1
        return normalized, logits

    def prefill(self, input_ids, *, progress=None):
        input_ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        if input_ids.size != self.sequence:
            raise ValueError(f"plan sequence {self.sequence}, input has {input_ids.size}")
        hidden = self.assets.embedding(input_ids)
        D = self.spec.hidden_size
        layer_keys, layer_values = [], []
        started = time.perf_counter()
        for layer in range(self.spec.num_layers):
            before = time.perf_counter()
            output = self._run(self.prefill_program, self._layer_inputs(layer, hidden))
            slot = self.cache_plan.slot_for(layer)
            head_dim = slot.head_dim
            hidden = output[:, :D]
            cache = output[:, D:]
            keys, values = [], []
            for kv in range(slot.num_kv_heads):
                base = 2 * kv * head_dim
                keys.append(np.ascontiguousarray(
                    cache[:, base:base + head_dim].T, dtype=np.float16))
                values.append(np.ascontiguousarray(
                    cache[:, base + head_dim:base + 2 * head_dim], dtype=np.float16))
            layer_keys.append(keys)
            layer_values.append(values)
            if progress is not None:
                progress("prefill", layer, hidden, time.perf_counter() - before)
        normalized, logits = self._logits(hidden[-1:])
        token = int(np.argmax(logits.astype(np.float32)))
        stats = self.stats()
        stats["wall_seconds"] = time.perf_counter() - started
        return SourcePrefillResult(
            input_ids, hidden, layer_keys, layer_values,
            normalized, logits, token, stats)

    def decode_token(self, token_id, position, keys, values, *, progress=None):
        """One token through all layers as fused programs; the host extends
        every layer's cache once per step."""
        hidden = self.assets.embedding([int(token_id)])
        D = self.spec.hidden_size
        context = int(position) + 1
        decode_program = self._decode_program(context)
        projections = []
        for layer in range(self.spec.num_layers):
            before = time.perf_counter()
            inputs = self._layer_inputs(layer, hidden)
            inputs["pos"] = np.asarray([[position]], dtype=np.float16)
            slot = self.cache_plan.slot_for(layer)
            for kv in range(slot.num_kv_heads):
                inputs[f"Kt{kv}"] = keys[layer][kv]
                inputs[f"Vc{kv}"] = values[layer][kv]
            output = self._run(decode_program, inputs)
            hidden = output[:, :D]
            projections.append(output[:, D:])
            if progress is not None:
                progress("decode", layer, hidden, time.perf_counter() - before)
        for layer, projected in enumerate(projections):
            slot = self.cache_plan.slot_for(layer)
            head_dim = slot.head_dim
            for kv in range(slot.num_kv_heads):
                base = 2 * kv * head_dim
                key = projected[:, base:base + head_dim]
                value = projected[:, base + head_dim:base + 2 * head_dim]
                keys[layer][kv] = np.ascontiguousarray(
                    np.concatenate((keys[layer][kv], key.T), axis=1), dtype=np.float16)
                values[layer][kv] = np.ascontiguousarray(
                    np.concatenate((values[layer][kv], value), axis=0), dtype=np.float16)
        _, logits = self._logits(hidden)
        return hidden, logits, int(np.argmax(logits.astype(np.float32)))

    def stats(self):
        return {
            "invocations": self.invocations,
            "elapsed_seconds": self.elapsed_seconds,
            "cached_weight_tensors": len(self._weights),
            "compile": self.compile_stats(),
        }
