"""Official Llama 3.2 3B prefill and decode on the extended source C-model."""
from __future__ import annotations

import time

import numpy as np

from . import driver, model
from .generation import SourceGenerationSession, SourcePrefillResult
from .model_spec import build_cache_plan, llama32_spec
from .v3_model import Llama32Assets
from .source_gemm_0818 import PackedRhsGemm


class Llama32SourceCompiler(SourceGenerationSession):
    """Compile static-shape layer programs and run them on the source C-model.

    A complete layer is one program and one expanded G-buffer invocation.  Decode
    uses one K/V projection program plus one attention/FFN program per layer; the
    host performs only the dynamic cache append between those static programs.
    """

    MODULE_SHAPES = {
        "self_attn.q_proj": lambda c: (c.D, c.D),
        "self_attn.k_proj": lambda c: (c.D, c.KV * c.HD),
        "self_attn.v_proj": lambda c: (c.D, c.KV * c.HD),
        "self_attn.o_proj": lambda c: (c.D, c.D),
        "mlp.gate_proj": lambda c: (c.D, c.F),
        "mlp.up_proj": lambda c: (c.D, c.F),
        "mlp.down_proj": lambda c: (c.F, c.D),
    }

    def __init__(self, sequence, assets=None, *, cache_weights=True):
        self.assets = assets or Llama32Assets()
        super().__init__(llama32_spec(self.assets.config), sequence)
        self.cfg = model.LLAMA_3_2_3B
        self.cache_plan = build_cache_plan(self.spec)
        self.cache_weights = bool(cache_weights)
        self._weights = {}
        self._lm_weight = None
        self.prefill_program = driver.compile_module(
            model.build_v3_prefill_layer_module(
                self.cfg, self.sequence, return_cache=True),
            backend="source-0818", reuse=True)
        self.kv_program = driver.compile_module(
            model.build_v3_decode_kv_module(self.cfg),
            backend="source-0818", reuse=True)
        self.norm_program = driver.compile_module(
            model.build_v3_final_norm_module(self.cfg, 1),
            backend="source-0818", reuse=True)
        self.lm_gemm = PackedRhsGemm(self.cfg.D, self.spec.vocab_size)
        self.decode_programs = {}

    def compile_stats(self):
        return {
            "prefill": self._program_stats(self.prefill_program),
            "decode_kv": self._program_stats(self.kv_program),
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

    def _weight(self, layer, module):
        key = (int(layer), module)
        if key in self._weights:
            return self._weights[key]
        value = self.assets.linear(layer, module, self.MODULE_SHAPES[module](self.cfg))
        if self.cache_weights:
            self._weights[key] = value
        return value

    def _norm(self, layer, post_attention=False):
        suffix = "post" if post_attention else "input"
        key = (int(layer), f"norm.{suffix}")
        if key in self._weights:
            return self._weights[key]
        value = self.assets.norm(layer, post_attention=post_attention).reshape(1, self.cfg.D)
        if self.cache_weights:
            self._weights[key] = value
        return value

    def _prefill_inputs(self, layer, hidden):
        return {
            "x": hidden,
            "Wn1": self._norm(layer),
            "Wn2": self._norm(layer, post_attention=True),
            "Wq": self._weight(layer, "self_attn.q_proj"),
            "Wk": self._weight(layer, "self_attn.k_proj"),
            "Wv": self._weight(layer, "self_attn.v_proj"),
            "Wo": self._weight(layer, "self_attn.o_proj"),
            "Wg": self._weight(layer, "mlp.gate_proj"),
            "Wu": self._weight(layer, "mlp.up_proj"),
            "Wd": self._weight(layer, "mlp.down_proj"),
        }

    def _decode_program(self, context):
        context = int(context)
        if context not in self.decode_programs:
            self.decode_programs[context] = driver.compile_module(
                model.build_v3_decode_layer_module(self.cfg, context),
                backend="source-0818", reuse=True)
        return self.decode_programs[context]

    def _decode_kv_inputs(self, layer, hidden, position):
        return {
            "x": hidden,
            "Wn1": self._norm(layer),
            "Wk": self._weight(layer, "self_attn.k_proj"),
            "Wv": self._weight(layer, "self_attn.v_proj"),
            "pos": np.asarray([[position]], dtype=np.float16),
        }

    def _decode_inputs(self, layer, hidden, position, keys, values):
        inputs = {
            "x": hidden,
            "Wn1": self._norm(layer),
            "Wn2": self._norm(layer, post_attention=True),
            "Wq": self._weight(layer, "self_attn.q_proj"),
            "Wo": self._weight(layer, "self_attn.o_proj"),
            "pos": np.asarray([[position]], dtype=np.float16),
            "Wg": self._weight(layer, "mlp.gate_proj"),
            "Wu": self._weight(layer, "mlp.up_proj"),
            "Wd": self._weight(layer, "mlp.down_proj"),
        }
        slot = self.cache_plan.slot_for(layer)
        for kv in range(slot.num_kv_heads):
            inputs[f"Kt{kv}"] = keys[kv]
            inputs[f"Vc{kv}"] = values[kv]
        return inputs

    def _logits(self, hidden):
        normalized = self._run(self.norm_program, {
            "x": hidden,
            "weight": self.assets.final_norm().reshape(1, self.cfg.D),
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
        layer_keys, layer_values = [], []
        started = time.perf_counter()
        for layer in range(self.spec.num_layers):
            before = time.perf_counter()
            output = self._run(self.prefill_program, self._prefill_inputs(layer, hidden))
            slot = self.cache_plan.slot_for(layer)
            head_dim = slot.head_dim
            width = 2 * slot.num_kv_heads * head_dim
            hidden = output[:, :self.cfg.D]
            cache = output[:, self.cfg.D:self.cfg.D + width]
            keys, values = [], []
            for kv in range(slot.num_kv_heads):
                base = 2 * kv * head_dim
                keys.append(np.ascontiguousarray(
                    cache[:, base:base + head_dim].T, dtype=np.float16))
                values.append(np.ascontiguousarray(
                    cache[:, base + head_dim:base + 2 * head_dim],
                    dtype=np.float16))
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
        """Consume one token, append every layer cache, and produce next logits."""
        hidden = self.assets.embedding([int(token_id)])
        context = int(position) + 1
        decode_program = self._decode_program(context)
        for layer in range(self.spec.num_layers):
            before = time.perf_counter()
            projected = self._run(
                self.kv_program, self._decode_kv_inputs(layer, hidden, position))
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
            hidden = self._run(decode_program, self._decode_inputs(
                layer, hidden, position, keys[layer], values[layer]))
            if progress is not None:
                progress("decode", layer, hidden, time.perf_counter() - before)
        _, logits = self._logits(hidden)
        return hidden, logits, int(np.argmax(logits.astype(np.float32)))

    def stats(self):
        return {
            "invocations": self.invocations,
            "elapsed_seconds": self.elapsed_seconds,
            "cached_weight_tensors": len(self._weights),
            "compile": self.compile_stats(),
        }
