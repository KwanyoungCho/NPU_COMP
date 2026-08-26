"""Official Gemma 4 E2B text prefill and decode on the extended source C-model.

Program reuse follows the layer shape classes, not layer indices: all 35
layers compile to four prefill programs (sliding/full x owner/shared —
double-wide FFN widths are part of the class).  KV caches live per CachePlan
slot; shared layers read their owner's slot, and during decode the owner's
freshly appended K/V is visible to later shared layers in the same step.

Greedy generation elides the monotonic final logit softcap (argmax-safe);
logits comparisons against HF apply the softcap on the comparison side.
"""
from __future__ import annotations

import time

import numpy as np

from . import driver
from .generation import SourceGenerationSession, SourcePrefillResult
from .gemma4_graph import (
    build_gemma4_decode_layer_module,
    build_gemma4_final_norm_module,
    build_gemma4_prefill_layer_module,
)
from .gemma4_model import Gemma4Assets, ModelAssetError
from .gemma4_ple import compute_rows
from .model_spec import build_cache_plan


class Gemma4SourceCompiler(SourceGenerationSession):
    """Compile per-class layer programs and run official Gemma 4 E2B text."""

    def __init__(self, sequence, assets=None, *, cache_weights=True,
                 backend="source-0818", quantize=None):
        self.assets = assets or Gemma4Assets()
        super().__init__(self.assets.spec, sequence, backend=backend, quantize=quantize)
        self.cache_plan = build_cache_plan(self.spec)
        self.cache_weights = bool(cache_weights)
        self._weights = {}
        self._lm_weight = None
        self._ple_table = None
        self._ple_weights = None
        self.prefill_programs = {}
        self.decode_programs = {}
        self.norm_program = driver.compile_module(
            build_gemma4_final_norm_module(self.spec, 1),
            backend=self.backend, reuse=True,
            quant_int8=self.quant_int8_names,
            quant_act=self.quant_act)
        self.lm_gemm = self.make_lm_gemm(self.spec.hidden_size, self.spec.vocab_size)

    def _layer_class(self, layer_index):
        layer = self.spec.layers[layer_index]
        return (layer.attention.kind, layer.owns_cache, layer.ffn_hidden)

    def _prefill_program(self, layer_index):
        key = self._layer_class(layer_index)
        if key not in self.prefill_programs:
            self.prefill_programs[key] = driver.compile_module(
                build_gemma4_prefill_layer_module(
                    self.spec, layer_index, self.sequence),
                backend=self.backend, reuse=True,
            quant_int8=self.quant_int8_names,
            quant_act=self.quant_act)
        return self.prefill_programs[key]

    def _decode_program(self, layer_index, context):
        key = self._layer_class(layer_index) + (int(context),)
        if key not in self.decode_programs:
            self.decode_programs[key] = driver.compile_module(
                build_gemma4_decode_layer_module(
                    self.spec, layer_index, context),
                backend=self.backend, reuse=True,
            quant_int8=self.quant_int8_names,
            quant_act=self.quant_act)
        return self.decode_programs[key]

    def compile_stats(self):
        def table(programs):
            return {"/".join(str(part) for part in key): self._program_stats(item)
                    for key, item in sorted(programs.items(), key=lambda kv: str(kv[0]))}
        return {
            "prefill": table(self.prefill_programs),
            "decode": table(self.decode_programs),
            "final_norm": self._program_stats(self.norm_program),
            "lm_head": {
                "program_words": len(self.lm_gemm),
                "gbuffer_entries": self.lm_gemm.gbuffer_entries,
                "rhs_layout": "Kx64 column panels",
            },
        }

    def _cached(self, key, loader):
        if key in self._weights:
            return self._weights[key]
        value = loader()
        if self.cache_weights:
            self._weights[key] = value
        return value

    def _ple_rows(self, token_ids):
        """Precomputed table rows; falls back to the proven-equivalent
        FP16-step computation while the offline table is still generating."""
        if self._ple_table is None:
            try:
                self._ple_table = self.assets.ple_table()
            except ModelAssetError:
                self._ple_table = False
        ids = [int(value) for value in np.asarray(token_ids).reshape(-1)]
        if self._ple_table is not False:
            return np.stack([self._ple_table[index] for index in ids])
        if self._ple_weights is None:
            self._ple_weights = (self.assets.per_layer_model_projection(),
                                 self.assets.per_layer_projection_norm())
        projection, norm_weight = self._ple_weights
        return compute_rows(self.spec, self.assets.embedding(ids),
                            self.assets.ple_rows(ids), projection, norm_weight)

    def _layer_inputs(self, layer_index, hidden, pli):
        layer = self.spec.layers[layer_index]
        cached = self._cached
        inputs = {
            "x": hidden,
            "pli": pli,
            "Wn1": cached((layer_index, "n1"), lambda: self.assets.norm(layer_index, "input_layernorm")),
            "Wn2": cached((layer_index, "n2"), lambda: self.assets.norm(layer_index, "post_attention_layernorm")),
            "Wn3": cached((layer_index, "n3"), lambda: self.assets.norm(layer_index, "pre_feedforward_layernorm")),
            "Wn4": cached((layer_index, "n4"), lambda: self.assets.norm(layer_index, "post_feedforward_layernorm")),
            "Wn5": cached((layer_index, "n5"), lambda: self.assets.norm(layer_index, "post_per_layer_input_norm")),
            "Wq": cached((layer_index, "q"), lambda: self.assets.linear(layer_index, "self_attn.q_proj")),
            "Wqn": cached((layer_index, "qn"), lambda: self.assets.qk_norm(layer_index, "q_norm")),
            "Wo": cached((layer_index, "o"), lambda: self.assets.linear(layer_index, "self_attn.o_proj")),
            "Wg": cached((layer_index, "g"), lambda: self.assets.linear(layer_index, "mlp.gate_proj")),
            "Wu": cached((layer_index, "u"), lambda: self.assets.linear(layer_index, "mlp.up_proj")),
            "Wd": cached((layer_index, "d"), lambda: self.assets.linear(layer_index, "mlp.down_proj")),
            "Wpg": cached((layer_index, "pg"), lambda: self.assets.linear(layer_index, "per_layer_input_gate")),
            "Wpp": cached((layer_index, "pp"), lambda: self.assets.linear(layer_index, "per_layer_projection")),
            "ls": cached((layer_index, "ls"), lambda: self.assets.layer_scalar(layer_index)),
        }
        if layer.owns_cache:
            inputs["Wk"] = cached((layer_index, "k"), lambda: self.assets.linear(layer_index, "self_attn.k_proj"))
            inputs["Wkn"] = cached((layer_index, "kn"), lambda: self.assets.qk_norm(layer_index, "k_norm"))
            inputs["Wv"] = cached((layer_index, "v"), lambda: self.assets.linear(layer_index, "self_attn.v_proj"))
        return inputs

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
        started = time.perf_counter()
        hidden = self.assets.embedding(input_ids)
        pli_all = self._ple_rows(input_ids)
        ple_dim = self.spec.layers[0].ple_dim
        D = self.spec.hidden_size
        slot_keys = [None] * len(self.cache_plan.slots)
        slot_values = [None] * len(self.cache_plan.slots)
        for layer in range(self.spec.num_layers):
            before = time.perf_counter()
            slot_index = self.cache_plan.layer_to_slot[layer]
            slot = self.cache_plan.slots[slot_index]
            pli = np.ascontiguousarray(
                pli_all[:, layer * ple_dim:(layer + 1) * ple_dim])
            inputs = self._layer_inputs(layer, hidden, pli)
            if self.spec.layers[layer].owns_cache:
                output = self._run(self._prefill_program(layer), inputs)
                head_dim = slot.head_dim
                hidden = output[:, :D]
                key = output[:, D:D + head_dim]
                value = output[:, D + head_dim:D + 2 * head_dim]
                slot_keys[slot_index] = np.ascontiguousarray(key.T, dtype=np.float16)
                slot_values[slot_index] = np.ascontiguousarray(value, dtype=np.float16)
            else:
                inputs["Kt"] = slot_keys[slot_index]
                inputs["Vc"] = slot_values[slot_index]
                hidden = self._run(self._prefill_program(layer), inputs)
            if progress is not None:
                progress("prefill", layer, hidden, time.perf_counter() - before)
        normalized, logits = self._logits(hidden[-1:])
        token = int(np.argmax(logits.astype(np.float32)))
        stats = self.stats()
        stats["wall_seconds"] = time.perf_counter() - started
        keys = [[array] for array in slot_keys]
        values = [[array] for array in slot_values]
        return SourcePrefillResult(
            input_ids, hidden, keys, values, normalized, logits, token, stats)

    def decode_token(self, token_id, position, keys, values, *, progress=None):
        """One token through all layers; each owner appends its K/V to the
        slot cache immediately so later shared layers see the full context."""
        hidden = self.assets.embedding([int(token_id)])
        pli_all = self._ple_rows([int(token_id)])
        ple_dim = self.spec.layers[0].ple_dim
        D = self.spec.hidden_size
        context = int(position) + 1
        appended = set()
        for layer in range(self.spec.num_layers):
            before = time.perf_counter()
            slot_index = self.cache_plan.layer_to_slot[layer]
            slot = self.cache_plan.slots[slot_index]
            pli = np.ascontiguousarray(
                pli_all[:, layer * ple_dim:(layer + 1) * ple_dim])
            inputs = self._layer_inputs(layer, hidden, pli)
            inputs["pos"] = np.asarray([[position]], dtype=np.float16)
            inputs["Kt"] = keys[slot_index][0]
            inputs["Vc"] = values[slot_index][0]
            output = self._run(self._decode_program(layer, context), inputs)
            if self.spec.layers[layer].owns_cache:
                head_dim = slot.head_dim
                hidden = output[:, :D]
                key = output[:, D:D + head_dim]
                value = output[:, D + head_dim:D + 2 * head_dim]
                assert slot_index not in appended
                appended.add(slot_index)
                keys[slot_index][0] = np.ascontiguousarray(
                    np.concatenate((keys[slot_index][0], key.T), axis=1),
                    dtype=np.float16)
                values[slot_index][0] = np.ascontiguousarray(
                    np.concatenate((values[slot_index][0], value), axis=0),
                    dtype=np.float16)
            else:
                hidden = output
            if progress is not None:
                progress("decode", layer, hidden, time.perf_counter() - before)
        _, logits = self._logits(hidden)
        return hidden, logits, int(np.argmax(logits.astype(np.float32)))

    def stats(self):
        return {
            "invocations": self.invocations,
            "elapsed_seconds": self.elapsed_seconds,
            "cached_weight_tensors": len(self._weights),
            # identity checks: the loaded table is a numpy memmap, and `in`
            # would compare it element-wise (ambiguous-truth ValueError).
            "ple_source": "computed" if (self._ple_table is None or
                                         self._ple_table is False) else "table",
            "compile": self.compile_stats(),
        }
