"""Model-independent source-0818 generation session (Stage G4.1).

Owns everything a model family does not: program execution with timing and
invocation counting, program size stats, and the greedy prefill+decode loop.
A family session (e.g. ``Llama32SourceCompiler``) subclasses this and provides
``prefill``, ``decode_token``, and ``stats`` on top of its own graph builders,
weight binding, and cache layout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import driver


@dataclass
class SourcePrefillResult:
    input_ids: np.ndarray
    hidden: np.ndarray
    keys: list
    values: list
    normalized: np.ndarray
    logits: np.ndarray
    next_token_id: int
    stats: dict


@dataclass
class SourceGenerationResult:
    input_ids: np.ndarray
    generated_ids: np.ndarray
    hidden: np.ndarray
    keys: list
    values: list
    logits: np.ndarray
    stats: dict


class SourceGenerationSession:
    """Spec-driven execution bookkeeping shared by all model families."""

    BACKENDS = ("source-0818", "v09")
    QUANT_MODES = (None, "w8a16", "w8a8")

    def __init__(self, spec, sequence, backend="source-0818", quantize=None):
        self.spec = spec
        self.sequence = int(sequence)
        if self.sequence < 1:
            raise ValueError("prefill sequence must be positive")
        if backend not in self.BACKENDS:
            raise ValueError(f"unsupported generation backend {backend!r}")
        if quantize not in self.QUANT_MODES:
            raise ValueError(f"unsupported quantization mode {quantize!r}")
        if quantize and backend != "v09":
            raise ValueError("weight quantization requires the v09 backend")
        self.backend = backend
        self.quantize = quantize
        self.invocations = 0
        self.elapsed_seconds = 0.0

    @property
    def quant_int8_names(self):
        """Projection-weight param names to quantize (None in FP16 mode)."""
        if self.quantize in ("w8a16", "w8a8"):
            from .quantize import QUANT_PARAM_NAMES
            return QUANT_PARAM_NAMES
        return None

    @property
    def quant_act(self):
        """True when activations are quantized too (W8A8)."""
        return self.quantize == "w8a8"

    def make_lm_gemm(self, inner, columns):
        """Wide-vocab panel GEMM for the session's backend (bit-identical
        K-tile accumulation order on both targets)."""
        if self.backend == "v09":
            from .source_gemm_v09 import V09PackedRhsGemm
            return V09PackedRhsGemm(inner, columns)
        from .source_gemm_0818 import PackedRhsGemm
        return PackedRhsGemm(inner, columns)

    @staticmethod
    def _program_stats(compiled):
        asm, memory = compiled
        return {"program_words": len(asm.words), "gbuffer_entries": memory.top}

    def _run(self, compiled, inputs):
        started = time.perf_counter()
        output = driver.run_compiled(*compiled, inputs)
        self.elapsed_seconds += time.perf_counter() - started
        self.invocations += 1
        return np.asarray(output, dtype=np.float16)

    def generate(self, input_ids, count, *, progress=None):
        count = int(count)
        if count < 1:
            raise ValueError("generation count must be positive")
        started = time.perf_counter()
        prefill = self.prefill(input_ids, progress=progress)
        generated = [prefill.next_token_id]
        hidden = prefill.hidden[-1:]
        logits = prefill.logits
        keys, values = prefill.keys, prefill.values
        while len(generated) < count:
            position = self.sequence + len(generated) - 1
            hidden, logits, next_id = self.decode_token(
                generated[-1], position, keys, values, progress=progress)
            generated.append(next_id)
        stats = self.stats()
        stats["wall_seconds"] = time.perf_counter() - started
        return SourceGenerationResult(
            np.asarray(input_ids, dtype=np.int64),
            np.asarray(generated, dtype=np.int64),
            hidden, keys, values, logits, stats)
