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

    def __init__(self, spec, sequence):
        self.spec = spec
        self.sequence = int(sequence)
        if self.sequence < 1:
            raise ValueError("prefill sequence must be positive")
        self.invocations = 0
        self.elapsed_seconds = 0.0

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
