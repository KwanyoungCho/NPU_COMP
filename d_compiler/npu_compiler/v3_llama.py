"""Official Llama 3.2 3B prefill orchestration for compiler V3."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import model
from .v3_executor import compile_module
from .v3_model import Llama32Assets, MODEL_REVISION
from .v3_runtime import VendorSession


@dataclass
class PrefillResult:
    input_ids: np.ndarray
    hidden: np.ndarray
    normalized: np.ndarray
    logits: np.ndarray
    next_token_id: int
    stats: dict


class Llama32PrefillCompiler:
    """Compile once per prompt length and stream all official layer weights."""

    def __init__(self, sequence, assets=None):
        self.assets = assets or Llama32Assets()
        self.sequence = int(sequence)
        if self.sequence < 1:
            raise ValueError("prefill sequence must be positive")
        self.cfg = model.LLAMA_3_2_3B
        self.layer_plan = compile_module(
            model.build_v3_prefill_layer_module(self.cfg, self.sequence))
        self.norm_plan = compile_module(
            model.build_v3_final_norm_module(self.cfg, self.sequence))
        self.lm_plan = compile_module(model.build_v3_lm_head_module(self.cfg))

    def layer_inputs(self, layer, hidden):
        D, KV, HD, F = self.cfg.D, self.cfg.KV, self.cfg.HD, self.cfg.F
        source = self.assets.linear_source
        return {
            "x": hidden,
            "Wn1": self.assets.norm(layer, post_attention=False).reshape(1, D),
            "Wn2": self.assets.norm(layer, post_attention=True).reshape(1, D),
            "Wq": source(layer, "self_attn.q_proj", (D, D)),
            "Wk": source(layer, "self_attn.k_proj", (D, KV * HD)),
            "Wv": source(layer, "self_attn.v_proj", (D, KV * HD)),
            "Wo": source(layer, "self_attn.o_proj", (D, D)),
            "Wg": source(layer, "mlp.gate_proj", (D, F)),
            "Wu": source(layer, "mlp.up_proj", (D, F)),
            "Wd": source(layer, "mlp.down_proj", (F, D)),
        }

    def run_layer(self, hidden, layer, vendor):
        return self.layer_plan.run(self.layer_inputs(layer, hidden), vendor=vendor)

    @staticmethod
    def _write_state(directory, layer, input_ids, hidden, stats):
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"hidden_after_layer_{layer:02d}.npy", hidden)
        metadata = {
            "model_revision": MODEL_REVISION,
            "input_ids": [int(value) for value in input_ids],
            "layers_completed": layer,
            "stats": stats,
        }
        (directory / "state.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def run(self, input_ids, *, vendor=None, start_layer=0, hidden=None,
            checkpoint_dir=None, progress=None):
        input_ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        if input_ids.size != self.sequence:
            raise ValueError(f"plan sequence {self.sequence}, input has {input_ids.size} tokens")
        if hidden is None:
            if start_layer:
                raise ValueError("resuming after layer 0 requires hidden")
            hidden = self.assets.embedding(input_ids)
        else:
            hidden = np.asarray(hidden, dtype=np.float16)
        if hidden.shape != (self.sequence, self.cfg.D):
            raise ValueError(f"hidden shape {hidden.shape} != {(self.sequence, self.cfg.D)}")
        checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else None
        owns_vendor = vendor is None
        if owns_vendor:
            vendor = VendorSession()
        started = time.perf_counter()
        try:
            for layer in range(start_layer, self.assets.config["num_hidden_layers"]):
                before = vendor.stats().copy()
                hidden = self.run_layer(hidden, layer, vendor)
                after = vendor.stats().copy()
                delta = {
                    "invocations": after["invocations"] - before["invocations"],
                    "elapsed_seconds": after["elapsed_seconds"] - before["elapsed_seconds"],
                }
                if checkpoint is not None:
                    self._write_state(checkpoint, layer + 1, input_ids, hidden, after)
                if progress is not None:
                    progress(layer, hidden, delta)
            normalized = self.norm_plan.run({
                "x": hidden,
                "weight": self.assets.final_norm().reshape(1, self.cfg.D),
            }, vendor=vendor)
            logits = self.lm_plan.run({
                "x": normalized[-1:],
                "weight": self.assets.lm_head_source(),
            }, vendor=vendor).reshape(-1)
            token = int(np.argmax(logits.astype(np.float32)))
            stats = vendor.stats().copy()
            stats["wall_seconds"] = time.perf_counter() - started
            return PrefillResult(input_ids, hidden, normalized, logits, token, stats)
        finally:
            if owns_vendor:
                vendor.close()
