"""Decode on the standard path: prefill_cache + per-context decode programs.

Two gates.  The float32 gate recomputes the whole sequence from scratch in
numpy at every step -- a KV cache is only an optimization, so cached decode
must produce the same greedy tokens as full recomputation.  The llvm gate
runs the identical programs on the CPU build per step, which pins the NPU
run against the same IR.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import npu_generate
from npu_compiler.nn_models import llama
from test_nn_frontend import _reference

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0)
SEQ, STEPS = 5, 3


class _Assets:
    """Random embeddings and weights standing in for a checkpoint."""

    def __init__(self, config, seed=0):
        rng = np.random.default_rng(seed)
        self.table = rng.normal(0, 0.5, (config["vocab_size"],
                                         config["hidden_size"]))
        self.rng = rng
        self.weights = None

    def embedding(self, token_ids):
        return self.table[np.asarray(token_ids, np.int64)].astype(np.float16)


def _family(assets):
    """The tiny model as a family object the generator understands."""

    class Family:
        build_generate = staticmethod(llama.build_generate)
        rope_inputs = staticmethod(llama.rope_inputs)

        @staticmethod
        def load_params(_assets, params, cfg):
            if assets.weights is None:
                assets.weights = [
                    assets.rng.normal(0, 0.15, p.shape).astype(np.float16)
                    for _, p in params]
            return assets.weights

        @staticmethod
        def runtime_inputs(_assets, cfg, tokens):
            seq = len(tokens)
            return {"input_embeds": assets.embedding(tokens),
                    "mask": llama.causal_mask(cfg.num_heads, seq)}

    return Family()


def _greedy_reference(config, assets, weights_by_name, prompt, steps):
    """Full recomputation per step, float32, no cache."""
    tokens = [int(t) for t in prompt]
    generated = []
    for _ in range(steps):
        seq = len(tokens)
        x = assets.embedding(tokens)
        cos, sin = llama.rope_inputs(config, np.arange(seq))
        mask = llama.causal_mask(config.num_heads, seq)
        logits = _reference(config, weights_by_name, x, cos, sin, mask)
        generated.append(int(np.argmax(logits[-1])))
        tokens.append(generated[-1])
    return generated


def test_cached_decode_matches_full_recomputation():
    assets = _Assets(TINY)
    family = _family(assets)

    mod, params, cfg = llama.build_generate(TINY, SEQ, SEQ + 1)
    family.load_params(assets, params, cfg)          # fix the weights
    by_name = {name: value
               for (name, _), value in zip(params, assets.weights)}
    prompt = [3, 1, 4, 1, 5]
    expected = _greedy_reference(cfg, assets, by_name, prompt, STEPS)

    on_llvm = npu_generate.generate(family, TINY, assets, prompt, STEPS,
                                    runner=npu_generate.llvm_runner)
    assert on_llvm == expected, (on_llvm, expected)
    print(f"  [PASS] llvm generation matches float32 recomputation: {expected}")

    on_npu = npu_generate.generate(family, TINY, assets, prompt, STEPS)
    assert on_npu == expected, (on_npu, expected)
    print(f"  [PASS] NPU generation matches: {on_npu}")


if __name__ == "__main__":
    test_cached_decode_matches_full_recomputation()
    print("ALL DECODE TESTS PASSED")
