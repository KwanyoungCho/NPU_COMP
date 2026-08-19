"""Qwen3-4B asset loader regression (skips when the checkpoint is absent)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.qwen3_model import Qwen3Assets, default_model_path


def test_qwen3_assets():
    path = default_model_path()
    if not (path / "model.safetensors.index.json").exists():
        print("  (skip: Qwen3-4B checkpoint not present)")
        return
    assets = Qwen3Assets(path)
    count = assets.validate_keyset()
    print(f"  keyset ok ({count} required keys)")
    assert assets.spec.num_layers == 36
    assert assets.spec.tie_word_embeddings
    assert assets.module_shape(0, "self_attn.q_proj") == (2560, 4096)
    assert assets.module_shape(0, "self_attn.k_proj") == (2560, 1024)
    assert assets.module_shape(0, "mlp.gate_proj") == (2560, 9728)
    weight = assets.linear(0, "self_attn.k_proj")
    assert weight.shape == (2560, 1024) and weight.dtype == np.float16
    assert np.isfinite(weight.astype(np.float32)).all()
    assert assets.qk_norm(0, "q_norm").shape == (1, 128)
    assert assets.qk_norm(0, "k_norm").shape == (1, 128)
    assert assets.norm(0, "input_layernorm").shape == (1, 2560)
    assert assets.final_norm().shape == (1, 2560)
    assert assets.embedding([9707]).shape == (1, 2560)
    ids = assets.tokenizer("Hello, NPU compiler!", return_tensors="np")["input_ids"][0]
    print(f"  prompt ids: {ids.tolist()}")
    assert ids.size >= 4


if __name__ == "__main__":
    test_qwen3_assets()
    print("ALL QWEN3 ASSET TESTS PASSED")
