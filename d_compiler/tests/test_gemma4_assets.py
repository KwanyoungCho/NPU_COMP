"""Gemma 4 E2B asset loader regression (skips when the checkpoint is absent)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.gemma4_model import Gemma4Assets, default_model_path

GOLDEN_IDS = [2, 9259, 236764, 646, 11152, 47133, 236888]


def test_gemma4_assets():
    path = default_model_path()
    if not (path / "model.safetensors").exists():
        print("  (skip: Gemma 4 E2B checkpoint not present)")
        return
    assets = Gemma4Assets(path)

    count = assets.validate_keyset()
    print(f"  keyset ok ({count} required text keys)")

    assert assets.spec.num_layers == 35

    # Per-layer geometry: sliding owner / full owner / shared double-wide.
    assert assets.module_shape(0, "self_attn.q_proj") == (1536, 2048)
    assert assets.module_shape(0, "self_attn.k_proj") == (1536, 256)
    assert assets.module_shape(0, "mlp.gate_proj") == (1536, 6144)
    assert assets.module_shape(4, "self_attn.q_proj") == (1536, 4096)
    assert assets.module_shape(4, "self_attn.v_proj") == (1536, 512)
    assert assets.module_shape(15, "mlp.gate_proj") == (1536, 12288)
    assert assets.module_shape(19, "self_attn.o_proj") == (4096, 1536)
    try:
        assets.module_shape(15, "self_attn.k_proj")
        raise AssertionError("shared layer must not expose k_proj")
    except KeyError:
        pass

    weight = assets.linear(0, "self_attn.k_proj")
    assert weight.shape == (1536, 256) and weight.dtype == np.float16
    assert np.isfinite(weight.astype(np.float32)).all()

    assert assets.qk_norm(0, "q_norm").shape == (1, 256)
    assert assets.qk_norm(4, "q_norm").shape == (1, 512)
    assert assets.norm(0, "post_per_layer_input_norm").shape == (1, 1536)
    assert assets.layer_scalar(0).shape == (1, 1)
    assert assets.final_norm().shape == (1, 1536)
    assert assets.per_layer_projection_norm().shape == (1, 256)
    assert assets.per_layer_model_projection().shape == (1536, 35 * 256)

    # Embedding scale is FP16 multiply on raw rows; the raw table stays tied to
    # the LM head.
    raw = assets.embedding([9259], scaled=False)
    scaled = assets.embedding([9259])
    assert np.array_equal(
        scaled, (raw * np.float16(np.sqrt(1536.0))).astype(np.float16))
    assert assets.ple_rows([9259]).shape == (1, 35 * 256)

    ids = assets.tokenizer("Hello, NPU compiler!", return_tensors="np")["input_ids"][0]
    assert ids.tolist() == GOLDEN_IDS


if __name__ == "__main__":
    test_gemma4_assets()
    print("ALL GEMMA4 ASSET TESTS PASSED")
