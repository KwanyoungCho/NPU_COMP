"""Official-weight Qwen3 layers vs HF intermediates (skips without assets)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler import driver
from npu_compiler.qwen3_graph import build_qwen3_prefill_layer_module
from npu_compiler.qwen3_model import Qwen3Assets, default_model_path

REFERENCE = os.path.join(ROOT, "d_compiler", "build",
                         "qwen3_layer_reference_hello.npz")


def metrics(actual, expected):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    delta = actual - expected
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "cosine": float(np.dot(actual, expected) /
                        (np.linalg.norm(actual) * np.linalg.norm(expected))),
    }


def test_qwen3_real_layers():
    path = default_model_path()
    if not (path / "model.safetensors.index.json").exists() \
            or not os.path.exists(REFERENCE):
        print("  (skip: checkpoint or layer reference not present)")
        return
    assets = Qwen3Assets(path)
    spec = assets.spec
    reference = np.load(REFERENCE)
    ids = reference["input_ids"]
    S = ids.size

    ours = assets.embedding(ids)
    embed = metrics(ours, reference["hidden_00"])
    assert embed["max_abs"] < 1e-3, embed

    module = build_qwen3_prefill_layer_module(spec, 0, S)
    compiled = driver.compile_module(module, backend="source-0818", reuse=True)
    D = spec.hidden_size
    for layer_index in (0, 18, 35):
        inputs = {
            "x": np.asarray(reference[f"hidden_{layer_index:02d}"], dtype=np.float16),
            "Wn1": assets.norm(layer_index, "input_layernorm"),
            "Wn2": assets.norm(layer_index, "post_attention_layernorm"),
            "Wq": assets.linear(layer_index, "self_attn.q_proj"),
            "Wqn": assets.qk_norm(layer_index, "q_norm"),
            "Wk": assets.linear(layer_index, "self_attn.k_proj"),
            "Wkn": assets.qk_norm(layer_index, "k_norm"),
            "Wv": assets.linear(layer_index, "self_attn.v_proj"),
            "Wo": assets.linear(layer_index, "self_attn.o_proj"),
            "Wg": assets.linear(layer_index, "mlp.gate_proj"),
            "Wu": assets.linear(layer_index, "mlp.up_proj"),
            "Wd": assets.linear(layer_index, "mlp.down_proj"),
        }
        output = np.asarray(driver.run_compiled(*compiled, inputs), dtype=np.float16)
        hidden = output[:, :D].astype(np.float64)
        if layer_index == spec.num_layers - 1:
            # HF's last hidden_states entry is post-final-norm.
            weight = assets.final_norm().astype(np.float64)
            ms = np.mean(hidden ** 2, axis=-1, keepdims=True) + spec.rms_norm_eps
            hidden = hidden / np.sqrt(ms) * weight
        result = metrics(hidden, reference[f"hidden_{layer_index + 1:02d}"])
        print(f"  [layer {layer_index:2d}] {result}")
        assert np.isfinite(hidden).all()
        assert result["cosine"] > 0.999, result


if __name__ == "__main__":
    test_qwen3_real_layers()
    print("ALL QWEN3 REAL LAYER TESTS PASSED")
