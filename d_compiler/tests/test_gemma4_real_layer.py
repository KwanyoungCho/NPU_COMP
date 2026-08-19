"""Official-weight Gemma 4 layers vs HF intermediates (skips without assets).

Ladder: PLE table math vs HF per_layer_inputs, then real decoder layers
(sliding owner 0, full owner 4, sliding owner 13 -> shared 15) each fed the
HF hidden state for that layer and compared to the next HF hidden state.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.gemma4_ple import compute_rows
from npu_compiler import driver
from npu_compiler.gemma4_graph import build_gemma4_prefill_layer_module
from npu_compiler.gemma4_model import Gemma4Assets, default_model_path

REFERENCE = os.path.join(ROOT, "d_compiler", "build",
                         "gemma4_layer_reference_hello.npz")


def metrics(actual, expected):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    delta = actual - expected
    cosine = float(np.dot(actual, expected) /
                   (np.linalg.norm(actual) * np.linalg.norm(expected)))
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "cosine": cosine,
    }


def layer_inputs(assets, layer_index, hidden, pli):
    layer = assets.spec.layers[layer_index]
    inputs = {
        "x": hidden,
        "pli": pli,
        "Wn1": assets.norm(layer_index, "input_layernorm"),
        "Wn2": assets.norm(layer_index, "post_attention_layernorm"),
        "Wn3": assets.norm(layer_index, "pre_feedforward_layernorm"),
        "Wn4": assets.norm(layer_index, "post_feedforward_layernorm"),
        "Wn5": assets.norm(layer_index, "post_per_layer_input_norm"),
        "Wq": assets.linear(layer_index, "self_attn.q_proj"),
        "Wqn": assets.qk_norm(layer_index, "q_norm"),
        "Wo": assets.linear(layer_index, "self_attn.o_proj"),
        "Wg": assets.linear(layer_index, "mlp.gate_proj"),
        "Wu": assets.linear(layer_index, "mlp.up_proj"),
        "Wd": assets.linear(layer_index, "mlp.down_proj"),
        "Wpg": assets.linear(layer_index, "per_layer_input_gate"),
        "Wpp": assets.linear(layer_index, "per_layer_projection"),
        "ls": assets.layer_scalar(layer_index),
    }
    if layer.owns_cache:
        inputs["Wk"] = assets.linear(layer_index, "self_attn.k_proj")
        inputs["Wkn"] = assets.qk_norm(layer_index, "k_norm")
        inputs["Wv"] = assets.linear(layer_index, "self_attn.v_proj")
    return inputs


def run_layer(assets, layer_index, inputs, S):
    module = build_gemma4_prefill_layer_module(assets.spec, layer_index, S)
    compiled = driver.compile_module(module, backend="source-0818", reuse=True)
    return np.asarray(driver.run_compiled(*compiled, inputs), dtype=np.float16)


def test_gemma4_real_layers():
    path = default_model_path()
    if not (path / "model.safetensors").exists() or not os.path.exists(REFERENCE):
        print("  (skip: checkpoint or layer reference not present)")
        return
    assets = Gemma4Assets(path)
    spec = assets.spec
    reference = np.load(REFERENCE)
    ids = reference["input_ids"]
    S = ids.size
    hf_pli = reference["per_layer_inputs"].reshape(S, -1)

    # Rung 1: embedding scale and PLE math against HF.
    ours_embeds = assets.embedding(ids)
    embed_metrics = metrics(ours_embeds, reference["inputs_embeds"])
    assert embed_metrics["max_abs"] < 0.05, embed_metrics
    ours_pli = compute_rows(
        spec, ours_embeds, assets.ple_rows(ids),
        assets.per_layer_model_projection(), assets.per_layer_projection_norm())
    ple_metrics = metrics(ours_pli, hf_pli)
    print(f"  [PLE] {ple_metrics}")
    assert ple_metrics["cosine"] > 0.999, ple_metrics

    # Rung 2: real decoder layers, each isolated on the HF input hidden state.
    def hf_hidden(index):
        return np.asarray(reference[f"hidden_{index:02d}"], dtype=np.float16)

    def hf_pli_layer(index):
        return np.ascontiguousarray(
            hf_pli[:, index * 256:(index + 1) * 256], dtype=np.float16)

    shared_kv = {}
    for layer_index in (0, 4, 13, 15):
        layer = spec.layers[layer_index]
        inputs = layer_inputs(
            assets, layer_index, hf_hidden(layer_index), hf_pli_layer(layer_index))
        if not layer.owns_cache:
            kt, value = shared_kv[layer.attention.kind]
            inputs["Kt"], inputs["Vc"] = kt, value
        output = run_layer(assets, layer_index, inputs, S)
        if layer.owns_cache:
            D, HD = spec.hidden_size, layer.attention.head_dim
            hidden, key, value = (output[:, :D], output[:, D:D + HD],
                                  output[:, D + HD:])
            shared_kv[layer.attention.kind] = (
                np.ascontiguousarray(key.T), np.ascontiguousarray(value))
        else:
            hidden = output
        result = metrics(hidden, hf_hidden(layer_index + 1))
        kind = layer.attention.kind + (" owner" if layer.owns_cache else " shared")
        print(f"  [layer {layer_index:2d} {kind}] {result}")
        assert np.isfinite(hidden.astype(np.float64)).all()
        assert result["cosine"] > 0.999, result


if __name__ == "__main__":
    test_gemma4_real_layers()
    print("ALL GEMMA4 REAL LAYER TESTS PASSED")
