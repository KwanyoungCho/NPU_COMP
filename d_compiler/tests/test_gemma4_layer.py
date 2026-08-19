"""Gemma 4 decoder-layer graph proxy regression (no checkpoint needed).

Runs all four layer shapes (sliding/full x owner/shared) on the source
C-model, checks the fixed-buffer streaming oracle agreement, and compares
against a float64 reference of the official layer math.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler import driver
from npu_compiler.gemma4_graph import (
    banded_causal_mask, build_gemma4_prefill_layer_module, gemma_freqs_row)
from npu_compiler.model_spec import AttentionSpec, LayerSpec, ModelSpec
from npu_compiler.source_runtime_0818 import build_source_model
from npu_compiler.v3_executor import compile_module as compile_streaming
from npu_compiler.v3_runtime import VendorSession

S = 5
SLIDING = AttentionSpec("sliding", 2, 1, 16, window=4, rope_theta=10000.0, qk_norm=True)
FULL = AttentionSpec("full", 2, 1, 32, rope_theta=1000000.0,
                     partial_rotary_factor=0.25, qk_norm=True)
SPEC = ModelSpec("gemma-proxy", 32, 64, 1e-6, (
    LayerSpec(0, SLIDING, 48, "gelu_tanh", 0, ple_dim=8),
    LayerSpec(1, FULL, 48, "gelu_tanh", 1, ple_dim=8),
    LayerSpec(2, SLIDING, 96, "gelu_tanh", 0, ple_dim=8),
    LayerSpec(3, FULL, 96, "gelu_tanh", 1, ple_dim=8),
), scale_embeddings=True, final_logit_softcapping=30.0)


def f16(value):
    return np.asarray(value, dtype=np.float16)


def make_inputs(layer_index, seed=4):
    layer = SPEC.layers[layer_index]
    attention = layer.attention
    D, HD, H, F, Pd = (SPEC.hidden_size, attention.head_dim,
                       attention.num_query_heads, layer.ffn_hidden, layer.ple_dim)
    rng = np.random.default_rng(seed)
    u = lambda *shape: f16(rng.uniform(-0.2, 0.2, shape))
    inputs = {
        "x": f16(rng.standard_normal((S, D)) * 0.5),
        "pli": f16(rng.standard_normal((S, Pd)) * 0.5),
        "Wn1": f16(rng.uniform(0.8, 1.2, (1, D))),
        "Wn2": f16(rng.uniform(0.8, 1.2, (1, D))),
        "Wn3": f16(rng.uniform(0.8, 1.2, (1, D))),
        "Wn4": f16(rng.uniform(0.8, 1.2, (1, D))),
        "Wn5": f16(rng.uniform(0.8, 1.2, (1, D))),
        "Wq": u(D, H * HD), "Wqn": f16(rng.uniform(0.8, 1.2, (1, HD))),
        "Wo": u(H * HD, D),
        "Wg": u(D, F), "Wu": u(D, F), "Wd": u(F, D),
        "Wpg": u(D, Pd), "Wpp": u(Pd, D),
        "ls": f16([[1.03125]]),
    }
    if layer.owns_cache:
        inputs["Wk"] = u(D, HD)
        inputs["Wkn"] = f16(rng.uniform(0.8, 1.2, (1, HD)))
        inputs["Wv"] = u(D, HD)
    return inputs


def ref_rms(x, w, eps):
    ms = np.mean(x ** 2, axis=-1, keepdims=True) + eps
    xn = x / np.sqrt(ms)
    return xn if w is None else xn * w


def ref_rope(x, attention):
    freqs = gemma_freqs_row(attention)[0]
    hd = x.shape[1]
    half = hd // 2
    positions = np.arange(x.shape[0]).reshape(-1, 1)
    cos = np.cos(positions * freqs)
    sin = np.sin(positions * freqs)
    rotated = np.concatenate([-x[:, half:], x[:, :half]], axis=1)
    return x * cos + rotated * sin


def ref_gelu(x):
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def ref_layer(layer_index, inputs, kt=None, value=None):
    layer = SPEC.layers[layer_index]
    attention = layer.attention
    HD, H = attention.head_dim, attention.num_query_heads
    eps = SPEC.rms_norm_eps
    g = lambda name: inputs[name].astype(np.float64)
    x = g("x")
    xn = ref_rms(x, g("Wn1"), eps)
    q_all = xn @ g("Wq")
    if layer.owns_cache:
        key = ref_rope(ref_rms(xn @ g("Wk"), g("Wkn"), eps), attention)
        value = ref_rms(xn @ g("Wv"), None, eps)
        kt = key.T
    window = attention.window if attention.kind == "sliding" else None
    mask = banded_causal_mask(S, window).astype(np.float64)
    contexts = []
    for head in range(H):
        query = q_all[:, head * HD:(head + 1) * HD]
        query = ref_rope(ref_rms(query, g("Wqn"), eps), attention)
        score = query @ kt + mask
        score = score - score.max(axis=1, keepdims=True)
        exp = np.exp(score)
        contexts.append((exp / exp.sum(axis=1, keepdims=True)) @ value)
    attention_out = np.concatenate(contexts, axis=1) @ g("Wo")
    h1 = x + ref_rms(attention_out, g("Wn2"), eps)
    f = ref_rms(h1, g("Wn3"), eps)
    ffn = (ref_gelu(f @ g("Wg")) * (f @ g("Wu"))) @ g("Wd")
    h2 = h1 + ref_rms(ffn, g("Wn4"), eps)
    gated = ref_gelu(h2 @ g("Wpg")) * g("pli")
    h3 = h2 + ref_rms(gated @ g("Wpp"), g("Wn5"), eps)
    y = h3 * g("ls")
    if layer.owns_cache:
        return y, kt, value
    return y, kt, value


def run_layer(layer_index, inputs):
    module = build_gemma4_prefill_layer_module(SPEC, layer_index, S)
    compiled = driver.compile_module(module, backend="source-0818")
    output = np.asarray(driver.run_compiled(*compiled, inputs), dtype=np.float16)
    plan = compile_streaming(module)
    with VendorSession(build_source_model()) as session:
        streaming = np.asarray(plan.run(inputs, vendor=session), dtype=np.float16)
    gap = float(np.max(np.abs(
        output.astype(np.float32) - streaming.astype(np.float32))))
    # The fixed-buffer oracle rounds the PE accumulator to FP16 between K
    # tiles; a single-program source run keeps it in float32.  Those ULP-level
    # differences amplify through the norm/residual chain after the FFN.
    assert gap <= 1e-2, f"layer {layer_index}: streaming oracle gap {gap}"
    return output


def check(name, actual, expected):
    actual = actual.astype(np.float64)
    delta = np.abs(actual - expected)
    denom = np.maximum(np.abs(expected), 1.0)
    rel = float((delta / denom).max())
    assert np.isfinite(actual).all(), f"{name}: non-finite"
    assert rel < 0.05, f"{name}: relative error {rel}"
    print(f"  [PASS] {name}: max rel {rel:.4f}")


def test_gemma4_layers():
    shared_cache = {}
    for layer_index in range(4):
        layer = SPEC.layers[layer_index]
        inputs = make_inputs(layer_index)
        if layer.owns_cache:
            output = run_layer(layer_index, inputs)
            D, HD = SPEC.hidden_size, layer.attention.head_dim
            hidden = output[:, :D]
            key = output[:, D:D + HD]
            value = output[:, D + HD:]
            expected, kt_ref, value_ref = ref_layer(layer_index, inputs)
            check(f"layer{layer_index} ({layer.attention.kind} owner)",
                  hidden, expected)
            check(f"layer{layer_index} key", key.astype(np.float64).T, kt_ref)
            check(f"layer{layer_index} value", value, value_ref)
            shared_cache[layer.attention.kind] = (
                np.ascontiguousarray(key.T), np.ascontiguousarray(value))
        else:
            kt, value = shared_cache[layer.attention.kind]
            inputs["Kt"] = kt
            inputs["Vc"] = value
            output = run_layer(layer_index, inputs)
            expected, _, _ = ref_layer(
                layer_index, inputs, kt.astype(np.float64), value.astype(np.float64))
            check(f"layer{layer_index} ({layer.attention.kind} shared)",
                  output, expected)


if __name__ == "__main__":
    test_gemma4_layers()
    print("ALL GEMMA4 LAYER TESTS PASSED")
