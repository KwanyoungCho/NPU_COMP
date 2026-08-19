"""Qwen3 decoder-layer graph proxy regression (no checkpoint needed).

Checks the prefill layer against a float64 reference and the streaming
oracle, the fused decode layer against a float64 decode step over the
prefill-seeded cache, and vendor a.out closure.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler import driver
from npu_compiler.gemma4_graph import banded_causal_mask, gemma_freqs_row
from npu_compiler.model_spec import AttentionSpec, LayerSpec, ModelSpec
from npu_compiler.qwen3_graph import (
    build_qwen3_decode_layer_module, build_qwen3_prefill_layer_module)
from npu_compiler.source_runtime_0818 import build_source_model
from npu_compiler.v3_executor import compile_module as compile_streaming
from npu_compiler.v3_runtime import VendorSession

S = 5
ATTENTION = AttentionSpec("full", 4, 2, 16, rope_theta=1000000.0, qk_norm=True)
SPEC = ModelSpec("qwen3-proxy", 32, 64, 1e-6, (
    LayerSpec(0, ATTENTION, 48, "silu", 0),
))


def f16(value):
    return np.asarray(value, dtype=np.float16)


def make_inputs(seed=11):
    attention = ATTENTION
    D, HD, H, KV = SPEC.hidden_size, attention.head_dim, 4, 2
    F = SPEC.layers[0].ffn_hidden
    rng = np.random.default_rng(seed)
    u = lambda *shape: f16(rng.uniform(-0.2, 0.2, shape))
    return {
        "x": f16(rng.standard_normal((S, D)) * 0.5),
        "Wn1": f16(rng.uniform(0.8, 1.2, (1, D))),
        "Wn2": f16(rng.uniform(0.8, 1.2, (1, D))),
        "Wq": u(D, H * HD), "Wqn": f16(rng.uniform(0.8, 1.2, (1, HD))),
        "Wk": u(D, KV * HD), "Wkn": f16(rng.uniform(0.8, 1.2, (1, HD))),
        "Wv": u(D, KV * HD),
        "Wo": u(H * HD, D),
        "Wg": u(D, F), "Wu": u(D, F), "Wd": u(F, D),
    }


def ref_rms(x, w, eps=1e-6):
    ms = np.mean(x ** 2, axis=-1, keepdims=True) + eps
    return x / np.sqrt(ms) * w


def ref_rope(x, positions):
    freqs = gemma_freqs_row(ATTENTION)[0]
    half = x.shape[1] // 2
    cos = np.cos(positions.reshape(-1, 1) * freqs)
    sin = np.sin(positions.reshape(-1, 1) * freqs)
    rotated = np.concatenate([-x[:, half:], x[:, :half]], axis=1)
    return x * cos + rotated * sin


def ref_layer(inputs, x, positions, cache=None):
    """Float64 Qwen3 layer; returns (y, kt_list, v_list) over positions."""
    HD, H, KV, GPK = 16, 4, 2, 2
    g = lambda name: inputs[name].astype(np.float64)
    x = x.astype(np.float64)
    xn = ref_rms(x, g("Wn1"))
    kts, vs = [], []
    for kv in range(KV):
        key = ref_rope(ref_rms(xn @ g("Wk")[:, kv * HD:(kv + 1) * HD], g("Wkn")),
                       positions)
        value = xn @ g("Wv")[:, kv * HD:(kv + 1) * HD]
        if cache is not None:
            key = np.concatenate([cache[0][kv].T, key], axis=0)
            value = np.concatenate([cache[1][kv], value], axis=0)
        kts.append(key.T)
        vs.append(value)
    total = kts[0].shape[1]
    if cache is None:
        mask = banded_causal_mask(len(positions)).astype(np.float64)
    else:
        mask = np.zeros((len(positions), total))
    contexts = []
    for head in range(H):
        query = ref_rope(ref_rms(xn @ g("Wq")[:, head * HD:(head + 1) * HD],
                                 g("Wqn")), positions)
        score = query @ kts[head // GPK] / np.sqrt(HD) + mask
        score = score - score.max(axis=1, keepdims=True)
        exp = np.exp(score)
        contexts.append((exp / exp.sum(axis=1, keepdims=True)) @ vs[head // GPK])
    attention_out = np.concatenate(contexts, axis=1) @ g("Wo")
    h1 = x + attention_out
    hn = ref_rms(h1, g("Wn2"))
    gate = hn @ g("Wg")
    ffn = (gate / (1.0 + np.exp(-gate)) * (hn @ g("Wu"))) @ g("Wd")
    return h1 + ffn, kts, vs


def check(name, actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    rel = float((np.abs(actual - expected) /
                 np.maximum(np.abs(expected), 1.0)).max())
    assert np.isfinite(actual).all(), f"{name}: non-finite"
    assert rel < 0.05, f"{name}: relative error {rel}"
    print(f"  [PASS] {name}: max rel {rel:.4f}")


def test_qwen3_layer():
    D, HD, KV = 32, 16, 2
    inputs = make_inputs()
    module = build_qwen3_prefill_layer_module(SPEC, 0, S)
    compiled = driver.compile_module(module, backend="source-0818")
    output = np.asarray(driver.run_compiled(*compiled, inputs), dtype=np.float16)
    plan = compile_streaming(module)
    with VendorSession(build_source_model()) as session:
        streaming = np.asarray(plan.run(inputs, vendor=session), dtype=np.float16)
    gap = float(np.max(np.abs(
        output.astype(np.float32) - streaming.astype(np.float32))))
    assert gap <= 1e-2, f"streaming oracle gap {gap}"

    expected, kt_ref, v_ref = ref_layer(inputs, inputs["x"], np.arange(S))
    hidden = output[:, :D]
    check("prefill hidden", hidden, expected)
    keys, values = [], []
    for kv in range(KV):
        base = D + 2 * kv * HD
        key = output[:, base:base + HD]
        value = output[:, base + HD:base + 2 * HD]
        check(f"prefill K{kv}", key.astype(np.float64), kt_ref[kv].T)
        check(f"prefill V{kv}", value, v_ref[kv])
        keys.append(np.ascontiguousarray(key.T))
        values.append(np.ascontiguousarray(value))

    # Fused decode over the seeded cache vs a float64 decode step.
    rng = np.random.default_rng(23)
    x_new = f16(rng.standard_normal((1, D)) * 0.5)
    decode_inputs = dict(inputs)
    decode_inputs["x"] = x_new
    decode_inputs["pos"] = f16([[S]])
    for kv in range(KV):
        decode_inputs[f"Kt{kv}"] = keys[kv]
        decode_inputs[f"Vc{kv}"] = values[kv]
    decode_module = build_qwen3_decode_layer_module(SPEC, 0, S + 1)
    decode_compiled = driver.compile_module(decode_module, backend="source-0818")
    decoded = np.asarray(
        driver.run_compiled(*decode_compiled, decode_inputs), dtype=np.float16)
    cache = ([kt.astype(np.float64) for kt in keys],
             [v.astype(np.float64) for v in values])
    expected_decode, _, _ = ref_layer(
        inputs, x_new, np.asarray([S]), cache=cache)
    check("decode hidden", decoded[:, :D], expected_decode)

    # Vendor closure on the real a.out.
    with VendorSession() as session:
        vendor = np.asarray(plan.run(inputs, vendor=session), dtype=np.float16)
    vendor_gap = float(np.max(np.abs(
        output.astype(np.float32) - vendor.astype(np.float32))))
    assert vendor_gap <= 1e-2, f"vendor gap {vendor_gap}"
    print(f"  [PASS] vendor a.out streaming agrees (max abs {vendor_gap:.2e})")


if __name__ == "__main__":
    test_qwen3_layer()
    print("ALL QWEN3 LAYER TESTS PASSED")
