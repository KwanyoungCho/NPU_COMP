"""Fused prefill-cache/decode graphs on the extended source C-model backend."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler import driver, model
from npu_compiler.source_runtime_0818 import build_source_model
from npu_compiler.v3_executor import compile_module as compile_streaming
from npu_compiler.v3_runtime import VendorSession


def _inputs(cfg, sequence):
    weights = model.make_weights(cfg, seed=818, ws=0.05)
    return weights, {
        "x": np.asarray(weights["x"][:sequence], dtype=np.float16),
        "Wn1": weights["Wn1"],
        "Wn2": weights["Wn2"],
        "Wq": np.concatenate([weights[f"Wq{head}"] for head in range(cfg.H)], axis=1),
        "Wk": np.concatenate([weights[f"Wk{kv}"] for kv in range(cfg.KV)], axis=1),
        "Wv": np.concatenate([weights[f"Wv{kv}"] for kv in range(cfg.KV)], axis=1),
        "Wo": np.concatenate([weights[f"Wo{head}"] for head in range(cfg.H)], axis=0),
        "Wg": weights["Wg"],
        "Wu": weights["Wu"],
        "Wd": weights["Wd"],
    }


def _compare_source_and_streaming(module, inputs):
    compiled = driver.compile_module(module, backend="source-0818")
    source_output = driver.run_compiled(*compiled, inputs)
    plan = compile_streaming(module)
    with VendorSession(build_source_model()) as session:
        streaming_output = plan.run(inputs, vendor=session)
    streaming_output = streaming_output.astype(np.float32)
    # A single source-model program keeps the PE accumulator live across K
    # tiles.  The fixed-buffer oracle snapshots it between processes, so the
    # final decode residual may differ by one FP16 ULP.
    assert float(np.max(np.abs(source_output - streaming_output))) <= 5e-4
    return source_output


def test_fused_prefill_cache_and_decode():
    cfg = model.REDUCED
    sequence = 3
    weights, inputs = _inputs(cfg, sequence)
    prefill = model.build_v3_prefill_layer_module(cfg, sequence, return_cache=True)
    output = _compare_source_and_streaming(prefill, inputs)
    assert output.shape == (sequence, cfg.D + 2 * cfg.KV * cfg.HD)

    hidden = np.asarray(weights["x"][sequence:sequence + 1], dtype=np.float16)
    position = np.asarray([[sequence]], dtype=np.float16)
    kv_inputs = {
        "x": hidden,
        "Wn1": inputs["Wn1"],
        "Wk": inputs["Wk"],
        "Wv": inputs["Wv"],
        "pos": position,
    }
    projected = _compare_source_and_streaming(
        model.build_v3_decode_kv_module(cfg), kv_inputs)

    decode_inputs = {
        "x": hidden,
        "Wn1": inputs["Wn1"],
        "Wn2": inputs["Wn2"],
        "Wq": inputs["Wq"],
        "Wo": inputs["Wo"],
        "pos": position,
        "Wg": inputs["Wg"],
        "Wu": inputs["Wu"],
        "Wd": inputs["Wd"],
    }
    offset = cfg.D
    for kv in range(cfg.KV):
        old_key = output[:, offset:offset + cfg.HD]
        offset += cfg.HD
        old_value = output[:, offset:offset + cfg.HD]
        offset += cfg.HD
        new_key = projected[:, 2 * kv * cfg.HD:(2 * kv + 1) * cfg.HD]
        new_value = projected[:, (2 * kv + 1) * cfg.HD:(2 * kv + 2) * cfg.HD]
        decode_inputs[f"Kt{kv}"] = np.concatenate((old_key, new_key), axis=0).T
        decode_inputs[f"Vc{kv}"] = np.concatenate((old_value, new_value), axis=0)
    decoded = _compare_source_and_streaming(
        model.build_v3_decode_layer_module(cfg, sequence + 1), decode_inputs)
    assert decoded.shape == (1, cfg.D)
    assert np.isfinite(decoded).all()


def test_single_program_decode_matches_split():
    """The one-program decode layer (in-program K/V append) must reproduce the
    split kv-projection + host-append + attention path exactly."""
    cfg = model.REDUCED
    sequence = 3
    weights, inputs = _inputs(cfg, sequence)
    prefill = model.build_v3_prefill_layer_module(cfg, sequence, return_cache=True)
    compiled = driver.compile_module(prefill, backend="source-0818")
    output = np.asarray(driver.run_compiled(*compiled, inputs), dtype=np.float16)

    hidden = np.asarray(weights["x"][sequence:sequence + 1], dtype=np.float16)
    position = np.asarray([[sequence]], dtype=np.float16)
    kv_compiled = driver.compile_module(
        model.build_v3_decode_kv_module(cfg), backend="source-0818")
    projected = np.asarray(driver.run_compiled(*kv_compiled, {
        "x": hidden, "Wn1": inputs["Wn1"], "Wk": inputs["Wk"],
        "Wv": inputs["Wv"], "pos": position,
    }), dtype=np.float16)

    split_inputs = {
        "x": hidden, "Wn1": inputs["Wn1"], "Wn2": inputs["Wn2"],
        "Wq": inputs["Wq"], "Wo": inputs["Wo"], "pos": position,
        "Wg": inputs["Wg"], "Wu": inputs["Wu"], "Wd": inputs["Wd"],
    }
    fused_inputs = {
        "x": hidden, "Wn1": inputs["Wn1"], "Wn2": inputs["Wn2"],
        "Wq": inputs["Wq"], "Wk": inputs["Wk"], "Wv": inputs["Wv"],
        "Wo": inputs["Wo"], "pos": position,
        "Wg": inputs["Wg"], "Wu": inputs["Wu"], "Wd": inputs["Wd"],
    }
    offset = cfg.D
    for kv in range(cfg.KV):
        old_key = output[:, offset:offset + cfg.HD]
        offset += cfg.HD
        old_value = output[:, offset:offset + cfg.HD]
        offset += cfg.HD
        new_key = projected[:, 2 * kv * cfg.HD:(2 * kv + 1) * cfg.HD]
        new_value = projected[:, (2 * kv + 1) * cfg.HD:(2 * kv + 2) * cfg.HD]
        split_inputs[f"Kt{kv}"] = np.concatenate((old_key, new_key), axis=0).T
        split_inputs[f"Vc{kv}"] = np.concatenate((old_value, new_value), axis=0)
        fused_inputs[f"Kt{kv}"] = np.ascontiguousarray(old_key.T)
        fused_inputs[f"Vc{kv}"] = old_value

    split_compiled = driver.compile_module(
        model.build_v3_decode_layer_module(cfg, sequence + 1), backend="source-0818")
    split_hidden = np.asarray(
        driver.run_compiled(*split_compiled, split_inputs), dtype=np.float16)

    fused = _compare_source_and_streaming(
        model.build_v3_decode_fused_layer_module(cfg, sequence + 1), fused_inputs)
    fused = np.asarray(fused, dtype=np.float16)
    assert fused.shape == (1, cfg.D + 2 * cfg.KV * cfg.HD)
    assert np.array_equal(fused[:, :cfg.D], split_hidden)
    assert np.array_equal(fused[:, cfg.D:], projected)


if __name__ == "__main__":
    test_fused_prefill_cache_and_decode()
    test_single_program_decode_matches_split()
    print("ALL V3 SOURCE PREFILL/DECODE TESTS PASSED")
