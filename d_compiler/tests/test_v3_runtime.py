"""Streaming vendor-runtime tests, including one real checkpoint projection."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.v3_model import Llama32Assets, ModelAssetError
from npu_compiler.v3_runtime import VendorSession


def _streaming_reference(lhs, rhs):
    """Match VendorSession.gemm's adaptive K tiles and FP16 boundary rounding."""
    m_total, k_total = lhs.shape
    n_total = rhs.shape[1]
    result = np.empty((m_total, n_total), dtype=np.float16)
    for m0 in range(0, m_total, 64):
        mt = min(64, m_total - m0)
        for n0 in range(0, n_total, 64):
            nt = min(64, n_total - n0)
            step = VendorSession._gemm_capacity(mt, nt)
            acc = np.zeros((mt, nt), dtype=np.float16)
            for k0 in range(0, k_total, step):
                kt = min(step, k_total - k0)
                partial = (lhs[m0:m0 + mt, k0:k0 + kt].astype(np.float32) @
                           rhs[k0:k0 + kt, n0:n0 + nt].astype(np.float32))
                acc = np.asarray(acc.astype(np.float32) + partial, dtype=np.float16)
            result[m0:m0 + mt, n0:n0 + nt] = acc
    return result


def test_streaming_gemm_and_primitives():
    rng = np.random.default_rng(30818)
    lhs = np.asarray(rng.normal(0, 0.2, (65, 70)), dtype=np.float16)
    rhs = np.asarray(rng.normal(0, 0.2, (70, 67)), dtype=np.float16)
    with VendorSession() as vendor:
        got = vendor.gemm(lhs, rhs)
        assert np.array_equal(got, _streaming_reference(lhs, rhs))

        a = np.asarray(rng.normal(0, 0.5, (9, 11)), dtype=np.float16)
        b = np.asarray(rng.normal(1, 0.5, (9, 11)), dtype=np.float16)
        assert np.array_equal(vendor.binary("add", a, b), np.asarray(a + b, np.float16))
        silu = vendor.unary("silu", a)
        expected_silu = np.asarray(a.astype(np.float32) /
                                   (1 + np.exp(-a.astype(np.float32))), dtype=np.float16)
        assert np.array_equal(silu, expected_silu)
        sums = vendor.reduce_sum_last(a)
        expected_sum = np.asarray(a.astype(np.float32).sum(-1, keepdims=True), dtype=np.float16)
        assert np.array_equal(sums, expected_sum)
        negative = -np.arange(1, 100, dtype=np.float16).reshape(9, 11)
        assert np.array_equal(vendor.reduce_max_last(negative),
                              negative.max(-1, keepdims=True))
        assert vendor.stats()["invocations"] > 0


def test_real_q_projection_tile_stream():
    try:
        assets = Llama32Assets()
    except ModelAssetError as error:
        print(f"SKIP: {error}")
        return
    rng = np.random.default_rng(3)
    hidden = np.asarray(rng.normal(0, 0.1, (1, 3072)), dtype=np.float16)
    loader = lambda ks, ns: assets.linear_tile(0, "self_attn.q_proj", ks, ns)
    with VendorSession() as vendor:
        got = vendor.gemm(hidden, rhs_loader=loader, n=128)
    # Build only this 3072x128 reference view; no full-model copy.
    weight = loader(slice(0, 3072), slice(0, 128))
    expected = _streaming_reference(hidden, weight)
    assert np.array_equal(got, expected)


if __name__ == "__main__":
    test_streaming_gemm_and_primitives()
    test_real_q_projection_tile_stream()
    print("ALL V3 STREAMING-RUNTIME TESTS PASSED")
