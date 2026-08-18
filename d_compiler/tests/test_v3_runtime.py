"""Streaming vendor-runtime tests, including one real checkpoint projection."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.v3_model import Llama32Assets, ModelAssetError
from npu_compiler.v3_runtime import ParallelVendorSession, VendorSession


def _streaming_reference(lhs, rhs):
    """Match grouped N tiles, adaptive K, and FP16 boundary rounding."""
    m_total, k_total = lhs.shape
    n_total = rhs.shape[1]
    result = np.empty((m_total, n_total), dtype=np.float16)
    for m0 in range(0, m_total, 64):
        mt = min(64, m_total - m0)
        starts = list(range(0, n_total, 64))
        widths = [min(64, n_total - start) for start in starts]
        index = 0
        while index < len(starts):
            count, step = VendorSession._select_gemm_group(mt, widths[index:], k_total)
            group_starts = starts[index:index + count]
            group_widths = widths[index:index + count]
            accumulators = [np.zeros((mt, width), dtype=np.float16)
                            for width in group_widths]
            for k0 in range(0, k_total, step):
                kt = min(step, k_total - k0)
                for item, (start, width) in enumerate(zip(group_starts, group_widths)):
                    partial = (lhs[m0:m0 + mt, k0:k0 + kt].astype(np.float32) @
                               rhs[k0:k0 + kt, start:start + width].astype(np.float32))
                    accumulators[item] = np.asarray(
                        accumulators[item].astype(np.float32) + partial, dtype=np.float16)
            for start, width, accumulator in zip(
                    group_starts, group_widths, accumulators):
                result[m0:m0 + mt, start:start + width] = accumulator
            index += count
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
        col = np.arange(9, dtype=np.float16).reshape(9, 1)
        assert np.array_equal(vendor.broadcast_to(col, (9, 137)),
                              np.broadcast_to(col, (9, 137)))
        row = np.arange(137, dtype=np.float16).reshape(1, 137)
        assert np.array_equal(vendor.broadcast_to(row, (9, 137)),
                              np.broadcast_to(row, (9, 137)))
        matrix = np.asarray(rng.normal(size=(67, 70)), dtype=np.float16)
        assert np.array_equal(vendor.transpose2d(matrix), matrix.T)
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


def test_parallel_columns_preserve_schedule():
    rng = np.random.default_rng(4)
    lhs = np.asarray(rng.normal(0, 0.1, (7, 130)), dtype=np.float16)
    rhs = np.asarray(rng.normal(0, 0.1, (130, 385)), dtype=np.float16)
    with VendorSession() as serial:
        expected = serial.gemm(lhs, rhs)
        serial_calls = serial.stats()["invocations"]
    with ParallelVendorSession(3) as parallel:
        actual = parallel.gemm(lhs, rhs)
        parallel_calls = parallel.stats()["invocations"]
    assert np.array_equal(actual, expected)
    assert parallel_calls == serial_calls


if __name__ == "__main__":
    test_streaming_gemm_and_primitives()
    test_real_q_projection_tile_stream()
    test_parallel_columns_preserve_schedule()
    print("ALL V3 STREAMING-RUNTIME TESTS PASSED")
