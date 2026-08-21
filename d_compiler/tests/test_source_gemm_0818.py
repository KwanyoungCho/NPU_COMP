"""Panel-packed source GEMM tests, including a stride above uint16."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.source_gemm_0818 import PackedRhsGemm


def _pack(rhs, panel=64):
    inner, columns = rhs.shape
    assert columns % panel == 0
    return np.ascontiguousarray(
        rhs.reshape(inner, columns // panel, panel).transpose(1, 0, 2)).reshape(-1)


def test_rhs_stride_above_uint16():
    inner, columns = 3, 65536
    lhs = np.asarray([[1.5, -2.0, 0.25]], dtype=np.float16)
    rhs = (np.arange(inner * columns, dtype=np.float32).reshape(inner, columns) % 17 - 8)
    rhs = np.asarray(rhs * 0.03125, dtype=np.float16)
    gemm = PackedRhsGemm(inner, columns)
    output = gemm.run(lhs, _pack(rhs))
    expected = np.asarray(lhs.astype(np.float32) @ rhs.astype(np.float32), dtype=np.float16)
    assert np.array_equal(output, expected)


def test_multi_k_tile_panel_gemm():
    rng = np.random.default_rng(818)
    lhs = np.asarray(rng.normal(0, 0.2, (1, 70)), dtype=np.float16)
    rhs = np.asarray(rng.normal(0, 0.2, (70, 128)), dtype=np.float16)
    output = PackedRhsGemm(70, 128).run(lhs, _pack(rhs))
    expected = np.asarray(lhs.astype(np.float32) @ rhs.astype(np.float32), dtype=np.float16)
    assert np.array_equal(output, expected)


if __name__ == "__main__":
    test_rhs_stride_above_uint16()
    test_multi_k_tile_panel_gemm()
    print("ALL PACKED SOURCE GEMM TESTS PASSED")
