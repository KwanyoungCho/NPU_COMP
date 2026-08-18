"""Relax -> host-resident execution plan -> 0818 vendor binary end-to-end."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler import model
from npu_compiler.v3_executor import compile_module
from npu_compiler.v3_runtime import VendorSession


class NumpyBoundarySession:
    """Reference with the same FP16 boundaries as the streaming vendor runtime."""

    @staticmethod
    def gemm(lhs, rhs):
        lhs, rhs = np.asarray(lhs, np.float16), np.asarray(rhs, np.float16)
        rows, inner = lhs.shape
        cols = rhs.shape[1]
        output = np.empty((rows, cols), np.float16)
        for row0 in range(0, rows, 64):
            part_rows = min(64, rows - row0)
            for col0 in range(0, cols, 64):
                part_cols = min(64, cols - col0)
                step = VendorSession._gemm_capacity(part_rows, part_cols)
                acc = np.zeros((part_rows, part_cols), np.float16)
                for inner0 in range(0, inner, step):
                    part_inner = min(step, inner - inner0)
                    product = (
                        lhs[row0:row0 + part_rows, inner0:inner0 + part_inner].astype(
                            np.float32)
                        @ rhs[inner0:inner0 + part_inner,
                              col0:col0 + part_cols].astype(np.float32)
                    )
                    acc = np.asarray(acc.astype(np.float32) + product, np.float16)
                output[row0:row0 + part_rows, col0:col0 + part_cols] = acc
        return output

    @staticmethod
    def binary(name, lhs, rhs):
        functions = {
            "add": np.add,
            "subtract": np.subtract,
            "multiply": np.multiply,
            "divide": np.divide,
        }
        return np.asarray(
            functions[name](np.asarray(lhs, np.float16).astype(np.float32),
                            np.asarray(rhs, np.float16).astype(np.float32)), np.float16)

    @staticmethod
    def unary(name, values):
        source = np.asarray(values, np.float16).astype(np.float32)
        functions = {
            "sqrt": np.sqrt,
            "exp": np.exp,
            "negative": np.negative,
            "cos": np.cos,
            "sin": np.sin,
            "silu": lambda x: x / (1.0 + np.exp(-x)),
        }
        return np.asarray(functions[name](source), np.float16)

    @staticmethod
    def reduce_sum_last(values):
        return np.asarray(np.asarray(values, np.float16).astype(np.float32).sum(
            -1, keepdims=True), np.float16)

    @staticmethod
    def reduce_max_last(values):
        return np.asarray(values, np.float16).max(-1, keepdims=True)

    @staticmethod
    def broadcast_to(values, shape):
        return np.array(np.broadcast_to(np.asarray(values, np.float16), shape),
                        dtype=np.float16, copy=True)

    @staticmethod
    def transpose2d(values):
        return np.ascontiguousarray(np.asarray(values, np.float16).T)


def test_reduced_prefill_relax_graph():
    sequence = 2
    module = model.build_prefill_layer_module(model.REDUCED, sequence)
    plan = compile_module(module)
    inputs = model.make_weights(model.REDUCED, seed=818, ws=0.04)
    inputs["x"] = inputs["x"][:sequence]

    expected = plan.run(inputs, vendor=NumpyBoundarySession())
    with VendorSession() as vendor:
        actual = plan.run(inputs, vendor=vendor)
        stats = vendor.stats()

    assert actual.shape == (sequence, model.REDUCED.D + 2 * model.REDUCED.KV *
                            model.REDUCED.HD)
    assert np.all(np.isfinite(actual))
    max_error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
    assert max_error <= 0.02, max_error
    summary = plan.summary()
    assert summary["ops"]["relax.matmul"] == 23
    assert summary["ops"]["relax.max"] == model.REDUCED.H
    assert summary["host_layout_bindings"] == 19
    assert stats["invocations"] > summary["bindings"] - summary["host_layout_bindings"]
    return max_error, stats, summary


if __name__ == "__main__":
    error, stats, summary = test_reduced_prefill_relax_graph()
    print(f"REDUCED PREFILL max_error={error} stats={stats}")
    print(f"PLAN {summary}")
    print("ALL V3 RELAX-EXECUTOR TESTS PASSED")
