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
            starts = list(range(0, cols, 64))
            widths = [min(64, cols - start) for start in starts]
            index = 0
            while index < len(starts):
                count, step = VendorSession._select_gemm_group(
                    part_rows, widths[index:], inner)
                group_starts = starts[index:index + count]
                group_widths = widths[index:index + count]
                accumulators = [np.zeros((part_rows, width), np.float16)
                                for width in group_widths]
                for inner0 in range(0, inner, step):
                    part_inner = min(step, inner - inner0)
                    for item, (col0, part_cols) in enumerate(
                            zip(group_starts, group_widths)):
                        product = (
                            lhs[row0:row0 + part_rows,
                                inner0:inner0 + part_inner].astype(np.float32)
                            @ rhs[inner0:inner0 + part_inner,
                                  col0:col0 + part_cols].astype(np.float32)
                        )
                        accumulators[item] = np.asarray(
                            accumulators[item].astype(np.float32) + product, np.float16)
                for col0, part_cols, accumulator in zip(
                        group_starts, group_widths, accumulators):
                    output[row0:row0 + part_rows,
                           col0:col0 + part_cols] = accumulator
                index += count
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


def test_fused_projection_prefill_graph():
    sequence = 2
    cfg = model.REDUCED
    weights = model.make_weights(cfg, seed=1818, ws=0.04)
    inputs = {
        "x": weights["x"][:sequence],
        "Wn1": weights["Wn1"],
        "Wn2": weights["Wn2"],
        "Wq": np.concatenate([weights[f"Wq{head}"] for head in range(cfg.H)], axis=1),
        "Wk": np.concatenate([weights[f"Wk{head}"] for head in range(cfg.KV)], axis=1),
        "Wv": np.concatenate([weights[f"Wv{head}"] for head in range(cfg.KV)], axis=1),
        "Wo": np.concatenate([weights[f"Wo{head}"] for head in range(cfg.H)], axis=0),
        "Wg": weights["Wg"],
        "Wu": weights["Wu"],
        "Wd": weights["Wd"],
    }
    plan = compile_module(model.build_v3_prefill_layer_module(cfg, sequence))
    expected = plan.run(inputs, vendor=NumpyBoundarySession())
    with VendorSession() as vendor:
        actual = plan.run(inputs, vendor=vendor)
        stats = vendor.stats()
    assert np.array_equal(actual, expected)
    assert plan.summary()["ops"]["relax.matmul"] == 15
    return stats


def test_rmsnorm_large_residual_stays_finite():
    sequence = 1
    cfg = model.REDUCED
    plan = compile_module(model.build_v3_final_norm_module(cfg, sequence))
    values = np.ones((sequence, cfg.D), dtype=np.float16)
    values[0, 3] = np.float16(330.75)
    weight = np.ones((1, cfg.D), dtype=np.float16)
    expected = plan.run({"x": values, "weight": weight}, vendor=NumpyBoundarySession())
    with VendorSession() as vendor:
        actual = plan.run({"x": values, "weight": weight}, vendor=vendor)
    assert np.array_equal(actual, expected)
    assert np.all(np.isfinite(actual))


if __name__ == "__main__":
    error, stats, summary = test_reduced_prefill_relax_graph()
    fused_stats = test_fused_projection_prefill_graph()
    test_rmsnorm_large_residual_stays_finite()
    print(f"REDUCED PREFILL max_error={error} stats={stats}")
    print(f"FUSED PREFILL stats={fused_stats}")
    print(f"PLAN {summary}")
    print("ALL V3 RELAX-EXECUTOR TESTS PASSED")
