"""Slow official-checkpoint layer test (enable with NPU_RUN_SLOW_OFFICIAL=1)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.v3_llama import Llama32PrefillCompiler
from npu_compiler.v3_runtime import ParallelVendorSession


def test_official_layer_zero():
    if os.environ.get("NPU_RUN_SLOW_OFFICIAL") != "1":
        print("SKIP: set NPU_RUN_SLOW_OFFICIAL=1 for the ~70 second vendor test")
        return
    reference_path = os.path.join(ROOT, "d_compiler", "build", "v3_reference_hello.npz")
    if not os.path.exists(reference_path):
        print(f"SKIP: independent reference is absent: {reference_path}")
        return
    reference = np.load(reference_path)
    compiler = Llama32PrefillCompiler(len(reference["input_ids"]))
    hidden = compiler.assets.embedding(reference["input_ids"])
    with ParallelVendorSession(4) as vendor:
        actual = compiler.run_layer(hidden, 0, vendor)
    expected = reference["hidden_states"][1]
    difference = actual.astype(np.float32) - expected.astype(np.float32)
    flat_a = actual.reshape(-1).astype(np.float64)
    flat_b = expected.reshape(-1).astype(np.float64)
    cosine = float(np.dot(flat_a, flat_b) /
                   (np.linalg.norm(flat_a) * np.linalg.norm(flat_b)))
    assert float(np.mean(np.abs(difference))) < 0.001
    assert cosine > 0.9999


if __name__ == "__main__":
    test_official_layer_zero()
    print("V3 OFFICIAL LAYER TEST PASSED")
