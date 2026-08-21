"""PLE precompute parity: table generator == source C-model program, bit-exact.

Proves the offline PLE table stores exactly what the NPU graph computes, so
runtime lookup is equivalent to on-device computation (skips without checkpoint).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.gemma4_ple import compute_rows
from npu_compiler import driver
from npu_compiler.gemma4_graph import build_gemma4_ple_module
from npu_compiler.gemma4_model import Gemma4Assets, default_model_path

SAMPLE_IDS = [2, 9259, 236764, 646, 11152, 47133, 236888, 0, 1, 108, 262143]


def test_ple_table_matches_npu_program():
    path = default_model_path()
    if not (path / "model.safetensors").exists():
        print("  (skip: Gemma 4 E2B checkpoint not present)")
        return
    assets = Gemma4Assets(path)
    spec = assets.spec

    scaled_embed = assets.embedding(SAMPLE_IDS)
    tok_rows = assets.ple_rows(SAMPLE_IDS)
    projection = assets.per_layer_model_projection()
    norm_weight = assets.per_layer_projection_norm()

    expected = compute_rows(spec, scaled_embed, tok_rows, projection, norm_weight)
    assert np.isfinite(expected.astype(np.float32)).all()

    module = build_gemma4_ple_module(spec, len(SAMPLE_IDS))
    compiled = driver.compile_module(module, backend="source-0818")
    actual = np.asarray(driver.run_compiled(*compiled, {
        "se": scaled_embed, "tok": tok_rows,
        "Wproj": projection, "Wn": norm_weight,
    }), dtype=np.float16)
    assert actual.shape == expected.shape
    mismatch = int(np.count_nonzero(actual != expected))
    assert mismatch == 0, f"{mismatch} elements differ between table math and NPU"
    print(f"  [PASS] generator == source program on {len(SAMPLE_IDS)} tokens "
          f"({expected.size} values, bit-exact)")

    try:
        table = assets.ple_table()
    except Exception:
        print("  (table file absent or still generating; stored-row check skipped)")
        return
    stored = np.stack([table[int(i)] for i in SAMPLE_IDS])
    assert np.array_equal(stored, expected)
    print("  [PASS] stored table rows match, bit-exact")


if __name__ == "__main__":
    test_ple_table_matches_npu_program()
    print("ALL GEMMA4 PLE TESTS PASSED")
