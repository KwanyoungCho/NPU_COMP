"""The host parameter transform, cached on disk.

A wrong cache is silent: the program links and runs, it just reads the wrong
weights.  So the gates are (1) what comes back from disk is bit-identical to
what the transform computes, and (2) a changed graph misses rather than
reusing an entry that no longer describes it.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

import tvm

from npu_compiler import npu_generate
from npu_compiler.nn_models import llama

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0)


def _model(config, seed):
    mod, params, _ = llama.build_generate(config, 5, 6)
    rng = np.random.default_rng(seed)
    weights = [tvm.nd.array(rng.normal(0, 0.15, p.shape).astype("float16"))
               for _, p in params]
    return npu_generate._lower(mod), weights


def test_cache_round_trips_and_keys_on_the_graph():
    cache = Path(tempfile.mkdtemp(prefix="npu_params_"))
    try:
        lowered, weights = _model(TINY, seed=0)
        direct = npu_generate._transform(lowered, "decode", weights)
        npu_generate._transform(lowered, "decode", weights, cache)   # fills
        reread = npu_generate._transform(lowered, "decode", weights, cache)

        assert len(reread) == len(direct), (len(reread), len(direct))
        for index, (computed, cached) in enumerate(zip(direct, reread)):
            assert computed.dtype == cached.dtype, index
            assert computed.shape == cached.shape, index
            assert np.array_equal(computed, np.asarray(cached)), \
                f"tensor {index} came back from disk changed"
        print(f"  [PASS] {len(reread)} tensors bit-identical from disk")

        entries = {path.name for path in cache.iterdir()}
        wider, wide_weights = _model(dict(TINY, intermediate_size=256), seed=1)
        npu_generate._transform(wider, "decode", wide_weights, cache)
        assert {path.name for path in cache.iterdir()} > entries, \
            "a changed graph reused the cache entry of the old one"
        print("  [PASS] a changed graph gets its own entry")
    finally:
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    test_cache_round_trips_and_keys_on_the_graph()
    print("ALL PARAMS CACHE TESTS PASSED")
