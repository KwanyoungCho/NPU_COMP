"""Official Llama 3.2 3B asset validation and slice-loading smoke tests."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.v3_model import Llama32Assets, ModelAssetError, default_model_path


def _assets_or_skip():
    try:
        return Llama32Assets()
    except ModelAssetError as error:
        print(f"SKIP: {error}")
        return None


def test_config_keyset_and_tokenizer():
    assets = _assets_or_skip()
    if assets is None:
        return
    assert assets.validate_keyset() == 254
    text = "Hello, NPU compiler!"
    ids = assets.tokenizer.encode(text, add_special_tokens=True)
    assert ids == [128000, 9906, 11, 452, 6459, 19979, 0]
    assert assets.tokenizer.decode(ids) == "<|begin_of_text|>" + text


def test_streamed_weight_views():
    assets = _assets_or_skip()
    if assets is None:
        return
    embeddings = assets.embedding([128000, 9906])
    assert embeddings.shape == (2, 3072) and embeddings.dtype == np.float16
    q_tile = assets.linear_tile(0, "self_attn.q_proj", slice(64, 128), slice(128, 192))
    assert q_tile.shape == (64, 64) and q_tile.dtype == np.float16
    lm_tile = assets.lm_head_tile(slice(64, 128), slice(128, 192))
    assert lm_tile.shape == (64, 64) and lm_tile.dtype == np.float16
    norm = assets.norm(27, post_attention=True)
    assert norm.shape == (3072,) and norm.dtype == np.float16


if __name__ == "__main__":
    test_config_keyset_and_tokenizer()
    test_streamed_weight_views()
    print(f"ALL V3 MODEL-ASSET TESTS PASSED ({default_model_path()})")
