"""Model-independent spec/cache-plan regression (Stage G4.1)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler.model_spec import (
    AttentionSpec, LayerSpec, ModelSpec,
    build_cache_plan, gemma4_e2b_spec, llama32_spec,
)
from npu_compiler.v3_model import Llama32Assets


LLAMA_CONFIG = dict(Llama32Assets.EXPECTED)
LLAMA_CONFIG.update({
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "rope_scaling": {"rope_type": "llama3"},
})


def test_llama32_spec():
    spec = llama32_spec(LLAMA_CONFIG)
    assert spec.num_layers == 28
    assert spec.hidden_size == 3072
    assert spec.vocab_size == 128256
    assert spec.tie_word_embeddings
    assert not spec.scale_embeddings
    assert spec.final_logit_softcapping is None
    for layer in spec.layers:
        attention = layer.attention
        assert attention.kind == "full"
        assert attention.num_query_heads == 24
        assert attention.num_kv_heads == 8
        assert attention.head_dim == 128
        assert attention.window is None
        assert attention.rope_theta == 500000.0
        assert attention.llama3_rope_scaling
        assert not attention.qk_norm
        assert layer.activation == "silu"
        assert layer.ffn_hidden == 8192
        assert layer.owns_cache
        assert layer.ple_dim == 0

    plan = build_cache_plan(spec)
    assert len(plan.slots) == 28
    assert plan.layer_to_slot == tuple(range(28))
    for layer in range(spec.num_layers):
        assert plan.is_owner(layer)
        slot = plan.slot_for(layer)
        assert slot.owner_layer == layer
        assert slot.shared_layer_ids == ()


def test_gemma4_e2b_spec():
    spec = gemma4_e2b_spec()
    assert spec.num_layers == 35
    assert spec.hidden_size == 1536
    assert spec.vocab_size == 262144
    assert spec.scale_embeddings
    assert spec.final_logit_softcapping == 30.0

    kinds = [layer.attention.kind for layer in spec.layers]
    assert kinds.count("full") == 7
    assert kinds.count("sliding") == 28
    for index, kind in enumerate(kinds):
        assert kind == ("full" if index % 5 == 4 else "sliding")

    for layer in spec.layers:
        attention = layer.attention
        assert attention.num_query_heads == 8
        assert attention.num_kv_heads == 1
        assert attention.qk_norm
        assert layer.activation == "gelu_tanh"
        # Checkpoint fact: the shared-KV last 20 layers use the double-wide MLP.
        assert layer.ffn_hidden == (12288 if layer.index >= 15 else 6144)
        assert layer.ple_dim == 256
        if attention.kind == "sliding":
            assert attention.head_dim == 256
            assert attention.window == 512
            assert attention.rope_theta == 10000.0
            assert attention.partial_rotary_factor == 1.0
        else:
            assert attention.head_dim == 512
            assert attention.window is None
            assert attention.rope_theta == 1000000.0
            assert attention.partial_rotary_factor == 0.25

    # First 15 layers own their slot; the shared 20 alias the latest owner of
    # their own attention kind (sliding -> layer 13, full -> layer 14).
    for layer in spec.layers[:15]:
        assert layer.owns_cache
    for layer in spec.layers[15:]:
        assert not layer.owns_cache
        assert layer.kv_owner == (14 if layer.attention.kind == "full" else 13)

    plan = build_cache_plan(spec)
    assert len(plan.slots) == 15
    shared_sliding = plan.slot_for(13).shared_layer_ids
    shared_full = plan.slot_for(14).shared_layer_ids
    assert len(shared_sliding) + len(shared_full) == 20
    assert all(kinds[layer] == "sliding" for layer in shared_sliding)
    assert all(kinds[layer] == "full" for layer in shared_full)
    for layer in range(15, 35):
        slot = plan.slot_for(layer)
        assert not plan.is_owner(layer)
        assert slot.kind == kinds[layer]
        assert layer in slot.shared_layer_ids
    for layer in range(15):
        assert plan.slot_for(layer).owner_layer == layer


def test_gemma4_spec_matches_downloaded_config():
    """When the official config.json is on disk, the built-in defaults must
    produce the identical spec."""
    import json
    path = os.path.join(
        ROOT, "d_compiler", "build", "gemma4_e2b_hf", "config.json")
    if not os.path.exists(path):
        print("  (skip: gemma4_e2b_hf/config.json not downloaded)")
        return
    with open(path) as file:
        text_config = json.load(file)["text_config"]
    assert gemma4_e2b_spec(text_config) == gemma4_e2b_spec()


def test_invalid_specs():
    full = AttentionSpec("full", 8, 1, 64)
    sliding = AttentionSpec("sliding", 8, 1, 64, window=128)
    try:
        AttentionSpec("sliding", 8, 1, 64)
        raise AssertionError("sliding without window must fail")
    except ValueError:
        pass
    try:
        AttentionSpec("full", 8, 1, 64, window=128)
        raise AssertionError("full with window must fail")
    except ValueError:
        pass
    try:
        AttentionSpec("full", 7, 2, 64)
        raise AssertionError("non-multiple GQA must fail")
    except ValueError:
        pass

    # Sharing must name an earlier owner of the same attention kind.
    mixed = ModelSpec("bad-owner", 64, 128, 1e-5, (
        LayerSpec(0, full, 64, "silu", 0),
        LayerSpec(1, sliding, 64, "silu", 0),
    ))
    try:
        build_cache_plan(mixed)
        raise AssertionError("cross-kind sharing must fail")
    except ValueError:
        pass
    forward = ModelSpec("bad-forward", 64, 128, 1e-5, (
        LayerSpec(0, full, 64, "silu", 1),
        LayerSpec(1, full, 64, "silu", 1),
    ))
    try:
        build_cache_plan(forward)
        raise AssertionError("forward owner reference must fail")
    except ValueError:
        pass


if __name__ == "__main__":
    test_llama32_spec()
    test_gemma4_e2b_spec()
    test_gemma4_spec_matches_downloaded_config()
    test_invalid_specs()
    print("ALL MODEL SPEC TESTS PASSED")
