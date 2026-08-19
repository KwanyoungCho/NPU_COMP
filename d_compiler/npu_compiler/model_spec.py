"""Model-independent structural specs for source-0818 generation (Stage G4.1).

These types describe what the orchestration and cache layers need to know about
a model family — layer count, attention geometry, RoPE parameters, and KV-cache
ownership — without encoding family-specific graph semantics.  Norm ordering,
PLE arithmetic, and activation lowering stay in the family graph builders.

KV sharing (Gemma 4) is expressed through ``LayerSpec.kv_owner``: a shared
layer names an earlier owner layer of the same attention kind and
``build_cache_plan`` turns that into slot aliases, never cache copies.
"""
from __future__ import annotations

from dataclasses import dataclass, field


ATTENTION_KINDS = ("full", "sliding")


@dataclass(frozen=True)
class AttentionSpec:
    """Geometry and RoPE contract of one attention block."""

    kind: str
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    window: int | None = None
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    llama3_rope_scaling: bool = False
    # QK-Norm models normalize Q/K and use score scale 1; otherwise the score
    # scale is 1/sqrt(head_dim).
    qk_norm: bool = False

    def __post_init__(self):
        if self.kind not in ATTENTION_KINDS:
            raise ValueError(f"unknown attention kind {self.kind!r}")
        if self.kind == "sliding" and not self.window:
            raise ValueError("sliding attention requires a positive window")
        if self.kind == "full" and self.window is not None:
            raise ValueError("full attention must not set a window")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be a multiple of KV heads")


@dataclass(frozen=True)
class LayerSpec:
    """One decoder layer: attention geometry, FFN width, and KV ownership."""

    index: int
    attention: AttentionSpec
    ffn_hidden: int
    activation: str
    kv_owner: int
    ple_dim: int = 0

    @property
    def owns_cache(self):
        return self.kv_owner == self.index


@dataclass(frozen=True)
class ModelSpec:
    """Whole-model structure consumed by generation orchestration."""

    name: str
    hidden_size: int
    vocab_size: int
    rms_norm_eps: float
    layers: tuple
    tie_word_embeddings: bool = True
    scale_embeddings: bool = False
    final_logit_softcapping: float | None = None

    @property
    def num_layers(self):
        return len(self.layers)

    def __post_init__(self):
        for position, layer in enumerate(self.layers):
            if layer.index != position:
                raise ValueError(f"layer {position} has index {layer.index}")


@dataclass(frozen=True)
class CacheSlot:
    """One physical K/V cache allocation and the layers that alias it."""

    owner_layer: int
    kind: str
    num_kv_heads: int
    head_dim: int
    window: int | None
    shared_layer_ids: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class CachePlan:
    """Slots in owner order plus the layer-index -> slot-position mapping."""

    slots: tuple
    layer_to_slot: tuple

    def slot_for(self, layer):
        return self.slots[self.layer_to_slot[layer]]

    def is_owner(self, layer):
        return self.slot_for(layer).owner_layer == layer


def build_cache_plan(spec):
    """Derive cache slots from ``kv_owner``; sharing is aliasing, not copying."""
    slot_index = {}
    shared = {}
    slots = []
    mapping = []
    for layer in spec.layers:
        attention = layer.attention
        if layer.owns_cache:
            slot_index[layer.index] = len(slots)
            shared[layer.index] = []
            slots.append(CacheSlot(
                layer.index, attention.kind, attention.num_kv_heads,
                attention.head_dim, attention.window))
            mapping.append(slot_index[layer.index])
            continue
        if layer.kv_owner not in slot_index:
            raise ValueError(
                f"layer {layer.index} shares KV with {layer.kv_owner}, "
                "which is not an earlier owner layer")
        owner = spec.layers[layer.kv_owner]
        if owner.attention.kind != attention.kind:
            raise ValueError(
                f"layer {layer.index} ({attention.kind}) cannot share KV with "
                f"layer {owner.index} ({owner.attention.kind})")
        shared[layer.kv_owner].append(layer.index)
        mapping.append(slot_index[layer.kv_owner])
    slots = [
        CacheSlot(slot.owner_layer, slot.kind, slot.num_kv_heads,
                  slot.head_dim, slot.window,
                  tuple(shared[slot.owner_layer]))
        for slot in slots
    ]
    return CachePlan(tuple(slots), tuple(mapping))


def llama32_spec(config):
    """Spec for official Llama 3.2 3B from its validated HF config dict."""
    attention = AttentionSpec(
        kind="full",
        num_query_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        rope_theta=float(config.get("rope_theta", 500000.0)),
        llama3_rope_scaling=bool(config.get("rope_scaling")),
    )
    layers = tuple(
        LayerSpec(index, attention, config["intermediate_size"], "silu", index)
        for index in range(config["num_hidden_layers"])
    )
    return ModelSpec(
        name="llama-3.2-3B",
        hidden_size=config["hidden_size"],
        vocab_size=config["vocab_size"],
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-5)),
        layers=layers,
        tie_word_embeddings=bool(config.get("tie_word_embeddings", True)),
    )


# Official Gemma 4 E2B ``text_config`` subset, verified against the checkpoint's
# config.json (google/gemma-4-E2B) and its safetensors tensor inventory.
GEMMA4_E2B_TEXT = {
    "hidden_size": 1536,
    "intermediate_size": 6144,
    "num_hidden_layers": 35,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "vocab_size": 262144,
    "rms_norm_eps": 1e-6,
    "sliding_window": 512,
    "num_kv_shared_layers": 20,
    "hidden_size_per_layer_input": 256,
    "final_logit_softcapping": 30.0,
    "hidden_activation": "gelu_pytorch_tanh",
    "use_double_wide_mlp": True,
    "tie_word_embeddings": True,
    "layer_types": [
        "full_attention" if index % 5 == 4 else "sliding_attention"
        for index in range(35)
    ],
    "rope_parameters": {
        "full_attention": {
            "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_theta": 10000.0},
    },
}


def gemma4_e2b_spec(config=None):
    """Spec for official Gemma 4 E2B text from its HF ``text_config`` dict.

    Layer kinds come from ``layer_types``; the last ``num_kv_shared_layers``
    layers alias the latest owner slot of their own attention kind and use the
    double-wide (2x ``intermediate_size``) MLP, matching the checkpoint's
    per-layer gate/up shapes.
    """
    config = config or GEMMA4_E2B_TEXT
    if config["hidden_activation"] != "gelu_pytorch_tanh":
        raise ValueError(f"unexpected activation {config['hidden_activation']}")
    rope = config["rope_parameters"]
    sliding = AttentionSpec(
        kind="sliding",
        num_query_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        window=config["sliding_window"],
        rope_theta=rope["sliding_attention"]["rope_theta"],
        qk_norm=True,
    )
    full = AttentionSpec(
        kind="full",
        num_query_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"],
        head_dim=config["global_head_dim"],
        rope_theta=rope["full_attention"]["rope_theta"],
        partial_rotary_factor=rope["full_attention"]["partial_rotary_factor"],
        qk_norm=True,
    )
    kinds = ["full" if name == "full_attention" else "sliding"
             for name in config["layer_types"]]
    if len(kinds) != config["num_hidden_layers"]:
        raise ValueError("layer_types does not cover num_hidden_layers")
    first_shared = config["num_hidden_layers"] - config["num_kv_shared_layers"]
    double_wide = bool(config.get("use_double_wide_mlp"))
    last_owner = {}
    layers = []
    for index, kind in enumerate(kinds):
        if index < first_shared:
            owner = index
            last_owner[kind] = index
        else:
            if kind not in last_owner:
                raise ValueError(f"no earlier {kind} owner for shared layer {index}")
            owner = last_owner[kind]
        shared = index >= first_shared
        layers.append(LayerSpec(
            index,
            sliding if kind == "sliding" else full,
            config["intermediate_size"] * (2 if double_wide and shared else 1),
            "gelu_tanh",
            owner,
            ple_dim=config["hidden_size_per_layer_input"],
        ))
    return ModelSpec(
        name="gemma-4-E2B",
        hidden_size=config["hidden_size"],
        vocab_size=config["vocab_size"],
        rms_norm_eps=float(config["rms_norm_eps"]),
        layers=tuple(layers),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", True)),
        scale_embeddings=True,
        final_logit_softcapping=config["final_logit_softcapping"],
    )
