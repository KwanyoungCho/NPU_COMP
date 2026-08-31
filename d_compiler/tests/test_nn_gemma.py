"""S7: the Gemma 4 E2B frontend against the recorded HF per-layer states.

``build/gemma4_layer_reference_hello.npz`` stores HF's hidden state entering
each layer, so a truncated model is checked against the state entering the
next one -- which pins the layer body exactly, rather than only the final
logits.  Needs the checkpoint; skips without it.
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

REFERENCE = HERE.parents[0] / "build" / "gemma4_layer_reference_hello.npz"


def _assets():
    from npu_compiler.gemma4_model import Gemma4Assets, ModelAssetError
    try:
        return Gemma4Assets()
    except (ModelAssetError, FileNotFoundError) as error:
        print(f"  [SKIP] Gemma assets unavailable: {error}")
        return None


def _hidden_after(layers):
    """-> (ours, HF's) hidden state after ``layers`` layers."""
    from npu_compiler.nn_models import gemma

    assets = _assets()
    if assets is None or not REFERENCE.exists():
        return None, None
    spec = dataclasses.replace(assets.spec, layers=assets.spec.layers[:layers])
    reference = np.load(REFERENCE)
    seq = int(reference["input_ids"].size)
    mod, params, _ = gemma.build_prefill(spec, seq)
    values = gemma.load_params(assets, params)
    inputs = gemma.host_inputs(spec, np.arange(seq))
    ple = np.ascontiguousarray(
        reference["per_layer_inputs"].transpose(1, 0, 2))[:layers]

    vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
    got = vm["hidden"](
        tvm.nd.array(reference["inputs_embeds"]), tvm.nd.array(ple),
        *[tvm.nd.array(inputs[name]) for name in
          ("cos_sliding", "sin_sliding", "cos_full", "sin_full",
           "mask_sliding", "mask_full")],
        [tvm.nd.array(v) for v in values]).numpy()
    # hidden_NN is the state entering layer NN, so after N layers it is N
    return got, reference[f"hidden_{layers:02d}"]


def test_first_layer_matches_hf():
    got, want = _hidden_after(1)
    if got is None:
        return
    a, b = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine > 0.9999, cosine
    print(f"  [PASS] gemma layer 0 vs HF: cosine {cosine:.6f}, "
          f"max|diff| {float(np.abs(a - b).max()):.4f}")


def test_parameters_cover_the_shared_kv_layers():
    """Shared layers must not carry K/V projections, and owners must."""
    from npu_compiler.nn_models import gemma

    assets = _assets()
    if assets is None:
        return
    spec = assets.spec
    _, params, _ = gemma.build_prefill(spec, seq=4)
    names = {name for name, _ in params}
    owners = [layer.index for layer in spec.layers if layer.owns_cache]
    shared = [layer.index for layer in spec.layers if not layer.owns_cache]
    assert shared, "this checkpoint should have shared-KV layers"
    for index in owners:
        assert f"layers.{index}.self_attn.k_proj.weight" in names, index
    for index in shared:
        assert f"layers.{index}.self_attn.k_proj.weight" not in names, index
        assert f"layers.{index}.self_attn.q_proj.weight" in names, index
    print(f"  [PASS] {len(owners)} owner layers carry K/V, "
          f"{len(shared)} shared layers do not")


if __name__ == "__main__":
    test_parameters_cover_the_shared_kv_layers()
    test_first_layer_matches_hf()
    print("ALL GEMMA FRONTEND TESTS PASSED")
