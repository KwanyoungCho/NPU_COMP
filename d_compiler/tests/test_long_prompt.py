"""Numerical accuracy at prompt lengths past one 64-row tile.

Every other gate runs at seq=7, where the machine reads each weight exactly
once and every kernel's staging fits SRAM with room to spare.  None of the
memory-planning work is visible there -- neither its gain nor its
regressions -- so this links a real-dimension layer at increasing prompt
lengths and scores it.

The reference is float32 numpy, not the llvm build: llvm accumulates fp16
matmuls in fp16 while the machine accumulates in fp32, so at K=3072..8192 it
is llvm that drifts (0.998 against us), and it cannot tell a scheduling bug
from its own rounding.  Running it at these dimensions also costs minutes,
because the llvm build executes the unscheduled loop nest.

Lengths the linker refuses are reported, not failed.  That limit -- kernels
that stage a whole tensor -- is what the stripe work removes, and this test
starts checking those lengths by itself once it does.
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))

import tvm
from tvm import relax

from npu_compiler import model, npu_legalize, npu_link, npu_memplan
from npu_compiler import tvm_pipeline as pipeline
from npu_compiler.nn_models import llama
from test_nn_frontend import _reference

SEQS = (64, 128, 256, 512, 1024)
# measured 0.999997 at seq 64 and 0.999996 at 128 -- the machine accumulates
# in fp32, so only the fp16 activations cost anything.  A staging or tiling
# error moves this far below the threshold rather than nudging it.
COSINE = 0.9999


def _config():
    """One real Llama 3.2 3B layer.  The vocabulary is shrunk so lm_head does
    not dominate the score -- the layer is what the planning work changes."""
    cfg = model.LLAMA_3_2_3B
    return dict(hidden_size=cfg.D, intermediate_size=cfg.F,
                num_hidden_layers=1, num_attention_heads=cfg.H,
                num_key_value_heads=cfg.KV, head_dim=cfg.HD,
                vocab_size=256, rms_norm_eps=1e-5, rope_theta=500000.0)


def _measure(seq):
    """-> (cosine, words, seconds), or None when the linker refuses this
    length."""
    started = time.perf_counter()
    mod, params, cfg = llama.build_prefill(_config(), seq=seq)
    rng = np.random.default_rng(5)
    weights = {name: rng.normal(0, 0.15, p.shape).astype(np.float16)
               for name, p in params}
    x = rng.normal(0, 0.5, (seq, cfg.hidden_size)).astype(np.float16)
    cos, sin = llama.rope_inputs(cfg, np.arange(seq))
    mask = llama.causal_mask(cfg.num_heads, seq)

    lowered = pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=True)(mod)
    try:
        asm, plan = npu_link.compile_program(lowered)
    except npu_link.LinkError as error:
        return None, str(error)

    # llvm only runs the lifted weight transform, which is transposes; the
    # transform is expressed as IR, so running that IR is the only way to get
    # exactly the values the image expects
    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    transformed = vm["prefill_transform_params"](
        [[tvm.nd.array(weights[name]) for name, _ in params]])

    planned, _ = npu_memplan.assign_addresses(lowered)
    func = planned["prefill"]
    values = dict(zip([p.name_hint for p in func.params],
                      [x, cos, sin, mask] + [t.numpy() for t in transformed]))
    reference = _reference(cfg, weights, x, cos, sin, mask)
    got, _ = npu_link.run_program(asm, plan, func, values, reference.shape)

    a = got.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return (cosine, len(asm.words), time.perf_counter() - started), None


def test_long_prompts_stay_accurate():
    linked, refused = [], []
    for seq in SEQS:
        result, error = _measure(seq)
        if result is None:
            refused.append((seq, error))
            print(f"  [LIMIT] seq {seq:4d}: does not link -- {error}")
            continue
        cosine, words, seconds = result
        linked.append(seq)
        assert cosine > COSINE, f"seq {seq}: cosine {cosine:.6f}"
        print(f"  [PASS]  seq {seq:4d}: cosine {cosine:.6f}, "
              f"{words:,} words, {seconds:.0f}s")
    assert linked, "no prompt length linked at all"
    if refused:
        print(f"  longest linkable prompt is {max(linked)}; "
              f"{len(refused)} longer length(s) still refused")


if __name__ == "__main__":
    test_long_prompts_stay_accurate()
    print("ALL LONG PROMPT TESTS PASSED")
