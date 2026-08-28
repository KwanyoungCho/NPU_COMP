"""S1 gate: the standard lowering pipeline.

The explicitly composed pipeline (LegalizeOps -> AnnotateTIROpPattern ->
FuseOps -> FuseTIR) must produce results identical to TVM's stock build, so
that inserting NPU-specific stages later starts from a verified baseline.
"""
import sys
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import tvm_pipeline as P
from npu_compiler.nn_models import llama

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0)
SEQ = 5


def _inputs(cfg, params):
    rng = np.random.default_rng(0)
    weights = {n: rng.normal(0, 0.15, p.shape).astype(np.float16) for n, p in params}
    x = rng.normal(0, 0.5, (SEQ, cfg.hidden_size)).astype(np.float16)
    cos, sin = llama.rope_inputs(cfg, np.arange(SEQ))
    mask = llama.causal_mask(cfg.num_heads, SEQ)
    return ([tvm.nd.array(v) for v in (x, cos, sin, mask)],
            [tvm.nd.array(weights[n]) for n, _ in params])


def _run(executable, args, weights):
    return relax.VirtualMachine(executable, tvm.cpu())["prefill"](*args, weights).numpy()


def test_pipeline_matches_stock_build():
    mod, params, cfg = llama.build_prefill(TINY, seq=SEQ)
    args, weights = _inputs(cfg, params)
    stock = _run(relax.build(mod, target="llvm"), args, weights)

    counts = {}
    for fuse in (False, True):
        # lift_params changes the calling convention; it is covered separately
        lowered = P.graph_pipeline(fuse=fuse, lift_params=False)(mod)
        counts[fuse] = P.kernel_summary(lowered)["prim_funcs"]
        got = _run(relax.build(lowered, target="llvm"), args, weights)
        np.testing.assert_array_equal(stock.view(np.uint16), got.view(np.uint16))
    assert counts[True] < counts[False], counts
    print(f"  [PASS] pipeline == stock build; PrimFuncs "
          f"{counts[False]} -> {counts[True]} with fusion")


def test_fusion_produces_expected_kernels():
    """Fusion must actually merge the LLM-shaped chains, not just rename."""
    mod, _, _ = llama.build_prefill(TINY, seq=SEQ)
    names = P.kernel_summary(P.graph_pipeline(lift_params=False)(mod))["names"]
    joined = " ".join(names)
    assert any("silu" in n and "matmul" in n for n in names), \
        f"SwiGLU chain not fused: {names}"
    assert any("softmax" in n for n in names), joined
    assert any(n.count("multiply") >= 2 and "concatenate" in n for n in names), \
        f"RoPE chain not fused: {names}"
    print(f"  [PASS] fused kernels include SwiGLU / softmax / RoPE chains "
          f"({len(names)} kernels)")


def test_lift_transform_params_is_value_preserving_and_shrinks_memory():
    """nn.Linear transposes its weight; without LiftTransformParams that
    transpose is materialized on every call (751 MiB for Llama's lm_head).
    The lifted form must compute the same values."""
    from npu_compiler import npu_memplan as M
    mod, params, cfg = llama.build_prefill(TINY, seq=SEQ)
    args, weights = _inputs(cfg, params)

    plain = P.graph_pipeline(lift_params=False)(mod)
    base = _run(relax.build(plain, target="llvm"), args, weights)

    lifted = P.graph_pipeline(lift_params=True)(mod)
    vm = relax.VirtualMachine(relax.build(lifted, target="llvm"), tvm.cpu())
    transformed = vm["prefill_transform_params"]([weights])
    got = vm["prefill"](*args, *transformed).numpy()
    np.testing.assert_array_equal(base.view(np.uint16), got.view(np.uint16))

    pools = lambda m: sum(M.assign_addresses(m)[1].pool_bytes.values())
    before, after = pools(plain), pools(lifted)
    assert after < before, (before, after)
    print(f"  [PASS] LiftTransformParams value-preserving; activation pools "
          f"{before:,} -> {after:,} B")


def test_layers_share_kernels():
    """Identical layers must collapse onto one PrimFunc each, so kernel count
    does not grow with depth (this is what makes whole-model codegen small)."""
    two = P.kernel_summary(P.graph_pipeline()(
        llama.build_prefill(dict(TINY, num_hidden_layers=2), seq=SEQ)[0]))
    four = P.kernel_summary(P.graph_pipeline()(
        llama.build_prefill(dict(TINY, num_hidden_layers=4), seq=SEQ)[0]))
    assert four["prim_funcs"] == two["prim_funcs"], (two, four)
    print(f"  [PASS] kernel count independent of depth "
          f"({two['prim_funcs']} for both 2 and 4 layers)")


if __name__ == "__main__":
    test_pipeline_matches_stock_build()
    test_fusion_produces_expected_kernels()
    test_lift_transform_params_is_value_preserving_and_shrinks_memory()
    test_layers_share_kernels()
    print("ALL TVM PIPELINE (S1) TESTS PASSED")
