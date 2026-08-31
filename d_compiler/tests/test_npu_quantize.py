"""S8: W8A16 as a Relax pass.

Checks three things: the rewrite fires on parameter weights and only those,
``LiftTransformParams`` hoists the packing by itself so nothing runs per
token, and the result matches a numpy mirror of the same arithmetic.
"""
import sys
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import npu_legalize, npu_quantize
from npu_compiler import tvm_pipeline as P
from npu_compiler.nn_models import llama
from test_nn_frontend import _reference

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0)
SEQ = 4


def _model():
    mod, params, cfg = llama.build_prefill(TINY, seq=SEQ)
    rng = np.random.default_rng(5)
    weights = {name: rng.normal(0, 0.15, p.shape).astype(np.float16)
               for name, p in params}
    x = rng.normal(0, 0.5, (SEQ, cfg.hidden_size)).astype(np.float16)
    cos, sin = llama.rope_inputs(cfg, np.arange(SEQ))
    mask = llama.causal_mask(cfg.num_heads, SEQ)
    return mod, params, cfg, weights, (x, cos, sin, mask)


def _run(mod, params, weights, inputs):
    lowered = P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                               fuse=False, lift_params=True)(mod)
    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    transformed = vm["prefill_transform_params"](
        [[tvm.nd.array(weights[name]) for name, _ in params]])
    out = vm["prefill"](*[tvm.nd.array(v) for v in inputs], *transformed).numpy()
    return out, transformed


def test_quantized_model_matches_float32_and_lifts_the_packing():
    mod, params, cfg, weights, inputs = _model()
    exact = _reference(cfg, weights, *inputs)
    dense, dense_params = _run(mod, params, weights, inputs)
    quantized_mod = npu_quantize.QuantizeWeightsW8A16()(mod)
    lean, lean_params = _run(quantized_mod, params, weights, inputs)

    def score(value):
        a = value.astype(np.float64).ravel()
        b = exact.astype(np.float64).ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    dtypes = {str(t.dtype) for t in lean_params}
    assert "int8" in dtypes and "float32" in dtypes, dtypes
    assert len(lean_params) > len(dense_params)      # each weight gained a scale
    assert score(lean) > 0.9999, score(lean)
    assert int(np.argmax(lean[-1])) == int(np.argmax(exact[-1]))
    print(f"  [PASS] quantized vs float32: cosine {score(lean):.6f} "
          f"(dense {score(dense):.6f}); {len(dense_params)} -> "
          f"{len(lean_params)} lifted params, dtypes {sorted(dtypes)}")


def test_only_parameter_weights_are_quantized():
    """A matmul of two runtime inputs must be left alone."""
    bb = relax.BlockBuilder()
    left = relax.Var("a", relax.TensorStructInfo([8, 128], "float16"))
    right = relax.Var("b", relax.TensorStructInfo([128, 256], "float16"))
    with bb.function("prefill", [left, right]):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(relax.op.matmul(left, right)))
        bb.emit_func_output(out)
    module = bb.finalize()
    module["prefill"] = module["prefill"].with_attr("num_input", 2)
    rewritten = npu_quantize.QuantizeWeightsW8A16()(module)
    names = [gv.name_hint for gv in rewritten.functions]
    assert not any("quantize" in name for name in names), names
    print("  [PASS] a matmul of two runtime inputs stays dense")


def test_op_matches_the_numpy_mirror():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.4, (8, 128)).astype(np.float16)
    weight = rng.normal(0, 0.2, (128, 256)).astype(np.float16)
    bb = relax.BlockBuilder()
    left = relax.Var("a", relax.TensorStructInfo([8, 128], "float16"))
    right = relax.Var("w", relax.TensorStructInfo([128, 256], "float16"))
    with bb.function("prefill", [left, right]):
        with bb.dataflow():
            out = bb.emit_output(bb.emit(relax.op.matmul(left, right)))
        bb.emit_func_output(out)
    module = bb.finalize()
    module["prefill"] = module["prefill"].with_attr("num_input", 1)
    rewritten = npu_quantize.QuantizeWeightsW8A16()(module)
    vm = relax.VirtualMachine(relax.build(rewritten, "llvm"), tvm.cpu())
    got = vm["prefill"](tvm.nd.array(x), tvm.nd.array(weight)).numpy()
    reference = npu_quantize.reference_w8a16(x, weight)
    error = float(np.abs(got.astype(np.float32)
                         - reference.astype(np.float32)).max())
    assert error < 0.01, error
    print(f"  [PASS] qmatmul vs numpy mirror: max|diff| {error:.5f}")


if __name__ == "__main__":
    test_only_parameter_weights_are_quantized()
    test_op_matches_the_numpy_mirror()
    test_quantized_model_matches_float32_and_lifts_the_packing()
    print("ALL QUANTIZE (S8) TESTS PASSED")
