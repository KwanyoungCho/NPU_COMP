"""Follow-up gates: bounded numerics, semantic dispatch and native failures."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tvm
from tvm import relax, tir, te

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu_compiler import native, npu_legalize, npu_link as L, npu_memplan as M
from npu_compiler import npu_quantize as Q, tvm_pipeline as P
from npu_compiler.npu_kernel import attention_kernel
from test_compiler_contracts import lower, matmul, traffic


def run(mod, values, shape, name="prefill"):
    asm, plan = L.compile_program(mod, name, native_mode="compare")
    planned, _ = M.assign_addresses(mod, name)
    got, _ = L.run_program(asm, plan, planned[name], values, shape)
    return got, asm, plan


def attention(axis=-1, key_width=64):
    bb = relax.BlockBuilder()
    q, k, v, mask = [relax.Var(name, relax.TensorStructInfo(shape, "float16"))
                     for name, shape in (("q", [2, 64, 64]),
                         ("k", [2, 64, key_width]), ("v", [2, key_width, 64]),
                         ("mask", [1, 64, key_width]))]
    with bb.function("prefill", [q, k, v, mask]):
        with bb.dataflow():
            score = bb.emit(relax.op.matmul(q, k))
            score = bb.emit(relax.op.multiply(score, relax.const(.125, "float16")))
            score = bb.emit(relax.op.add(score, mask))
            score = bb.emit(relax.op.nn.softmax(score, axis=axis))
            out = bb.emit_output(relax.op.matmul(score, v))
        bb.emit_func_output(out)
    return bb.finalize()


def rms(epsilon, width=64):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([7, width], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([width], "float16"))
    with bb.function("prefill", [x, w]):
        with bb.dataflow():
            out = bb.emit_output(relax.op.nn.rms_norm(x, w, axes=[-1], epsilon=epsilon))
        bb.emit_func_output(out)
    return bb.finalize()


class PriorityTests(unittest.TestCase):
    def test_abs_preserves_a_computed_operand_until_its_last_read(self):
        bb = relax.BlockBuilder()
        x = relax.Var("x", relax.TensorStructInfo([2,64], "float16"))
        def compound(data):
            return te.compute(data.shape, lambda i,j: te.abs(data[i,j] * tir.const(2, "float16")))
        with bb.function("prefill", [x]):
            with bb.dataflow():
                out = bb.emit_output(bb.call_te(compound, x))
            bb.emit_func_output(out)
        values = np.linspace(-3,3,128).reshape(2,64).astype("float16")
        got, _, plan = run(lower(bb.finalize()), {"x": values}, values.shape)
        np.testing.assert_array_equal(got, np.abs(values * 2))
        self.assertTrue(all(plan.native.values()), plan.native_diagnostics)

    def test_long_pointwise_row_splits_at_vlen_limit(self):
        width = 65537
        bb = relax.BlockBuilder()
        x = relax.Var("x", relax.TensorStructInfo([2,width], "float16"))
        with bb.function("prefill", [x]):
            with bb.dataflow():
                out = bb.emit_output(relax.op.add(x, x))
            bb.emit_func_output(out)
        values = (np.arange(2 * width) % 127).reshape(2,width).astype("float16")
        got, _, plan = run(lower(bb.finalize()), {"x": values}, values.shape)
        np.testing.assert_array_equal(got, values * 2)
        self.assertTrue(all(plan.native.values()), plan.native_diagnostics)

    def test_rms_dynamic_range_and_small_epsilon(self):
        rng = np.random.default_rng(3)
        for width in (64, 3072):
            x = rng.normal(size=(7, width)).astype("float16")
            x[0] = 0
            x[1] = 1000
            x[2] = 65504
            x[3] *= np.float16(1e-4)
            x[4] = 0
            x[4, 0] = 65504
            x[5] *= np.float16(1000)
            x[6] = np.float16(2**-24)
            w = np.ones(width, "float16")
            for epsilon in (1e-5, 1e-10):
                with self.subTest(width=width, epsilon=epsilon):
                    got, asm, plan = run(lower(rms(epsilon, width)), {"x": x, "w": w}, x.shape)
                    reference = x.astype("float64") / np.sqrt(
                        np.mean(x.astype("float64")**2, axis=-1, keepdims=True) + epsilon)
                    self.assertTrue(np.isfinite(got).all())
                    np.testing.assert_allclose(got, reference, rtol=.012, atol=.003)
                    self.assertTrue(all(plan.native.values()), plan.native_diagnostics)
                    print(json.dumps({"rms_width": width, "epsilon": epsilon,
                                      "words": len(asm.words),
                                      "max_abs": float(np.max(np.abs(got-reference)))}))

    def test_rms_rejects_unrepresentable_epsilon(self):
        for epsilon in (0, 1e-20, float("inf")):
            with self.assertRaisesRegex((ValueError, tvm.error.TVMError), "epsilon"):
                relax.transform.LegalizeOps(npu_legalize.legalize_map())(rms(epsilon))

    def test_attention_rename_is_semantic(self):
        mod = lower(attention())
        gv = next(g for g, f in mod.functions.items()
                  if isinstance(f, tir.PrimFunc) and attention_kernel(f))
        def rename(node):
            if isinstance(node, tir.Block) and node.name_hint != "root":
                return tir.Block(node.iter_vars, node.reads, node.writes,
                                 "opaque_" + str(names.setdefault(node.name_hint, len(names))),
                                 node.body, node.init, node.alloc_buffers,
                                 node.match_buffers, node.annotations)
            return None
        names = {}
        changed = tvm.IRModule(mod.functions)
        changed[gv] = mod[gv].with_body(tir.stmt_functor.ir_transform(
            mod[gv].body, None, rename, ["tir.Block"]))
        self.assertIsNotNone(attention_kernel(changed[gv]))
        rng = np.random.default_rng(12)
        vals = {name: rng.normal(0, .1, shape).astype("float16")
                for name, shape in (("q", (2,64,64)), ("k", (2,64,64)),
                                    ("v", (2,64,64)), ("mask", (1,64,64)))}
        got, asm, plan = run(changed, vals, (2,64,64))
        baseline, _ = L.compile_program(mod, native_mode="compare")
        np.testing.assert_array_equal(asm.words, baseline.words)
        scores = vals["q"].astype("float32") @ vals["k"].astype("float32") * .125 + vals["mask"]
        exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        expected = exp / exp.sum(axis=-1, keepdims=True) @ vals["v"].astype("float32")
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=.01)
        self.assertTrue(all(plan.native.values()), plan.native_diagnostics)

    def test_attention_pattern_rejects_wrong_axis_and_ragged_keys(self):
        for axis, width in ((0, 64), (-1, 65)):
            matched = relax.transform.FuseOpsByPattern(P._attention_patterns())(
                attention(axis, width))
            self.assertNotIn('"npu.attention', matched.script())
            if axis == 0:
                with self.assertRaisesRegex((ValueError, tvm.error.TVMError), "last axis"):
                    lower(attention(axis, width))
                continue
            mod = lower(attention(axis, width))
            self.assertFalse(any(attention_kernel(f) for f in mod.functions.values()
                                 if isinstance(f, tir.PrimFunc)))
            # The Relax pattern must decline before a compound kernel is formed.
            self.assertFalse(any("fused_relax_matmul" in g.name_hint
                                 for g in mod.functions))

    def test_fake_softmax_names_do_not_select_attention(self):
        mod = lower(attention())
        prim = next(f for f in mod.functions.values()
                    if isinstance(f, tir.PrimFunc) and attention_kernel(f))
        # Keep every original block name but destroy exp/sum normalization.
        bad = prim.with_body(tir.stmt_functor.ir_transform(
            prim.body, None, lambda expr: tir.Mul(expr.a, expr.b)
            if isinstance(expr, tir.Div) else None, ["tir.Div"]))
        self.assertIsNone(attention_kernel(bad))

    def test_hf_fusion_and_constant_quantization_match_torch(self):
        from test_nn_hf import _tiny, _reference, _cosine
        from npu_compiler.nn_models import hf
        model, cfg = _tiny()
        seq = 5
        x = np.random.default_rng(0).normal(0,.5,(1,seq,64)).astype("float16")
        mask = hf.causal_mask4d(seq)
        expected = _reference(model, x, mask)
        for fuse, quant in ((False, False), (True, False), (False, True)):
            mod = hf.import_prefill(model, seq)
            if quant:
                mod = Q.QuantizeW8A8()(mod)
                self.assertGreater(int(mod.attrs["npu.quantized_matmuls"]), 0)
            mod = P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                                   fuse=fuse, lift_params=False)(mod)
            got, asm, plan = run(mod, {"inputs_embeds": x, "attention_mask": mask},
                                 (1,seq,cfg.vocab_size), "main")
            cosine = _cosine(got, expected)
            self.assertGreater(cosine, .9999)
            self.assertEqual(int(got[0,-1].argmax()), int(expected[0,-1].argmax()))
            self.assertEqual(plan.frontend_precision, "fp16")
            print(json.dumps({"hf_fuse": fuse, "hf_w8a8": quant, "cosine": cosine,
                              "words": len(asm.words), "calls": len(plan.kernel_span)}))

    def test_constant_weights_quantize_offline(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, .2, (3,64)).astype("float16")
        w = rng.normal(0, .1, (64,64)).astype("float16")
        for quantizer in (Q.QuantizeWeightsW8A16, Q.QuantizeW8A8):
            bb = relax.BlockBuilder()
            inp = relax.Var("x", relax.TensorStructInfo(x.shape, "float16"))
            with bb.function("prefill", [inp]):
                with bb.dataflow():
                    out = bb.emit_output(relax.op.matmul(inp, relax.const(w)))
                bb.emit_func_output(out)
            quantized = quantizer()(bb.finalize())
            self.assertEqual(int(quantized.attrs["npu.quantized_matmuls"]), 1)
            mod = lower(quantized)
            self.assertEqual(len([f for f in mod.functions.values()
                                  if isinstance(f, tir.PrimFunc)]), 1)
            got, _, _ = run(mod, {"x": x}, (3,64))
            np.testing.assert_allclose(got, x.astype("float32") @ w.astype("float32"),
                                       atol=.005, rtol=.02)

    def test_native_errors_are_not_fallback(self):
        mod = matmul(64,64,64)
        prim = next(f for f in mod.functions.values() if isinstance(f, tir.PrimFunc))
        diagnostics = {}
        with patch.object(native, "function", return_value=lambda *a: ["unsupported", "probe"]):
            self.assertIsNone(native.codegen_kernel(prim, [], {}, [], {}, None, diagnostics))
            self.assertEqual(diagnostics, {"status": "fallback", "reason": "probe"})
        with patch.object(native, "function", return_value=lambda *a: (_ for _ in ()).throw(
                tvm.error.TVMError("internal failure"))):
            with self.assertRaisesRegex(tvm.error.TVMError, "internal failure"):
                native.codegen_kernel(prim, [], {}, [], {}, None)
        with self.assertRaisesRegex(tvm.error.TVMError, "address count"):
            native.codegen_kernel(prim, [], {}, [], {}, None)

    def test_fp32_vector_storage_rejected(self):
        bb = relax.BlockBuilder()
        x = relax.Var("x", relax.TensorStructInfo([2,64], "float32"))
        with bb.function("prefill", [x]):
            with bb.dataflow():
                out = bb.emit_output(relax.op.add(x, x))
            bb.emit_func_output(out)
        with self.assertRaisesRegex(Exception, "FP16 storage"):
            L.compile_program(lower(bb.finalize()), native_mode="compare")

    def test_softmax_sum_range_is_checked(self):
        for width in (65504, 65505):
            bb = relax.BlockBuilder()
            x = relax.Var("x", relax.TensorStructInfo([1,width], "float16"))
            with bb.function("prefill", [x]):
                with bb.dataflow():
                    out = bb.emit_output(relax.op.nn.softmax(x, axis=-1))
                bb.emit_func_output(out)
            mod = bb.finalize()
            if width > 65504:
                with self.assertRaisesRegex((ValueError, tvm.error.TVMError), "65504"):
                    lower(mod)
            else:
                got, _, plan = run(lower(mod), {"x": np.zeros((1,width), "float16")}, (1,width))
                self.assertTrue(np.isfinite(got).all())
                np.testing.assert_array_equal(got, np.full((1,width), 1/width, "float16"))
                self.assertTrue(all(plan.native.values()), plan.native_diagnostics)


if __name__ == "__main__":
    unittest.main()
