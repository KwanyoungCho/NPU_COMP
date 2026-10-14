"""Regression gates for the 0910 audit: semantics, resources and attribution.

Run with the npu-tvm Python environment.  These are boundary/end-to-end tests,
not snapshots of one particular schedule or instruction spelling.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import tvm
from tvm import relax, te, tir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu_compiler import npu_legalize, npu_link as L, npu_memplan as M
from npu_compiler import npu_quantize as Q, tvm_pipeline as P
from npu_compiler.device import PROFILES
from npu_compiler.nn_models import hf
from npu_compiler.peephole import eliminate_dead_stores


def lower(mod, lift=False):
    return P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                            fuse=False, lift_params=lift)(mod)


def matmul(m, k, n, quant=False, chain=False):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([m, k], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([k, n], "float16"))
    with bb.function("prefill", [x, w]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x, w))
            if chain:
                y = bb.emit(relax.op.add(y, y))
            out = bb.emit_output(y)
        bb.emit_func_output(out)
    mod = bb.finalize()
    if quant:
        mod["prefill"] = mod["prefill"].with_attr("num_input", 1)
        mod = Q.QuantizeW8A8()(mod)
    return lower(mod, quant)


def traffic(words):
    count, i = 0, 0
    while i < len(words):
        if int(words[i]) & 255 in (0xA0, 0xA8):
            size = int(words[i + 3])
            count += (size >> 16) * (size & 65535) * 4
            i += 4
        else:
            i += 1
    return count


def bounce_guard_result(walker_type=None):
    """An unaligned gather larger than scratch must preserve its neighbour."""
    from npu_compiler.v09_asm import V09Asm
    from npu_compiler.tir_codegen_v09 import SramEmitter, Walker
    from npu_compiler.v09_runtime import run

    asm = V09Asm()
    stage = SramEmitter(asm)
    walker = (walker_type or Walker)(asm, {}, stage)
    walker.scratch_slots = tuple(0x100 + 0x20 * i for i in range(6))  # 96 bytes
    guard_nib = 0x1C0
    image = np.zeros(1024, dtype="uint8")
    image[:264] = np.arange(264, dtype="uint16").astype("uint8")
    image[320:336] = 0x5A
    stage.dma_in(320, guard_nib, 16)
    walker._bounce_dma(2, [(0x4000, 130), (0x5000, 130)], True)
    stage.dma_out(512, guard_nib, 16)
    asm.halt()
    images, _ = run(asm.words, image.view("<u4"))
    return np.ascontiguousarray(images[-1]).view("uint8")[512:528]


class MemoryContractTests(unittest.TestCase):
    def test_pool_size_and_nonzero_offset_are_bytes(self):
        for dtype in ("int8", "float16", "float32"):
            with self.subTest(dtype=dtype):
                bb = relax.BlockBuilder()
                with bb.function("prefill", [], pure=False):
                    storage = bb.emit(relax.op.memory.alloc_storage(
                        relax.ShapeExpr([128]), 0, "global", dtype), "pool")
                    tensor = bb.emit(relax.op.memory.alloc_tensor(
                        storage, 16, relax.ShapeExpr([8]), dtype), "tensor")
                    bb.emit_func_output(tensor)
                _, plan = M.assign_addresses(bb.finalize())
                self.assertEqual(plan.pool_bytes["pool"], 128)
                self.assertEqual(plan.address["tensor"] - plan.pool_base["pool"], 8)
                self.assertEqual(plan.top * plan.unit_bytes, 128)

    def test_tvm_generated_pools_retain_their_byte_size(self):
        for dtype in ("int8", "float16", "float32"):
            bb = relax.BlockBuilder()
            x = relax.Var("x", relax.TensorStructInfo([4, 64], dtype))
            with bb.function("prefill", [x]):
                with bb.dataflow():
                    a = bb.emit(relax.op.add(x, x))
                    b = bb.emit(relax.op.multiply(a, x))
                    out = bb.emit_output(relax.op.add(b, x))
                bb.emit_func_output(out)
            planned, plan = M.assign_addresses(lower(bb.finalize()))
            self.assertTrue(plan.pool_bytes)
            for block in planned["prefill"].body.blocks:
                for binding in block.bindings:
                    call = binding.value
                    if isinstance(call, relax.Call) and getattr(call.op, "name", "") == "relax.memory.alloc_storage":
                        self.assertEqual(plan.pool_bytes[binding.var.name_hint],
                                         int(call.args[0].values[0]))


def rms_graph(exponent=2., axis=-1, keepdims=True, runtime_eps=False,
              forward_dtype="float32", back_dtype="float16", multiply=False,
              epsilon=1e-5, weight_shape=(4,), input_dtype="float16"):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([4, 4], input_dtype))
    w = relax.Var("w", relax.TensorStructInfo(list(weight_shape), back_dtype))
    eps = relax.Var("eps", relax.TensorStructInfo([], forward_dtype))
    with bb.function("main", [x, w] + ([eps] if runtime_eps else [])):
        with bb.dataflow():
            f = bb.emit(relax.op.astype(x, forward_dtype))
            squared = bb.emit(relax.op.multiply(f, f) if multiply else
                              relax.op.power(f, relax.const(exponent, forward_dtype)))
            mean = bb.emit(relax.op.mean(squared, axis=[axis], keepdims=keepdims))
            shifted = bb.emit(relax.op.add(mean, eps if runtime_eps else
                                           relax.const(epsilon, forward_dtype)))
            inv = bb.emit(relax.op.rsqrt(shifted))
            normed = bb.emit(relax.op.multiply(f, inv))
            back = bb.emit(relax.op.astype(normed, back_dtype))
            out = bb.emit_output(relax.op.multiply(back, w))
        bb.emit_func_output(out)
    return bb.finalize()


class RewriteContractTests(unittest.TestCase):
    def test_npu_rms_legalization_rejects_unsupported_axis_and_precision(self):
        for dtype, axis, message in (("float16", 0, "last axis"),
                                     ("float32", -1, "FP16")):
            bb = relax.BlockBuilder()
            x = relax.Var("x", relax.TensorStructInfo([4, 4], dtype))
            w = relax.Var("w", relax.TensorStructInfo([4], dtype))
            with bb.function("prefill", [x, w]):
                with bb.dataflow():
                    out = bb.emit_output(relax.op.nn.rms_norm(x, w, axes=[axis]))
                bb.emit_func_output(out)
            with self.assertRaisesRegex((ValueError, tvm.error.TVMError), message):
                # Exercise our legalization boundary directly.  TVM's earlier
                # FoldConstant pass also attempts TOPI legalization and can
                # reject axis=0 with its own internal error first.
                relax.transform.LegalizeOps(npu_legalize.legalize_map())(bb.finalize())

    def test_rms_rewrite_requires_the_complete_semantic_contract(self):
        cases = [(dict(), True), (dict(multiply=True), True), (dict(axis=1), True),
                 (dict(exponent=3.), False), (dict(axis=0), False),
                 (dict(keepdims=False), False), (dict(runtime_eps=True), False),
                 (dict(forward_dtype="float16"), False),
                 (dict(back_dtype="float32"), False),
                 (dict(epsilon=-1.), False), (dict(weight_shape=(4, 4)), False),
                 (dict(input_dtype="float32"), True)]
        for options, expected in cases:
            with self.subTest(options=options):
                mod = rms_graph(**options)
                pattern, rewrite = hf._rms_norm_pattern()
                changed = relax.dpl.rewrite_call(pattern, rewrite, mod["main"])
                self.assertEqual("R.nn.rms_norm" in changed.script(), expected)
                if not expected:
                    tvm.ir.assert_structural_equal(mod["main"], changed)

    def test_gamma_stays_after_the_fp16_rounding_boundary(self):
        for dtype in ("float16", "float32"):
            mod = rms_graph(input_dtype=dtype)
            pattern, rewrite = hf._rms_norm_pattern()
            changed = tvm.IRModule({"main": relax.dpl.rewrite_call(pattern, rewrite, mod["main"])})
            rng = np.random.default_rng(94)
            args = [tvm.nd.array(rng.normal(0, 1000, (4, 4)).astype(dtype)),
                    tvm.nd.array(rng.normal(0, .5, (4,)).astype("float16"))]
            before = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())["main"](*args).numpy()
            after = relax.VirtualMachine(relax.build(changed, "llvm"), tvm.cpu())["main"](*args).numpy()
            np.testing.assert_array_equal(before.view("uint16"), after.view("uint16"))


class BackendContractTests(unittest.TestCase):
    def test_bounce_dma_preserves_the_buffer_after_scratch(self):
        np.testing.assert_array_equal(bounce_guard_result(), np.full(16, 0x5A, "uint8"))

    def test_real_width_ragged_matmul_does_not_reread_weights(self):
        for m in (65, 129, 257, 513):
            with self.subTest(m=m):
                asm, plan = L.compile_program(matmul(m, 3072, 3072))
                once = 2 * (m * 3072 + 3072 * 3072 + m * 3072)
                self.assertEqual(traffic(asm.words), once + len(plan.constant_values) * 2 + 4)
                self.assertLessEqual(max(plan.sram_peak.values()), PROFILES["v09"].sram_nibbles)

    def test_panel_search_finds_a_non_halving_divisor_within_budget(self):
        from npu_compiler.npu_schedule import _fit_kernel

        planned, _ = M.assign_addresses(matmul(3072, 3072, 3072))
        gv = next(g for g, fn in planned.functions.items() if isinstance(fn, tir.PrimFunc))
        diagnostics = {}
        _fit_kernel(planned, gv, planned[gv], PROFILES["v09"], 24, 6, diagnostics)
        chosen = diagnostics["selected"]
        self.assertLessEqual(chosen["estimated_operand_load_bytes"], 75497472)
        self.assertLessEqual(chosen["sram_peak_nibbles"], PROFILES["v09"].sram_nibbles)
        self.assertTrue(any(a["status"] == "capacity_rejected" for a in diagnostics["attempts"]))

    def test_renamed_matmul_and_ragged_panels_execute_correctly(self):
        rng = np.random.default_rng(91)
        shapes = [(m, 128, 192) for m in (1, 63, 64, 65, 127, 128, 129)]
        shapes += [(65, 128, 65), (65, 65, 128), (65, 65, 65), (129, 128, 129)]
        for m, k, n in shapes:
            with self.subTest(shape=(m, k, n)):
                mod = matmul(m, k, n)
                for gv, fn in list(mod.functions.items()):
                    if isinstance(fn, tir.PrimFunc):
                        mod[gv] = tvm.script.from_source(fn.script().replace(
                            'T.block("matmul")', 'T.block("dot")'))
                asm, plan = L.compile_program(mod, native_mode="compare")
                planned, _ = M.assign_addresses(mod)
                x = (rng.integers(-3, 4, (m, k)) / 16).astype("float16")
                w = (rng.integers(-3, 4, (k, n)) / 16).astype("float16")
                got, _ = L.run_program(asm, plan, planned["prefill"],
                                       {"x": x, "w": w}, (m, n))
                np.testing.assert_array_equal(got, (x.astype("float32") @ w.astype("float32")).astype("float16"))
                self.assertEqual(len(plan.kernel_span), 1)

    def test_odd_width_dma_staging_does_not_overwrite_live_sram(self):
        rng = np.random.default_rng(96)
        for shape in ((65, 65), (129, 129)):
            bb = relax.BlockBuilder()
            args = [relax.Var(name, relax.TensorStructInfo(list(shape), "float16"))
                    for name in ("x", "y")]
            with bb.function("prefill", args):
                with bb.dataflow():
                    out = bb.emit_output(relax.op.add(*args))
                bb.emit_func_output(out)
            mod = lower(bb.finalize())
            asm, plan = L.compile_program(mod, native_mode="compare")
            planned, _ = M.assign_addresses(mod)
            x, y = [rng.normal(0, .5, shape).astype("float16") for _ in range(2)]
            got, _ = L.run_program(asm, plan, planned["prefill"],
                                   {"x": x, "y": y}, shape)
            np.testing.assert_array_equal(got.view("uint16"), (x + y).view("uint16"))

    def test_w8a8_zero_rows_and_all_calls_are_accounted(self):
        from reference.quantize import w8a8_reference

        mod = matmul(7, 128, 192, quant=True, chain=True)
        vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
        rng = np.random.default_rng(92)
        w = rng.normal(0, .2, (128, 192)).astype("float16")
        transformed = vm["prefill_transform_params"]([tvm.nd.array(w)])
        planned, _ = M.assign_addresses(mod)
        func = planned["prefill"]
        q_w = next(t.numpy() for t in transformed if str(t.dtype) == "int8")
        scale = next(t.numpy() for t in transformed if str(t.dtype) == "float32")
        for all_zero in (False, True):
            x = np.zeros((7, 128), "float16")
            if not all_zero:
                x[:] = rng.normal(0, .4, x.shape)
                x[[0, 3, 6]] = 0
                x[1] = np.float16(2 ** -24)
            reference = w8a8_reference(x, q_w, scale)[0]
            reference = (reference + reference).astype("float16")
            values = dict(zip([p.name_hint for p in func.params],
                              [x] + [t.numpy() for t in transformed]))
            for peephole in (False, True):
                asm, plan = L.compile_program(mod, snapshot_at={0, 1}, peephole=peephole)
                got, _ = L.run_program(asm, plan, func, values, (7, 192))
                np.testing.assert_array_equal(got.view("uint16"), reference.view("uint16"))
                self.assertEqual(asm.kernel_count, 2)
                self.assertEqual(len(plan.kernel_span), 2)
                self.assertEqual(len(plan.kernel_records), 2)
                self.assertEqual([s[0] for s in plan.snapshots], [0, 1])
                self.assertIn(plan.kernel_records[0]["backend"], ("native", "python"))
                self.assertEqual(plan.schedules[plan.kernel_records[0]["name"]]["kind"], "w8a8_tir")
                spans = plan.kernel_span
                self.assertLessEqual(spans[-1][2], len(asm.words))
                self.assertEqual(sum(traffic(asm.words[a:b]) for _, a, b in spans),
                                 traffic(asm.words) - (len(plan.constant_values) * 2 + 4))
                for record, (name, a, b) in zip(plan.kernel_records, spans):
                    self.assertEqual(record["span"], (a, b))
                    self.assertGreater(plan.sram_peak[name], 0)
                    self.assertLessEqual(plan.sram_peak[name], PROFILES["v09"].sram_nibbles)

    def test_unscaled_int8_to_fp32_is_rejected_at_compile_time(self):
        a, b = [te.placeholder((64, 64), "int8", name=n) for n in ("a", "b")]
        k = te.reduce_axis((0, 64), "k")
        out = te.compute((64, 64), lambda i, j: te.sum(
            a[i, k].astype("float32") * b[k, j].astype("float32"), k), name="dot")
        prim = te.create_prim_func([a, b, out])
        from npu_compiler.npu_intrin import schedule_matmul_sram
        with self.assertRaisesRegex(ValueError, "W8A8 scale/output contract"):
            schedule_matmul_sram(tvm.IRModule({"main": prim}), "main")

    def test_w8a8_panel_tail_uses_the_available_capacity(self):
        from dataclasses import replace
        from reference.quantize import w8a8_reference

        m, k, n = 193, 128, 192
        mod = matmul(m, k, n, quant=True)
        vm = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())
        rng = np.random.default_rng(95)
        w = rng.normal(0, .2, (k, n)).astype("float16")
        x = rng.normal(0, .4, (m, k)).astype("float16")
        x[[0, 127, 128, 192]] = 0
        transformed = vm["prefill_transform_params"]([tvm.nd.array(w)])
        profile = replace(PROFILES["v09"], sram_bytes=80000)
        asm, plan = L.compile_program(mod, profile=profile)
        schedule = next(iter(plan.schedules.values()))
        self.assertEqual((schedule["panel_rows"], schedule["row_panels"]), (128, 2))
        self.assertLessEqual(schedule["sram_peak_nibbles"], profile.sram_nibbles)
        planned, _ = M.assign_addresses(mod)
        func = planned["prefill"]
        values = dict(zip([p.name_hint for p in func.params],
                          [x] + [t.numpy() for t in transformed]))
        got, _ = L.run_program(asm, plan, func, values, (m, n))
        q_w = next(t.numpy() for t in transformed if str(t.dtype) == "int8")
        scale = next(t.numpy() for t in transformed if str(t.dtype) == "float32")
        expected = w8a8_reference(x, q_w, scale)[0]
        np.testing.assert_array_equal(got.view("uint16"), expected.view("uint16"))

    def test_peephole_boundary_map_includes_dma_payload_and_end(self):
        from npu_compiler.isa_v09 import enc_gload, enc_vlen
        words = [enc_vlen(64)] * 2 + list(enc_gload(0x82, 0x82, 0, 1, 1)) + [enc_vlen(64)]
        out, mapping = eliminate_dead_stores(words, boundaries=range(len(words) + 1))
        for i in range(len(words) + 1):
            if i not in (3, 4, 5):  # prefixes must end outside a DMA packet
                self.assertEqual(mapping[i], len(eliminate_dead_stores(words[:i])))
        self.assertEqual([mapping[i] for i in (2, 3, 4, 5, 6)], [1, 2, 3, 4, 5])
        self.assertEqual(mapping[len(words)], len(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
