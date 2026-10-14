"""Lifetime reuse must preserve loop values, alias reads and the pool boundary."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import tvm
from tvm.script import tir as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu_compiler import npu_link as L, npu_memplan as M, npu_schedule as S
from npu_compiler.device import PROFILES
from npu_compiler.npu_errors import LinkError
from npu_compiler.npu_sram import plan_sram
from npu_compiler.v09_asm import V09Asm
from npu_compiler.tir_codegen_v09 import SramEmitter
from npu_compiler.v09_runtime import run as run_v09
from test_compiler_contracts import lower, traffic
from test_priority_contracts import rms


@T.prim_func
def chain(X: T.Buffer((64,), "float16"), O: T.Buffer((64,), "float16")):
    A = T.alloc_buffer((64,), "float16", scope="global.sram")
    B = T.alloc_buffer((64,), "float16", scope="global.sram")
    C = T.alloc_buffer((64,), "float16", scope="global.sram")
    for i in range(64):
        A[i] = X[i] * T.float16(2)
    for i in range(64):
        B[i] = A[i] + T.float16(1)
    for i in range(64):
        C[i] = B[i] * T.float16(2)
    for i in range(64):
        O[i] = C[i]


@T.prim_func
def repeated(X: T.Buffer((64,), "float16"), O: T.Buffer((3,64), "float16")):
    A = T.alloc_buffer((64,), "float16", scope="global.sram")
    B = T.alloc_buffer((64,), "float16", scope="global.sram")
    for i in range(64):
        A[i] = X[i]
    for j in range(3):
        for i in range(64):
            O[j,i] = A[i]
        for i in range(64):
            B[i] = X[i] * T.Cast("float16", j+1)
        for i in range(64):
            O[j,i] = O[j,i] + B[i]


@T.prim_func
def alias_read(X: T.Buffer((64,), "float16"), O: T.Buffer((64,), "float16")):
    A = T.alloc_buffer((64,), "float16", scope="global.sram")
    B = T.alloc_buffer((64,), "float16", scope="global.sram")
    for i in range(64):
        A[i] = X[i]
    for i in range(64):
        B[i] = X[i] * T.float16(2)
    with T.block("view"):
        V = T.match_buffer(A[0:64], (64,), "float16", scope="global.sram")
        for i in range(64):
            O[i] = V[i] + B[i]


@T.prim_func
def pending_matrix(X: T.Buffer((64,64), "float16")):
    C = T.alloc_buffer((64,64), "float16", scope="global.sram")
    D = T.alloc_buffer((64,64), "float16", scope="global.sram")
    T.evaluate(T.call_extern("int32", "npu_fill_zero", C.access_ptr("w"), 64))
    T.evaluate(T.call_extern("int32", "npu_gemm_acc", C.access_ptr("rw"), 64,
                            X.access_ptr("r"), 64, X.access_ptr("r"), 64))
    for i,j in T.grid(64,64):
        D[i,j] = X[i,j]


def layout(prim):
    return plan_sram(prim, S._sram_buffers(prim))


class LifetimeTests(unittest.TestCase):
    def test_disjoint_lifetimes_reuse_but_operands_do_not_alias(self):
        allocation = layout(chain)
        a,b,c = (next(i for i,v in enumerate(allocation.buffers) if v.name == n)
                 for n in ("A","B","C"))
        self.assertEqual(allocation.offsets[a], allocation.offsets[c])
        self.assertNotEqual(allocation.offsets[a], allocation.offsets[b])
        self.assertIn(b, allocation.conflicts[a])
        self.assertEqual(allocation.peak_nibbles, 2 * 64 * 4)
        restored = tvm.ir.load_json(tvm.ir.save_json(chain))
        self.assertEqual(layout(restored).offsets, allocation.offsets)

    def test_buffer_read_in_next_iteration_stays_live(self):
        allocation = layout(repeated)
        self.assertIn(1, allocation.conflicts[0])
        self.assertNotEqual(*allocation.offsets)

    def test_match_buffer_alias_extends_the_owners_lifetime(self):
        allocation = layout(alias_read)
        self.assertEqual(len(allocation.buffers), 2)
        self.assertIn(1, allocation.conflicts[0])
        self.assertNotEqual(*allocation.offsets)

    def test_pending_matrix_destination_is_pinned(self):
        allocation = layout(pending_matrix)
        c = next(i for i,v in enumerate(allocation.buffers) if v.name == "C")
        self.assertIn(c, allocation.pinned)
        self.assertEqual(allocation.peak_nibbles, allocation.baseline_nibbles)

    def test_validator_rejects_conflicting_overlap_and_capacity(self):
        allocation = layout(chain)
        with self.assertRaisesRegex(LinkError, "lifetime conflict"):
            replace(allocation, offsets=(0,0,0)).validate()
        with self.assertRaisesRegex(LinkError, "capacity"):
            allocation.place(cursor=24, capacity=allocation.peak_nibbles)
        with self.assertRaisesRegex(LinkError, "nonnegative"):
            allocation.place(cursor=-8)

    def test_reuse_is_bit_exact_and_respects_the_pool_end(self):
        rng = np.random.default_rng(45)
        for width in (65,3072):
            mod = lower(rms(1e-5,width))
            vals = {"x": rng.normal(0,.4,(7,width)).astype("float16"),
                    "w": rng.normal(1,.1,(width,)).astype("float16")}
            outputs, peaks = [], []
            for reuse in (False,True):
                asm,plan = L.compile_program(mod, profile=replace(PROFILES["v09"],reuse_sram=reuse),
                                             native_mode="compare")
                func = M.assign_addresses(mod)[0]["prefill"]
                got,_ = L.run_program(asm,plan,func,vals,(7,width))
                outputs.append(got)
                peaks.append(max(plan.sram_peak.values()))
                self.assertTrue(all(plan.native.values()))
                if not reuse:
                    continue
                # Place 32 known bytes immediately after the declared SRAM
                # high-water mark, then read them back after the real program.
                raw = L.build_image(plan,func,vals).view("uint8")
                size = (raw.size+3)//4*4
                image = np.zeros(size+64,"uint8")
                image[:raw.size] = raw
                image[size:size+32] = np.arange(32,dtype="uint8") ^ 0xA5
                prefix,suffix = V09Asm(),V09Asm()
                SramEmitter(prefix).dma_in(size,peaks[-1],32)
                SramEmitter(suffix).dma_out(size+32,peaks[-1],32)
                suffix.halt()
                images,_ = run_v09(prefix.words+asm.words[:-1]+suffix.words,image.view("<u4"))
                final = np.ascontiguousarray(images[-1]).view("uint8")
                np.testing.assert_array_equal(final[size+32:], image[size:size+32])
            np.testing.assert_array_equal(outputs[0].view("uint16"), outputs[1].view("uint16"))
            self.assertLess(peaks[1],peaks[0])

    def test_capacity_selection_uses_the_same_layout_as_the_linker(self):
        mod = lower(rms(1e-5,3072))
        profiles = [replace(PROFILES["v09"],sram_bytes=200*1024,reuse_sram=b) for b in (False,True)]
        programs = [L.compile_program(mod,profile=p,native_mode="compare") for p in profiles]
        for (_,plan),profile in zip(programs,profiles):
            self.assertLessEqual(max(plan.sram_peak.values()),profile.sram_nibbles)
            for name,record in plan.schedules.items():
                self.assertEqual(record["selected"]["sram_peak_nibbles"],plan.sram_peak[name])
        baseline, reused = (next(iter(plan.schedules.values()))["selected"] for _,plan in programs)
        self.assertIsNotNone(baseline["stripe_rows"])
        self.assertIsNone(reused["stripe_rows"])
        self.assertLess(traffic(programs[1][0].words),traffic(programs[0][0].words))
        vals={"x":np.random.default_rng(4).normal(size=(7,3072)).astype("float16"),
              "w":np.ones(3072,"float16")}
        func=M.assign_addresses(mod)[0]["prefill"]
        results=[L.run_program(asm,plan,func,vals,(7,3072))[0] for asm,plan in programs]
        np.testing.assert_array_equal(*[r.view("uint16") for r in results])


if __name__ == "__main__":
    unittest.main(verbosity=2)
