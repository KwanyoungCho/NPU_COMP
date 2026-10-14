"""Small, executable TVM Schedule examples; no compiler implementation changes.

Run with the npu-tvm Python. The 2-row tiles explain APIs, not NPU tile sizes.
Only the global-memory compute_at/reverse_compute_at examples run on LLVM.
SRAM scope examples are inspected as TIR, not executed on the CPU.
"""
import json
from pathlib import Path

import numpy as np
import tvm
from tvm import te, tir

OUT = Path(__file__).resolve().parent


def module(two_steps=True):
    a = te.placeholder((4, 4), "float32", name="A")
    if two_steps:
        b = te.compute((4, 4), lambda i, j: a[i, j] + 1, name="B")
        c = te.compute((4, 4), lambda i, j: b[i, j] * 2, name="C")
    else:
        c = te.compute((4, 4), lambda i, j: a[i, j] + 1, name="C")
    return tvm.IRModule({"main": te.create_prim_func([a, c])})


def save(name, sch):
    (OUT / f"{name}.py").write_text(sch.mod.script() + "\n")
    (OUT / f"{name}.trace.txt").write_text("\n".join(sch.trace.as_python()) + "\n")


def check(mod):
    run = tvm.build(mod, target="llvm")
    a = np.arange(16, dtype="float32").reshape(4, 4)
    output = tvm.nd.empty((4, 4), "float32")
    run(tvm.nd.array(a), output)
    np.testing.assert_array_equal(output.numpy(), (a + 1) * 2)


base = module()
save("01_before", tir.Schedule(base))

sch = tir.Schedule(base)
outer, inner = sch.split(sch.get_loops(sch.get_block("C"))[0], factors=[None, 2])
sch.compute_at(sch.get_block("B"), outer)
save("02_compute_at", sch)
check(sch.mod)

sch = tir.Schedule(base)
outer, inner = sch.split(sch.get_loops(sch.get_block("B"))[0], factors=[None, 2])
sch.reverse_compute_at(sch.get_block("C"), outer)
save("03_reverse_compute_at", sch)
check(sch.mod)

sch = tir.Schedule(module(False))
copy = sch.cache_read(sch.get_block("C"), 0, "global.sram")
save("04_cache_read", sch)
outer, inner = sch.split(sch.get_loops(sch.get_block("C"))[0], factors=[None, 2])
sch.compute_at(copy, outer)
save("05_cache_read_compute_at", sch)
compacted = tvm.transform.Sequential([
    tir.transform.PlanAndUpdateBufferAllocationLocation(),
    tir.transform.ConvertBlocksToOpaque(),
    tir.transform.CompactBufferAllocation(),
])(sch.mod)
(OUT / "08_cache_read_compacted.py").write_text(compacted.script() + "\n")

sch = tir.Schedule(base)
sch.set_scope(sch.get_block("B"), 0, "global.sram")
save("06_set_scope", sch)

sch = tir.Schedule(module(False))
sch.cache_write(sch.get_block("C"), 0, "global.sram")
save("07_cache_write", sch)

(OUT / "checks.json").write_text(json.dumps({
    "tvm_version": tvm.__version__,
    "compute_at_llvm": "exact match: (A + 1) * 2",
    "reverse_compute_at_llvm": "exact match: (A + 1) * 2",
    "sram_examples": "schedule construction only; not executed on LLVM",
}, indent=2) + "\n")
print("Wrote eight TIR examples plus schedule traces; both LLVM checks passed.")
