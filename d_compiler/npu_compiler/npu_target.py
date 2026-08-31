"""The NPU as a TVM target, and the build entry point that goes with it.

Two things live here.

**A target.** ``npu_target()`` is a real ``tvm.target.Target`` of kind
``ext_dev`` keyed ``npu``, so passes and schedules can dispatch on it the way
they do for any other backend.  ``ext_dev`` only accepts TVM's own attribute
set, so the machine's numbers -- SRAM size, tile, lanes -- sit in an
:class:`NpuProfile` selected by the target's ``model``, rather than being
constants scattered through the linker and the schedules.

**A pipeline.** The graph half is registered under the name ``npu``, so
``relax.get_pipeline("npu")`` reaches it like any stock pipeline.

``relax.build`` itself is deliberately not the entry point.  It ends in a
``tvm.runtime.Module`` loaded by a runtime that allocates memory and calls
functions; this machine has neither -- a program is one straight-line
instruction stream over one flat memory image, with every address decided at
compile time.  :func:`build` therefore returns an :class:`NpuExecutable`,
which is that stream plus the memory plan needed to fill the image.
"""
from __future__ import annotations

import tvm
from tvm import relax

from . import npu_legalize, npu_link, npu_memplan, tvm_pipeline
from .device import NPU_KEY, NpuProfile, PROFILES, npu_target, profile_of

def npu_pipeline(fuse=False, lift_params=True):
    """The graph half of the NPU flow, as a standard Relax pipeline.

    Fusion is off by default: a fused kernel computes a compound expression
    per element, but the vector unit applies one operation to a whole vector,
    so the body has to be re-serialized into vector steps and nothing is
    saved (measured: 29,970 words fused against 29,946 unfused).
    """
    return tvm_pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=fuse, lift_params=lift_params)


if "npu" not in relax.pipeline.PIPELINE_MAP:
    relax.register_pipeline("npu")(npu_pipeline)


class NpuExecutable:
    """A linked program: instruction words plus the memory plan for the image."""

    def __init__(self, words, plan, func, kernel_count, target, module=None):
        self.words = words
        self.module = module              # the lowered IRModule it came from
        self.plan = plan
        self.func = func                  # the planned Relax function
        self.kernel_count = kernel_count
        self.target = target

    @property
    def param_names(self):
        return [param.name_hint for param in self.func.params]

    @property
    def image_bytes(self):
        return self.plan.top * 2

    def build_image(self, values):
        """The initial global memory image, from parameter name -> array."""
        return npu_link.build_image(self.plan, self.func, values)

    def run(self, values, output_shape):
        """Execute on the v09 C-model; -> (output array, counters)."""
        return npu_link.run_program(self.words, self.plan, self.func,
                                    values, output_shape)

    def save(self, path):
        """Write the instruction stream as little-endian 32-bit words."""
        import numpy as np

        np.asarray(self.words, dtype="<u4").tofile(str(path))
        return path

    def __repr__(self):
        return (f"NpuExecutable({len(self.words):,} words, "
                f"{self.kernel_count} kernels, "
                f"{self.image_bytes / 2**20:.1f} MiB image)")


def build(mod, target=None, func_name="prefill", pipeline=None,
          peephole=True):
    """IRModule (pre-lowering) -> :class:`NpuExecutable`.

    Mirrors ``relax.build(mod, target)`` in shape: run the pipeline for the
    target, then hand the result to that target's code generator, which here
    is the linker in :mod:`npu_link`.
    """
    target = tvm.target.Target(target) if target is not None else npu_target()
    lowered = (pipeline or npu_pipeline())(mod)
    asm, plan = npu_link.compile_program(lowered, func_name, peephole=peephole,
                                         profile=profile_of(target))
    planned, _ = npu_memplan.assign_addresses(lowered, func_name)
    return NpuExecutable(asm.words, plan, planned[func_name],
                         asm.kernel_count, target, lowered)


__all__ = ["NPU_KEY", "NpuProfile", "PROFILES", "npu_target",
           "profile_of", "npu_pipeline", "NpuExecutable", "build"]
