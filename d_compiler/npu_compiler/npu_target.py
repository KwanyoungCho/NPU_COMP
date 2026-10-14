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

from . import npu_legalize, npu_link, tvm_pipeline
# Public target/profile convenience exports.
from .device import NPU_KEY, NpuProfile, PROFILES, npu_target, profile_of
from .v09_contract import NUMERIC_POLICY

def npu_pipeline(fuse=False, lift_params=True, numeric_policy=NUMERIC_POLICY,
                 fold_repeated_gemv=False):
    """The graph half of the NPU flow, as a standard Relax pipeline.

    Fusion is opt-in. Generic multi-output vector fusion is supported when
    staging fits SRAM; compound reductions and matmul epilogues still need a
    legal schedule. Each vector step retains an FP16 store boundary.
    """
    return tvm_pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(numeric_policy),
        fuse=fuse, lift_params=lift_params, fold_repeated_gemv=fold_repeated_gemv)


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
        return self.plan.top * self.plan.unit_bytes

    def build_image(self, values):
        """The initial global memory image, from parameter name -> array."""
        return npu_link.build_image(self.plan, self.func, values)

    def run(self, values, output_shape=None):
        """Execute on the v09 C-model; -> (output array, counters)."""
        return npu_link.run_program(self.words, self.plan, self.func,
                                    values, output_shape)

    def save(self, path):
        """Write the instruction stream as little-endian 32-bit words."""
        import numpy as np

        np.asarray(self.words, dtype="<u4").tofile(str(path))
        return path

    def export_bundle(self, directory, values=None):
        """Export ISA, IR, placement and costs for independent model/RTL review.

        With values, the bundle includes a C-model-ready global memory image.
        It contains no invented cycles or hardware performance estimates.
        """
        import hashlib
        import json
        from pathlib import Path
        from .v09_stats import analyze
        from .npu_provenance import describe
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        program = directory / "program_memory.bin"
        self.save(program)
        metadata = dict(format="npu-v09-review-v1", target=str(self.target),
                        program_sha256=hashlib.sha256(program.read_bytes()).hexdigest(),
                        codegen_contract=self.plan.codegen_contract,
                        numeric_policy=self.plan.numeric_policy,
                        frontend_precision=self.plan.frontend_precision,
                        statistics=analyze(self.words, self.plan),
                        memory=dict(unit_bytes=self.plan.unit_bytes,
                                    addresses=self.plan.address, sizes_bytes=self.plan.nbytes),
                        schedules=self.plan.schedules, sram_layouts=self.plan.sram_layouts,
                        joint_memory=getattr(self.plan, 'joint_memory', None),
                        call_schedules=getattr(self.plan, 'call_schedules', []),
                        kernel_records=self.plan.kernel_records,
                        dram_compaction=getattr(self.plan,'dram_compaction',None),
                        fusion=getattr(self.plan, "fusion", None),
                        residency=getattr(self.plan, "residency", None),
                        views=self.plan.views, compilation=self.plan.compilation,
                        outputs=self.plan.outputs,
                        cache_append=getattr(self.plan, "cache_append", None),
                        provenance=describe(),
                        graph_optimizations=getattr(self.plan, 'graph_optimizations', {}),
                        hardware_measurement=None)
        if self.module is not None:
            (directory / "lowered.py").write_text(self.module.script() + "\n")
        for index, prim in getattr(self.plan, 'scheduled_calls', {}).items():
            (directory / f"scheduled_{index:04d}.py").write_text(prim.script() + "\n")
        if values is not None:
            self.build_image(values).tofile(directory / "global_memory.bin")
        (directory / "program.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return directory

    def __repr__(self):
        return (f"NpuExecutable({len(self.words):,} words, "
                f"{self.kernel_count} kernels, "
                f"{self.image_bytes / 2**20:.1f} MiB image)")


def build(mod, target=None, func_name="prefill", pipeline=None,
          peephole=True, fusion="auto", residency=True, profile=None):
    """Lower and link an IRModule with cost-selected fusion and SRAM forwarding."""
    from .npu_compilation import compilation
    with compilation() as context:
        result = _build(mod, target, func_name, pipeline, peephole, fusion, residency, profile)
        result.plan.compilation = context.describe()
        return result


def _build(mod, target=None, func_name="prefill", pipeline=None,
           peephole=True, fusion="auto", residency=True, profile=None):
    """IRModule (pre-lowering) -> :class:`NpuExecutable`.

    Mirrors ``relax.build(mod, target)`` in shape: run the pipeline for the
    target, then hand the result to that target's code generator, which here
    is the linker in :mod:`npu_link`.

    ``fusion="auto"`` verifies each FuseOps region and checks whole-program
    DMA/word/SRAM cost. ``on`` requests unrestricted general fusion; ``off``
    disables pattern and generic fusion. An explicit pipeline controls
    graph lowering itself. ``residency`` enables versioned SRAM forwarding.
    """
    target = tvm.target.Target(target) if target is not None else npu_target()
    profile = profile or profile_of(target)
    from .npu_compilation import reachable_module
    mod = reachable_module(mod, [func_name])
    if pipeline is None:
        from .npu_build import lower_and_select
        lowered, compiled, _ = lower_and_select(mod, profile, [func_name], fusion,
                                               peephole=peephole, residency=residency)
        asm, plan = compiled[func_name]
        return NpuExecutable(asm.words, plan, plan.func, asm.kernel_count, target, lowered)
    if fusion not in ("auto", "off", "on"):
        raise ValueError("fusion must be auto, off, or on")
    lowered = pipeline(mod)
    asm, plan = npu_link.compile_program(lowered, func_name, peephole=peephole,
                                         profile=profile, residency=residency)
    return NpuExecutable(asm.words, plan, plan.func,
                         asm.kernel_count, target, lowered)


__all__ = ["NPU_KEY", "NpuProfile", "PROFILES", "npu_target",
           "profile_of", "npu_pipeline", "NpuExecutable", "build"]
