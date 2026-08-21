"""Relax pass pipeline for the ver.08 (0818) backends.

The 0818 backend consumes normalized Relax dataflow bindings directly, so the
pipeline's job is to run the standard TVM cleanup/optimization passes over the
family graphs before planning and codegen.  Applied by ``driver.compile_module``
for the 0818/source-0818 backends only; legacy (0710/hybrid/tir) paths are
untouched.
"""
from __future__ import annotations

import tvm
from tvm import relax


def npu_pipeline():
    """Standard Relax cleanup for 0818 codegen.

    FoldConstant bakes constant-only subgraphs (e.g. prefill RoPE cos/sin from
    the static position ramp) into initial G-buffer constants, removing their
    on-device instructions; CSE/DCE/canonicalization tidy the binding list the
    backend walks.  Verified byte-exact against the unoptimized programs on the
    family layer graphs.
    """
    return tvm.transform.Sequential([
        relax.transform.CanonicalizeBindings(),
        relax.transform.EliminateCommonSubexpr(),
        relax.transform.FoldConstant(),
        relax.transform.DeadCodeElimination(),
    ], name="npu0818_pipeline")
