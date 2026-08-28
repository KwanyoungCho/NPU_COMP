"""Standard TVM lowering pipeline for the NPU target.

The stages are the ones TVM defines; we do not replace them.  Our target
specific work is *inserted*: a custom legalization map (Relax op -> TIR) and,
later, the schedule and codegen stages.

  graph_pipeline()  Relax -> legalized+fused TIR   (target independent)
  build_llvm()      the same module built for CPU  (validation path)

Keeping the graph pipeline separate from the build pipeline means one module
can be checked on CPU and then handed to the NPU backend unchanged.
"""
from __future__ import annotations

import tvm
from tvm import relax


def graph_pipeline(custom_legalize=None, fuse=True, lift_params=True):
    """Relax -> TIR PrimFuncs, following the standard stage order.

    ``custom_legalize`` is TVM's per-op legalization override map; NPU specific
    op decompositions are registered there rather than in a bespoke pass.

    ``lift_params`` runs ``LiftTransformParams``, which hoists parameter-only
    computation (notably the ``permute_dims`` that ``nn.Linear`` applies to its
    weight) into a separate ``transform_params`` function the host runs once at
    load time.  Without it those transposes are materialized on every
    invocation -- for Llama's lm_head that is a 751 MiB tensor.
    """
    stages = [
        relax.transform.CanonicalizeBindings(),
        relax.transform.EliminateCommonSubexpr(),
        relax.transform.FoldConstant(),
        relax.transform.RewriteDataflowReshape(),
    ]
    if lift_params:
        stages.append(relax.transform.LiftTransformParams())
    stages += [
        relax.transform.LegalizeOps(custom_legalize),
        relax.transform.AnnotateTIROpPattern(),
    ]
    if fuse:
        stages += [relax.transform.FuseOps(), relax.transform.FuseTIR()]
    stages += [relax.transform.DeadCodeElimination()]
    return tvm.transform.Sequential(stages, name="npu_graph_pipeline")


def build_llvm(mod, custom_legalize=None, fuse=True, lift_params=True):
    """Validation build: run the graph pipeline, then the stock CPU build."""
    return relax.build(graph_pipeline(custom_legalize, fuse, lift_params)(mod),
                       target="llvm")


def kernel_summary(mod):
    """PrimFunc inventory after lowering — the granularity codegen must handle."""
    from tvm import tir
    funcs = {gv.name_hint: fn for gv, fn in mod.functions.items()
             if isinstance(fn, tir.PrimFunc)}
    return {"prim_funcs": len(funcs), "names": sorted(funcs)}
