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


def _masked_softmax_patterns():
    """Attention's score chain: ``softmax(add(mul(scores, scale), mask))``.

    Left unfused it is a separate pass over a [heads, seq, seq] tensor per op,
    which is the largest thing in a long prompt: at 2048 tokens those kernels
    move 1.6 GB of the 3.7 GB a layer moves.  Fused they share one staged
    stripe.

    Two patterns, most specific first.  Gemma folds the scale into the query
    instead of scaling the scores, so its chain has no multiply, and only the
    second pattern matches it.
    """
    from tvm.relax.dpl.pattern import is_op, wildcard

    scores, scale, mask = wildcard(), wildcard(), wildcard()
    scaled = is_op("relax.nn.softmax")(
        is_op("relax.add")(is_op("relax.multiply")(scores, scale), mask))
    plain = is_op("relax.nn.softmax")(is_op("relax.add")(scores, mask))
    return [("npu.masked_softmax", scaled),
            ("npu.masked_softmax_unscaled", plain)]


def graph_pipeline(custom_legalize=None, fuse=True, lift_params=True,
                   fuse_attention=True):
    """Relax -> TIR PrimFuncs, following the standard stage order.

    ``custom_legalize`` is TVM's per-op legalization override map; NPU specific
    op decompositions are registered there rather than in a bespoke pass.

    ``lift_params`` runs ``LiftTransformParams``, which hoists parameter-only
    computation (notably the ``permute_dims`` that ``nn.Linear`` applies to its
    weight) into a separate ``transform_params`` function the host runs once at
    load time.  Without it those transposes are materialized on every
    invocation -- for Llama's lm_head that is a 751 MiB tensor.

    ``fuse_attention`` fuses only the attention score chain (see
    ``_masked_softmax_pattern``).  It is separate from ``fuse``, which turns on
    TVM's general fusion and is off by default because a fused body still has
    to be re-serialized into vector steps and that is unverified.
    """
    stages = [
        relax.transform.CanonicalizeBindings(),
        relax.transform.EliminateCommonSubexpr(),
        relax.transform.FoldConstant(),
        relax.transform.RewriteDataflowReshape(),
    ]
    if lift_params:
        stages.append(relax.transform.LiftTransformParams())
    if fuse_attention:
        stages.append(relax.transform.FuseOpsByPattern(
            _masked_softmax_patterns()))
    stages += [
        relax.transform.LegalizeOps(custom_legalize),
        relax.transform.AnnotateTIROpPattern(),
    ]
    if fuse:
        stages += [relax.transform.FuseOps(), relax.transform.FuseTIR()]
    elif fuse_attention:
        # only the pattern above became a composite function; FuseTIR merges
        # its PrimFuncs and leaves everything else alone
        stages.append(relax.transform.FuseTIR())
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
