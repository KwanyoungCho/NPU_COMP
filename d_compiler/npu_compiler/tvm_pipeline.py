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


def _attention_patterns(tile=64):
    """The whole attention: ``matmul -> scale -> mask -> softmax -> matmul``.

    Fusing the score chain alone (``_masked_softmax_patterns``) removes two
    passes over the score matrix but still materializes it.  Fusing the two
    matmuls in as well lets the schedule block the whole thing over Q row
    blocks, and then the score matrix never leaves SRAM: at seq 2048 the
    attention moves 1,057 MB as five kernels and 235 MB as one.

    Tried before the score-chain patterns, so an attention this cannot take
    still gets the smaller fusion -- which is also the fallback when ``check``
    below refuses.
    """
    from tvm.relax.dpl.pattern import is_op, wildcard
    from tvm.relax.transform import FusionPattern

    def supported(context):
        exprs = context.annotated_expr
        def shape(name):
            info = exprs[name].struct_info
            if info.dtype != "float16" or info.shape is None or info.ndim != 3:
                return None
            try:
                return tuple(int(dim) for dim in info.shape)
            except (TypeError, ValueError):
                return None
        q, k, v = (shape(name) for name in ("q", "k", "v"))
        if q is None or k is None or v is None:
            return False
        if int(exprs["softmax"].attrs.axis) not in (-1, 2):
            return False
        h, m, d = q
        # This compound schedule handles static, aligned rank-3 attention.
        # Other shapes retain separate matmuls and the smaller score fusion.
        return (h > 0 and k[0] == h and v[0] == h and k[1] == d
                and v[1] == k[2]
                and all(x > 0 and x % tile == 0 for x in (m, d, k[2], v[2])))

    patterns = []
    for scaled in (True, False):
        q, k, v = wildcard(), wildcard(), wildcard()
        scores = is_op("relax.matmul")(q, k)
        if scaled:
            scores = is_op("relax.multiply")(scores, wildcard())
        masked = is_op("relax.add")(scores, wildcard())
        softmax = is_op("relax.nn.softmax")(masked)
        output = is_op("relax.matmul")(softmax, v)
        patterns.append(FusionPattern(
            "npu.attention" if scaled else "npu.attention_unscaled", output,
            {"q": q, "k": k, "v": v, "softmax": softmax}, supported))
    return patterns


def graph_prefix(lift_params=True):
    from .npu_attention import CanonicalizeAttention
    stages = [CanonicalizeAttention(), relax.transform.CanonicalizeBindings(),
              relax.transform.EliminateCommonSubexpr(), relax.transform.RewriteDataflowReshape()]
    if lift_params:
        stages.append(relax.transform.LiftTransformParams())
    return tvm.transform.Sequential(stages, name="npu_graph_prefix")


def _matrix_producer_patterns():
    from tvm.relax.dpl.pattern import is_op, wildcard
    patterns = []
    for wrapper in (None, "relax.repeat", "relax.expand_dims"):
        rhs = is_op("relax.concat")(wildcard())
        if wrapper:
            rhs = is_op(wrapper)(rhs)
        patterns.append(("npu.matrix_producer_" + (wrapper or "concat").split(".")[-1],
                         is_op("relax.matmul")(wildcard(), rhs)))
    return patterns


def graph_pipeline(custom_legalize=None, fuse=True, lift_params=True,
                   fuse_attention=True, fold_repeated_gemv=False,
                   fuse_producers=False, prepared=False):
    """Relax -> TIR PrimFuncs, following the standard stage order.

    ``custom_legalize`` is TVM's per-op legalization override map; NPU specific
    op decompositions are registered there rather than in a bespoke pass.

    ``lift_params`` runs ``LiftTransformParams``, which hoists parameter-only
    computation (notably the ``permute_dims`` that ``nn.Linear`` applies to its
    weight) into a separate ``transform_params`` function the host runs once at
    load time.  Without it those transposes are materialized on every
    invocation -- for Llama's lm_head that is a 751 MiB tensor.

    ``fuse_attention`` fuses the attention, whole where the shape allows
    (``_attention_patterns``) and otherwise just its score chain
    (``_masked_softmax_patterns``).  It is separate from ``fuse``, which turns
    on TVM's general fusion. That remains opt-in: vector steps and multiple
    outputs are covered, but arbitrary compound schedules and automatic
    fusion profitability selection are not yet supported.
    """
    stages = [] if prepared else [graph_prefix(lift_params)]
    if fold_repeated_gemv:
        from .npu_graph_opt import FoldRepeatedGemv
        stages.append(FoldRepeatedGemv())
    if fuse_producers:
        stages.append(relax.transform.FuseOpsByPattern(_matrix_producer_patterns()))
    if fuse_attention not in (False, True, "score", "full"):
        raise ValueError("fuse_attention must be False, True, 'score', or 'full'")
    if fuse_attention:
        stages.append(relax.transform.FuseOpsByPattern(
            ([] if fuse_attention == "score" else _attention_patterns())
            + _masked_softmax_patterns()))
    from .npu_numeric import FoldConstants
    stages += [
        relax.transform.LegalizeOps(custom_legalize),
        FoldConstants() if custom_legalize else relax.transform.FoldConstant(),
        relax.transform.AnnotateTIROpPattern(),
    ]
    if fuse:
        stages += [relax.transform.FuseOps(), relax.transform.FuseTIR()]
    elif fuse_attention or fuse_producers:
        # only the pattern above became a composite function; FuseTIR merges
        # its PrimFuncs and leaves everything else alone
        stages.append(relax.transform.FuseTIR())
    stages += [relax.transform.DeadCodeElimination()]
    return tvm.transform.Sequential(stages, name="npu_graph_pipeline")


def kernel_summary(mod):
    """PrimFunc inventory after lowering — the granularity codegen must handle."""
    from tvm import tir
    funcs = {gv.name_hint: fn for gv, fn in mod.functions.items()
             if isinstance(fn, tir.PrimFunc)}
    return {"prim_funcs": len(funcs), "names": sorted(funcs)}
