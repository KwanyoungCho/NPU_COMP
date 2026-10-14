"""Select TVM FuseOps regions with target legality and emitted program cost.

Rejected regions are inlined using Relax's Function.inline_functions before
the final FuseTIR. Independent legal regions remain fused. No model names or
handwritten operator chains participate in selection.
"""
import tvm
from tvm import relax, tir

from . import npu_link
from .npu_errors import LinkError
from .npu_kernel import matmul_blocks
from .npu_w8a8 import is_w8a8
from .tir_codegen_v09 import V09TirError
from .v09_stats import analyze, cost


def _probe(module, function, profile):
    mod = tvm.IRModule(module.functions, attrs=module.attrs)
    mod["__npu_fusion_probe"] = function.without_attr("Primitive").with_attr(
        "global_symbol", "__npu_fusion_probe")
    from .npu_compilation import reachable_module, current
    mod = reachable_module(mod, ["__npu_fusion_probe"])
    mode = npu_link._native_default()
    def create():
        asm, plan = npu_link.compile_program(mod, "__npu_fusion_probe", profile=profile,
                                            residency=True,native_mode=mode)
        return analyze(asm.words, plan, detail=False)
    context = current()
    return context.region_cost(mod,'__npu_fusion_probe',(profile,mode),create) if context else create()


def _wrapper(gv, fn):
    bb = relax.BlockBuilder()
    with bb.function("__npu_fusion_probe", fn.params):
        with bb.dataflow():
            out = bb.emit_output(relax.call_tir(gv, relax.Tuple(fn.params),
                                                out_sinfo=fn.ret_struct_info))
        bb.emit_func_output(out)
    return bb.get()["__npu_fusion_probe"]


def select(module, profile, entries=("prefill",), matmul_epilogues=False):
    """Already legalized module -> selected module and JSON diagnostics.

    Matrix epilogues and quantized composites are fusion boundaries until a
    schedule supports their full producer/consumer graph. Existing attention
    composites remain intact. Generic regions are verified by code generation,
    including multi-output/stripe restrictions, before accepting fusion.
    """
    base = tvm.IRModule(module.functions, attrs=module.attrs)
    boundaries = []
    for gv, fn in list(base.functions.items()):
        if isinstance(fn, tir.PrimFunc) and (is_w8a8(fn) or (
                matmul_blocks(fn) and (not matmul_epilogues or len(matmul_blocks(fn)) != 1))):
            base[gv] = fn.with_attr("op_pattern", 8)  # TVM kOpaque
            boundaries.append(gv.name_hint)
    outlined = relax.transform.FuseOps()(base)
    fused = relax.transform.FuseTIR()(outlined)
    reachable = set()
    pending = [outlined.get_global_var(entry) for entry in entries]
    while pending:
        gv = pending.pop()
        if gv in reachable:
            continue
        reachable.add(gv)
        fn = outlined[gv]
        if isinstance(fn, relax.Function):
            def visit(expr):
                if isinstance(expr, relax.Call) and isinstance(expr.op, relax.GlobalVar):
                    pending.append(expr.op)
            relax.analysis.post_order_visit(fn.body, visit)
    rejected, records = {}, []
    for gv, fn in outlined.functions.items():
        if not (gv in reachable and isinstance(fn, relax.Function) and fn.attrs
                and fn.attrs.get("Primitive", 0)):
            continue
        record = dict(region=gv.name_hint)
        # The unfused region is the mandatory reference. A failure here is an
        # unsupported input or compiler error, not a reason to hide an error.
        reference = _probe(outlined, fn, profile)
        record["unfused"] = reference
        try:
            candidate = _probe(fused, _wrapper(fused.get_global_var(gv.name_hint), fn), profile)
        except (LinkError, V09TirError) as error:
            record.update(selected=False, reason=str(error), status="unsupported")
        else:
            record["fused"] = candidate
            selected = cost(candidate) < cost(reference)
            record.update(selected=selected,
                          status="selected" if selected else "unprofitable",
                          reason="lexicographic DMA bytes, words, SRAM peak")
        if not record["selected"]:
            rejected[gv] = fn
        records.append(record)
    if rejected:
        outlined = tvm.IRModule(outlined.functions, attrs=outlined.attrs)
        for gv, fn in list(outlined.functions.items()):
            if isinstance(fn, relax.Function) and gv not in rejected:
                outlined[gv] = fn.inline_functions(rejected)
        for gv in rejected:
            del outlined[gv]
    selected = relax.transform.FuseTIR()(outlined)
    selected = relax.transform.DeadCodeElimination()(selected)
    return selected, dict(policy="per-region emitted DMA/words/SRAM",
                          boundaries=sorted(boundaries), regions=records)


def select_programs(module, profile, entries=('prefill',), *, peephole=True,
                    residency=True, reference_module=None):
    """Select one module only if every requested device entry avoids regression.

    Multiple entries can share PrimFuncs and transformed-parameter ABI. An
    atomic module choice keeps those contracts identical for all runners.
    Return linked programs as well so build/generation can use the evaluated
    streams without linking them again.
    """
    from .npu_compilation import compilation
    with compilation() as context:
        baseline = reference_module if reference_module is not None else module
        before, after, records = {}, {}, {}
        for entry in entries:
            # Establish a valid reference before trying graph rewrites. Errors
            # in the original input and internal backend bugs still propagate.
            before[entry] = npu_link.compile_program(baseline, entry, profile=profile,
                                                    peephole=peephole, residency=residency)
        failure = None
        try:
            with context.phase('fusion_regions'):
                candidate, decisions = select(module, profile, entries)
        except (LinkError, V09TirError) as error:
            failure = str(error)
            decisions = dict(regions=[], boundaries=[], candidate_rejected=failure)
        for entry in entries:
            reason = failure
            if reason is None:
                try:
                    after[entry] = npu_link.compile_program(candidate, entry, profile=profile,
                                                   peephole=peephole, residency=residency)
                except (LinkError, V09TirError) as error:
                    reason = str(error)
            with context.phase('program_cost'):
                a = analyze(after[entry][0].words, after[entry][1], detail=False) if reason is None else None
                b = analyze(before[entry][0].words, before[entry][1], detail=False)
            records[entry] = dict(fused=a, unfused=b, improves=a is not None and cost(a) < cost(b),
                                  nonregressing=a is not None and cost(a) <= cost(b))
            if reason is not None:
                records[entry].update(status='unsupported',reason=reason)
        selected = (all(r['nonregressing'] for r in records.values())
                    and any(r['improves'] for r in records.values()))
        choice = 'fused' if selected else 'unfused'
        decisions['programs'] = {entry: dict(record, selected=choice)
                                for entry, record in records.items()}
        decisions['module_policy'] = 'all device entries nonregressing, at least one improves'
        if len(entries) == 1:
            decisions['program'] = decisions['programs'][entries[0]]
        compiled = after if selected else before
        for asm, plan in compiled.values():
            plan.fusion = decisions
            plan.compilation = context.describe()
        return candidate if selected else baseline, compiled, decisions
