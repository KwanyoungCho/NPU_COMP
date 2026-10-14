"""One graph selection entry for public builds, generation and CLI runners."""
import tvm
from . import npu_link, npu_legalize, tvm_pipeline
from .npu_compilation import compilation, reachable_module, EntryCache, OptimizationBudgetExceeded
from .npu_errors import LinkError
from .tir_codegen_v09 import V09TirError
from .v09_stats import analyze, cost


def lower_and_select(mod, profile, entries, fusion="auto", *, peephole=True,
                     residency=True, lift_params=True):
    if fusion not in ("off", "on", "auto"):
        raise ValueError("fusion must be auto, off, or on")
    entries = list(entries)
    mod = tvm.IRModule(mod.functions, attrs=mod.attrs)
    for entry in entries:
        mod[entry] = mod[entry].with_attr("global_symbol", entry)
    mod = reachable_module(mod, entries)
    mod = tvm_pipeline.graph_prefix(lift_params)(mod)
    def lower(attention, gqa=False, fuse=False, producers=False):
        return tvm_pipeline.graph_pipeline(
            custom_legalize=npu_legalize.legalize_map(profile.numeric_policy),
            lift_params=lift_params, fuse=fuse, fuse_attention=attention,
            fold_repeated_gemv=gqa, fuse_producers=producers, prepared=True)(mod)
    with compilation() as context:
        cache = EntryCache(context)
        native_mode = npu_link._native_default()
        def compile_all(module, incumbent=None):
            programs, statistics = {}, {}
            if context.schedule_database is not None:
                with context.phase('meta_schedule_extract_tasks'):
                    context.schedule_database.extract_tasks(module, entries, profile)
            for entry in entries:
                programs[entry],statistics[entry] = cache.compile(module,entry,
                    (profile,peephole,residency,native_mode),
                    lambda: npu_link.compile_program(module,entry,profile=profile,
                        peephole=peephole,residency=residency,native_mode=native_mode))
                # The module policy requires EVERY entry to be nonregressing.
                # One worse completed entry proves this candidate cannot win;
                # compiling its remaining entries cannot change that fact.
                if incumbent is not None and cost(statistics[entry]) > incumbent[entry]:
                    return programs,statistics,entry
            return programs,statistics,None
        if fusion != "auto":
            module = lower(fusion == "on", fuse=fusion == "on")
            programs,_,_ = compile_all(module)
            return module, programs, {"policy": fusion}
        from .npu_fusion import select
        records, seen, sources, winner, best_programs, best_costs = [], [], [], None, None, None
        def options():
            # Graph transformations remain TVM passes; only identical sources
            # are dropped before the cost-selected generic fusion probes.
            for attention in (False, "score", "full"):
                for gqa, producers in ((False, False), (True, False), (True, True)):
                    source = lower(attention, gqa, producers=producers)
                    if any(tvm.ir.structural_equal(source, old) for old in sources):
                        continue
                    sources.append(source)
                    for generic in (False, "vectors", "matmul"):
                        label = dict(attention=attention,repeated_gemv=gqa,
                                     matrix_producers=producers,generic_fusion=generic)
                        yield source,generic,label
        budget_exhausted = False
        with context.optimization_budget(profile.graph_search_seconds) as arm_budget:
            for source,generic,label in options():
                context.label = label
                context.report('candidate_begin', label=label)
                try:
                    context.check_budget()
                    candidate, regions = select(source, profile, entries,
                        matmul_epilogues=generic == "matmul") if generic else (source, None)
                    if any(tvm.ir.structural_equal(candidate, old) for old in seen):
                        context.report('candidate_end', label=label, status='duplicate', accepted=False)
                        continue
                    seen.append(candidate)
                    programs,stats,regressing_entry = compile_all(candidate,best_costs)
                    costs = {entry: cost(s) for entry, s in stats.items()}
                    selected = regressing_entry is None and (best_costs is None or (
                        all(costs[e] <= best_costs[e] for e in entries)
                        and any(costs[e] < best_costs[e] for e in entries)))
                    record = dict(label, status="noncompetitive" if regressing_entry else "legal",
                                  costs=stats, regions=regions,
                                  accepted=selected)
                    if regressing_entry:
                        record.update(regressing_entry=regressing_entry,
                            skipped_entries=[e for e in entries if e not in stats],
                            reason="completed entry is worse than incumbent; module policy forbids selection")
                    if selected:
                        winner, best_programs, best_costs = candidate, programs, costs
                        chosen = label
                        chosen_regions = regions
                        arm_budget()
                except OptimizationBudgetExceeded as error:
                    budget_exhausted = True
                    records.append(dict(label,status='budget_exhausted',reason=str(error),accepted=False))
                    context.report('candidate_end',label=label,status='budget_exhausted',accepted=False)
                    break
                except (LinkError, V09TirError) as error:
                    record = dict(label, status="unsupported", reason=str(error))
                records.append(record)
                context.report('candidate_end',label=label,status=record['status'],
                               accepted=record.get('accepted',False))
        if winner is None:
            reasons = "; ".join(r.get("reason", "") for r in records[:3])
            raise LinkError(f"no legal graph candidate for {entries}: {reasons}")
        decisions = dict(policy="bounded graph alternatives; emitted DMA/words/SRAM",
                         module_policy="all entries nonregressing; shared ABI chosen atomically",
                         pareto_scope="fully compiled legal candidates; early-rejected candidates have partial costs",
                         search_budget=dict(additional_seconds=profile.graph_search_seconds,
                             exhausted=budget_exhausted,
                             policy='arm after first legal module; stop between compiler operations'),
                         candidates=records, selected=chosen)
        legal = [(index, record) for index, record in enumerate(records) if record["status"] == "legal"]
        def metrics(record):
            return tuple(value for entry in entries for value in cost(record["costs"][entry]))
        decisions["pareto_candidates"] = [index for index, record in legal if not any(
            all(a <= b for a,b in zip(metrics(other), metrics(record)))
            and any(a < b for a,b in zip(metrics(other), metrics(record)))
            for _,other in legal)]
        for record in records:
            record["selected"] = all(record.get(key) == value for key,value in chosen.items())
        decisions.update({k: v for k, v in (chosen_regions or dict(regions=[], boundaries=[])).items()
                          if k != "policy"})
        for _, plan in best_programs.values():
            plan.fusion = decisions
            plan.compilation = context.describe()
        return winner, best_programs, decisions
