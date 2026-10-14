"""NPU schedule selection and actual SRAM footprint validation.

This module consumes legalized TIR and produces scheduled TIR plus its local
buffer requirements.  It does not bind global addresses or emit instructions.
"""
from tvm import tir

from . import npu_intrin
from .npu_errors import LinkError
from .npu_sram import SramLayout, plan_sram
from .npu_kernel import single_matmul, attention_kernel
from .npu_w8a8 import is_w8a8


def _is_matmul(prim):
    return single_matmul(prim) is not None


def _is_attention(prim):
    return attention_kernel(prim) is not None


def _schedule(module, gvar, prim, tile=64, stripe=None, mapping=None,
              tile_pad=False, trace_out=None):
    """Apply the NPU schedule this kernel needs; every kernel ends up staged.

    ``stripe`` narrows a non-matmul kernel to that many result rows at a time;
    None stages whole tensors, which is what every kernel that fits gets.
    """
    # A kernel schedule only needs its own PrimFunc. Giving Schedule every
    # layer's Relax/TIR increases analysis work without changing this recipe.
    import tvm
    module = tvm.IRModule({gvar:prim}, attrs=module.attrs)
    name = gvar.name_hint
    try:
        if is_w8a8(prim):
            import tvm
            from .npu_w8a8_schedule import schedule
            scheduled = schedule(prim, stripe)
            return tvm.transform.Sequential([
                tir.transform.LowerMatchBuffer(),
                tir.transform.ConvertBlocksToOpaque(),
            ])(tvm.IRModule({"main": scheduled}))["main"]
        if _is_attention(prim):
            # ``stripe`` is the query row block here; the whole-tensor form
            # does not exist for this kernel, so there is always a block
            scheduled = npu_intrin.schedule_attention_blocked(
                module, name, stripe or tile, tile)
            if scheduled is None:
                raise LinkError(f"{name}: attention has an unexpected shape")
        elif _is_matmul(prim):
            config = mapping if isinstance(mapping, dict) else None
            hold, panel = mapping if mapping and config is None else (None, None)
            scheduled = npu_intrin.schedule_matmul_sram(module, name, tile,
                hold, panel, tile_pad, config, return_schedule=trace_out is not None)
            if trace_out is not None:
                trace_out['trace'] = scheduled.trace
                scheduled = scheduled.mod
        elif stripe is None:
            scheduled = npu_intrin.schedule_generic_sram(module, name)
        else:
            scheduled = npu_intrin.schedule_generic_striped(module, name,
                                                            stripe)
            if scheduled is None:
                raise LinkError(f"{name}: cannot be striped")
    except LinkError:
        raise
    except (tir.ScheduleError, ValueError) as error:
        raise LinkError(f"{name}: {type(error).__name__}: "
                        f"{str(error).splitlines()[-1][:120]}") from error
    return _lower_scheduled(scheduled[name])


def _lower_scheduled(prim):
    """Shared lowering for a newly constructed or replayed TVM schedule."""
    # cache_read allocates a buffer with the producer's full shape even when a
    # single tile is staged, so real weight shapes would need tens of MiB of
    # SRAM.  The standard passes shrink each buffer to the region actually
    # accessed; they require init blocks lowered first, which turns reductions
    # into a guarded store the codegen understands.
    from tvm import IRModule, tir as _tir
    import tvm as _tvm
    single = IRModule({'main': prim})
    single = _tvm.transform.Sequential([
        _tir.transform.LowerInitBlock(),
        _tir.transform.PlanAndUpdateBufferAllocationLocation(),
        _tir.transform.ConvertBlocksToOpaque(),
        _tir.transform.CompactBufferAllocation(),
    ])(single)
    from .npu_accumulator import lower_stores, forward_output_tiles
    scheduled = forward_output_tiles(lower_stores(single['main']))
    return _coalesce_pointwise(scheduled)


def _coalesce_pointwise(prim):
    """Fuse contiguous pointwise axes with TVM; retain broadcasts/strided rows.

    Matrix tiles remain two-dimensional. Only complete serial nests whose
    loads and store are linear in the same flattened index are eligible.
    Reuse the existing scratch capacity rather than increasing SRAM pressure.
    """
    import tvm
    from math import gcd
    from .v09_contract import MAX_VLEN
    capacity = min(MAX_VLEN, _scratch_row(prim))
    analyzer = tvm.arith.Analyzer()
    sch = tir.Schedule(prim)
    names = []
    tir.stmt_functor.post_order_visit(prim.body, lambda n:
        names.append(n.name_hint) if isinstance(n, tir.Block)
        and isinstance(n.body, tir.BufferStore) and n.body.buffer.dtype == 'float16'
        and not isinstance(n.body.value, (tir.BufferLoad, tir.FloatImm, tir.IntImm))
        and not n.init and not n.match_buffers and not n.alloc_buffers else None)
    width = 0
    for name in names:
        block = sch.get_block(name)
        info = sch.get(block)
        loops = list(sch.get_loops(block))
        if len(loops) < 2 or info.iter_vars:
            continue
        nodes = [sch.get(loop) for loop in loops]
        start = len(nodes)-1
        while start and nodes[start-1].body.same_as(nodes[start]):
            start -= 1
        loops, nodes = loops[start:], nodes[start:]
        if len(nodes) < 2 or any(n.kind != tir.ForKind.SERIAL or
                not isinstance(n.extent, tir.IntImm) for n in nodes):
            continue
        realize = nodes[-1].body
        if not isinstance(realize, tir.BlockRealize) or not analyzer.can_prove(realize.predicate):
            continue
        ordinal, count = tir.const(0, nodes[-1].loop_var.dtype), 1
        for node in reversed(nodes):
            ordinal += (node.loop_var-node.min)*count
            count *= int(node.extent)
        vector_width = count if count <= capacity else gcd(count, capacity)
        if vector_width <= int(nodes[-1].extent):
            continue
        zero = {n.loop_var:n.min for n in nodes}
        def flat(buffer, indices):
            strides = list(buffer.strides)
            if not strides:
                stride = 1
                for dim in reversed(buffer.shape):
                    strides.insert(0,stride); stride *= dim
            return analyzer.simplify(sum(i*s for i,s in zip(indices,strides)))
        store = info.body
        target = flat(store.buffer,store.indices)
        def delta(index):
            return analyzer.simplify(index-tir.stmt_functor.substitute(index,zero))
        if not analyzer.can_prove_equal(delta(target),ordinal):
            continue
        loads, unsupported = [], []
        def inspect(node):
            if isinstance(node,tir.BufferLoad): loads.append(node)
            if isinstance(node,tir.Call) and node.op.name == 'tir.if_then_else': unsupported.append(node)
        tir.stmt_functor.post_order_visit(store.value,inspect)
        if unsupported or not loads:
            continue
        valid = True
        for load in loads:
            index = flat(load.buffer,load.indices)
            if (load.buffer.dtype != 'float16' or load.buffer.scope() != 'global.sram'
                    or not (analyzer.can_prove_equal(delta(index),ordinal)
                            or analyzer.can_prove_equal(delta(index),0))
                    or (load.buffer.data.same_as(store.buffer.data)
                        and not analyzer.can_prove_equal(index,target))):
                valid = False; break
        # Index-valued arithmetic outside buffer accesses is not pointwise
        # data arithmetic supported by the vector emitter.
        scalar = tir.stmt_functor.ir_transform(tir.Evaluate(store.value), None,
            lambda n: tir.const(0,n.dtype) if isinstance(n,tir.BufferLoad) else None,
            ['tir.BufferLoad']).value
        tir.stmt_functor.post_order_visit(scalar, lambda n:
            unsupported.append(n) if isinstance(n,tir.Var) and n in zero else None)
        if valid and not unsupported:
            fused = sch.fuse(*loops)
            if count > vector_width:
                sch.split(fused, [None, vector_width])
            width = max(width,vector_width)
    result = sch.mod['main']
    return result.with_attr('npu.pointwise_vector_width',width) if width else result


def _sram_buffers(prim):
    """The kernel's cache buffers, in allocation order.

    Buffers may be allocated at any block after
    PlanAndUpdateBufferAllocationLocation moves them inward, so collect them
    from the whole body rather than the root block. The USMP layout is
    computed once per scheduled candidate and relocated at each call site.
    """
    buffers = []
    seen = set()

    def visit(node):
        if isinstance(node, tir.Block):
            for buffer in node.alloc_buffers:
                if buffer.scope() == "local.accumulator":
                    continue  # explicit matrix register state, no SRAM storage
                if buffer.data not in seen:
                    seen.add(buffer.data)
                    buffers.append(buffer)

    tir.stmt_functor.post_order_visit(prim.body, visit)
    return buffers


def _place_sram(buffers, cursor=0, capacity=None):
    """Bind a planned local layout, or bump-place a legacy buffer list."""
    layout = buffers if isinstance(buffers, SramLayout) else plan_sram(None, buffers, reuse=False)
    return layout.place(cursor, capacity)


def _scratch_row(prim):
    """Longest row an expression temporary in this kernel may have to hold.

    ``_materialize`` serializes an expression over a whole row at a time, so a
    slot has to be at least as wide as the widest row the kernel touches.  A
    fixed size silently corrupted the neighbouring slot for anything wider --
    Qwen3's 9728-wide FFN against 8192-element slots, where Llama's 8192-wide
    one happened to fit exactly.
    """
    if prim.attrs and prim.attrs.get("npu.w8a8_lowered", 0):
        return 0  # all intrinsic workspace is explicit in the TIR
    widest = int(prim.attrs.get('npu.pointwise_vector_width',1)) if prim.attrs else 1
    buffers = list(prim.buffer_map.values())

    def visit(node):
        if isinstance(node, tir.Block):
            buffers.extend(node.alloc_buffers)

    tir.stmt_functor.post_order_visit(prim.body, visit)
    for buffer in buffers:
        if len(buffer.shape):
            widest = max(widest, int(buffer.shape[-1]))
    return widest


def _collect_constants(prim):
    """Scalar literals a pointwise block needs materialized in memory."""
    values = set()

    def visit(node):
        if isinstance(node, tir.BufferStore):
            def scan(expr):
                if isinstance(expr, tir.FloatImm):
                    values.add(float(expr.value))
            tir.stmt_functor.post_order_visit(node.value, scan)

    tir.stmt_functor.post_order_visit(prim.body, visit)
    return values


def _measure(planned, gvar, prim, profile, scratch_base, slot_count,
             stripe=None, mapping=None, tile_pad=False):
    """Return scheduled TIR, scratch width, reusable SRAM layout and peak."""
    return _measure_trial(planned, gvar, prim, profile, scratch_base, slot_count,
                          stripe, mapping, tile_pad)['measured']


def _measure_trial(planned, gvar, prim, profile, scratch_base, slot_count,
                   stripe=None, mapping=None, tile_pad=False, candidate_schedule=None):
    """Materialize a candidate once, independently of available capacity.

    Generic/attention/W8A8 stripes need this reuse as well as matmul panels:
    a different retained-value reservation often changes only the capacity.
    """
    from .npu_compilation import current
    context = current()
    capture = (candidate_schedule is not None or
        context is not None and context.schedule_database is not None and _is_matmul(prim))
    def create():
        from .npu_compilation import phase
        trace_out = {} if capture else None
        with phase('schedule_tir', kernel=gvar.name_hint, mapping=str(mapping), stripe=str(stripe)):
            if candidate_schedule is not None:
                from .npu_meta_schedule import restore_labels
                scheduled = restore_labels(_lower_scheduled(candidate_schedule.mod['main']), prim)
                trace_out['trace'] = candidate_schedule.trace
            elif capture:
                from .npu_meta_schedule import canonical_module, restore_labels
                module = canonical_module(prim, planned.attrs)
                scheduled = _schedule(module, module.get_global_var('main'), module['main'],
                    profile.tile, stripe, mapping, tile_pad, trace_out=trace_out)
                scheduled = restore_labels(scheduled, prim)
            else:
                scheduled = _schedule(planned, gvar, prim, profile.tile, stripe, mapping, tile_pad)
            elems = _scratch_row(scheduled)
        with phase('schedule_sram'):
            buffers = plan_sram(scheduled, _sram_buffers(scheduled), profile.reuse_sram)
        start = scratch_base + slot_count * elems * 4
        _, top = _place_sram(buffers, start)
        return dict(measured=(scheduled, elems, buffers, top), emitted=None,
                    trace=trace_out.get('trace') if trace_out is not None else None)
    frozen = tuple(sorted(mapping.items())) if isinstance(mapping, dict) else mapping
    settings = (profile.tile, profile.reuse_sram, scratch_base, slot_count,
                stripe, frozen, tile_pad, capture)
    return context.trial(prim, settings, create) if context else create()


def legacy_matmul_candidates(extents, tile=64):
    """Enumerate aligned divisor panels, ordered by operand DMA estimate.

    Padding is a producer placement decision: stationary mappings move that
    producer to the same loop as its operand.  Axis divisors are enumerated
    completely rather than obtained by repeated halving.  The cost accounts
    for real input elements; SRAM fit is measured on compacted, padded TIR.
    This is a restricted mapping search, not a cycle or global cost model.
    """
    rows, cols, depth = extents
    m, n, k = ((v + tile - 1) // tile * tile for v in extents)
    padded = (m, n, k) != extents
    candidates = []
    for hold, extent in (("A", m), ("B", n)):
        for panel in range(tile, extent + 1, tile):
            if extent % panel:
                continue
            cost = (rows * depth + depth * cols * (m // panel)
                    if hold == "A" else
                    depth * cols + rows * depth * (n // panel)) * 2
            candidates.append(dict(mapping=(hold, panel), tile_pad=padded,
                                   estimated_operand_load_bytes=cost,
                                   footprint_hint=(panel * k + k * tile + tile * tile) * 2))
            if hold == "B":
                candidates.append(dict(mapping=("B_stream", panel), tile_pad=padded,
                    estimated_operand_load_bytes=2*(depth*cols+rows*depth*(n//tile)),
                    footprint_hint=2*(panel*k+2*tile*tile)))
    # The whole padded operand can be cheaper than a panel when it fits.
    # Retain that legal schedule as a candidate, including for tiny decode.
    cost = 2 * (rows * depth * (1 if rows != m or depth != k else n // tile)
                + depth * cols * (1 if cols != n or depth != k else m // tile))
    candidates.append(dict(mapping=None, tile_pad=False,
                           estimated_operand_load_bytes=cost,
                           footprint_hint=2 * (
                               (m * k if rows != m or depth != k else tile * tile)
                               + (k * n if cols != n or depth != k else tile * tile)
                               + tile * tile)))
    if padded:
        candidates.append(dict(mapping=None, tile_pad=True,
                               estimated_operand_load_bytes=2 * (
                                   rows * depth * (n // tile) + depth * cols * (m // tile)),
                               footprint_hint=6 * tile * tile))
    return sorted(candidates, key=lambda c: (c["estimated_operand_load_bytes"],
                                            c["footprint_hint"]))


def matmul_candidates(extents, tile=64, resident=()):
    from .npu_schedule_space import candidates
    return candidates(extents, tile, resident)


def _fit_matmul(planned, gvar, prim, profile, scratch_base, slot_count,
                diagnostics=None):
    """Try legal candidates by estimated cost; validate actual SRAM usage."""
    extents = npu_intrin.matmul_extents(planned, gvar.name_hint)
    attempts = []
    best, best_cost, best_record, legal, words_examined = None, None, None, 0, 0
    if diagnostics is not None:
        diagnostics.update(kind="matmul", extents=extents,
                           cost_model="FP16 operand DMA per batch", attempts=attempts)
    if profile.schedule_search not in ('space', 'legacy', 'meta_schedule'):
        raise LinkError(f"unknown schedule search: {profile.schedule_search}")
    from .npu_compilation import current, phase
    context = current()
    database = context.schedule_database if context else None
    workload = None
    best_trace = None
    stored_cost = None
    if database is not None and profile.schedule_search != 'legacy':
        workload = database.workload(prim, planned.attrs, profile, scratch_base, slot_count)
        record = database.lookup(workload)
        if record is not None:
            import tvm
            from .npu_meta_schedule import restore_labels
            try:
                with phase('meta_schedule_replay', kernel=gvar.name_hint):
                    sch = tir.Schedule(workload)
                    record.trace.apply_to_schedule(sch, remove_postproc=False)
                    scheduled = restore_labels(_lower_scheduled(sch.mod['main']), prim)
                    elems = _scratch_row(scheduled)
                    buffers = plan_sram(scheduled, _sram_buffers(scheduled), profile.reuse_sram)
                    _, top = _place_sram(buffers, scratch_base + slot_count * elems * 4)
                    if top > profile.sram_nibbles:
                        raise LinkError('stored schedule exceeds current SRAM capacity')
                    stats = _probe_accumulator_chain(scheduled, buffers, elems, scratch_base, slot_count)
            except (LinkError, tvm.error.TVMError) as error:
                database.rejections += 1
                if diagnostics is not None:
                    diagnostics['database_rejection'] = str(error)
            else:
                database.hits += 1
                if database.mode == 'reuse' and diagnostics is not None:
                    diagnostics.update(search_space='meta_schedule_database', examined_candidates=0,
                        generated_candidates=0, legal_candidates=1, stop_reason='database_hit',
                        probe_words=stats['words'],
                        selection_policy='reuse previously selected trace under identical contract',
                        selected=dict(origin='meta_schedule_database', status='selected',
                            emitted=stats, sram_peak_nibbles=top, sram_allocator=buffers.algorithm))
                if database.mode == 'reuse':
                    return scheduled, elems, buffers
                # Tuning starts from the stored result and can only improve it.
                best = scheduled, elems, buffers
                best_cost = stored_cost = (stats['dma_bytes'], stats['words'], top)
                best_trace = record.trace
                best_record = dict(origin='meta_schedule_database', status='legal',
                    emitted=stats, sram_peak_nibbles=top, sram_allocator=buffers.algorithm)
                attempts.append(best_record)
    # The previous scheduler is an incumbent, not the definition of the new
    # space. Keep its legal result even when a bounded new search stops early.
    if profile.schedule_search != 'legacy' and best is None:
        from dataclasses import replace
        seed = {}
        try:
            best = _fit_matmul(planned,gvar,prim,replace(profile,schedule_search='legacy'),
                               scratch_base,slot_count,seed)
        except LinkError:
            pass  # a finer placement in the new space may still fit
        else:
            best_record = dict(seed['selected'], origin='legacy_incumbent', status='legal')
            stats = best_record['emitted']
            best_cost = (stats['dma_bytes'],stats['words'],best_record['sram_peak_nibbles'])
            attempts.append(best_record)
        if diagnostics is not None:
            diagnostics['incumbent_search'] = {key:seed.get(key) for key in (
                'examined_candidates','generated_candidates','stop_reason','attempt_limit')}
    info = single_matmul(prim)
    resident = []
    operands = []
    def find_resident(node):
        if isinstance(node, tir.Block) and node.name_hint == info.name:
            for i,value in enumerate((node.body.value.b.a, node.body.value.b.b)):
                while isinstance(value, tir.Cast):
                    value = value.value
                if isinstance(value, tir.BufferLoad):
                    operands.append(value.buffer)
                    if value.buffer.scope() == 'global.sram':
                        resident.append(i)
    tir.stmt_functor.post_order_visit(prim.body, find_resident)
    pool = (legacy_matmul_candidates(extents, profile.tile) if profile.schedule_search == 'legacy'
            else matmul_candidates(extents, profile.tile, resident))
    generated = len(pool)
    if profile.schedule_search != 'legacy':
        from .npu_schedule_space import prioritize
        m, n, k = extents
        expected = ((m,k), (k,n))
        parameter_data = {b.data for b in prim.buffer_map.values()}
        ordinary = (len(operands) == 2 and not operands[0].data.same_as(operands[1].data)
            and all(v % profile.tile == 0 for v in extents)
            and all(b.data in parameter_data and b.dtype == 'float16'
                    and b.scope() in ('global', 'global.sram')
                    and tuple(map(int,b.shape)) == shape for b,shape in zip(operands,expected)))
        # Only the fixed prefix is subtracted for the proof; scratch and
        # epilogue estimates are deliberately not used to reject a candidate.
        available = max(0, (profile.sram_nibbles-scratch_base)//2)
        pool, capacity_detail = prioritize(pool, available, prove_operand_bound=ordinary)
        if diagnostics is not None:
            diagnostics.update(capacity_detail, operand_bound_proven=ordinary,
                               candidate_capacity_bytes=available)
    if profile.schedule_search == 'meta_schedule':
        from .npu_meta_schedule import replay_candidates
        candidates = replay_candidates(prim, planned.attrs, profile, pool)
    else:
        candidates = pool
    if diagnostics is not None:
        diagnostics.update(search_space=profile.schedule_search, generated_candidates=generated)
        if profile.schedule_search == 'meta_schedule':
            diagnostics.update(strategy='TVM MetaSchedule ReplayFunc', seed=profile.schedule_seed,
                               latency_feedback=False)
    limit = profile.schedule_candidates if profile.schedule_search == 'legacy' else profile.schedule_space_trials
    if limit < 1:
        raise LinkError('schedule trial limit must be positive')
    attempt_limit = profile.schedule_max_attempts
    if attempt_limit is not None and attempt_limit < 1:
        raise LinkError('schedule_max_attempts must be positive or None')
    examined = 0
    stop_reason = 'exhausted'
    # Scale the work cap to the kernel itself. A large incumbent must not use
    # up the entire comparison budget with a single similarly sized trial.
    word_basis = best_record['emitted']['words'] if best_record is not None else 0
    word_budget = max(profile.schedule_probe_word_budget, word_basis * limit)
    try:
        for candidate in candidates:
            if attempt_limit is not None and examined >= attempt_limit:
                stop_reason = 'attempt_budget'
                break
            examined += 1
            record = {k: v for k, v in candidate.items() if k not in ('footprint_hint','_schedule')}
            attempts.append(record)
            try:
                if 'generation_error' in candidate:
                    raise LinkError(candidate['generation_error'])
                trial = _measure_trial(planned, gvar, prim, profile, scratch_base, slot_count,
                    mapping=candidate['mapping'], tile_pad=candidate['tile_pad'],
                    candidate_schedule=candidate.get('_schedule'))
                scheduled, elems, buffers, top = trial['measured']
            except LinkError as error:
                record.update(status="schedule_rejected", reason=str(error))
                continue
            record["sram_peak_nibbles"] = top
            record["sram_allocator"] = buffers.algorithm
            if top > profile.sram_nibbles:
                record.update(status="capacity_rejected", reason="exceeds SRAM capacity")
                continue
            try:
                if 'codegen_error' in trial:
                    raise LinkError(trial['codegen_error'])
                if trial['emitted'] is None:
                    trial['emitted'] = _probe_accumulator_chain(scheduled, buffers, elems, scratch_base, slot_count)
                stats = trial['emitted']
            except LinkError as error:
                trial['codegen_error'] = str(error)
                record.update(status="codegen_rejected", reason=str(error))
                continue
            record.update(status="legal", emitted=stats)
            score = (stats["dma_bytes"],stats["words"],top)
            legal += 1; words_examined += stats["words"]
            if not word_basis:
                word_basis = stats['words']
                word_budget = max(word_budget, word_basis * limit)
            if best_cost is None or score < best_cost:
                best, best_cost, best_record = (scheduled,elems,buffers), score, record
                best_trace = trial.get('trace')
            if legal >= limit or (legal >= min(2,limit) and words_examined >= word_budget):
                stop_reason = 'legal_budget' if legal >= limit else 'word_budget'
                break
    finally:
        if profile.schedule_search == 'meta_schedule':
            candidates.close()
    if stop_reason == 'exhausted' and attempt_limit is not None and examined >= attempt_limit:
        stop_reason = 'attempt_budget'
    if diagnostics is not None:
        diagnostics.update(examined_candidates=examined,attempt_limit=attempt_limit,
                           stop_reason=stop_reason,legal_candidates=legal,probe_words=words_examined,
                           effective_probe_word_budget=word_budget, probe_word_basis=word_basis)
    if best is not None:
        best_record["status"] = "selected"
        if diagnostics is not None:
            diagnostics.update(selected=dict(best_record),
                selection_policy="bounded emitted DMA/words/SRAM with aligned parameter bindings",
                legal_candidates=legal, probe_words=words_examined)
        if workload is not None and (stored_cost is None or best_cost < stored_cost):
            # Trial reuse retrieves the winning trace, including when the
            # bounded search kept its legacy incumbent. No other candidate is
            # persisted, and static costs are never stored as run_secs.
            if best_trace is None:
                trial = _measure_trial(planned, gvar, prim, profile, scratch_base, slot_count,
                    mapping=best_record['mapping'], tile_pad=best_record['tile_pad'])
                best_trace = trial['trace']
            database.commit(workload, best_trace, profile, replace_existing=True)
        return best
    if stop_reason == 'attempt_budget':
        raise LinkError(f"{gvar.name_hint}: schedule attempt budget exhausted "
                        f"({examined}/{len(pool)}); no legal schedule found within budget")
    reasons = "; ".join(f"{a['mapping']}: {a.get('reason')}" for a in attempts[-3:])
    raise LinkError(f"{gvar.name_hint}: no legal matmul schedule fits; {reasons}")


def _probe_accumulator_chain(prim, layout, elems, scratch_base, slot_count):
    """Exercise emission before accepting a schedule with multiple K tiles.

    Padding, dequantization and odd-cell DMA copies use the PE. A schedule
    must keep those outside its FP32 MAC chain. Relative, aligned addresses
    suffice here; final linking checks concrete view offsets again.
    """
    import numpy as np
    from .v09_asm import V09Asm
    from .tir_codegen_v09 import Walker, SramEmitter, V09TirError
    asm = V09Asm(); walker = Walker(asm, {}, SramEmitter(asm))
    walker.scratch_slots = tuple(scratch_base + i*elems*4 for i in range(slot_count))
    walker.constants = {v: 0 for v in _collect_constants(prim) | {1.0,2.0}}
    walker.one_fp32 = 0
    cursor = 0
    for param in prim.params:
        b = prim.buffer_map[param]
        walker.bases[b.data] = cursor//2
        walker.dtypes[b.data] = str(b.dtype)
        size = int(np.prod(list(map(int,b.shape)))) * np.dtype(str(b.dtype)).itemsize
        cursor += (size+3)//4*4
    sram, top = layout.place(scratch_base + slot_count*elems*4)
    # Externally retained inputs are not kernel-local allocations. Give them
    # disjoint synthetic SRAM bindings for the address-independent probe.
    from .npu_sram import size_nibbles, align
    for p in prim.params:
        b = prim.buffer_map[p]
        if b.scope() == 'global.sram':
            top = align(top)
            sram[b] = top
            top += size_nibbles(b)
    for b, offset in sram.items():
        walker.declare_sram(b,offset)
    try:
        import os
        from .npu_compilation import phase
        from .v09_stats import analyze
        from .peephole import eliminate_dead_stores
        words = None
        if os.environ.get("NPU_NATIVE", "1") != "0":
            from . import native
            with phase('schedule_probe_codegen'):
                words = native.codegen_kernel(prim, [walker.bases[prim.buffer_map[p].data] for p in prim.params],
                    sram, walker.scratch_slots, walker.constants, walker.one_fp32)
                if words is not None: words = list(map(int,words))
        if words is None:
            with phase('schedule_probe_python'):
                walker.run(prim,{})
            words = asm.words
        with phase('schedule_probe_peephole'):
            words = eliminate_dead_stores(words)
        with phase('schedule_probe_statistics'):
            return analyze(words)
    except V09TirError as error:
        raise LinkError(str(error)) from error


def _fit_attention(planned, gvar, prim, profile, scratch_base, slot_count,
                   diagnostics=None):
    """Pick the query row block, largest first.

    A bigger block reads K and V fewer times -- they are re-read once per
    block -- so this wants the largest that fits, the same direction as the
    matmul panel.  Unlike the panel there is no whole-tensor fallback to land
    on, so failing all the way down is a link error rather than a silent
    retreat.
    """
    rows = npu_intrin.attention_rows(planned, gvar.name_hint) or profile.tile
    floor = min(rows, profile.tile)   # an axis under a tile cannot be halved
    block, why = rows, "no block was tried"
    attempts = []
    if diagnostics is not None:
        diagnostics.update(kind="attention", attempts=attempts,
                           selection_policy="largest query block, halving on failure")
    while True:
        record = {"query_rows": block}
        attempts.append(record)
        try:
            scheduled, elems, buffers, top = _measure(
                planned, gvar, prim, profile, scratch_base, slot_count,
                stripe=block)
        except LinkError as error:
            why, top = str(error), None
            record.update(status="schedule_rejected", reason=why)
        else:
            record["sram_peak_nibbles"] = top
            record["sram_allocator"] = buffers.algorithm
            if top <= profile.sram_nibbles:
                record["status"] = "selected"
                if diagnostics is not None:
                    diagnostics["selected"] = dict(record)
                return scheduled, elems, buffers
            why = f"needs {top / 2 / 1024 / 1024:.2f} MiB"
            record.update(status="capacity_rejected", reason=why)
        if block <= floor:
            break
        block = max(floor, block // 2)
    # say which block failed and why: a schedule that silently falls back is
    # how the panel bug in report_0908 s8.1 stayed hidden
    raise LinkError(f"{gvar.name_hint}: attention does not fit down to a "
                    f"{floor}-row block -- {why}")


def _fit_kernel(planned, gvar, prim, profile, scratch_base, slot_count,
                diagnostics=None):
    """Schedule a kernel, narrowing its stripe until its buffers fit SRAM.

    The first attempt stages whole tensors, so a kernel that already fits is
    scheduled exactly as it was before this loop existed.  Only when it does
    not does a stripe appear.  What is checked is the placement after the TIR
    passes have sized the buffers, not an estimate -- pad_einsum's padding,
    CompactBufferAllocation's shrinking and the kernel's scratch width are
    only known then.
    """
    if is_w8a8(prim):
        from .npu_w8a8_schedule import workspace_bytes
        rows = int(prim.buffer_map[prim.params[0]].shape[0])
        k = int(prim.buffer_map[prim.params[0]].shape[1])
        n = int(prim.buffer_map[prim.params[1]].shape[1])
        panel = rows
        attempts = []
        available = profile.sram_bytes - (scratch_base + 7) // 8 * 4
        if workspace_bytes(panel, k, n) > available:
            attempts.append(dict(panel_rows=panel, status="capacity_rejected",
                                 method="explicit workspace bound",
                                 sram_peak_nibbles=scratch_base + workspace_bytes(panel,k,n)*2))
            maximum = (available - workspace_bytes(0, k, n)) // (3 * k + 4)
            panel = max(min(64, rows), min((rows - 1) // 64, maximum // 64) * 64)
        while True:
            scheduled, elems, buffers, top = _measure(
                planned, gvar, prim, profile, scratch_base, slot_count, stripe=panel)
            record = dict(panel_rows=panel, sram_peak_nibbles=top,
                          sram_allocator=buffers.algorithm)
            if buffers.baseline_nibbles != workspace_bytes(panel, k, n) * 2:
                raise LinkError("W8A8 workspace predictor disagrees with explicit TIR allocations")
            attempts.append(record)
            if top <= profile.sram_nibbles:
                record["status"] = "selected"
                if diagnostics is not None:
                    diagnostics.update(kind="w8a8_tir", contract_version=1,
                                       selection_policy="explicit workspace bound, actual TIR fit",
                                       attempts=attempts, selected=record,
                                       panel_rows=panel, row_panels=(rows + panel - 1) // panel,
                                       sram_peak_nibbles=top)
                return scheduled, elems, buffers
            record["status"] = "capacity_rejected"
            if panel <= min(64, rows):
                raise LinkError(f"{gvar.name_hint}: W8A8 tile exceeds SRAM capacity")
            panel = max(64, (panel - 1) // 64 * 64)
    if _is_attention(prim):
        return _fit_attention(planned, gvar, prim, profile, scratch_base,
                              slot_count, diagnostics)
    if _is_matmul(prim):
        return _fit_matmul(planned, gvar, prim, profile, scratch_base,
                           slot_count, diagnostics)
    attempts = []
    if diagnostics is not None:
        diagnostics.update(kind="generic", attempts=attempts,
                           selection_policy="whole tensor, descending rectangular stripes on capacity overflow")
    # Include the full row count: pulling producers under a single stripe can
    # still reduce lifetimes compared with whole-tensor staging.
    def stripes():
        yield None
        yield from npu_intrin.generic_stripes(planned, gvar.name_hint)
    last_error = None
    for stripe in stripes():
        try:
            scheduled, elems, buffers, top = _measure(
                planned, gvar, prim, profile, scratch_base, slot_count, stripe=stripe)
        except LinkError as error:
            if stripe is None:
                raise
            # One placement can repeat producers while a smaller aligned one
            # does not. Reject that placement, not all remaining stripes.
            last_error = str(error)
            attempts.append(dict(stripe_rows=stripe, status='schedule_rejected',
                                 reason=last_error))
            continue
        record = dict(stripe_rows=stripe, sram_peak_nibbles=top,
                      sram_allocator=buffers.algorithm)
        attempts.append(record)
        if top <= profile.sram_nibbles:
            record["status"] = "selected"
            if diagnostics is not None:
                diagnostics["selected"] = dict(record)
            return scheduled, elems, buffers
        record.update(status="capacity_rejected", reason="exceeds SRAM capacity")
    if last_error is not None:
        raise LinkError(f"{gvar.name_hint}: no supported rectangular stripe; {last_error}")
    raise LinkError(
        f"{gvar.name_hint}: needs "
        f"{top / 2 / 1024 / 1024:.2f} MiB of "
        f"{profile.sram_nibbles / 2 / 1024 / 1024:.2f} MiB and "
        f"cannot be narrowed further")
