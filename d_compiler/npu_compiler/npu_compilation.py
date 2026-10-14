"""Transient, structurally checked schedule reuse within one compilation.

Kernel recipes/costs are rebound at call sites. Complete entry artifacts may be
reused only for an identical reachable module and execution contract. All caches
are build-local; no checkpoint-dependent data survives the context.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter

import tvm

_ACTIVE = ContextVar('npu_compilation', default=None)


@dataclass(frozen=True)
class _RejectedTrial:
    reason: str


class OptimizationBudgetExceeded(RuntimeError):
    """Stop optional search; the graph selector already owns a legal module."""


class CompilationContext:
    def __init__(self, cache=True, progress=None, schedule_database=None, schedule_database_mode=None):
        self.cache = cache
        self.schedule_database = None
        if schedule_database_mode is not None and schedule_database is None:
            raise ValueError('schedule_database_mode requires schedule_database')
        if schedule_database is not None:
            from .npu_meta_schedule import SelectedScheduleDatabase
            self.schedule_database = SelectedScheduleDatabase(schedule_database, schedule_database_mode or 'reuse')
        self.recipes = {}
        self.schedule_trials = {}
        self.trial_hits = self.trial_misses = 0
        self.hits = self.misses = 0
        self.probe_costs = {}
        self.probe_hits = self.probe_misses = 0
        self.phases = {}
        self.phase_self = {}
        self.phase_calls = {}
        self._phase_stack = []
        self.progress = progress
        self.label = {}
        self.native_fallbacks = {}
        self.native_rejections = {}
        self.entry_hits = self.entry_misses = 0
        self.region_hits = self.region_misses = 0
        self.region_costs = {}
        self._optimization_deadline = None
        self.start = perf_counter()

    @contextmanager
    def phase(self, name, **metadata):
        self.check_budget()
        start = perf_counter()
        inherited = self._phase_stack[-1][2] if self._phase_stack else {}
        info = dict(inherited, **metadata)
        frame = [start, 0., info]
        self._phase_stack.append(frame)
        self.report('phase_begin', phase=name, **info)
        try:
            yield
        finally:
            duration = perf_counter()-start
            self._phase_stack.pop()
            if self._phase_stack: self._phase_stack[-1][1] += duration
            self.phases[name] = self.phases.get(name, 0.) + duration
            self.phase_self[name] = self.phase_self.get(name, 0.) + duration-frame[1]
            self.phase_calls[name] = self.phase_calls.get(name, 0) + 1
            self.report('phase_end', phase=name, seconds=duration, **info)

    def check_budget(self):
        if self._optimization_deadline is not None and perf_counter() >= self._optimization_deadline:
            raise OptimizationBudgetExceeded('additional graph optimization time budget exhausted')

    @contextmanager
    def optimization_budget(self, seconds):
        """Arm only after a legal module; restore the enclosing context on exit."""
        import math
        if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
            raise ValueError('graph_search_seconds must be finite and nonnegative, or None')
        previous = self._optimization_deadline
        self._optimization_deadline = None
        def arm():
            if seconds is not None and self._optimization_deadline is None:
                self._optimization_deadline = perf_counter()+seconds
                self.report('optimization_budget_started',seconds=seconds)
        try:
            yield arm
        finally:
            self._optimization_deadline = previous

    def report(self, event, **metadata):
        if self.progress is not None:
            self.progress(dict(event=event, elapsed_seconds=perf_counter()-self.start,
                               candidate=self.label, **metadata))

    def recipe(self, prim, profile, scratch_base, slot_count, create):
        # global_symbol and op_pattern only label the graph boundary. Neither
        # participates in a kernel schedule; everything else remains in the key.
        fn = prim.without_attr('global_symbol').without_attr('op_pattern')
        from dataclasses import replace
        # Graph search controls do not change a kernel recipe. Capacity and
        # scoped input contracts do, and remain in this key.
        local_profile = replace(profile, joint_memory=False, memory_beam_width=1,
                                graph_search_seconds=None)
        key = (local_profile, scratch_base, slot_count, tvm.ir.structural_hash(fn))
        if self.cache:
            for reference, result, diagnostics in self.recipes.get(key, ()):
                if tvm.ir.structural_equal(reference, fn):
                    self.hits += 1
                    return result, deepcopy(diagnostics)
        self.misses += 1
        with self.phase('schedule_and_sram', kernel=str(prim.attrs.get('global_symbol','kernel')) if prim.attrs else 'kernel'):
            result, diagnostics = create()
        if self.cache:
            self.recipes.setdefault(key, []).append((fn, result, deepcopy(diagnostics)))
        return result, diagnostics

    def describe(self):
        return dict(scope='one build; structural recipes, costs and identical entry artifacts',
                    elapsed_seconds=perf_counter()-self.start,
                    schedule_hits=self.hits, schedule_misses=self.misses,
                    trial_hits=self.trial_hits, trial_misses=self.trial_misses,
                    cached_recipes=sum(map(len, self.recipes.values())),
                    cost_probe_hits=self.probe_hits, cost_probe_misses=self.probe_misses,
                    native_fallbacks=dict(self.native_fallbacks),
                    native_rejections=dict(self.native_rejections),
                    entry_hits=self.entry_hits, entry_misses=self.entry_misses,
                    region_hits=self.region_hits, region_misses=self.region_misses,
                    seconds=dict(self.phases), self_seconds=dict(self.phase_self),
                    phase_calls=dict(self.phase_calls),
                    meta_schedule=self.schedule_database.describe() if self.schedule_database else
                        dict(enabled=False))

    def probe(self, prim, settings, create, enabled=True):
        """Share candidate costs, never bound instructions, across calls.

        A hit defers codegen until the winning program is known. The caller
        checks that winner's actual cost and retries without reuse on any
        mismatch. Settings include physical SRAM bindings and DMA alignment.
        """
        fn = prim.without_attr('global_symbol').without_attr('op_pattern')
        key = (settings, tvm.ir.structural_hash(fn))
        if self.cache and enabled:
            for reference, statistics in self.probe_costs.get(key, ()):
                if tvm.ir.structural_equal(reference, fn):
                    self.probe_hits += 1
                    return None, None, {}, deepcopy(statistics)
        self.probe_misses += 1
        result = create()
        if self.cache and enabled:
            self.probe_costs.setdefault(key, []).append((fn, deepcopy(result[3])))
        return result

    def region_cost(self, module, entry, settings, create):
        reference, key = entry_identity(module, entry, settings)
        # Region costs have no externally named runtime input ABI.
        key = (key[0], key[2], key[3])
        if self.cache:
            for old, statistics in self.region_costs.get(key, ()):
                if tvm.ir.structural_equal(old, reference):
                    self.region_hits += 1
                    return deepcopy(statistics)
        self.region_misses += 1
        statistics = create()
        if self.cache:
            self.region_costs.setdefault(key, []).append((reference, deepcopy(statistics)))
        return statistics

    def trial(self, prim, settings, create):
        """Reuse one schedule's measured layout across SRAM budget choices.

        Capacity is checked by the caller. Parameter scopes, tile, allocation
        policy and scratch requirements remain in the key. No bound addresses
        or final words are reused here.
        """
        from .npu_errors import LinkError
        fn=prim.without_attr('global_symbol').without_attr('op_pattern')
        key=(settings,tvm.ir.structural_hash(fn))
        if self.cache:
            for reference,result in self.schedule_trials.get(key,()):
                if tvm.ir.structural_equal(reference,fn):
                    self.trial_hits+=1
                    if isinstance(result, _RejectedTrial):
                        raise LinkError(result.reason)
                    return result
        self.trial_misses+=1
        try:
            result=create()
        except LinkError as error:
            # Store the reason, not an exception with live traceback frames.
            # Capacity failures are checked outside this pure trial cache.
            result = _RejectedTrial(str(error))
        if self.cache:self.schedule_trials.setdefault(key,[]).append((fn,result))
        if isinstance(result, _RejectedTrial):
            raise LinkError(result.reason)
        return result


@contextmanager
def compilation(cache=True, progress=None, schedule_database=None, schedule_database_mode=None):
    """Nested probes/link calls share the outer build's context safely."""
    previous = _ACTIVE.get()
    if previous is not None:
        if schedule_database is not None or schedule_database_mode is not None:
            raise ValueError('set schedule_database on the outermost compilation context')
        yield previous
        return
    context = CompilationContext(cache, progress, schedule_database, schedule_database_mode)
    token = _ACTIVE.set(context)
    try:
        yield context
    finally:
        _ACTIVE.reset(token)


def current():
    return _ACTIVE.get()


def phase(name, **metadata):
    context = current()
    return context.phase(name, **metadata) if context else nullcontext()


def reachable_module(module, entries):
    """Keep exactly the requested public roots with TVM's reachability pass.

    DeadCodeElimination also preserves every other global_symbol by design.
    Privatize those functions on a copy before using it for an isolated probe
    or host parameter transform. The caller's module is never changed.
    """
    from tvm import relax
    roots = set(entries)
    attrs = dict(module.attrs) if module.attrs else {}
    attrs.pop('npu.fusion_selection', None)
    private = tvm.IRModule({gv: fn if gv.name_hint in roots else fn.without_attr('global_symbol')
                           for gv, fn in module.functions.items()}, attrs=attrs)
    return relax.transform.DeadCodeElimination(list(entries))(private)


def entry_identity(module, entry, settings, private_names=None):
    """Compile identity, including named public ABI and all unknown attributes.

    op_pattern guides graph fusion, which has already finished at this boundary.
    repeated_gemv is a module-wide diagnostic count, not an execution contract.
    Neither is removed from the actual module or exported diagnostics.
    """
    from tvm import tir, relax
    isolated = reachable_module(module, [entry])
    attrs = dict(isolated.attrs) if isolated.attrs else {}
    attrs.pop('npu.repeated_gemv', None)
    isolated = tvm.IRModule({gv: fn.without_attr('op_pattern') if isinstance(fn,tir.PrimFunc)
                            else fn for gv,fn in isolated.functions.items()}, attrs=attrs)
    # GlobalVar names participate in TVM structural equality. A changed decode
    # can renumber identical prefill helpers. Canonicalize private symbols in
    # first-reference order, using TVM's cross-IR replacement for all uses.
    ordered, visited = [], set()
    def visit(gv):
        if gv in visited or gv not in isolated.functions: return
        visited.add(gv)
        if gv.name_hint != entry: ordered.append(gv)
        fn = isolated[gv]
        def collect(node):
            if isinstance(node,tvm.ir.GlobalVar): visit(node)
            elif isinstance(node,tir.Call) and isinstance(node.op,tvm.ir.GlobalVar): visit(node.op)
        if isinstance(fn,relax.Function): relax.analysis.post_order_visit(fn,collect)
        elif isinstance(fn,tir.PrimFunc): tir.stmt_functor.post_order_visit(fn.body,collect)
    visit(isolated.get_global_var(entry))
    prefix = '__npu_private_'
    while entry.startswith(prefix): prefix = '_'+prefix
    replacements = {gv:f'{prefix}{i}' for i,gv in enumerate(ordered)}
    if private_names is not None:
        private_names.update({canonical:gv for gv,canonical in replacements.items()})
    isolated = isolated.replace_global_vars(replacements)
    names = tuple(p.name_hint for p in isolated[entry].params)
    return isolated, (entry, names, settings, tvm.ir.structural_hash(isolated))


def _clone_metadata(value):
    """Copy mutable Python plans while sharing immutable TVM IR nodes."""
    memo, seen = {}, set()
    def collect(item):
        if id(item) in seen: return
        seen.add(id(item))
        if isinstance(item, tvm.runtime.Object):
            memo[id(item)] = item
        elif isinstance(item, dict):
            for k,v in item.items(): collect(k); collect(v)
        elif isinstance(item, (list,tuple,set,frozenset)):
            for child in item: collect(child)
        elif hasattr(item,'__dict__'):
            collect(vars(item))
    collect(value)
    return deepcopy(value,memo)


def _relabel_plan(plan, old_names, new_names, entry):
    """Align cached private function references and diagnostics with the caller."""
    from tvm import relax
    aliases = {old.name_hint:new_names[key] for key,old in old_names.items()}
    replacements = {}
    def collect(node):
        if isinstance(node,tvm.ir.GlobalVar) and node.name_hint in aliases:
            replacements[node] = aliases[node.name_hint]
    relax.analysis.post_order_visit(plan.func,collect)
    plan.func = tvm.IRModule({entry:plan.func}).replace_global_vars(replacements)[entry]
    labels = {name:gv.name_hint for name,gv in aliases.items()}
    for field in ('native','native_diagnostics','sram_peak','schedules','sram_layouts'):
        value = getattr(plan,field,{})
        setattr(plan,field,{labels.get(k,k):v for k,v in value.items()})
    plan.kernel_span = [(labels.get(name,name),a,b) for name,a,b in plan.kernel_span]
    def records(value):
        if isinstance(value,dict):
            if isinstance(value.get('name'),str): value['name'] = labels.get(value['name'],value['name'])
            for child in value.values(): records(child)
        elif isinstance(value,(list,tuple)):
            for child in value: records(child)
    for field in ('kernel_records','call_schedules','joint_memory','residency'):
        records(getattr(plan,field,None))
    for index,prim in getattr(plan,'scheduled_calls',{}).items():
        symbol = prim.attrs.get('global_symbol') if prim.attrs else None
        if symbol is not None and str(symbol) in labels:
            plan.scheduled_calls[index] = prim.with_attr('global_symbol',labels[str(symbol)])


class EntryCache:
    """Bounded LRU for graph selection; packed words and private plan copies.

    Only the graph selector owns this cache, and it dies at the end of a build.
    The limits bound packed instruction bytes and metadata entry count; they do
    not change the search or its selected result. Oversized entries just miss.
    """
    def __init__(self, context, max_bytes=256*1024*1024, max_entries=32):
        self.context, self.max_bytes, self.max_entries = context, max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = self.serial = 0

    def compile(self, module, entry, settings, create):
        import numpy as np
        from .v09_asm import V09Asm
        from .v09_stats import analyze
        names = {}
        with self.context.phase('entry_identity', entry=entry):
            reference, key = entry_identity(module,entry,settings,names)
        if self.context.cache:
            for token,(old,words,metadata,plan,statistics,old_names) in list(self.entries.items()):
                if token[0] == key and tvm.ir.structural_equal(old,reference):
                    self.entries.move_to_end(token)
                    self.context.entry_hits += 1
                    asm = V09Asm()
                    asm.__dict__.update(_clone_metadata(metadata))
                    asm.words = words.tolist()
                    restored = _clone_metadata(plan)
                    _relabel_plan(restored,old_names,names,entry)
                    restored.graph_optimizations = dict(repeated_gemv=int(
                        module.attrs.get('npu.repeated_gemv',0)) if module.attrs else 0)
                    self.context.report('entry_reused',entry=entry,words=len(words))
                    return (asm,restored),deepcopy(statistics)
        self.context.entry_misses += 1
        with self.context.phase('entry_compile',entry=entry):
            asm,plan = create()
        with self.context.phase('entry_statistics',entry=entry):
            statistics = analyze(asm.words,plan,detail=False)
        if self.context.cache and len(asm.words)*4 <= self.max_bytes and self.max_entries:
            words = np.asarray(asm.words,dtype='<u4')
            metadata = {k:v for k,v in vars(asm).items() if k != 'words'}
            while self.entries and (self.bytes+words.nbytes > self.max_bytes
                                    or len(self.entries)>=self.max_entries):
                _,record = self.entries.popitem(last=False)
                self.bytes -= record[1].nbytes
            self.entries[(key,self.serial)] = (reference,words,_clone_metadata(metadata),
                                               _clone_metadata(plan),deepcopy(statistics),names)
            self.serial += 1
            self.bytes += words.nbytes
        self.context.report('entry_compiled',entry=entry,words=len(asm.words))
        return (asm,plan),statistics
