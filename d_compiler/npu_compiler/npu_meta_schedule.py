"""Opt-in MetaSchedule search and storage for FP16 matmul schedules.

TVM owns workload equality, trace serialization and Relax task extraction.
The NPU still chooses schedules by emitted DMA/words/SRAM. Records deliberately
have run_secs=None: these are selected recipes, not hardware latency samples,
so standard latency-ranked query_schedule/GetTopK must not select them.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import tvm
from tvm import meta_schedule as ms


def canonical_module(prim, attrs=None):
    return tvm.IRModule({'main': prim.without_attr('global_symbol').without_attr('op_pattern')},
                        attrs=attrs)


def restore_labels(scheduled, original):
    for name in ('global_symbol', 'op_pattern'):
        if original.attrs and name in original.attrs:
            scheduled = scheduled.with_attr(name, original.attrs[name])
    return scheduled


def replay_candidates(prim, module_attrs, profile, pool):
    """TVM ReplayFunc samples the SAME normalized space as the ordered search.

    The callback reruns the NPU schedule rule for each sampled configuration.
    This intentionally uses ReplayFunc, not trace mutation: cache placement
    involves Python control flow after reading the categorical decision.
    A rejected schedule consumes one attempt. Empty postprocs avoid ReplayFunc's
    unbounded retry-on-postproc-failure loop. Latency is never fabricated.
    """
    from . import npu_intrin
    from .device import npu_target
    from .npu_errors import LinkError
    if profile.schedule_max_attempts is None or profile.schedule_max_attempts < 1:
        raise LinkError('MetaSchedule ReplayFunc requires a finite positive attempt limit')
    if not pool:
        return
    module = canonical_module(prim, module_attrs)
    latest = {}
    seen = set()
    def make_schedule(sch):
        index = int(sch.get(sch.sample_categorical(
            list(range(len(pool))), [1.0/len(pool)]*len(pool))))
        latest.clear()
        latest.update(pool[index])
        latest['sample_index'] = index
        if index in seen:
            latest['generation_error'] = 'duplicate sample; no schedule regenerated'
            return sch
        seen.add(index)
        try:
            return npu_intrin.schedule_matmul_sram(module,'main',profile.tile,
                tile_pad=pool[index]['tile_pad'],config=pool[index]['mapping'],
                return_schedule=True,schedule=sch)
        except (tir.ScheduleError, ValueError, LinkError) as error:
            latest['generation_error'] = str(error).splitlines()[-1][:240]
            return sch
    from tvm import tir
    context = ms.TuneContext(mod=module,target=npu_target(profile.model),
        space_generator=ms.space_generator.ScheduleFn(make_schedule,sch_rules=[],
            postprocs=[],mutator_probs={}),
        search_strategy=ms.search_strategy.ReplayFunc(),rand_state=profile.schedule_seed,
        num_threads=1,task_name='npu_matmul')
    context.pre_tuning(max_trials=profile.schedule_max_attempts,num_trials_per_iter=1,
                       design_spaces=[])
    try:
        while True:
            from .npu_compilation import current, phase
            if current(): current().check_budget()
            with phase('meta_schedule_generate'):
                batch = context.generate_measure_candidates()
            if batch is None: break
            result = dict(latest)
            if 'generation_error' not in result:
                result['_schedule'] = batch[0].sch
            # ReplayFunc only advances its trial counter here; it does not use
            # latency feedback. The common NPU evaluator selects by static cost.
            context.notify_runner_results(batch,[ms.runner.RunnerResult(None,None)])
            yield result
    finally:
        context.post_tuning()


class SelectedScheduleDatabase:
    """One writer per directory; compatible builds may reopen the JSON database.

    Exact workload matching includes scopes, profile, scratch requirements and
    module attributes. A source/TVM/ISA/native fingerprint isolates versions
    BEFORE TVM loads traces, including any TensorIntrin references in them.
    The latest accepted improvement is indexed for each execution contract.
    Old traces remain in TVM's append-only history, with no invented timings.
    """
    def __init__(self, directory, mode='reuse'):
        if mode not in ('reuse', 'tune'):
            raise ValueError('schedule_database_mode must be reuse or tune')
        self.mode = mode
        from .npu_provenance import describe
        from .npu_intrin import register_all
        register_all()
        provenance = describe()
        identity = {key: provenance[key] for key in (
            'compiler_source_sha256', 'tvm_version', 'tvm_commit',
            'contract_sha256', 'native_sha256')}
        identity['schema'] = 2
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.directory = Path(directory) / fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = ms.database.JSONDatabase(work_dir=str(self.directory),
                                                  module_equality='structural')
        self.records = {}
        for record in self.database.get_all_tuning_records():
            self._index(record)
        self.hits = self.misses = self.writes = self.rejections = 0
        self.task_inventories = []

    def _index(self, record):
        key = tvm.ir.structural_hash(record.workload.mod)
        bucket = self.records.setdefault(key, [])
        for i, old in enumerate(bucket):
            if tvm.ir.structural_equal(old.workload.mod, record.workload.mod):
                bucket[i] = record
                return
        bucket.append(record)

    def workload(self, prim, module_attrs, profile, scratch_base, slot_count):
        import os
        execution = asdict(profile)
        # Search policy determines how a recipe is found, not whether it can
        # execute here. Keep every other (including future unknown) field.
        for key in ('joint_memory', 'memory_beam_width', 'graph_search_seconds',
                    'schedule_candidates', 'schedule_space_trials', 'schedule_max_attempts',
                    'schedule_probe_word_budget', 'schedule_search', 'schedule_seed'):
            execution.pop(key)
        contract = dict(profile=execution, scratch_base=scratch_base,
                        slot_count=slot_count, native_probe=os.environ.get('NPU_NATIVE', '1'))
        return canonical_module(prim, module_attrs).with_attr(
            'npu.selected_schedule_contract', json.dumps(contract, sort_keys=True))

    def lookup(self, module):
        for record in self.records.get(tvm.ir.structural_hash(module), ()):
            if tvm.ir.structural_equal(record.workload.mod, module):
                return record
        self.misses += 1
        return None

    def commit(self, module, trace, profile, *, replace_existing=False):
        # The caller has revalidated the old trace and compared emitted costs.
        # Replacements are strictly better recipes or recovery from bad replay.
        for record in self.records.get(tvm.ir.structural_hash(module), ()):
            if tvm.ir.structural_equal(record.workload.mod, module) and not replace_existing:
                return
        from .device import npu_target
        workload = self.database.commit_workload(module)
        record = ms.database.TuningRecord(trace, workload, run_secs=None,
                                          target=npu_target(profile.model))
        self.database.commit_tuning_record(record)
        self._index(record)
        self.writes += 1

    def extract_tasks(self, module, entries, profile):
        """Inventory only: different SRAM call-site contracts remain separate."""
        from .device import npu_target
        from .npu_compilation import reachable_module
        tasks = ms.relax_integration.extract_tasks(
            reachable_module(module, entries), npu_target(profile.model),
            module_equality='structural')
        inventory = dict(unique_tasks=len(tasks), call_sites=sum(int(t.weight) for t in tasks),
            tasks=[dict(name=str(t.task_name), weight=int(t.weight)) for t in tasks])
        self.task_inventories.append(inventory)
        return inventory

    def describe(self):
        return dict(enabled=True, mode=self.mode, directory=str(self.directory), hits=self.hits,
            misses=self.misses, writes=self.writes, rejections=self.rejections,
            records=sum(map(len, self.records.values())),
            policy='selected FP16 matmul traces; exact contract; revalidate lowering/SRAM/codegen',
            latency_measured=False, task_inventories=list(self.task_inventories))
