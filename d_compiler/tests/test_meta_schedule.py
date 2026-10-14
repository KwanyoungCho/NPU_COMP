"""Small, bounded integration checks; no checkpoint or full-model compilation."""
from dataclasses import replace
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import unittest
from unittest.mock import patch

import numpy as np
import tvm
from tvm import relax, tir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu_compiler import npu_link, npu_schedule, npu_legalize, tvm_pipeline
from npu_compiler.device import PROFILES
from npu_compiler.npu_compilation import compilation
from npu_compiler.npu_memory_search import parameter_scopes
from npu_compiler.npu_meta_schedule import SelectedScheduleDatabase
from npu_compiler.v09_stats import analyze, check_counters
from test_architecture_contracts import graph, lower


PROFILE = replace(PROFILES['v09'], schedule_candidates=1, schedule_space_trials=2,
                  schedule_max_attempts=8, graph_search_seconds=0)


class MetaScheduleReuse(unittest.TestCase):
    def test_sampled_schedule_can_be_saved_and_replayed_without_search(self):
        mod=lower(graph([(64,128),(128,64)],relax.op.matmul))
        profile=replace(PROFILE,schedule_search='meta_schedule',schedule_seed=929)
        with tempfile.TemporaryDirectory() as root:
            # Force this check to exercise a sampled trace, not an incumbent
            # chosen by the old schedule rule.
            with compilation(schedule_database=root) as context:
                with patch.object(npu_schedule,'legacy_matmul_candidates',return_value=[]):
                    first=npu_link.compile_program(mod,profile=profile,native_mode='compare')
                self.assertGreater(context.schedule_database.writes,0)
                traces=context.schedule_database.database.get_all_tuning_records()
                self.assertTrue(any('sample_categorical' in str(r.trace) for r in traces))
            with compilation(schedule_database=root) as context:
                with patch('npu_compiler.npu_meta_schedule.replay_candidates',
                           side_effect=AssertionError('disk replay resampled')):
                    second=npu_link.compile_program(mod,profile=profile,native_mode='compare')
                self.assertGreater(context.schedule_database.hits,0)
                self.assertEqual(context.schedule_database.rejections,0)
            self.assertEqual(first[0].words,second[0].words)
            x=np.ones((64,128),dtype='float16');w=np.ones((128,64),dtype='float16')
            got,counters=npu_link.run_program(*second,second[1].func,dict(x=x,w=w))
            np.testing.assert_array_equal(got,np.full((64,64),128,dtype='float16'))
            check_counters(second[0].words,counters)

    def test_replayfunc_search_is_bounded_reproducible_and_executes(self):
        mod = lower(graph([(64,128),(128,64)],relax.op.matmul))
        profile = replace(PROFILE,schedule_search='meta_schedule',schedule_seed=929)
        programs=[]
        for _ in range(2):
            with compilation():
                asm,plan=npu_link.compile_program(mod,profile=profile,native_mode='compare')
                programs.append((asm,plan))
            detail=next(iter(plan.schedules.values()))
            self.assertEqual(detail['strategy'],'TVM MetaSchedule ReplayFunc')
            self.assertLessEqual(detail['examined_candidates'],profile.schedule_max_attempts)
        self.assertEqual(programs[0][0].words,programs[1][0].words)
        x=np.ones((64,128),dtype='float16'); w=np.ones((128,64),dtype='float16')
        out,counters=npu_link.run_program(*programs[0],programs[0][1].func,dict(x=x,w=w))
        np.testing.assert_array_equal(out,np.full((64,64),128,dtype='float16'))
        check_counters(programs[0][0].words,counters)

    def test_disk_replay_preserves_programs_outputs_and_contracts(self):
        # The repeated chain represents same-shape layer calls. The fused case
        # includes a multi-K-tile reduction and the FP16 epilogue boundary.
        shapes = [(64,128), (128,64), (64,64)]
        fused = tvm_pipeline.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
            fuse=True, lift_params=False, fuse_attention=False)(graph(shapes,
            lambda x,w,v: relax.op.multiply(relax.op.nn.silu(relax.op.matmul(x,w)),v)))
        repeated = lower(graph([(64,64)]*3, lambda x,w,v:
            relax.op.matmul(relax.op.matmul(relax.op.matmul(x,w),v),w)))
        padded = lower(graph([(7,128),(128,64)], relax.op.matmul))
        cases = [('repeated', repeated, [(64,64)]*3, True),
                 ('fused', fused, shapes, False),
                 ('padded', padded, [(7,128),(128,64)], False)]
        rows = []
        with tempfile.TemporaryDirectory() as root:
            for name, mod, dims, residency in cases:
                with self.subTest(case=name):
                    results = []
                    for mode in ('baseline', 'record', 'replay'):
                        start = perf_counter()
                        with compilation(schedule_database=root if mode != 'baseline' else None) as ctx:
                            if mode == 'record':
                                inventory = ctx.schedule_database.extract_tasks(mod, ['prefill'], PROFILE)
                                if name == 'repeated':
                                    self.assertEqual(inventory['unique_tasks'], 1)
                                    self.assertEqual(inventory['call_sites'], 3)
                            # Reopening the database must avoid ALL matmul
                            # candidate construction, not just hit a RAM cache.
                            with ExitStack() as patches:
                                if mode == 'replay':
                                    for function in ('matmul_candidates','legacy_matmul_candidates'):
                                        patches.enter_context(patch.object(npu_schedule,function,
                                            side_effect=AssertionError('replay searched candidates')))
                                asm,plan = npu_link.compile_program(mod, profile=PROFILE,
                                    residency=residency, native_mode='compare')
                            metrics = ctx.describe()
                            if mode == 'record':
                                self.assertGreater(metrics['meta_schedule']['writes'], 0)
                                self.assertTrue(all(r.run_secs is None
                                    for r in ctx.schedule_database.database.get_all_tuning_records()))
                            if mode == 'replay':
                                self.assertGreater(metrics['meta_schedule']['hits'], 0)
                                self.assertEqual(metrics['meta_schedule']['writes'], 0)
                                self.assertEqual(metrics['meta_schedule']['rejections'], 0)
                        results.append((asm, plan))
                        rows.append(dict(case=name, mode=mode, seconds=perf_counter()-start,
                            statistics=analyze(asm.words,plan,detail=False),
                            schedule_tir_calls=metrics['phase_calls'].get('schedule_tir',0),
                            meta_schedule=metrics['meta_schedule']))
                    self.assertEqual(results[0][0].words, results[1][0].words)
                    self.assertEqual(results[0][0].words, results[2][0].words)
                    rng = np.random.default_rng(929)
                    values = {key:rng.normal(0,.1,shape).astype('float16')
                              for key,shape in zip(('x','w','v'),dims)}
                    outputs = []
                    for asm,plan in (results[0],results[2]):
                        out,counters = npu_link.run_program(asm,plan,plan.func,values)
                        check_counters(asm.words,counters)
                        outputs.append(out)
                    np.testing.assert_array_equal(outputs[0].view('uint16'),outputs[1].view('uint16'))
        destination = os.environ.get('NPU_META_SCHEDULE_EVIDENCE')
        if destination:
            from npu_compiler.npu_provenance import describe
            path = Path(destination); path.parent.mkdir(parents=True, exist_ok=True)
            # Temporary paths have expired; do not present them as artifacts.
            for row in rows:
                row['meta_schedule'].pop('directory',None)
            path.write_text(json.dumps(dict(scope='three small FP16 cases; bounded schedules',
                full_llama_compiled=False, evolutionary_search_tested=False,
                instruction_words_identical=True, outputs_bit_exact=True,
                cmodel_counters_match=True, records=rows, provenance=describe()),indent=2)+'\n')

    def test_workload_contract_separates_execution_from_search_policy(self):
        mod = lower(graph([(64,64)]*2, relax.op.matmul))
        prim = next(fn for fn in mod.functions.values() if isinstance(fn,tir.PrimFunc))
        other = lower(graph([(64,128),(128,64)], relax.op.matmul))
        changed = next(fn for fn in other.functions.values() if isinstance(fn,tir.PrimFunc))
        with tempfile.TemporaryDirectory() as root:
            database = SelectedScheduleDatabase(root)
            def key(fn=prim, profile=PROFILE, scratch=24, slots=6):
                return database.workload(fn, mod.attrs, profile, scratch, slots)
            reference = key()
            self.assertTrue(tvm.ir.structural_equal(reference,
                key(prim.with_attr('global_symbol','another_layer'))))
            self.assertTrue(tvm.ir.structural_equal(reference,
                key(profile=replace(PROFILE, graph_search_seconds=600, memory_beam_width=8))))
            self.assertTrue(tvm.ir.structural_equal(reference,
                key(profile=replace(PROFILE, schedule_search='meta_schedule',schedule_seed=42,
                    schedule_candidates=8,schedule_space_trials=32,schedule_max_attempts=128,
                    schedule_probe_word_budget=1000000))))
            for variant in (key(parameter_scopes(prim,(0,))), key(changed),
                    key(profile=replace(PROFILE,sram_bytes=65536)), key(scratch=32),
                    key(slots=5), key(profile=replace(PROFILE,tile=32)),
                    key(profile=replace(PROFILE,numeric_policy='different')),
                    key(prim.with_attr('npu.test_contract','changed'))):
                self.assertFalse(tvm.ir.structural_equal(reference,variant))

    def test_tuning_improves_disk_recipe_then_fast_build_reuses_across_search_settings(self):
        mod=lower(graph([(128,128),(128,128)],relax.op.matmul))
        slow=dict(mapping=None,tile_pad=False)
        with tempfile.TemporaryDirectory() as root:
            # Start with a valid but deliberately poor tiled schedule.
            with compilation(schedule_database=root):
                with patch.object(npu_schedule,'legacy_matmul_candidates',return_value=[slow]), \
                     patch.object(npu_schedule,'matmul_candidates',return_value=[]):
                    first=npu_link.compile_program(mod,profile=PROFILE,native_mode='compare')
            with compilation(schedule_database=root,schedule_database_mode='tune') as ctx:
                improved=npu_link.compile_program(mod,profile=PROFILE,native_mode='compare')
                self.assertEqual(ctx.schedule_database.hits,1)
                self.assertEqual(ctx.schedule_database.writes,1)
            self.assertLess(analyze(improved[0].words)['dma_bytes'],analyze(first[0].words)['dma_bytes'])
            fast=replace(PROFILE,schedule_search='meta_schedule',schedule_seed=100,
                         schedule_space_trials=1,schedule_max_attempts=1,schedule_probe_word_budget=1)
            with compilation(schedule_database=root) as ctx:
                with patch.object(npu_schedule,'matmul_candidates',side_effect=AssertionError('replay searched')):
                    replay=npu_link.compile_program(mod,profile=fast,native_mode='compare')
                self.assertEqual(ctx.schedule_database.hits,1)
            self.assertEqual(replay[0].words,improved[0].words)
            # A subsequent unsuccessful tuning run must not overwrite the winner.
            with compilation(schedule_database=root,schedule_database_mode='tune') as ctx:
                with patch.object(npu_schedule,'matmul_candidates',return_value=[slow]):
                    kept=npu_link.compile_program(mod,profile=PROFILE,native_mode='compare')
                self.assertEqual(ctx.schedule_database.writes,0)
            self.assertEqual(kept[0].words,improved[0].words)
            rng=np.random.default_rng(930)
            values=dict(x=rng.normal(0,.1,(128,128)).astype('float16'),
                        w=rng.normal(0,.1,(128,128)).astype('float16'))
            outputs=[]
            for asm,plan in (first,replay):
                got,counters=npu_link.run_program(asm,plan,plan.func,values)
                check_counters(asm.words,counters);outputs.append(got)
            np.testing.assert_array_equal(outputs[0].view('uint16'),outputs[1].view('uint16'))

    def test_nested_context_reuses_database_and_rejects_reconfiguration(self):
        with tempfile.TemporaryDirectory() as root:
            with compilation(schedule_database=root) as outer:
                with compilation() as inner:
                    self.assertIs(inner,outer)
                with self.assertRaisesRegex(ValueError,'outermost'):
                    with compilation(schedule_database=root):
                        self.fail('nested reconfiguration was accepted')


if __name__ == '__main__':
    unittest.main()
