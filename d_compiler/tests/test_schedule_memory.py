"""End-to-end contracts for schedule space and retained intermediate values."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tvm
from tvm import relax
from tvm import tir

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_architecture_contracts import graph, lower
from npu_compiler import npu_link
from npu_compiler.device import PROFILES
from npu_compiler.npu_schedule_space import candidates
from npu_compiler.v09_stats import analyze


class ScheduleMemory(unittest.TestCase):
    def test_operand_bound_is_conservative_and_hints_do_not_prune(self):
        from npu_compiler.npu_schedule_space import prioritize
        from npu_compiler import npu_schedule as schedule
        from npu_compiler.npu_compilation import compilation
        pool=candidates((128,256,128))
        ranked,detail=prioritize(pool,24000,prove_operand_bound=False)
        self.assertEqual(len(ranked),len(pool))
        pruned,detail=prioritize(pool,24000,prove_operand_bound=True)
        self.assertGreater(detail['capacity_pruned'],0)
        mod=lower(graph([(128,128),(128,256)],relax.op.matmul))
        gv,prim=next((gv,fn) for gv,fn in mod.functions.items() if isinstance(fn,tir.PrimFunc))
        rejected=[c for c in pool if c not in pruned]
        samples=[pool[0],pool[-1],pruned[0],pruned[-1],rejected[0],rejected[-1]]
        with compilation():
            for candidate in samples:
                measured=schedule._measure_trial(mod,gv,prim,PROFILES['v09'],24,6,
                    mapping=candidate['mapping'],tile_pad=False)['measured']
                self.assertLessEqual(sum(candidate['operand_cache_bytes']),measured[2].peak_nibbles//2)

    def test_equivalent_k_partitions_emit_identical_programs(self):
        from npu_compiler import npu_schedule as schedule
        from npu_compiler.npu_compilation import compilation
        from npu_compiler.npu_schedule_space import _iteration_signature,loop_spec
        config=dict(m_panel=64,n_panel=64,k_chunk=64,outer_order='mn',inner_order='mn',
                    a_at='m_panel',b_at='n_panel')
        profile=replace(PROFILES['v09'],schedule_space_trials=1,joint_memory=False)
        for rows in (7,64):
            mod=lower(graph([(rows,256),(256,128)],relax.op.matmul))
            words=[]
            for chunk in (64,128,256):
                candidate=dict(mapping=dict(config,k_chunk=chunk),tile_pad=rows==7)
                with compilation(), \
                     patch.object(schedule,'legacy_matmul_candidates',return_value=[]), \
                     patch.object(schedule,'matmul_candidates',return_value=[candidate]):
                    asm,_=npu_link.compile_program(mod,profile=profile,native_mode='compare')
                    words.append(asm.words)
            self.assertEqual(words[0],words[1]);self.assertEqual(words[1],words[2])
        # Copying once per K group remains a genuinely different placement.
        prefixes=[_iteration_signature(loop_spec((64,128,256),dict(config,k_chunk=k))[:5])
                  for k in (64,128,256)]
        self.assertEqual(len(set(prefixes)),3)

    def test_equivalent_spatial_partitions_emit_identical_programs(self):
        from npu_compiler import npu_schedule as schedule
        from npu_compiler.npu_compilation import compilation
        # The second spelling splits an otherwise identical contiguous N
        # traversal; cache placement is after BOTH parts of that N traversal.
        configurations=[dict(m_panel=128,n_panel=64,k_chunk=64,outer_order='mn',
                             inner_order='mn',a_at='m_panel',b_at='n_panel'),
                        dict(m_panel=128,n_panel=128,k_chunk=64,outer_order='mn',
                             inner_order='nm',a_at='m_panel',b_at='n_tile')]
        profile=replace(PROFILES['v09'],schedule_space_trials=1,joint_memory=False)
        for rows in (127,128):
            mod=lower(graph([(rows,128),(128,256)],relax.op.matmul))
            words=[]
            for config in configurations:
                with compilation(), \
                     patch.object(schedule,'legacy_matmul_candidates',return_value=[]), \
                     patch.object(schedule,'matmul_candidates',return_value=[dict(
                         mapping=config,tile_pad=rows!=128)]):
                    asm,_=npu_link.compile_program(mod,profile=profile,native_mode='compare')
                    words.append(asm.words)
            self.assertEqual(words[0],words[1])

    def test_small_word_setting_still_compares_multiple_legal_candidates(self):
        from npu_compiler.npu_compilation import compilation
        profile=replace(PROFILES['v09'],schedule_probe_word_budget=1,
                        schedule_candidates=1,schedule_space_trials=3,joint_memory=False)
        mod=lower(graph([(64,128),(128,128)],relax.op.matmul))
        with compilation():
            _,plan=npu_link.compile_program(mod,profile=profile,native_mode='compare')
        detail=next(iter(plan.schedules.values()))
        self.assertGreaterEqual(detail['legal_candidates'],2)
        self.assertGreater(detail['effective_probe_word_budget'],1)

    def compare(self, source, shapes, profile=None):
        mod=lower(source)
        profile=profile or PROFILES['v09']
        baseline=replace(profile,joint_memory=False,schedule_search='legacy')
        runs=[]
        rng=np.random.default_rng(926)
        values={name:rng.normal(0,.1,shape).astype('float16')
                for name,shape in zip(('x','w','v','mask'),shapes)}
        for target in (baseline,profile):
            asm,plan=npu_link.compile_program(mod,profile=target,residency=True,native_mode='compare')
            result,_=npu_link.run_program(asm,plan,plan.func,values)
            runs.append((result,analyze(asm.words,plan,detail=False),plan,asm))
        def exact(a,b):
            if isinstance(a,tuple):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):exact(x,y)
            else:np.testing.assert_array_equal(a.view('uint16'),b.view('uint16'))
        exact(runs[0][0],runs[1][0])
        return runs

    def test_space_contains_two_axis_blocks_and_intermediate_k_groups(self):
        space=candidates((256,256,512))
        self.assertTrue(any(c['mapping']['m_panel']==128 and c['mapping']['n_panel']==128
            and c['mapping']['k_chunk'] in (128,256) for c in space))
        self.assertGreater(len(space),4)

    def test_parameterized_loop_orders_and_k_groups_execute(self):
        shapes=[(128,256),(256,256)]
        mod=lower(graph(shapes,lambda x,w:relax.op.matmul(x,w)))
        profile=replace(PROFILES['v09'],joint_memory=False)
        rng=np.random.default_rng(926)
        values={name:rng.normal(0,.1,shape).astype('float16')
                for name,shape in zip(('x','w'),shapes)}
        asm,plan=npu_link.compile_program(mod,
            profile=replace(profile,schedule_search='legacy'),native_mode='compare')
        reference,_=npu_link.run_program(asm,plan,plan.func,values)
        # Force non-incumbent schedules: a winning incumbent would otherwise
        # hide a broken reorder or an incorrectly placed K-group copy.
        for outer,inner in (('mn','nm'),('nm','mn')):
            config=dict(m_panel=128,n_panel=128,k_chunk=128,
                        outer_order=outer,inner_order=inner,a_at='n_panel',b_at='k_group')
            candidate=dict(mapping=config,tile_pad=False,estimated_operand_load_bytes=0)
            with self.subTest(outer=outer,inner=inner), \
                 patch('npu_compiler.npu_schedule.legacy_matmul_candidates',return_value=[]), \
                 patch('npu_compiler.npu_schedule.matmul_candidates',return_value=[candidate]):
                asm,plan=npu_link.compile_program(mod,profile=profile,native_mode='compare')
                actual,_=npu_link.run_program(asm,plan,plan.func,values)
                np.testing.assert_array_equal(reference.view('uint16'),actual.view('uint16'))
                self.assertEqual(plan.schedules['matmul']['selected']['mapping'],config)

    def test_tiled_producer_is_retained_for_following_matmul(self):
        shapes=[(64,64),(64,128),(128,64)]
        old,new=self.compare(graph(shapes,lambda x,w,v:relax.op.matmul(relax.op.matmul(x,w),v)),shapes)
        self.assertLess(new[1]['dma_load_bytes'],old[1]['dma_load_bytes'])
        self.assertEqual(old[1]['dma_store_bytes']-new[1]['dma_store_bytes'],64*128*2)
        kept=new[2].joint_memory['retained_values']
        self.assertTrue(kept)
        self.assertEqual(kept[0]['size_bytes'],64*128*2)
        self.assertTrue(new[2].joint_memory['program_comparison']['selected'])
        self.assertEqual(new[1]['opcodes'].get('0x17',0),0,
                         'retained matrix output should not require row-wise vector copies')

    def test_shared_value_survives_two_consumers(self):
        shapes=[(64,128),(128,64),(128,64)]
        def op(x,w,v):
            shared=relax.op.add(x,x)
            return relax.op.add(relax.op.matmul(shared,w),relax.op.matmul(shared,v))
        old,new=self.compare(graph(shapes,op),shapes)
        self.assertLess(new[1]['dma_bytes'],old[1]['dma_bytes'])
        self.assertTrue(any(len(v['readers'])>=2 for v in new[2].joint_memory['retained_values']))

    def test_capacity_changes_producer_schedule_to_reserve_output(self):
        shapes=[(64,64),(64,128),(128,64)]
        old,new=self.compare(graph(shapes,lambda x,w,v:relax.op.matmul(relax.op.matmul(x,w),v)),
                             shapes,replace(PROFILES['v09'],sram_bytes=44*1024))
        self.assertLessEqual(new[1]['sram_peak_bytes'],44*1024)
        self.assertLess(new[1]['dma_bytes'],old[1]['dma_bytes'])
        producer=new[2].joint_memory['calls'][0]
        reference=old[2].schedules[producer['name']]['selected']
        self.assertLess(producer['selected']['sram_peak_nibbles'],reference['sram_peak_nibbles'])
        # Direct output placement can now make the old mapping fit too.
        # What matters is the measured workspace plus the retained result.
        self.assertLessEqual(producer['workspace_peak_bytes']+producer['retained_bytes'],44*1024)
        self.assertTrue(new[2].scheduled_calls[0].attrs['npu.forwarded_output_tiles'])

    def test_trial_reuse_preserves_capacity_dependent_program(self):
        from npu_compiler.npu_compilation import compilation
        shapes=[(64,64),(64,128),(128,64)]
        mod=lower(graph(shapes,lambda x,w,v:relax.op.matmul(relax.op.matmul(x,w),v)))
        words=[]
        for enabled in (False,True):
            with compilation(cache=enabled):
                programs=[]
                for capacity in (48*1024,44*1024):
                    asm,plan=npu_link.compile_program(mod,
                        profile=replace(PROFILES['v09'],sram_bytes=capacity),
                        residency=True,native_mode='compare')
                    programs.append(asm.words)
                words.append(programs)
                if enabled:self.assertGreater(plan.compilation['trial_hits'],0)
        self.assertEqual(words[0],words[1])

    def test_capacity_fallback_and_ragged_inputs(self):
        for shapes,capacity in [([(64,128),(128,64)],30000), ([(7,65),(65,7)],65536)]:
            old,new=self.compare(graph(shapes,lambda x,w:relax.op.matmul(relax.op.add(x,x),w)),
                                 shapes,replace(PROFILES['v09'],sram_bytes=capacity))
            self.assertLessEqual(new[1]['sram_peak_bytes'],capacity)

    def test_returned_intermediate_keeps_dram_store(self):
        shapes=[(64,128),(128,64)]
        def op(x,w):
            shared=relax.op.add(x,x)
            return relax.Tuple([shared,relax.op.matmul(shared,w)])
        old,new=self.compare(graph(shapes,op),shapes)
        self.assertEqual(old[1]['dma_store_bytes'],new[1]['dma_store_bytes'])

    def test_reshape_alias_reads_retained_value(self):
        shapes=[(64,128),(128,64)]
        def op(x,w):
            shared=relax.op.add(x,x)
            viewed=relax.op.reshape(relax.op.reshape(shared,(1,64,128)),(64,128))
            return relax.op.matmul(viewed,w)
        old,new=self.compare(graph(shapes,op),shapes)
        self.assertLess(new[1]['dma_store_bytes'],old[1]['dma_store_bytes'])

    def test_offset_views_use_the_right_retained_slice(self):
        shapes=[(128,64),(64,64)]
        def op(x,w):
            shared=relax.op.add(x,x)
            pieces=relax.op.split(shared,2,axis=0)
            first=relax.op.matmul(relax.TupleGetItem(pieces,0),w)
            second=relax.op.matmul(relax.TupleGetItem(pieces,1),w)
            return relax.op.add(first,second)
        old,new=self.compare(graph(shapes,op),shapes)
        self.assertLess(new[1]['dma_bytes'],old[1]['dma_bytes'])

    def test_snapshots_keep_original_memory_contract(self):
        shapes=[(64,64),(64,64)]
        mod=lower(graph(shapes,lambda x,w:relax.op.matmul(relax.op.add(x,x),w)))
        asm,plan=npu_link.compile_program(mod,residency=True,snapshot_at={0},native_mode='compare')
        self.assertEqual(plan.joint_memory['policy'],'disabled')
        self.assertIn('snapshots',plan.joint_memory['reason'])

    def test_repeated_calls_and_storage_versions(self):
        shapes=[(64,64)]
        def op(x):
            a=relax.op.add(x,x)
            b=relax.op.multiply(a,a)
            c=relax.op.add(b,b)
            return relax.op.add(c,a)
        old,new=self.compare(graph(shapes,op),shapes)
        self.assertLess(new[1]['dma_bytes'],old[1]['dma_bytes'])

    def test_inplace_consumer_cannot_redirect_its_next_version(self):
        from npu_compiler.npu_memory_search import discover
        def op(x):
            a=relax.op.add(x,x)
            b=relax.op.multiply(a,a)
            return relax.op.add(b,b)
        mod=lower(graph([(64,64)],op))
        _,plan=npu_link.compile_program(mod,
            profile=replace(PROFILES['v09'],joint_memory=False),native_mode='compare')
        # Model a read/write consumer in the alias analysis without relying
        # on LegalizeOps to generate an in-place primitive.
        working=tvm.IRModule(mod.functions,attrs=mod.attrs)
        gv=next(g for g in mod.get_global_vars() if g.name_hint=='multiply')
        prim=mod[gv]
        incoming=prim.buffer_map[prim.params[0]]
        output=prim.buffer_map[prim.params[-1]]
        def rewrite(node):
            if isinstance(node,tir.BufferStore) and node.buffer.same_as(output):
                return tir.BufferStore(incoming,node.value,node.indices)
        working[gv]=prim.with_body(tir.stmt_functor.ir_transform(prim.body,None,rewrite))
        _,values,reasons=discover(working,plan)
        self.assertFalse(any(v.key[0]==0 for v in values))
        self.assertTrue(any(r['reason']=='in-place consumer update' for r in reasons))


if __name__=='__main__': unittest.main(verbosity=2)
