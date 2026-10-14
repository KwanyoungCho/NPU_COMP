"""Compile reuse must preserve ABI, constants, selection and executed outputs."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import tvm
from tvm import relax, tir
from tvm.script import tir as T

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from npu_compiler import npu_link, native, npu_build
from npu_compiler.device import PROFILES
from npu_compiler.npu_compilation import CompilationContext, EntryCache, compilation, entry_identity
from npu_compiler.npu_errors import LinkError
from npu_compiler.npu_memory_search import emit
from npu_compiler.tir_codegen_v09 import V09TirError
from test_architecture_contracts import lower


def source(decode_multiply=False, name='x', constant=1):
    bb=relax.BlockBuilder()
    for entry in ('prefill','decode'):
        x=relax.Var(name,relax.TensorStructInfo((64,), 'float16'))
        with bb.function(entry,[x]):
            with bb.dataflow():
                op=relax.op.multiply if entry=='decode' and decode_multiply else relax.op.add
                out=bb.emit_output(op(x,relax.const(np.full((64,),constant,'float16'))))
            bb.emit_func_output(out)
    return bb.finalize()


@T.prim_func
def invalid_chain(A:T.Buffer((64,64),'float16',scope='global.sram'),
                  C:T.Buffer((64,64),'float16',scope='global.sram')):
    T.evaluate(T.call_extern('int32','npu_gemm_acc',C.access_ptr('rw'),64,
                            A.access_ptr('r'),64,A.access_ptr('r'),64))


class CompileSpeed(unittest.TestCase):
    def test_native_invalid_chain_matches_python_rejection(self):
        sram={invalid_chain.buffer_map[p]:i*16384 for i,p in enumerate(invalid_chain.params)}
        for mode,error in [('use',LinkError),(False,V09TirError)]:
            with self.assertRaisesRegex(error,'FP32 matrix chain interrupted'):
                emit(invalid_chain,[0,0],sram,[32768,49152],{},None,mode)
        # Missing native coverage remains eligible for the Python fallback.
        with patch.object(native,'function',return_value=lambda *a:['unsupported','coverage']):
            self.assertIsNone(native.codegen_kernel(invalid_chain,[0,0],sram,[],{},None))

    def test_unchanged_entry_reused_after_other_entry_changes_and_copies_are_private(self):
        a,b=lower(source()),lower(source(True))
        ctx=CompilationContext(); cache=EntryCache(ctx)
        def compile(mod):
            return cache.compile(mod,'prefill',(),lambda:npu_link.compile_program(mod,'prefill'))[0]
        first=compile(a)
        expected_words=list(first[0].words)
        first[0].words[0]=0
        first[1].address['corrupt']=123
        with patch.object(npu_link,'compile_program',side_effect=AssertionError('compiled twice')):
            hit=compile(b)
        self.assertEqual(ctx.entry_hits,1)
        self.assertEqual(hit[0].words,expected_words)
        self.assertNotIn('corrupt',hit[1].address)
        self.assertEqual(hit[1].func.params[0].name_hint,'x')
        x=np.linspace(-2,2,64).astype('float16')
        actual,_=npu_link.run_program(*hit,hit[1].func,{'x':x})
        np.testing.assert_array_equal(actual,(x+np.float16(1)).astype('float16'))
        hit[1].address['corrupt']=456
        self.assertNotIn('corrupt',compile(a)[1].address)

    def test_identity_includes_input_names_constants_attributes_and_settings(self):
        mod=lower(source())
        reference,key=entry_identity(mod,'prefill',('profile-a',))
        for other,settings in [(lower(source(name='renamed')),('profile-a',)),
                (lower(source(constant=2)),('profile-a',)),
                (mod.with_attr('npu.cache_abi','changed'),('profile-a',)),
                (mod,('profile-b',))]:
            changed,other_key=entry_identity(other,'prefill',settings)
            self.assertFalse(key==other_key and tvm.ir.structural_equal(reference,changed))
        diagnostic=mod.with_attr('npu.repeated_gemv',99)
        diagnostic=tvm.IRModule({gv:fn.with_attr('op_pattern',8) if isinstance(fn,tir.PrimFunc)
            else fn for gv,fn in diagnostic.functions.items()},attrs=diagnostic.attrs)
        changed,other_key=entry_identity(diagnostic,'prefill',('profile-a',))
        self.assertEqual(key,other_key)
        self.assertTrue(tvm.ir.structural_equal(reference,changed))

    def test_private_symbol_renumbering_reuses_words_and_relabels_plan(self):
        mod=lower(source())
        renames={gv:'renumbered_'+gv.name_hint for gv,fn in mod.functions.items()
                 if isinstance(fn,tir.PrimFunc)}
        renamed=mod.replace_global_vars(renames)
        ctx=CompilationContext(); cache=EntryCache(ctx)
        expected,_=cache.compile(mod,'prefill',(),lambda:npu_link.compile_program(mod,'prefill'))
        actual,_=cache.compile(renamed,'prefill',(),lambda:self.fail('renumbering recompiled'))
        self.assertEqual(ctx.entry_hits,1)
        self.assertEqual(expected[0].words,actual[0].words)
        self.assertTrue(all(name.startswith('renumbered_') for name,_,_ in actual[1].kernel_span))
        referenced=[]
        relax.analysis.post_order_visit(actual[1].func,
            lambda e:referenced.append(e) if isinstance(e,tvm.ir.GlobalVar) else None)
        self.assertTrue(all(gv in renamed.functions for gv in referenced))
        x=np.arange(64,dtype='float16')/64
        got,_=npu_link.run_program(*actual,actual[1].func,{'x':x})
        np.testing.assert_array_equal(got,x+np.float16(1))

    def test_real_llama_prefill_identity_survives_decode_gqa_rewrite(self):
        from npu_compiler.nn_models import llama
        from npu_compiler import tvm_pipeline,npu_legalize
        config=llama.benchmark_config()
        mod=tvm_pipeline.graph_prefix(True)(llama.build_generate(config,7,9,last_logits=True)[0])
        identities=[]
        for fold in (False,True):
            candidate=tvm_pipeline.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                lift_params=True,fuse=False,fuse_attention=False,fold_repeated_gemv=fold,
                prepared=True)(mod)
            identities.append(entry_identity(candidate,'prefill_cache',()))
        self.assertEqual(identities[0][1],identities[1][1])
        self.assertTrue(tvm.ir.structural_equal(identities[0][0],identities[1][0]))

    def test_hash_collisions_and_capacity_do_not_reuse_wrong_program(self):
        ctx=CompilationContext(); cache=EntryCache(ctx,max_entries=2)
        a,b=lower(source()),lower(source(constant=2))
        with patch.object(tvm.ir,'structural_hash',return_value=0):
            for mod in (a,b,a):
                cache.compile(mod,'prefill',(),lambda:npu_link.compile_program(mod,'prefill'))
        self.assertEqual(ctx.entry_hits,1)
        self.assertEqual(ctx.entry_misses,2)
        self.assertEqual(len(cache.entries),2)
        cache.max_entries=1
        c=lower(source(constant=3))
        cache.compile(c,'prefill',(),lambda:npu_link.compile_program(c,'prefill'))
        self.assertEqual(len(cache.entries),1)
        tiny=EntryCache(ctx,max_bytes=1)
        tiny.compile(a,'prefill',(),lambda:npu_link.compile_program(a,'prefill'))
        self.assertFalse(tiny.entries)

    def test_region_cost_does_not_repeat_identical_compile(self):
        from npu_compiler.npu_fusion import _probe
        mod=lower(source())
        with compilation() as ctx:
            first=_probe(mod,mod['prefill'],PROFILES['v09'])
            with patch.object(npu_link,'compile_program',side_effect=AssertionError('compiled twice')):
                second=_probe(mod,mod['prefill'],PROFILES['v09'])
            self.assertEqual(first,second)
            self.assertEqual(ctx.region_hits,1)

    def test_auto_selection_cached_and_uncached_execute_identically(self):
        results=[]
        for enabled in (False,True):
            with compilation(cache=enabled):
                results.append(npu_build.lower_and_select(source(True),PROFILES['v09'],
                    ['prefill','decode'],lift_params=False))
        self.assertEqual(results[0][2]['selected'],results[1][2]['selected'])
        for entry in ('prefill','decode'):
            expected=results[0][1][entry]
            actual=results[1][1][entry]
            self.assertEqual(expected[0].words,actual[0].words)
            x=np.arange(64,dtype='float16')/64
            before,_=npu_link.run_program(*expected,expected[1].func,{'x':x})
            after,_=npu_link.run_program(*actual,actual[1].func,{'x':x})
            np.testing.assert_array_equal(before,after)

    def test_failed_trials_count_toward_budget_and_legal_incumbent_survives(self):
        from npu_compiler import npu_schedule as schedule
        from test_architecture_contracts import graph
        mod=lower(graph([(64,128),(128,128)],relax.op.matmul))
        gv,prim=next((gv,fn) for gv,fn in mod.functions.items() if isinstance(fn,tir.PrimFunc))
        profile=replace(PROFILES['v09'],schedule_max_attempts=2)
        # The new space can reject every candidate; the independently legal
        # incumbent must still be returned after exactly two failed attempts.
        failed=[dict(mapping={'reject':i},tile_pad=False) for i in range(100)]
        original=schedule._measure_trial
        def trial(*args,**kwargs):
            if isinstance(kwargs.get('mapping'),dict) and 'reject' in kwargs['mapping']:
                raise LinkError('deliberately infeasible')
            return original(*args,**kwargs)
        detail={}
        with patch.object(schedule,'matmul_candidates',return_value=failed), \
             patch.object(schedule,'_measure_trial',side_effect=trial):
            scheduled,_,_=schedule._fit_matmul(mod,gv,prim,profile,24,6,detail)
        self.assertIsInstance(scheduled,tir.PrimFunc)
        self.assertEqual(detail['examined_candidates'],2)
        self.assertEqual(detail['stop_reason'],'attempt_budget')
        self.assertEqual(detail['selected']['origin'],'legacy_incumbent')
        self.assertEqual(sum(a.get('reason')=='deliberately infeasible' for a in detail['attempts']),2)

    def test_zero_graph_budget_returns_complete_baseline_and_resets_context(self):
        profile=replace(PROFILES['v09'],graph_search_seconds=0)
        mod=source(True)
        with compilation() as ctx:
            _,actual,decisions=npu_build.lower_and_select(mod,profile,['prefill','decode'],
                                                       lift_params=False)
            self.assertTrue(decisions['search_budget']['exhausted'])
            self.assertEqual(set(actual),{'prefill','decode'})
            ctx.check_budget()  # the exited selector must not poison later compilation
            _,baseline,_=npu_build.lower_and_select(mod,profile,['prefill','decode'],
                                                 fusion='off',lift_params=False)
        for entry in actual:
            self.assertEqual(actual[entry][0].words,baseline[entry][0].words)

    def test_budget_expiration_inside_optional_probe_keeps_complete_module(self):
        from npu_compiler.npu_compilation import OptimizationBudgetExceeded
        from npu_compiler import npu_fusion
        def expired(*a,**kw):
            raise OptimizationBudgetExceeded('test optional probe expiration')
        with patch.object(npu_fusion,'select',side_effect=expired):
            _,programs,decisions=npu_build.lower_and_select(source(),PROFILES['v09'],
                ['prefill','decode'],lift_params=False)
        self.assertTrue(decisions['search_budget']['exhausted'])
        self.assertTrue(all(asm.words[-1]==0xff for asm,_ in programs.values()))
        x=np.arange(64,dtype='float16')
        for asm,plan in programs.values():
            got,_=npu_link.run_program(asm,plan,plan.func,{'x':x})
            np.testing.assert_array_equal(got,x+np.float16(1))


if __name__=='__main__': unittest.main(verbosity=2)
