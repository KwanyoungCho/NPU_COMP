"""Functional gates for views, compilation reuse and precise SRAM effects."""
import json
import sys
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import tvm
from tvm import relax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu_compiler import npu_link as L, npu_memplan as M, npu_target as Target
from npu_compiler import npu_generate as G, npu_fusion as F
from npu_compiler.npu_compilation import compilation
from npu_compiler.npu_residency import forward
from npu_compiler.v09_effects import WriteEffects
from npu_compiler.v09_asm import V09Asm
from npu_compiler.isa_v09 import SRC1, DST, DT_FP16, DT_FP32
from npu_compiler.v09_runtime import run
from npu_compiler.v09_stats import analyze, check_counters, cost
from npu_compiler.device import PROFILES
from test_compiler_contracts import lower, matmul
from test_completion_contracts import chain


def view_graph():
    bb = relax.BlockBuilder()
    x = relax.Var('x', relax.TensorStructInfo((3, 5), 'float16'))
    with bb.function('prefill', [x]):
        with bb.dataflow():
            # Keep a reshaped producer alive across additional allocations.
            a = bb.emit(relax.op.add(x, x))
            v = bb.emit(relax.op.reshape(a, (5, 3)))
            b = bb.emit(relax.op.negative(x))
            c = bb.emit(relax.op.multiply(b, b))
            d = bb.emit(relax.op.reshape(c, (5, 3)))
            z = bb.emit(relax.op.add(v, d))
            out = bb.emit_output(relax.op.reshape(z, (3, 5)))
        bb.emit_func_output(out)
    return lower(bb.finalize())


class OptimizationTests(unittest.TestCase):
    def test_repeated_gemv_uses_broadcast_and_actual_program_cost(self):
        from npu_compiler.npu_graph_opt import FoldRepeatedGemv
        for kv,group,k,n in ((1,2,2,2),(1,3,2,3),(2,2,16,130),(2,4,64,7)):
            with self.subTest(shape=(kv,group,k,n)):
                h=kv*group
                bb=relax.BlockBuilder()
                a=relax.Var('a',relax.TensorStructInfo((h,1,k),'float16'))
                b=relax.Var('b',relax.TensorStructInfo((kv,k,n),'float16'))
                with bb.function('prefill',[a,b]):
                    with bb.dataflow():
                        repeated=bb.emit(relax.op.repeat(b,group,axis=0))
                        out=bb.emit_output(relax.op.matmul(a,repeated))
                    bb.emit_func_output(out)
                raw=bb.finalize()
                old=Target.build(raw,fusion='off')
                new=Target.build(raw,fusion='auto')
                self.assertLessEqual(cost(analyze(new.words,new.plan)),cost(analyze(old.words,old.plan)))
                rng=np.random.default_rng(11)
                vals={'a':rng.normal(0,.1,(h,1,k)).astype('float16'),
                      'b':rng.normal(0,.1,(kv,k,n)).astype('float16')}
                before,_=old.run(vals,(h,1,n));after,_=new.run(vals,(h,1,n))
                np.testing.assert_array_equal(before.view('uint16'),after.view('uint16'))
                L.compile_program(new.module,native_mode='compare')
                if k==16:
                    # The repeated tensor can now stay in SRAM. Auto may
                    # prefer that program over the graph rewrite; compare the
                    # actual program cost above, and test the rewrite itself
                    # independently of the new storage choice.
                    legacy=replace(PROFILES['v09'],joint_memory=False,schedule_search='legacy')
                    folded=Target.build(raw,fusion='auto',profile=legacy)
                    self.assertEqual(folded.plan.graph_optimizations['repeated_gemv'],1)
                    folded_value,_=folded.run(vals,(h,1,n))
                    np.testing.assert_array_equal(before.view('uint16'),folded_value.view('uint16'))
                # A second, non-matmul consumer retains the repeated tensor.
                # Verify the rewrite does not delete an externally visible value.
                sharedbb=relax.BlockBuilder()
                with sharedbb.function('prefill',[a,b]):
                    with sharedbb.dataflow():
                        repeated=sharedbb.emit(relax.op.repeat(b,group,axis=0))
                        value=sharedbb.emit(relax.op.matmul(a,repeated))
                        sharedout=sharedbb.emit_output(relax.Tuple([value,repeated]))
                    sharedbb.emit_func_output(sharedout)
                sharedmod=sharedbb.finalize()
                rewritten=FoldRepeatedGemv()(sharedmod)
                names=[]
                relax.analysis.post_order_visit(rewritten['prefill'].body,lambda e:names.append(e.op.name)
                    if isinstance(e,relax.Call) and isinstance(e.op,tvm.ir.Op) else None)
                self.assertIn('relax.repeat',names)

    def test_views_preserve_lifetime_shape_and_result_bits(self):
        mod = view_graph()
        values = {'x': np.linspace(-1, 1, 15).astype('float16').reshape(3, 5)}
        results = []
        programs = []
        for views in (False, True):
            a, p = L.compile_program(mod, views=views, native_mode='compare')
            results.append(L.run_program(a, p, p.func, values, (3, 5))[0])
            programs.append((a, p))
        np.testing.assert_array_equal(results[0].view('uint16'), results[1].view('uint16'))
        before, after = programs
        self.assertTrue(after[1].views)
        self.assertLess(after[0].kernel_count, before[0].kernel_count)
        self.assertLess(analyze(after[0].words)['dma_bytes'], analyze(before[0].words)['dma_bytes'])
        for view in after[1].views:
            self.assertEqual(after[1].address[view['name']], after[1].address[view['source']])
        # An already-formed Relax reshape can also be a function result.
        x = relax.Var('x', relax.TensorStructInfo((3,5), 'float16'))
        bb = relax.BlockBuilder()
        with bb.function('prefill',[x]):
            with bb.dataflow():
                v = bb.emit_output(relax.op.reshape(x,(5,3)))
            bb.emit_func_output(v)
        a,p = L.compile_program(bb.finalize())
        self.assertEqual(a.kernel_count,0)
        self.assertEqual(p.address[L.output_var(p.func)],p.address['x'])
        got,_ = L.run_program(a,p,p.func,values,(5,3))
        np.testing.assert_array_equal(got,values['x'].reshape(5,3))
        raw=np.array([0,0x8000,0x7c01,0x7fff,0xfc01]*3,dtype='uint16').view('float16').reshape(3,5)
        got,_=L.run_program(a,p,p.func,{'x':raw},(5,3))
        np.testing.assert_array_equal(got.view('uint16'),raw.reshape(5,3).view('uint16'))

    def test_recipe_cache_reuses_structure_but_separates_profile_and_pool(self):
        mod = lower(chain())
        with compilation(cache=False):
            baseline = Target.build(chain()).words
        with compilation() as ctx:
            a,p = L.compile_program(mod)
            hits = ctx.hits
            # JSON round trip changes object identities, retaining TIR semantics.
            clone = tvm.ir.load_json(tvm.ir.save_json(mod))
            b,q = L.compile_program(clone)
            self.assertEqual(a.words,b.words)
            self.assertGreater(ctx.hits,hits)
            misses = ctx.misses
            L.compile_program(clone,profile=replace(PROFILES['v09'],sram_bytes=1024*1024))
            self.assertGreater(ctx.misses,misses)
            self.assertEqual(Target.build(chain()).words,baseline)
            json.dumps(ctx.describe())
        # Scratch prefix belongs to the key even when the PrimFunc is identical.
        gv = next(gv for gv,fn in mod.functions.items() if isinstance(fn,tvm.tir.PrimFunc))
        with compilation() as ctx:
            for base in (16,24,16):
                ctx.recipe(mod[gv],PROFILES['v09'],base,6,lambda: ((base,),{}))
            self.assertEqual((ctx.hits,ctx.misses),(1,2))
        other=mod[gv].with_attr('npu.test_distinct_contract',1)
        with compilation() as ctx, patch.object(tvm.ir,'structural_hash',return_value=0):
            first,_=ctx.recipe(mod[gv],PROFILES['v09'],16,6,lambda: ((1,),{}))
            second,_=ctx.recipe(other,PROFILES['v09'],16,6,lambda: ((2,),{}))
            self.assertEqual((first,second),((1,),(2,)))
            self.assertEqual(ctx.hits,0)

    def test_generation_guard_rejects_one_regressing_entry(self):
        m = chain()
        m['decode'] = chain()['prefill'].with_attr('global_symbol','decode')
        m = lower(m)
        original = L.compile_program
        def penalize(module,entry='prefill',**kwargs):
            asm,plan = original(module,entry,**kwargs)
            if entry == 'decode' and any(r['name'].startswith('fused_') for r in plan.kernel_records):
                # A target cost regression in one device entry must reject the
                # shared candidate even when another entry improves.
                asm.words += [0xa0,0,65535,(65535<<16)|65535]
            return asm,plan
        with patch.object(L,'compile_program',side_effect=penalize):
            selected,programs,decisions = F.select_programs(m,PROFILES['v09'],['prefill','decode'])
        self.assertTrue(all(r['selected']=='unfused' for r in decisions['programs'].values()))
        self.assertFalse(decisions['programs']['decode']['nonregressing'])
        self.assertTrue(tvm.ir.structural_equal(selected,m))
        # The returned executable is the already-measured unfused program.
        a,p=programs['decode']
        expected,ref=L.compile_program(m,'decode',residency=True)
        self.assertEqual(a.words,expected.words)
        from npu_compiler.npu_errors import LinkError
        with patch.object(F,'select',side_effect=LinkError('unsupported graph candidate')):
            selected,programs,decisions=F.select_programs(m,PROFILES['v09'],['prefill','decode'])
        self.assertEqual(decisions['programs']['decode']['status'],'unsupported')
        self.assertTrue(tvm.ir.structural_equal(selected,m))
        with patch.object(F,'select',side_effect=RuntimeError('internal backend bug')):
            with self.assertRaisesRegex(RuntimeError,'internal backend bug'):
                F.select_programs(m,PROFILES['v09'],['prefill','decode'])

    def test_effects_use_result_length_and_preserve_disjoint_cached_cells(self):
        asm=V09Asm()
        asm.gload(0,16,0,1,16)
        asm.gstore(32,16,0,1,16)
        boundary=len(asm.words)
        asm.vlen(16);asm.addr(SRC1,0);asm.shape_dt(SRC1,1,16,1,DT_FP16)
        asm.load(0,SRC1);asm.v_reduce_sum()
        # Reducing created exactly one FP32 result, despite this new vlen.
        asm.vlen(128);asm.addr(DST,0);asm.shape_dt(DST,1,128,1,DT_FP32);asm.save(0)
        asm.gload(40,8,64,1,8)  # disjoint right half survives; same address
        asm.gstore(64,8,64,1,8)
        asm.gload(32,1,0,1,1)   # overwritten first cell must be fetched
        asm.gstore(72,1,0,1,1)
        end=len(asm.words);asm.halt()
        records=[dict(index=0,span=(0,boundary),arguments=[]),
                 dict(index=1,span=(boundary,end),arguments=[])]
        words,_,info=forward(asm.words,records)
        self.assertEqual(info['saved_load_bytes'],32)
        self.assertEqual(info['precise_writes'],1)
        image=np.zeros(80,dtype='<u4');image[:16]=np.arange(32,dtype='float16').view('<u4')
        a,_=run(asm.words,image);b,counters=run(words,image)
        np.testing.assert_array_equal(a[-1],b[-1]);check_counters(words,counters)

    def test_matrix_stride_and_quantization_effect_bounds(self):
        e=WriteEffects();a=V09Asm()
        a.vlen(8);a.addr(SRC1,0);a.shape_dt(SRC1,2,4,0,DT_FP16)
        a.shape_dt(SRC1,2,2,1,DT_FP16);a.load(1,SRC1);a.m_mul(0,1)
        a.addr(DST,128);a.shape_dt(DST,2,8,0,DT_FP16);a.shape_dt(DST,2,2,1,DT_FP16)
        a.save(1)
        effects=[e.step(w) for w in a.words]
        self.assertEqual(effects[-1],(128,40))
        from npu_compiler.isa_v09 import DT_INT4
        a=V09Asm();a.addr(DST,132);a.shape_dt(DST,1,3,1,DT_INT4);a.vlen(3);a.vquant()
        for w in a.words:write=e.step(w)
        self.assertEqual(write,(132,3))
        self.assertIsNone(e.step(0xfe))
        self.assertIsNone(e.step(0x98))

    def test_w8a8_bound_avoids_linear_schedule_search(self):
        m=matmul(4097,128,192,quant=True)
        a,p=L.compile_program(m,profile=replace(PROFILES['v09'],sram_bytes=80000))
        record=next(iter(p.schedules.values()))
        self.assertEqual(record['panel_rows'],128)
        self.assertEqual(len(record['attempts']),2)
        self.assertLessEqual(max(p.sram_peak.values()),160000)

    def test_overlapping_global_store_rows_do_not_forward_an_old_version(self):
        a=V09Asm();a.gload(0,8,0,1,8);a.gstore(32,2,0,2,4)
        boundary=len(a.words)
        a.gload(32,4,0,1,4);a.gstore(64,4,0,1,4)
        end=len(a.words);a.halt()
        records=[dict(index=0,span=(0,boundary)),dict(index=1,span=(boundary,end))]
        words,_,info=forward(a.words,records)
        self.assertEqual(info['saved_load_bytes'],0)
        image=np.arange(80,dtype='<u4')
        before,_=run(a.words,image);after,_=run(words,image)
        np.testing.assert_array_equal(before[-1],after[-1])

    def test_runtime_bindings_use_names_not_dictionary_order(self):
        a=relax.Var('cos',relax.TensorStructInfo((1,4),'float16'))
        b=relax.Var('sin',relax.TensorStructInfo((1,4),'float16'))
        fn=relax.Function([a,b],relax.op.subtract(a,b),a.struct_info)
        values=G._bind_inputs(fn,{'sin':2,'cos':1},[])
        self.assertEqual([values[p.name_hint] for p in fn.params],[1,2])
        with self.assertRaisesRegex(ValueError,'runtime inputs'):
            G._bind_inputs(fn,{'cos':1,'missing':2},[])

    def test_parameter_cache_includes_reachable_tir_bodies(self):
        from tempfile import TemporaryDirectory
        from test_params_cache import _model, TINY
        m,weights=_model(TINY,seed=3)
        host=G._host_transform_module(m,'decode')
        self.assertNotIn('decode',[g.name_hint for g in host.functions])
        gv=next(g for g,f in host.functions.items() if isinstance(f,tvm.tir.PrimFunc))
        prim=m[gv]
        def negate(node):
            if isinstance(node,tvm.tir.BufferStore):
                return tvm.tir.BufferStore(node.buffer,-node.value,node.indices)
            return None
        modified=tvm.IRModule(m.functions,attrs=m.attrs)
        modified[gv]=prim.with_body(tvm.tir.stmt_functor.ir_transform(
            prim.body,None,negate,['tir.BufferStore']))
        with TemporaryDirectory() as directory:
            original=G._transform(m,'decode',weights,directory)
            changed=G._transform(modified,'decode',weights,directory)
            self.assertEqual(len(list(Path(directory).iterdir())),2)
            self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(original,changed)))


if __name__=='__main__':
    unittest.main(verbosity=2)
