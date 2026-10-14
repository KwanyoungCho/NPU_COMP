"""Integration gates for cost-selected fusion, forwarding and cache effects."""
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from dataclasses import replace
from itertools import product

import numpy as np
import tvm
from tvm import relax, tir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu_compiler import npu_link as L, npu_memplan as M, npu_target
from npu_compiler.npu_cache import declare, AppendPlan
from npu_compiler.device import PROFILES
from npu_compiler.v09_stats import analyze, cost
from npu_compiler.v09_runtime import run
from npu_compiler.v09_asm import V09Asm
from npu_compiler.npu_residency import forward
from test_compiler_contracts import lower, matmul


def chain(width=64, rows=3):
    bb = relax.BlockBuilder()
    x = relax.Var('x', relax.TensorStructInfo((rows, width), 'float16'))
    with bb.function('prefill', [x]):
        with bb.dataflow():
            a = bb.emit(relax.op.add(x, x))
            b = bb.emit(relax.op.multiply(a, a))
            c = bb.emit(relax.op.add(b, a))
            out = bb.emit_output(relax.op.negative(c))
        bb.emit_func_output(out)
    return bb.finalize()


class CompletionTests(unittest.TestCase):
    def test_rejected_fusion_region_is_inlined_without_losing_others(self):
        from npu_compiler import npu_fusion
        from npu_compiler.tir_codegen_v09 import V09TirError
        bb=relax.BlockBuilder()
        x=relax.Var('x',relax.TensorStructInfo((3,64),'float16'))
        w=relax.Var('w',relax.TensorStructInfo((64,64),'float16'))
        with bb.function('prefill',[x,w]):
            with bb.dataflow():
                a=bb.emit(relax.op.add(x,x));b=bb.emit(relax.op.multiply(a,a))
                dot=bb.emit(relax.op.matmul(b,w))
                c=bb.emit(relax.op.negative(dot));out=bb.emit_output(relax.op.add(c,c))
            bb.emit_func_output(out)
        mod=lower(bb.finalize())
        probe=npu_fusion._probe
        rejected=[]
        def backend_limit(module,fn,profile):
            calls=[]
            relax.analysis.post_order_visit(fn.body,lambda e:calls.append(e)
                if isinstance(e,relax.Call) and e.op==tvm.ir.Op.get('relax.call_tir') else None)
            # Simulate a target-specific compound rejection at its public
            # boundary. The region restoration and resulting execution are real.
            if len(calls)==1 and calls[0].args[0].name_hint.startswith('fused_') and not rejected:
                rejected.append(calls[0].args[0].name_hint)
                raise V09TirError('test target compound is unsupported')
            return probe(module,fn,profile)
        with patch.object(npu_fusion,'_probe',side_effect=backend_limit):
            # Isolate region rollback from cross-kernel retention, which can
            # independently remove the unfused region's traffic.
            selected,info=npu_fusion.select(mod,replace(PROFILES['v09'],joint_memory=False))
        self.assertEqual(sum(r['status']=='unsupported' for r in info['regions']),1)
        self.assertTrue(any(r['selected'] for r in info['regions']))
        rng=np.random.default_rng(33)
        vals={'x':rng.normal(0,.1,(3,64)).astype('float16'),
              'w':rng.normal(0,.1,(64,64)).astype('float16')}
        outputs=[]
        for module in (mod,selected):
            asm,plan=L.compile_program(module)
            func=M.assign_addresses(module)[0]['prefill']
            outputs.append(L.run_program(asm,plan,func,vals,(3,64))[0])
        np.testing.assert_array_equal(outputs[0].view('uint16'),outputs[1].view('uint16'))

    def test_scaled_accumulator_protocol_errors_propagate(self):
        from npu_compiler import native
        from npu_compiler.tir_codegen_v09 import Walker,SramEmitter
        c=tir.decl_buffer((64,64),'float32',scope='local.accumulator',name='C')
        d=tir.decl_buffer((64,64),'float16',scope='global.sram',name='D')
        body=tir.Evaluate(tir.call_extern('int32','npu_store_scaled',
                          c.access_ptr('r'),d.access_ptr('w'),64))
        block=tir.Block([],[],[],'root',body,alloc_buffers=[c,d])
        prim=tir.PrimFunc([],tir.BlockRealize([],True,block))
        asm=V09Asm();walker=Walker(asm,{},SramEmitter(asm));walker.declare_sram(d,0)
        with self.assertRaisesRegex(RuntimeError,'matching accumulator'):
            walker.run(prim,{})
        with self.assertRaisesRegex(tvm.error.TVMError,'matching accumulator'):
            native.codegen_kernel(prim,[],{d:0},[],{},0)

    def test_auto_fusion_selects_regions_by_actual_cost(self):
        values = {'x': np.random.default_rng(1).normal(size=(3,64)).astype('float16')}
        baseline = npu_target.build(chain(), pipeline=None, fusion='off', residency=False)
        selected = npu_target.build(chain(), fusion='auto', residency=False)
        before, _ = baseline.run(values, (3,64))
        after, _ = selected.run(values, (3,64))
        np.testing.assert_array_equal(before.view('uint16'), after.view('uint16'))
        self.assertLess(cost(analyze(selected.words, selected.plan)), cost(analyze(baseline.words, baseline.plan)))
        self.assertTrue(any(r['selected'] for r in selected.plan.fusion['regions']))
        json.dumps(selected.plan.fusion)
        import tempfile, hashlib
        with tempfile.TemporaryDirectory() as directory:
            selected.export_bundle(directory, values)
            manifest=json.loads((Path(directory)/'program.json').read_text())
            binary=(Path(directory)/'program_memory.bin').read_bytes()
            self.assertEqual(manifest['program_sha256'],hashlib.sha256(binary).hexdigest())
            self.assertEqual(len(binary),4*len(selected.words))

    def test_forwarding_preserves_every_kernel_snapshot_and_counts(self):
        mod = lower(chain())
        base, plan0 = L.compile_program(mod, snapshot_at={0,1,2,3}, residency=False)
        opt, plan1 = L.compile_program(mod, snapshot_at={0,1,2,3}, residency=True)
        func = M.assign_addresses(mod)[0]['prefill']
        values = {'x': np.random.default_rng(5).normal(size=(3,64)).astype('float16')}
        image = L.build_image(plan0, func, values).view('<u4')
        a, ca = run(base.words, image)
        b, cb = run(opt.words, image)
        self.assertEqual(len(a), len(b))
        for left, right in zip(a,b):
            np.testing.assert_array_equal(left, right)
        self.assertGreater(plan1.residency['saved_load_bytes'], 0)
        stats = analyze(opt.words, plan1)
        self.assertEqual(stats['dma_load_bytes'], cb['dma_cells_loaded'] * 4)
        self.assertEqual(stats['dma_store_bytes'], cb['dma_cells_stored'] * 4)
        self.assertEqual(stats['instructions'], cb['words_executed'])
        self.assertEqual(sum(analyze(opt.words[s:e])['dma_bytes'] for _,s,e in plan1.kernel_span),
                         stats['dma_bytes'] - (len(plan1.constant_values)*2+4))

    def test_forwarding_rejects_stale_versions_and_noncanonical_copies(self):
        # Pure DMA may contain arbitrary FP16 bit patterns (including payload
        # NaNs); it must not be replaced with a numeric vector copy.
        asm = V09Asm()
        asm.gload(0, 4, 0, 1, 4)
        asm.gstore(4, 4, 0, 1, 4)
        boundary = len(asm.words)
        asm.gload(4, 4, 128, 1, 4)
        records = [dict(index=0, span=(0,boundary), arguments=[]),
                   dict(index=1, span=(boundary,len(asm.words)), arguments=[
                       dict(dtype='float16', address_bytes=16, size_bytes=16)])]
        words, _, info = forward(asm.words, records)
        self.assertEqual(words, asm.words)
        self.assertFalse(info['events'])
        # An overlapping global write invalidates an older SRAM copy.
        asm = V09Asm()
        asm.gload(0,4,0,1,4)
        asm.gstore(4,4,0,1,4)
        asm.gload(8,4,128,1,4)
        asm.gstore(4,4,128,1,4)
        boundary = len(asm.words)
        asm.gload(4,4,0,1,4)
        records[0]['span']=(0,boundary)
        records[1]['span']=(boundary,len(asm.words))
        words,_,info = forward(asm.words,records)
        self.assertEqual(words,asm.words)
        self.assertFalse(info['events'])

    def test_w8a8_tir_matches_reference_and_exposes_workspace(self):
        from npu_compiler.npu_schedule import _fit_kernel
        rng = np.random.default_rng(8)
        for m,k,n,budget in ((7,128,192,8<<20),(193,128,192,80000),(65,3072,128,8<<20)):
            mod=matmul(m,k,n,quant=True)
            vm=relax.VirtualMachine(relax.build(mod,'llvm'),tvm.cpu())
            w=rng.normal(0,.2,(k,n)).astype('float16')
            packed=vm['prefill_transform_params']([tvm.nd.array(w)])
            x=rng.normal(0,.4,(m,k)).astype('float16');x[0]=0;x[-1]=2**-24
            profile=replace(PROFILES['v09'],sram_bytes=budget)
            outputs=[]
            for mode in ('reference','tir'):
                asm,plan=L.compile_program(mod,profile=profile,w8a8_mode=mode,
                                          native_mode="compare")
                func=M.assign_addresses(mod)[0]['prefill']
                values=dict(zip([p.name_hint for p in func.params],[x]+[p.numpy() for p in packed]))
                output,_=L.run_program(asm,plan,func,values,(m,n))
                outputs.append(output)
                if mode=='tir':
                    self.assertTrue(plan.sram_layouts)
                    self.assertTrue(all(plan.native.values()),plan.native_diagnostics)
                    self.assertTrue(all(r['backend']=='python' for r in plan.kernel_records))
                    self.assertLessEqual(max(plan.sram_peak.values()),budget*2)
                    gv=next(g for g,f in mod.functions.items() if isinstance(f,tir.PrimFunc) and f.attrs.get('npu.kernel')=='w8a8')
                    scheduled, scratch, layout=_fit_kernel(mod,gv,mod[gv],profile,16,6)
                    script=scheduled.script()
                    self.assertIn('npu_quantize_row',script)
                    self.assertIn('local.accumulator',script)
                    self.assertEqual(scratch,0)
                    self.assertFalse(any(b.scope()=='local.accumulator' for b in layout.buffers))
            np.testing.assert_array_equal(outputs[0].view('uint16'),outputs[1].view('uint16'))

    def test_incremental_cache_dma_all_positions_and_cell_neighbours(self):
        for rows,capacity in product((3,129),(1,2,5,6)):
            for width in (1,3,4,17):
                # Return existing new-token parameters; append is an explicit
                # effect in a separate program, so no compute is needed here.
                bb=relax.BlockBuilder()
                cache=relax.Var('cache',relax.TensorStructInfo((rows,capacity,width),'float16'))
                new=relax.Var('new',relax.TensorStructInfo((rows,1,width),'float16'))
                mask=relax.Var('mask',relax.TensorStructInfo((1,1,capacity+1),'float16'))
                with bb.function('decode',[cache,new,mask]):
                    bb.emit_func_output(relax.Tuple([new]))
                mod=bb.finalize()
                mod['decode']=declare(mod['decode'],[('cache',0,1)])
                planned,plan=M.assign_addresses(mod,'decode')
                append=AppendPlan(planned['decode'],plan,sram_bytes=512)
                expected=np.arange(rows*capacity*width).reshape(rows,capacity,width).astype('float16')
                for position in range(capacity):
                    fresh=(np.arange(rows*width).reshape(rows,1,width)*-.125-position).astype('float16')
                    values={'cache':expected.copy(),'new':fresh,'mask':np.zeros((1,1,capacity+1),'float16')}
                    image=L.build_image(plan,planned['decode'],values)
                    reference=image.copy()
                    expected[:,position:position+1,:]=fresh
                    start=plan.address['cache']
                    reference[start:start+expected.size]=expected.reshape(-1)
                    words=append.words(position)+[0xff]
                    images,counters=run(words,image.view('<u4'))
                    np.testing.assert_array_equal(images[-1],reference.view('<u4'))
                    self.assertEqual(analyze(words)['dma_bytes'],
                                     4*(counters['dma_cells_loaded']+counters['dma_cells_stored']))
                with self.assertRaises(ValueError):append.words(capacity)
                with self.assertRaises(ValueError):append.words(-1)

    def test_static_analyzer_rejects_truncated_dma_payload(self):
        with self.assertRaisesRegex(ValueError,'truncated'):
            analyze([0xa0,0x42,0x98])

    def test_image_binding_rejects_wrong_shape_and_dtype(self):
        executable=npu_target.build(chain(),fusion='off',residency=False)
        for value in (np.zeros((3,64),'float32'),np.zeros((4,64),'float16')):
            with self.assertRaisesRegex(L.LinkError,'expected'):
                executable.build_image({'x':value})


if __name__=='__main__':
    unittest.main(verbosity=2)
