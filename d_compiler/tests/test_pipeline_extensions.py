"""Independent checks for attention, storage reuse and generation extensions."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import tvm
from tvm import relax, tir
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_architecture_contracts import graph
from npu_compiler import npu_target, npu_link, npu_generate
from npu_compiler.device import PROFILES
from npu_compiler.npu_errors import LinkError
from npu_compiler.v09_stats import analyze


class PipelineExtensions(unittest.TestCase):
    def test_dequant_producer_rename_fits_small_sram_without_partial_spill(self):
        from npu_compiler import npu_quantize,npu_legalize,tvm_pipeline
        source=graph([(64,256),(256,256)],relax.op.matmul)
        source['prefill']=source['prefill'].with_attr('num_input',1)
        source=npu_quantize.QuantizeWeightsW8A16()(source)
        mod=tvm_pipeline.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
            fuse=False,fuse_attention=False)(source)
        renamed=tvm.IRModule(mod.functions,attrs=mod.attrs)
        def rename(node):
            if isinstance(node,tir.Block) and node.name_hint=='w_dequant':
                return tir.Block(node.iter_vars,node.reads,node.writes,'independent_producer_name',
                    node.body,node.init,node.alloc_buffers,node.match_buffers,node.annotations)
        for gv,fn in mod.functions.items():
            if isinstance(fn,tir.PrimFunc):
                renamed[gv]=fn.with_body(tir.stmt_functor.ir_transform(fn.body,None,rename,['tir.Block']))
        rng=np.random.default_rng(831)
        weight=tvm.nd.array(rng.normal(0,.1,(256,256)).astype('float16'))
        vm=relax.VirtualMachine(relax.build(mod,'llvm'),tvm.cpu())
        packed=[p.numpy() for p in vm['prefill_transform_params']([weight])]
        x=rng.normal(0,.1,(64,256)).astype('float16'); outputs=[]; words=[]
        for module in (mod,renamed):
            asm,plan=npu_link.compile_program(module,profile=replace(PROFILES['v09'],sram_bytes=80000))
            self.assertLessEqual(max(plan.sram_peak.values()),160000)
            values=dict(zip([p.name_hint for p in plan.func.params],[x]+packed))
            output,_=npu_link.run_program(asm,plan,plan.func,values)
            outputs.append(output); words.append(asm.words)
        np.testing.assert_array_equal(outputs[0],outputs[1]);self.assertEqual(words[0],words[1])

    def test_hf_host_fold_defers_fp16_arithmetic(self):
        from npu_compiler.nn_models.hf import _fold_host_constants
        x=np.ones((64,64),dtype='float16'); x[:,0]=2048; x[:,-1]=-2048
        source=graph([x.shape],lambda bias:relax.op.add(
            relax.op.matmul(relax.const(x),relax.const(np.ones_like(x))),bias))
        ex=npu_target.build(_fold_host_constants(source),fusion='off')
        got,_=ex.run({'x':np.zeros_like(x)})
        np.testing.assert_array_equal(got,np.full_like(x,62))

    def test_highlevel_attention_bias_causal_window_independent_oracle(self):
        rng=np.random.default_rng(72)
        shapes=[(1,3,2,8),(1,5,2,8),(1,5,2,4),(1,2,3,5)]
        q,k,v,bias=[rng.normal(0,.1,s).astype('float16') for s in shapes]
        for causal,window in [(None,None),('TopLeft',None),('BottomRight',None),('BottomRight',3)]:
            mod=graph(shapes,lambda q,k,v,b:relax.op.nn.attention(q,k,v,b,
                causal_mask=causal,window_size=window))
            ex=npu_target.build(mod,fusion='off')
            got,_=ex.run(dict(zip(['x','w','v','mask'],[q,k,v,bias])))
            qh,kh,vh=[a.transpose(0,2,1,3).astype('float32') for a in (q,k,v)]
            scores=(qh@kh.swapaxes(-1,-2)).astype('float16')
            scores=(scores/np.sqrt(np.float16(8))).astype('float16')
            scores=(scores+bias).astype('float16')
            row,col=np.indices((3,5)); offset=2 if causal=='BottomRight' else 0
            valid=col<=row+offset if causal else np.ones((3,5),dtype=bool)
            if window: valid &= col>=row+offset-window+1
            masked=np.where(valid,scores,np.float16(0)) if causal else scores
            exponent=np.exp((masked-masked.max(axis=-1,keepdims=True)).astype('float16')).astype('float16')
            exponent=np.where(valid,exponent,np.float16(0))
            total=exponent.astype('float32').sum(axis=-1,keepdims=True).astype('float16')
            probabilities=(exponent/total).astype('float16')
            expected=(probabilities.astype('float32')@vh).astype('float16').transpose(0,2,1,3)
            np.testing.assert_allclose(got,expected,rtol=.003,atol=5e-5)
        mod=graph(shapes[:3],lambda q,k,v:relax.op.nn.attention(q,k,v,window_size=3))
        with self.assertRaisesRegex(LinkError,'explicit causal'):
            npu_target.build(mod)

    def test_epilogue_fusion_preserves_stores_and_reduces_traffic(self):
        mod=graph([(64,64)]*3,lambda x,w,b:relax.op.nn.silu(relax.op.add(relax.op.matmul(x,w),b)))
        rng=np.random.default_rng(712)
        values={n:rng.normal(0,.3,(64,64)).astype('float16') for n in ('x','w','v')}
        # Isolate fusion's effect. Cross-kernel retention can now remove the
        # same DRAM stores even when matmul remains a separate kernel.
        profile=replace(PROFILES['v09'],joint_memory=False)
        plain=npu_target.build(mod,fusion='off',profile=profile)
        fused=npu_target.build(mod,profile=profile)
        a,_=plain.run(values); b,_=fused.run(values)
        np.testing.assert_array_equal(a.view('uint16'),b.view('uint16'))
        self.assertEqual(fused.plan.fusion['selected']['generic_fusion'],'matmul')
        self.assertLess(analyze(fused.words)['dma_bytes'],analyze(plain.words)['dma_bytes'])

    def test_weight_content_cache_identity_and_corruption(self):
        from test_params_cache import _model,TINY
        lower,weights=_model(TINY,seed=41)
        with tempfile.TemporaryDirectory() as cache:
            original=npu_generate._transform(lower,'decode',weights,cache)
            changed=[tvm.nd.array(w.numpy()+np.float16(.125)) for w in weights]
            other=npu_generate._transform(lower,'decode',changed,cache)
            self.assertEqual(len(list(Path(cache).iterdir())),2)
            self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(original,other)))
            path=next(Path(cache).glob('*/0.npy'))
            data=np.load(path,mmap_mode='r+'); data.flat[0]+=np.float16(1); data.flush()
            with self.assertRaisesRegex(ValueError,'corrupt'):
                # Find which source corresponds to the modified entry.
                for source in (weights,changed):
                    npu_generate._transform(lower,'decode',source,cache)

    def test_w8_workspace_reuse_and_forged_body(self):
        from test_completion_contracts import matmul
        from npu_compiler.npu_verify import validate_w8a8
        mod=matmul(7,128,192,quant=True)
        gv=next(g for g,f in mod.functions.items() if isinstance(f,tir.PrimFunc) and f.attrs.get('npu.kernel')=='w8a8')
        prim=mod[gv]
        validate_w8a8(prim)
        def alter(node):
            if isinstance(node,tir.FloatImm) and node.value==0:
                return tir.const(1,node.dtype)
        forged=prim.with_body(tir.stmt_functor.ir_transform(prim.body,None,alter))
        with self.assertRaisesRegex(LinkError,'semantic body'):
            validate_w8a8(forged)
        rng=np.random.default_rng(812)
        vm=relax.VirtualMachine(relax.build(mod,'llvm'),tvm.cpu())
        packed=vm['prefill_transform_params']([tvm.nd.array(rng.normal(0,.2,(128,192)).astype('float16'))])
        x=rng.normal(0,.4,(7,128)).astype('float16'); x[0]=0
        outputs=[]; peaks=[]
        for reuse in (False,True):
            asm,plan=npu_link.compile_program(mod,profile=replace(PROFILES['v09'],reuse_sram=reuse),native_mode='compare')
            values=dict(zip([p.name_hint for p in plan.func.params],[x]+[p.numpy() for p in packed]))
            image=npu_link.build_image(plan,plan.func,values)
            from npu_compiler.v09_runtime import run
            images,_=run(asm.words,image.view('uint32'))
            outputs.append(images[-1]); peaks.append(max(plan.sram_peak.values()))
        np.testing.assert_array_equal(outputs[0],outputs[1])
        self.assertLess(peaks[1],peaks[0])

    def test_persistent_transport_all_cache_slots_and_neighbours(self):
        from test_completion_contracts import declare,AppendPlan,M
        from npu_compiler.v09_runtime import run,PersistentTransport
        for capacity in (5,6):
            rows,width=3,3
            cache=relax.Var('cache',relax.TensorStructInfo((rows,capacity,width),'float16'))
            fresh=relax.Var('new',relax.TensorStructInfo((rows,1,width),'float16'))
            mask=relax.Var('mask',relax.TensorStructInfo((1,1,capacity+1),'float16'))
            bb=relax.BlockBuilder()
            with bb.function('decode',[cache,fresh,mask]):bb.emit_func_output(relax.Tuple([fresh]))
            mod=bb.finalize();mod['decode']=declare(mod['decode'],[('cache',0,1)])
            planned,plan=M.assign_addresses(mod,'decode'); append=AppendPlan(planned['decode'],plan,sram_bytes=512)
            values=dict(cache=np.arange(rows*capacity*width,dtype='float16').reshape(rows,capacity,width),
                        new=np.zeros((rows,1,width),'float16'),mask=np.zeros((1,1,capacity+1),'float16'))
            expected=npu_link.build_image(plan,planned['decode'],values).view('uint32')
            transport=PersistentTransport(expected)
            try:
                for position in range(capacity):
                    data=(np.arange(rows*width,dtype='float16')-position).view('uint8')
                    offset=plan.address['new']*plan.unit_bytes
                    expected.view('uint8')[offset:offset+data.size]=data
                    words=append.words(position)+[0xff]
                    images,_=run(words,expected);expected=images[-1]
                    actual,_=transport.run(words,[(offset,data)])
                    np.testing.assert_array_equal(actual,expected)
                    self.assertEqual(transport.last_transfer['host_weight_rewrite_bytes'],0)
            finally:transport.close()

    def test_buckets_cross_boundary_without_recompile_per_token(self):
        from test_nn_decode import _Assets,_family,TINY
        cfg=dict(TINY,num_hidden_layers=1); assets=_Assets(cfg); family=_family(assets)
        results=[]; records=[]
        for buckets,runner in [(None,None),([8,16],None),([8,16],npu_generate.llvm_runner)]:
            diagnostics={}
            results.append(npu_generate.generate(family,cfg,assets,[3,1,4,1,5],6,
                capacity=16,cache_buckets=buckets,diagnostics=diagnostics,runner=runner,fusion='off'))
            records.append(diagnostics)
        self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])
        self.assertEqual(records[1]['compiled_decode_programs'],2)
        self.assertEqual([r['position'] for r in records[1]['transitions']],[8])
        self.assertLess(records[1]['steps'][1]['words'],records[0]['steps'][1]['words'])

    def test_generation_last_logits_matches_full_output(self):
        from test_nn_decode import _Assets,_family,TINY
        cfg=dict(TINY,num_hidden_layers=1);assets=_Assets(cfg);family=_family(assets)
        results=[]
        for last in (False,True):
            mod,params,config=family.build_generate(cfg,5,8,last_logits=last)
            weights=[tvm.nd.array(v) for v in family.load_params(assets,params,config)]
            lowered=npu_generate._lower(mod,fusion='off',entries=['prefill_cache'])
            inp=family.runtime_inputs(assets,config,[3,1,4,1,5]); cos,sin=family.rope_inputs(config,np.arange(5))
            outputs,_,_=npu_generate.run_tuple_program(lowered,'prefill_cache',dict(inp,cos=cos,sin=sin),weights)
            results.append(outputs)
        np.testing.assert_array_equal(results[0][0][-1:],results[1][0])
        np.testing.assert_array_equal(results[0][1],results[1][1]);np.testing.assert_array_equal(results[0][2],results[1][2])


if __name__=='__main__':unittest.main(verbosity=2)
