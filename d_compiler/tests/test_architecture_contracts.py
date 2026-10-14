"""Numerical, frontend and pipeline regressions from the September 12 audit."""
import sys
import unittest
from pathlib import Path
import numpy as np
import tvm
from tvm import relax
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu_compiler import npu_target, npu_legalize, tvm_pipeline, npu_link
from npu_compiler.device import NpuProfile
from npu_compiler.npu_errors import LinkError


def graph(shapes, op, dtypes=None):
    names = ["x", "w", "v", "mask"]
    args = [relax.Var(names[i], relax.TensorStructInfo(s, (dtypes or ["float16"]*len(shapes))[i]))
            for i, s in enumerate(shapes)]
    bb = relax.BlockBuilder()
    with bb.function("prefill", args):
        with bb.dataflow():
            out = bb.emit_output(op(*args))
        bb.emit_func_output(out)
    return bb.finalize()


def lower(mod, attention=False):
    return tvm_pipeline.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=False, fuse_attention=attention)(mod)


class ArchitectureContracts(unittest.TestCase):
    def test_unknown_call_rejected_with_location(self):
        mod = graph([(64,)], lambda x: relax.call_dps_packed("audit_unimplemented", [x],
                    out_sinfo=relax.TensorStructInfo((64,), "float16")))
        with self.assertRaisesRegex(LinkError, "prefill/.*audit_unimplemented"):
            npu_target.build(mod, fusion="off")

    def test_identity_and_int8_output_view(self):
        for dtype in ("int8", "float16", "float32"):
            mod = graph([(4,)], lambda x: relax.op.reshape(x, (2,2)), [dtype])
            ex = npu_target.build(mod, fusion="off")
            x = np.arange(4, dtype=dtype)
            out, _ = ex.run(dict(x=x))
            self.assertEqual(out.dtype, x.dtype)
            np.testing.assert_array_equal(out, x.reshape(2,2))
            self.assertEqual(ex.kernel_count, 0)
            with self.assertRaisesRegex(LinkError, "output_shape"):
                ex.run(dict(x=x), (4,))

    def test_direct_parameter_and_nested_tuple_outputs(self):
        x = relax.Var("x", relax.TensorStructInfo((4,), "float16"))
        y = relax.const(np.array([7, 9], dtype="int8"))
        fn = relax.Function([x], relax.Tuple([x, relax.Tuple([y,x])]))
        ex = npu_target.build(tvm.IRModule({"prefill": fn}), fusion="off")
        out, _ = ex.run(dict(x=np.arange(4,dtype="float16")))
        np.testing.assert_array_equal(out[0], out[1][1])
        np.testing.assert_array_equal(out[1][0], [7,9])

    def test_matmul_fp32_accumulation_constant_and_parameter(self):
        x = np.ones((64,64), dtype="float16"); x[:,0] = 2048; x[:,-1] = -2048
        w = np.ones_like(x)
        mod = graph([x.shape,w.shape], relax.op.matmul)
        semantic = lower(mod)
        vm = relax.VirtualMachine(relax.build(semantic, "llvm"), tvm.cpu())
        cpu = vm["prefill"](tvm.nd.array(x), tvm.nd.array(w)).numpy()
        np.testing.assert_array_equal(cpu, np.full_like(x, 62))
        for bound in (False, True):
            source = relax.transform.BindParams("prefill", {"x":x,"w":w})(mod) if bound else mod
            ex = npu_target.build(source, fusion="off")
            got, _ = ex.run({} if bound else dict(x=x,w=w))
            np.testing.assert_array_equal(got, cpu)

    def test_long_reduction_invariants(self):
        x = np.ones((1,4096), dtype="float16")
        cases = [(graph([x.shape], lambda x: relax.op.nn.softmax(x,axis=-1)),dict(x=x),1/4096),
                 (graph([x.shape,(4096,)], lambda x,w: relax.op.nn.rms_norm(x,w,axes=[-1],epsilon=1e-5)),
                  dict(x=x,w=x[0]),1)]
        for mod, values, expected in cases:
            semantic = lower(mod)
            vm = relax.VirtualMachine(relax.build(semantic,"llvm"), tvm.cpu())
            cpu = vm["prefill"](*[tvm.nd.array(v) for v in values.values()]).numpy()
            ex = npu_target.build(mod, fusion="off")
            got, _ = ex.run(values)
            np.testing.assert_array_equal(cpu, np.full_like(x,expected))
            np.testing.assert_array_equal(got,cpu)

    def test_multi_tile_cancellation_keeps_fp32_state_and_constant_order(self):
        x=np.ones((1,128),dtype="float16"); x[0,0]=65504; x[0,64]=-65504
        w=np.ones((128,1),dtype="float16"); w[0,0]=w[64,0]=65504
        mod=graph([x.shape,w.shape],relax.op.matmul)
        results=[]
        for bound in (False,True):
            source=relax.transform.BindParams("prefill",dict(x=x,w=w))(mod) if bound else mod
            ex=npu_target.build(source,fusion="off")
            got,_=ex.run({} if bound else dict(x=x,w=w))
            results.append(got)
        # The two rounded FP32 tile sums cancel. Neither a half-spilled
        # accumulator (inf), nor flat CPU accumulation (63), is valid here.
        np.testing.assert_array_equal(results[0],np.zeros((1,1),dtype="float16"))
        np.testing.assert_array_equal(results[1],results[0])

    def test_mean_reduction_uses_explicit_accumulator_and_rejects_half_tir(self):
        mod=graph([(1,4096)],lambda x:relax.op.mean(x,axis=-1,keepdims=True))
        ex=npu_target.build(mod,fusion='off')
        got,_=ex.run({'x':np.ones((1,4096),dtype='float16')})
        np.testing.assert_array_equal(got,np.ones((1,1),dtype='float16'))
        raw=relax.transform.LegalizeOps()(graph([(1,4096)],lambda x:relax.op.sum(x,axis=-1)))
        with self.assertRaisesRegex(LinkError,'explicitly use FP32'):
            npu_link.compile_program(raw)

    def test_unit_and_ragged_matmul_values(self):
        rng = np.random.default_rng(919)
        for a,b in [((1,2),(2,1)),((1,64),(64,1)),((2,1,64),(2,64,1)),
                    ((64,64),(64,1)),((1,64),(64,2)),((2,3,65),(2,65,7))]:
            x,w = [rng.normal(0,.1,s).astype("float16") for s in (a,b)]
            ex = npu_target.build(graph([a,b],relax.op.matmul),fusion="off")
            got,_ = ex.run(dict(x=x,w=w))
            np.testing.assert_allclose(got, (x.astype("float32")@w.astype("float32")).astype("float16"),rtol=.002,atol=2e-5)

    def test_attention_auto_rolls_back_on_small_sram(self):
        def attention(q,k,v,mask):
            return relax.op.matmul(relax.op.nn.softmax(relax.op.add(relax.op.matmul(q,k),mask),axis=-1),v)
        mod = graph([(1,64,64)]*3+[(64,64)],attention)
        ex = npu_target.build(mod,profile=NpuProfile(sram_bytes=32000))
        self.assertLessEqual(max(ex.plan.sram_peak.values())//2,32000)
        self.assertNotEqual(ex.plan.fusion["selected"]["attention"], "full")

    def test_unreachable_unsupported_entry_does_not_break_build(self):
        mod = graph([(4,)],lambda x: relax.op.add(x,x))
        other = graph([(4,)],lambda x: relax.call_dps_packed("missing",[x],out_sinfo=x.struct_info))
        mod["unused"] = other["prefill"].with_attr("global_symbol","unused")
        ex = npu_target.build(mod,fusion="off")
        self.assertGreater(ex.kernel_count,0)

    def test_shared_importer_value_retains_original_dtype(self):
        import torch
        from npu_compiler.nn_models.hf import HFImporter
        class Shared(torch.nn.Module):
            def forward(self,x,y):
                return x+y,x*x
        x=torch.tensor([1.1,2.1],dtype=torch.float16); y=torch.zeros(2,dtype=torch.float32)
        mod=HFImporter().from_exported_program(torch.export.export(Shared(),(x,y)),
              keep_params_as_input=True,unwrap_unit_return_tuple=True,no_bind_return_tuple=False)
        vm=relax.VirtualMachine(relax.build(mod,"llvm"),tvm.cpu())
        result=vm["main"](tvm.nd.array(x.numpy()),tvm.nd.array(y.numpy()))
        for got,expected in zip(result,Shared()(x,y)):
            self.assertEqual(str(got.dtype),str(expected.numpy().dtype))
            np.testing.assert_array_equal(got.numpy(),expected.numpy())

    def test_squeeze_preserves_broadcast_and_output_dtype(self):
        from npu_compiler.nn_models.hf import _SqueezeMatmulBatch
        for rhs,dtype in [((4,3),"float32"),((2,4,3),"float16")]:
            mod=graph([(1,2,4),rhs],lambda x,w: relax.op.matmul(x,w,out_dtype=dtype))
            old=mod["prefill"].ret_struct_info
            new=_SqueezeMatmulBatch().visit_expr(mod["prefill"]).ret_struct_info
            self.assertTrue(tvm.ir.structural_equal(old,new))

    def test_rope_parameters_against_independent_formula(self):
        from npu_compiler.nn_models.llama import rope_freqs
        base=10000.; dim=128; unscaled=1/(base**(np.arange(0,dim,2)/dim))
        for factor in (2.,8.):
            got=rope_freqs(dim,base,dict(rope_type="linear",factor=factor))[:dim//2]
            np.testing.assert_allclose(got,unscaled/factor,rtol=1e-14)
        cfg=dict(rope_type="llama3",factor=8.,low_freq_factor=1.,high_freq_factor=4.,original_max_position_embeddings=8192)
        wave=2*np.pi/unscaled; smooth=(8192/wave-1)/3
        ref=np.where(wave>8192,unscaled/8,np.where(wave<2048,unscaled,(1-smooth)*unscaled/8+smooth*unscaled))
        np.testing.assert_allclose(rope_freqs(dim,base,cfg)[:dim//2],ref,rtol=1e-14)
        with self.assertRaisesRegex(ValueError,"unsupported RoPE"):
            rope_freqs(dim,base,dict(rope_type="dynamic",factor=8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
