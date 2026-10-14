"""Executed regressions for vector coalescing, direct stores and DRAM relocation."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tvm import relax

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_architecture_contracts import graph, lower
from npu_compiler import npu_link, npu_legalize, tvm_pipeline
from npu_compiler.v09_stats import analyze, check_counters


class CompilerRefinement(unittest.TestCase):
    def run_program(self,mod,shapes,residency=True):
        rng=np.random.default_rng(928)
        values={name:rng.normal(0,.2,shape).astype('float16')
                for name,shape in zip(('x','w','v'),shapes)}
        asm,plan=npu_link.compile_program(mod,residency=residency,native_mode='compare')
        result,counters=npu_link.run_program(asm,plan,plan.func,values)
        check_counters(asm.words,counters)
        return result,analyze(asm.words,plan,detail=False),plan,asm

    def assert_exact(self,a,b):
        np.testing.assert_array_equal(a[0].view('uint16'),b[0].view('uint16'))

    def test_fused_silu_uses_long_vectors_without_changing_rounding(self):
        shapes=[(64,128),(128,128),(64,128)]
        source=graph(shapes,lambda x,w,v:relax.op.multiply(relax.op.nn.silu(relax.op.matmul(x,w)),v))
        mod=tvm_pipeline.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
            fuse=True,lift_params=False,fuse_attention=False)(source)
        with patch('npu_compiler.npu_schedule._coalesce_pointwise',side_effect=lambda p:p):
            before=self.run_program(mod,shapes,residency=False)
        after=self.run_program(mod,shapes,residency=False)
        self.assert_exact(before,after)
        self.assertLess(after[1]['opcodes']['0x0f'],before[1]['opcodes']['0x0f'])
        self.assertLess(after[1]['words'],before[1]['words'])
        self.assertEqual(after[1]['dma_bytes'],before[1]['dma_bytes'])

    def test_row_broadcast_remains_correct_when_silu_is_coalesced(self):
        shapes=[(64,128),(128,128),(128,)]
        source=graph(shapes,lambda x,w,v:relax.op.nn.silu(relax.op.add(relax.op.matmul(x,w),v)))
        mod=tvm_pipeline.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
            fuse=True,lift_params=False,fuse_attention=False)(source)
        with patch('npu_compiler.npu_schedule._coalesce_pointwise',side_effect=lambda p:p):
            before=self.run_program(mod,shapes,residency=False)
        after=self.run_program(mod,shapes,residency=False)
        self.assert_exact(before,after)
        self.assertLess(after[1]['words'],before[1]['words'])

    def test_direct_output_handles_column_offset_stride_and_multiple_k_tiles(self):
        shapes=[(64,128),(128,128),(128,64)]
        mod=lower(graph(shapes,lambda x,w,v:relax.op.matmul(relax.op.matmul(x,w),v)))
        with patch('npu_compiler.npu_accumulator.forward_output_tiles',side_effect=lambda p:p):
            before=self.run_program(mod,shapes)
        after=self.run_program(mod,shapes)
        self.assert_exact(before,after)
        self.assertTrue(after[2].scheduled_calls[0].attrs['npu.forwarded_output_tiles'])
        self.assertLess(after[1]['opcodes']['0x98'],before[1]['opcodes']['0x98'])

    def test_dram_compaction_moves_outputs_and_constants_without_dma_changes(self):
        shapes=[(64,64),(64,128),(128,64)]
        mod=lower(graph(shapes,lambda x,w,v:relax.op.matmul(relax.op.matmul(x,w),v)))
        with patch('npu_compiler.npu_memplan.compact_unused_storage'):
            before=self.run_program(mod,shapes)
        after=self.run_program(mod,shapes)
        self.assert_exact(before,after)
        self.assertEqual(before[1]['global_image_bytes']-after[1]['global_image_bytes'],16384)
        self.assertEqual(after[1]['global_image_bytes'],49164)
        self.assertEqual(before[1]['dma_bytes'],after[1]['dma_bytes'])
        self.assertEqual(before[1]['words'],after[1]['words'])
        plan=after[2]
        self.assertEqual(plan.dram_compaction['saved_bytes'],16384)
        self.assertTrue(plan.dram_compaction['removed_tensors'])
        self.assertTrue(all(n not in plan.address for n in plan.dram_compaction['removed_tensors']))
        self.assertLess(plan.outputs['offset_bytes'],before[2].outputs['offset_bytes'])
        self.assertLess(plan.constant_base,before[2].constant_base)

    def test_dram_used_by_later_versions_is_preserved(self):
        shapes=[(64,64)]
        def op(x):
            a=relax.op.add(x,x)
            b=relax.op.multiply(a,a)
            c=relax.op.add(b,b)
            return relax.op.add(c,a)
        mod=lower(graph(shapes,op))
        with patch('npu_compiler.npu_memplan.compact_unused_storage'):
            before=self.run_program(mod,shapes,residency=False)
        after=self.run_program(mod,shapes,residency=False)
        self.assert_exact(before,after)
        self.assertEqual(after[2].dram_compaction['saved_bytes'],0)

    def repeated_layers(self):
        def op(x,w,v):
            for weight in (w,v,w,w,v,w):
                x=relax.op.matmul(x,weight)
            return x
        shapes=[(64,64)]*3
        return lower(graph(shapes,op)),shapes

    def test_repeated_layer_cost_reuse_keeps_actual_instructions_and_output(self):
        from npu_compiler import npu_memory_search
        original=npu_memory_search.probe
        mod,shapes=self.repeated_layers()
        def uncached(*args,**kwargs):
            return original(*args,**dict(kwargs,reuse=False))
        with patch.object(npu_memory_search,'probe',side_effect=uncached):
            before=self.run_program(mod,shapes)
        after=self.run_program(mod,shapes)
        self.assertGreater(after[2].compilation['cost_probe_hits'],0)
        self.assertEqual(before[3].words,after[3].words)
        self.assert_exact(before,after)

    def test_inaccurate_reused_cost_retries_with_actual_emission(self):
        from npu_compiler.npu_compilation import CompilationContext
        original=CompilationContext.probe
        mod,shapes=self.repeated_layers()
        before=self.run_program(mod,shapes)
        def corrupt(self,*args,**kwargs):
            words,native,detail,statistics=original(self,*args,**kwargs)
            if words is None:
                statistics=dict(statistics,words=0)
            return words,native,detail,statistics
        with patch.object(CompilationContext,'probe',corrupt):
            after=self.run_program(mod,shapes)
        self.assertIn('cost_reuse_fallback',after[2].joint_memory)
        self.assertEqual(before[3].words,after[3].words)
        self.assert_exact(before,after)


if __name__=='__main__': unittest.main(verbosity=2)
