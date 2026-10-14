"""Bounded regressions for conditional fusion and unit-length DMA staging."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import unittest
import tvm
from tvm import relax, tir
from tvm.script import tir as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu_compiler import native, npu_fusion, npu_link, npu_target, tvm_pipeline
from npu_compiler.device import PROFILES
from npu_compiler.npu_errors import LinkError
from npu_compiler.npu_schedule import _fit_kernel, _schedule, _sram_buffers, _place_sram
from npu_compiler.tir_codegen_v09 import Walker, SramEmitter
from npu_compiler.v09_asm import V09Asm


def cache_stack(layers=3, tokens=8, heads=2, width=16):
    args = [relax.Var(f'x{i}', relax.TensorStructInfo((tokens, heads, width), 'float16'))
            for i in range(layers)]
    bb = relax.BlockBuilder()
    with bb.function('prefill', args):
        with bb.dataflow():
            parts = []
            for x in args:
                y = bb.emit(relax.op.permute_dims(x, axes=[1, 2, 0]))
                parts.append(bb.emit(relax.op.expand_dims(y, 0)))
            out = bb.emit_output(relax.op.concat(parts, 0))
        bb.emit_func_output(out)
    return tvm_pipeline.graph_pipeline(fuse=False, fuse_attention=False,
                                       lift_params=False)(bb.get())


def test_repeated_producers_are_rejected_before_codegen():
    # The same dimensions/28-way stack as full Llama prefill, without weights
    # or model compilation. No huge instruction stream is ever materialized.
    mod = relax.transform.FuseTIR()(relax.transform.FuseOps()(cache_stack(28, 256, 8, 128)))
    gv, prim = next((g, f) for g, f in mod.functions.items() if isinstance(f, tir.PrimFunc))
    with unittest.TestCase().assertRaisesRegex(LinkError, 'generic stripe repeats producer'):
        _fit_kernel(mod, gv, prim, PROFILES['v09'], 24, 6)


def test_fusion_falls_back_and_preserves_cache_stack_values():
    mod = cache_stack()
    profile = replace(PROFILES['v09'], sram_bytes=3000, joint_memory=False)
    selected, decision = npu_fusion.select(mod, profile, ['prefill'])
    assert any(not r['selected'] and 'repeats producer' in r['reason']
               for r in decision['regions'])
    asm, plan = npu_link.compile_program(selected, profile=profile, native_mode='compare')
    # Execute the exact selected program, including the unfused fallback.
    ex = npu_target.NpuExecutable(asm.words, plan, plan.func, asm.kernel_count, None, selected)
    values = {f'x{i}': (np.arange(8*2*16).reshape(8,2,16)+i*300).astype('float16')
              for i in range(3)}
    got, _ = ex.run(values)
    expected = np.stack([x.transpose(1,2,0) for x in values.values()])
    np.testing.assert_array_equal(got, expected)
    assert len(asm.words) < 10000


def test_stripes_do_not_write_uncomputed_rectangular_holes():
    from npu_compiler.npu_intrin import generic_stripes
    mod = cache_stack()
    gv, prim = next((g, f) for g, f in mod.functions.items()
                    if isinstance(f, tir.PrimFunc) and len(f.params) == 4)
    choices = generic_stripes(mod, gv.name_hint)
    assert choices == [96, 32, 16, 8, 4, 2, 1]
    with unittest.TestCase().assertRaisesRegex(LinkError, 'rectangular output boundary'):
        _schedule(mod, gv, prim, stripe=48)


@T.prim_func
def unit_rows(a: T.Buffer((7, 5), 'float16'), out: T.Buffer((7,), 'float16')):
    temp = T.alloc_buffer((7,), 'float16', scope='global.sram')
    for i, j in T.grid(7, 1):
        with T.block('stage'):
            vi, vj = T.axis.remap('SS', [i, j])
            temp[vi] = a[vi, 2]
    for i in range(7):
        with T.block('write'):
            vi = T.axis.spatial(7, i)
            out[vi] = temp[vi]


def test_native_unit_rows_match_python_odd_cell_dma():
    if not native.load():
        raise unittest.SkipTest('native codegen is not built')
    scheduled = tir.transform.ConvertBlocksToOpaque()(tvm.IRModule({'main': unit_rows}))['main']
    sram, _ = _place_sram(_sram_buffers(scheduled), 4096)
    addresses, slots = [0, 128], tuple(i*128 for i in range(6))
    detail = {}
    actual = native.codegen_kernel(scheduled, addresses, sram, slots, {}, 0, diagnostics=detail)
    assert detail['status'] == 'emitted', detail
    asm = V09Asm()
    walker = Walker(asm, {}, SramEmitter(asm))
    walker.scratch_slots = slots
    for p, address in zip(scheduled.params, addresses):
        b = scheduled.buffer_map[p]
        walker.bases[b.data] = address
        walker.dtypes[b.data] = str(b.dtype)
    for b, address in sram.items():
        walker.declare_sram(b, address)
    walker.run(scheduled, {})
    np.testing.assert_array_equal(actual, asm.words)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_repeated_producers_are_rejected_before_codegen,
        test_fusion_falls_back_and_preserves_cache_stack_values,
        test_stripes_do_not_write_uncomputed_rectangular_holes,
        test_native_unit_rows_match_python_odd_cell_dma))


if __name__ == '__main__':
    unittest.main()
