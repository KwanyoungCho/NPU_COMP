"""TVM reference scheduling for the hardware's 64-element partial sums."""
import tvm
from tvm import tir, relax
from .npu_kernel import matmul_blocks


def tile_order_reference(module):
    """Use standard split/rfactor to reproduce the device's FP32 grouping.

    TVM reductions permit reassociation. CPU validation and constant evaluation
    explicitly use the device grouping, avoiding a CPU-dependent constant
    result at cancellation boundaries.
    """
    result = tvm.IRModule(module.functions, attrs=module.attrs)
    for gv, fn in module.functions.items():
        if not isinstance(fn, tir.PrimFunc):
            continue
        dots = [d for d in matmul_blocks(fn) if d.dtypes == ("float16","float16","float32")
                and d.extents[2] > 64]
        if not dots:
            continue
        schedule = tir.Schedule(tvm.IRModule({gv: fn}))
        for dot in dots:
            block = schedule.get_block(dot.name, func_name=gv.name_hint)
            outer, _ = schedule.split(schedule.get_loops(block)[-1], [None, 64])
            schedule.rfactor(outer, factor_axis=0)
        result[gv] = schedule.mod[gv]
    return result


@tvm.transform.module_pass(opt_level=0, name="FoldNpuConstants")
class FoldConstants:
    def transform_module(self, module, context):
        folded = relax.transform.FoldConstant()(tile_order_reference(module))
        # Device scheduling still operates on the unfactored logical dot,
        # which tensorization splits into exactly the same 64-wide partials.
        # Keep the original bodies for all calls that were not constant.
        for gv, fn in module.functions.items():
            if isinstance(fn, tir.PrimFunc) and gv in folded.functions:
                folded[gv] = fn
        return folded
