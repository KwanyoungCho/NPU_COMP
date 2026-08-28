"""Link scheduled kernels into one v09 instruction stream.

The machine has no call mechanism: a program is a single straight-line
instruction sequence over one flat memory.  This module walks the planned
Relax function -- whose bindings are direct PrimFunc calls after
``CallTIRRewrite`` -- and, for each call, schedules the kernel, binds its
buffers to the static addresses from :mod:`npu_memplan`, and appends its
instructions to a shared assembler.

Fusion is off on this path.  A fused kernel computes a compound expression
per element, but the vector unit applies one operation to a whole vector at a
time, so a fused body would have to be re-serialized into vector steps with
SRAM temporaries.  That re-serialization is a separate design question; until
then each PrimFunc is a single operation.
"""
from __future__ import annotations

from tvm import relax, tir

from . import npu_intrin, npu_memplan
from .backend_v09 import V09Asm
from .tir_codegen_v09 import SramEmitter, V09TirError, Walker

SRAM_NIBBLES = 8 * 1024 * 1024 * 2


class LinkError(RuntimeError):
    pass


def _is_matmul(prim):
    found = []

    def visit(node):
        if isinstance(node, tir.Block) and node.name_hint == "matmul":
            found.append(node)

    tir.stmt_functor.post_order_visit(prim.body, visit)
    return bool(found)


def _schedule(module, gvar, prim):
    """Apply the NPU schedule this kernel needs; every kernel ends up staged."""
    name = gvar.name_hint
    try:
        if _is_matmul(prim):
            return npu_intrin.schedule_matmul_sram(module, name)[name]
        return npu_intrin.schedule_generic_sram(module, name)[name]
    except Exception as error:
        raise LinkError(f"{name}: {type(error).__name__}: "
                        f"{str(error).splitlines()[-1][:120]}") from error


def _sram_layout(prim, cursor=0):
    """Bump-allocate the kernel's cache buffers in SRAM."""
    placement = {}
    body = prim.body
    if not isinstance(body, tir.BlockRealize):
        return placement, cursor
    for buffer in body.block.alloc_buffers:
        if buffer.scope() != npu_intrin.SRAM_SCOPE:
            continue
        size = 1
        for dim in buffer.shape:
            size *= int(dim)
        placement[buffer] = cursor
        cursor += size * 4
        cursor = (cursor + 7) // 8 * 8
        if cursor > SRAM_NIBBLES:
            raise LinkError("kernel exceeds SRAM capacity")
    return placement, cursor


def _collect_constants(prim):
    """Scalar literals a pointwise block needs materialized in memory."""
    values = set()

    def visit(node):
        if isinstance(node, tir.BufferStore):
            def scan(expr):
                if isinstance(expr, tir.FloatImm):
                    values.add(float(expr.value))
            tir.stmt_functor.post_order_visit(node.value, scan)

    tir.stmt_functor.post_order_visit(prim.body, visit)
    return values


def compile_program(mod, func_name="prefill"):
    """Lowered IRModule -> (assembler, StaticPlan).

    ``mod`` must already have gone through the graph pipeline with fusion off.
    """
    planned, plan = npu_memplan.assign_addresses(mod, func_name)

    # scalar literals used by pointwise kernels live in a small pool that the
    # host fills; the program stages it into SRAM once, before any kernel
    constants = sorted(set().union(*[
        _collect_constants(fn) for _, fn in planned.functions.items()
        if isinstance(fn, tir.PrimFunc)] or [set()]))
    constants = sorted(set(constants) | {1.0})     # rsqrt needs a literal one
    if len(constants) % 2:
        constants.append(0.0)
    plan.constant_values = constants
    plan.constant_base = plan.top
    plan.top += len(constants)

    asm = V09Asm()
    emitter = SramEmitter(asm)
    const_nib = 0
    if constants:
        emitter.dma_in(plan.constant_base, const_nib, len(constants))
    const_addr = {value: const_nib + index * 4
                  for index, value in enumerate(constants)}
    # temporaries for serializing expression trees into vector steps
    scratch_elems = 8192
    slot_count = 6
    scratch_base = (len(constants) * 4 + 7) // 8 * 8
    scratch_slots = tuple(scratch_base + i * scratch_elems * 4
                          for i in range(slot_count))
    sram_start = scratch_base + slot_count * scratch_elems * 4
    kernels = 0
    for block in planned[func_name].body.blocks:
        for binding in block.bindings:
            call = binding.value
            if not (isinstance(call, relax.Call)
                    and isinstance(call.op, relax.GlobalVar)):
                continue
            prim = planned[call.op]
            if not isinstance(prim, tir.PrimFunc):
                continue
            scheduled = _schedule(planned, call.op, prim)
            addresses = []
            for arg in call.args:
                if not isinstance(arg, relax.Var):
                    raise LinkError(f"{call.op.name_hint}: non-var argument {arg}")
                if arg.name_hint not in plan.address:
                    raise LinkError(f"{call.op.name_hint}: unplaced {arg.name_hint}")
                addresses.append(plan.address[arg.name_hint])
            walker = Walker(asm, {}, emitter)
            walker.constants = const_addr
            walker.scratch_slots = scratch_slots
            for buffer, address in zip(scheduled.buffer_map.values(), addresses):
                walker.bases[buffer.data] = address
            for buffer, nibble in _sram_layout(scheduled, sram_start)[0].items():
                walker.declare_sram(buffer, nibble)
            walker.run(scheduled, {})
            kernels += 1
    asm.halt()
    asm.kernel_count = kernels
    return asm, plan
