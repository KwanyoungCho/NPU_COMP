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

import os

from tvm import relax, tir

from . import native, npu_intrin, npu_memplan, npu_w8a8
from .backend_v09 import V09Asm
from .device import PROFILES
from .peephole import eliminate_dead_stores
from .tir_codegen_v09 import SramEmitter, V09TirError, Walker




class LinkError(RuntimeError):
    pass


def _is_matmul(prim):
    found = []

    def visit(node):
        if isinstance(node, tir.Block) and node.name_hint == "matmul":
            found.append(node)

    tir.stmt_functor.post_order_visit(prim.body, visit)
    return bool(found)


def _schedule(module, gvar, prim, tile=64):
    """Apply the NPU schedule this kernel needs; every kernel ends up staged."""
    name = gvar.name_hint
    try:
        if _is_matmul(prim):
            scheduled = npu_intrin.schedule_matmul_sram(module, name, tile)
        else:
            scheduled = npu_intrin.schedule_generic_sram(module, name)
    except Exception as error:
        raise LinkError(f"{name}: {type(error).__name__}: "
                        f"{str(error).splitlines()[-1][:120]}") from error
    # cache_read allocates a buffer with the producer's full shape even when a
    # single tile is staged, so real weight shapes would need tens of MiB of
    # SRAM.  The standard passes shrink each buffer to the region actually
    # accessed; they require init blocks lowered first, which turns reductions
    # into a guarded store the codegen understands.
    from tvm import IRModule, tir as _tir
    import tvm as _tvm
    single = IRModule({gvar: scheduled[name]})
    single = _tvm.transform.Sequential([
        _tir.transform.LowerInitBlock(),
        _tir.transform.PlanAndUpdateBufferAllocationLocation(),
        _tir.transform.ConvertBlocksToOpaque(),
        _tir.transform.CompactBufferAllocation(),
    ])(single)
    return single[gvar]


def _sram_buffers(prim):
    """The kernel's cache buffers, in allocation order.

    Buffers may be allocated at any block after
    PlanAndUpdateBufferAllocationLocation moves them inward, so collect them
    from the whole body rather than the root block.  Kept separate from
    placement because the walk is over the whole body while placement is a
    handful of additions -- and the same kernel is placed once per call site.
    """
    buffers = []
    seen = set()

    def visit(node):
        if isinstance(node, tir.Block):
            for buffer in node.alloc_buffers:
                if buffer.data not in seen:
                    seen.add(buffer.data)
                    buffers.append(buffer)

    tir.stmt_functor.post_order_visit(prim.body, visit)
    return buffers


def _place_sram(buffers, cursor=0, capacity=None):
    """Bump-allocate collected buffers from ``cursor``."""
    placement = {}
    widths = {"float16": 4, "int8": 2, "float32": 8}
    for buffer in buffers:
        # every kernel-local temporary lives in SRAM: compute units cannot
        # address global memory, so a buffer allocated inside a kernel has
        # nowhere else to go (padding buffers from pad_einsum included)
        size = 1
        for dim in buffer.shape:
            size *= int(dim)
        placement[buffer] = cursor
        cursor += size * widths[str(buffer.dtype)]
        cursor = (cursor + 7) // 8 * 8
        if capacity is not None and cursor > capacity:
            raise LinkError(
                f"kernel exceeds SRAM capacity: {buffer.name} "
                f"{[int(d) for d in buffer.shape]} {buffer.dtype} takes it to "
                f"{cursor / 2 / 1024 / 1024:.2f} MiB of "
                f"{capacity / 2 / 1024 / 1024:.2f} MiB")
    return placement, cursor


def _scratch_row(prim):
    """Longest row an expression temporary in this kernel may have to hold.

    ``_materialize`` serializes an expression over a whole row at a time, so a
    slot has to be at least as wide as the widest row the kernel touches.  A
    fixed size silently corrupted the neighbouring slot for anything wider --
    Qwen3's 9728-wide FFN against 8192-element slots, where Llama's 8192-wide
    one happened to fit exactly.
    """
    widest = 1
    buffers = list(prim.buffer_map.values())

    def visit(node):
        if isinstance(node, tir.Block):
            buffers.extend(node.alloc_buffers)

    tir.stmt_functor.post_order_visit(prim.body, visit)
    for buffer in buffers:
        if len(buffer.shape):
            widest = max(widest, int(buffer.shape[-1]))
    return widest


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


def _native_default():
    """Emit natively when the library is there; ``NPU_NATIVE=0`` turns it off.

    Safe as a default because a kernel the native walker does not cover falls
    back to Python, and the two are gated word-for-word by
    tests/test_native_codegen.py.
    """
    if os.environ.get("NPU_NATIVE", "1") == "0":
        return None
    return "use" if native.load() else None


def _compare_native(name, python_words, native_words):
    """Fail loudly on the first word where the two walkers disagree.

    A silent divergence here would be a wrong program with no symptom other
    than wrong numbers, so the check is exact and the message says where.
    """
    if len(python_words) != len(native_words):
        raise LinkError(f"{name}: native emitted {len(native_words)} words, "
                        f"python {len(python_words)}")
    for index, (mine, theirs) in enumerate(zip(python_words, native_words)):
        if int(mine) != int(theirs):
            raise LinkError(f"{name}: word {index} differs -- python "
                            f"{int(mine):#010x}, native {int(theirs):#010x}")


def compile_program(mod, func_name="prefill", snapshot_at=None,
                    peephole=True, profile=None, native_mode=None):
    """Lowered IRModule -> (assembler, StaticPlan).

    ``mod`` must already have gone through the graph pipeline with fusion off.

    ``peephole`` drops descriptor writes that store the value the register
    already holds, which the straight-line stream makes decidable exactly.

    ``snapshot_at`` is a set of kernel indices after which to emit SNAPSHOT.
    Each one appends the whole memory image at that point, which is how a
    divergence is traced back to the kernel that caused it; the captured
    outputs are listed in ``plan.snapshots`` as (index, var name, struct info).

    ``native_mode`` selects the native (C++) walker: ``"use"`` emits with it
    wherever it has coverage and falls back to Python elsewhere, ``"compare"``
    emits with Python but checks the native words match exactly and records
    the outcome in ``plan.native``.  Default is Python only.  Each kernel is
    walked independently -- the walker flushes the accumulator before it
    returns -- so mixing the two per kernel is safe.
    """
    profile = profile or PROFILES["v09"]
    if native_mode is None:
        native_mode = _native_default()
    elif native_mode == "python":
        native_mode = None              # explicit opt-out, for the gate below
    planned, plan = npu_memplan.assign_addresses(mod, func_name)

    # scalar literals used by pointwise kernels live in a small pool that the
    # host fills; the program stages it into SRAM once, before any kernel.
    # Only kernels the device function actually calls contribute -- lifted
    # parameter transforms run on the host and their literals (reduce
    # identities among them) may not even fit FP16.
    called = set()
    for block in planned[func_name].body.blocks:
        for binding in block.bindings:
            value = binding.value
            if isinstance(value, relax.Call) and isinstance(
                    value.op, relax.GlobalVar):
                called.add(value.op)
    constants = sorted(set().union(*[
        _collect_constants(planned[gv]) for gv in called
        if isinstance(planned[gv], tir.PrimFunc)] or [set()]))
    # rsqrt and tanh need literal 1 and 2 whether or not the TIR mentions them
    constants = sorted(set(constants) | {1.0, 2.0})
    if len(constants) % 2:
        constants.append(0.0)
    plan.constant_values = constants
    plan.constant_base = plan.top
    plan.top += len(constants)

    # one fp32 literal 1.0 rides just after the fp16 pool: VDEQUANT's scale
    # read is fp32, and pointing it at 1.0 makes the conversion pure
    plan.one_fp32_base = plan.top
    plan.top += 2
    asm = V09Asm()
    emitter = SramEmitter(asm)
    const_nib = 0
    pool_bytes = len(constants) * 2 + 4
    emitter.dma_in(plan.constant_base * 2, const_nib, pool_bytes)
    const_addr = {value: const_nib + index * 4
                  for index, value in enumerate(constants)}
    one_fp32_nib = len(constants) * 4
    # temporaries for serializing expression trees into vector steps; each
    # kernel gets slots wide enough for its own longest row (see _scratch_row)
    slot_count = profile.scratch_slots
    scratch_base = (len(constants) * 4 + 8 + 7) // 8 * 8
    kernels = 0
    recipes = {}        # GlobalVar -> (scheduled, scratch row, SRAM buffers)
    for block in planned[func_name].body.blocks:
        for binding in block.bindings:
            call = binding.value
            if not (isinstance(call, relax.Call)
                    and isinstance(call.op, relax.GlobalVar)):
                continue
            prim = planned[call.op]
            if not isinstance(prim, tir.PrimFunc):
                continue
            if npu_w8a8.is_w8a8(prim):
                # the W8A8 matmul is emitted as one validated sequence; its
                # TIR body only defines the semantics the llvm build runs
                addresses = []
                for arg in call.args:
                    addresses.append(plan.address[arg.name_hint])
                asm_start = len(asm.words)
                scratch_elems = _scratch_row(prim)
                sram_start = scratch_base + slot_count * scratch_elems * 4
                npu_w8a8.emit(asm, emitter, prim, addresses, sram_start)
                kernels += 1
                continue
            # A 28-layer model calls ~32 distinct kernels 1,206 times, and
            # scheduling is a pure function of the PrimFunc and the tile, so
            # doing it per call site was 64% of the link.
            recipe = recipes.get(call.op)
            if recipe is None:
                scheduled = _schedule(planned, call.op, prim, profile.tile)
                recipe = (scheduled, _scratch_row(scheduled),
                          _sram_buffers(scheduled))
                recipes[call.op] = recipe
            scheduled, scratch_elems, sram_buffers = recipe
            addresses = []
            for arg in call.args:
                if isinstance(arg, relax.Constant):
                    array = arg.data.numpy()
                    key = (str(array.dtype), array.shape, array.tobytes())
                    addresses.append(plan.graph_constants[key])
                    continue
                if isinstance(arg, relax.Tuple):
                    for field in arg.fields:
                        addresses.append(plan.address[field.name_hint])
                    continue
                if not isinstance(arg, relax.Var):
                    raise LinkError(f"{call.op.name_hint}: non-var argument {arg}")
                if arg.name_hint not in plan.address:
                    raise LinkError(f"{call.op.name_hint}: unplaced {arg.name_hint}")
                addresses.append(plan.address[arg.name_hint])
            scratch_slots = tuple(scratch_base + index * scratch_elems * 4
                                  for index in range(slot_count))
            sram_start = scratch_base + slot_count * scratch_elems * 4
            sram_map, sram_top = _place_sram(sram_buffers, sram_start,
                                             profile.sram_nibbles)
            # what this kernel actually occupies, scratch slots included --
            # the headroom question (SRAM is 8 MiB; how much is idle?) is
            # answered from here
            plan.sram_peak[call.op.name_hint] = sram_top
            native_words = None
            if native_mode:
                native_words = native.codegen_kernel(
                    scheduled, addresses, sram_map, scratch_slots,
                    const_addr, one_fp32_nib)
                plan.native[call.op.name_hint] = native_words is not None
            if native_mode == "use" and native_words is not None:
                asm.words.extend(int(word) for word in native_words)
                asm.tags.extend([None] * len(native_words))
            else:
                start = len(asm.words)
                walker = Walker(asm, {}, emitter)
                walker.constants = const_addr
                walker.one_fp32 = one_fp32_nib
                walker.scratch_slots = scratch_slots
                # bind by parameter order -- buffer_map is a map, and its
                # iteration order is not the signature's
                for param, address in zip(scheduled.params, addresses):
                    buffer = scheduled.buffer_map[param]
                    walker.bases[buffer.data] = address
                    walker.dtypes[buffer.data] = str(buffer.dtype)
                for buffer, nibble in sram_map.items():
                    walker.declare_sram(buffer, nibble)
                walker.run(scheduled, {})
                if native_mode == "compare" and native_words is not None:
                    _compare_native(call.op.name_hint, asm.words[start:],
                                    native_words)
            if snapshot_at is not None and kernels in snapshot_at:
                asm.snapshot()
                target = call.args[-1]
                plan.snapshots.append(
                    (kernels, getattr(target, "name_hint", None),
                     target.struct_info))
            kernels += 1
    asm.halt()
    if peephole:
        asm.words[:] = eliminate_dead_stores(asm.words)
    asm.kernel_count = kernels
    return asm, plan


def build_image(plan, func, values):
    """Assemble the initial global memory image the program expects.

    ``values`` maps parameter name -> array, in the planned function's own
    parameter names (after LiftTransformParams these are the transformed
    weights, not the checkpoint tensors).
    """
    import numpy as np

    image = np.zeros(plan.top, dtype="<f2")
    raw = image.view(np.uint8)

    def place(address, array):
        data = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        start = address * 2
        raw[start:start + data.size] = data

    for param in func.params:
        name = param.name_hint
        if name not in values:
            raise LinkError(f"missing value for parameter {name}")
        place(plan.address[name], values[name])
    for address, array in plan.const_data:
        place(address, array)
    if getattr(plan, "constant_values", None):
        place(plan.constant_base,
              np.asarray(plan.constant_values, dtype="<f2"))
    if getattr(plan, "one_fp32_base", None) is not None:
        place(plan.one_fp32_base, np.asarray([1.0], dtype="<f4"))
    return image


def output_var(func):
    body = func.body
    return body.body.name_hint if hasattr(body.body, "name_hint") else None


def run_program(asm, plan, func, values, output_shape):
    """Execute a linked program on the v09 C-model and return its output.

    ``asm`` may be an assembler or the word list itself.
    """
    import numpy as np

    from .v09_runtime import run as run_v09

    image = build_image(plan, func, values)
    if image.size % 2:
        image = np.concatenate([image, np.zeros(1, dtype="<f2")])
    images, counters = run_v09(getattr(asm, "words", asm), image.view("<u4"))
    final = np.ascontiguousarray(images[-1]).view("<f2")
    address = plan.address[output_var(func)]
    count = int(np.prod(output_shape))
    return final[address:address + count].reshape(output_shape), counters
