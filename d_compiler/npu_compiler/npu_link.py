"""Link scheduled kernels into one v09 instruction stream.

The machine has no call mechanism: a program is a single straight-line
instruction sequence over one flat memory.  This module walks the planned
Relax function -- whose bindings are direct PrimFunc calls after
``CallTIRRewrite`` -- and, for each call, schedules the kernel, binds its
buffers to the static addresses from :mod:`npu_memplan`, and appends its
instructions to a shared assembler.

The public build chooses FuseOps/FuseTIR regions through target legality and
cost probes. This linker accepts the resulting supported scheduled compounds;
the lower-level boolean graph pipeline remains available for explicit builds.
"""
from __future__ import annotations

import os

from tvm import relax, tir

from . import native, npu_memplan, npu_w8a8
from .v09_asm import V09Asm
from .device import PROFILES
from .v09_contract import NUMERIC_POLICY, CONTRACT_HASH
from .peephole import eliminate_dead_stores
from .npu_compilation import compilation, current
from .tir_codegen_v09 import SramEmitter, Walker
from .npu_errors import LinkError
from .npu_abi import bindings, validate_entry, read_outputs, tensor_leaves
from .npu_schedule import _collect_constants, _fit_kernel, _place_sram


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
                    peephole=True, profile=None, native_mode=None,
                    w8a8_mode="tir", residency=False, views=True):
    """Link lowered IR; reuse structural schedules within the enclosing build.

    See :func:`_compile_program` for target/emitter options. ``views=False``
    keeps reshape copies as an independent validation baseline.
    """
    with compilation() as context:
        from dataclasses import replace
        from .v09_stats import analyze, cost
        effective = profile or PROFILES['v09']
        baseline = None
        if residency and effective.joint_memory and not snapshot_at and w8a8_mode == 'tir':
            baseline = _compile_program(mod, func_name, snapshot_at, peephole,
                replace(effective, joint_memory=False), native_mode, w8a8_mode, residency, views)
            from .npu_memory_search import discover
            if not discover(mod,baseline[1])[1]:
                asm,plan=baseline
                plan.joint_memory=dict(policy='no retainable intermediate',applied=False,
                                       retained_values=[],calls=[])
                plan.compilation=context.describe()
                return asm,plan
        asm, plan = _compile_program(mod, func_name, snapshot_at, peephole,
                                     profile, native_mode, w8a8_mode, residency, views)
        if baseline is not None:
            before, after = (analyze(a.words,p,detail=False) for a,p in (baseline,(asm,plan)))
            selected = cost(after) < cost(before)
            detail = plan.joint_memory
            detail['program_comparison'] = dict(baseline=before,candidate=after,selected=selected)
            detail['applied'] = selected
            if not selected:
                asm,plan=baseline
                plan.joint_memory=dict(policy=detail['policy'],applied=False,
                    retained_values=[],calls=[],program_comparison=detail['program_comparison'],
                    candidate_plan=detail)
        with context.phase('dram_compaction'):
            npu_memplan.compact_unused_storage(asm.words,plan)
        if plan.cache_append is not None and plan.dram_compaction['saved_bytes']:
            _cache_append(plan,effective)
        plan.compilation = context.describe()
        return asm, plan


def _compile_program(mod, func_name="prefill", snapshot_at=None,
                     peephole=True, profile=None, native_mode=None,
                     w8a8_mode="tir", residency=False, views=True):
    """Lowered IRModule -> (assembler, StaticPlan).

    ``mod`` must already have gone through the graph pipeline. Compound
    kernels still need a supported schedule and vector expression contract.

    ``peephole`` drops descriptor writes that store the value the register
    already holds, which the straight-line stream makes decidable exactly.

    ``snapshot_at`` is a set of kernel indices after which to emit SNAPSHOT.
    Each one appends the whole memory image at that point, which is how a
    divergence is traced back to the kernel that caused it; the captured
    outputs are listed in ``plan.snapshots`` as (index, var name, struct info).

    ``native_mode`` selects the native (C++) walker: ``"use"`` emits with it
    wherever it has coverage and falls back to Python elsewhere, ``"compare"``
    emits with Python but checks the native words match exactly and records
    the outcome in ``plan.native``.  Default uses native when available;
    ``"python"`` explicitly disables it.  Each kernel is
    walked independently -- the walker flushes the accumulator before it
    returns -- so mixing the two per kernel is safe.
    """
    profile = profile or PROFILES["v09"]
    if w8a8_mode not in ("tir", "reference"):
        raise ValueError("w8a8_mode must be tir or reference")
    if profile.tile != 64 or profile.cell_bits != 32:
        raise LinkError("v09 intrinsic/ISA requires tile=64 and cell_bits=32")
    if profile.numeric_policy != NUMERIC_POLICY:
        raise LinkError(f"unsupported v09 numeric policy: {profile.numeric_policy}")
    if native_mode is None:
        native_mode = _native_default()
    elif native_mode == "python":
        native_mode = None              # explicit opt-out, for the gate below
    context = current()
    with context.phase("memory_planning"):
        planned, plan = npu_memplan.assign_addresses(mod, func_name, views=views)
    plan.func = planned[func_name]
    validate_entry(planned, func_name, plan)
    plan.numeric_policy = profile.numeric_policy
    plan.codegen_contract = CONTRACT_HASH
    plan.graph_optimizations = dict(repeated_gemv=int(mod.attrs.get('npu.repeated_gemv',0)) if mod.attrs else 0)
    if mod.attrs and "npu.fusion_selection" in mod.attrs:
        import json
        plan.fusion = json.loads(str(mod.attrs["npu.fusion_selection"]))
    plan.frontend_precision = str(planned[func_name].attrs.get(
        "npu.frontend_precision", "unspecified")) if planned[func_name].attrs else "unspecified"

    # scalar literals used by pointwise kernels live in a small pool that the
    # host fills; the program stages it into SRAM once, before any kernel.
    # Only kernels the device function actually calls contribute -- lifted
    # parameter transforms run on the host and their literals (reduce
    # identities among them) may not even fit FP16.
    called = set()
    for block_bindings in [bindings(plan.func)]:
        for binding in block_bindings:
            value = binding.value
            if isinstance(value, relax.Call) and isinstance(
                    value.op, relax.GlobalVar):
                called.add(value.op)
    # W8A8 has explicit quantizer workspace and its own scalar recipe;
    from .npu_verify import validate_primitive
    for gv in called:
        validate_primitive(planned[gv], gv.name_hint)
    # collecting semantic TIR literals only drags in its reduce identity
    # (-FLT_MAX), which cannot be represented in the fp16 pool and lands in the
    # image as -inf
    constants = sorted(set().union(*[
        _collect_constants(planned[gv]) for gv in called
        if isinstance(planned[gv], tir.PrimFunc)
        and not npu_w8a8.is_w8a8(planned[gv])] or [set()]))
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
    # kernel gets slots wide enough for its own longest row (see npu_schedule._scratch_row)
    slot_count = profile.scratch_slots
    scratch_base = (len(constants) * 4 + 8 + 7) // 8 * 8
    joint_choices = {}
    plan.joint_memory = dict(policy='disabled', calls=[])
    if residency and profile.joint_memory and not snapshot_at and w8a8_mode == 'tir':
        from .npu_memory_search import plan_program
        with context.phase('joint_schedule_memory'):
            joint_choices, plan.joint_memory = plan_program(
                planned, plan, profile, scratch_base, slot_count, const_addr,
                one_fp32_nib, native_mode)
    elif snapshot_at:
        plan.joint_memory['reason'] = 'snapshots require the original global memory image'
    plan.call_schedules = []
    plan.scheduled_calls = {}
    kernels = 0
    recipes = {}        # GlobalVar -> (scheduled, scratch row, SRAM layout)
    for block_bindings in [bindings(plan.func)]:
        for binding in block_bindings:
            call = binding.value
            if not (isinstance(call, relax.Call)
                    and isinstance(call.op, relax.GlobalVar)):
                continue
            prim = planned[call.op]
            if not isinstance(prim, tir.PrimFunc):
                continue
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
            if len(addresses) != len(prim.params):
                raise LinkError(f"{call.op.name_hint}: bound {len(addresses)} "
                                f"buffers for {len(prim.params)} parameters")
            span_start = len(asm.words)
            backend = "python"
            plan.native[call.op.name_hint] = False
            if kernels in joint_choices:
                choice = joint_choices[kernels]
                plan.scheduled_calls[kernels] = choice.scheduled
                asm.words.extend(choice.words)
                asm.tags.extend([None] * len(choice.words))
                sram_top = choice.top
                backend = 'native' if native_mode == 'use' and choice.native else 'python'
                plan.native[call.op.name_hint] = choice.native
                plan.native_diagnostics[call.op.name_hint] = choice.native_diagnostics
                plan.schedules[call.op.name_hint] = choice.diagnostics
                plan.call_schedules.append(dict(index=kernels, name=call.op.name_hint,
                    diagnostics=choice.diagnostics, parameter_sram_nibbles=choice.parameter_sram))
                plan.sram_layouts[call.op.name_hint] = choice.layout.describe(choice.workspace_start)
            elif npu_w8a8.is_w8a8(prim) and w8a8_mode == "reference":
                backend = "w8a8"
                # The direct emitter owns its temporaries; Walker's six
                # expression slots are not used on this path.
                diagnostics = {}
                plan.schedules[call.op.name_hint] = diagnostics
                try:
                    sram_top = npu_w8a8.emit(
                        asm, emitter, prim, addresses, scratch_base,
                        capacity=profile.sram_nibbles, diagnostics=diagnostics)
                except ValueError as error:
                    raise LinkError(f"{call.op.name_hint}: {error}") from error
            else:
                # A 28-layer model calls ~32 distinct kernels 1,206 times, and
                # scheduling is a pure function of the PrimFunc and the tile, so
                # doing it per call site was 64% of the link.
                recipe = recipes.get(call.op)
                if recipe is None:
                    def create_recipe():
                        diagnostics = {}
                        result = _fit_kernel(planned, call.op, prim, profile,
                                             scratch_base, slot_count, diagnostics)
                        return result, diagnostics
                    recipe, diagnostics = context.recipe(
                        prim, profile, scratch_base, slot_count, create_recipe)
                    plan.schedules[call.op.name_hint] = diagnostics
                    recipes[call.op] = recipe
                scheduled, scratch_elems, sram_layout = recipe
                scratch_slots = tuple(scratch_base + index * scratch_elems * 4
                                      for index in range(slot_count))
                sram_start = scratch_base + slot_count * scratch_elems * 4
                sram_map, sram_top = _place_sram(sram_layout, sram_start,
                                                 profile.sram_nibbles)
                if call.op.name_hint not in plan.sram_layouts:
                    plan.sram_layouts[call.op.name_hint] = sram_layout.describe(sram_start)
                # what this kernel actually occupies, scratch slots included --
                # the headroom question (SRAM is 8 MiB; how much is idle?) is
                # answered from here
                native_words = None
                if native_mode:
                    native_diagnostics = {}
                    plan.native_diagnostics[call.op.name_hint] = native_diagnostics
                    with context.phase("native_codegen", kernel=call.op.name_hint):
                        native_words = native.codegen_kernel(
                            scheduled, addresses, sram_map, scratch_slots,
                            const_addr, one_fp32_nib, diagnostics=native_diagnostics)
                    plan.native[call.op.name_hint] = native_words is not None
                if native_mode == "use" and native_words is not None:
                    backend = "native"
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
                    with context.phase("python_codegen", kernel=call.op.name_hint,
                                       fallback_reason=plan.native_diagnostics.get(
                                           call.op.name_hint, {}).get('reason')):
                        walker.run(scheduled, {})
                    if native_mode == "compare" and native_words is not None:
                        _compare_native(call.op.name_hint, asm.words[start:],
                                        native_words)
            plan.sram_peak[call.op.name_hint] = max(sram_top,plan.sram_peak.get(call.op.name_hint,0))
            output_stores = []
            parameters = {b.data for b in prim.buffer_map.values()}
            tir.stmt_functor.post_order_visit(prim.body, lambda node:
                output_stores.append(node) if isinstance(node, tir.BufferStore)
                and node.buffer.data in parameters else None)
            plan.kernel_records.append(dict(
                index=kernels, name=call.op.name_hint, backend=backend,
                sram_peak_nibbles=sram_top,
                canonical_fp16_output=bool(output_stores) and all(
                    str(s.buffer.dtype) == "float16" and not isinstance(s.value, tir.BufferLoad)
                    for s in output_stores),
                arguments=[dict(address_bytes=address * 2,
                                size_bytes=npu_memplan._numel(buffer.shape) *
                                           npu_memplan.DTYPE_BYTES[str(buffer.dtype)],
                                dtype=str(buffer.dtype))
                           for address, buffer in zip(addresses,
                               [prim.buffer_map[p] for p in prim.params])],
                span=(span_start, len(asm.words))))
            # which words came from which kernel, so traffic can be attributed
            # to a kernel rather than to a tensor (storage pools are reused, so
            # an address does not name one tensor).  Indices are into the
            # stream as emitted; remapped to final offsets after peephole.
            plan.kernel_span.append(
                (call.op.name_hint, span_start, len(asm.words)))
            if snapshot_at is not None and kernels in snapshot_at:
                asm.snapshot()
                target = call.args[-1]
                plan.snapshots.append(
                    (kernels, getattr(target, "name_hint", None),
                     target.struct_info))
            kernels += 1
    asm.halt()
    if residency:
        from .npu_residency import forward
        with context.phase("residency"):
            asm.words[:], remap, plan.residency = forward(asm.words, plan.kernel_records)
        plan.kernel_span = [(name, remap[first], remap[last])
                            for name, first, last in plan.kernel_span]
        for record, (_, first, last) in zip(plan.kernel_records, plan.kernel_span):
            record["span"] = (first, last)
        asm.tags[:] = [None] * len(asm.words)
    if peephole:
        boundaries = {offset for _, first, last in plan.kernel_span
                      for offset in (first, last)}
        with context.phase("peephole"):
            asm.words[:], remap = eliminate_dead_stores(asm.words, boundaries=boundaries)
        plan.kernel_span = [(name, remap[first], remap[last])
                            for name, first, last in plan.kernel_span]
        for record, (_, first, last) in zip(plan.kernel_records, plan.kernel_span):
            record["span"] = (first, last)
        # Linking does not assign role tags; keep this assembler invariant
        # even when descriptor elimination changes its word count.
        asm.tags[:] = [None] * len(asm.words)
    asm.kernel_count = kernels
    _cache_append(plan,profile)
    return asm, plan


def _cache_append(plan,profile):
    from .npu_cache import ATTRIBUTE, AppendPlan
    if plan.func.attrs and ATTRIBUTE in plan.func.attrs:
        from dataclasses import asdict
        import json
        plan.append_plan = AppendPlan(plan.func, plan, sram_bytes=profile.sram_bytes)
        plan.cache_append = dict(contract=json.loads(str(plan.func.attrs[ATTRIBUTE])),
            slices=[asdict(s) for s in plan.append_plan.slices],
            workspace_bytes=plan.append_plan.workspace_bytes,
            logical_update_bytes=plan.append_plan.logical_update_bytes,
            execution="pure base program; DecodeSession appends the declared DMA effect before HALT")
    else:
        plan.cache_append = None


def build_image(plan, func, values):
    """Assemble the initial global memory image the program expects.

    ``values`` maps parameter name -> array, in the planned function's own
    parameter names (after LiftTransformParams these are the transformed
    weights, not the checkpoint tensors).
    """
    import numpy as np

    image = np.zeros((plan.top * plan.unit_bytes + 1) // 2, dtype="<f2")
    raw = image.view(np.uint8)

    def place(address, array):
        data = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        start = address * plan.unit_bytes
        if start < 0 or start + data.size > raw.size:
            raise LinkError("input placement exceeds the memory image")
        raw[start:start + data.size] = data

    for param in func.params:
        name = param.name_hint
        if name not in values:
            raise LinkError(f"missing value for parameter {name}")
        def check_tuple(info, value, where):
            if isinstance(info, relax.TupleStructInfo):
                if not isinstance(value, (tuple, list)) or len(value) != len(info.fields):
                    raise LinkError(f"parameter {where}: expected tuple with {len(info.fields)} fields")
                for index, (field, item) in enumerate(zip(info.fields, value)):
                    check_tuple(field, item, f"{where}.{index}")
        check_tuple(param.struct_info, values[name], name)
        for leaf, shape, dtype in tensor_leaves(param.struct_info, name):
            value = values[name]
            for index in leaf[len(name):].split('.')[1:]:
                value = value[int(index)]
            value = np.asarray(value)
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise LinkError(f"parameter {leaf}: expected {shape} {dtype}, "
                                f"got {value.shape} {value.dtype}")
            place(plan.address[leaf], value)
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
    value = body.body if isinstance(body, relax.SeqExpr) else body
    return value.name_hint if isinstance(value, relax.Var) else None


def run_program(asm, plan, func, values, output_shape=None):
    """Execute a linked program on the v09 C-model and return its output.

    ``asm`` may be an assembler or the word list itself.
    """
    import numpy as np

    from .v09_runtime import run as run_v09

    if output_shape is not None:
        if isinstance(plan.outputs, list) or tuple(output_shape) != tuple(plan.outputs["shape"]):
            raise LinkError(f"output_shape must match the compiled output ABI: {plan.outputs}")
    image = build_image(plan, func, values)
    if image.size % 2:
        image = np.concatenate([image, np.zeros(1, dtype="<f2")])
    images, counters = run_v09(getattr(asm, "words", asm), image.view("<u4"))
    return read_outputs(images[-1], plan.outputs), counters
