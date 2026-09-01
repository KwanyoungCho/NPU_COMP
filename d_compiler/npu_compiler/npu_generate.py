"""Autoregressive generation on the standard path.

A deployed model is exactly two compiled programs: ``prefill_cache`` at the
prompt length and ONE ``decode`` at a fixed cache capacity.  The machine has
no dynamic shapes, so decode always attends over ``capacity + 1`` slots and
an additive mask input carries the current length -- slots past it get the
fp16 floor and softmax zeroes them.  The KV cache lives on the host at full
capacity (keys transposed to [L, kv, hd, C], values [L, kv, C, hd]); each
step the graph returns this token's K/V rows and the host writes them into
the next slot.  No per-step re-linking.
"""
from __future__ import annotations

import numpy as np

from . import npu_legalize, npu_link, npu_memplan
from . import tvm_pipeline as pipeline


def _lower(mod):
    return pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=True)(mod)


def _transform(lowered, entry, weights):
    import tvm
    from tvm import relax

    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    return [t.numpy() for t in vm[entry + "_transform_params"]([weights])]


def run_tuple_program(lowered, entry, inputs, weights, cached=None):
    """Link ``entry`` once and run it on the C-model; -> (outputs, words,
    program).  Passing the returned program back as ``cached`` reuses the
    linked words and plan with only the inputs changed -- the decode loop's
    whole point.
    """
    if cached is None:
        transformed = _transform(lowered, entry, weights)
        asm, plan = npu_link.compile_program(lowered, entry)
        planned, _ = npu_memplan.assign_addresses(lowered, entry)
        func = planned[entry]
        cached = (transformed, asm, plan, func)
    transformed, asm, plan, func = cached
    values = dict(zip([p.name_hint for p in func.params],
                      [inputs[name] for name in inputs] + transformed))

    image = npu_link.build_image(plan, func, values)
    if image.size % 2:
        image = np.concatenate([image, np.zeros(1, dtype="<f2")])
    from .v09_runtime import run as run_v09

    images, counters = run_v09(asm.words, image.view("<u4"))
    flat = np.ascontiguousarray(images[-1]).view("<f2")

    out_name = npu_link.output_var(func)
    members = plan.tuples.get(out_name, [out_name])
    body = func.body.body
    infos = (body.struct_info.fields
             if hasattr(body.struct_info, "fields") else [body.struct_info])
    outputs = []
    for member, info in zip(members, infos):
        shape = [int(d) for d in info.shape]
        address = plan.address[member]
        count = int(np.prod(shape))
        outputs.append(flat[address:address + count].reshape(shape).copy())
    return outputs, len(asm.words), cached


def length_mask(capacity, length, dtype=np.float16):
    """Additive [1, 1, capacity+1]: zero over the ``length`` valid slots and
    over this token's own slot (the last), the fp16 floor elsewhere."""
    mask = np.full((1, 1, capacity + 1), -65504.0, dtype=np.float32)
    mask[0, 0, :length] = 0.0
    mask[0, 0, capacity] = 0.0
    return mask.astype(dtype)


def generate(family, config, assets, token_ids, steps, capacity=None,
             runner=None, progress=None):
    """Greedy generation: prompt ``token_ids`` -> ``steps`` new tokens.

    Links two programs once -- prefill_cache and decode at ``capacity``
    (default: just enough for the requested tokens) -- then loops, reusing
    the decode program with only its inputs changing.  ``runner`` defaults
    to the C-model; the llvm runner reruns the identical programs on CPU,
    which is how the tests cross-check each step.
    """
    runner = runner or run_tuple_program
    seq = len(token_ids)
    tokens = [int(t) for t in token_ids]
    capacity = capacity or (seq + steps - 1)
    if capacity < seq + steps - 1:
        raise ValueError(f"capacity {capacity} cannot hold {seq}+{steps - 1}")

    mod, params, cfg = family.build_generate(config, seq, capacity)
    weights = [_nd(value) for value in family.load_params(assets, params, cfg)]
    lowered = _lower(mod)

    inputs = family.runtime_inputs(assets, cfg, tokens)
    cos, sin = family.rope_inputs(cfg, np.arange(seq))
    prefill_inputs = {"input_embeds": inputs["input_embeds"],
                      "cos": cos, "sin": sin, "mask": inputs["mask"]}
    (logits, k_rows, v_rows), words, _ = _run3(
        runner, lowered, "prefill_cache", prefill_inputs, weights)
    generated = [int(np.argmax(logits[-1].astype(np.float32)))]
    if progress:
        progress("prefill", seq, generated[-1], words)

    layers, kv, hd = k_rows.shape[0], k_rows.shape[1], k_rows.shape[2]
    k_cache = np.zeros((layers, kv, hd, capacity), dtype=np.float16)
    v_cache = np.zeros((layers, kv, capacity, hd), dtype=np.float16)
    k_cache[:, :, :, :seq] = k_rows
    v_cache[:, :, :seq, :] = v_rows

    decode = None
    for step in range(1, steps):
        position = seq + step - 1
        cos, sin = family.rope_inputs(cfg, [position])
        step_inputs = {
            "input_embeds": assets.embedding(
                [generated[-1]]).astype(np.float16),
            "cos": cos, "sin": sin,
            "k_cache": k_cache, "v_cache": v_cache,
            "mask": length_mask(capacity, position),
        }
        (logits, k_new, v_new), words, decode = _run3(
            runner, lowered, "decode", step_inputs, weights, cached=decode)
        k_cache[:, :, :, position] = k_new[:, :, :, 0]
        v_cache[:, :, position, :] = v_new[:, :, 0, :]
        generated.append(int(np.argmax(logits[-1].astype(np.float32))))
        if progress:
            progress("decode", position + 1, generated[-1], words)
    return generated


def _run3(runner, lowered, entry, inputs, weights, cached=None):
    outputs, words, cached = runner(lowered, entry, inputs, weights,
                                    cached=cached)
    if len(outputs) != 3:
        raise RuntimeError(f"{entry}: expected 3 outputs, got {len(outputs)}")
    return tuple(outputs), words, cached


def _nd(value):
    import tvm

    return tvm.nd.array(value)


def llvm_runner(lowered, entry, inputs, weights, cached=None):
    """The same programs on the llvm build -- the per-step cross-check."""
    import tvm
    from tvm import relax

    if cached is None:
        vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
        cached = (vm, vm[entry + "_transform_params"]([weights]))
    vm, transformed = cached
    out = vm[entry](*[tvm.nd.array(inputs[name]) for name in inputs],
                    *transformed)
    return [t.numpy() for t in out], 0, cached
