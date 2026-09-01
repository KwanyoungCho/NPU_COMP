"""Autoregressive generation on the standard path.

The machine has no dynamic shapes, so generation is a sequence of static
programs: one ``prefill_cache`` at the prompt length, then one ``decode``
program per context length.  The KV cache lives on the host between steps --
keys transposed to [L, kv, hd, ctx] (the layout the score matmul reads) and
values as [L, kv, ctx, hd]; each decode returns its own K/V rows and the host
appends them, exactly the convention the validated hand-written path used.
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


def run_tuple_program(lowered, entry, inputs, weights):
    """Link ``entry`` and run it on the C-model; -> list of output arrays.

    ``inputs`` maps the entry's runtime parameter names to arrays; the
    function's result is a tuple, read member by member from the plan.
    """
    transformed = _transform(lowered, entry, weights)
    asm, plan = npu_link.compile_program(lowered, entry)
    planned, _ = npu_memplan.assign_addresses(lowered, entry)
    func = planned[entry]
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
    return outputs, len(asm.words), counters


def generate(family, config, assets, token_ids, steps, runner=None,
             progress=None):
    """Greedy generation: prompt ``token_ids`` -> ``steps`` new tokens.

    ``runner`` defaults to the C-model (:func:`run_tuple_program`); passing a
    different callable (e.g. an llvm VM wrapper) reruns the identical
    programs elsewhere, which is how the tests cross-check each step.
    """
    runner = runner or run_tuple_program
    seq = len(token_ids)
    tokens = [int(t) for t in token_ids]
    context = seq + 1

    mod, params, cfg = family.build_generate(config, seq, context)
    weights = [_nd(value) for value in family.load_params(assets, params, cfg)]
    lowered = _lower(mod)

    inputs = family.runtime_inputs(assets, cfg, tokens)
    cos, sin = family.rope_inputs(cfg, np.arange(seq))
    prefill_inputs = {"input_embeds": inputs["input_embeds"],
                      "cos": cos, "sin": sin, "mask": inputs["mask"]}
    (logits, k_cache, v_cache), words, _ = _run3(
        runner, lowered, "prefill_cache", prefill_inputs, weights)
    generated = [int(np.argmax(logits[-1].astype(np.float32)))]
    if progress:
        progress("prefill", seq, generated[-1], words)

    for step in range(1, steps):
        position = seq + step - 1
        # this step's token sits at ``position``; it attends over everything
        # up to and including itself
        context = position + 1
        mod, params, cfg = family.build_generate(config, seq, context)
        lowered = _lower(mod)
        cos, sin = family.rope_inputs(cfg, [position])
        step_inputs = {
            "input_embeds": assets.embedding(
                [generated[-1]]).astype(np.float16),
            "cos": cos, "sin": sin,
            "k_cache": np.ascontiguousarray(k_cache),
            "v_cache": np.ascontiguousarray(v_cache),
        }
        (logits, k_new, v_new), words, _ = _run3(
            runner, lowered, "decode", step_inputs, weights)
        k_cache = np.concatenate([k_cache, k_new], axis=3)
        v_cache = np.concatenate([v_cache, v_new], axis=2)
        generated.append(int(np.argmax(logits[-1].astype(np.float32))))
        if progress:
            progress("decode", position + 1, generated[-1], words)
    return generated


def _run3(runner, lowered, entry, inputs, weights):
    outputs, words, counters = runner(lowered, entry, inputs, weights)
    if len(outputs) != 3:
        raise RuntimeError(f"{entry}: expected 3 outputs, got {len(outputs)}")
    return tuple(outputs), words, counters


def _nd(value):
    import tvm

    return tvm.nd.array(value)


def llvm_runner(lowered, entry, inputs, weights):
    """The same programs on the llvm build -- the per-step cross-check."""
    import tvm
    from tvm import relax

    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    transformed = vm[entry + "_transform_params"]([weights])
    out = vm[entry](*[tvm.nd.array(inputs[name]) for name in inputs],
                    *transformed)
    return [t.numpy() for t in out], 0, {}
