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

import hashlib
from pathlib import Path

import numpy as np

from . import npu_legalize, npu_link, npu_memplan
from . import tvm_pipeline as pipeline


def _lower(mod):
    return pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=True)(mod)


def _run_transform(lowered, entry, weights):
    """Execute the lifted parameter transform on the host.

    ``LiftTransformParams`` expresses the transform as IR (the weight
    transposes, and the scale/quantize of a quantized build), so running that
    IR is the only way to get exactly the values the linked program's image
    expects -- reimplementing it in numpy would put the same rule in two
    places, and a 1-ULP disagreement in a scale is enough to flip quantized
    weights.
    """
    import tvm
    from tvm import relax

    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    return [t.numpy() for t in vm[entry + "_transform_params"]([weights])]


def _transform(lowered, entry, weights, cache=None):
    """The host-side transform, computed once and reused from disk.

    The transform does not change between runs of the same model, but running
    it means building the whole module for CPU and recomputing several GiB of
    weights every time.  ``cache`` is a directory the caller makes specific to
    the checkpoint; within it the entries are keyed by the transform
    function's own IR, so changing the graph or the quantization mode misses
    the cache rather than reusing stale weights.

    Cached arrays come back as read-only memmaps: ``build_image`` only copies
    them into the image, so they never need to be resident.
    """
    if cache is None:
        return _run_transform(lowered, entry, weights)
    name = entry + "_transform_params"
    key = hashlib.sha256(lowered[name].script().encode()).hexdigest()[:16]
    directory = Path(cache) / f"{entry}-{key}"
    marker = directory / "COMPLETE"
    if marker.exists():
        count = int(marker.read_text())
        return [np.load(directory / f"{index}.npy", mmap_mode="r")
                for index in range(count)]
    transformed = _run_transform(lowered, entry, weights)
    directory.mkdir(parents=True, exist_ok=True)
    for index, array in enumerate(transformed):
        np.save(directory / f"{index}.npy", array)
    marker.write_text(str(len(transformed)))   # written last: a partial dump
    return transformed                         # must not look complete


def run_tuple_program(lowered, entry, inputs, weights, cached=None,
                      params_cache=None):
    """Link ``entry`` once and run it on the C-model; -> (outputs, words,
    program).  Passing the returned program back as ``cached`` reuses the
    linked words and plan with only the inputs changed -- the decode loop's
    whole point.
    """
    if cached is None:
        transformed = _transform(lowered, entry, weights, params_cache)
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
             runner=None, progress=None, params_cache=None):
    """Greedy generation: prompt ``token_ids`` -> ``steps`` new tokens.

    Links two programs once -- prefill_cache and decode at ``capacity``
    (default: just enough for the requested tokens) -- then loops, reusing
    the decode program with only its inputs changing.  ``runner`` defaults
    to the C-model; the llvm runner reruns the identical programs on CPU,
    which is how the tests cross-check each step.

    ``params_cache`` is a directory holding this checkpoint's transformed
    weights; see :func:`_transform`.
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
        runner, lowered, "prefill_cache", prefill_inputs, weights,
        params_cache=params_cache)
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
            runner, lowered, "decode", step_inputs, weights, cached=decode,
            params_cache=params_cache)
        k_cache[:, :, :, position] = k_new[:, :, :, 0]
        v_cache[:, :, position, :] = v_new[:, :, 0, :]
        generated.append(int(np.argmax(logits[-1].astype(np.float32))))
        if progress:
            progress("decode", position + 1, generated[-1], words)
    return generated


def _run3(runner, lowered, entry, inputs, weights, cached=None,
          params_cache=None):
    outputs, words, cached = runner(lowered, entry, inputs, weights,
                                    cached=cached, params_cache=params_cache)
    if len(outputs) != 3:
        raise RuntimeError(f"{entry}: expected 3 outputs, got {len(outputs)}")
    return tuple(outputs), words, cached


def _nd(value):
    import tvm

    return tvm.nd.array(value)


def llvm_runner(lowered, entry, inputs, weights, cached=None,
                params_cache=None):
    """The same programs on the llvm build -- the per-step cross-check.

    The transform runs inside this build, so the disk cache does not apply.
    """
    import tvm
    from tvm import relax

    if cached is None:
        vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
        cached = (vm, vm[entry + "_transform_params"]([weights]))
    vm, transformed = cached
    out = vm[entry](*[tvm.nd.array(inputs[name]) for name in inputs],
                    *transformed)
    return [t.numpy() for t in out], 0, cached
