"""Autoregressive generation on the standard path.

A deployed model is exactly two compiled programs: ``prefill_cache`` at the
prompt length and ONE ``decode`` at a fixed cache capacity.  The machine has
no dynamic shapes, so decode always attends over ``capacity + 1`` slots and
an additive mask input carries the current length -- slots past it get the
fp16 floor and softmax zeroes them. The default runner retains a global memory
image (K: [L, kv, hd, C], V: [L, kv, C, hd]) and appends the new K/V rows with
a device DMA tail. The C-model still serializes this image on each invocation.
Host/LLVM reference runners update their cache arrays explicitly. No per-step
re-linking occurs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import npu_legalize, npu_link
from . import tvm_pipeline as pipeline


def _lower(mod, fusion="off", compiled=None, entries=None):
    if fusion not in ("auto", "off", "on"):
        raise ValueError("fusion must be auto, off, or on")
    if entries is not None:
        from .npu_compilation import reachable_module
        mod = reachable_module(mod, entries)
    if fusion == "auto":
        import json
        from tvm import relax
        from .npu_build import lower_and_select
        from .device import PROFILES
        roots = list(entries) if entries is not None else [
            gv.name_hint for gv, fn in mod.functions.items() if isinstance(fn, relax.Function)]
        lowered, programs, decisions = lower_and_select(mod, PROFILES["v09"], roots, fusion)
        if compiled is not None:
            compiled.update(programs)
        lowered = lowered.with_attr("npu.fusion_selection", json.dumps(decisions))
    else:
        lowered = pipeline.graph_pipeline(
            custom_legalize=npu_legalize.legalize_map(),
            fuse=fusion == "on", lift_params=True, fuse_attention=fusion != "off")(mod)
    return lowered


def _host_transform_module(lowered, entry):
    from .npu_compilation import reachable_module
    return reachable_module(lowered, [entry + "_transform_params"])


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

    host = _host_transform_module(lowered, entry)
    vm = relax.VirtualMachine(relax.build(host, "llvm"), tvm.cpu())
    return [t.numpy() for t in vm[entry + "_transform_params"]([weights])]


def _transform(lowered, entry, weights, cache=None):
    """The host-side transform, computed once and reused from disk.

    The transform does not change between runs of the same model, but running
    it means building the whole module for CPU and recomputing several GiB of
    weights every time.  ``cache`` is a directory the caller makes specific to
    the checkpoint; within it the entries are keyed by the transform
    reachable host IR, so changing a transform body or quantization mode misses
    the cache rather than reusing stale weights.

    Cached arrays come back as read-only memmaps: ``build_image`` only copies
    them into the image, so they never need to be resident.
    """
    if cache is None:
        return _run_transform(lowered, entry, weights)
    import tvm
    # Include reachable PrimFunc bodies and constant data, not only the Relax
    # wrapper's names/signature. A changed quantizer must miss the old cache.
    import json
    import os
    import fcntl
    digest = hashlib.sha256(tvm.ir.save_json(
        _host_transform_module(lowered, entry)).encode())
    # Content identity makes a cache directory safe across checkpoints. Hash
    # one tensor at a time so validation does not duplicate the full model.
    for weight in weights:
        array = np.ascontiguousarray(weight.numpy() if hasattr(weight, "numpy") else weight)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
    directory = Path(cache) / f"{entry}-{digest.hexdigest()}"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "COMPLETE"
    with (directory / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            manifest = json.loads(marker.read_text())
            arrays = []
            for index, spec in enumerate(manifest["arrays"]):
                array = np.load(directory / f"{index}.npy", mmap_mode="r", allow_pickle=False)
                if (list(array.shape) != spec["shape"] or array.dtype.str != spec["dtype"]
                        or hashlib.sha256(memoryview(array).cast("B")).hexdigest() != spec["sha256"]):
                    raise ValueError(f"parameter cache is corrupt: {directory}/{index}.npy")
                arrays.append(array)
            return arrays
        transformed = _run_transform(lowered, entry, weights)
        specs = []
        for index, array in enumerate(transformed):
            array = np.ascontiguousarray(array)
            path = directory / f"{index}.npy"
            temporary = directory / f"{index}.tmp"
            with temporary.open("wb") as output:
                np.save(output, array, allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            specs.append(dict(shape=list(array.shape), dtype=array.dtype.str,
                              sha256=hashlib.sha256(memoryview(array).cast("B")).hexdigest()))
        temporary = directory / "COMPLETE.tmp"
        temporary.write_text(json.dumps(dict(version=2, arrays=specs)))
        os.replace(temporary, marker)
        return transformed


def run_tuple_program(lowered, entry, inputs, weights, cached=None,
                      params_cache=None, compiled=None):
    """Link ``entry`` once and run it on the C-model; -> (outputs, words,
    program).  Passing the returned program back as ``cached`` reuses the
    linked words and plan with only the inputs changed -- the decode loop's
    whole point.
    """
    if cached is None:
        transformed = _transform(lowered, entry, weights, params_cache)
        asm, plan = compiled or npu_link.compile_program(lowered, entry)
        func = plan.func
        cached = (transformed, asm, plan, func)
    transformed, asm, plan, func = cached
    values = _bind_inputs(func, inputs, transformed)

    image = npu_link.build_image(plan, func, values)
    if image.size % 2:
        image = np.concatenate([image, np.zeros(1, dtype="<f2")])
    from .v09_runtime import run as run_v09

    images, counters = run_v09(asm.words, image.view("<u4"))
    from .npu_abi import read_outputs
    result = read_outputs(images[-1],plan.outputs)
    outputs = list(result) if isinstance(result,tuple) else [result]
    return outputs, len(asm.words), cached


def _bind_inputs(func, inputs, transformed):
    runtime_count = len(func.params) - len(transformed)
    names = [p.name_hint for p in func.params[:runtime_count]]
    if set(inputs) != set(names):
        raise ValueError(f"runtime inputs must be {names}, got {list(inputs)}")
    return {**inputs, **dict(zip([p.name_hint for p in func.params[runtime_count:]], transformed))}


def length_mask(capacity, length, dtype=np.float16):
    """Additive [1, 1, capacity+1]: zero over the ``length`` valid slots and
    over this token's own slot (the last), the fp16 floor elsewhere."""
    if capacity <= 0 or not 0 <= length <= capacity:
        raise ValueError("invalid cache capacity/valid length")
    mask = np.full((1, 1, capacity + 1), -65504.0, dtype=np.float32)
    mask[0, 0, :length] = 0.0
    mask[0, 0, capacity] = 0.0
    return mask.astype(dtype)


def generate(family, config, assets, token_ids, steps, capacity=None,
             runner=None, progress=None, params_cache=None, fusion="auto",
             cache_buckets=None, diagnostics=None):
    """Greedy generation: prompt ``token_ids`` -> ``steps`` new tokens.

    Links two programs once -- prefill_cache and decode at ``capacity``
    (default: just enough for the requested tokens) -- then loops, reusing
    the decode program with only its inputs changing.  ``runner`` defaults
    to the C-model; the llvm runner reruns the identical programs on CPU,
    which is how the tests cross-check each step.

    ``cache_buckets`` optionally supplies increasing capacities ending at
    ``capacity``. All required programs compile before execution; switching
    buckets copies the valid cache prefix once. ``diagnostics`` receives
    program costs, transitions and transport bytes. The default remains a
    single fixed-capacity program.
    """
    runner = runner or run_stateful_tuple_program
    stateful = runner is run_stateful_tuple_program
    seq = len(token_ids)
    if seq == 0 or steps <= 0:
        raise ValueError("generation requires a nonempty prompt and positive steps")
    tokens = [int(t) for t in token_ids]
    if cache_buckets is not None:
        buckets = list(cache_buckets)
        if (not buckets or any(type(c) is not int or c <= 0 for c in buckets)
                or buckets != sorted(set(buckets))):
            raise ValueError("cache_buckets must be strictly increasing positive integers")
        capacity = buckets[-1] if capacity is None else capacity
        if buckets[-1] != capacity:
            raise ValueError("the last cache bucket must equal capacity")
    else:
        buckets = None
    capacity = seq + steps - 1 if capacity is None else capacity
    if capacity < seq + steps - 1:
        raise ValueError(f"capacity {capacity} cannot hold {seq}+{steps - 1}")

    def bucket(position):
        return next(c for c in buckets if c > position) if buckets else capacity
    used = sorted({bucket(seq + step - 1) for step in range(1, steps)}) or [capacity]
    from .npu_compilation import compilation
    modules, programs = {}, {}
    with compilation():
        for index, cap in enumerate(used):
            mod, params, cfg = family.build_generate(config, seq, cap, last_logits=True)
            signature = [(name, tuple(p.shape), str(p.dtype)) for name,p in params]
            if index == 0:
                weights = [_nd(value) for value in family.load_params(assets, params, cfg)]
                first_signature = signature
            elif signature != first_signature:
                raise ValueError("cache buckets changed the model's parameter ABI")
            programs[cap] = {}
            entries = ("prefill_cache", "decode") if index == 0 else ("decode",)
            modules[cap] = _lower(mod, fusion=fusion, compiled=programs[cap], entries=entries)
    info = diagnostics if diagnostics is not None else {}
    info.update(capacities=used, compiled_decode_programs=len(used), transitions=[], steps=[])
    def execute(cap, entry, inputs, cached=None):
        kwargs = dict(cached=cached, params_cache=params_cache)
        if runner in (run_stateful_tuple_program, run_tuple_program):
            kwargs["compiled"] = programs[cap].get(entry)
        outputs, words, handle = runner(modules[cap], entry, inputs, weights, **kwargs)
        if len(outputs) != 3:
            raise RuntimeError(f"{entry}: expected 3 outputs, got {len(outputs)}")
        record = dict(entry=entry, capacity=cap, words=words)
        if isinstance(handle, DecodeSession):
            record.update(handle.history[-1])
        info["steps"].append(record)
        return outputs, words, handle

    inputs = family.runtime_inputs(assets, cfg, tokens)
    cos, sin = family.rope_inputs(cfg, np.arange(seq))
    prefill_inputs = {"input_embeds": inputs["input_embeds"],
                      "cos": cos, "sin": sin, "mask": inputs["mask"]}
    (logits, k_rows, v_rows), words, _ = execute(used[0], "prefill_cache", prefill_inputs)
    generated = [int(np.argmax(logits[-1].astype(np.float32)))]
    if progress:
        progress("prefill", seq, generated[-1], words)

    layers, kv, hd = k_rows.shape[0], k_rows.shape[1], k_rows.shape[2]
    k_cache = np.zeros((layers, kv, hd, capacity), dtype=np.float16)
    v_cache = np.zeros((layers, kv, capacity, hd), dtype=np.float16)
    k_cache[:, :, :, :seq] = k_rows
    v_cache[:, :, :seq, :] = v_rows

    decode, active = None, used[0]
    try:
        for step in range(1, steps):
            position = seq + step - 1
            cap = bucket(position)
            if cap != active:
                if isinstance(decode, DecodeSession):
                    decode.close()
                decode = None
                info["transitions"].append(dict(position=position, old_capacity=active,
                    capacity=cap, valid_cache_bytes=2*layers*kv*hd*position*2,
                    input_cache_bytes=2*layers*kv*hd*cap*2))
                active = cap
            cos, sin = family.rope_inputs(cfg, [position])
            step_inputs = {"input_embeds": assets.embedding([generated[-1]]).astype(np.float16),
                "cos": cos, "sin": sin, "mask": length_mask(cap, position)}
            if not stateful or decode is None:
                step_inputs.update(k_cache=np.ascontiguousarray(k_cache[:,:,:,:cap]),
                                   v_cache=np.ascontiguousarray(v_cache[:,:,:cap,:]))
            (logits, k_new, v_new), words, decode = execute(cap, "decode", step_inputs, decode)
            if not stateful or buckets:
                k_cache[:, :, :, position] = k_new[:, :, :, 0]
                v_cache[:, :, position, :] = v_new[:, :, 0, :]
            generated.append(int(np.argmax(logits[-1].astype(np.float32))))
            if progress:
                progress("decode", position + 1, generated[-1], words)
    finally:
        if isinstance(decode, DecodeSession):
            decode.close()
    return generated


class DecodeSession:
    """One linked decode, immutable transformed weights, persistent global image.

    SRAM starts fresh in the functional C-model on every call. Its file protocol
    still serializes the full global image; that I/O is not incremental DMA.
    The device program's append traffic and host input patch bytes are recorded
    separately from that simulator transport cost.
    """
    def __init__(self, lowered, entry, inputs, weights, params_cache=None, compiled=None):
        from .npu_cache import AppendPlan
        transformed = _transform(lowered, entry, weights, params_cache)
        self.asm, self.plan = compiled or npu_link.compile_program(lowered, entry, residency=True)
        self.func = self.plan.func
        self.append = getattr(self.plan, "append_plan", None) or AppendPlan(self.func, self.plan)
        self.cache_names = {s.parameter for s in self.append.slices}
        self.runtime_names = set(inputs)
        values = dict(inputs)
        missing = [p.name_hint for p in self.func.params if p.name_hint not in values]
        if len(missing) != len(transformed):
            raise ValueError("decode initial inputs do not match the linked signature")
        values.update(zip(missing, transformed))
        self.image = npu_link.build_image(self.plan, self.func, values)
        from .v09_runtime import PersistentTransport
        self.transport = PersistentTransport(self.image)
        self.next_position = self.append.position(inputs[self.append.mask])
        self.history = []

    def run(self, inputs):
        position = self.append.position(inputs[self.append.mask])
        if position != self.next_position:
            raise ValueError(f"cache append must be sequential: expected {self.next_position}, got {position}")
        allowed = {p.name_hint: p.struct_info for p in self.func.params
                   if p.name_hint in self.runtime_names}
        patches = []
        for name, array in inputs.items():
            if name in self.cache_names:
                if self.history:
                    raise ValueError("persistent cache inputs must be omitted after initialization")
                continue
            if name not in allowed:
                raise ValueError(f"unknown decode input {name}")
            info = allowed[name]
            array = np.asarray(array)
            if tuple(array.shape) != tuple(map(int, info.shape)) or str(array.dtype) != str(info.dtype):
                raise ValueError(f"decode input shape/dtype mismatch: {name}")
            patches.append((self.plan.address[name] * 2,
                            np.ascontiguousarray(array).view(np.uint8).reshape(-1)))
        # Do not permit silently reusing a previous token's mask/embedding/RoPE.
        required = self.runtime_names - self.cache_names
        if not required <= inputs.keys():
            raise ValueError(f"missing decode inputs: {sorted(required - inputs.keys())}")
        tail = self.append.words(position)
        words = self.asm.words[:-1] + tail + [0xff]
        final, counters = self.transport.run(words, patches)
        self.image = final.view("<f2")
        self.next_position += 1
        from .npu_abi import read_outputs
        outputs = list(read_outputs(self.image,self.plan.outputs))
        from .v09_stats import analyze
        self.history.append(dict(position=position, input_patch_bytes=sum(len(a) for _, a in patches),
                                 cache_update_bytes=self.append.logical_update_bytes,
                                 append=analyze(tail), counters=counters,
                                 transport=dict(self.transport.last_transfer)))
        return outputs, len(words), self

    def close(self):
        self.transport.close()


def run_stateful_tuple_program(lowered, entry, inputs, weights, cached=None, params_cache=None,
                              compiled=None):
    from .npu_cache import ATTRIBUTE
    if not lowered[entry].attrs or ATTRIBUTE not in lowered[entry].attrs:
        return run_tuple_program(lowered, entry, inputs, weights, cached, params_cache, compiled)
    session = cached or DecodeSession(lowered, entry, inputs, weights, params_cache, compiled)
    return session.run(inputs)


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
        from .npu_numeric import tile_order_reference
        vm = relax.VirtualMachine(relax.build(tile_order_reference(lowered), "llvm"), tvm.cpu())
        cached = (vm, vm[entry + "_transform_params"]([weights]))
    vm, transformed = cached
    values = _bind_inputs(lowered[entry], inputs, transformed)
    runtime_count = len(lowered[entry].params) - len(transformed)
    out = vm[entry](*[tvm.nd.array(values[p.name_hint]) for p in lowered[entry].params[:runtime_count]],
                    *transformed)
    return [t.numpy() for t in out], 0, cached
