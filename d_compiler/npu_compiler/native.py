"""Loader for the native (C++) v09 codegen.

The Python and native walkers consume :mod:`npu_compiler.v09_contract` and
its generated C++ peer. The native library is an accelerated implementation,
gated by tests that compare the two word streams exactly. An absent library
falls back to Python; a mismatched contract or internal compiler error fails
explicitly instead of silently using a different implementation.

Why it exists: emission is O(unrolled iterations) on a machine with no branch
or loop instruction, and in Python each iteration touches TVM objects whose
attribute reads and hashes are FFI calls.  TVM's own codegens (LLVM, CUDA, C)
are C++ visitors for the same reason.

Build it with ``npu_codegen/build.sh``; set ``NPU_CODEGEN_SO`` to use a
library from somewhere else.
"""
from __future__ import annotations

import os
from pathlib import Path

_LIBRARY = Path(__file__).resolve().parents[1] / "npu_codegen" / "libnpu_codegen.so"
_loaded = None


def library_path():
    return Path(os.environ.get("NPU_CODEGEN_SO", _LIBRARY))


def load():
    """Load the library once; -> True if the native entry points are usable."""
    global _loaded
    if _loaded is not None:
        return _loaded
    path = library_path()
    if not path.exists():
        _loaded = False
        return False
    import ctypes

    import tvm

    # RTLD_GLOBAL so the library shares TVM's registry rather than getting a
    # private copy of its symbols
    ctypes.CDLL(str(path), ctypes.RTLD_GLOBAL)
    from .v09_contract import CONTRACT_HASH
    signature = tvm.get_global_func("npu.codegen_contract_hash", allow_missing=True)
    if signature is None or str(signature()) != CONTRACT_HASH:
        raise RuntimeError(f"native codegen contract mismatch: rebuild {path} "
                           "with npu_codegen/build.sh")
    import importlib.util
    identity_path = _LIBRARY.parent / "build_identity.py"
    spec = importlib.util.spec_from_file_location("npu_build_identity", identity_path)
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    source = tvm.get_global_func("npu.codegen_source_hash", allow_missing=True)
    if source is None or str(source()) != identity.source_hash(_LIBRARY.parent):
        raise RuntimeError(f"native codegen source mismatch: rebuild {path} with npu_codegen/build.sh")
    _loaded = tvm.get_global_func("npu.native_available", allow_missing=True) \
        is not None
    return _loaded


def function(name):
    """A registered native entry point, or None when unavailable."""
    if not load():
        return None
    import tvm

    return tvm.get_global_func(name, allow_missing=True)


def encode_selftest():
    """The fixed encoder script the bit-exactness test compares against."""
    entry = function("npu.encode_selftest")
    if entry is None:
        raise RuntimeError(f"native codegen not built: {library_path()}")
    return entry().numpy()


def codegen_kernel(prim, addresses, sram, scratch_slots, constants,
                   one_fp32, diagnostics=None):
    """Emit one scheduled kernel natively; -> word array, or None.

    None means an absent library or missing native coverage. Proven invalid
    schedules raise LinkError without repeating the same check in Python. Internal
    compiler errors propagate; they must not be hidden by Python fallback.

    ``sram`` maps local buffers or explicitly retained parameters to nibble
    addresses. Retained bindings take precedence over global addresses.
    ``constants``
    maps a scalar literal to the nibble holding it.
    """
    entry = function("npu.codegen_kernel_status")
    if entry is None:
        if diagnostics is not None:
            diagnostics.update(status="unavailable", reason="native entry point is not loaded")
        return None
    from tvm import tir

    buffers, nibbles = zip(*sram.items()) if sram else ((), ())
    values, addrs = zip(*constants.items()) if constants else ((), ())
    result = entry(prim,
                     [int(address) for address in addresses],
                     list(buffers), [int(n) for n in nibbles],
                     [int(slot) for slot in scratch_slots],
                     [tir.FloatImm("float64", float(v)) for v in values],
                     [int(address) for address in addrs],
                     int(one_fp32 if one_fp32 is not None else -1))
    status = str(result[0])
    if status == "ok":
        if diagnostics is not None:
            diagnostics["status"] = "emitted"
        return result[1].numpy()
    if status == "unsupported":
        from .npu_compilation import current
        context = current()
        if context:
            reason = str(result[1])
            context.native_fallbacks[reason] = context.native_fallbacks.get(reason,0)+1
        if diagnostics is not None:
            diagnostics.update(status="fallback", reason=str(result[1]))
        return None
    if status == "invalid_schedule":
        from .npu_compilation import current
        from .npu_errors import LinkError
        context = current()
        reason = str(result[1])
        if context:
            context.native_rejections[reason] = context.native_rejections.get(reason,0)+1
        if diagnostics is not None:
            diagnostics.update(status="invalid_schedule", reason=reason)
        raise LinkError(reason)
    raise RuntimeError(f"invalid native codegen status: {status}")
