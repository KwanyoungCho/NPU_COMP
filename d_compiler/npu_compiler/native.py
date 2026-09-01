"""Loader for the native (C++) v09 codegen.

The Python walker in :mod:`npu_compiler.tir_codegen_v09` is the definition of
what a scheduled kernel lowers to, and it stays that way: the native library
is an accelerated second implementation, gated by tests that compare the two
word streams exactly.  Nothing here is required for the compiler to work --
if the library is absent, every caller falls back to Python.

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
                   one_fp32):
    """Emit one scheduled kernel natively; -> word array, or None.

    None means "this kernel is not covered" -- the native walker raises on
    anything it does not recognise, and the caller then runs the Python
    walker, which is the definition of the lowering.  So a gap costs speed,
    never correctness.

    ``sram`` maps a kernel-local buffer to its nibble address, ``constants``
    maps a scalar literal to the nibble holding it.
    """
    entry = function("npu.codegen_kernel")
    if entry is None:
        return None
    import tvm
    from tvm import tir

    buffers, nibbles = zip(*sram.items()) if sram else ((), ())
    values, addrs = zip(*constants.items()) if constants else ((), ())
    try:
        return entry(prim,
                     [int(address) for address in addresses],
                     list(buffers), [int(n) for n in nibbles],
                     [int(slot) for slot in scratch_slots],
                     [tir.FloatImm("float64", float(v)) for v in values],
                     [int(address) for address in addrs],
                     int(one_fp32 if one_fp32 is not None else -1)).numpy()
    except tvm.error.TVMError:
        return None                     # not covered; the caller falls back
