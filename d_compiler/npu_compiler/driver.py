"""End-to-end driver: Relax IRModule -> NPU ISA -> run on given mysim -> output.

Ties together memplan + codegen + runtime. B0 (logical) path.
"""
import numpy as np
from . import memplan as _memplan
from . import codegen as _codegen
from . import runtime as _runtime


def compile_func(func, tile=None):
    """Relax function -> (Asm program, MemPlan). tile=64 enables B0.5 K-tiling."""
    mp = _memplan.plan(func)
    asm = _codegen.compile_func(func, mp, tile=tile)
    return asm, mp


def run_module(mod, inputs, func_name="main", maxrun=None, tile=None, backend="direct"):
    """Run a Relax module on mysim.

    inputs: dict {param_name -> np.ndarray}. Returns the output np.ndarray
    (float32, FP16-rounded by the NPU) reshaped to the output tensor shape.
    """
    if backend == "tir":                      # matmul-only modules via pure TIR path
        from . import tir_backend
        asm, mp = tir_backend.compile_func(mod, func_name)
        func = None
    elif backend == "hybrid":                 # whole graph: matmul->TIR, rest->direct
        func = mod[func_name]
        mp = _memplan.plan(func)
        asm = _codegen.compile_func(func, mp, tile=64, mm_backend="tir")
        func = None
    else:
        func = mod[func_name]
        asm, mp = compile_func(func, tile=tile)

    gbuf = np.zeros(mp.top, dtype=np.float32)
    for c in mp.constants:                         # bake constant data into initial G-buffer
        data = mp.const_data[c].astype(np.float32).reshape(-1)
        off = mp.offset[c]
        gbuf[off:off + data.size] = data
    # inputs: dict keyed by param name, OR list/tuple positional (matches param order)
    for i, p in enumerate(mp.params):
        if isinstance(inputs, dict):
            if p.name_hint not in inputs:
                raise KeyError(f"missing input for param '{p.name_hint}'")
            src = inputs[p.name_hint]
        else:
            src = inputs[i]
        arr = np.asarray(src, dtype=np.float32).reshape(-1)
        off = mp.offset[p]
        if arr.size != _numel(mp.shape[p]):
            raise ValueError(f"param #{i} '{p.name_hint}' size {arr.size} != {mp.shape[p]}")
        gbuf[off:off + arr.size] = arr

    out_var = mp.output
    out_off = mp.offset[out_var]
    out_shape = mp.shape[out_var]
    n_out = _numel(out_shape)

    full = _runtime.run(asm, gbuf, gn=mp.top, maxrun=maxrun)
    return full[out_off:out_off + n_out].reshape(out_shape)


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n
