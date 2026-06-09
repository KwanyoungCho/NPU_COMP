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


def run_module(mod, inputs, func_name="main", maxrun=None, tile=None):
    """Run a Relax module on mysim.

    inputs: dict {param_name -> np.ndarray}. Returns the output np.ndarray
    (float32, FP16-rounded by the NPU) reshaped to the output tensor shape.
    """
    func = mod[func_name]
    asm, mp = compile_func(func, tile=tile)

    gbuf = np.zeros(mp.top, dtype=np.float32)
    for c in mp.constants:                         # bake constant data into initial G-buffer
        data = mp.const_data[c].astype(np.float32).reshape(-1)
        off = mp.offset[c]
        gbuf[off:off + data.size] = data
    for p in mp.params:
        name = p.name_hint
        if name not in inputs:
            raise KeyError(f"missing input for param '{name}'")
        arr = np.asarray(inputs[name], dtype=np.float32).reshape(-1)
        off = mp.offset[p]
        if arr.size != _numel(mp.shape[p]):
            raise ValueError(f"param '{name}' size {arr.size} != {mp.shape[p]}")
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
