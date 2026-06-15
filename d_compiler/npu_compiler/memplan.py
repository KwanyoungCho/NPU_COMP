"""Static G-buffer memory planning.

The NPU has no dynamic allocation: every tensor lives at a fixed G-buffer offset
decided at compile time. For B0 we use a simple bump allocator (no reuse yet) over
a Relax function's params + dataflow-binding vars. Row-major contiguous layout.
"""
import numpy as np
from tvm import relax

_EW_OPS = {"relax.add", "relax.subtract", "relax.multiply", "relax.divide"}


def _shape_dtype(sinfo):
    assert isinstance(sinfo, relax.TensorStructInfo), f"expected tensor, got {type(sinfo)}"
    shape = [int(d) for d in sinfo.shape]
    return shape, sinfo.dtype


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


class MemPlan:
    def __init__(self):
        self.offset = {}      # Var|Constant -> int offset (FP16 element units)
        self.shape = {}       # Var|Constant -> list[int]
        self.dtype = {}       # Var|Constant -> str
        self.top = 0
        self.params = []      # ordered list of param Vars
        self.constants = []   # relax.Constant nodes (data baked into initial G-buffer)
        self.const_data = {}  # Constant -> numpy array
        self.tuple_of = {}    # tuple-typed Var -> [field vars] (torch import outputs)
        self.output = None    # returned Var

    def alloc(self, var):
        shape, dtype = _shape_dtype(var.struct_info)
        off = self.top
        self.offset[var] = off
        self.shape[var] = shape
        self.dtype[var] = dtype
        self.top += _numel(shape)
        return off

    def scratch_alloc(self, n):
        """Allocate codegen-internal scratch (e.g. tiling gather/partial buffers)."""
        off = self.top
        self.top += n
        return off

    def alloc_const(self, c, broadcast_to=None):
        shape, dtype = _shape_dtype(c.struct_info)
        data = c.data.numpy()
        if broadcast_to is not None and list(shape) != list(broadcast_to):
            data = np.broadcast_to(data, broadcast_to).copy()   # host-expand (e.g. bias)
            shape = list(broadcast_to)
        self.offset[c] = self.top
        self.shape[c] = shape
        self.dtype[c] = dtype
        self.top += _numel(shape)
        self.constants.append(c)
        self.const_data[c] = data


def plan(func):
    """Plan a Relax function: assign G-buffer offsets to params, constants, and
    every binding var. Returns a MemPlan. Assumes one dataflow block returning a Var."""
    mp = MemPlan()
    for p in func.params:
        mp.params.append(p)
        mp.alloc(p)
    seq = func.body
    assert isinstance(seq, relax.SeqExpr), "expected SeqExpr body"
    for block in seq.blocks:
        for binding in block.bindings:
            val = binding.value
            if isinstance(val, relax.Tuple):         # e.g. output (lv,) from torch import
                mp.tuple_of[binding.var] = list(val.fields)
                continue
            if isinstance(val, relax.Var):           # alias binding (e.g. gv = lv)
                mp.offset[binding.var] = mp.offset[val]
                mp.shape[binding.var] = mp.shape[val]
                mp.dtype[binding.var] = mp.dtype[val]
                continue
            if isinstance(val, relax.Call):
                # elementwise ops may broadcast a smaller constant operand (e.g. bias)
                # -> host-expand that constant to the output shape.
                is_ew = getattr(val.op, "name", "") in _EW_OPS
                bsh = [int(d) for d in binding.var.struct_info.shape] if is_ew else None
                for arg in val.args:
                    if isinstance(arg, relax.Constant) and arg not in mp.offset:
                        mp.alloc_const(arg, broadcast_to=bsh)
            mp.alloc(binding.var)
    out = seq.body
    while out in mp.tuple_of:                         # unwrap 1-tuple output
        out = mp.tuple_of[out][0]
    assert isinstance(out, relax.Var), f"expected Var output, got {type(out)}"
    mp.output = out
    return mp
