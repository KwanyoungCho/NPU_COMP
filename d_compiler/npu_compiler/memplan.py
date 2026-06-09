"""Static G-buffer memory planning.

The NPU has no dynamic allocation: every tensor lives at a fixed G-buffer offset
decided at compile time. For B0 we use a simple bump allocator (no reuse yet) over
a Relax function's params + dataflow-binding vars. Row-major contiguous layout.
"""
from tvm import relax


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

    def alloc_const(self, c):
        shape, dtype = _shape_dtype(c.struct_info)
        self.offset[c] = self.top
        self.shape[c] = shape
        self.dtype[c] = dtype
        self.top += _numel(shape)
        self.constants.append(c)
        self.const_data[c] = c.data.numpy()


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
            if isinstance(val, relax.Call):
                for arg in val.args:
                    if isinstance(arg, relax.Constant) and arg not in mp.offset:
                        mp.alloc_const(arg)
            if isinstance(val, relax.Var):       # alias binding (e.g. output gv = lv)
                mp.offset[binding.var] = mp.offset[val]
                mp.shape[binding.var] = mp.shape[val]
                mp.dtype[binding.var] = mp.dtype[val]
            else:
                mp.alloc(binding.var)
    out = seq.body
    assert isinstance(out, relax.Var), f"expected Var output, got {type(out)}"
    mp.output = out
    return mp
