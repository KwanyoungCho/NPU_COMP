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


def _ceil(x, T=64):
    return (x + T - 1) // T * T


# ---- tile-blocked layout (A4): [R,N] logical <-> [Rt,Nt,T,T] physical, zero-padded ----
# Canonical convention shared by memplan alloc, codegen TILE-mode matmul, and runtime.
def pack_tiled(arr, T=64):
    """[R,N] row-major -> flat tile-blocked [ceil(R/T),ceil(N/T),T,T] (pad with 0)."""
    R, N = arr.shape
    Rt, Nt = _ceil(R, T) // T, _ceil(N, T) // T
    padded = np.zeros((Rt * T, Nt * T), dtype=arr.dtype)
    padded[:R, :N] = arr
    return padded.reshape(Rt, T, Nt, T).transpose(0, 2, 1, 3).reshape(-1)


def unpack_tiled(flat, R, N, T=64):
    """flat tile-blocked -> [R,N] row-major (drops padding). Inverse of pack_tiled."""
    Rt, Nt = _ceil(R, T) // T, _ceil(N, T) // T
    blk = np.asarray(flat).reshape(Rt, Nt, T, T).transpose(0, 2, 1, 3).reshape(Rt * T, Nt * T)
    return blk[:R, :N]


def tiled_numel(shape, T=64):
    """Physical element count of a [R,N] tensor stored tile-blocked (padded to T)."""
    R, N = shape
    return (_ceil(R, T) // T) * (_ceil(N, T) // T) * T * T


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
        self.packed_meta = {} # Constant -> Nt (tile-blocked weight: stored [Kt,Nt,64,64])
        self.layout = {}      # Var|Constant -> 'row' (default) | 'tile' (A4 tile-blocked)

    def alloc(self, var):
        shape, dtype = _shape_dtype(var.struct_info)
        off = self.top
        self.offset[var] = off
        self.shape[var] = shape
        self.dtype[var] = dtype
        self.layout[var] = "row"
        self.top += _numel(shape)
        return off

    def alloc_tiled(self, var):
        """Allocate a binding var in tile-blocked layout (A4). Logical shape is kept;
        physical footprint is padded-to-64 tiles. codegen TILE-mode matmul reads/writes
        each 64x64 tile contiguously (no gather/scatter)."""
        shape, dtype = _shape_dtype(var.struct_info)
        off = self.top
        self.offset[var] = off
        self.shape[var] = shape
        self.dtype[var] = dtype
        self.layout[var] = "tile"
        self.top += tiled_numel(shape)
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
        self.layout[c] = "row"
        self.top += _numel(shape)
        self.constants.append(c)
        self.const_data[c] = data

    def alloc_const_packed(self, c, T=64):
        """Weight pre-packing: store a matmul weight constant [K,N] in tile-blocked
        layout [Kt,Nt,T,T] (flattened) so the matmul backend reads each TxT tile
        contiguously (stride T -> NO gather). Logical shape [K,N] is preserved;
        only the data byte-order is reordered (host-side, offline). Requires 64-mult."""
        shape, dtype = _shape_dtype(c.struct_info)
        K, N = shape
        data = np.asarray(c.data.numpy())                      # [K,N] row-major
        Kt, Nt = K // T, N // T
        packed = data.reshape(Kt, T, Nt, T).transpose(0, 2, 1, 3).reshape(-1)  # [Kt,Nt,T,T]
        self.offset[c] = self.top
        self.shape[c] = shape                                  # logical shape unchanged
        self.dtype[c] = dtype
        self.layout[c] = "tile"
        self.top += K * N
        self.constants.append(c)
        self.const_data[c] = packed
        self.packed_meta[c] = Nt


def _packable(c):
    shp = [int(d) for d in c.struct_info.shape]
    return len(shp) == 2 and shp[0] % 64 == 0 and shp[1] % 64 == 0


def plan(func, pack=True, pack_params=False):
    """Plan a Relax function: assign G-buffer offsets to params, constants, and
    every binding var. Returns a MemPlan. Assumes one dataflow block returning a Var.

    pack=True        : pre-pack matmul weight CONSTANTS (tile-blocked) — automatic.
    pack_params=True : ALSO mark matmul weight PARAMS (name starts 'W', static model
                       weights) as packed. The param's byte-size is unchanged (same
                       K*N); only packed_meta is recorded so codegen reads it packed
                       (no gather) and run_compiled packs the fed data. Cache params
                       (Kt*/Vc*, filled at runtime) are NOT weights -> not packed."""
    mp = MemPlan()
    param_set = set(func.params)
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
                opname = getattr(val.op, "name", "")
                is_ew = opname in _EW_OPS
                bsh = [int(d) for d in binding.var.struct_info.shape] if is_ew else None
                for idx, arg in enumerate(val.args):
                    if isinstance(arg, relax.Constant) and arg not in mp.offset:
                        if pack and opname == "relax.matmul" and idx == 1 and _packable(arg):
                            mp.alloc_const_packed(arg)        # weight pre-packing (tile-blocked)
                        else:
                            mp.alloc_const(arg, broadcast_to=bsh)
                    elif (pack_params and opname == "relax.matmul" and idx == 1
                          and arg in param_set and arg not in mp.packed_meta
                          and arg.name_hint.startswith("W") and _packable(arg)):
                        mp.packed_meta[arg] = int(arg.struct_info.shape[1]) // 64   # static weight param
            mp.alloc(binding.var)
    out = seq.body
    while out in mp.tuple_of:                         # unwrap 1-tuple output
        out = mp.tuple_of[out][0]
    assert isinstance(out, relax.Var), f"expected Var output, got {type(out)}"
    mp.output = out
    return mp
