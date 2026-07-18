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


# ---- A4 5c: layout assignment (tile-blocked activations for matmul chains) ----
_EW_LAYOUT = {"relax.add", "relax.subtract", "relax.multiply", "relax.divide",
              "relax.nn.silu", "relax.exp", "relax.sqrt", "relax.negative",
              "relax.cos", "relax.sin"}


def _opn(c):
    return getattr(c.op, "name", "") if isinstance(c, relax.Call) else ""


def _arg_vars(val):
    out = []
    if isinstance(val, relax.Call):
        for a in val.args:
            if isinstance(a, relax.Var):
                out.append(a)
            elif isinstance(a, relax.Tuple):
                out += [f for f in a.fields if isinstance(f, relax.Var)]
    elif isinstance(val, relax.Tuple):
        out += [f for f in val.fields if isinstance(f, relax.Var)]
    elif isinstance(val, relax.Var):
        out.append(val)
    return out


def assign_layouts(bindings, output_var):
    """Decide 'row'/'tile' per binding var (fixpoint). A matmul output (64-mult M,K,N)
    or a layout-transparent elementwise output is 'tile' iff EVERY consumer is
    TILE-compatible (matmul-A, or a 'tile' elementwise) and (for elementwise) every
    tensor input is 'tile'; else 'row'. Consistency: a var feeding a row op is demoted
    to row, so tile vars only feed tile consumers -> no explicit relayout needed
    (a matmul reads a row A via its own gather; a tile A skips the gather)."""
    val = {b.var: b.value for b in bindings}
    consumers = {}
    mm_a = {}                                          # v -> {matmul binding vars reading v as A}
    for b in bindings:
        for v in _arg_vars(b.value):
            consumers.setdefault(v, []).append(b.var)
        c = b.value
        if isinstance(c, relax.Call) and _opn(c) == "relax.matmul" and isinstance(c.args[0], relax.Var):
            mm_a.setdefault(c.args[0], set()).add(b.var)  # Var hash-eq keys (identity `is` is unreliable)

    def is_mm64(v):
        c = val.get(v)
        if not (isinstance(c, relax.Call) and _opn(c) == "relax.matmul"):
            return False
        M, K = [int(d) for d in c.args[0].struct_info.shape]
        _, N = [int(d) for d in c.args[1].struct_info.shape]
        return M % 64 == 0 and K % 64 == 0 and N % 64 == 0

    def is_ew(v):
        c = val.get(v)
        return isinstance(c, relax.Call) and _opn(c) in _EW_LAYOUT

    layout = {b.var: ("tile" if (is_mm64(b.var) or is_ew(b.var)) else "row") for b in bindings}
    layout[output_var] = "row"
    changed = True
    while changed:
        changed = False
        for b in bindings:
            v = b.var
            if layout.get(v) != "tile":
                continue
            demote = (v is output_var)
            if not demote and is_ew(v):                       # TILE ew: all inputs TILE
                for a in val[v].args:
                    if isinstance(a, (relax.Var, relax.Constant)) and layout.get(a, "row") != "tile":
                        demote = True; break
                    if isinstance(a, relax.Tuple) and any(layout.get(f, "row") != "tile" for f in a.fields):
                        demote = True; break
            if not demote:
                v_mm_a = mm_a.get(v, set())
                for cv in consumers.get(v, []):               # all consumers TILE-compatible
                    if cv in v_mm_a and is_mm64(cv):
                        continue                              # read as A by a 64-mult matmul
                    if is_ew(cv) and layout.get(cv) == "tile":
                        continue                              # feeds a TILE ew
                    demote = True; break
            if demote:
                layout[v] = "row"; changed = True
    return layout


def plan(func, pack=True, pack_params=False, layouts=True):
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
    bindings = [b for block in seq.blocks for b in block.bindings]

    # resolve output var (unwrap 1-tuple) then assign layouts (A4 5c)
    for b in bindings:
        if isinstance(b.value, relax.Tuple):
            mp.tuple_of[b.var] = list(b.value.fields)
    out = seq.body
    while out in mp.tuple_of:
        out = mp.tuple_of[out][0]
    assert isinstance(out, relax.Var), f"expected Var output, got {type(out)}"
    layout = assign_layouts(bindings, out) if layouts else {}

    for binding in bindings:
        val = binding.value
        if isinstance(val, relax.Tuple):             # tuple_of already recorded
            continue
        if isinstance(val, relax.Var):               # alias binding (e.g. gv = lv)
            mp.offset[binding.var] = mp.offset[val]
            mp.shape[binding.var] = mp.shape[val]
            mp.dtype[binding.var] = mp.dtype[val]
            mp.layout[binding.var] = mp.layout.get(val, "row")
            continue
        if isinstance(val, relax.Call):
            # elementwise ops may broadcast a smaller constant operand (e.g. bias)
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
                    mp.packed_meta[arg] = int(arg.struct_info.shape[1]) // 64
        if layout.get(binding.var) == "tile":
            mp.alloc_tiled(binding.var)
        else:
            mp.alloc(binding.var)
    mp.output = out
    return mp
