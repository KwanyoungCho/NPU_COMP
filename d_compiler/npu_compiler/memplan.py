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
        self.tiled_inputs = set()  # param Vars fed pre-packed (tile-blocked activation input, A4 5d)

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

    def alloc_const_tiled(self, c, broadcast_to=None, T=64):
        """Store an elementwise-operand constant [R,C] in tile-blocked layout so a
        TILE elementwise op reads it in the same tile order as its other operands
        (A4 5d). Host-side reorder (offline) — no device cost. Requires 64-mult."""
        shape, dtype = _shape_dtype(c.struct_info)
        data = c.data.numpy()
        if broadcast_to is not None and list(shape) != list(broadcast_to):
            data = np.broadcast_to(data, broadcast_to).copy()
            shape = list(broadcast_to)
        self.offset[c] = self.top
        self.shape[c] = shape
        self.dtype[c] = dtype
        self.layout[c] = "tile"
        self.top += tiled_numel(shape)
        self.constants.append(c)
        self.const_data[c] = pack_tiled(np.asarray(data), T)

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


def assign_layouts(bindings, output_var, tile_inputs=False):
    """Decide 'row'/'tile' per binding var (fixpoint). A matmul output (64-mult M,K,N)
    or a layout-transparent elementwise output is 'tile' iff EVERY consumer is
    TILE-compatible (matmul-A, or a 'tile' elementwise) and (for elementwise) every
    tensor input is 'tile'; else 'row'. Consistency: a var feeding a row op is demoted
    to row, so tile vars only feed tile consumers -> no explicit relayout needed
    (a matmul reads a row A via its own gather; a tile A skips the gather)."""
    val = {b.var: b.value for b in bindings}
    consumers = {}
    mm_a = {}                                          # v -> {matmul binding vars reading v as A}
    mm_b = {}                                          # v -> {matmul binding vars reading v as B}
    for b in bindings:
        for v in _arg_vars(b.value):
            consumers.setdefault(v, []).append(b.var)
        c = b.value
        if isinstance(c, relax.Call) and _opn(c) == "relax.matmul":
            if isinstance(c.args[0], relax.Var):
                mm_a.setdefault(c.args[0], set()).add(b.var)  # Var hash-eq keys (identity `is` unreliable)
            if isinstance(c.args[1], relax.Var):
                mm_b.setdefault(c.args[1], set()).add(b.var)  # tile activation B (A4 5d-2: Kt/V)

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

    def is_bcast(v):                                    # broadcast_to producing a 64-mult 2D dst
        c = val.get(v)
        if not (isinstance(c, relax.Call) and _opn(c) == "relax.broadcast_to"):
            return False
        sh = [int(d) for d in v.struct_info.shape]
        return len(sh) == 2 and sh[0] % 64 == 0 and sh[1] % 64 == 0

    def is_reduce(v):                                  # sum reading a 64-mult 2D src (tile-native)
        c = val.get(v)
        if not (isinstance(c, relax.Call) and _opn(c) == "relax.sum"):
            return False
        sh = [int(d) for d in c.args[0].struct_info.shape]
        return len(sh) == 2 and sh[0] % 64 == 0 and sh[1] % 64 == 0

    def is_concat(v):                                  # concat relayouts its tile inputs itself
        c = val.get(v)
        return isinstance(c, relax.Call) and _opn(c) == "relax.concat"

    def _is64_2d(v):
        sh = [int(d) for d in v.struct_info.shape]
        return len(sh) == 2 and sh[0] % 64 == 0 and sh[1] % 64 == 0

    # A4 5d-2 attention: layout-transforming producers that need a TILE input.
    def is_transpose(v):                               # permute_dims [R,C]->[C,R]
        c = val.get(v)
        return isinstance(c, relax.Call) and _opn(c) == "relax.permute_dims" and _is64_2d(v)

    def is_tslice(v):                                  # strided_slice, tile-COLUMN-aligned last axis
        c = val.get(v)
        if not (isinstance(c, relax.Call) and _opn(c) == "relax.strided_slice" and _is64_2d(v)):
            return False
        axes = [int(f.value) for f in c.args[1]]
        if axes[0] not in (1, -1):
            return False
        src_c = int(c.args[0].struct_info.shape[1])
        beg = int(c.args[2][0].value)
        b0 = beg + (src_c if beg < 0 else 0)
        return b0 % 64 == 0

    def is_concat64(v):                                # concat with a 64-mult 2D output (tile if all in tile)
        return is_concat(v) and _is64_2d(v)

    def is_max(v):                                     # row-max reduce reading a 64-mult 2D src
        c = val.get(v)
        if not (isinstance(c, relax.Call) and _opn(c) == "relax.max"):
            return False
        sh = [int(d) for d in c.args[0].struct_info.shape]
        return len(sh) == 2 and sh[0] % 64 == 0 and sh[1] % 64 == 0

    # A4 5d: activation INPUT params (residual stream x) can be tile — but ONLY when the
    # caller pre-packs the fed data host-side (tile_inputs, set on the real generation
    # path). Otherwise the input stays row so the direct oracle (which doesn't honor tile
    # layout) and tir agree byte-exact. Candidates: 2D 64-mult Vars read as matmul-A or an
    # elementwise/reduce operand, NOT defined by a binding (real params) nor a weight.
    defined = set(val.keys())
    tile_params = set()
    if tile_inputs:
        for b in bindings:
            c = b.value
            if isinstance(c, relax.Call):
                nm = _opn(c)
                if nm == "relax.matmul" and isinstance(c.args[0], relax.Var):
                    tile_params.add(c.args[0])
                elif nm in _EW_LAYOUT or nm == "relax.sum":
                    for a in c.args:
                        if isinstance(a, relax.Var):
                            tile_params.add(a)
        tile_params = {p for p in tile_params if p not in defined
                       and len(p.struct_info.shape) == 2
                       and int(p.struct_info.shape[0]) % 64 == 0 and int(p.struct_info.shape[1]) % 64 == 0}

    layout = {b.var: ("tile" if (is_mm64(b.var) or is_ew(b.var) or is_bcast(b.var)
                                 or is_transpose(b.var) or is_tslice(b.var) or is_concat64(b.var))
                      else "row") for b in bindings}
    for p in tile_params:
        layout[p] = "tile"
    layout[output_var] = "row"
    nodes = [b.var for b in bindings] + list(tile_params)
    changed = True
    while changed:
        changed = False
        for v in nodes:
            if layout.get(v) != "tile":
                continue
            demote = (v is output_var)
            if not demote and is_ew(v):                       # TILE ew: all inputs TILE (or packable const)
                for a in val[v].args:
                    if isinstance(a, relax.Constant):
                        csh = [int(d) for d in a.struct_info.shape]
                        n = 1
                        for d in csh:
                            n *= d
                        if n == 1:                            # scalar: layout-agnostic
                            continue
                        if len(csh) == 2 and csh[0] % 64 == 0 and csh[1] % 64 == 0:
                            continue                          # 2D 64-mult: packed tile (host-side)
                        demote = True; break
                    if isinstance(a, relax.Var) and layout.get(a, "row") != "tile":
                        demote = True; break
                    if isinstance(a, relax.Tuple) and any(layout.get(f, "row") != "tile" for f in a.fields):
                        demote = True; break
            if not demote and (is_transpose(v) or is_tslice(v)):  # producer needs a TILE input
                if layout.get(val[v].args[0], "row") != "tile":
                    demote = True
            if not demote and is_concat64(v):                 # tile concat: every input TILE
                if any(layout.get(f, "row") != "tile" for f in val[v].args[0].fields):
                    demote = True
            if not demote:
                v_mm_a = mm_a.get(v, set())
                v_mm_b = mm_b.get(v, set())
                for cv in consumers.get(v, []):               # all consumers TILE-compatible
                    if cv in v_mm_a and is_mm64(cv):
                        continue                              # read as A by a 64-mult matmul
                    if cv in v_mm_b and is_mm64(cv):
                        continue                              # read as B by a 64-mult matmul (tile act B)
                    if is_ew(cv) and layout.get(cv) == "tile":
                        continue                              # feeds a TILE ew
                    if is_reduce(cv) or is_max(cv):
                        continue                              # feeds a tile-native reduce (sum/max)
                    if (is_transpose(cv) or is_tslice(cv)) and layout.get(cv) == "tile":
                        continue                              # feeds a tile transpose/slice
                    if is_concat(cv):
                        continue                              # concat relayouts/places tile inputs
                    demote = True; break
            if demote:
                layout[v] = "row"; changed = True
    return layout


def _footprint(shape, lay):
    return tiled_numel(shape) if lay == "tile" else _numel(shape)


def _liveness(seq, bindings, fuse_oproj=True):
    """A1: fusion-aware last-read per binding var, in codegen EMISSION order.
    O-proj groups are emitted at the root and read the leaves' inputs (ctx_h/Wo_h)
    there, so those inputs stay live until the root — naive graph liveness would free
    them one step too early. Folded (consumed non-root) vars are never materialised.
    Must mirror codegen's fusion decision: only fuse when fuse_oproj (else the adds are
    emitted normally and their liveness is graph-natural). Returns (emit_idx, reads_at,
    last_read, consumed, fused_root)."""
    from . import codegen                                  # module-level helper (lazy: avoids cycle)
    fused_root, consumed = codegen._detect_oproj_groups(seq) if fuse_oproj else ({}, set())
    val_of = {b.var: b.value for b in bindings}
    emit_idx, reads_at, i = {}, {}, 0
    for b in bindings:
        v = b.var
        if v in consumed and v not in fused_root:         # folded away, not emitted
            continue
        if v in fused_root:                               # group reads every leaf's inputs here
            rd = []
            for leaf in fused_root[v]:
                mm = val_of[leaf]
                rd += [a for a in mm.args if isinstance(a, relax.Var)]
            reads_at[i] = rd
        else:
            reads_at[i] = _arg_vars(b.value)
        emit_idx[v] = i
        i += 1
    last_read = {}
    for idx, rds in reads_at.items():
        for u in rds:
            if last_read.get(u, -1) < idx:
                last_read[u] = idx
    return emit_idx, reads_at, last_read, consumed, fused_root


def plan(func, pack=True, pack_params=False, layouts=True, reuse=False, fuse_oproj=True):
    """Plan a Relax function: assign G-buffer offsets to params, constants, and
    every binding var. Returns a MemPlan. Assumes one dataflow block returning a Var.

    pack=True        : pre-pack matmul weight CONSTANTS (tile-blocked) — automatic.
    pack_params=True : ALSO mark matmul weight PARAMS (name starts 'W', static model
                       weights) as packed. The param's byte-size is unchanged (same
                       K*N); only packed_meta is recorded so codegen reads it packed
                       (no gather) and run_compiled packs the fed data. Cache params
                       (Kt*/Vc*, filled at runtime) are NOT weights -> not packed.
    reuse=True       : A1 liveness-based offset reuse for ACTIVATION binding vars — a
                       var's slot is freed at its last read and handed to a later
                       same-size var (exact-size free-list). Value-invariant (data
                       location only) -> byte-exact vs reuse=False. Safe now that A4
                       (5d-2) eliminated all gather, so there is no gather_cache
                       write-once conflict."""
    mp = MemPlan()
    param_set = set(func.params)
    seq = func.body
    assert isinstance(seq, relax.SeqExpr), "expected SeqExpr body"
    bindings = [b for block in seq.blocks for b in block.bindings]

    # resolve output var (unwrap 1-tuple) then assign layouts (A4 5c/5d)
    for b in bindings:
        if isinstance(b.value, relax.Tuple):
            mp.tuple_of[b.var] = list(b.value.fields)
    out = seq.body
    while out in mp.tuple_of:
        out = mp.tuple_of[out][0]
    assert isinstance(out, relax.Var), f"expected Var output, got {type(out)}"
    layout = assign_layouts(bindings, out, tile_inputs=pack_params) if layouts else {}

    # params AFTER layout so a tile-marked activation input (residual x) gets tile
    # footprint; run_compiled packs its fed data host-side (no device relayout).
    for p in func.params:
        mp.params.append(p)
        if layout.get(p) == "tile":
            mp.alloc_tiled(p)
            mp.tiled_inputs.add(p)
        else:
            mp.alloc(p)

    # A1: fusion-aware liveness + exact-size free-list (activation offset reuse)
    emit_idx = reads_at = last_read = None
    consumed_set, fused_root = set(), {}
    keep_alive = set(param_set)
    # A physical offset may be reached through more than one graph value after
    # earlier reuse.  A set prevents duplicate free-list entries from assigning
    # that same offset to two simultaneously-live later values.
    free_list = {}                                         # footprint -> {freed offsets}
    if reuse:
        emit_idx, reads_at, last_read, consumed_set, fused_root = _liveness(seq, bindings, fuse_oproj)
        val_of = {b.var: b.value for b in bindings}
        o = out                                           # output + its alias chain never freed
        keep_alive.add(o)
        while isinstance(val_of.get(o), relax.Var):
            o = val_of[o]; keep_alive.add(o)

    def _alloc_reuse(v, lay):
        shape, dtype = _shape_dtype(v.struct_info)
        size = _footprint(shape, lay)
        pool = free_list.get(size)
        if pool:
            off = pool.pop()
        else:
            off = mp.top
            mp.top += size
        mp.offset[v] = off; mp.shape[v] = shape; mp.dtype[v] = dtype; mp.layout[v] = lay
        for u in reads_at.get(emit_idx.get(v, -1), []):   # free inputs dead after this binding
            if (last_read.get(u) == emit_idx[v] and u not in keep_alive
                    and u in mp.offset and u not in mp.const_data):
                free_list.setdefault(
                    _footprint(mp.shape[u], mp.layout[u]), set()).add(mp.offset[u])

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
            ew_tile = is_ew and layout.get(binding.var) == "tile"
            for idx, arg in enumerate(val.args):
                if isinstance(arg, relax.Constant) and arg not in mp.offset:
                    if pack and opname == "relax.matmul" and idx == 1 and _packable(arg):
                        mp.alloc_const_packed(arg)        # weight pre-packing (tile-blocked)
                    elif ew_tile and bsh is not None and len(bsh) == 2 and bsh[0] % 64 == 0 and bsh[1] % 64 == 0:
                        mp.alloc_const_tiled(arg, broadcast_to=bsh)   # TILE ew operand (A4 5d)
                    else:
                        mp.alloc_const(arg, broadcast_to=bsh)
                elif (pack_params and opname == "relax.matmul" and idx == 1
                      and arg in param_set and arg not in mp.packed_meta
                      and arg.name_hint.startswith("W") and _packable(arg)):
                    mp.packed_meta[arg] = int(arg.struct_info.shape[1]) // 64
        v = binding.var
        lay = layout.get(v, "row")
        if reuse:
            if v in consumed_set and v not in fused_root:  # folded non-root: never emitted/read
                shape, dtype = _shape_dtype(v.struct_info)   # dead -> dummy offset (never read)
                mp.offset[v] = 0; mp.shape[v] = shape; mp.dtype[v] = dtype; mp.layout[v] = lay
            else:                                          # normal var OR O-proj root (emitted)
                _alloc_reuse(v, lay)
        elif lay == "tile":
            mp.alloc_tiled(v)
        else:
            mp.alloc(v)
    mp.output = out
    return mp
