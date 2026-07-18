"""T0: TIR+tensorize codegen backend.

Pipeline:  Relax matmul --LegalizeOps--> TIR (scalar loops)
           --tir.Schedule (split 64^3 / reorder / decompose_reduction / tensorize)-->
           outer loops + npu_fill_zero / npu_gemm_acc intrinsic calls
           --_Walker (unroll loops, evaluate index exprs, map intrinsics)--> NPU ISA.

The walker is the "TIR -> ISA codegen": it interprets structured TIR (For /
BlockRealize / match_buffers / call_extern) and emits instructions via isa.Asm.
Numeric semantics preserve mysim's FP16-round-on-save order:
  C = 0;  C = fp16(C + fp16(partial_k))  for each k-tile
which is byte-identical to the existing oracle (fp16(0+x) == fp16(x)).

T0 scope: matmul with M, N, K all multiples of 64 (padding comes later).
"""
import tvm
import tvm.tir as tir
from tvm import relax, arith
from tvm.script import tir as T

from . import memplan as _memplan
from .isa import Asm, SRC1, SRC2, DST, IMM, VECTOR

TILE = 64

# Pure-Python evaluator dispatch for the affine index math in scheduled GEMM TIR.
# Walking these in Python (below) avoids a TVM FFI substitute+simplify round-trip per
# index expr — the compile hot path. Anything not here falls back to arith.simplify.
_TIR_BINOPS = {
    tir.Add: lambda a, b: a + b,
    tir.Sub: lambda a, b: a - b,
    tir.Mul: lambda a, b: a * b,
    tir.FloorDiv: lambda a, b: a // b,       # matches TVM floordiv for all signs
    tir.FloorMod: lambda a, b: a % b,        # matches TVM floormod for all signs
    tir.Min: min,
    tir.Max: max,
}


# ============================ intrinsic definitions ============================
# desc = what the 64x64 block computes (for pattern matching)
# impl = a call_extern marker the walker recognizes (carrying ptr+stride args)
@T.prim_func
def _fill_desc(c: T.handle):
    sc = T.int32()
    C = T.match_buffer(c, (TILE, TILE), "float16", strides=[sc, 1], offset_factor=1)
    with T.block("root"):
        T.reads()
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.block("fill"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.float16(0)


@T.prim_func
def _fill_impl(c: T.handle):
    sc = T.int32()
    C = T.match_buffer(c, (TILE, TILE), "float16", strides=[sc, 1], offset_factor=1)
    with T.block("root"):
        T.reads()
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("int32", "npu_fill_zero", C.access_ptr("w"), sc))


@T.prim_func
def _gemm_desc(a: T.handle, b: T.handle, c: T.handle):
    sa = T.int32(); sb = T.int32(); sc = T.int32()
    A = T.match_buffer(a, (TILE, TILE), "float16", strides=[sa, 1], offset_factor=1)
    B = T.match_buffer(b, (TILE, TILE), "float16", strides=[sb, 1], offset_factor=1)
    C = T.match_buffer(c, (TILE, TILE), "float16", strides=[sc, 1], offset_factor=1)
    with T.block("root"):
        T.reads(C[0:TILE, 0:TILE], A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j, k in T.grid(TILE, TILE, TILE):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def _gemm_impl(a: T.handle, b: T.handle, c: T.handle):
    sa = T.int32(); sb = T.int32(); sc = T.int32()
    A = T.match_buffer(a, (TILE, TILE), "float16", strides=[sa, 1], offset_factor=1)
    B = T.match_buffer(b, (TILE, TILE), "float16", strides=[sb, 1], offset_factor=1)
    C = T.match_buffer(c, (TILE, TILE), "float16", strides=[sc, 1], offset_factor=1)
    with T.block("root"):
        T.reads(C[0:TILE, 0:TILE], A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("int32", "npu_gemm_acc",
                                 C.access_ptr("rw"), sc,
                                 A.access_ptr("r"), sa,
                                 B.access_ptr("r"), sb))


def _register():
    for name, d, i in [("npu_fill_zero", _fill_desc, _fill_impl),
                       ("npu_gemm_acc", _gemm_desc, _gemm_impl)]:
        try:
            tir.TensorIntrin.register(name, d, i)
        except Exception:
            pass                       # already registered (re-import)


_register()


# ============================ schedule ============================
class TirBackendError(Exception):
    pass


def schedule_matmul(mod, func_name):
    """Split the legalized matmul into 64^3 tiles and tensorize init/update."""
    sch = tir.Schedule(mod)
    blk = sch.get_block("matmul", func_name=func_name)
    i, j, k = sch.get_loops(blk)
    io, ii = sch.split(i, [None, TILE])
    jo, ji = sch.split(j, [None, TILE])
    ko, ki = sch.split(k, [None, TILE])
    sch.reorder(io, jo, ko, ii, ji, ki)
    init_blk = sch.decompose_reduction(blk, ko)
    sch.tensorize(sch.get_loops(blk)[3], "npu_gemm_acc")     # update at ii
    sch.tensorize(sch.get_loops(init_blk)[2], "npu_fill_zero")  # init at ii
    return sch.mod


# ============================ TIR -> ISA walker ============================
class _Walker:
    """Interpret scheduled TIR: unroll For loops, bind block/match-buffer vars,
    evaluate index expressions to constants, emit ISA for intrinsic calls."""

    def __init__(self, asm, mp, data_base, verbose=False, gather_cache=None):
        self.a = asm
        self.mp = mp
        self.base = dict(data_base)        # buffer-data Var -> G-buffer base offset
        self.env = {}                      # tir Var -> IntImm (current binding)
        self.ana = arith.Analyzer()
        self.verbose = verbose             # learning hook: trace each intrinsic/gather
        # T1 input-reuse state; gather_cache may be SHARED across matmuls (write-once
        # sources => same (off,stride) = same data) so a shared activation tile (e.g.
        # xn read by all Q/K/V projs) is gathered ONCE for the whole layer.
        self.gather_cache = gather_cache if gather_cache is not None else {}  # (src_off,stride)->scratch
        self.zeroed = set()                # C-tile offsets currently == 0 (fill not yet materialized)
        self.cbuf = {}                     # strided C-tile addr -> (contiguous accum scratch, stride)
        self.packed_src = {}               # buffer.data -> (packed_base, Nt) for tile-blocked weights
        self.suppress_fill = False         # group-accumulate: keep prior C accumulator across matmuls

    # ---- expression / pointer evaluation ----
    def ev(self, expr):
        v = self._ev_fast(expr)                          # pure-Python fast path (no FFI)
        if v is not None:
            return v
        env = {k: tir.IntImm(k.dtype if k.dtype else "int64", i) for k, i in self.env.items()}
        e = tir.stmt_functor.substitute(expr, env)       # fallback for unhandled node types
        e = self.ana.simplify(e)
        if not isinstance(e, tir.IntImm):
            raise TirBackendError(f"cannot const-evaluate: {expr}")
        return int(e.value)

    def _ev_fast(self, e):
        """Evaluate an affine index expr from self.env (Var->int) without TVM FFI.
        Returns an int, or None if the expr contains a node type we don't handle
        (-> ev() falls back to substitute+simplify). env holds plain Python ints."""
        t = type(e)
        if t is tir.Var or t is tir.SizeVar:
            return self.env.get(e)                       # int, or None (unbound -> fallback)
        if t is tir.IntImm:
            return int(e.value)
        op = _TIR_BINOPS.get(t)
        if op is not None:
            a = self._ev_fast(e.a)
            if a is None:
                return None
            b = self._ev_fast(e.b)
            return None if b is None else op(a, b)
        if t is tir.Cast:
            return self._ev_fast(e.value)
        return None

    def _bind(self, var, value):
        self.env[var] = value                            # plain Python int (see _ev_fast)

    def ptr(self, call):                  # tvm_access_ptr(type, data, elem_offset, extent, mask)
        data = call.args[1]
        if data not in self.base:
            raise TirBackendError(f"unknown buffer data var {data}")
        return self.base[data] + self.ev(call.args[2])

    # ---- statement dispatch ----
    def walk(self, stmt):
        if isinstance(stmt, tir.For):
            lo, n = self.ev(stmt.min), self.ev(stmt.extent)
            for v in range(lo, lo + n):
                self._bind(stmt.loop_var, v)
                self.walk(stmt.body)
            self.env.pop(stmt.loop_var, None)
        elif isinstance(stmt, tir.SeqStmt):
            for s in stmt.seq:
                self.walk(s)
        elif isinstance(stmt, tir.BlockRealize):
            blk = stmt.block
            for iv, val in zip(blk.iter_vars, stmt.iter_values):
                self._bind(iv.var, self.ev(val))
            for mbr in blk.match_buffers:
                self._bind_match(mbr)
            if blk.init is not None:
                self.walk(blk.init)
            self.walk(blk.body)
        elif isinstance(stmt, tir.Evaluate):
            self._intrinsic(stmt.value)
        elif isinstance(stmt, (tir.AttrStmt, tir.DeclBuffer)):
            self.walk(stmt.body)
        elif isinstance(stmt, tir.LetStmt):
            self._bind(stmt.var, self.ev(stmt.value))
            self.walk(stmt.body)
        else:
            raise TirBackendError(f"walker: unhandled stmt {type(stmt).__name__}")

    def _bind_match(self, mbr):
        """match_buffer: bind the view's data/elem_offset/stride symbols to the
        source buffer's concrete values (source assumed compact 2D root)."""
        buf, src = mbr.buffer, mbr.source
        sbuf = src.buffer
        if sbuf.data not in self.base:
            raise TirBackendError(f"match_buffer source not a root param: {sbuf.name}")
        if sbuf.data in self.packed_src:                     # tile-blocked weight: tile is contiguous
            packed_base, Nt = self.packed_src[sbuf.data]
            rt = self.ev(src.region[0].min) // TILE          # row tile index (ko)
            ct = self.ev(src.region[1].min) // TILE          # col tile index (jo)
            off = (rt * Nt + ct) * TILE * TILE               # blocked offset
            self.base[buf.data] = packed_base
            if isinstance(buf.elem_offset, tir.Var):
                self._bind(buf.elem_offset, off)
            if len(buf.strides) and isinstance(buf.strides[0], tir.Var):
                self._bind(buf.strides[0], TILE)             # stride==64 -> _gather_cached skips gather
            return
        row_stride = int(sbuf.shape[1])                       # compact 2D
        off = self.ev(src.region[0].min) * row_stride + self.ev(src.region[1].min)
        self.base[buf.data] = self.base[sbuf.data]            # alias to root base
        if isinstance(buf.elem_offset, tir.Var):
            self._bind(buf.elem_offset, off)
        if len(buf.strides) and isinstance(buf.strides[0], tir.Var):
            self._bind(buf.strides[0], row_stride)

    # ---- intrinsic emission ----
    def _intrinsic(self, call):
        if not (isinstance(call, tir.Call) and call.op.name == "tir.call_extern"):
            raise TirBackendError(f"walker: unhandled call {call}")
        name = call.args[0].value
        if name == "npu_fill_zero":
            c, sc = self.ptr(call.args[1]), self.ev(call.args[2])
            if self.verbose:
                print(f"    [walker] fill_zero  C@{c}(stride {sc}) -> mark tile as zero")
            self.emit_fill(c, sc)
        elif name == "npu_gemm_acc":
            c, sc = self.ptr(call.args[1]), self.ev(call.args[2])
            a, sa = self.ptr(call.args[3]), self.ev(call.args[4])
            b, sb = self.ptr(call.args[5]), self.ev(call.args[6])
            if self.verbose:
                print(f"    [walker] gemm_acc   C@{c}(s{sc}) += A@{a}(s{sa}) · B@{b}(s{sb})")
            self.emit_acc(c, sc, a, sa, b, sb)
        else:
            raise TirBackendError(f"unknown extern {name}")

    def emit_fill(self, c, sc):
        """T1: don't materialize the zero fill — just mark the C tile as zero.
        The first accumulate stores the partial directly (fp16(0+x)==fp16(x)).
        In group-accumulate mode (suppress_fill) the C tile keeps the running
        cross-matmul sum, so the re-init from a later matmul is skipped."""
        if self.suppress_fill:
            return
        self.zeroed.add(c)

    def _gather_cached(self, off, stride):
        """T1 input reuse: a tile is read-only, so gather each distinct
        (src, stride) tile at most ONCE and reuse the contiguous copy.
        Already-contiguous tiles (stride==64) need no gather at all."""
        if stride == TILE:
            if self.verbose:
                print(f"      gather  SKIP  @{off} already contiguous (stride==64)")
            return off
        key = (off, stride)
        hit = self.gather_cache.get(key)
        if hit is not None:
            if self.verbose:
                print(f"      gather  REUSE @{off}(s{stride}) -> scratch@{hit} (memoized)")
            return hit
        dst = self.mp.scratch_alloc(TILE * TILE)
        a = self.a
        with a.role("gather"):
            for r in range(TILE):
                a.vlen(TILE)
                a.addr(SRC1, off + r * stride); a.load(0, 0)
                a.v_add(mode=IMM, imm=0)
                a.addr(DST, dst + r * TILE); a.save(0)
        self.gather_cache[key] = dst
        if self.verbose:
            print(f"      gather  NEW   @{off}(s{stride}) -> scratch@{dst} (copy {TILE} rows)")
        return dst

    def emit_acc(self, c, sc, aoff, sa, boff, sb):
        """C[64,64] += A[64,64] @ B[64,64]. 0710: K-accumulation uses the matmul MAC
        bit (pout += A@B) instead of a save/load/add round-trip — the running
        accumulator is copied into pout, then m_mul(mac) adds this k-tile in place.
        (Each k-tile still FP16-rounds on store; both matmul paths use the identical
        sequence so direct==tir stays byte-exact.)"""
        a = self.a
        asrc = self._gather_cached(aoff, sa)
        bsrc = self._gather_cached(boff, sb)
        first = c in self.zeroed
        self.zeroed.discard(c)
        if sc == TILE:                               # C contiguous -> accumulate in place
            acc = c
        else:                                        # strided C -> contiguous accumulator, scatter at flush
            if first:
                self.cbuf[c] = (self.mp.scratch_alloc(TILE * TILE), sc)
            acc = self.cbuf[c][0]
        if not first:                                # preload running sum into pout for MAC
            with a.role("accum"):
                a.vlen(TILE * TILE)
                a.addr(SRC1, acc); a.load(0, 0); a.v_copy()
        with a.role("mmul"):                         # 64x64 matmul (MAC-accumulate if not first)
            a.tile(0, TILE, TILE); a.tile(1, TILE, TILE)
            a.addr(SRC1, asrc); a.load(1, 0)
            a.addr(SRC2, bsrc); a.load(1, 1)
            a.m_mul(mode=VECTOR, mac=not first)
            a.addr(DST, acc); a.save(1)

    def flush(self):
        """Scatter finished contiguous C-accumulators to their strided locations."""
        a = self.a
        with a.role("scatter"):
            for c, (ctile, sc) in self.cbuf.items():
                for r in range(TILE):
                    a.vlen(TILE)
                    a.addr(SRC1, ctile + r * TILE); a.load(0, 0)
                    a.v_add(mode=IMM, imm=0)
                    a.addr(DST, c + r * sc); a.save(0)
        self.cbuf.clear()


# ============================ padding helpers ============================
def _ceil64(x):
    return (x + TILE - 1) // TILE * TILE


def _copy2d(asm, dst_off, dst_stride, src_off, src_stride, rows, cols):
    """Per-row copy of a [rows,cols] block (a+0); either side may be strided."""
    with asm.role("pad"):
        for r in range(rows):
            asm.vlen(cols)
            asm.addr(SRC1, src_off + r * src_stride); asm.load(0, 0)
            asm.v_add(mode=IMM, imm=0)
            asm.addr(DST, dst_off + r * dst_stride); asm.save(0)


def _copy_block(asm, dst_off, src_off, n, CH=8192):
    """Contiguous copy of n elements (chunked to the PE buffer size)."""
    with asm.role("pad"):
        for base in range(0, n, CH):
            m = min(CH, n - base)
            asm.vlen(m)
            asm.addr(SRC1, src_off + base); asm.load(0, 0)
            asm.v_add(mode=IMM, imm=0)
            asm.addr(DST, dst_off + base); asm.save(0)


# ============================ single-matmul emit (hybrid entry) ============================
def emit_matmul_into(asm, mp, c_off, a_off, b_off, M, K, N, b_pack_nt=None, gather_cache=None,
                     a_tiled=False, c_tiled=False):
    """Emit one matmul C[M,N]=A[M,K]@B[K,N] into an existing asm at the given
    G-buffer offsets, via the TIR+tensorize+walker path (T1 input reuse).

    Non-64-multiple dims (e.g. arbitrary prefill SEQ) are PADDED to 64-multiples
    so matmul-the-op is TIR-only (no direct fallback). Padding is byte-exact vs
    direct: padded K-terms are 0 (fresh scratch is zero-initialised by the driver),
    and padded M/N rows/cols are sliced off. Fast path: when only M is non-64
    (K,N already 64-multiple, the common projection case) A/B are read in place and
    only the output is staged in scratch -> no input copy.

    b_pack_nt: if set, B is a tile-blocked (pre-packed) weight stored at b_off as
    [Kt,Nt,64,64] (Nt=b_pack_nt); the walker reads each B tile contiguously (no gather)."""
    Mp, Kp, Np = _ceil64(M), _ceil64(K), _ceil64(N)
    if (Mp, Kp, Np) == (M, K, N):
        _emit_tir_gemm(asm, mp, c_off, a_off, b_off, M, K, N, b_pack_nt=b_pack_nt,
                       gather_cache=gather_cache,
                       a_tiled=(K // TILE if a_tiled else None),
                       c_tiled=(N // TILE if c_tiled else None))
        return
    assert not (a_tiled or c_tiled), \
        f"tile-blocked A/C need 64-multiple M,K,N; got {M}x{K}x{N}"
    if Kp == K and Np == N:                       # M-only padding (cheap: no input copy)
        cpad = mp.scratch_alloc(Mp * N)
        _emit_tir_gemm(asm, mp, cpad, a_off, b_off, Mp, K, N, b_pack_nt=b_pack_nt, gather_cache=gather_cache)
        _copy_block(asm, c_off, cpad, M * N)                    # first M rows are contiguous
        return
    # general padding: stage A,B in fresh(=zero) scratch so padded K is 0, slice C out
    # (B packing not applied here — only triggers when N or K non-64, never for weights)
    apad = mp.scratch_alloc(Mp * Kp)
    bpad = mp.scratch_alloc(Kp * Np)
    cpad = mp.scratch_alloc(Mp * Np)
    _copy2d(asm, apad, Kp, a_off, K, M, K)        # A[M,K] -> apad[0:M,0:K] (rest stays 0)
    _copy2d(asm, bpad, Np, b_off, N, K, N)        # B[K,N] -> bpad[0:K,0:N]
    _emit_tir_gemm(asm, mp, cpad, apad, bpad, Mp, Kp, Np)
    _copy2d(asm, c_off, N, cpad, Np, M, N)        # cpad[0:M,0:N] -> C[M,N]


def _scheduled_gemm(M, K, N):
    """Legalize+schedule a [M,K]@[K,N] matmul; return the scheduled prim_func
    (params: A_in, B_in, C_out). Pure shape function -> safe to cache/reuse."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([M, K], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([K, N], "float16"))
    with bb.function("mm", [x, w]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x, w)); gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    mod = relax.transform.LegalizeOps()(bb.finalize())
    gvar = None
    for blk in mod["mm"].body.blocks:
        for bd in blk.bindings:
            if isinstance(bd.value, relax.Call) and bd.value.op.name == "relax.call_tir":
                gvar = bd.value.args[0]
    sched = schedule_matmul(mod, gvar.name_hint)
    return sched[gvar.name_hint]


def _bind_gemm(wk, pf, a_off, b_off, c_off, b_pack_nt=None, a_tiled=None, c_tiled=None):
    """Point a (reused) walker's root buffers at this matmul's G-buffer offsets.
    a_tiled/c_tiled (A4): if set (=K//64 / =N//64), A/C are tile-blocked so each 64x64
    tile is contiguous -> the walker skips the A-gather / C-scatter (stride==64)."""
    aD = pf.buffer_map[pf.params[0]].data
    bD = pf.buffer_map[pf.params[1]].data
    cD = pf.buffer_map[pf.params[2]].data
    wk.base[aD] = a_off; wk.base[bD] = b_off; wk.base[cD] = c_off
    wk.packed_src = {}
    if b_pack_nt is not None:                               # B tile-blocked weight -> no gather
        wk.packed_src[bD] = (b_off, b_pack_nt)
    if a_tiled is not None:                                 # A tile-blocked -> no A-gather
        wk.packed_src[aD] = (a_off, a_tiled)
    if c_tiled is not None:                                 # C tile-blocked -> no C-scatter
        wk.packed_src[cD] = (c_off, c_tiled)


def _emit_tir_gemm(asm, mp, c_off, a_off, b_off, M, K, N, b_pack_nt=None, gather_cache=None,
                   a_tiled=None, c_tiled=None):
    """Core TIR+tensorize emit for a 64-multiple matmul (no padding)."""
    for d in (M, K, N):
        assert d % TILE == 0, f"_emit_tir_gemm needs 64-multiples, got {M}x{K}x{N}"
    pf = _scheduled_gemm(M, K, N)
    wk = _Walker(asm, mp, {}, gather_cache=gather_cache)
    _bind_gemm(wk, pf, a_off, b_off, c_off, b_pack_nt, a_tiled=a_tiled, c_tiled=c_tiled)
    wk.walk(pf.body)
    wk.flush()


def emit_matmul_accumulate_group(asm, mp, c_off, terms, gather_cache=None):
    """Emit  C[M,N] = sum_t (A_t[M,K_t] @ B_t[K_t,N])  with a SINGLE output flush.

    Used for the per-head attention output projection: instead of materialising
    each head's full [M,N] product (one strided scatter per head) and adding them,
    all head products accumulate in one shared set of contiguous C-tile buffers
    (in-G-buffer 'accum'), scattered to the strided output exactly once. Turns
    H output scatters into 1.

    terms: list of (a_off, b_off, M, K, N, b_pack_nt) sharing the same (M, N).
    M may be non-64 (decode, M=1): computed padded then the first M rows copied out.
    K, N must be 64-multiples (true for o_proj: K=head_dim=128, N=hidden)."""
    M, N = terms[0][2], terms[0][4]
    Mp = _ceil64(M)
    assert N % TILE == 0, f"group N must be 64-multiple, got {N}"
    out = c_off
    cpad = None
    if Mp != M:                                            # decode: pad M, copy valid rows out
        cpad = mp.scratch_alloc(Mp * N); out = cpad
    wk = _Walker(asm, mp, {}, gather_cache=gather_cache)
    sched_cache = {}
    for i, (a_off, b_off, M_t, K_t, N_t, nt) in enumerate(terms):
        assert K_t % TILE == 0 and N_t == N, f"bad group term {M_t}x{K_t}x{N_t}"
        pf = sched_cache.get((Mp, K_t, N))
        if pf is None:
            pf = _scheduled_gemm(Mp, K_t, N); sched_cache[(Mp, K_t, N)] = pf
        _bind_gemm(wk, pf, a_off, b_off, out, nt)
        wk.suppress_fill = (i > 0)                         # only the first term zero-inits C
        wk.walk(pf.body)
    wk.flush()                                             # ONE scatter for the whole sum
    if cpad is not None:
        _copy_block(asm, c_off, cpad, M * N)


# ============================ compile entry ============================
def compile_func(relax_mod, func_name="main"):
    """Whole Relax function -> (Asm, MemPlan), matmul on the TIR path.
    Thin wrapper over codegen with mm_backend='tir' (elementwise stays direct)."""
    from . import codegen, memplan
    func = relax_mod[func_name]
    mp = memplan.plan(func)
    # tile=64 so non-64-multiple matmuls (router fallback) use hardware-legal
    # direct tiling, not the logical one-shot path.
    asm = codegen.compile_func(func, mp, tile=64, mm_backend="tir")
    return asm, mp
