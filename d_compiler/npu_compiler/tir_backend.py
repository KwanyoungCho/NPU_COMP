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

    def __init__(self, asm, mp, data_base):
        self.a = asm
        self.mp = mp
        self.base = dict(data_base)        # buffer-data Var -> G-buffer base offset
        self.env = {}                      # tir Var -> IntImm (current binding)
        self.ana = arith.Analyzer()
        self.sP = mp.scratch_alloc(TILE * TILE)   # matmul partial
        # T1 input-reuse state:
        self.gather_cache = {}             # (src_off, stride) -> scratch offset (gather once)
        self.zeroed = set()                # C-tile offsets currently == 0 (fill not yet materialized)
        self.cbuf = {}                     # strided C-tile addr -> (contiguous accum scratch, stride)

    # ---- expression / pointer evaluation ----
    def ev(self, expr):
        e = tir.stmt_functor.substitute(expr, self.env)
        e = self.ana.simplify(e)
        if not isinstance(e, tir.IntImm):
            raise TirBackendError(f"cannot const-evaluate: {expr}")
        return int(e.value)

    def _bind(self, var, value):
        self.env[var] = tir.IntImm(var.dtype if var.dtype else "int64", value)

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
            self.emit_fill(self.ptr(call.args[1]), self.ev(call.args[2]))
        elif name == "npu_gemm_acc":
            self.emit_acc(self.ptr(call.args[1]), self.ev(call.args[2]),
                          self.ptr(call.args[3]), self.ev(call.args[4]),
                          self.ptr(call.args[5]), self.ev(call.args[6]))
        else:
            raise TirBackendError(f"unknown extern {name}")

    def emit_fill(self, c, sc):
        """T1: don't materialize the zero fill — just mark the C tile as zero.
        The first accumulate stores the partial directly (fp16(0+x)==fp16(x))."""
        self.zeroed.add(c)

    def _gather_cached(self, off, stride):
        """T1 input reuse: a tile is read-only, so gather each distinct
        (src, stride) tile at most ONCE and reuse the contiguous copy.
        Already-contiguous tiles (stride==64) need no gather at all."""
        if stride == TILE:
            return off
        key = (off, stride)
        hit = self.gather_cache.get(key)
        if hit is not None:
            return hit
        dst = self.mp.scratch_alloc(TILE * TILE)
        a = self.a
        for r in range(TILE):
            a.vlen(TILE)
            a.addr(SRC1, off + r * stride); a.load(0, 0)
            a.v_add(mode=IMM, imm=0)
            a.addr(DST, dst + r * TILE); a.save(0)
        self.gather_cache[key] = dst
        return dst

    def emit_acc(self, c, sc, aoff, sa, boff, sb):
        """C[64,64] += A[64,64] @ B[64,64]  (each save FP16-rounds, matching mysim)."""
        a = self.a
        asrc = self._gather_cached(aoff, sa)
        bsrc = self._gather_cached(boff, sb)
        first = c in self.zeroed
        self.zeroed.discard(c)
        a.tile(0, TILE, TILE); a.tile(1, TILE, TILE)
        a.addr(SRC1, asrc); a.load(1, 0)
        a.addr(SRC2, bsrc); a.load(1, 1)
        a.m_mul(mode=VECTOR)
        if sc == TILE:                               # C contiguous
            if first:
                a.addr(DST, c); a.save(1)            # store partial directly (no add)
            else:
                a.addr(DST, self.sP); a.save(1)
                a.vlen(TILE * TILE)
                a.addr(SRC1, c); a.load(0, 0)
                a.addr(SRC2, self.sP); a.load(0, 1); a.v_add(mode=VECTOR)
                a.addr(DST, c); a.save(0)
        else:                                        # strided C: accumulate in a
            if first:                                # contiguous tile buffer, scatter at flush
                ctile = self.mp.scratch_alloc(TILE * TILE)
                self.cbuf[c] = (ctile, sc)
                a.addr(DST, ctile); a.save(1)        # partial -> contiguous accumulator
            else:
                ctile = self.cbuf[c][0]
                a.addr(DST, self.sP); a.save(1)
                a.vlen(TILE * TILE)                  # accumulator += partial (one shot)
                a.addr(SRC1, ctile); a.load(0, 0)
                a.addr(SRC2, self.sP); a.load(0, 1); a.v_add(mode=VECTOR)
                a.addr(DST, ctile); a.save(0)

    def flush(self):
        """Scatter finished contiguous C-accumulators to their strided locations."""
        a = self.a
        for c, (ctile, sc) in self.cbuf.items():
            for r in range(TILE):
                a.vlen(TILE)
                a.addr(SRC1, ctile + r * TILE); a.load(0, 0)
                a.v_add(mode=IMM, imm=0)
                a.addr(DST, c + r * sc); a.save(0)
        self.cbuf.clear()


# ============================ single-matmul emit (hybrid entry) ============================
def emit_matmul_into(asm, mp, c_off, a_off, b_off, M, K, N):
    """Emit one matmul C[M,N]=A[M,K]@B[K,N] into an existing asm at the given
    G-buffer offsets, via the TIR+tensorize+walker path (T1 input reuse).
    codegen.py calls this so a whole-graph compile can route matmul here while
    keeping elementwise/transpose on the direct path (hybrid)."""
    for d in (M, K, N):
        if d % TILE:
            raise TirBackendError(f"TIR matmul needs dims % {TILE} == 0, got {M}x{K}x{N} (pad TODO)")
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
    pf = sched[gvar.name_hint]                              # params: A_in, B_in, C_out
    data_base = {pf.buffer_map[pf.params[0]].data: a_off,
                 pf.buffer_map[pf.params[1]].data: b_off,
                 pf.buffer_map[pf.params[2]].data: c_off}
    wk = _Walker(asm, mp, data_base)
    wk.walk(pf.body)
    wk.flush()


# ============================ compile entry ============================
def compile_func(relax_mod, func_name="main"):
    """Whole Relax function -> (Asm, MemPlan), matmul on the TIR path.
    Thin wrapper over codegen with mm_backend='tir' (elementwise stays direct)."""
    from . import codegen, memplan
    func = relax_mod[func_name]
    mp = memplan.plan(func)
    asm = codegen.compile_func(func, mp, mm_backend="tir")
    return asm, mp
