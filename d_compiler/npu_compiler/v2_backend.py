"""v2 codegen (Path A) — unified TIR->ISA lowering, built ALONGSIDE v1 (v1 = oracle).

Direction (see COMPILER_V2_PLAN.md): every op reaches ONE walker as a `call_extern`
marker and is lowered to NPU ISA there, replacing v1's per-op hand emitters in
codegen.py. v1 already does this for matmul (`npu_gemm_acc`/`npu_fill_zero`, a real
tensorize intrinsic — tir_backend.py). This module extends the SAME walker to
non-matmul ops, starting with elementwise (Phase 2-A). Byte-exact vs v1 (tests/test_v2.py).

Marker convention (the "already-lowered" legalization of an op):
  binary ew:  call_extern("npu_ew2_<op>", C_ptr, A_ptr, B_ptr, N)   op in add/sub/mul/div
  unary  ew:  call_extern("npu_ew1_<op>", C_ptr, A_ptr, N)          op in sqrt/exp/neg/cos/sin
  silu:       call_extern("npu_silu",     C_ptr, A_ptr, N)
where N is the element count (walker chunks it to the PE buffer, exactly like emit_ew).
"""
import tvm.tir as tir
from tvm.script import tir as T
from tvm import relax as _relax

from .isa import SRC1, SRC2, DST, IMM, VECTOR, Asm
from .tir_backend import _Walker, TILE, TirBackendError, emit_matmul_into
from . import memplan as _memplan

CH = 8192  # PE-buffer chunk; matches codegen.emit_ew (16-bit vlen field)

# marker op suffix -> Asm method name (mirrors codegen.py EW2/EW1)
_EW2 = {"add": "v_add", "subtract": "v_sub", "multiply": "v_mul", "divide": "v_div"}
_EW1 = {"sqrt": "v_sqrt", "exp": "v_exp", "negative": "v_sign_inv", "cos": "v_cos", "sin": "v_sin"}


class V2Walker(_Walker):
    """v1 `_Walker` + elementwise markers. Unknown markers fall through to v1
    (`npu_gemm_acc`/`npu_fill_zero`), so matmul lowering is unchanged."""

    def _bind_match(self, mbr):
        """N-D generalization of v1's 2D-only _bind_match: resolve a match_buffer view
        over a root buffer to (base, elem_offset, leading-stride) for ANY rank (1D
        elementwise, 2D matmul, ...). Keeps v1's packed_src (tile-blocked) / a_m1_src
        special cases; the general branch is compact row-major over the region mins."""
        src = mbr.source
        sbuf = src.buffer
        sdata = sbuf.data
        if sdata not in self.base:
            raise TirBackendError(f"match_buffer source not a root param: {sbuf.name}")
        buf = mbr.buffer
        bdata = buf.data
        region = src.region
        packed = self.packed_src.get(sdata)
        if packed is not None:                                # 2D tile-blocked (matmul)
            packed_base, Nt = packed
            off = (self.ev(region[0].min) // TILE * Nt + self.ev(region[1].min) // TILE) * TILE * TILE
            self.base[bdata] = packed_base; strd = TILE
        elif sdata in self.a_m1_src:                          # 2D decode M=1
            rs = int(sbuf.shape[1])
            off = self.ev(region[0].min) * rs + self.ev(region[1].min)
            self.base[bdata] = self.base[sdata]; strd = TILE
        else:                                                 # general compact row-major, any rank
            shape = [int(s) for s in sbuf.shape]
            nd = len(region)
            strides = [1] * nd
            for k in range(nd - 2, -1, -1):
                strides[k] = strides[k + 1] * shape[k + 1]
            off = sum(self.ev(region[k].min) * strides[k] for k in range(nd))
            self.base[bdata] = self.base[sdata]
            strd = strides[0] if nd >= 2 else 1
        eoff = buf.elem_offset
        if isinstance(eoff, tir.Var):
            self._bind(eoff, off)
        bstrides = buf.strides
        if len(bstrides) and isinstance(bstrides[0], tir.Var):
            self._bind(bstrides[0], strd)

    def _intrinsic(self, call):
        if isinstance(call, tir.Call) and call.op.name == "tir.call_extern":
            name = call.args[0].value
            if name.startswith("npu_ew2_"):
                return self._emit_ew2(_EW2[name[8:]], call)
            if name.startswith("npu_ew1_"):
                return self._emit_ew1(_EW1[name[8:]], call)
            if name == "npu_silu":
                return self._emit_silu(call)
            if name == "npu_copy":
                return self._emit_copy(call)
            if name == "npu_ttile":
                return self._emit_ttile(call)
            if name == "npu_rsum_row":
                return self._emit_rsum_row(call)
            if name == "npu_rsum_tile":
                return self._emit_rsum_tile(call)
            if name == "npu_rmax_row":
                return self._emit_rmax_row(call)
            if name == "npu_rmax_tile":
                return self._emit_rmax_tile(call)
            if name == "npu_bcast_row":
                return self._emit_bcast_row(call)
            if name == "npu_bcast_col":
                return self._emit_bcast_col(call)
            if name == "npu_transpose":
                return self._emit_transpose_row(call)
            if name == "npu_slice":
                return self._emit_slice(call)
            if name == "npu_colplace":
                return self._emit_colplace(call)
        return super()._intrinsic(call)

    def _emit_ew2(self, method, call):
        c = self.ptr(call.args[1]); x0 = self.ptr(call.args[2]); x1 = self.ptr(call.args[3])
        n = self.ev(call.args[4]); a = self.a; op = getattr(a, method)
        for base in range(0, n, CH):
            a.vlen(min(CH, n - base))
            a.addr(SRC1, x0 + base); a.load(0, 0)
            a.addr(SRC2, x1 + base); a.load(0, 1)
            op(mode=VECTOR)
            a.addr(DST, c + base); a.save(0)

    def _emit_ew1(self, method, call):
        c = self.ptr(call.args[1]); x0 = self.ptr(call.args[2]); n = self.ev(call.args[3])
        a = self.a; op = getattr(a, method)
        for base in range(0, n, CH):
            a.vlen(min(CH, n - base))
            a.addr(SRC1, x0 + base); a.load(0, 0)
            op()
            a.addr(DST, c + base); a.save(0)

    def _emit_silu(self, call):
        c = self.ptr(call.args[1]); x0 = self.ptr(call.args[2]); n = self.ev(call.args[3])
        a = self.a
        for base in range(0, n, CH):
            a.vlen(min(CH, n - base))
            a.addr(SRC1, x0 + base); a.load(0, 0)
            a.m_add(mode=IMM, imm=0, act=True)          # 0710 native SiLU (act bit on m_add+0)
            a.addr(DST, c + base); a.save(0)

    # ---- thin native primitives (layout building blocks) ----
    def _emit_copy(self, call):                         # vector copy (0x17), n elems
        c = self.ptr(call.args[1]); s = self.ptr(call.args[2]); n = self.ev(call.args[3]); a = self.a
        for base in range(0, n, CH):
            a.vlen(min(CH, n - base))
            a.addr(SRC1, s + base); a.load(0, 0); a.v_copy()
            a.addr(DST, c + base); a.save(0)

    def _emit_ttile(self, call):                        # transpose one 64x64 tile (strided-load 0x90)
        d = self.ptr(call.args[1]); s = self.ptr(call.args[2]); a = self.a
        a.tile(0, 64, 64); a.addr(SRC1, s)
        a.load(1, 0, strided=1, ncols=64, start=0)      # column-major read = 64x64 transpose
        a.vlen(4096); a.v_copy()
        a.addr(DST, d); a.save(0)

    def _copy2d(self, dst_off, dst_stride, src_off, src_stride, rows, cols):
        a = self.a
        for r in range(rows):
            a.vlen(cols)
            a.addr(SRC1, src_off + r * src_stride); a.load(0, 0)
            a.v_add(mode=IMM, imm=0)
            a.addr(DST, dst_off + r * dst_stride); a.save(0)

    # ---- broadcast (row-major) — relocation of codegen.emit_broadcast row/col branches ----
    def _emit_bcast_row(self, call):                    # src[1,C] -> dst[R,C] (native copy)
        d = self.ptr(call.args[1]); s = self.ptr(call.args[2])
        R = self.ev(call.args[3]); C = self.ev(call.args[4]); a = self.a
        for r in range(R):
            a.vlen(C); a.addr(SRC1, s); a.load(0, 0); a.v_copy()
            a.addr(DST, d + r * C); a.save(0)

    def _emit_bcast_col(self, call):                    # src[R,1] -> dst[R,C] (ones-matmul: 0x15 imm can't address per-row)
        d = self.ptr(call.args[1]); s = self.ptr(call.args[2])
        Rd = self.ev(call.args[3]); Cd = self.ev(call.args[4]); a = self.a
        T = TILE
        ones = self.mp.scratch_alloc(T); sP = self.mp.scratch_alloc(T * T)
        a.vlen(T); a.v_broadcast(mode=IMM, imm=1); a.addr(DST, ones); a.save(0)
        for mi in range(0, Rd, T):
            mt = min(T, Rd - mi)
            for nj in range(0, Cd, T):
                nt = min(T, Cd - nj)
                a.tile(0, mt, 1); a.tile(1, 1, nt)
                a.addr(SRC1, s + mi); a.load(1, 0)
                a.addr(SRC2, ones); a.load(1, 1)
                a.m_mul(mode=VECTOR)
                if nt == Cd:
                    a.addr(DST, d + mi * Cd); a.save(1)
                else:
                    a.addr(DST, sP); a.save(1)
                    self._copy2d(d + mi * Cd + nj, Cd, sP, nt, mt, nt)

    # ---- data-movement (row-major): transpose / slice / concat-place ----
    def _emit_transpose_row(self, call):                # src[R,C] -> dst[C,R] (strided load 0x90)
        d0 = self.ptr(call.args[1]); s0 = self.ptr(call.args[2])
        R = self.ev(call.args[3]); C = self.ev(call.args[4]); a = self.a
        assert C < 256, (f"v2 transpose row-path: 8-bit stride field needs C<256 (got {C}); "
                         f"wider transposes need the tile-blocked path (not yet in v2.compile_module)")
        T = TILE
        scr = self.mp.scratch_alloc(T * T)
        for ci in range(0, C, T):
            ct = min(T, C - ci)
            for ri in range(0, R, T):
                rt = min(T, R - ri)
                a.tile(0, rt, C); a.addr(SRC1, s0 + ri * C + ci)
                a.load(1, 0, strided=1, ncols=ct, start=0)
                a.vlen(ct * rt); a.v_copy()
                if rt == R:
                    a.addr(DST, d0 + ci * R); a.save(0)
                else:
                    a.addr(DST, scr); a.save(0)
                    self._copy2d(d0 + ci * R + ri, R, scr, rt, ct, rt)

    def _emit_slice(self, call):                        # src[R,C][:, b:b+w] -> dst[R,w]
        d = self.ptr(call.args[1]); s = self.ptr(call.args[2])
        R = self.ev(call.args[3]); C = self.ev(call.args[4])
        b = self.ev(call.args[5]); w = self.ev(call.args[6]); a = self.a
        for r in range(R):
            a.vlen(w); a.addr(SRC1, s + r * C + b); a.load(0, 0); a.v_add(mode=IMM, imm=0)
            a.addr(DST, d + r * w); a.save(0)

    def _emit_colplace(self, call):                     # src[R,Cs] -> dst[R,Cd] at column `col`
        d = self.ptr(call.args[1]); s = self.ptr(call.args[2])
        R = self.ev(call.args[3]); Cs = self.ev(call.args[4])
        Cd = self.ev(call.args[5]); col = self.ev(call.args[6]); a = self.a
        for r in range(R):
            a.vlen(Cs); a.addr(SRC1, s + r * Cs); a.load(0, 0); a.v_add(mode=IMM, imm=0)
            a.addr(DST, d + r * Cd + col); a.save(0)

    # ---- reduce (row/tile x sum/max) — verbatim relocation of codegen.emit_row_sum/max ----
    def _emit_rsum_row(self, call):                     # src[R,C] row-major -> dst[R,1]
        sd = self.ptr(call.args[1]); ssrc = self.ptr(call.args[2])
        R = self.ev(call.args[3]); C = self.ev(call.args[4]); a = self.a
        for r in range(R):
            a.vlen(C)
            a.addr(SRC1, ssrc + r * C); a.load(0, 0)
            a.v_reduce_sum()
            a.addr(DST, sd + r); a.save(0)

    def _emit_rsum_tile(self, call):                    # src[R,C] tile-blocked -> dst[R,1]
        sd = self.ptr(call.args[1]); ssrc = self.ptr(call.args[2])
        R = self.ev(call.args[3]); C = self.ev(call.args[4]); a = self.a
        Ct = C // 64
        A = self.mp.scratch_alloc(64 * 64)
        for rt in range(R // 64):
            base = ssrc + (rt * Ct) * 4096
            a.vlen(4096); a.addr(SRC1, base); a.load(0, 0)
            a.v_copy(); a.addr(DST, A); a.save(0)
            for ct in range(1, Ct):
                a.vlen(4096); a.addr(SRC1, base + ct * 4096); a.load(0, 0)
                a.addr(SRC2, A); a.load(0, 1); a.v_add(mode=VECTOR)
                a.addr(DST, A); a.save(0)
            for ir in range(64):
                a.vlen(64); a.addr(SRC1, A + ir * 64); a.load(0, 0)
                a.v_reduce_sum(); a.addr(DST, sd + rt * 64 + ir); a.save(0)

    def _emit_rmax_row(self, call):                     # src[R,C] row-major -> dst[R,1] (R,C<256)
        d0 = self.ptr(call.args[1]); s0 = self.ptr(call.args[2])
        R = self.ev(call.args[3]); C = self.ev(call.args[4]); a = self.a
        assert R < 256 and C < 256, (f"v2 reduce-max row-path: 8-bit tile field needs R,C<256 (got "
                                     f"{R}x{C}); >=256 (e.g. softmax at SEQ>=256) needs the tile-blocked "
                                     f"path (not yet in v2.compile_module)")
        acc = self.mp.scratch_alloc(R)
        a.tile(0, R, C); a.addr(SRC1, s0)
        a.load(1, 0, strided=1, ncols=1, start=0)
        a.vlen(R); a.v_copy(); a.addr(DST, acc); a.save(0)
        for j in range(1, C):
            a.tile(0, R, C); a.addr(SRC1, s0)
            a.load(1, 0, strided=1, ncols=1, start=j)
            a.vlen(R); a.addr(SRC2, acc); a.load(0, 1)
            a.v_max(mode=VECTOR)
            a.addr(DST, acc); a.save(0)
        a.vlen(R); a.addr(SRC1, acc); a.load(0, 0)
        a.v_copy(); a.addr(DST, d0); a.save(0)

    def _emit_rmax_tile(self, call):                    # src[R,C] tile-blocked -> dst[R,1]
        d0 = self.ptr(call.args[1]); s0 = self.ptr(call.args[2])
        R = self.ev(call.args[3]); C = self.ev(call.args[4]); a = self.a
        Ct = C // 64
        B = self.mp.scratch_alloc(64 * 64)
        ac = self.mp.scratch_alloc(64)
        for rt in range(R // 64):
            base = s0 + (rt * Ct) * 4096
            a.vlen(4096); a.addr(SRC1, base); a.load(0, 0)
            a.v_copy(); a.addr(DST, B); a.save(0)
            for ct in range(1, Ct):
                a.vlen(4096); a.addr(SRC1, base + ct * 4096); a.load(0, 0)
                a.addr(SRC2, B); a.load(0, 1); a.v_max(mode=VECTOR)
                a.addr(DST, B); a.save(0)
            a.tile(0, 64, 64); a.addr(SRC1, B)
            a.load(1, 0, strided=1, ncols=1, start=0)
            a.vlen(64); a.v_copy(); a.addr(DST, ac); a.save(0)
            for j in range(1, 64):
                a.tile(0, 64, 64); a.addr(SRC1, B)
                a.load(1, 0, strided=1, ncols=1, start=j)
                a.vlen(64); a.addr(SRC2, ac); a.load(0, 1); a.v_max(mode=VECTOR)
                a.addr(DST, ac); a.save(0)
            a.vlen(64); a.addr(SRC1, ac); a.load(0, 0)
            a.v_copy(); a.addr(DST, d0 + rt * 64); a.save(0)


# ---- marker PrimFunc builders (stand-ins for Phase-3 legalization) ----
def ew2_marker(op, N):
    intrin = "npu_ew2_" + op

    @T.prim_func
    def f(x0: T.handle, x1: T.handle, y: T.handle):
        A = T.match_buffer(x0, (N,), "float16")
        B = T.match_buffer(x1, (N,), "float16")
        C = T.match_buffer(y, (N,), "float16")
        with T.block("root"):
            T.reads(A[0:N], B[0:N]); T.writes(C[0:N])
            T.evaluate(T.call_extern("int32", intrin,
                                     C.access_ptr("w"), A.access_ptr("r"), B.access_ptr("r"), N))
    return f


def ew1_marker(op, N):
    intrin = "npu_silu" if op == "silu" else "npu_ew1_" + op

    @T.prim_func
    def f(x0: T.handle, y: T.handle):
        A = T.match_buffer(x0, (N,), "float16")
        C = T.match_buffer(y, (N,), "float16")
        with T.block("root"):
            T.reads(A[0:N]); T.writes(C[0:N])
            T.evaluate(T.call_extern("int32", intrin, C.access_ptr("w"), A.access_ptr("r"), N))
    return f


def copy_marker(n):
    @T.prim_func
    def f(x0: T.handle, y: T.handle):
        A = T.match_buffer(x0, (n,), "float16")
        C = T.match_buffer(y, (n,), "float16")
        with T.block("root"):
            T.reads(A[0:n]); T.writes(C[0:n])
            T.evaluate(T.call_extern("int32", "npu_copy", C.access_ptr("w"), A.access_ptr("r"), n))
    return f


def ttile_marker():
    @T.prim_func
    def f(x0: T.handle, y: T.handle):
        A = T.match_buffer(x0, (4096,), "float16")
        C = T.match_buffer(y, (4096,), "float16")
        with T.block("root"):
            T.reads(A[0:4096]); T.writes(C[0:4096])
            T.evaluate(T.call_extern("int32", "npu_ttile", C.access_ptr("w"), A.access_ptr("r")))
    return f


def reduce_marker(intrin, R, C, src_numel):
    @T.prim_func
    def f(s: T.handle, d: T.handle):
        S = T.match_buffer(s, (src_numel,), "float16")
        D = T.match_buffer(d, (R,), "float16")
        with T.block("root"):
            T.reads(S[0:src_numel]); T.writes(D[0:R])
            T.evaluate(T.call_extern("int32", intrin, D.access_ptr("w"), S.access_ptr("r"), R, C))
    return f


def bcast_marker(intrin, Rd, Cd, src_numel):
    @T.prim_func
    def f(s: T.handle, d: T.handle):
        S = T.match_buffer(s, (src_numel,), "float16")
        D = T.match_buffer(d, (Rd * Cd,), "float16")
        with T.block("root"):
            T.reads(S[0:src_numel]); T.writes(D[0:Rd * Cd])
            T.evaluate(T.call_extern("int32", intrin, D.access_ptr("w"), S.access_ptr("r"), Rd, Cd))
    return f


def transpose_marker(R, C):
    @T.prim_func
    def f(s: T.handle, d: T.handle):
        S = T.match_buffer(s, (R * C,), "float16")
        D = T.match_buffer(d, (R * C,), "float16")
        with T.block("root"):
            T.reads(S[0:R * C]); T.writes(D[0:R * C])
            T.evaluate(T.call_extern("int32", "npu_transpose", D.access_ptr("w"), S.access_ptr("r"), R, C))
    return f


def slice_marker(R, C, b, w):
    @T.prim_func
    def f(s: T.handle, d: T.handle):
        S = T.match_buffer(s, (R * C,), "float16")
        D = T.match_buffer(d, (R * w,), "float16")
        with T.block("root"):
            T.reads(S[0:R * C]); T.writes(D[0:R * w])
            T.evaluate(T.call_extern("int32", "npu_slice",
                                     D.access_ptr("w"), S.access_ptr("r"), R, C, b, w))
    return f


def colplace_marker(R, Cs, Cd, col):
    @T.prim_func
    def f(s: T.handle, d: T.handle):
        S = T.match_buffer(s, (R * Cs,), "float16")
        D = T.match_buffer(d, (R * Cd,), "float16")
        with T.block("root"):
            T.reads(S[0:R * Cs]); T.writes(D[0:R * Cd])
            T.evaluate(T.call_extern("int32", "npu_colplace",
                                     D.access_ptr("w"), S.access_ptr("r"), R, Cs, Cd, col))
    return f


def _slice_last_begin(pf):
    """Recover the last-axis begin from a legalized strided_slice PrimFunc: the input
    load's last index is (spatial_var + begin), or just spatial_var when begin == 0."""
    res = [0]

    def visit(s):
        if isinstance(s, tir.BufferLoad) and len(s.indices):
            idx = s.indices[-1]
            if isinstance(idx, tir.Add):
                for part in (idx.a, idx.b):
                    if isinstance(part, tir.IntImm):
                        res[0] = max(res[0], int(part))
    tir.stmt_functor.post_order_visit(pf.body, visit)
    return res[0]


def walk_marker(asm, pf, offsets, mp=None):
    """Lower a marker PrimFunc into `asm` with each param bound to a G-buffer offset.
    mp: MemPlan providing scratch_alloc (needed by reduce tile/row-max paths)."""
    wk = V2Walker(asm, mp, {})
    for p, off in zip(pf.params, offsets):
        wk.base[pf.buffer_map[p].data] = off
    wk.walk(pf.body)
    wk.flush()


# ==== REAL pipeline: native vector-op tensor intrinsics (tensorize legalized ew TIR) ====
# desc = CH-elem compute (matches the legalized elementwise block); impl = the walker's
# call_extern marker. Split the legalized loop by CH + tensorize -> the marker -> walker.
@T.prim_func
def _vadd_desc(a: T.handle, b: T.handle, c: T.handle):
    A = T.match_buffer(a, (CH,), "float16", offset_factor=1)
    B = T.match_buffer(b, (CH,), "float16", offset_factor=1)
    C = T.match_buffer(c, (CH,), "float16", offset_factor=1)
    with T.block("root"):
        T.reads(A[0:CH], B[0:CH]); T.writes(C[0:CH])
        for i in range(CH):
            with T.block("u"):
                vi = T.axis.remap("S", [i]); C[vi] = A[vi] + B[vi]


@T.prim_func
def _vadd_impl(a: T.handle, b: T.handle, c: T.handle):
    sa = T.int32(); sb = T.int32(); sc = T.int32()
    A = T.match_buffer(a, (CH,), "float16", strides=[sa], offset_factor=1)
    B = T.match_buffer(b, (CH,), "float16", strides=[sb], offset_factor=1)
    C = T.match_buffer(c, (CH,), "float16", strides=[sc], offset_factor=1)
    with T.block("root"):
        T.reads(A[0:CH], B[0:CH]); T.writes(C[0:CH])
        T.evaluate(T.call_extern("int32", "npu_ew2_add",
                                 C.access_ptr("w"), A.access_ptr("r"), B.access_ptr("r"), CH))


def _register_v2():
    try:
        tir.TensorIntrin.register("npu_vadd", _vadd_desc, _vadd_impl)
    except Exception:
        pass


_register_v2()


def schedule_ew_binary(mod, func_name, block_name, intrin):
    """Fuse the legalized elementwise loops, split by CH, tensorize the inner block to
    a native vector intrinsic. Requires numel % CH == 0 (remainder handling = later)."""
    sch = tir.Schedule(mod)
    blk = sch.get_block(block_name, func_name=func_name)
    loops = sch.get_loops(blk)
    fused = sch.fuse(*loops) if len(loops) > 1 else loops[0]
    _, ii = sch.split(fused, [None, CH])
    sch.tensorize(ii, intrin)
    return sch.mod


# ==== v2 compile orchestration: legalized Relax IRModule -> NPU ISA ====
def _numel(sh):
    n = 1
    for d in sh:
        n *= d
    return n


def _first_block(pf):
    """Name of the first non-root TIR block (the op's compute block)."""
    names = []
    tir.stmt_functor.post_order_visit(
        pf.body, lambda s: names.append(s.name_hint)
        if isinstance(s, tir.Block) and s.name_hint != "root" else None)
    return names[0]


class _Op:
    """One legalized op in emission order (a decoded `call_tir` binding)."""
    __slots__ = ("opname", "gvar", "ins", "outn", "begin")

    def __init__(self, opname, gvar, ins, outn, begin=0):
        self.opname = opname; self.gvar = gvar
        self.ins = ins; self.outn = outn; self.begin = begin


def _parse(mod):
    """Pass F5-read: legalized Relax "main" (a call_tir sequence) -> structure only.
    Returns (params, ops, shp{key->shape}, const_arrs{key->ndarray}, out_name). Constants
    are keyed by (binding, arg) POSITION — TVM re-wraps FFI nodes so id() is unstable."""
    main = mod["main"]
    shp, params = {}, []
    for p in main.params:
        shp[p.name_hint] = [int(d) for d in p.struct_info.shape]
        params.append(p.name_hint)
    binds = [bd for blk in main.body.blocks for bd in blk.bindings
             if isinstance(bd.value, _relax.Call)
             and getattr(bd.value.op, "name", "") == "relax.call_tir"]
    const_arrs, ops = {}, []
    for i, bd in enumerate(binds):
        val = bd.value
        gvar = val.args[0].name_hint
        opname = gvar.rstrip("0123456789")                   # "matmul1" -> "matmul"
        if opname.startswith("tir_"):                        # "tir_sqrt" -> "sqrt"
            opname = opname[4:]
        ins = []
        for j, a in enumerate(val.args[1].fields):
            if isinstance(a, _relax.Constant):
                key = "__const_%d_%d" % (i, j)
                arr = a.data.numpy()
                const_arrs[key] = arr; shp[key] = list(arr.shape); ins.append(key)
            else:
                ins.append(a.name_hint)
        outn = bd.var.name_hint
        shp[outn] = [int(d) for d in bd.var.struct_info.shape]
        begin = _slice_last_begin(mod[gvar]) if opname == "strided_slice" else 0
        ops.append(_Op(opname, gvar, ins, outn, begin))
    out = main.body.body                                     # resolve returned var thru aliases
    val_of = {bd.var: bd.value for blk in main.body.blocks for bd in blk.bindings}
    while isinstance(val_of.get(out), _relax.Var):
        out = val_of[out]
    return params, ops, shp, const_arrs, out.name_hint


def _plan_memory(params, ops, shp, const_arrs, out_name, reuse=True):
    """Pass M1: assign a flat G-buffer offset to every tensor.

    Params are all live at t=0 (host loads them up-front) and constants are materialized
    by legalize -> both persist for the whole program (never reused). reuse=True adds v1's
    A1 optimization for INTERMEDIATES: liveness (last read per tensor) + an exact-size
    free-list, so a tensor's slot is handed to a later same-size tensor once it's dead.
    Value-invariant (relocates data only) -> byte-exact vs reuse=False; only `top` shrinks.
    Returns (off, top, const_inits, no_reuse_top)."""
    off, top = {}, [0]

    def bump(nm):
        off[nm] = top[0]; top[0] += _numel(shp[nm])

    for p in params:
        bump(p)
    const_keys = list(const_arrs)
    for k in const_keys:
        bump(k)
    keep = set(params) | set(const_keys) | {out_name}        # never freed (persist / output)
    last_read = {}                                            # last op index reading each tensor
    for i, op in enumerate(ops):
        for k in op.ins:
            last_read[k] = i
    no_reuse_top = top[0] + sum(_numel(shp[op.outn]) for op in ops)   # bump-only footprint
    free = {}                                                # footprint -> [freed offsets]
    for i, op in enumerate(ops):
        sz = _numel(shp[op.outn])                            # allocate output BEFORE freeing
        lst = free.get(sz) if reuse else None                # inputs (no self-alias in one op)
        off[op.outn] = lst.pop() if lst else top[0]
        if not lst:
            top[0] += sz
        if reuse:
            for k in dict.fromkeys(op.ins):                  # dedup: free each dead input once
                if last_read.get(k) == i and k not in keep:
                    free.setdefault(_numel(shp[k]), []).append(off[k])
    const_inits = [(off[k], const_arrs[k]) for k in const_keys]
    return off, top[0], const_inits, no_reuse_top


def _emit(ops, off, shp, top):
    """Pass C1: lower each op to NPU ISA via the unified walker (markers) / v1 matmul.
    `mp` supplies codegen-internal scratch above the planned `top`."""
    mp = _memplan.MemPlan(); mp.top = top
    asm = Asm()
    _EW1_OPS = set(_EW1) | {"silu"}
    for op in ops:
        opname, ins, outn = op.opname, op.ins, op.outn
        if opname == "matmul":                               # matmul -> v1 proven path
            M, K = shp[ins[0]]; _, N = shp[ins[1]]
            emit_matmul_into(asm, mp, off[outn], off[ins[0]], off[ins[1]], M, K, N)
        elif opname in _EW2:                                 # binary elementwise -> marker
            N = _numel(shp[outn])                            # operands must be full output size
            assert _numel(shp[ins[0]]) == N and _numel(shp[ins[1]]) == N, \
                f"v2 {opname}: broadcast operand (size != output) unsupported; legalize must materialize full-size"
            walk_marker(asm, ew2_marker(opname, N), [off[ins[0]], off[ins[1]], off[outn]], mp)
        elif opname in _EW1_OPS:                             # unary elementwise -> marker
            N = _numel(shp[outn])
            assert _numel(shp[ins[0]]) == N, f"v2 {opname}: operand size != output"
            walk_marker(asm, ew1_marker(opname, N), [off[ins[0]], off[outn]], mp)
        elif opname in ("sum", "max"):                       # reduce over LAST axis -> [R,1]
            assert len(shp[ins[0]]) == 2 and shp[outn][-1] == 1, \
                f"v2 {opname}: only last-axis 2D reduce (keepdims), got {shp[ins[0]]}->{shp[outn]}"
            R, C = shp[ins[0]]
            walk_marker(asm, reduce_marker("npu_rsum_row" if opname == "sum" else "npu_rmax_row",
                                           R, C, R * C), [off[ins[0]], off[outn]], mp)
        elif opname == "broadcast_to":                       # [R,1]->col or [1,C]->row (reject scalar/other)
            Rd, Cd = shp[outn]; si = list(shp[ins[0]]) + [1, 1]
            sr, sc = si[0], si[1]
            col = sc == 1 and sr == Rd
            row = sr == 1 and sc == Cd
            assert col or row, f"v2 broadcast_to: only [R,1]->[R,C] or [1,C]->[R,C], got {shp[ins[0]]}->{shp[outn]}"
            walk_marker(asm, bcast_marker("npu_bcast_col" if col else "npu_bcast_row",
                                          Rd, Cd, Rd if col else Cd), [off[ins[0]], off[outn]], mp)
        elif opname == "transpose":                          # permute_dims [1,0]: [R,C]->[C,R]
            R, C = shp[ins[0]]
            walk_marker(asm, transpose_marker(R, C), [off[ins[0]], off[outn]], mp)
        elif opname == "strided_slice":                      # last-axis x[:, b:b+w] only
            R, C = shp[ins[0]]
            assert len(shp[outn]) == 2 and shp[outn][0] == R, \
                f"v2 strided_slice: only last-axis (rows unchanged), got {shp[ins[0]]}->{shp[outn]}"
            w = shp[outn][1]
            walk_marker(asm, slice_marker(R, C, op.begin, w), [off[ins[0]], off[outn]], mp)
        elif opname == "concatenate":                        # last-axis concat -> place columns
            Rd, Cd = shp[outn]; ccol = 0
            assert all(shp[ins[k]][0] == Rd for k in range(len(ins))), \
                "v2 concatenate: only last-axis (all inputs share row count)"
            for k in range(len(ins)):
                Rk, Ck = shp[ins[k]]
                walk_marker(asm, colplace_marker(Rk, Ck, Cd, ccol), [off[ins[k]], off[outn]], mp)
                ccol += Ck
        else:
            raise TirBackendError(f"v2.compile_module: op '{opname}' not handled yet")
    return asm


def compile_module(mod, reuse=True):
    """Compile a `LegalizeOps`'d Relax IRModule ("main" = call_tir sequence) to NPU ISA
    through an explicit 3-pass pipeline:
      _parse   (F5-read) : legalized Relax -> ops + tensor table + constants
      _plan_memory (M1)  : liveness + exact-size free-list -> offsets (A1 reuse)
      _emit    (C1)      : unified TIR->ISA lowering (walker markers / v1 matmul)
    reuse=False disables A1 (flat bump) — used to measure the reuse win.

    Returns (asm, offsets{name->off}, shapes{name->shape}, top, out_name, const_inits)."""
    params, ops, shp, const_arrs, out_name = _parse(mod)
    off, top, const_inits, _ = _plan_memory(params, ops, shp, const_arrs, out_name, reuse)
    asm = _emit(ops, off, shp, top)
    return asm, off, shp, top, out_name, const_inits
