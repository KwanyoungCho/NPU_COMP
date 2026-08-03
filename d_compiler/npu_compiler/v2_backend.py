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
from .tir_backend import _Walker, TILE, TirBackendError, emit_matmul_into, emit_matmul_accumulate_group
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
            if name == "npu_bcast_row_tile":
                return self._emit_bcast_row_tile(call)
            if name == "npu_bcast_col_tile":
                return self._emit_bcast_col_tile(call)
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

    # ---- broadcast (tile-blocked dst, A4) — relocation of codegen.emit_broadcast tile branch ----
    def _emit_bcast_col_tile(self, call):               # src[R,1] row -> tile dst[R,C] (ones-matmul per row-tile)
        d = self.ptr(call.args[1]); s = self.ptr(call.args[2])
        Rd = self.ev(call.args[3]); Cd = self.ev(call.args[4]); a = self.a
        Rt, Ct = Rd // 64, Cd // 64
        B = self.mp.scratch_alloc(64 * 64); ones = self.mp.scratch_alloc(64)
        a.vlen(64); a.v_broadcast(mode=IMM, imm=1); a.addr(DST, ones); a.save(0)
        for rt in range(Rt):
            a.tile(0, 64, 1); a.tile(1, 1, 64)
            a.addr(SRC1, s + rt * 64); a.load(1, 0)         # src col [64,1]
            a.addr(SRC2, ones); a.load(1, 1)                # ones [1,64]
            a.m_mul(mode=VECTOR); a.addr(DST, B); a.save(1)  # B[ir,:]=src[rt*64+ir]
            for ct in range(Ct):
                a.vlen(4096); a.addr(SRC1, B); a.load(0, 0)
                a.v_copy(); a.addr(DST, d + (rt * Ct + ct) * 4096); a.save(0)

    def _emit_bcast_row_tile(self, call):               # src[1,C] row -> tile dst[R,C]
        d = self.ptr(call.args[1]); s = self.ptr(call.args[2])
        Rd = self.ev(call.args[3]); Cd = self.ev(call.args[4]); a = self.a
        Rt, Ct = Rd // 64, Cd // 64
        B = self.mp.scratch_alloc(64 * 64)
        for ct in range(Ct):
            seg = s + ct * 64                               # src[1,C] contiguous 64-segment
            for ir in range(64):                            # B[ir,:] = seg
                a.vlen(64); a.addr(SRC1, seg); a.load(0, 0)
                a.v_copy(); a.addr(DST, B + ir * 64); a.save(0)
            for rt in range(Rt):
                a.vlen(4096); a.addr(SRC1, B); a.load(0, 0)
                a.v_copy(); a.addr(DST, d + (rt * Ct + ct) * 4096); a.save(0)

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


# op families that are layout-transparent elementwise (tile iff all inputs tile)
_EW_LAYOUT = {"add", "subtract", "multiply", "divide",
              "silu", "exp", "sqrt", "negative", "cos", "sin"}


def _assign_layouts(params, ops, shp, const_arrs, out_name, tile_inputs=True):
    """Pass F4: decide 'row'/'tile' per tensor (port of memplan.assign_layouts to the v2
    legalized op list). A tensor seeds 'tile' if its producer is tile-capable, then a
    demotion fixpoint drops it to 'row' unless EVERY consumer is tile-compatible and (for
    elementwise/transpose/slice/concat producers) its own inputs are tile. The invariant
    (tile tensors only feed tile consumers) means NO explicit relayout is needed at
    boundaries — a matmul reads a row operand via its own gather, a tile operand skips it.
    Also returns packed_params{name->Nt}: 64-mult matmul-weight PARAMS read pre-packed."""
    param_set, const_set = set(params), set(const_arrs)
    prod = {op.outn: op for op in ops}                       # producer op of each binding var

    def _is64_2d(key):
        sh = shp[key]
        return len(sh) == 2 and sh[0] % 64 == 0 and sh[1] % 64 == 0

    def is_mm64(key):
        op = prod.get(key)
        if op is None or op.opname != "matmul":
            return False
        M, K = shp[op.ins[0]]; _, N = shp[op.ins[1]]
        return M % 64 == 0 and K % 64 == 0 and N % 64 == 0

    def is_ew(key):
        op = prod.get(key); return op is not None and op.opname in _EW_LAYOUT

    def is_bcast(key):
        op = prod.get(key); return op is not None and op.opname == "broadcast_to" and _is64_2d(key)

    # NOTE: transpose/strided_slice/concatenate are ROW-ONLY in v2's _emit, so they are
    # never seeded 'tile' and never count as tile-compatible consumers below (a tile tensor
    # feeding one is demoted to row). This keeps the RoPE chain row at ANY head dim — at
    # HD>=128 the rotate-half slice is 64-wide/64-aligned, which a tile-seeding rule would
    # wrongly mark tile and then crash the row-only emit (adversarial review V2-019).

    def _reduce_src64(key, name):
        op = prod.get(key)
        if op is None or op.opname != name:
            return False
        sh = shp[op.ins[0]]
        return len(sh) == 2 and sh[0] % 64 == 0 and sh[1] % 64 == 0

    is_reduce = lambda k: _reduce_src64(k, "sum")
    is_max = lambda k: _reduce_src64(k, "max")

    consumers = {}                                           # key -> list of consumer ops
    for op in ops:
        for k in op.ins:
            consumers.setdefault(k, []).append(op)

    layout = {}
    for op in ops:
        v = op.outn
        layout[v] = "tile" if (is_mm64(v) or is_ew(v) or is_bcast(v)) else "row"
    for k in param_set | const_set:
        layout.setdefault(k, "row")
    tile_params = set()
    if tile_inputs:                                          # activation input params fed pre-packed (A4 5d)
        for op in ops:
            if op.opname == "matmul" and op.ins[0] in param_set:
                tile_params.add(op.ins[0])
            elif op.opname in _EW_LAYOUT or op.opname == "sum":
                tile_params |= {k for k in op.ins if k in param_set}
        tile_params = {p for p in tile_params if _is64_2d(p)}
        for p in tile_params:
            layout[p] = "tile"
    layout[out_name] = "row"
    nodes = [op.outn for op in ops] + list(tile_params)

    changed = True
    while changed:
        changed = False
        for v in nodes:
            if layout.get(v) != "tile":
                continue
            demote = (v == out_name)
            op = prod.get(v)
            if not demote and is_ew(v):                      # tile ew: every input tile (or scalar/packable const)
                for k in op.ins:
                    if k in const_set:
                        csh = shp[k]; n = 1
                        for dd in csh:
                            n *= dd
                        if n == 1 or (len(csh) == 2 and csh[0] % 64 == 0 and csh[1] % 64 == 0):
                            continue                         # scalar (layout-agnostic) / 2D 64-mult (packed tile)
                        demote = True; break
                    if layout.get(k, "row") != "tile":
                        demote = True; break
            if not demote:
                for cv in consumers.get(v, []):              # every consumer tile-compatible
                    if cv.opname == "matmul" and cv.ins[0] == v and is_mm64(cv.outn):
                        continue                             # read as A by a 64-mult matmul
                    if cv.opname == "matmul" and cv.ins[1] == v and is_mm64(cv.outn):
                        continue                             # read as B (tile activation / weight)
                    if is_ew(cv.outn) and layout.get(cv.outn) == "tile":
                        continue                             # feeds a tile elementwise
                    if is_reduce(cv.outn) or is_max(cv.outn):
                        continue                             # feeds a tile-native reduce (reads tile src)
                    demote = True; break                     # transpose/slice/concat row-only in v2 -> demote
            if demote:
                layout[v] = "row"; changed = True

    # a 2D-64-mult CONSTANT feeding a tile elementwise must be stored tile-blocked too
    # (else the layout-transparent ew reads a row-major const in tile order) — v1's
    # alloc_const_tiled. Mark it tile so _plan_memory packs it (pack_tiled).
    for op in ops:
        if op.opname in _EW_LAYOUT and layout.get(op.outn) == "tile":
            for k in op.ins:
                if k in const_set and len(shp[k]) == 2 and shp[k][0] % 64 == 0 and shp[k][1] % 64 == 0:
                    layout[k] = "tile"

    packed_params = {}                                       # 64-mult matmul-weight params -> Nt (pre-packed B)
    for op in ops:
        if op.opname == "matmul" and op.ins[1] in param_set and _is64_2d(op.ins[1]):
            packed_params[op.ins[1]] = shp[op.ins[1]][1] // 64
    ok = {k: True for k in packed_params}                    # pack only if used EXCLUSIVELY as matmul B
    for op in ops:
        for i, k in enumerate(op.ins):
            if k in ok and not (op.opname == "matmul" and i == 1):
                ok[k] = False
    packed_params = {k: nt for k, nt in packed_params.items() if ok[k]}
    return layout, packed_params


def _detect_oproj_groups(ops, shp):
    """Pass F3 (fusion): find maximal add-trees whose leaves are all same-shape 64-mult
    matmuls — the per-head O-projection attn = sum_h (ctx_h @ Wo_h). Port of v1
    codegen._detect_oproj_groups to the v2 op list. Returns (fused_root{root_key ->
    [leaf matmul keys in fold order]}, consumed{every key folded away: tree adds + leaves}).
    Only fuses when each folded node is single-use, so skipping its emit is safe (the root
    var is still produced, now by one in-place accumulate group instead of H scatters)."""
    prod = {op.outn: op for op in ops}
    uses = {}
    for op in ops:
        for k in op.ins:
            uses[k] = uses.get(k, 0) + 1

    def _op(key, opn):
        op = prod.get(key)
        return op if (op is not None and op.opname == opn) else None

    def _leaves(key, shape):
        ad = _op(key, "add")
        if ad is not None:
            l = _leaves(ad.ins[0], shape); r = _leaves(ad.ins[1], shape)
            return None if (l is None or r is None) else (l + r)
        mm = _op(key, "matmul")
        return [key] if (mm is not None and tuple(shp[key]) == shape) else None

    def _legal_term(key):                                   # leaf matmul with 64-mult K, N
        mm = prod[key]
        return shp[mm.ins[0]][1] % 64 == 0 and shp[mm.ins[1]][1] % 64 == 0

    cand = {}
    for op in ops:
        if op.opname != "add":
            continue
        shape = tuple(shp[op.outn])
        lv = _leaves(op.outn, shape)
        if lv is not None and len(lv) >= 2 and all(_legal_term(x) for x in lv):
            cand[op.outn] = lv
    used_by_cand = set()
    for v in cand:
        ad = prod[v]
        for ar in (ad.ins[0], ad.ins[1]):
            if ar in cand:
                used_by_cand.add(ar)
    fused_root, consumed = {}, set()
    for v, leaves in cand.items():
        if v in used_by_cand:                               # not the maximal root
            continue
        nodes, stack, ok = set(), [v], True
        while stack:                                        # collect tree adds + leaves
            x = stack.pop(); nodes.add(x)
            ad = _op(x, "add")
            if ad is not None:
                stack += [ad.ins[0], ad.ins[1]]
        for x in nodes:                                     # every folded node except root single-use
            if x != v and uses.get(x, 0) != 1:
                ok = False; break
        if ok:
            fused_root[v] = leaves
            consumed |= nodes
    return fused_root, consumed


def _footprint(key, shp, layout):
    return _memplan.tiled_numel(shp[key]) if layout.get(key) == "tile" else _numel(shp[key])


def _plan_memory(params, ops, shp, const_arrs, out_name, layout, packed_params,
                 fused_root, consumed, reuse=True):
    """Pass M1: assign a flat G-buffer offset to every tensor.

    Params are all live at t=0 (host loads them up-front) and constants are materialized
    by legalize -> both persist for the whole program (never reused). reuse=True adds v1's
    A1 optimization for INTERMEDIATES: liveness (last read per tensor) + an exact-size
    free-list, so a tensor's slot is handed to a later same-size tensor once it's dead.
    Value-invariant (relocates data only) -> byte-exact vs reuse=False; only `top` shrinks.
    Tile tensors get a tile-blocked footprint (tiled_numel, == numel for 64-mult dims);
    tile elementwise-operand constants are stored pre-packed (pack_tiled).

    Fusion-aware (F3): an O-proj group's folded nodes (leaf matmuls + tree adds, except the
    root) are NOT materialized (no slot); the group reads every leaf's inputs (ctx_h, Wo_h)
    at the ROOT emit position, so those inputs stay live until the group is emitted. Returns
    (off, top, const_inits, no_reuse_top, tiled_feed)."""
    off, top = {}, [0]
    fp = lambda k: _footprint(k, shp, layout)
    prod = {op.outn: op for op in ops}

    def bump(nm):
        off[nm] = top[0]; top[0] += fp(nm)

    for p in params:
        bump(p)
    const_keys = list(const_arrs)
    for k in const_keys:
        bump(k)
    keep = set(params) | set(const_keys) | {out_name}        # never freed (persist / output)
    # emission steps: (produced_key, [read_keys]) skipping folded non-root nodes; a group
    # root produces the summed result and reads every leaf matmul's A,B at its position.
    steps = []
    for op in ops:
        if op.outn in consumed and op.outn not in fused_root:
            continue                                         # folded away -> not materialized
        if op.outn in fused_root:
            reads = []
            for leaf in fused_root[op.outn]:
                reads += [prod[leaf].ins[0], prod[leaf].ins[1]]
            steps.append((op.outn, reads))
        else:
            steps.append((op.outn, list(op.ins)))
    last_read = {}                                           # last emit step reading each tensor
    for i, (_, reads) in enumerate(steps):
        for k in reads:
            last_read[k] = i
    no_reuse_top = top[0] + sum(fp(pk) for pk, _ in steps)   # bump-only footprint
    free = {}                                                # footprint -> [freed offsets]
    for i, (pk, reads) in enumerate(steps):
        sz = fp(pk)                                          # allocate output BEFORE freeing
        lst = free.get(sz) if reuse else None                # inputs (no self-alias in one step)
        off[pk] = lst.pop() if lst else top[0]
        if not lst:
            top[0] += sz
        if reuse:
            for k in dict.fromkeys(reads):                   # dedup: free each dead input once
                if last_read.get(k) == i and k not in keep:
                    free.setdefault(fp(k), []).append(off[k])
    const_inits = []                                         # tile ew-operand consts stored pre-packed
    for k in const_keys:
        arr = const_arrs[k]
        if layout.get(k) == "tile":
            arr = _memplan.pack_tiled(arr, 64)
        const_inits.append((off[k], arr))
    tiled_feed = set(packed_params) | {p for p in params if layout.get(p) == "tile"}
    return off, top[0], const_inits, no_reuse_top, tiled_feed


def _emit(ops, off, shp, top, layout, packed_params, fused_root, consumed):
    """Pass C1: lower each op to NPU ISA via the unified walker (markers) / v1 matmul,
    routing to a ROW or TILE path per the layout map. An O-proj group (F3) is emitted once
    at its root as an in-place accumulate group; its folded nodes are skipped. `mp` supplies
    codegen-internal scratch above the planned `top`."""
    mp = _memplan.MemPlan(); mp.top = top
    asm = Asm()
    prod = {op.outn: op for op in ops}
    _EW1_OPS = set(_EW1) | {"silu"}
    lay = lambda k: layout.get(k, "row")
    cnt = lambda k: _memplan.tiled_numel(shp[k]) if lay(k) == "tile" else _numel(shp[k])
    for op in ops:
        opname, ins, outn = op.opname, op.ins, op.outn
        if outn in consumed and outn not in fused_root:      # folded into an O-proj group
            continue
        if outn in fused_root:                               # emit the per-head O-proj as ONE group
            terms = []                                       # (a_off, b_off, M, K, N, b_pack_nt, a_tiled)
            for leaf in fused_root[outn]:
                A, B = prod[leaf].ins[0], prod[leaf].ins[1]
                M, K = shp[A]; _, N = shp[B]
                a_tiled = (K // 64) if (lay(A) == "tile" and M % 64 == 0 and K % 64 == 0) else None
                nt = (N // 64) if (lay(B) == "tile" or B in packed_params) else None
                terms.append((off[A], off[B], M, K, N, nt, a_tiled))
            M, N = terms[0][2], terms[0][4]
            c_tiled = (N // 64) if (lay(outn) == "tile" and M % 64 == 0 and N % 64 == 0) else None
            emit_matmul_accumulate_group(asm, mp, off[outn], terms, c_tiled=c_tiled)
            continue
        if opname == "matmul":                               # v1 matmul, tile flags per operand layout
            M, K = shp[ins[0]]; _, N = shp[ins[1]]
            a_tiled = lay(ins[0]) == "tile"
            c_tiled = lay(outn) == "tile"
            b_pack_nt = (N // 64) if (lay(ins[1]) == "tile" or ins[1] in packed_params) else None
            emit_matmul_into(asm, mp, off[outn], off[ins[0]], off[ins[1]], M, K, N,
                             b_pack_nt=b_pack_nt, a_tiled=a_tiled, c_tiled=c_tiled)
        elif opname in _EW2:                                 # binary elementwise (layout-transparent)
            n = cnt(outn)
            assert cnt(ins[0]) == n and cnt(ins[1]) == n, \
                f"v2 {opname}: operand size != output ({cnt(ins[0])},{cnt(ins[1])} vs {n})"
            walk_marker(asm, ew2_marker(opname, n), [off[ins[0]], off[ins[1]], off[outn]], mp)
        elif opname in _EW1_OPS:                             # unary elementwise
            n = cnt(outn)
            assert cnt(ins[0]) == n, f"v2 {opname}: operand size != output"
            walk_marker(asm, ew1_marker(opname, n), [off[ins[0]], off[outn]], mp)
        elif opname in ("sum", "max"):                       # reduce over LAST axis -> [R,1]
            assert len(shp[ins[0]]) == 2 and shp[outn][-1] == 1, \
                f"v2 {opname}: only last-axis 2D reduce (keepdims), got {shp[ins[0]]}->{shp[outn]}"
            R, C = shp[ins[0]]
            if lay(ins[0]) == "tile":                        # tile src: cross-tile fold (no 256 field limit)
                intrin = "npu_rsum_tile" if opname == "sum" else "npu_rmax_tile"
                srcn = _memplan.tiled_numel(shp[ins[0]])
            else:
                intrin = "npu_rsum_row" if opname == "sum" else "npu_rmax_row"
                srcn = R * C
            walk_marker(asm, reduce_marker(intrin, R, C, srcn), [off[ins[0]], off[outn]], mp)
        elif opname == "broadcast_to":                       # [R,1]->col or [1,C]->row (reject scalar/other)
            Rd, Cd = shp[outn]; si = list(shp[ins[0]]) + [1, 1]
            sr, sc = si[0], si[1]
            col = sc == 1 and sr == Rd
            row = sr == 1 and sc == Cd
            assert col or row, f"v2 broadcast_to: only [R,1]->[R,C] or [1,C]->[R,C], got {shp[ins[0]]}->{shp[outn]}"
            if lay(outn) == "tile":                          # tile dst (softmax max/sum -> tile island)
                intrin = "npu_bcast_col_tile" if col else "npu_bcast_row_tile"
            else:
                intrin = "npu_bcast_col" if col else "npu_bcast_row"
            walk_marker(asm, bcast_marker(intrin, Rd, Cd, Rd if col else Cd), [off[ins[0]], off[outn]], mp)
        elif opname == "transpose":                          # permute_dims [1,0]: [R,C]->[C,R]
            assert lay(ins[0]) != "tile" and lay(outn) != "tile", \
                "v2 transpose is row-only; F4 fixpoint must keep it row (see _assign_layouts)"
            R, C = shp[ins[0]]
            walk_marker(asm, transpose_marker(R, C), [off[ins[0]], off[outn]], mp)
        elif opname == "strided_slice":                      # last-axis x[:, b:b+w] only
            assert lay(outn) != "tile", \
                "v2 strided_slice is row-only; F4 fixpoint must keep it row (see _assign_layouts)"
            R, C = shp[ins[0]]
            assert len(shp[outn]) == 2 and shp[outn][0] == R, \
                f"v2 strided_slice: only last-axis (rows unchanged), got {shp[ins[0]]}->{shp[outn]}"
            w = shp[outn][1]
            walk_marker(asm, slice_marker(R, C, op.begin, w), [off[ins[0]], off[outn]], mp)
        elif opname == "concatenate":                        # last-axis concat -> place columns
            assert lay(outn) != "tile" and all(lay(k) != "tile" for k in ins), \
                "v2 concatenate is row-only; F4 fixpoint must keep it (and its inputs) row"
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


def compile_module(mod, reuse=True, tile=True, fuse=True):
    """Compile a `LegalizeOps`'d Relax IRModule ("main" = call_tir sequence) to NPU ISA
    through an explicit 5-pass pipeline:
      _parse           (F5-read): legalized Relax -> ops + tensor table + constants
      _assign_layouts   (F4)    : row/tile per tensor + packed weights (A4 tile-blocked)
      _detect_oproj_groups (F3) : per-head O-proj add-tree -> one accumulate group
      _plan_memory     (M1)     : liveness (fusion-aware) + exact-size free-list -> offsets (A1)
      _emit            (C1)     : unified TIR->ISA lowering (row/tile per layout)
    tile=False forces all-row (no A4); reuse=False disables A1; fuse=False disables O-proj fusion.

    Returns (asm, off, shp, top, out_name, const_inits, tiled_feed). tiled_feed = the set
    of PARAM names whose fed data must be pre-packed tile-blocked (memplan.pack_tiled)."""
    params, ops, shp, const_arrs, out_name = _parse(mod)
    if tile:
        layout, packed_params = _assign_layouts(params, ops, shp, const_arrs, out_name)
    else:
        layout, packed_params = {}, {}                       # all-row (default row via .get)
    fused_root, consumed = _detect_oproj_groups(ops, shp) if fuse else ({}, set())
    off, top, const_inits, _, tiled_feed = _plan_memory(
        params, ops, shp, const_arrs, out_name, layout, packed_params, fused_root, consumed, reuse)
    asm = _emit(ops, off, shp, top, layout, packed_params, fused_root, consumed)
    return asm, off, shp, top, out_name, const_inits, tiled_feed
