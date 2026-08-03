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

from .isa import SRC1, SRC2, DST, IMM, VECTOR
from .tir_backend import _Walker, TILE, TirBackendError

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
