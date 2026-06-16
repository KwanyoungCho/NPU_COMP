"""Relax graph -> NPU ISA codegen (operator-level, B0 / logical).

The NPU is coarse-grained: a matmul or an elementwise op is a single instruction
sequence over a whole (logical) tile. So we map each high-level Relax op directly
to NPU instructions, using G-buffer offsets from memplan. No TIR lowering and no
64x64 tiling here (B0 = logical bring-up; mysim accepts tile dims <=255).

Unsupported model ops (softmax/rms_norm/silu/...) are expected to already be
decomposed into these NPU-supported ops by Relax-level legalize passes (later).

Op handling:
  - relax.matmul       -> TIR+tensorize backend (true GEMM, 64-multiple dims).
                          The direct tiling here is an oracle/test + non-64 fallback.
  - relax.sum          -> emit_row_sum   (row reduction, matrix-engine ones-matmul)
  - relax.broadcast_to -> emit_broadcast (outer-product ones-matmul)
  - elementwise / permute_dims / strided_slice / concat -> emitted directly here.
Reductions/broadcasts are dedicated ops (not matmul) so matmul-the-op is TIR-only.
"""
from tvm import relax
from . import isa
from .isa import Asm, SRC1, SRC2, DST, VECTOR, IMM


def _opname(call):
    op = call.op
    return op.name if hasattr(op, "name") else str(op)


class CodegenError(Exception):
    pass


def compile_func(func, mp, tile=None, mm_backend="direct", emit_log=None):
    """Emit an Asm for a planned Relax function. Returns Asm (ending in halt).

    tile=None  -> B0 logical matmul (single m_mul, dims<=255, simulator-only).
    tile=64    -> B0.5/B1 hardware-legal direct tiling (gather/scatter, contiguous-skip).
    mm_backend="tir" -> route matmul to the TIR+tensorize path (T1 input reuse);
                  elementwise/transpose still emitted directly here (hybrid).
    emit_log -> if a list, append (op_name, dst_shape, arg_shapes, start, end) per binding
                for per-OP command attribution (used by the HF-graph overhead analysis).
    """
    a = Asm()
    off = mp.offset

    def copy2d(dst_off, dst_stride, src_off, src_stride, rows, cols):
        """Copy a [rows,cols] block; one side may be strided. Per-row copy
        (a+0). Used for gather (strided src->contiguous) and scatter (reverse)."""
        for r in range(rows):
            a.vlen(cols)
            a.addr(SRC1, src_off + r * src_stride); a.load(0, 0)
            a.v_add(mode=IMM, imm=0)
            a.addr(DST, dst_off + r * dst_stride); a.save(0)

    def emit_matmul(dst, x, w):
        M, K = mp.shape[x]
        K2, N = mp.shape[w]
        if K != K2:
            raise CodegenError(f"matmul K mismatch {K} vs {K2}")
        # matmul-the-op is TIR-only: reductions/broadcasts are dedicated
        # relax.sum / relax.broadcast_to (see emit_row_sum/emit_broadcast), and the
        # TIR path now pads non-64-multiple dims itself (emit_matmul_into), so EVERY
        # relax.matmul routes to TIR. The direct tiling below is retained only as a
        # byte-exact oracle for the direct/tir-comparison tests (backend="direct").
        if mm_backend == "tir":
            from . import tir_backend
            nt = mp.packed_meta.get(w)        # tile-blocked weight (no B-gather) if pre-packed
            tir_backend.emit_matmul_into(a, mp, off[dst], off[x], off[w], M, K, N, b_pack_nt=nt)
            return
        if tile is None:
            if max(M, K, N) > 255:
                raise CodegenError(f"B0 logical matmul needs dims<=255, got {M}x{K}x{N} "
                                   f"(use tile=64 for 64x64-legal tiling)")
            a.tile(0, M, K)               # A: rows=M, cols=K
            a.tile(1, K, N)               # B: rows=K, cols=N
            a.addr(SRC1, off[x]); a.load(1, 0)
            a.addr(SRC2, off[w]); a.load(1, 1)
            a.m_mul(mode=VECTOR)          # real matrix multiply
            a.addr(DST, off[dst]); a.save(1)
            return
        # ---- B1: general M/N/K tiling, hardware-legal (every m_mul <=64x64) ----
        T = tile
        ax, aw, ac = off[x], off[w], off[dst]
        sA = mp.scratch_alloc(T * T)      # gathered A tile  [mt, kt]
        sB = mp.scratch_alloc(T * T)      # gathered B tile  [kt, nt]
        sP = mp.scratch_alloc(T * T)      # partial product  [mt, nt]
        sC = mp.scratch_alloc(T * T)      # output-tile accumulator [mt, nt]

        for mi in range(0, M, T):                         # output row tiles
            mt = min(T, M - mi)
            for nj in range(0, N, T):                     # output col tiles
                nt = min(T, N - nj)
                # output tile C[mi:mi+mt, nj:nj+nt]. If it spans full width (nt==N)
                # it's contiguous in C -> accumulate directly there (no scatter).
                cdst = (ac + mi * N) if nt == N else sC
                for ti, kk in enumerate(range(0, K, T)):  # accumulate over K
                    kt = min(T, K - kk)
                    # A tile: contiguous iff full K (kt==K) -> load directly, else gather
                    if kt == K:
                        a_src = ax + mi * K
                    else:
                        copy2d(sA, kt, ax + mi * K + kk, K, mt, kt); a_src = sA
                    # B tile: contiguous iff full width (nt==N) -> load directly, else gather
                    if nt == N:
                        b_src = aw + kk * N
                    else:
                        copy2d(sB, nt, aw + kk * N + nj, N, kt, nt); b_src = sB
                    a.tile(0, mt, kt); a.tile(1, kt, nt)
                    a.addr(SRC1, a_src); a.load(1, 0)
                    a.addr(SRC2, b_src); a.load(1, 1)
                    a.m_mul(mode=VECTOR)
                    if ti == 0:
                        a.addr(DST, cdst); a.save(1)      # accumulator = first partial
                    else:
                        a.addr(DST, sP); a.save(1)        # partial -> sP (FP16 round)
                        a.vlen(mt * nt)                   # acc = acc + partial (FP16 round)
                        a.addr(SRC1, cdst); a.load(0, 0)
                        a.addr(SRC2, sP); a.load(0, 1); a.v_add(mode=VECTOR)
                        a.addr(DST, cdst); a.save(0)
                # scatter only when the output tile is NOT contiguous (nt<N)
                if nt != N:
                    copy2d(ac + mi * N + nj, N, sC, nt, mt, nt)

    def emit_row_sum(dst, src):
        """Row reduction src[R,C] -> dst[R,1] (sum over last dim). The NPU has no
        reduce instruction, so the *efficient* lowering on this matrix engine is
        a ones-matmul: dst = src @ ones[C,1] (degenerate N=1, hardware-legal,
        no 64x padding -> same m_mul count as a true GEMM's K reduction).
        This keeps `relax.sum` out of the matmul-the-op path (which is TIR-only)."""
        R, C = mp.shape[src]
        assert mp.shape[dst][-1] == 1, f"emit_row_sum expects [R,1] dst, got {mp.shape[dst]}"
        T = tile or 64
        ssrc, sd = off[src], off[dst]
        ones = mp.scratch_alloc(T)          # ones[kt,1] column, filled once
        sA = mp.scratch_alloc(T * T)        # gathered A tile
        sP = mp.scratch_alloc(T)            # partial column [mt,1]
        a.vlen(T); a.addr(SRC1, ssrc); a.load(0, 0)          # load T valid elems -> pin1
        a.v_move(mode=IMM, imm=1); a.addr(DST, ones); a.save(0)   # ones = 1.0
        for mi in range(0, R, T):
            mt = min(T, R - mi)
            cdst = sd + mi                                    # dst[R,1]: row mi at +mi
            for ti, kk in enumerate(range(0, C, T)):
                kt = min(T, C - kk)
                if kt == C:
                    a_src = ssrc + mi * C
                else:
                    copy2d(sA, kt, ssrc + mi * C + kk, C, mt, kt); a_src = sA
                a.tile(0, mt, kt); a.tile(1, kt, 1)          # A[mt,kt] @ ones[kt,1]
                a.addr(SRC1, a_src); a.load(1, 0)
                a.addr(SRC2, ones); a.load(1, 1)
                a.m_mul(mode=VECTOR)                          # -> [mt,1]
                if ti == 0:
                    a.addr(DST, cdst); a.save(1)
                else:
                    a.addr(DST, sP); a.save(1)               # partial -> sP (FP16 round)
                    a.vlen(mt)
                    a.addr(SRC1, cdst); a.load(0, 0)
                    a.addr(SRC2, sP); a.load(0, 1); a.v_add(mode=VECTOR)
                    a.addr(DST, cdst); a.save(0)

    def emit_broadcast(dst, src):
        """Broadcast src -> dst[R,C], either [R,1]->[R,C] (col) or [1,C]->[R,C]
        (row). No broadcast instruction either, so lower via ones-matmul (outer
        product), degenerate K=1: col = src[R,1] @ ones[1,C]; row = ones[R,1] @ src[1,C].
        Keeps `relax.broadcast_to` out of the matmul-the-op (TIR) path."""
        Rd, Cd = mp.shape[dst]
        sshape = list(mp.shape[src]) + [1, 1]
        sr, sc = sshape[0], sshape[1]
        T = tile or 64
        ssrc, sd = off[src], off[dst]
        col_bcast = (sc == 1 and sr == Rd)                   # [R,1] -> [R,C]
        row_bcast = (sr == 1 and sc == Cd)                   # [1,C] -> [R,C]
        if not (col_bcast or row_bcast):
            raise CodegenError(f"broadcast {sshape[:2]} -> {[Rd,Cd]} unsupported")
        ones = mp.scratch_alloc(T)
        sP = mp.scratch_alloc(T * T)
        a.vlen(T); a.addr(SRC1, ssrc); a.load(0, 0)
        a.v_move(mode=IMM, imm=1); a.addr(DST, ones); a.save(0)   # ones = 1.0
        for mi in range(0, Rd, T):
            mt = min(T, Rd - mi)
            for nj in range(0, Cd, T):
                nt = min(T, Cd - nj)
                a.tile(0, mt, 1); a.tile(1, 1, nt)           # K=1 outer product
                if col_bcast:
                    a.addr(SRC1, ssrc + mi); a.load(1, 0)    # src column [mt,1]
                    a.addr(SRC2, ones); a.load(1, 1)         # ones row  [1,nt]
                else:
                    a.addr(SRC1, ones); a.load(1, 0)         # ones col  [mt,1]
                    a.addr(SRC2, ssrc + nj); a.load(1, 1)    # src row   [1,nt]
                a.m_mul(mode=VECTOR)                          # -> [mt,nt] (= broadcast)
                if nt == Cd:
                    a.addr(DST, sd + mi * Cd); a.save(1)
                else:
                    a.addr(DST, sP); a.save(1)
                    copy2d(sd + mi * Cd + nj, Cd, sP, nt, mt, nt)

    def emit_transpose(dst, src):
        """2D transpose [R,C]->[C,R] via per-element copy (no transpose/strided ISA).
        copy = load 1 elem, add immediate 0, save. O(R*C) -> instruction-heavy
        (this overhead is exactly what we measure for 'is a transpose ISA needed')."""
        shp = mp.shape[src]
        if len(shp) != 2:
            raise CodegenError(f"transpose expects 2D, got {shp}")
        R, C = shp
        s0, d0 = off[src], off[dst]
        for r in range(R):
            for c in range(C):
                a.vlen(1)
                a.addr(SRC1, s0 + r * C + c); a.load(0, 0)
                a.v_add(mode=IMM, imm=0)                 # identity copy (a + 0)
                a.addr(DST, d0 + c * R + r); a.save(0)

    def emit_strided_slice(dst, call):
        """2D last-axis slice x[:, b:e] -> contiguous dst (per-row copy)."""
        src = call.args[0]
        axes = [int(f.value) for f in call.args[1]]
        begin = [int(f.value) for f in call.args[2]]
        end = [int(f.value) for f in call.args[3]]
        R, C = mp.shape[src]
        if axes[0] not in (1, -1):
            raise CodegenError(f"strided_slice: only last-axis 2D (axes={axes})")
        b0 = begin[0] + (C if begin[0] < 0 else 0)
        e0 = min(end[0], C)
        w = e0 - b0
        for r in range(R):
            a.vlen(w)
            a.addr(SRC1, off[src] + r * C + b0); a.load(0, 0)
            a.v_add(mode=IMM, imm=0)
            a.addr(DST, off[dst] + r * w); a.save(0)

    def emit_concat(dst, call):
        """2D last-axis concat: copy each input's columns into the output (per-row)."""
        srcs = list(call.args[0].fields)
        if int(call.attrs.axis) not in (1, -1):
            raise CodegenError(f"concat: only last-axis 2D (axis={call.attrs.axis})")
        Cd = mp.shape[dst][1]
        col = 0
        for s in srcs:
            Rs, Cs = mp.shape[s]
            for r in range(Rs):
                a.vlen(Cs)
                a.addr(SRC1, off[s] + r * Cs); a.load(0, 0)
                a.v_add(mode=IMM, imm=0)
                a.addr(DST, off[dst] + r * Cd + col); a.save(0)
            col += Cs

    def emit_ew(dst, op_method, args, n):
        """Elementwise vector op over n contiguous elements. args: 1 or 2 vars.
        Chunked to <=8192 (16-bit vlen field max is 65535; 8192 matches the
        documented PE buffer) so large activations (e.g. [SEQ,F]=1M) don't overflow."""
        CH = 8192
        o0 = off[args[0]]
        o1 = off[args[1]] if len(args) == 2 else None
        od = off[dst]
        for base in range(0, n, CH):
            m = min(CH, n - base)
            a.vlen(m)
            a.addr(SRC1, o0 + base); a.load(0, 0)
            if o1 is not None:
                a.addr(SRC2, o1 + base); a.load(0, 1)
                op_method(mode=VECTOR)
            else:
                op_method()           # unary (sqrt/exp) — no mode/operand2
            a.addr(DST, od + base); a.save(0)

    EW2 = {"relax.add": a.v_add, "relax.subtract": a.v_sub,
           "relax.multiply": a.v_mul, "relax.divide": a.v_div}
    EW1 = {"relax.sqrt": a.v_sqrt, "relax.exp": a.v_exp}

    seq = func.body
    for block in seq.blocks:
        for binding in block.bindings:
            dst = binding.var
            call = binding.value
            if isinstance(call, (relax.Var, relax.Tuple)):  # alias / tuple output: no emit
                continue
            if not isinstance(call, relax.Call):
                raise CodegenError(f"unsupported binding value {type(call)}")
            name = _opname(call)
            _start = len(a.words)
            if name == "relax.matmul":
                emit_matmul(dst, call.args[0], call.args[1])   # roles tagged inside (walker)
            elif name == "relax.sum":
                axis = [int(x) % len(mp.shape[call.args[0]]) for x in call.attrs.axis]
                if axis != [len(mp.shape[call.args[0]]) - 1]:
                    raise CodegenError(f"sum: only last-axis keepdims supported (axis={axis})")
                with a.role("reduce"):
                    emit_row_sum(dst, call.args[0])
            elif name == "relax.broadcast_to":
                with a.role("broadcast"):
                    emit_broadcast(dst, call.args[0])
            elif name == "relax.permute_dims":
                with a.role("transpose"):
                    emit_transpose(dst, call.args[0])
            elif name == "relax.strided_slice":
                with a.role("layout"):
                    emit_strided_slice(dst, call)
            elif name == "relax.concat":
                with a.role("layout"):
                    emit_concat(dst, call)
            elif name in EW2:
                n = 1
                for d in mp.shape[dst]:
                    n *= d
                with a.role("elementwise"):
                    emit_ew(dst, EW2[name], [call.args[0], call.args[1]], n)
            elif name in EW1:
                n = 1
                for d in mp.shape[dst]:
                    n *= d
                with a.role("elementwise"):
                    emit_ew(dst, EW1[name], [call.args[0]], n)
            else:
                raise CodegenError(f"unsupported op for B0 codegen: {name}")
            if emit_log is not None:
                argshapes = [mp.shape.get(ar) for ar in call.args if ar in mp.shape]
                emit_log.append((name, mp.shape.get(dst), argshapes, _start, len(a.words)))
    a.halt()
    return a
