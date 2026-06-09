"""Relax graph -> NPU ISA codegen (operator-level, B0 / logical).

The NPU is coarse-grained: a matmul or an elementwise op is a single instruction
sequence over a whole (logical) tile. So we map each high-level Relax op directly
to NPU instructions, using G-buffer offsets from memplan. No TIR lowering and no
64x64 tiling here (B0 = logical bring-up; mysim accepts tile dims <=255).

Unsupported model ops (softmax/rms_norm/silu/...) are expected to already be
decomposed into these NPU-supported ops by Relax-level legalize passes (later).
"""
from tvm import relax
from . import isa
from .isa import Asm, SRC1, SRC2, DST, VECTOR, IMM


def _opname(call):
    op = call.op
    return op.name if hasattr(op, "name") else str(op)


class CodegenError(Exception):
    pass


def compile_func(func, mp, tile=None):
    """Emit an Asm for a planned Relax function. Returns Asm (ending in halt).

    tile=None  -> B0 logical matmul (single m_mul, dims<=255, simulator-only).
    tile=64    -> B0.5 hardware-legal: split K into <=64 chunks, accumulate
                  partial products via save->load->add (FP16 rounding each save).
                  (This first version tiles K only; M,N must be <=tile.)
    """
    a = Asm()
    off = mp.offset

    def emit_matmul(dst, x, w):
        M, K = mp.shape[x]
        K2, N = mp.shape[w]
        if K != K2:
            raise CodegenError(f"matmul K mismatch {K} vs {K2}")
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
        # ---- B0.5 K-tiled, hardware-legal (<=64x64 per m_mul) ----
        T = tile
        if M > T or N > T:
            raise CodegenError(f"B0.5 tiles K only; M,N must be <={T}, got M={M} N={N} "
                               f"(M/N tiling is B1)")
        ax, aw, ac = off[x], off[w], off[dst]
        sA = mp.scratch_alloc(M * T)      # gathered A tile [M, kt]
        sP = mp.scratch_alloc(M * N)      # partial product [M, N]
        nkt = (K + T - 1) // T
        for ti in range(nkt):
            kk = ti * T
            kt = min(T, K - kk)
            # gather A[:, kk:kk+kt] (rows strided by K) -> sA contiguous [M,kt]
            for r in range(M):
                a.vlen(kt); a.addr(SRC1, ax + r * K + kk); a.load(0, 0)
                a.v_add(mode=IMM, imm=0)                  # identity copy
                a.addr(DST, sA + r * kt); a.save(0)
            # B K-tile is contiguous rows: w[kk:kk+kt, :] at aw + kk*N, shape [kt,N]
            a.tile(0, M, kt); a.tile(1, kt, N)
            a.addr(SRC1, sA); a.load(1, 0)
            a.addr(SRC2, aw + kk * N); a.load(1, 1)
            a.m_mul(mode=VECTOR)
            if ti == 0:
                a.addr(DST, ac); a.save(1)               # C = first partial (FP16 round)
            else:
                a.addr(DST, sP); a.save(1)               # partial -> sP (FP16 round)
                a.vlen(M * N)                            # C = C + partial (FP16 round)
                a.addr(SRC1, ac); a.load(0, 0)
                a.addr(SRC2, sP); a.load(0, 1); a.v_add(mode=VECTOR)
                a.addr(DST, ac); a.save(0)

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

    def emit_ew(dst, op_method, args, n):
        """Elementwise vector op over n contiguous elements. args: 1 or 2 vars."""
        a.vlen(n)
        a.addr(SRC1, off[args[0]]); a.load(0, 0)
        if len(args) == 2:
            a.addr(SRC2, off[args[1]]); a.load(0, 1)
            op_method(mode=VECTOR)
        else:
            op_method()               # unary (sqrt/exp) — no mode/operand2
        a.addr(DST, off[dst]); a.save(0)

    EW2 = {"relax.add": a.v_add, "relax.subtract": a.v_sub,
           "relax.multiply": a.v_mul, "relax.divide": a.v_div}
    EW1 = {"relax.sqrt": a.v_sqrt, "relax.exp": a.v_exp}

    seq = func.body
    for block in seq.blocks:
        for binding in block.bindings:
            dst = binding.var
            call = binding.value
            if isinstance(call, relax.Var):          # alias (offset already shared in memplan)
                continue
            if not isinstance(call, relax.Call):
                raise CodegenError(f"unsupported binding value {type(call)}")
            name = _opname(call)
            if name == "relax.matmul":
                emit_matmul(dst, call.args[0], call.args[1])
            elif name == "relax.permute_dims":
                emit_transpose(dst, call.args[0])
            elif name in EW2:
                n = 1
                for d in mp.shape[dst]:
                    n *= d
                emit_ew(dst, EW2[name], [call.args[0], call.args[1]], n)
            elif name in EW1:
                n = 1
                for d in mp.shape[dst]:
                    n *= d
                emit_ew(dst, EW1[name], [call.args[0]], n)
            else:
                raise CodegenError(f"unsupported op for B0 codegen: {name}")
    a.halt()
    return a
