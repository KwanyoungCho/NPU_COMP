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


def _detect_oproj_groups(seq):
    """Find maximal add-trees whose leaves are all same-shape matmuls
    (the per-head attention output projection sum_h ctx_h @ Wo_h).

    Returns (fused_root, consumed):
      fused_root: {root_add_var -> [leaf matmul var, ...] in fold order}
      consumed:   set of every binding var folded away (tree adds + leaf matmuls).
    Only fuses when each leaf/intermediate is single-use, so skipping its own
    emit is safe (the root add var is still produced by the group)."""
    val = {b.var: b.value for blk in seq.blocks for b in blk.bindings}
    uses = {}
    def _count(x):
        if isinstance(x, relax.Var):
            uses[x] = uses.get(x, 0) + 1
        elif isinstance(x, relax.Tuple):
            for f in x.fields:
                _count(f)
    for blk in seq.blocks:
        for b in blk.bindings:
            c = b.value
            if isinstance(c, relax.Call):
                for ar in c.args:
                    _count(ar)
            else:
                _count(c)

    def _op(v, opn):
        c = val.get(v)
        return c if (isinstance(c, relax.Call) and _opname(c) == opn) else None

    def _leaves(v, shp):
        ad = _op(v, "relax.add")
        if ad is not None:
            l = _leaves(ad.args[0], shp); r = _leaves(ad.args[1], shp)
            return None if (l is None or r is None) else (l + r)
        mm = _op(v, "relax.matmul")
        if mm is not None and tuple(v.struct_info.shape) == shp:
            return [v]
        return None

    def _legal_term(v):                             # leaf matmul with 64-multiple K, N
        mm = val[v]
        K = int(mm.args[0].struct_info.shape[1]); N = int(mm.args[1].struct_info.shape[1])
        return K % 64 == 0 and N % 64 == 0

    cand = {}
    for blk in seq.blocks:
        for b in blk.bindings:
            if _op(b.var, "relax.add") is None:
                continue
            shp = tuple(b.var.struct_info.shape)
            lv = _leaves(b.var, shp)
            if lv is not None and len(lv) >= 2 and all(_legal_term(x) for x in lv):
                cand[b.var] = lv
    used_by_cand = set()
    for v in cand:
        ad = _op(v, "relax.add")
        for ar in (ad.args[0], ad.args[1]):
            if ar in cand:
                used_by_cand.add(ar)

    fused_root, consumed = {}, set()
    for v, leaves in cand.items():
        if v in used_by_cand:                       # not the maximal root
            continue
        nodes, stack, ok = set(), [v], True
        while stack:                                # collect tree adds + leaves
            x = stack.pop(); nodes.add(x)
            ad = _op(x, "relax.add")
            if ad is not None:
                stack += [ad.args[0], ad.args[1]]
        # safety: every folded node except the root must be single-use
        for x in nodes:
            if x is not v and uses.get(x, 0) != 1:
                ok = False; break
        if ok:
            fused_root[v] = leaves
            consumed |= nodes
    return fused_root, consumed


def compile_func(func, mp, tile=None, mm_backend="direct", emit_log=None, reuse_act=True,
                 fuse_oproj=True):
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
    # shared across all matmuls so a read-only activation tile (e.g. xn read by every
    # Q/K/V proj) is gathered ONCE for the whole layer, not per-matmul. (SSA + bump
    # allocator => (off,stride) uniquely identifies write-once data, so reuse is safe.)
    mm_gather_cache = {} if reuse_act else None

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
            tir_backend.emit_matmul_into(a, mp, off[dst], off[x], off[w], M, K, N,
                                         b_pack_nt=nt, gather_cache=mm_gather_cache)
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
                    if ti != 0:                       # preload running sum into pout for MAC
                        a.vlen(mt * nt)
                        a.addr(SRC1, cdst); a.load(0, 0); a.v_copy()
                    a.tile(0, mt, kt); a.tile(1, kt, nt)
                    a.addr(SRC1, a_src); a.load(1, 0)
                    a.addr(SRC2, b_src); a.load(1, 1)
                    a.m_mul(mode=VECTOR, mac=(ti != 0))   # 0710 MAC: acc += A@B in place
                    a.addr(DST, cdst); a.save(1)
                # scatter only when the output tile is NOT contiguous (nt<N)
                if nt != N:
                    copy2d(ac + mi * N + nj, N, sC, nt, mt, nt)

    def emit_row_sum(dst, src):
        """Row reduction src[R,C] -> dst[R,1] (sum over last dim). 0710 ISA has a
        NATIVE reduce-sum (0x14): for each row, load the contiguous row (C elems)
        and reduce to one scalar. Replaces the old ones-matmul lowering."""
        R, C = mp.shape[src]
        assert mp.shape[dst][-1] == 1, f"emit_row_sum expects [R,1] dst, got {mp.shape[dst]}"
        ssrc, sd = off[src], off[dst]
        for r in range(R):
            a.vlen(C)
            a.addr(SRC1, ssrc + r * C); a.load(0, 0)     # contiguous row [C]
            a.v_reduce_sum()                              # -> [sum]
            a.addr(DST, sd + r); a.save(0)                # dst[r,0] = sum

    def emit_row_max(dst, src):
        """Row reduction src[R,C] -> dst[R,1] (max over last dim). 0710 ISA still has
        NO reduce-max, so fold it with vector max (0x12 bit[28]): strided-load each
        column j (all R rows at once, 0710 strided load) and v_max into a running
        accumulator. O(C) vector-max ops (over R lanes) — not O(R*C). Enables the
        max-subtraction of a *stable* softmax."""
        R, C = mp.shape[src]
        assert mp.shape[dst][-1] == 1, f"emit_row_max expects [R,1] dst, got {mp.shape[dst]}"
        s0, d0 = off[src], off[dst]
        acc = mp.scratch_alloc(R)
        a.tile(0, R, C); a.addr(SRC1, s0)
        a.load(1, 0, strided=1, ncols=1, start=0)         # pin1 = column 0 [R]
        a.vlen(R); a.v_copy(); a.addr(DST, acc); a.save(0)  # acc = col0
        for j in range(1, C):
            a.tile(0, R, C); a.addr(SRC1, s0)
            a.load(1, 0, strided=1, ncols=1, start=j)     # pin1 = column j [R]
            a.vlen(R); a.addr(SRC2, acc); a.load(0, 1)    # pin2 = acc
            a.v_max(mode=VECTOR)                           # pout = max(colj, acc)
            a.addr(DST, acc); a.save(0)
        a.vlen(R); a.addr(SRC1, acc); a.load(0, 0)
        a.v_copy(); a.addr(DST, d0); a.save(0)             # dst[R,1] = row max

    def emit_broadcast(dst, src):
        """Broadcast src -> dst[R,C].
        Row-broadcast ([1,C]->[R,C]): replicate the source row via native copy
        (0x17) — a full-address load, so any offset is fine.
        Col-broadcast ([R,1]->[R,C]): the 0710 native broadcast (0x15) addresses
        its scalar source with a 16-bit IMMEDIATE (no set-addr), so it can't reach
        offsets >64K. A per-row-varying scalar therefore stays on the matrix engine:
        col[R,1] @ ones[1,C] outer product (full addressing, degenerate K=1)."""
        Rd, Cd = mp.shape[dst]
        sshape = list(mp.shape[src]) + [1, 1]
        sr, sc = sshape[0], sshape[1]
        ssrc, sd = off[src], off[dst]
        col_bcast = (sc == 1 and sr == Rd)                   # [R,1] -> [R,C]
        row_bcast = (sr == 1 and sc == Cd)                   # [1,C] -> [R,C]
        if not (col_bcast or row_bcast):
            raise CodegenError(f"broadcast {sshape[:2]} -> {[Rd,Cd]} unsupported")
        if row_bcast:
            for r in range(Rd):
                a.vlen(Cd)
                a.addr(SRC1, ssrc); a.load(0, 0); a.v_copy()  # replicate source row [1,C]
                a.addr(DST, sd + r * Cd); a.save(0)
            return
        # col-broadcast: ones-matmul outer product (full addressing)
        T = tile or 64
        ones = mp.scratch_alloc(T)
        sP = mp.scratch_alloc(T * T)
        # ones = 1.0 via native broadcast IMMEDIATE fill (0x15). v_move would read the
        # (as-yet-unloaded) pin1 -> OOB; broadcast-immediate writes pout without reading pin1.
        a.vlen(T); a.v_broadcast(mode=IMM, imm=1); a.addr(DST, ones); a.save(0)
        for mi in range(0, Rd, T):
            mt = min(T, Rd - mi)
            for nj in range(0, Cd, T):
                nt = min(T, Cd - nj)
                a.tile(0, mt, 1); a.tile(1, 1, nt)           # K=1 outer product
                a.addr(SRC1, ssrc + mi); a.load(1, 0)        # src column [mt,1]
                a.addr(SRC2, ones); a.load(1, 1)             # ones row  [1,nt]
                a.m_mul(mode=VECTOR)                          # -> [mt,nt] (= broadcast)
                if nt == Cd:
                    a.addr(DST, sd + mi * Cd); a.save(1)
                else:
                    a.addr(DST, sP); a.save(1)
                    copy2d(sd + mi * Cd + nj, Cd, sP, nt, mt, nt)

    def emit_transpose(dst, src):
        """2D transpose [R,C]->[C,R] via the 0710 native strided load (0x90 bit[29]).
        A strided load of a [rt,C] block reads columns [0,ct) COLUMN-major, i.e. the
        transposed block [ct,rt] in row-major order, in ONE instruction (was O(R*C)
        per-element copies). v_copy moves it pin1->pout; when the tile spans the full
        src-row dimension (rt==R) the dst block is contiguous (single save)."""
        shp = mp.shape[src]
        if len(shp) != 2:
            raise CodegenError(f"transpose expects 2D, got {shp}")
        R, C = shp
        s0, d0 = off[src], off[dst]
        T = tile or 64
        scr = mp.scratch_alloc(T * T)
        for ci in range(0, C, T):                        # tile dst rows (= src cols)
            ct = min(T, C - ci)
            for ri in range(0, R, T):                    # tile dst cols (= src rows)
                rt = min(T, R - ri)
                a.tile(0, rt, C)                         # R0=rt rows read, R1=C src stride
                a.addr(SRC1, s0 + ri * C + ci)
                a.load(1, 0, strided=1, ncols=ct, start=0)   # pin1[c*rt+r]=src[ri+r,ci+c]
                a.vlen(ct * rt); a.v_copy()              # pin1 -> pout (dst block row-major)
                if rt == R:                              # block = full dst rows -> contiguous
                    a.addr(DST, d0 + ci * R); a.save(0)
                else:
                    a.addr(DST, scr); a.save(0)
                    copy2d(d0 + ci * R + ri, R, scr, rt, ct, rt)

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
    # SiLU: 0710 native activation on a matrix op (m_add(+0) with act bit = SiLU).
    # negative -> sign-inversion (0x16); cos/sin -> 0x18 (RoPE on-device angle -> cos/sin).
    EW1 = {"relax.sqrt": a.v_sqrt, "relax.exp": a.v_exp,
           "relax.nn.silu": lambda: a.m_add(mode=IMM, imm=0, act=True),
           "relax.negative": a.v_sign_inv, "relax.cos": a.v_cos, "relax.sin": a.v_sin}

    seq = func.body
    # ---- detect per-head O-proj add-tree: attn = sum_h (ctx_h @ Wo_h) ----------
    # H separate same-shape matmuls feeding one left-folded add chain. Fuse them
    # so the wide [M,N] result is scattered ONCE (shared accumulator) instead of
    # once per head. TIR-only optimisation (uses the accumulate-group emit).
    fused_root, consumed = _detect_oproj_groups(seq) if (fuse_oproj and mm_backend == "tir") \
        else ({}, set())

    def emit_oproj_group(root_var, leaves):
        from . import tir_backend
        terms = []
        for lv in leaves:
            mm = val_of[lv]                                  # relax.matmul call
            av, bv = mm.args[0], mm.args[1]
            M, K = mp.shape[av]; _, N = mp.shape[bv]
            terms.append((off[av], off[bv], M, K, N, mp.packed_meta.get(bv)))
        tir_backend.emit_matmul_accumulate_group(a, mp, off[root_var], terms,
                                                 gather_cache=mm_gather_cache)
        return terms[0]                                      # (.,.,M,K,N,.) for logging

    val_of = {b.var: b.value for blk in seq.blocks for b in blk.bindings}
    for block in seq.blocks:
        for binding in block.bindings:
            dst = binding.var
            call = binding.value
            if dst in consumed:                              # folded into an O-proj group
                if dst in fused_root:
                    _start = len(a.words)
                    _, _, M, K, N, _ = emit_oproj_group(dst, fused_root[dst])
                    if emit_log is not None:                 # attribute to the matmul (O proj)
                        emit_log.append(("relax.matmul", mp.shape.get(dst),
                                         [(M, K), (K, N)], _start, len(a.words)))
                continue
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
            elif name == "relax.max":
                axis = [int(x) % len(mp.shape[call.args[0]]) for x in call.attrs.axis]
                if axis != [len(mp.shape[call.args[0]]) - 1]:
                    raise CodegenError(f"max: only last-axis keepdims supported (axis={axis})")
                with a.role("reduce"):
                    emit_row_max(dst, call.args[0])
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
