"""The W8A8 matmul, emitted as one validated instruction sequence.

Activation quantization is dynamic -- the scale is this row's max, known only
at runtime -- so unlike weight quantization it must happen on the machine.
The vector unit provides exactly the pieces: ``|x| = max(x, -x)``, the seeded
reduce-max, a divide-by-127 stored as FP32 (the scale width the matmul's
scale registers read), and VQUANT.  The matmul itself then runs INT8xINT8
with the descriptors' dtype bits set, and the machine multiplies each
partial sum by ``w_scale[col] * a_scale[row]`` as it enters the FP32
accumulator -- the scales are constant along K, so this equals scaling the
whole dot product.

This kernel is emitted directly rather than walked from TIR: the sequence is
the hand-written backend's, bit-exact against ``quantize.w8a8_reference``,
and TIR would have to be pattern-matched back into exactly these
instructions anyway.  The TIR body still defines the semantics the llvm
cross-check runs.

Shape contract (the same one the oracle enforced): K and N are multiples of
the 64-lane tile; M is free (partial row tiles ride in the descriptors);
every activation row fits one vector instruction (K <= 65535).
"""
from __future__ import annotations

from .isa_0818 import DST, IMM, SRC1, SRC2, VECTOR
from .isa_v09 import DT_FP16, DT_FP32, DT_INT8

TILE = 64


def is_w8a8(prim):
    """Does this PrimFunc carry the W8A8 marker block?"""
    import tvm.tir as tir

    found = []

    def visit(node):
        if isinstance(node, tir.Block) and node.name_hint == "a_quant":
            found.append(node)

    tir.stmt_functor.post_order_visit(prim.body, visit)
    return bool(found)


def emit(asm, stage, prim, addresses, sram_base):
    """Emit the whole W8A8 matmul; returns the SRAM high-water mark."""
    # bind by parameter order -- buffer_map's iteration order is not the
    # signature's
    x_buf, qw_buf, ws_buf, out_buf = [prim.buffer_map[p] for p in prim.params]
    rows, inner = [int(d) for d in x_buf.shape]
    cols = int(qw_buf.shape[1])
    x_addr, qw_addr, ws_addr, out_addr = addresses
    if inner % TILE or cols % TILE:
        raise ValueError(f"W8A8 needs K,N as {TILE}-multiples, got "
                         f"[{inner},{cols}]")
    if inner > 0xFFFF:
        raise ValueError(f"activation row of {inner} exceeds the vlen field")

    cursor = [sram_base]

    def alloc(nibbles):
        base = cursor[0]
        cursor[0] += (nibbles + 7) // 8 * 8
        return base

    x_nib = alloc(rows * inner * 4)
    qa_nib = alloc(rows * inner * 2)
    a_scale_nib = alloc(rows * 8)
    tmp_nib = alloc(inner * 4)
    acc_nib = alloc(8)
    w_scale_nib = alloc(cols * 8)
    qw_tile_nib = alloc(TILE * TILE * 2)
    dst_nib = alloc(rows * cols * 4)

    # ---- staging: activation and the fp32 scale vector, whole --------------
    stage.dma_in(x_addr * 2, x_nib, rows * inner * 2)
    stage.dma_in(ws_addr * 2, w_scale_nib, cols * 4)

    # ---- per-row dynamic activation quantization (the oracle recipe) -------
    for row in range(rows):
        row_nib = x_nib + row * inner * 4
        asm.vlen(inner)
        asm.addr(SRC1, row_nib, 1)
        asm.shape_dt(SRC1, 1, inner, 1, DT_FP16)
        asm.load(0, SRC1)
        asm.v_sign_inv()
        asm.addr(DST, tmp_nib, 1)
        asm.shape_dt(DST, 1, inner, 1, DT_FP16)
        asm.save(0)
        asm.addr(SRC1, row_nib, 1)
        asm.load(0, SRC1)
        asm.addr(SRC2, tmp_nib, 1)
        asm.shape_dt(SRC2, 1, inner, 1, DT_FP16)
        asm.load(0, SRC2)
        asm.v_max(VECTOR)
        asm.addr(DST, tmp_nib, 1)
        asm.save(0)
        asm.addr(SRC1, tmp_nib, 1)
        asm.load(0, SRC1)
        asm.v_reduce_max()                    # seeded: exact for any sign
        asm.addr(DST, acc_nib, 1)
        asm.save(0)
        asm.vlen(1)
        asm.addr(SRC1, acc_nib, 1)
        asm.shape_dt(SRC1, 1, 1, 1, DT_FP16)
        asm.load(0, SRC1)
        asm.v_div(IMM, 127)
        asm.addr(DST, a_scale_nib + row * 8, 1)
        asm.shape_dt(DST, 1, 1, 1, DT_FP32)   # the width the scale reads use
        asm.save(0)
        asm.vlen(inner)
        asm.ascale(a_scale_nib + row * 8)
        asm.addr(SRC1, row_nib, 1)
        asm.shape_dt(SRC1, 1, inner, 1, DT_FP16)
        asm.addr(DST, qa_nib + row * inner * 2, 1)
        asm.shape_dt(DST, 1, inner, 1, DT_INT8)
        asm.vquant()
    asm.shape_dt(DST, 1, 1, 1, DT_FP16)       # dtype is sticky: restore

    # ---- INT8 x INT8 tiled gemm, scales applied inside the MAC -------------
    def region(operand, nib0, stride, all_rows, row, col, part_rows,
               part_cols, dtype, width):
        asm.addr(operand, nib0, 0)
        asm.shape_dt(operand, all_rows, stride, 0, dtype)
        asm.addr(operand, nib0 + (row * stride + col) * width, 1)
        asm.shape_dt(operand, part_rows, part_cols, 1, dtype)

    for row in range(0, rows, TILE):
        part_rows = min(TILE, rows - row)
        asm.ascale(a_scale_nib + row * 8)     # tile-local row indexing
        for col in range(0, cols, TILE):
            asm.wscale(w_scale_nib + col * 8)
            for k_index, k0 in enumerate(range(0, inner, TILE)):
                region(SRC1, qa_nib, inner, rows, row, k0,
                       part_rows, TILE, DT_INT8, 2)
                asm.load(1, SRC1)
                # weight tile arrives fresh from global for each use
                stage.dma_2d(qw_addr * 2 + k0 * cols + col, cols,
                             qw_tile_nib, TILE, TILE, True)
                region(SRC2, qw_tile_nib, TILE, TILE, 0, 0,
                       TILE, TILE, DT_INT8, 2)
                asm.load(1, SRC2)
                asm.m_mul(VECTOR, mac=k_index != 0)
            region(DST, dst_nib, cols, rows, row, col,
                   part_rows, TILE, DT_FP16, 4)
            asm.save(1)

    # descriptor dtype is sticky state: the gemm left SRC1/SRC2 at INT8,
    # which the next kernel's vector loads would trip over
    asm.shape_dt(SRC1, 1, 1, 1, DT_FP16)
    asm.shape_dt(SRC2, 1, 1, 1, DT_FP16)
    asm.shape_dt(DST, 1, 1, 1, DT_FP16)
    stage.dma_out(out_addr * 2, dst_nib, rows * cols * 2)
    return cursor[0]
