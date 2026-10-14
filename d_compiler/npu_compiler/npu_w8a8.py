"""Reference W8A8 emitter, retained for migration and numerical comparisons.

The default compiler now uses npu_w8a8_schedule and the common Python/native
walkers. ``compile_program(w8a8_mode="reference")`` selects this older path.

Activation quantization is dynamic -- the scale is this row's max, known only
at runtime -- so unlike weight quantization it must happen on the machine.
The vector unit provides exactly the pieces: ``|x| = max(x, -x)``, the seeded
reduce-max, a divide-by-127 stored as FP32 (the scale width the matmul's
scale registers read), and VQUANT.  The matmul itself then runs INT8xINT8
with the descriptors' dtype bits set, and the machine multiplies each
partial sum by ``w_scale[col] * a_scale[row]`` as it enters the FP32
accumulator -- the scales are constant along K, so this equals scaling the
whole dot product.

This reference emits the entire kernel directly and is bit-exact against
``quantize.w8a8_reference``. The scheduled path makes its memory and ordering
explicit in TIR; its decomposition need not preserve this reference's words.

Shape contract (the same one the oracle enforced): K and N are multiples of
the 64-lane tile; M is free (partial row tiles ride in the descriptors);
every activation row fits one vector instruction (K <= 65535).
"""
from __future__ import annotations

from .isa_compute import DST, IMM, SRC1, SRC2, VECTOR
from .isa_v09 import DT_FP16, DT_FP32, DT_INT8

TILE = 64


def is_w8a8(prim):
    """The quantization pass explicitly declares the emitter's semantics."""
    return prim.attrs is not None and prim.attrs.get("npu.kernel") == "w8a8"


def emit(asm, stage, prim, addresses, sram_base, capacity=None, diagnostics=None):
    """Emit the whole W8A8 matmul; returns the SRAM high-water mark."""
    if not is_w8a8(prim) or int(prim.attrs.get("npu.kernel_version", 0)) != 1:
        raise ValueError("unsupported W8A8 semantic contract version")
    if len(prim.params) != 4 or len(addresses) != 4:
        raise ValueError("W8A8 requires (FP16 X, INT8 W, FP32 scale, FP16 out)")
    # bind by parameter order -- buffer_map's iteration order is not the
    # signature's
    x_buf, qw_buf, ws_buf, out_buf = [prim.buffer_map[p] for p in prim.params]
    buffers = (x_buf, qw_buf, ws_buf, out_buf)
    if (tuple(str(b.dtype) for b in buffers) !=
            ("float16", "int8", "float32", "float16")
            or tuple(len(b.shape) for b in buffers) != (2, 2, 1, 2)):
        raise ValueError("invalid W8A8 buffer dtypes or ranks")
    rows, inner = [int(d) for d in x_buf.shape]
    cols = int(qw_buf.shape[1])
    if (min(rows, inner, cols) <= 0 or int(qw_buf.shape[0]) != inner
            or int(ws_buf.shape[0]) != cols
            or tuple(int(d) for d in out_buf.shape) != (rows, cols)):
        raise ValueError("inconsistent W8A8 buffer shapes")
    x_addr, qw_addr, ws_addr, out_addr = addresses
    if inner % TILE or cols % TILE:
        raise ValueError(f"W8A8 needs K,N as {TILE}-multiples, got "
                         f"[{inner},{cols}]")
    if inner > 0xFFFF:
        raise ValueError(f"activation row of {inner} exceeds the vlen field")

    # How many rows at a time.  Everything below is sized by this, so it is
    # decided first and taken as large as fits -- the same rule the scheduled
    # kernels follow.  Taking all the rows is what this did before, and it is
    # why the kernel stopped linking past seq 256: at [512, 8192] the fp16
    # activation alone is 8 MiB.
    fixed = inner * TILE * 2 + inner * 4 + 8 + cols * 8 + 64
    per_row = inner * 6 + 8 + TILE * 4
    panel = rows
    if capacity is not None:
        room = capacity - sram_base - fixed
        minimum_rows = min(rows, TILE)
        if room < per_row * minimum_rows:
            raise ValueError(
                f"W8A8 [{rows},{inner}]x[{inner},{cols}] does not fit: "
                f"{minimum_rows} rows need "
                f"{(fixed + minimum_rows * per_row) / 2 / 1024 / 1024:.2f} MiB of "
                f"{(capacity - sram_base) / 2 / 1024 / 1024:.2f} MiB")
        panel = min(rows, room // per_row)
        if panel >= TILE:
            panel = panel // TILE * TILE
            # This emitter handles a tail with min(panel, remaining).  Unlike
            # TIR compute_at, it does not require panel to divide M.

    cursor = [sram_base]

    def alloc(nibbles):
        base = cursor[0]
        cursor[0] += (nibbles + 7) // 8 * 8
        return base

    x_nib = alloc(panel * inner * 4)
    qa_nib = alloc(panel * inner * 2)
    a_scale_nib = alloc(panel * 8)
    tmp_nib = alloc(inner * 4)
    acc_nib = alloc(8)
    w_scale_nib = alloc(cols * 8)
    # one column of the weight with every k tile of it, so a column crosses
    # the bus once per row panel instead of once per (row tile, column, k) --
    # that re-read is what made W8A8 lose to FP16 past seq 128
    qw_strip_nib = alloc(inner * TILE * 2)
    # one column tile of the result, not the whole thing: a wide output (an
    # lm_head is [7, 128256]) would need a row stride the 16-bit shape field
    # cannot hold, and the encoder used to mask it silently
    dst_nib = alloc(panel * TILE * 4)
    if capacity is not None and cursor[0] > capacity:
        raise ValueError("W8A8 allocations exceed SRAM capacity")
    if diagnostics is not None:
        diagnostics.update(kind="w8a8", contract_version=1, rows=rows,
                           panel_rows=panel, row_panels=(rows + panel - 1) // panel,
                           sram_peak_nibbles=cursor[0])

    stage.dma_in(ws_addr * 2, w_scale_nib, cols * 4)

    def region(operand, nib0, stride, all_rows, row, col, part_rows,
               part_cols, dtype, width):
        asm.addr(operand, nib0, 0)
        asm.shape_dt(operand, all_rows, stride, 0, dtype)
        asm.addr(operand, nib0 + (row * stride + col) * width, 1)
        asm.shape_dt(operand, part_rows, part_cols, 1, dtype)

    for base_row in range(0, rows, panel):
        part = min(panel, rows - base_row)
        stage.dma_in(x_addr * 2 + base_row * inner * 2, x_nib,
                     part * inner * 2)

        # ---- per-row dynamic activation quantization (the oracle recipe) ---
        for row in range(part):
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
            asm.v_reduce_max()                # seeded: exact for any sign
            asm.addr(DST, acc_nib, 1)
            asm.save(0)
            asm.vlen(1)
            asm.addr(SRC1, acc_nib, 1)
            asm.shape_dt(SRC1, 1, 1, 1, DT_FP16)
            asm.load(0, SRC1)
            # TE defines scale=1 for an all-zero row.  Compute
            # (absmax + 127 * (absmax == 0)) / 127 without a branch.
            # The numerator round-trip through FP16 preserves every finite
            # nonzero absmax exactly.  acc's second FP16 cell holds the mask.
            asm.v_compare(IMM, 0)
            asm.addr(DST, acc_nib + 4, 1)
            asm.shape_dt(DST, 1, 1, 1, DT_FP16)
            asm.save(0)
            asm.addr(SRC1, acc_nib + 4, 1)
            asm.load(0, SRC1)
            asm.v_mul(IMM, 127)
            asm.save(0)
            asm.addr(SRC2, acc_nib + 4, 1)
            asm.shape_dt(SRC2, 1, 1, 1, DT_FP16)
            asm.load(0, SRC2)
            asm.addr(SRC1, acc_nib, 1)
            asm.load(0, SRC1)
            asm.v_add(VECTOR)
            asm.addr(DST, acc_nib, 1)
            asm.save(0)
            asm.load(0, SRC1)
            asm.v_div(IMM, 127)
            asm.addr(DST, a_scale_nib + row * 8, 1)
            asm.shape_dt(DST, 1, 1, 1, DT_FP32)   # the width scale reads use
            asm.save(0)
            asm.vlen(inner)
            asm.ascale(a_scale_nib + row * 8)
            asm.addr(SRC1, row_nib, 1)
            asm.shape_dt(SRC1, 1, inner, 1, DT_FP16)
            asm.addr(DST, qa_nib + row * inner * 2, 1)
            asm.shape_dt(DST, 1, inner, 1, DT_INT8)
            asm.vquant()

        # ---- INT8 x INT8 tiled gemm, scales applied inside the MAC --------
        for col in range(0, cols, TILE):
            asm.wscale(w_scale_nib + col * 8)
            stage.dma_2d(qw_addr * 2 + col, cols, qw_strip_nib, inner, TILE,
                         True)
            for row in range(0, part, TILE):
                part_rows = min(TILE, part - row)
                asm.ascale(a_scale_nib + row * 8)   # panel-local row indexing
                for k_index, k0 in enumerate(range(0, inner, TILE)):
                    region(SRC1, qa_nib, inner, part, row, k0,
                           part_rows, TILE, DT_INT8, 2)
                    asm.load(1, SRC1)
                    region(SRC2, qw_strip_nib, TILE, inner, k0, 0,
                           TILE, TILE, DT_INT8, 2)
                    asm.load(1, SRC2)
                    asm.m_mul(VECTOR, mac=k_index != 0)
                region(DST, dst_nib, TILE, part_rows, 0, 0,
                       part_rows, TILE, DT_FP16, 4)
                asm.save(1)
                # drain this tile before the next one reuses the buffer
                stage.dma_2d(
                    out_addr * 2 + ((base_row + row) * cols + col) * 2,
                    cols * 2, dst_nib, part_rows, TILE * 2, False)

    # descriptor dtype is sticky state: the gemm left SRC1/SRC2 at INT8,
    # which the next kernel's vector loads would trip over
    asm.shape_dt(SRC1, 1, 1, 1, DT_FP16)
    asm.shape_dt(SRC2, 1, 1, 1, DT_FP16)
    asm.shape_dt(DST, 1, 1, 1, DT_FP16)
    return cursor[0]
