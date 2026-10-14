"""v09 implementation of the W8A8 row primitive (no allocation or DMA).

Scratch operands are caller-owned clobber buffers. FP32 scales are preserved;
activation input and temporary vector storage are FP16.
"""
from .isa_compute import DST, IMM, SRC1, SRC2, VECTOR
from .isa_v09 import DT_FP16, DT_FP32, DT_INT8


def quantize_row(asm, row_nib, qa_nib, a_scale_nib, tmp_nib, acc_nib, inner):
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
    asm.addr(DST, a_scale_nib, 1)
    asm.shape_dt(DST, 1, 1, 1, DT_FP32)   # the width scale reads use
    asm.save(0)
    asm.vlen(inner)
    asm.ascale(a_scale_nib)
    asm.addr(SRC1, row_nib, 1)
    asm.shape_dt(SRC1, 1, inner, 1, DT_FP16)
    asm.addr(DST, qa_nib, 1)
    asm.shape_dt(DST, 1, inner, 1, DT_INT8)
    asm.vquant()
