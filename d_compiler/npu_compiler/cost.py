"""Static resource analyzer (NOT latency — see PLAN §4.1).

Counts what a compiled program does, with zero hardware data:
  - instruction count by opcode kind
  - matmul tiles, gather/copy rows, accumulate adds
  - G-buffer footprint
Used to measure input-reuse optimizations (gather count before/after).
"""
from collections import Counter

_OPN = {0x80: "ADDR", 0x82: "VLEN", 0x88: "TILE", 0x90: "LOAD", 0x98: "SAVE",
        0x42: "MATMUL", 0x40: "M_ADD", 0x01: "VADD", 0x02: "VSUB",
        0x0A: "VMUL", 0x0B: "VDIV", 0x0E: "VSQRT", 0x0F: "VEXP", 0xFF: "HALT"}


def analyze(asm, mp=None):
    """asm: Asm (or .words list). mp: optional MemPlan for footprint. -> dict."""
    words = asm.words if hasattr(asm, "words") else list(asm)
    by_op = Counter()
    matmul_tiles = 0       # real matrix multiplies (op 0x42, vector mode)
    copy_rows = 0          # identity copies (a+0) = gather/scatter/fill rows
    accum_adds = 0         # vector add with two operands = accumulation
    for w in words:
        op = w & 0xFF
        if w == 0:
            by_op["NOP"] += 1; continue
        by_op[_OPN.get(op, hex(op))] += 1
        mode = (w >> 30) & 3
        imm = (w >> 8) & 0xFFFF
        if op == 0x42 and mode == 2:
            matmul_tiles += 1
        if op == 0x01:                      # VADD
            if mode == 0 and imm == 0:
                copy_rows += 1              # a + 0  -> a (copy)
            elif mode == 2:
                accum_adds += 1             # a + b  -> accumulate
    return {
        "total": len(words),
        "by_op": dict(by_op),
        "matmul_tiles": matmul_tiles,
        "copy_ops": copy_rows,             # gather/scatter/fill identity-copies
        "accum_adds": accum_adds,
        "gbuffer_elems": (mp.top if mp is not None else None),
        "gbuffer_bytes": (mp.top * 2 if mp is not None else None),
    }


def summary_line(stats, label=""):
    g = f", G-buf {stats['gbuffer_elems']}" if stats["gbuffer_elems"] is not None else ""
    return (f"{label:<14} instrs={stats['total']:<7} matmul_tiles={stats['matmul_tiles']:<4} "
            f"copies={stats['copy_ops']:<6} accum_adds={stats['accum_adds']:<5}{g}")
