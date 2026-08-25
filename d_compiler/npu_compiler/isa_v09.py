"""ISA ver.09 word encoders (memory layer DMA/control + v09 compute deltas).

Units follow d_compiler/ISA_V09.md: global addresses/strides/cols are 32-bit
cells; SRAM addresses are 4-bit nibbles (24 effective bits) and must sit on an
8-nibble (cell) boundary for DMA.  ver.08 compute encodings are reused
verbatim (import them from :mod:`npu_compiler.isa_0818`); this module adds only
the v09 deltas: per-descriptor dtype bits, scale-address setup, drain/reduce
carry flags, FP32 vector save, and VQUANT/VDEQUANT.
"""
from __future__ import annotations

from .isa_0818 import (  # noqa: F401  (re-exported ver.08 encodings)
    DST, IMM, MAIN, PARTIAL, SCALAR, SRC1, SRC2, VECTOR,
    enc_addr_hi, enc_addr_lo, enc_load, enc_save, enc_vlen,
)
from .isa_0818 import enc_mcols as _enc_mcols_0818
from .isa_0818 import enc_mrows as _enc_mrows_0818

OP_NOP = 0x00
OP_SNAPSHOT = 0xF0
OP_HALT = 0xFF
OP_GLOAD = 0xA0
OP_GSTORE = 0xA8
OP_ASCALE = 0x8A
OP_WSCALE = 0x8B
OP_VQUANT = 0x1A
OP_VDEQUANT = 0x1B

DT_FP16, DT_FP32, DT_INT8, DT_INT4 = 0, 1, 2, 3
ACT_OFF, ACT_GELU_STD, ACT_SILU, ACT_GELU_LEGACY = 0, 1, 2, 3

SRAM_NIBBLES = 8 * 1024 * 1024 * 2  # 8 MiB, 2^24 nibbles
DMA_WORDS = 4


class V09EncodeError(ValueError):
    pass


def _check(value, bits, name):
    value = int(value)
    if not 0 <= value < (1 << bits):
        raise V09EncodeError(f"{name}={value} does not fit in {bits} bits")
    return value


def enc_nop():
    return OP_NOP


def enc_snapshot():
    return OP_SNAPSHOT


def enc_halt():
    return OP_HALT


def _enc_dma(opcode, g_addr, g_stride, sram_addr, rows, cols):
    g_addr = _check(g_addr, 32, "g_addr")
    g_stride = _check(g_stride, 32, "g_stride")
    sram_addr = int(sram_addr)
    if not 0 <= sram_addr < SRAM_NIBBLES:
        raise V09EncodeError(f"sram_addr={sram_addr} outside 2^24 nibbles")
    if sram_addr % 8:
        raise V09EncodeError(f"sram_addr={sram_addr} not 8-nibble aligned")
    rows = _check(rows, 16, "rows")
    cols = _check(cols, 16, "cols")
    if rows == 0 or cols == 0:
        raise V09EncodeError("rows and cols must be nonzero")
    # The 24-bit SRAM address rides in the opcode word's reserved field.
    return [(sram_addr << 8) | opcode, g_addr, g_stride, (rows << 16) | cols]


def enc_gload(g_addr, g_stride, sram_addr, rows, cols):
    """global[g_addr + r*g_stride ..+cols) cells -> SRAM[sram_addr..] nibbles."""
    return _enc_dma(OP_GLOAD, g_addr, g_stride, sram_addr, rows, cols)


def enc_gstore(g_addr, g_stride, sram_addr, rows, cols):
    """SRAM[sram_addr..] nibbles -> global[g_addr + r*g_stride ..+cols) cells."""
    return _enc_dma(OP_GSTORE, g_addr, g_stride, sram_addr, rows, cols)


# ---------------------------------------------------------------- compute

def enc_mrows(operand, rows, partial=PARTIAL, dtype=DT_FP16):
    """ver.08 rows word plus the v09 dtype in spare bits [26:25]."""
    return _enc_mrows_0818(operand, rows, partial) | ((dtype & 3) << 25)


def enc_mcols(operand, cols, partial=PARTIAL, dtype=DT_FP16):
    return _enc_mcols_0818(operand, cols, partial) | ((dtype & 3) << 25)


def enc_scale_addr(which, value, high=False):
    """0x8A (a_scale) / 0x8B (w_scale): FP32 scale-vector SRAM nibble address,
    two-half form like 0x80."""
    if which not in (OP_ASCALE, OP_WSCALE):
        raise V09EncodeError(f"scale opcode must be 0x8A/0x8B, got {which:#x}")
    half = (int(value) >> 16 if high else int(value)) & 0xFFFF
    return ((1 if high else 0) << 29) | (half << 8) | which


def enc_ascale(address):
    return [enc_scale_addr(OP_ASCALE, address, False),
            enc_scale_addr(OP_ASCALE, address, True)]


def enc_wscale(address):
    return [enc_scale_addr(OP_WSCALE, address, False),
            enc_scale_addr(OP_WSCALE, address, True)]


def enc_vquant():
    """FP16 -> packed integer store; the format (INT8/INT4) comes from the
    destination descriptor's dtype, not from an instruction bit."""
    return OP_VQUANT


def enc_vdequant():
    """Packed integer -> FP16-bound floats; format from the source
    descriptor's dtype."""
    return OP_VDEQUANT


def decode_dma(words):
    """Round-trip decoder for one 4-word DMA instruction (tests/tools)."""
    if len(words) != DMA_WORDS:
        raise V09EncodeError(f"DMA instruction needs {DMA_WORDS} words")
    opcode = words[0] & 0xFF
    if opcode not in (OP_GLOAD, OP_GSTORE):
        raise V09EncodeError(f"not a DMA opcode: {opcode:#x}")
    return {
        "op": "gload" if opcode == OP_GLOAD else "gstore",
        "g_addr": int(words[1]),
        "g_stride": int(words[2]),
        "sram_addr": int(words[0]) >> 8,
        "rows": int(words[3]) >> 16,
        "cols": int(words[3]) & 0xFFFF,
    }
