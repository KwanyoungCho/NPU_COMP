"""ISA ver.09 word encoders (memory layer: DMA + control).

Units follow d_compiler/ISA_V09.md: global addresses/strides/cols are 32-bit
cells; SRAM addresses are 4-bit nibbles (24 effective bits) and must sit on an
8-nibble (cell) boundary for DMA.  Compute-side encodings arrive with N3.
"""
from __future__ import annotations

OP_NOP = 0x00
OP_SNAPSHOT = 0xF0
OP_HALT = 0xFF
OP_GLOAD = 0xA0
OP_GSTORE = 0xA8

SRAM_NIBBLES = 8 * 1024 * 1024 * 2  # 8 MiB, 2^24 nibbles
DMA_WORDS = 5


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
    return [opcode, g_addr, g_stride, sram_addr, (rows << 16) | cols]


def enc_gload(g_addr, g_stride, sram_addr, rows, cols):
    """global[g_addr + r*g_stride ..+cols) cells -> SRAM[sram_addr..] nibbles."""
    return _enc_dma(OP_GLOAD, g_addr, g_stride, sram_addr, rows, cols)


def enc_gstore(g_addr, g_stride, sram_addr, rows, cols):
    """SRAM[sram_addr..] nibbles -> global[g_addr + r*g_stride ..+cols) cells."""
    return _enc_dma(OP_GSTORE, g_addr, g_stride, sram_addr, rows, cols)


def decode_dma(words):
    """Round-trip decoder for one 5-word DMA instruction (tests/tools)."""
    if len(words) != DMA_WORDS:
        raise V09EncodeError(f"DMA instruction needs {DMA_WORDS} words")
    opcode = words[0] & 0xFF
    if opcode not in (OP_GLOAD, OP_GSTORE):
        raise V09EncodeError(f"not a DMA opcode: {opcode:#x}")
    return {
        "op": "gload" if opcode == OP_GLOAD else "gstore",
        "g_addr": int(words[1]),
        "g_stride": int(words[2]),
        "sram_addr": int(words[3]),
        "rows": int(words[4]) >> 16,
        "cols": int(words[4]) & 0xFFFF,
    }
