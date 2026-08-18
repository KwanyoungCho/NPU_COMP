"""Encoder for the vendor NPU instruction format ver.08 (2026-08-03).

This module intentionally models the vendor format, including its stateful
main/partial matrix descriptors.  It does not contain compatibility shims for
the 0710 encoder; the old :mod:`npu_compiler.isa` remains the v07 oracle.
"""
from __future__ import annotations

import struct

SRC1, SRC2, DST = 0, 1, 2
IMM, SCALAR, VECTOR = 0, 1, 2
MAIN, PARTIAL = 0, 1
ACT_OFF, ACT_RESERVED, ACT_SILU, ACT_GELU = 0, 1, 2, 3
AND, OR, NOT, XOR, NAND, NOR = 0, 1, 2, 3, 4, 5
INT2FLOAT, FLOAT2INT = 0, 1
MASK32 = 0xFFFFFFFF


def _u16(x):
    return int(x) & 0xFFFF


def enc_nop():
    return 0


def enc_finish():
    """Write ``saved_G_buffer_data.bin``.  The vendor does not actually halt."""
    return 0xF0


def enc_addr_half(operand, value, high=False, partial=PARTIAL):
    half = (_u16(value >> 16) if high else _u16(value))
    return (((operand & 3) << 30) | ((1 if high else 0) << 29)
            | ((partial & 1) << 28) | (half << 8) | 0x80)


def enc_addr_lo(operand, value, partial=PARTIAL):
    return enc_addr_half(operand, value, False, partial)


def enc_addr_hi(operand, value, partial=PARTIAL):
    return enc_addr_half(operand, value, True, partial)


def enc_vlen(n):
    return (_u16(n) << 8) | 0x82


def enc_mrows(operand, rows, partial=PARTIAL):
    return (((operand & 3) << 30) | ((partial & 1) << 29)
            | (_u16(rows) << 8) | 0x88)


def enc_mcols(operand, cols, partial=PARTIAL):
    return (((operand & 3) << 30) | ((partial & 1) << 29)
            | (_u16(cols) << 8) | 0x89)


def enc_load(matrix, operand, strided=0, ncols=0, start=0):
    return (((matrix & 1) << 31) | ((operand & 1) << 30)
            | ((strided & 1) << 29) | ((ncols & 0xFF) << 16)
            | ((start & 0xFF) << 8) | 0x90)


def enc_save(matrix, strided=0, ncols=0, start=0):
    return (((matrix & 1) << 31) | ((strided & 1) << 29)
            | ((ncols & 0xFF) << 16) | ((start & 0xFF) << 8) | 0x98)


def _enc_simple(op, mode, imm):
    return ((mode & 3) << 30) | (_u16(imm) << 8) | (op & 0xFF)


def enc_add(mode=VECTOR, imm=0): return _enc_simple(0x01, mode, imm)
def enc_sub(mode=VECTOR, imm=0): return _enc_simple(0x02, mode, imm)
def enc_mul(mode=VECTOR, imm=0): return _enc_simple(0x0A, mode, imm)
def enc_div(mode=VECTOR, imm=0): return _enc_simple(0x0B, mode, imm)
def enc_muladd(mode=VECTOR, imm=0): return _enc_simple(0x0C, mode, imm)
def enc_move(mode=VECTOR, imm=0): return _enc_simple(0x0D, mode, imm)
def enc_compare(mode=VECTOR, imm=0): return _enc_simple(0x11, mode, imm)
def enc_sqrt(): return 0x0E
def enc_exp(): return 0x0F


def enc_logical(sub, mode=VECTOR, imm=0):
    return ((mode & 3) << 30) | ((sub & 7) << 27) | (_u16(imm) << 8) | 0x08


def enc_shift(amount, mode=IMM):
    return _enc_simple(0x09, mode, amount)


def enc_minmax(is_max, mode=VECTOR, imm=0):
    return ((mode & 3) << 30) | ((1 if is_max else 0) << 28) | (_u16(imm) << 8) | 0x12


def enc_convert(direction):
    return ((direction & 1) << 31) | 0x13


def enc_reduce_sum(): return 0x14
def enc_reduce_max(): return 0x19
def enc_sign_inv(): return 0x16
def enc_copy(): return 0x17
def enc_cossin(is_sin): return ((1 if is_sin else 0) << 27) | 0x18


def enc_broadcast(mode=SCALAR, imm=0, high=False):
    return (((mode & 3) << 30) | ((1 if high else 0) << 29)
            | (_u16(imm) << 8) | 0x15)


def _enc_matrix(op, mode, imm, activation=ACT_OFF, mac=False):
    return (((mode & 3) << 30) | ((activation & 3) << 28)
            | ((1 if mac else 0) << 27) | (_u16(imm) << 8) | (op & 0xFF))


def enc_m_add(mode=VECTOR, imm=0, activation=ACT_OFF):
    return _enc_matrix(0x40, mode, imm, activation)


def enc_m_sub(mode=VECTOR, imm=0, activation=ACT_OFF):
    return _enc_matrix(0x41, mode, imm, activation)


def enc_m_mul(mode=VECTOR, imm=0, activation=ACT_OFF, mac=False):
    return _enc_matrix(0x42, mode, imm, activation, mac)


def enc_m_move(mode=VECTOR, imm=0, activation=ACT_OFF):
    return _enc_matrix(0x43, mode, imm, activation)


class Asm:
    """State-independent ver.08 assembler.

    ``addr`` defaults to PARTIAL because vendor load/save consumes the partial
    address even for vectors.  Matrix clients should use ``matrix_region`` so
    the main stride and the transferred sub-region are both explicit.
    """

    format_version = "0818"

    def __init__(self):
        self.words = []
        self.tags = []
        self._role = None

    def role(self, name):
        asm = self
        class _Role:
            def __enter__(s):
                s.prev = asm._role
                asm._role = name
            def __exit__(s, *exc):
                asm._role = s.prev
        return _Role()

    def _emit(self, word):
        self.words.append(word & MASK32)
        self.tags.append(self._role)
        return self

    def nop(self): return self._emit(enc_nop())
    def finish(self): return self._emit(enc_finish())

    def addr(self, operand, value, partial=PARTIAL):
        self._emit(enc_addr_lo(operand, value, partial))
        return self._emit(enc_addr_hi(operand, value, partial))

    def shape(self, operand, rows, cols, partial=PARTIAL):
        self._emit(enc_mrows(operand, rows, partial))
        return self._emit(enc_mcols(operand, cols, partial))

    def matrix_region(self, operand, main_addr, main_rows, main_cols,
                      partial_addr=None, partial_rows=None, partial_cols=None):
        if partial_addr is None:
            partial_addr = main_addr
        if partial_rows is None:
            partial_rows = main_rows
        if partial_cols is None:
            partial_cols = main_cols
        self.addr(operand, main_addr, MAIN).shape(operand, main_rows, main_cols, MAIN)
        return self.addr(operand, partial_addr, PARTIAL).shape(
            operand, partial_rows, partial_cols, PARTIAL)

    def vlen(self, n): return self._emit(enc_vlen(n))
    def load(self, matrix, operand, strided=0, ncols=0, start=0):
        return self._emit(enc_load(matrix, operand, strided, ncols, start))
    def save(self, matrix, strided=0, ncols=0, start=0):
        return self._emit(enc_save(matrix, strided, ncols, start))

    def v_add(self, mode=VECTOR, imm=0): return self._emit(enc_add(mode, imm))
    def v_sub(self, mode=VECTOR, imm=0): return self._emit(enc_sub(mode, imm))
    def v_mul(self, mode=VECTOR, imm=0): return self._emit(enc_mul(mode, imm))
    def v_div(self, mode=VECTOR, imm=0): return self._emit(enc_div(mode, imm))
    def v_muladd(self, mode=VECTOR, imm=0): return self._emit(enc_muladd(mode, imm))
    def v_move(self, mode=VECTOR, imm=0): return self._emit(enc_move(mode, imm))
    def v_compare(self, mode=VECTOR, imm=0): return self._emit(enc_compare(mode, imm))
    def v_sqrt(self): return self._emit(enc_sqrt())
    def v_exp(self): return self._emit(enc_exp())
    def v_logical(self, sub, mode=VECTOR, imm=0): return self._emit(enc_logical(sub, mode, imm))
    def v_shift(self, amount, mode=IMM): return self._emit(enc_shift(amount, mode))
    def v_min(self, mode=VECTOR, imm=0): return self._emit(enc_minmax(False, mode, imm))
    def v_max(self, mode=VECTOR, imm=0): return self._emit(enc_minmax(True, mode, imm))
    def v_convert(self, direction): return self._emit(enc_convert(direction))
    def v_reduce_sum(self): return self._emit(enc_reduce_sum())
    def v_reduce_max(self): return self._emit(enc_reduce_max())
    def v_broadcast(self, mode=SCALAR, imm=0, high=False):
        return self._emit(enc_broadcast(mode, imm, high))
    def v_sign_inv(self): return self._emit(enc_sign_inv())
    def v_copy(self): return self._emit(enc_copy())
    def v_cos(self): return self._emit(enc_cossin(False))
    def v_sin(self): return self._emit(enc_cossin(True))
    def m_add(self, mode=VECTOR, imm=0, activation=ACT_OFF):
        return self._emit(enc_m_add(mode, imm, activation))
    def m_sub(self, mode=VECTOR, imm=0, activation=ACT_OFF):
        return self._emit(enc_m_sub(mode, imm, activation))
    def m_mul(self, mode=VECTOR, imm=0, activation=ACT_OFF, mac=False):
        return self._emit(enc_m_mul(mode, imm, activation, mac))
    def m_move(self, mode=VECTOR, imm=0, activation=ACT_OFF):
        return self._emit(enc_m_move(mode, imm, activation))

    def to_bytes(self):
        return b"".join(struct.pack("<I", w) for w in self.words) + b"\n"


def reencode(word):
    word &= MASK32
    if word == 0: return 0
    op = word & 0xFF
    mode, imm = (word >> 30) & 3, (word >> 8) & 0xFFFF
    if op == 0xF0: return enc_finish()
    if op == 0x80:
        return enc_addr_half(mode, ((word >> 8) & 0xFFFF) << (16 if ((word >> 29) & 1) else 0),
                             bool((word >> 29) & 1), (word >> 28) & 1)
    if op == 0x82: return enc_vlen(imm)
    if op == 0x88: return enc_mrows(mode, imm, (word >> 29) & 1)
    if op == 0x89: return enc_mcols(mode, imm, (word >> 29) & 1)
    if op == 0x90:
        return enc_load((word >> 31) & 1, (word >> 30) & 1, (word >> 29) & 1,
                        (word >> 16) & 0xFF, (word >> 8) & 0xFF)
    if op == 0x98:
        return enc_save((word >> 31) & 1, (word >> 29) & 1,
                        (word >> 16) & 0xFF, (word >> 8) & 0xFF)
    if op in (0x01, 0x02, 0x0A, 0x0B, 0x0C, 0x0D, 0x11): return _enc_simple(op, mode, imm)
    if op in (0x0E, 0x0F, 0x14, 0x16, 0x17, 0x19): return op
    if op == 0x08: return enc_logical((word >> 27) & 7, mode, imm)
    if op == 0x09: return enc_shift(imm, mode)
    if op == 0x12: return enc_minmax((word >> 28) & 1, mode, imm)
    if op == 0x13: return enc_convert((word >> 31) & 1)
    if op == 0x15: return enc_broadcast(mode, imm, bool((word >> 29) & 1))
    if op == 0x18: return enc_cossin((word >> 27) & 1)
    if op in (0x40, 0x41, 0x42, 0x43):
        return _enc_matrix(op, mode, imm, (word >> 28) & 3, bool((word >> 27) & 1))
    return None


def read_program_bin(path):
    data = open(path, "rb").read()
    n = len(data) // 4
    return list(struct.unpack("<%dI" % n, data[:n * 4]))
