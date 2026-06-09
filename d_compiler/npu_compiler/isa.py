"""NPU ISA instruction encoder.

THIS IS NOT A SIMULATOR / EXECUTOR. It only encodes 32-bit instruction words
into the `program_memory.bin` format that the GIVEN c-model `_poc/mysim` decodes.
The single source of truth for the bit layout is `_poc/mysim.cpp`'s decode logic
(see the `for(pc...)` loop). The encodings below were cross-checked byte-exact
against the original example generators in `b_program/inst_*/w_support_program_mem_gen.cpp`.

Instruction word (uint32, little-endian in the file). Low byte = opcode.
Common fields:
  [31:30] operand mode for compute (0=immediate, 1=scalar, 2=vector/matrix)
  [29]    activation flag (matrix compute)
  [23:8]  16-bit immediate / address-half / length / shift amount
Control opcodes use [31] / [30] / [29] as selector bits (see each encoder).
"""
import struct

# ---- operand selectors (for set_addr / load) ----
SRC1, SRC2, DST = 0, 1, 2
# ---- compute operand modes ----
IMM, SCALAR, VECTOR = 0, 1, 2
# ---- logical sub-ops (opcode 0x08, bits [29:27]) ----
AND, OR, NOT, XOR, NAND, NOR = 0, 1, 2, 3, 4, 5
# ---- type-convert direction (opcode 0x13, bit [31]) ----
INT2FLOAT, FLOAT2INT = 0, 1

MASK32 = 0xFFFFFFFF


def _u16(x):
    return x & 0xFFFF


# ============================ low-level encoders ============================
# Control
def enc_nop():
    return 0

def enc_halt():
    return 0xFF                                  # mysim: op==0xFF -> HALT

def enc_addr_lo(operand, value):
    return ((operand & 3) << 30) | (0 << 29) | (_u16(value) << 8) | 0x80

def enc_addr_hi(operand, value):
    return ((operand & 3) << 30) | (1 << 29) | (_u16(value >> 16) << 8) | 0x80

def enc_vlen(n):
    return (_u16(n) << 8) | 0x82

def enc_tile(sel, d1, d2):
    # sel: 0 = first(A), 1 = second(B). d1 -> tA [15:8], d2 -> tB [23:16]
    return ((sel & 1) << 31) | ((d2 & 0xFF) << 16) | ((d1 & 0xFF) << 8) | 0x88

def enc_load(matrix, operand):
    # operand 0 -> PE_in_1, 1 -> PE_in_2
    return ((matrix & 1) << 31) | ((operand & 1) << 30) | 0x90

def enc_save(matrix):
    return ((matrix & 1) << 31) | 0x98

# Vector / matrix compute
def _enc_simple(op, mode, imm):
    return ((mode & 3) << 30) | (_u16(imm) << 8) | (op & 0xFF)

def enc_add(mode=VECTOR, imm=0):     return _enc_simple(0x01, mode, imm)
def enc_sub(mode=VECTOR, imm=0):     return _enc_simple(0x02, mode, imm)
def enc_mul(mode=VECTOR, imm=0):     return _enc_simple(0x0A, mode, imm)
def enc_div(mode=VECTOR, imm=0):     return _enc_simple(0x0B, mode, imm)
def enc_muladd(mode=VECTOR, imm=0):  return _enc_simple(0x0C, mode, imm)
def enc_move(mode=VECTOR, imm=0):    return _enc_simple(0x0D, mode, imm)
def enc_compare(mode=VECTOR, imm=0): return _enc_simple(0x11, mode, imm)

def enc_sqrt():                      return 0x0E
def enc_exp():                       return 0x0F

def enc_logical(sub, mode=VECTOR, imm=0):
    return ((mode & 3) << 30) | ((sub & 7) << 27) | (_u16(imm) << 8) | 0x08

def enc_shift(amount, mode=IMM):
    # amount is a signed 16-bit shift (negative => right shift, a * 2^amount)
    return ((mode & 3) << 30) | (_u16(amount) << 8) | 0x09

def enc_minmax(is_max, mode=VECTOR, imm=0):
    return ((mode & 3) << 30) | ((1 if is_max else 0) << 28) | (_u16(imm) << 8) | 0x12

def enc_convert(direction):
    return ((direction & 1) << 31) | 0x13

def _enc_matrix(op, mode, imm, act):
    return ((mode & 3) << 30) | ((1 if act else 0) << 29) | (_u16(imm) << 8) | (op & 0xFF)

def enc_m_add(mode=VECTOR, imm=0, act=False):  return _enc_matrix(0x40, mode, imm, act)
def enc_m_sub(mode=VECTOR, imm=0, act=False):  return _enc_matrix(0x41, mode, imm, act)
def enc_m_mul(mode=VECTOR, imm=0, act=False):  return _enc_matrix(0x42, mode, imm, act)
def enc_m_move(mode=VECTOR, imm=0, act=False): return _enc_matrix(0x43, mode, imm, act)


# ============================ assembler builder ============================
class Asm:
    """Accumulates NPU instruction words. Emits program_memory.bin bytes.

    Mirrors how the example generators sequence instructions, but is written
    fresh from the mysim decode spec (not copied from b_program_examples/npu.py).
    """

    def __init__(self):
        self.words = []

    def _emit(self, w):
        self.words.append(w & MASK32)
        return self

    # control
    def nop(self):                 return self._emit(enc_nop())
    def halt(self):                return self._emit(enc_halt())
    def addr(self, operand, value):
        self._emit(enc_addr_lo(operand, value))
        return self._emit(enc_addr_hi(operand, value))
    def vlen(self, n):             return self._emit(enc_vlen(n))
    def tile(self, sel, d1, d2):   return self._emit(enc_tile(sel, d1, d2))
    def load(self, matrix, operand): return self._emit(enc_load(matrix, operand))
    def save(self, matrix):        return self._emit(enc_save(matrix))

    # vector compute
    def v_add(self, mode=VECTOR, imm=0):     return self._emit(enc_add(mode, imm))
    def v_sub(self, mode=VECTOR, imm=0):     return self._emit(enc_sub(mode, imm))
    def v_mul(self, mode=VECTOR, imm=0):     return self._emit(enc_mul(mode, imm))
    def v_div(self, mode=VECTOR, imm=0):     return self._emit(enc_div(mode, imm))
    def v_muladd(self, mode=VECTOR, imm=0):  return self._emit(enc_muladd(mode, imm))
    def v_move(self, mode=VECTOR, imm=0):    return self._emit(enc_move(mode, imm))
    def v_compare(self, mode=VECTOR, imm=0): return self._emit(enc_compare(mode, imm))
    def v_sqrt(self):                        return self._emit(enc_sqrt())
    def v_exp(self):                         return self._emit(enc_exp())
    def v_logical(self, sub, mode=VECTOR, imm=0): return self._emit(enc_logical(sub, mode, imm))
    def v_shift(self, amount, mode=IMM):     return self._emit(enc_shift(amount, mode))
    def v_min(self, mode=VECTOR, imm=0):     return self._emit(enc_minmax(False, mode, imm))
    def v_max(self, mode=VECTOR, imm=0):     return self._emit(enc_minmax(True, mode, imm))
    def v_convert(self, direction):          return self._emit(enc_convert(direction))

    # matrix compute
    def m_add(self, mode=VECTOR, imm=0, act=False):  return self._emit(enc_m_add(mode, imm, act))
    def m_sub(self, mode=VECTOR, imm=0, act=False):  return self._emit(enc_m_sub(mode, imm, act))
    def m_mul(self, mode=VECTOR, imm=0, act=False):  return self._emit(enc_m_mul(mode, imm, act))
    def m_move(self, mode=VECTOR, imm=0, act=False): return self._emit(enc_m_move(mode, imm, act))

    def to_bytes(self):
        # uint32 little-endian words + trailing newline (matches the original .bin format)
        return b"".join(struct.pack("<I", w) for w in self.words) + b"\n"


# ============================ decode / round-trip ============================
def reencode(w):
    """Canonicalize a 32-bit word by extracting its meaningful fields (per mysim
    decode) and rebuilding it. For words produced by the real generators this
    returns the identical word, which makes it a strong field-layout validator:
        assert reencode(w) == w   for every word in every real program_memory.bin
    Returns None for an unrecognized opcode (so tests can flag it)."""
    w &= MASK32
    if w == 0:
        return 0
    op = w & 0xFF
    if op == 0xFF:
        return 0xFF
    if op == 0x80:
        operand = (w >> 30) & 3; hi = (w >> 29) & 1; v = (w >> 8) & 0xFFFF
        return ((operand) << 30) | (hi << 29) | (v << 8) | 0x80
    if op == 0x82:
        return enc_vlen((w >> 8) & 0xFFFF)
    if op == 0x88:
        return enc_tile((w >> 31) & 1, (w >> 8) & 0xFF, (w >> 16) & 0xFF)
    if op == 0x90:
        return enc_load((w >> 31) & 1, (w >> 30) & 1)
    if op == 0x98:
        return enc_save((w >> 31) & 1)
    mode = (w >> 30) & 3; imm = (w >> 8) & 0xFFFF
    if op in (0x01, 0x02, 0x0A, 0x0B, 0x0C, 0x0D, 0x11):
        return _enc_simple(op, mode, imm)
    if op in (0x0E, 0x0F):
        return op
    if op == 0x08:
        return enc_logical((w >> 27) & 7, mode, imm)
    if op == 0x09:
        return enc_shift(imm, mode)
    if op == 0x12:
        return enc_minmax((w >> 28) & 1, mode, imm)
    if op == 0x13:
        return enc_convert((w >> 31) & 1)
    if op in (0x40, 0x41, 0x42, 0x43):
        return _enc_matrix(op, mode, (w >> 8) & 0xFFFF, (w >> 29) & 1)
    return None


def read_program_bin(path):
    """Read a program_memory.bin into a list of uint32 words (ignores a trailing
    partial/newline byte, matching mysim's pn = filesize/4)."""
    data = open(path, "rb").read()
    n = len(data) // 4
    return list(struct.unpack("<%dI" % n, data[: n * 4]))
