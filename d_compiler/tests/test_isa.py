"""Validate the ISA encoder against the GIVEN c-model's real instruction binaries.

Two independent checks:
  (A) golden constants taken verbatim from b_program/inst_*/w_support_program_mem_gen.cpp
      -> proves the encoder API produces the exact words the original generators did.
  (B) decode->re-encode round-trip over EVERY word of all 55 example program_memory.bin
      (+ a_npu) -> proves the bit-field layout matches the real binaries with no
      stray/unmodeled bits.

Run:  python d_compiler/tests/test_isa.py
"""
import os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                 # repo root
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))
from npu_compiler import isa
from npu_compiler.isa import Asm, reencode, read_program_bin, SRC1, DST, IMM, SCALAR, VECTOR, AND


def test_golden_constants():
    """Each expected value is the literal expression from a real generator .cpp."""
    cases = {
        # inst_0001 vector_add_immediate
        "addr lo src1=5":   (isa.enc_addr_lo(SRC1, 5),        (0 << 30) + (0 << 29) + (5 << 8) + 0x80),
        "addr hi src1":     (isa.enc_addr_hi(SRC1, 5),        (0 << 30) + (1 << 29) + (0 << 8) + 0x80),
        "addr lo dst=32":   (isa.enc_addr_lo(DST, 32),        (2 << 30) + (0 << 29) + (32 << 8) + 0x80),
        "vlen 8":           (isa.enc_vlen(8),                 (8 << 8) + 0x82),
        "load vec src1":    (isa.enc_load(0, 0),              (0 << 31) + (0 << 30) + 0x90),
        "add imm 3":        (isa.enc_add(IMM, 3),             (0 << 30) + (3 << 8) + 0x01),
        "save vec":         (isa.enc_save(0),                 (0 << 31) + 0x98),
        # inst_0021 logical AND immediate (3)
        "logical AND imm3": (isa.enc_logical(AND, IMM, 3),    (0 << 30) + (0 << 27) + (3 << 8) + 0x08),
        # inst_0031 shift immediate 0xFFFF (=-1, divide by 2)
        "shift -1 imm":     (isa.enc_shift(0xFFFF, IMM),      (0 << 30) + (0xFFFF << 8) + 0x09),
        # inst_0091 compare immediate 8
        "compare imm8":     (isa.enc_compare(IMM, 8),         (0 << 30) + (8 << 8) + 0x11),
        # inst_0112 max scalar / inst_0102 min scalar (imm 7)
        "max scalar 7":     (isa.enc_minmax(True, SCALAR, 7), (1 << 30) + (1 << 28) + (7 << 8) + 0x12),
        "min scalar 7":     (isa.enc_minmax(False, SCALAR, 7),(1 << 30) + (0 << 28) + (7 << 8) + 0x12),
        # inst_0124/0125 convert
        "convert i2f":      (isa.enc_convert(0),              (0 << 31) + 0x13),
        "convert f2i":      (isa.enc_convert(1),              (1 << 31) + 0x13),
        # inst_0081 sqrt / inst_0123 exp
        "sqrt":             (isa.enc_sqrt(),                  0x0E),
        "exp":              (isa.enc_exp(),                   0x0F),
        # inst_1023 matrix mul matrix: tiles, matrix load, matmul, matrix save
        "tile A 2x3":       (isa.enc_tile(0, 2, 3),           (0 << 31) + (3 << 16) + (2 << 8) + 0x88),
        "tile B 3x2":       (isa.enc_tile(1, 3, 2),           (1 << 31) + (2 << 16) + (3 << 8) + 0x88),
        "load matrix src1": (isa.enc_load(1, 0),              (1 << 31) + (0 << 30) + 0x90),
        "matmul vector":    (isa.enc_m_mul(VECTOR),           (2 << 30) + 0x42),
        "save matrix":      (isa.enc_save(1),                 (1 << 31) + 0x98),
    }
    bad = {k: (got, exp) for k, (got, exp) in cases.items() if got != exp}
    assert not bad, "golden mismatch: " + "; ".join(
        f"{k}: got {hex(g)} exp {hex(e)}" for k, (g, e) in bad.items())
    return len(cases)


def test_roundtrip_all_bins():
    bins = sorted(glob.glob(os.path.join(ROOT, "b_program", "*", "program_memory.bin")))
    bins += [os.path.join(ROOT, "a_npu", "program_memory.bin")]
    bins = [b for b in bins if os.path.exists(b)]
    assert bins, "no program_memory.bin found"
    total_words = 0
    nfiles = 0
    for path in bins:
        words = read_program_bin(path)
        nfiles += 1
        for idx, w in enumerate(words):
            re = reencode(w)
            assert re is not None, f"{os.path.relpath(path, ROOT)} word#{idx}: unrecognized opcode in {hex(w)}"
            assert re == w, f"{os.path.relpath(path, ROOT)} word#{idx}: reencode {hex(re)} != {hex(w)}"
            total_words += 1
    return nfiles, total_words


def test_asm_builder_matches_encoders():
    """The Asm builder must emit exactly the low-level encoders (incl. 2-word addr)."""
    a = Asm()
    a.addr(SRC1, 0x12345).vlen(8).load(0, 0).v_add(IMM, 3).addr(DST, 32).save(0).halt()
    expect = [
        isa.enc_addr_lo(SRC1, 0x12345), isa.enc_addr_hi(SRC1, 0x12345),
        isa.enc_vlen(8), isa.enc_load(0, 0), isa.enc_add(IMM, 3),
        isa.enc_addr_lo(DST, 32), isa.enc_addr_hi(DST, 32), isa.enc_save(0), isa.enc_halt(),
    ]
    assert a.words == expect, f"{a.words} != {expect}"
    # 32-bit address really splits into lo/hi 16
    assert isa.enc_addr_lo(SRC1, 0x12345) == ((0x2345) << 8) | 0x80
    assert isa.enc_addr_hi(SRC1, 0x12345) == (1 << 29) | ((0x1) << 8) | 0x80
    return len(a.words)


if __name__ == "__main__":
    ng = test_golden_constants()
    print(f"[PASS] golden constants: {ng} cases match the original generators")
    nb = test_asm_builder_matches_encoders()
    print(f"[PASS] Asm builder: {nb} words match low-level encoders")
    nfiles, nwords = test_roundtrip_all_bins()
    print(f"[PASS] round-trip: {nwords} words across {nfiles} real program_memory.bin reencode identically")
    print("ALL ISA TESTS PASSED")
