"""Bit-exact tests for the supplied ver.08 instruction format."""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler import isa_0818 as isa
from npu_compiler.isa_0818 import DST, MAIN, PARTIAL, SRC1, SRC2, Asm


def test_new_field_layout():
    cases = {
        "src1 main address": isa.enc_addr_lo(SRC1, 2, MAIN),
        "src1 partial address": isa.enc_addr_lo(SRC1, 4, PARTIAL),
        "src2 partial rows": isa.enc_mrows(SRC2, 2, PARTIAL),
        "src2 main cols": isa.enc_mcols(SRC2, 9, MAIN),
        "dst partial address": isa.enc_addr_lo(DST, 202, PARTIAL),
        "gelu add": isa.enc_m_add(activation=isa.ACT_GELU),
        "mac matmul": isa.enc_m_mul(mac=True),
        "reduce max": isa.enc_reduce_max(),
        "finish": isa.enc_finish(),
    }
    expected = {
        "src1 main address": 0x00000280,
        "src1 partial address": 0x10000480,
        "src2 partial rows": 0x60000288,
        "src2 main cols": 0x40000989,
        "dst partial address": 0x9000CA80,
        "gelu add": 0xB0000040,
        "mac matmul": 0x88000042,
        "reduce max": 0x19,
        "finish": 0xF0,
    }
    assert cases == expected


def test_matrix_region_matches_vendor_generator():
    a = Asm()
    a.matrix_region(SRC1, 2, 8, 10, 4, 2, 3)
    a.matrix_region(SRC2, 6, 6, 9, 9, 2, 3)
    expected = [
        0x00000280, 0x20000080, 0x00000888, 0x00000A89,
        0x10000480, 0x30000080, 0x20000288, 0x20000389,
        0x40000680, 0x60000080, 0x40000688, 0x40000989,
        0x50000980, 0x70000080, 0x60000288, 0x60000389,
    ]
    assert a.words == expected


def test_roundtrip_every_0818_example():
    bins = sorted(glob.glob(os.path.join(
        ROOT, "0818_npu_update", "b_program", "*", "program_memory.bin")))
    assert len(bins) == 64
    count = 0
    for path in bins:
        for index, word in enumerate(isa.read_program_bin(path)):
            encoded = isa.reencode(word)
            assert encoded == word, f"{path} word {index}: {encoded!r} != {word:#x}"
            count += 1
    assert count == 13766


if __name__ == "__main__":
    test_new_field_layout()
    test_matrix_region_matches_vendor_generator()
    test_roundtrip_every_0818_example()
    print("ALL 0818 ISA TESTS PASSED")
