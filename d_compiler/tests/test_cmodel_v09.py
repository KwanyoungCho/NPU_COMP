"""N1/N2 gate: v09 simulator — control flow, host I/O, counters, DMA."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler.isa_v09 import (
    V09EncodeError, decode_dma, enc_gload, enc_gstore, enc_halt)
from npu_compiler.v09_runtime import V09Error, run

HALT = 0x000000FF
SNAPSHOT = 0x000000F0
NOP = 0x00000000


def cells(*values):
    return np.asarray(values, dtype=np.uint32)


def _expect_error(program, image, needle):
    try:
        run(program, image)
    except V09Error as error:
        assert needle in str(error), f"unexpected error text: {error}"
        return error
    raise AssertionError(f"expected V09Error containing {needle!r}")


def test_halt_writes_input_image_unchanged():
    image = cells(0x11223344, 0xDEADBEEF, 0x00000000, 0xFFFFFFFF)
    images, counters = run([HALT], image)
    assert len(images) == 1
    np.testing.assert_array_equal(images[0], image)
    assert counters["halt"] == 1
    assert counters["words_executed"] == 1


def test_snapshot_appends_checkpoint_then_halt_appends_final():
    image = cells(1, 2, 3)
    images, counters = run([NOP, SNAPSHOT, NOP, HALT], image)
    assert len(images) == 2
    np.testing.assert_array_equal(images[0], image)
    np.testing.assert_array_equal(images[1], image)
    assert counters["snapshot"] == 1
    assert counters["halt"] == 1
    assert counters["nop"] == 2
    assert counters["words_executed"] == 4


def test_missing_halt_is_error():
    _expect_error([NOP, NOP], cells(7), "without HALT")


def test_unknown_opcode_is_error():
    _expect_error([0x00000042, HALT], cells(7), "unknown opcode")


def test_words_after_halt_do_not_execute():
    images, counters = run([HALT, 0x00000042], cells(5))
    assert len(images) == 1
    assert counters["words_executed"] == 1


def test_counters_reported_on_error_path():
    error = _expect_error([NOP, 0x00000041], cells(1), "unknown opcode")
    assert error.counters.get("nop") == 1


# ---------------------------------------------------------------- N2: DMA


def test_dma_roundtrip_is_identity_for_raw_cells():
    rng = np.random.default_rng(9)
    image = rng.integers(0, 2**32, size=64, dtype=np.uint32)
    program = (enc_gload(0, 16, 0, rows=4, cols=16)
               + enc_gstore(64 - 64, 16, 0, rows=4, cols=16)  # overwrite in place
               + [enc_halt()])
    images, counters = run(program, image)
    np.testing.assert_array_equal(images[0], image)
    assert counters["dma_cells_loaded"] == 64
    assert counters["dma_cells_stored"] == 64


def test_dma_moves_block_to_new_region():
    image = np.zeros(32, dtype=np.uint32)
    image[:8] = np.arange(1, 9, dtype=np.uint32)
    program = (enc_gload(0, 8, 0, rows=1, cols=8)
               + enc_gstore(16, 8, 0, rows=1, cols=8)
               + [enc_halt()])
    images, _ = run(program, image)
    np.testing.assert_array_equal(images[0][16:24], image[:8])
    np.testing.assert_array_equal(images[0][:16], image[:16])


def test_dma_strided_subblock_gather():
    # global matrix: 4 rows x 8 cells (row stride 8); move the 2-cell-wide
    # column block starting at cell 3 of rows 1..2 into a packed region.
    image = np.arange(64, dtype=np.uint32)
    src = 8 * 1 + 3
    program = (enc_gload(src, 8, 0, rows=2, cols=2)
               + enc_gstore(40, 2, 0, rows=2, cols=2)
               + [enc_halt()])
    images, _ = run(program, image)
    expected = image.reshape(8, 8)[1:3, 3:5].reshape(-1)
    np.testing.assert_array_equal(images[0][40:44], expected)


def test_dma_wide_stride_beyond_16bit():
    stride = 100_000  # V3-027 class: stride no longer capped at 16 bits
    image = np.zeros(stride + 2, dtype=np.uint32)
    image[0], image[stride] = 0xAAAA5555, 0x1234ABCD
    program = (enc_gload(0, stride, 0, rows=2, cols=1)
               + enc_gstore(1, 1, 0, rows=2, cols=1)
               + [enc_halt()])
    images, _ = run(program, image)
    np.testing.assert_array_equal(images[0][1:3],
                                  [0xAAAA5555, 0x1234ABCD])


def test_dma_little_endian_fp16_cell_mapping():
    # Two FP16 values packed per cell: after a load+store roundtrip through
    # SRAM the byte-level layout (section 1.2 mapping) must be untouched.
    values = np.asarray([1.0, -2.5, 0.5, 65504.0], dtype="<f2")
    image = np.concatenate([values.view("<u4"), np.zeros(2, dtype=np.uint32)])
    program = (enc_gload(0, 2, 8, rows=1, cols=2)
               + enc_gstore(2, 2, 8, rows=1, cols=2)
               + [enc_halt()])
    images, _ = run(program, image)
    roundtrip = images[0][2:4].astype("<u4").view("<f2")
    np.testing.assert_array_equal(roundtrip, values)


def test_dma_bounds_and_alignment_errors():
    image = np.zeros(16, dtype=np.uint32)
    bad_sram_align = [0xA0, 0, 1, 4, (1 << 16) | 1]        # nibble 4: mid-cell
    bad_sram_high = [0xA0, 0, 1, 1 << 24, (1 << 16) | 1]   # upper bits set
    bad_global = [0xA0, 15, 1, 0, (1 << 16) | 2]           # runs past cell 16
    bad_sram_range = [0xA0, 0, 1, (2**24 - 8), (1 << 16) | 2]
    zero_cols = [0xA0, 0, 1, 0, (1 << 16) | 0]
    truncated = [0xA0, 0, 1]
    _expect_error(bad_sram_align + [enc_halt()], image, "aligned")
    _expect_error(bad_sram_high + [enc_halt()], image, "24-bit")
    _expect_error(bad_global + [enc_halt()], image, "global range")
    _expect_error(bad_sram_range + [enc_halt()], image, "SRAM range")
    _expect_error(zero_cols + [enc_halt()], image, "zero rows or cols")
    _expect_error(truncated, image, "truncated")


def test_isa_v09_dma_encode_decode_roundtrip():
    words = enc_gload(0x1234, 0x1F000, 512, rows=64, cols=32)
    decoded = decode_dma(words)
    assert decoded == {"op": "gload", "g_addr": 0x1234, "g_stride": 0x1F000,
                       "sram_addr": 512, "rows": 64, "cols": 32}
    words = enc_gstore(7, 1, 8, rows=1, cols=1)
    assert decode_dma(words)["op"] == "gstore"
    for bad in (
        lambda: enc_gload(1 << 32, 1, 0, 1, 1),
        lambda: enc_gload(0, 1, 4, 1, 1),        # misaligned nibble
        lambda: enc_gload(0, 1, 1 << 24, 1, 1),  # beyond SRAM
        lambda: enc_gload(0, 1, 0, 0, 1),        # zero rows
        lambda: enc_gload(0, 1, 0, 1, 1 << 16),  # cols too wide
    ):
        try:
            bad()
        except V09EncodeError:
            pass
        else:
            raise AssertionError("expected V09EncodeError")


if __name__ == "__main__":
    test_halt_writes_input_image_unchanged()
    test_snapshot_appends_checkpoint_then_halt_appends_final()
    test_missing_halt_is_error()
    test_unknown_opcode_is_error()
    test_words_after_halt_do_not_execute()
    test_counters_reported_on_error_path()
    test_dma_roundtrip_is_identity_for_raw_cells()
    test_dma_moves_block_to_new_region()
    test_dma_strided_subblock_gather()
    test_dma_wide_stride_beyond_16bit()
    test_dma_little_endian_fp16_cell_mapping()
    test_dma_bounds_and_alignment_errors()
    test_isa_v09_dma_encode_decode_roundtrip()
    print("ALL V09 C-MODEL N1+N2 TESTS PASSED")
