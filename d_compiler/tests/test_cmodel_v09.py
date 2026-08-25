"""N1 gate: v09 simulator skeleton — control flow, host I/O, counters."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

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


if __name__ == "__main__":
    test_halt_writes_input_image_unchanged()
    test_snapshot_appends_checkpoint_then_halt_appends_final()
    test_missing_halt_is_error()
    test_unknown_opcode_is_error()
    test_words_after_halt_do_not_execute()
    test_counters_reported_on_error_path()
    print("ALL V09 C-MODEL N1 TESTS PASSED")
