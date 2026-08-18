"""Vendor-versus-source parity tests for the 0818 C-model."""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "d_compiler"))

from npu_compiler.isa_0818 import ACT_GELU, DST, IMM, SCALAR, SRC1, SRC2, Asm
from npu_compiler.runtime_0818 import GBUF_CAPACITY, VENDOR_BIN, _libstdcxx_dir, run


def build_source_model(output):
    subprocess.run([
        "g++", "-std=c++11", "-O2", str(ROOT / "_poc" / "mysim_0818.cpp"),
        "-o", str(output),
    ], check=True)


def _source_binary():
    tmp = tempfile.TemporaryDirectory(prefix="mysim0818-test-")
    binary = Path(tmp.name) / "mysim_0818"
    build_source_model(binary)
    return tmp, binary


def _assert_parity(program, gbuf, source):
    vendor, vendor_trace = run(program, gbuf, capture_trace=True)
    ours, source_trace = run(program, gbuf, capture_trace=True, vendor_bin=source)
    assert np.array_equal(vendor, ours, equal_nan=True)
    # The vendor's 20-word header dump reads uninitialized words after a short
    # input program.  Compare the deterministic executed-instruction trace body.
    def body(trace):
        lines = trace.splitlines()
        first = next(index for index, line in enumerate(lines)
                     if line.startswith("p_counter :"))
        normalized = [line.rstrip() for line in lines[first:]]
        while normalized and not normalized[-1]:
            normalized.pop()
        return normalized
    assert body(vendor_trace) == body(source_trace)
    return vendor


def _raw_snapshot(executable, program):
    with tempfile.TemporaryDirectory(prefix="npu0818-multisave-") as directory:
        directory = Path(directory)
        (directory / "program_memory.bin").write_bytes(program.to_bytes())
        zeros = np.zeros(GBUF_CAPACITY, dtype="<f2")
        (directory / "G_buffer_data.bin").write_bytes(zeros.tobytes() + b"\n")
        env = os.environ.copy()
        library = _libstdcxx_dir()
        if library:
            env["LD_LIBRARY_PATH"] = library + (
                ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        subprocess.run([str(executable)], cwd=directory, env=env, check=True,
                       stdout=subprocess.DEVNULL)
        return (directory / "saved_G_buffer_data.bin").read_bytes()


def _snapshot_exists(executable, program):
    with tempfile.TemporaryDirectory(prefix="npu0818-nosave-") as directory:
        directory = Path(directory)
        (directory / "program_memory.bin").write_bytes(program.to_bytes())
        zeros = np.zeros(GBUF_CAPACITY, dtype="<f2")
        (directory / "G_buffer_data.bin").write_bytes(zeros.tobytes() + b"\n")
        env = os.environ.copy()
        library = _libstdcxx_dir()
        if library:
            env["LD_LIBRARY_PATH"] = library + (
                ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        subprocess.run([str(executable)], cwd=directory, env=env, check=True,
                       stdout=subprocess.DEVNULL)
        return (directory / "saved_G_buffer_data.bin").exists()


def test_targeted_vendor_quirks():
    tmp, source = _source_binary()
    try:
        # Arbitrary row-major sub-region, not a packed 64x64 global tile.
        gbuf = np.zeros(100, dtype=np.float16)
        gbuf[:12] = np.arange(1, 13)
        gbuf[20:32] = np.arange(11, 23)
        a = Asm()
        a.matrix_region(SRC1, 0, 3, 4, 4, 2, 3).load(1, SRC1)
        a.matrix_region(SRC2, 20, 3, 4, 24, 2, 3).load(1, SRC2)
        a.m_add()
        a.matrix_region(DST, 40, 3, 4, 44, 2, 3).save(1).finish()
        out = _assert_parity(a, gbuf, source)
        assert np.array_equal(out[[44, 45, 46, 48, 49, 50]], [20, 22, 24, 28, 30, 32])

        # Native reduce-max intentionally reproduces the vendor's zero seed bug.
        a = Asm()
        a.addr(SRC1, 0).vlen(3).load(0, SRC1).v_reduce_max()
        a.addr(DST, 4).save(0).finish()
        out = _assert_parity(a, [-5, -2, -7], source)
        assert out[4] == 0

        # Vendor GELU is x*sigmoid(2*x), not a standard erf/tanh GELU.
        values = np.asarray([-2, -1, 0, 1, 2], dtype=np.float16)
        a = Asm()
        a.matrix_region(SRC1, 0, 1, 5).load(1, SRC1)
        a.m_add(IMM, 0, ACT_GELU)
        a.matrix_region(DST, 8, 1, 5).save(1).finish()
        out = _assert_parity(a, values, source)
        expected = np.asarray(values.astype(np.float32) /
                              (1 + np.exp(-2 * values.astype(np.float32))), dtype=np.float16)
        assert np.array_equal(out[8:13], expected.astype(np.float32))

        # Bit 27 accumulates a second matrix product into the existing PE output.
        gbuf = np.zeros(20, dtype=np.float16)
        gbuf[0:4] = [1, 2, 3, 4]
        a = Asm()
        for mac in (False, True):
            a.matrix_region(SRC1, 0, 1, 2).load(1, SRC1)
            a.matrix_region(SRC2, 2, 2, 1).load(1, SRC2)
            a.m_mul(mac=mac)
        a.matrix_region(DST, 8, 1, 1).save(1).finish()
        out = _assert_parity(a, gbuf, source)
        assert out[8] == 22

        # Native scalar broadcast reads its source directly from G-buffer.
        gbuf[5] = 7
        a = Asm()
        a.vlen(4).v_broadcast(SCALAR, 5).addr(DST, 10).save(0).finish()
        out = _assert_parity(a, gbuf, source)
        assert np.array_equal(out[10:14], np.full(4, 7, dtype=np.float32))

        # 0xF0 is not HALT: repeated snapshots are concatenated and get one final newline.
        a = Asm().finish().finish()
        vendor_saved = _raw_snapshot(VENDOR_BIN, a)
        source_saved = _raw_snapshot(source, a)
        assert source_saved == vendor_saved
        assert len(source_saved) == 2 * GBUF_CAPACITY * 2 + 1
    finally:
        tmp.cleanup()


def test_all_supplied_program_snapshots():
    tmp, source = _source_binary()
    try:
        raw = (ROOT / "0818_npu_update" / "a_npu" / "G_buffer_data.bin").read_bytes()
        gbuf = np.frombuffer(raw[:(len(raw) // 2) * 2], dtype="<f2")[:8192]
        programs = sorted((ROOT / "0818_npu_update" / "b_program").glob(
            "*/program_memory.bin"))
        assert len(programs) == 64
        for path in programs:
            data = path.read_bytes()
            words = list(struct.unpack(f"<{len(data) // 4}I", data[:len(data) // 4 * 4]))
            if 0xF0 not in words:
                words.append(0xF0)
            _assert_parity(words, gbuf, source)
    finally:
        tmp.cleanup()


def test_program_file_is_not_limited_to_32768_words():
    """The vendor fetches the complete malloc-backed file, including word 33000."""
    tmp, source = _source_binary()
    try:
        program = Asm()
        program.words = [0] * 33000 + [0xF0]
        program.tags = [None] * len(program.words)
        zeros = np.zeros(1, dtype=np.float16)
        vendor_saved = run(program, zeros)
        source_saved = run(program, zeros, vendor_bin=source)
        assert np.array_equal(source_saved, vendor_saved)
        assert vendor_saved.size == GBUF_CAPACITY
    finally:
        tmp.cleanup()


def test_process_exit_does_not_implicitly_save():
    tmp, source = _source_binary()
    try:
        program = Asm()
        for _ in range(20):
            program.nop()
        assert not _snapshot_exists(VENDOR_BIN, program)
        assert not _snapshot_exists(source, program)
    finally:
        tmp.cleanup()


if __name__ == "__main__":
    test_targeted_vendor_quirks()
    test_all_supplied_program_snapshots()
    test_program_file_is_not_limited_to_32768_words()
    test_process_exit_does_not_implicitly_save()
    print("ALL 0818 C-MODEL PARITY TESTS PASSED")
