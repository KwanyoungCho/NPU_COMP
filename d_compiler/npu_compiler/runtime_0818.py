"""Runtime harness for the supplied 0818 vendor C-model.

The executable is the execution oracle.  This module only materializes its two
fixed-name inputs and reads the fixed-name G-buffer snapshot produced by 0xF0.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
VENDOR_BIN = _ROOT / "0818_npu_update" / "a_npu" / "a.out"
GBUF_CAPACITY = 8192
PROGRAM_CAPACITY = 32768


def _program_bytes(program):
    if isinstance(program, (bytes, bytearray)):
        return bytes(program)
    words = program.words if hasattr(program, "words") else list(program)
    arr = (np.asarray(words, dtype=np.uint64) & np.uint64(0xFFFFFFFF)).astype("<u4")
    return arr.tobytes() + b"\n"


def _libstdcxx_dir():
    configured = os.environ.get("NPU0818_LIBSTDCPP")
    if configured:
        return configured
    candidates = [
        Path("/home/chokwans99/anaconda3/envs/npu-tvm/lib"),
        Path(os.environ.get("CONDA_PREFIX", "")) / "lib",
    ]
    for path in candidates:
        if path and (path / "libstdc++.so.6").exists():
            return str(path)
    return None


def run(program, gbuf, *, capture_trace=False, vendor_bin=VENDOR_BIN):
    """Run ver.08 and return its full 8192-entry FP16 G-buffer snapshot.

    The real executable cannot represent a larger G-buffer or program.  Rejecting
    overflow here prevents the vendor's silent memory corruption.
    """
    pbytes = _program_bytes(program)
    nwords = len(pbytes) // 4
    if nwords > PROGRAM_CAPACITY:
        raise ValueError(f"0818 program has {nwords} words; vendor capacity is {PROGRAM_CAPACITY}")
    arr = np.asarray(gbuf, dtype=np.float16).reshape(-1)
    if arr.size > GBUF_CAPACITY:
        raise ValueError(f"0818 G-buffer has {arr.size} FP16 values; vendor capacity is {GBUF_CAPACITY}")
    padded = np.zeros(GBUF_CAPACITY, dtype="<f2")
    padded[:arr.size] = arr

    vendor_bin = Path(vendor_bin).resolve()
    if not vendor_bin.exists():
        raise FileNotFoundError(vendor_bin)
    with tempfile.TemporaryDirectory(prefix="npu0818-") as tmp:
        tmp = Path(tmp)
        (tmp / "program_memory.bin").write_bytes(pbytes)
        (tmp / "G_buffer_data.bin").write_bytes(padded.tobytes() + b"\n")
        env = os.environ.copy()
        lib = _libstdcxx_dir()
        if lib:
            env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"]
                                                    if env.get("LD_LIBRARY_PATH") else "")
        proc = subprocess.run([str(vendor_bin)], cwd=tmp, env=env, check=True,
                              stdout=(subprocess.PIPE if capture_trace else subprocess.DEVNULL),
                              stderr=(subprocess.STDOUT if capture_trace else subprocess.DEVNULL))
        saved = tmp / "saved_G_buffer_data.bin"
        if not saved.exists():
            raise RuntimeError("vendor produced no saved_G_buffer_data.bin; program must execute 0xF0")
        raw = saved.read_bytes()
        if len(raw) != GBUF_CAPACITY * 2 + 1:
            raise RuntimeError(f"unexpected vendor snapshot size {len(raw)}")
        out = np.frombuffer(raw[:GBUF_CAPACITY * 2], dtype="<f2").astype(np.float32)
    if capture_trace:
        return out, proc.stdout.decode(errors="replace")
    return out
