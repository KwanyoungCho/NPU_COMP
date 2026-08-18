"""Runtime for the source-level ver.08 C-model with an extended flat G-buffer.

The arithmetic and instruction quirks remain vendor-compatible.  Only storage is
extended: the source model consumes the complete program file and grows its flat
G-buffer beyond the vendor executable's 8192-entry static array.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from .runtime_0818 import GBUF_CAPACITY, _libstdcxx_dir, _program_bytes

_ROOT = Path(__file__).resolve().parents[2]
SOURCE = _ROOT / "_poc" / "mysim_0818.cpp"
SOURCE_BIN = _ROOT / "d_compiler" / "build" / "mysim_0818"
_BUILD_LOCK = threading.Lock()


def build_source_model(output=SOURCE_BIN):
    """Build the checked-in parity model when its cached binary is stale."""
    output = Path(output).resolve()
    with _BUILD_LOCK:
        if output.exists() and output.stat().st_mtime_ns >= SOURCE.stat().st_mtime_ns:
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        subprocess.run(
            ["g++", "-std=c++17", "-O2", str(SOURCE), "-o", str(temporary)],
            check=True,
        )
        temporary.replace(output)
    return output


def _environment():
    env = os.environ.copy()
    env["NPU0818_QUIET"] = "1"
    library = _libstdcxx_dir()
    if library:
        env["LD_LIBRARY_PATH"] = library + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def run(program, gbuf, *, capture_trace=False, source_bin=None):
    """Execute a program with a dynamically sized source-model G-buffer."""
    executable = Path(source_bin).resolve() if source_bin else build_source_model()
    if not executable.exists():
        raise FileNotFoundError(executable)
    values = np.asarray(gbuf, dtype=np.float16).reshape(-1)
    capacity = max(GBUF_CAPACITY, values.size)
    initial = np.zeros(capacity, dtype="<f2")
    initial[:values.size] = values
    with tempfile.TemporaryDirectory(prefix="npu0818-source-") as directory:
        directory = Path(directory)
        (directory / "program_memory.bin").write_bytes(_program_bytes(program))
        (directory / "G_buffer_data.bin").write_bytes(initial.tobytes() + b"\n")
        env = _environment()
        if capture_trace:
            env.pop("NPU0818_QUIET", None)
        proc = subprocess.run(
            [str(executable)], cwd=directory, env=env, check=True,
            stdout=subprocess.PIPE if capture_trace else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if capture_trace else subprocess.DEVNULL,
        )
        saved = directory / "saved_G_buffer_data.bin"
        if not saved.exists():
            raise RuntimeError("source model produced no snapshot; program must execute 0xF0")
        raw = saved.read_bytes()
        if not raw or raw[-1:] != b"\n" or (len(raw) - 1) % 2:
            raise RuntimeError(f"malformed source-model snapshot size {len(raw)}")
        output = np.frombuffer(raw[:-1], dtype="<f2").astype(np.float32)
    if capture_trace:
        return output, proc.stdout.decode(errors="replace")
    return output
