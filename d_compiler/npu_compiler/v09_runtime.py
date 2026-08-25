"""Runtime for the ISA ver.09 C-model (our next-generation design).

Contract (see d_compiler/ISA_V09.md):
  * ``global_memory.bin`` — initial global image, little-endian 32-bit cells.
  * ``program_memory.bin`` — little-endian 32-bit instruction words.
  * ``saved_global_memory.bin`` — one full global image appended per
    SNAPSHOT (0xF0) plus one appended by HALT (0xFF); HALT is the only
    normal termination.
  * ``perf_counters.txt`` — ``key value`` lines written at exit.
"""
from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
SOURCE = _ROOT / "_poc" / "mysim_v09.cpp"
SOURCE_BIN = _ROOT / "d_compiler" / "build" / "mysim_v09"
_BUILD_LOCK = threading.Lock()


class V09Error(RuntimeError):
    """Simulator signalled a hard error (bounds, alignment, bad opcode...)."""

    def __init__(self, message, counters=None):
        super().__init__(message)
        self.counters = counters or {}


def build_v09_model(output=SOURCE_BIN):
    """Build the v09 simulator when its cached binary is stale."""
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


def _read_counters(directory):
    path = directory / "perf_counters.txt"
    counters = {}
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition(" ")
            if value:
                counters[key] = int(value)
    return counters


def run(program_words, global_cells):
    """Execute a v09 program; return (list of snapshot images, counters).

    ``program_words`` is an iterable of 32-bit instruction words and
    ``global_cells`` the initial global image as uint32 cells.  Each returned
    image is a uint32 array (SNAPSHOT appends one; HALT appends the final one).
    """
    executable = build_v09_model()
    program = np.asarray(list(program_words), dtype="<u4")
    cells = np.ascontiguousarray(np.asarray(global_cells, dtype=np.uint32),
                                 dtype="<u4").reshape(-1)
    with tempfile.TemporaryDirectory(prefix="npu-v09-") as name:
        directory = Path(name)
        (directory / "program_memory.bin").write_bytes(program.tobytes())
        (directory / "global_memory.bin").write_bytes(cells.tobytes())
        proc = subprocess.run(
            [str(executable)], cwd=directory,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        counters = _read_counters(directory)
        if proc.returncode != 0:
            raise V09Error(proc.stderr.decode(errors="replace").strip()
                           or f"v09 simulator exited {proc.returncode}",
                           counters)
        saved = directory / "saved_global_memory.bin"
        raw = saved.read_bytes() if saved.exists() else b""
    image_bytes = cells.size * 4
    if image_bytes == 0:
        images = []
    else:
        if len(raw) % image_bytes:
            raise V09Error(f"malformed snapshot file size {len(raw)}")
        images = [np.frombuffer(raw[base:base + image_bytes], dtype="<u4").copy()
                  for base in range(0, len(raw), image_bytes)]
    return images, counters
