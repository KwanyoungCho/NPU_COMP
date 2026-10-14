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

import os
import shutil
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


class PersistentTransport:
    """Reuse the simulator's input files and patch only changed host inputs.

    C-model execution and its mandatory full-image read/write are unchanged.
    Atomic rename makes the last device image the next input without copying
    immutable weights through Python on every token. This is transport reuse,
    not a hardware latency model or an in-process simulator.
    """
    def __init__(self, image):
        self.executable = build_v09_model()
        self.temporary = tempfile.TemporaryDirectory(prefix="npu-v09-session-",
            dir=os.environ.get("NPU_V09_TMPDIR") or None)
        self.directory = Path(self.temporary.name)
        array = np.ascontiguousarray(image).view(np.uint8).reshape(-1)
        if array.size % 4:
            raise ValueError("persistent global image must contain whole 32-bit cells")
        self.image_bytes = array.size
        array.tofile(self.directory / "global_memory.bin")
        self.program = np.empty(0,dtype="<u4")
        self.last_transfer = {}

    def run(self, words, patches):
        program = np.asarray(words,dtype="<u4")
        if program.ndim != 1 or not len(program) or int(program[-1]) != 0xff:
            raise ValueError("persistent program must terminate with HALT")
        for offset, data in patches:
            if offset < 0 or offset + np.asarray(data).nbytes > self.image_bytes:
                raise ValueError("runtime input patch exceeds global image")
        common = min(len(program),len(self.program))
        changed = np.flatnonzero(program[:common] != self.program[:common])
        first = int(changed[0]) if len(changed) else common
        path = self.directory / "program_memory.bin"
        with path.open("r+b" if path.exists() else "w+b") as file:
            file.seek(first*4)
            program[first:].tofile(file)
            file.truncate(program.nbytes)
        self.program = program.copy()
        global_path = self.directory / "global_memory.bin"
        with global_path.open("r+b") as file:
            for offset, data in patches:
                file.seek(offset)
                np.ascontiguousarray(data).view(np.uint8).tofile(file)
        proc = subprocess.run([str(self.executable)],cwd=self.directory,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        counters = _read_counters(self.directory)
        if proc.returncode:
            raise V09Error(proc.stderr.decode(errors="replace").strip(),counters)
        saved = self.directory / "saved_global_memory.bin"
        if not saved.exists() or saved.stat().st_size != self.image_bytes:
            raise V09Error("persistent decode requires exactly one final memory image",counters)
        saved.replace(global_path)
        self.last_transfer = dict(host_input_patch_bytes=sum(np.asarray(v).nbytes for _,v in patches),
            host_program_patch_bytes=(len(program)-first)*4,
            cmodel_image_read_bytes=self.image_bytes, cmodel_image_write_bytes=self.image_bytes,
            host_weight_rewrite_bytes=0)
        return np.memmap(global_path,dtype="<u4",mode="r+"),counters

    def close(self):
        self.temporary.cleanup()


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
    # Full-model invocations write multi-100MB images; NPU_V09_TMPDIR points
    # them at a volume with room (e.g. /data2) when the system tmp is tight.
    tmp_root = os.environ.get("NPU_V09_TMPDIR") or None
    with tempfile.TemporaryDirectory(prefix="npu-v09-", dir=tmp_root) as name:
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
            # Almost always a short write: a full-model image is several GB
            # and the simulator writes one per snapshot plus one at HALT, so a
            # volume with less room than that leaves a truncated file behind.
            # Reporting only "malformed" sends the reader looking at the
            # compiler for a disk problem.
            free = shutil.disk_usage(tmp_root or tempfile.gettempdir()).free
            raise V09Error(
                f"snapshot file is {len(raw):,} bytes, not a multiple of the "
                f"{image_bytes:,}-byte image -- the simulator's write was cut "
                f"short ({free / 2**30:.1f} GiB free on "
                f"{tmp_root or tempfile.gettempdir()}; each image needs "
                f"{image_bytes / 2**30:.1f} GiB and the run writes two)",
                counters)
        images = [np.frombuffer(raw[base:base + image_bytes], dtype="<u4").copy()
                  for base in range(0, len(raw), image_bytes)]
    return images, counters
