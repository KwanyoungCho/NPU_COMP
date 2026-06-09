"""Runtime harness: run a compiled NPU program on the GIVEN c-model `_poc/mysim`.

We do NOT reimplement NPU execution. This only:
  1. builds _poc/mysim.cpp (cached),
  2. writes program_memory.bin + G_buffer_data.bin,
  3. invokes mysim (--gout for FP16 write-back),
  4. reads gout.bin back as floats.
The G-buffer in mysim stores FP16 (rounding on every save).
"""
import os, subprocess, tempfile
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))               # repo root
MYSIM_SRC = os.path.join(_ROOT, "_poc", "mysim.cpp")
_BUILD_DIR = os.path.join(_ROOT, "d_compiler", "build")
MYSIM_BIN = os.path.join(_BUILD_DIR, "mysim")


def build_mysim(force=False):
    """Compile the given mysim.cpp. Cached: rebuilds only if source is newer."""
    os.makedirs(_BUILD_DIR, exist_ok=True)
    if (not force and os.path.exists(MYSIM_BIN)
            and os.path.getmtime(MYSIM_BIN) >= os.path.getmtime(MYSIM_SRC)):
        return MYSIM_BIN
    subprocess.run(["g++", "-O2", "-std=c++17", MYSIM_SRC, "-o", MYSIM_BIN],
                   check=True)
    return MYSIM_BIN


def _program_bytes(program):
    """program: Asm | list[int] | bytes -> program_memory.bin bytes."""
    if isinstance(program, (bytes, bytearray)):
        return bytes(program)
    words = program.words if hasattr(program, "words") else list(program)
    import struct
    return b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in words) + b"\n"


def _gbuffer_bytes(gbuf):
    """gbuf: 1-D float array-like -> FP16 little-endian bytes (+ trailing newline)."""
    arr = np.asarray(gbuf, dtype=np.float16)
    return arr.tobytes() + b"\n"


def run(program, gbuf, gn=None, maxrun=None, capture_trace=False):
    """Run `program` on mysim with initial G-buffer `gbuf` (list/np of floats).

    Returns the written-back G-buffer (np.float32 of length gn).
    gn:     #entries to write back (default len(gbuf)).
    maxrun: max instructions (default = program length; HALT stops earlier).
    """
    build_mysim()
    gbuf = np.asarray(gbuf, dtype=np.float32)
    if gn is None:
        gn = len(gbuf)
    pbytes = _program_bytes(program)
    nwords = (len(pbytes)) // 4
    if maxrun is None:
        maxrun = nwords + 1

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "program_memory.bin"), "wb") as f:
            f.write(pbytes)
        with open(os.path.join(d, "G_buffer_data.bin"), "wb") as f:
            f.write(_gbuffer_bytes(gbuf))
        gout = os.path.join(d, "gout.bin")
        proc = subprocess.run(
            [MYSIM_BIN, "--run", str(maxrun), "--gout", gout, "--gn", str(gn)],
            cwd=d, check=True,
            stdout=(subprocess.PIPE if capture_trace else subprocess.DEVNULL),
            stderr=subprocess.DEVNULL)
        raw = open(gout, "rb").read()
        out = np.frombuffer(raw[: gn * 2], dtype=np.float16).astype(np.float32)
    if capture_trace:
        return out, proc.stdout.decode(errors="replace")
    return out
