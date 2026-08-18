"""Streaming tensor runtime for the fixed-size 0818 vendor C-model.

Host memory owns full tensors.  Every arithmetic operation is decomposed into a
small program whose complete working set fits the real 8192-entry G-buffer.  The
host only packs tiles and carries FP16 snapshots between vendor invocations.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .isa_0818 import ACT_SILU, DST, IMM, SRC1, SRC2, VECTOR, Asm
from .runtime_0818 import (
    GBUF_CAPACITY,
    PROGRAM_CAPACITY,
    VENDOR_BIN,
    _libstdcxx_dir,
    _program_bytes,
)


class VendorRuntimeError(RuntimeError):
    pass


class VendorSession:
    """Reuse one work directory while launching the unmodified vendor executable."""

    def __init__(self, vendor_bin=VENDOR_BIN):
        self.vendor_bin = Path(vendor_bin).resolve()
        if not self.vendor_bin.exists():
            raise FileNotFoundError(self.vendor_bin)
        self._temp = tempfile.TemporaryDirectory(prefix="npu-v3-")
        self.workdir = Path(self._temp.name)
        self.env = os.environ.copy()
        library = _libstdcxx_dir()
        if library:
            self.env["LD_LIBRARY_PATH"] = library + (
                ":" + self.env["LD_LIBRARY_PATH"] if self.env.get("LD_LIBRARY_PATH") else "")
        self.program_cache = {}
        self.invocations = 0
        self.program_words = 0
        self.elapsed_seconds = 0.0

    def close(self):
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def run_program(self, program, gbuf):
        """Execute one vendor program with stdout discarded and return FP16 G-buffer."""
        values = np.asarray(gbuf, dtype=np.float16).reshape(-1)
        if values.size > GBUF_CAPACITY:
            raise ValueError(f"working set {values.size} exceeds vendor G-buffer {GBUF_CAPACITY}")
        program_bytes = _program_bytes(program)
        words = len(program_bytes) // 4
        if words > PROGRAM_CAPACITY:
            raise ValueError(f"program {words} exceeds vendor program memory {PROGRAM_CAPACITY}")
        padded = np.zeros(GBUF_CAPACITY, dtype="<f2")
        padded[:values.size] = values
        (self.workdir / "program_memory.bin").write_bytes(program_bytes)
        (self.workdir / "G_buffer_data.bin").write_bytes(padded.tobytes() + b"\n")
        saved = self.workdir / "saved_G_buffer_data.bin"
        if saved.exists():
            saved.unlink()
        start = time.perf_counter()
        subprocess.run(
            [str(self.vendor_bin)],
            cwd=self.workdir,
            env=self.env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.elapsed_seconds += time.perf_counter() - start
        self.invocations += 1
        self.program_words += words
        if not saved.exists():
            raise VendorRuntimeError("vendor did not produce saved_G_buffer_data.bin")
        raw = saved.read_bytes()
        if len(raw) != GBUF_CAPACITY * 2 + 1:
            raise VendorRuntimeError(f"unexpected vendor snapshot size {len(raw)}")
        return np.frombuffer(raw[:GBUF_CAPACITY * 2], dtype="<f2").copy()

    @staticmethod
    def _gemm_capacity(m, n):
        """Largest K<=64 satisfying A[m,k] + B[k,n] + C[m,n] <= 8192."""
        remaining = GBUF_CAPACITY - m * n
        return min(64, remaining // (m + n)) if remaining > 0 else 0

    def _gemm_program(self, m, k, n, accumulate):
        key = ("gemm", m, k, n, bool(accumulate))
        if key in self.program_cache:
            return self.program_cache[key]
        a_size, b_size = m * k, k * n
        a_off, b_off, c_off = 0, a_size, a_size + b_size
        asm = Asm()
        if accumulate:
            asm.addr(SRC1, c_off).vlen(m * n).load(0, SRC1).v_copy()
        asm.matrix_region(SRC1, a_off, m, k).load(1, SRC1)
        asm.matrix_region(SRC2, b_off, k, n).load(1, SRC2)
        asm.m_mul(VECTOR, mac=accumulate)
        asm.matrix_region(DST, c_off, m, n).save(1).finish()
        self.program_cache[key] = asm
        return asm

    def gemm(self, lhs, rhs=None, *, rhs_loader=None, n=None):
        """FP16 streaming GEMM using only vendor matrix/MAC operations.

        ``lhs`` is [M,K]. Supply either a resident ``rhs`` [K,N], or ``rhs_loader``
        called as ``loader(k_slice, n_slice) -> B_tile[Kt,Nt]`` for safetensors.
        The result is FP16 and is rounded after every streamed K partial, matching
        what crosses the vendor G-buffer between invocations.
        """
        lhs = np.asarray(lhs, dtype=np.float16)
        if lhs.ndim != 2:
            raise ValueError(f"gemm lhs must be rank 2, got {lhs.shape}")
        m_total, k_total = lhs.shape
        if rhs is not None:
            rhs = np.asarray(rhs, dtype=np.float16)
            if rhs.ndim != 2 or rhs.shape[0] != k_total:
                raise ValueError(f"gemm rhs mismatch {rhs.shape} for lhs {lhs.shape}")
            n_total = rhs.shape[1]
            loader = lambda ks, ns: rhs[ks, ns]
        else:
            if rhs_loader is None or n is None:
                raise ValueError("gemm requires rhs or rhs_loader plus n")
            n_total = int(n)
            loader = rhs_loader
        result = np.empty((m_total, n_total), dtype=np.float16)
        for m0 in range(0, m_total, 64):
            mt = min(64, m_total - m0)
            for n0 in range(0, n_total, 64):
                nt = min(64, n_total - n0)
                max_k = self._gemm_capacity(mt, nt)
                if max_k < 1:
                    raise VendorRuntimeError(f"no legal GEMM tile for m={mt}, n={nt}")
                accumulator = np.zeros((mt, nt), dtype=np.float16)
                first = True
                for k0 in range(0, k_total, max_k):
                    kt = min(max_k, k_total - k0)
                    a_tile = np.ascontiguousarray(lhs[m0:m0 + mt, k0:k0 + kt])
                    b_tile = np.asarray(
                        loader(slice(k0, k0 + kt), slice(n0, n0 + nt)), dtype=np.float16)
                    if b_tile.shape != (kt, nt):
                        raise ValueError(f"rhs loader returned {b_tile.shape}, expected {(kt, nt)}")
                    b_tile = np.ascontiguousarray(b_tile)
                    a_size, b_size = a_tile.size, b_tile.size
                    c_off = a_size + b_size
                    working = np.empty(c_off + accumulator.size, dtype=np.float16)
                    working[:a_size] = a_tile.reshape(-1)
                    working[a_size:c_off] = b_tile.reshape(-1)
                    working[c_off:] = accumulator.reshape(-1)
                    program = self._gemm_program(mt, kt, nt, accumulate=not first)
                    snapshot = self.run_program(program, working)
                    accumulator = snapshot[c_off:c_off + accumulator.size].reshape(mt, nt)
                    first = False
                result[m0:m0 + mt, n0:n0 + nt] = accumulator
        return result

    def _binary_program(self, name, length):
        key = ("binary", name, length)
        if key in self.program_cache:
            return self.program_cache[key]
        methods = {
            "add": "v_add",
            "subtract": "v_sub",
            "multiply": "v_mul",
            "divide": "v_div",
            "maximum": "v_max",
        }
        if name not in methods:
            raise ValueError(name)
        asm = Asm().vlen(length).addr(SRC1, 0).load(0, SRC1)
        asm.addr(SRC2, length).load(0, SRC2)
        getattr(asm, methods[name])(VECTOR)
        asm.addr(DST, 2 * length).save(0).finish()
        self.program_cache[key] = asm
        return asm

    def binary(self, name, lhs, rhs):
        lhs = np.asarray(lhs, dtype=np.float16)
        rhs = np.asarray(rhs, dtype=np.float16)
        if lhs.shape != rhs.shape:
            raise ValueError(f"binary shape mismatch {lhs.shape} vs {rhs.shape}")
        left, right = lhs.reshape(-1), rhs.reshape(-1)
        output = np.empty(left.size, dtype=np.float16)
        chunk = GBUF_CAPACITY // 3
        for start in range(0, left.size, chunk):
            length = min(chunk, left.size - start)
            working = np.concatenate((left[start:start + length], right[start:start + length],
                                      np.zeros(length, dtype=np.float16)))
            snapshot = self.run_program(self._binary_program(name, length), working)
            output[start:start + length] = snapshot[2 * length:3 * length]
        return output.reshape(lhs.shape)

    def _unary_program(self, name, length):
        key = ("unary", name, length)
        if key in self.program_cache:
            return self.program_cache[key]
        methods = {
            "sqrt": "v_sqrt",
            "exp": "v_exp",
            "negative": "v_sign_inv",
            "cos": "v_cos",
            "sin": "v_sin",
        }
        asm = Asm()
        if name == "silu":
            asm.matrix_region(SRC1, 0, 1, length).load(1, SRC1)
            # Activation is attached to a matrix ALU instruction.  Use an
            # immediate zero so this is an identity add followed by SiLU;
            # VECTOR mode would require an uninitialised SRC2 matrix.
            asm.m_add(IMM, 0, activation=ACT_SILU)
            asm.matrix_region(DST, length, 1, length).save(1).finish()
        else:
            if name not in methods:
                raise ValueError(name)
            asm.vlen(length).addr(SRC1, 0).load(0, SRC1)
            getattr(asm, methods[name])()
            asm.addr(DST, length).save(0).finish()
        self.program_cache[key] = asm
        return asm

    def unary(self, name, values):
        values = np.asarray(values, dtype=np.float16)
        flat = values.reshape(-1)
        output = np.empty(flat.size, dtype=np.float16)
        chunk = 64 if name == "silu" else GBUF_CAPACITY // 2
        for start in range(0, flat.size, chunk):
            length = min(chunk, flat.size - start)
            working = np.concatenate((flat[start:start + length],
                                      np.zeros(length, dtype=np.float16)))
            snapshot = self.run_program(self._unary_program(name, length), working)
            output[start:start + length] = snapshot[length:2 * length]
        return output.reshape(values.shape)

    def reduce_sum_last(self, values):
        values = np.asarray(values, dtype=np.float16)
        if values.ndim < 1:
            raise ValueError("reduce_sum_last requires at least one dimension")
        cols = values.shape[-1]
        if cols + 1 > GBUF_CAPACITY:
            raise ValueError(f"row length {cols} needs chunked reduction (not yet implemented)")
        rows = values.reshape(-1, cols)
        output = np.empty((rows.shape[0], 1), dtype=np.float16)
        key = ("reduce_sum", cols)
        if key not in self.program_cache:
            asm = Asm().vlen(cols).addr(SRC1, 0).load(0, SRC1).v_reduce_sum()
            asm.vlen(1).addr(DST, cols).save(0).finish()
            self.program_cache[key] = asm
        for row, data in enumerate(rows):
            working = np.concatenate((data, np.zeros(1, dtype=np.float16)))
            output[row, 0] = self.run_program(self.program_cache[key], working)[cols]
        return output.reshape(values.shape[:-1] + (1,))

    def reduce_max_last(self, values):
        """Safe all-negative reduction through repeated vendor vector maximum."""
        values = np.asarray(values, dtype=np.float16)
        if values.shape[-1] < 1:
            raise ValueError("cannot reduce an empty axis")
        rows = values.reshape(-1, values.shape[-1])
        output = np.empty((rows.shape[0], 1), dtype=np.float16)
        max_pairs = GBUF_CAPACITY // 3
        for row, data in enumerate(rows):
            current = data.copy()
            while current.size > 1:
                pairs = current.size // 2
                merged = []
                for start in range(0, pairs, max_pairs):
                    length = min(max_pairs, pairs - start)
                    left = current[2 * start:2 * (start + length):2]
                    right = current[2 * start + 1:2 * (start + length):2]
                    merged.append(self.binary("maximum", left, right))
                next_values = np.concatenate(merged) if merged else np.empty(0, np.float16)
                if current.size & 1:
                    next_values = np.concatenate((next_values, current[-1:]))
                current = next_values
            output[row, 0] = current[0]
        return output.reshape(values.shape[:-1] + (1,))

    def stats(self):
        return {
            "invocations": self.invocations,
            "program_words": self.program_words,
            "elapsed_seconds": self.elapsed_seconds,
            "cached_programs": len(self.program_cache),
        }
