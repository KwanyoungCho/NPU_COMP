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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .isa_0818 import ACT_SILU, DST, IMM, SCALAR, SRC1, SRC2, VECTOR, Asm
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

    @staticmethod
    def _gemm_group_capacity(m, widths):
        """Largest K for one A tile and several independent B/C output tiles."""
        total_n = sum(widths)
        remaining = GBUF_CAPACITY - m * total_n
        return min(64, remaining // (m + total_n)) if remaining > 0 else 0

    @classmethod
    def _select_gemm_group(cls, m, widths, k_total, max_group=8):
        """Pick the local N-tile group with the fewest invocations per output."""
        best = None
        for count in range(1, min(max_group, len(widths)) + 1):
            capacity = cls._gemm_group_capacity(m, widths[:count])
            if capacity < 1:
                break
            calls = (k_total + capacity - 1) // capacity
            candidate = (calls / count, calls, -capacity, count, capacity)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None:
            raise VendorRuntimeError(f"no legal grouped GEMM tile for m={m}, widths={widths}")
        return best[3], best[4]

    def _gemm_group_program(self, m, k, widths, accumulate):
        key = ("gemm_group", m, k, tuple(widths), bool(accumulate))
        if key in self.program_cache:
            return self.program_cache[key]
        a_size = m * k
        offset = a_size
        regions = []
        for n in widths:
            b_offset = offset
            offset += k * n
            c_offset = offset
            offset += m * n
            regions.append((n, b_offset, c_offset))
        asm = Asm()
        for n, b_offset, c_offset in regions:
            if accumulate:
                asm.addr(SRC1, c_offset).vlen(m * n).load(0, SRC1).v_copy()
            asm.matrix_region(SRC1, 0, m, k).load(1, SRC1)
            asm.matrix_region(SRC2, b_offset, k, n).load(1, SRC2)
            asm.m_mul(VECTOR, mac=accumulate)
            asm.matrix_region(DST, c_offset, m, n).save(1)
        self.program_cache[key] = asm.finish()
        return self.program_cache[key]

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
            n_starts = list(range(0, n_total, 64))
            n_widths = [min(64, n_total - start) for start in n_starts]
            n_index = 0
            while n_index < len(n_starts):
                count, max_k = self._select_gemm_group(
                    mt, n_widths[n_index:], k_total)
                starts = n_starts[n_index:n_index + count]
                widths = n_widths[n_index:n_index + count]
                accumulators = [np.zeros((mt, width), dtype=np.float16)
                                for width in widths]
                first = True
                for k0 in range(0, k_total, max_k):
                    kt = min(max_k, k_total - k0)
                    a_tile = np.ascontiguousarray(lhs[m0:m0 + mt, k0:k0 + kt])
                    pieces = [a_tile.reshape(-1)]
                    b_tiles = []
                    for start, width, accumulator in zip(starts, widths, accumulators):
                        b_tile = np.asarray(
                            loader(slice(k0, k0 + kt), slice(start, start + width)),
                            dtype=np.float16)
                        if b_tile.shape != (kt, width):
                            raise ValueError(
                                f"rhs loader returned {b_tile.shape}, expected {(kt, width)}")
                        b_tile = np.ascontiguousarray(b_tile)
                        b_tiles.append(b_tile)
                        pieces.extend((b_tile.reshape(-1), accumulator.reshape(-1)))
                    working = np.concatenate(pieces)
                    program = self._gemm_group_program(
                        mt, kt, widths, accumulate=not first)
                    snapshot = self.run_program(program, working)
                    offset = a_tile.size
                    next_accumulators = []
                    for width, b_tile, accumulator in zip(widths, b_tiles, accumulators):
                        offset += b_tile.size
                        size = accumulator.size
                        next_accumulators.append(
                            snapshot[offset:offset + size].reshape(mt, width))
                        offset += size
                    accumulators = next_accumulators
                    first = False
                for start, width, accumulator in zip(starts, widths, accumulators):
                    result[m0:m0 + mt, start:start + width] = accumulator
                n_index += count
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

    def broadcast_to(self, values, shape):
        """Broadcast scalar/row/column tensors with the native 0x15 opcode."""
        values = np.asarray(values, dtype=np.float16)
        shape = tuple(int(dim) for dim in shape)
        if values.shape == shape:
            return values.copy()
        if len(shape) != 2:
            raise ValueError(f"vendor broadcast requires rank-2 output, got {shape}")
        rows, cols = shape
        output = np.empty(shape, dtype=np.float16)

        if values.size == 1:
            flat = output.reshape(-1)
            max_output = GBUF_CAPACITY - 1
            for start in range(0, flat.size, max_output):
                length = min(max_output, flat.size - start)
                key = ("broadcast_scalar", length)
                if key not in self.program_cache:
                    self.program_cache[key] = (
                        Asm().vlen(length).v_broadcast(SCALAR, 0)
                        .addr(DST, 1).save(0).finish()
                    )
                working = np.concatenate((values.reshape(1),
                                          np.zeros(length, dtype=np.float16)))
                flat[start:start + length] = self.run_program(
                    self.program_cache[key], working)[1:1 + length]
            return output

        row_source = values.shape in ((cols,), (1, cols))
        if row_source:
            row = values.reshape(cols)
            rows_per_call = (GBUF_CAPACITY - cols) // cols
            if rows_per_call < 1:
                raise ValueError(f"row broadcast width {cols} exceeds streaming capacity")
            for row0 in range(0, rows, rows_per_call):
                count = min(rows_per_call, rows - row0)
                key = ("broadcast_row", cols, count)
                if key not in self.program_cache:
                    asm = Asm()
                    for index in range(count):
                        asm.vlen(cols).addr(SRC1, 0).load(0, SRC1).v_copy()
                        asm.addr(DST, cols + index * cols).save(0)
                    self.program_cache[key] = asm.finish()
                working = np.concatenate((row, np.zeros(count * cols, dtype=np.float16)))
                snapshot = self.run_program(self.program_cache[key], working)
                output[row0:row0 + count] = snapshot[cols:cols + count * cols].reshape(
                    count, cols)
            return output

        if values.shape == (rows, 1):
            rows_per_call = GBUF_CAPACITY // (cols + 1)
            if rows_per_call < 1:
                raise ValueError(f"column broadcast width {cols} exceeds streaming capacity")
            for row0 in range(0, rows, rows_per_call):
                count = min(rows_per_call, rows - row0)
                key = ("broadcast_col", cols, count)
                if key not in self.program_cache:
                    asm = Asm()
                    for index in range(count):
                        asm.vlen(cols).v_broadcast(SCALAR, index)
                        asm.addr(DST, count + index * cols).save(0)
                    self.program_cache[key] = asm.finish()
                source = values[row0:row0 + count, 0]
                working = np.concatenate((source, np.zeros(count * cols, dtype=np.float16)))
                snapshot = self.run_program(self.program_cache[key], working)
                output[row0:row0 + count] = snapshot[count:count + count * cols].reshape(
                    count, cols)
            return output

        raise ValueError(f"unsupported vendor broadcast {values.shape} -> {shape}")

    def transpose2d(self, values):
        """Transpose a row-major tensor through strided arbitrary sub-tile loads."""
        values = np.asarray(values, dtype=np.float16)
        if values.ndim != 2:
            raise ValueError(f"transpose2d expects rank 2, got {values.shape}")
        rows, cols = values.shape
        output = np.empty((cols, rows), dtype=np.float16)
        for row0 in range(0, rows, 64):
            part_rows = min(64, rows - row0)
            for col0 in range(0, cols, 64):
                part_cols = min(64, cols - col0)
                tile = np.ascontiguousarray(
                    values[row0:row0 + part_rows, col0:col0 + part_cols])
                key = ("transpose", part_rows, part_cols)
                if key not in self.program_cache:
                    size = part_rows * part_cols
                    asm = Asm().matrix_region(SRC1, 0, part_rows, part_cols)
                    asm.load(1, SRC1, strided=1, ncols=part_cols, start=0)
                    asm.shape(SRC1, part_cols, part_rows).m_add(IMM, 0)
                    asm.matrix_region(DST, size, part_cols, part_rows).save(1).finish()
                    self.program_cache[key] = asm
                working = np.concatenate((tile.reshape(-1),
                                          np.zeros(tile.size, dtype=np.float16)))
                snapshot = self.run_program(self.program_cache[key], working)
                output[col0:col0 + part_cols, row0:row0 + part_rows] = snapshot[
                    tile.size:2 * tile.size].reshape(part_cols, part_rows)
        return output

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


class ParallelVendorSession:
    """Column-parallel wrapper over independent vendor executable workdirs.

    GEMM output-column groups have no dependency on each other.  They may run in
    parallel, while every K accumulation chain stays ordered inside one
    :class:`VendorSession`.  Non-GEMM primitives use the first session.
    """

    def __init__(self, workers=4, vendor_bin=VENDOR_BIN):
        workers = int(workers)
        if workers < 1:
            raise ValueError("vendor workers must be positive")
        self.sessions = [VendorSession(vendor_bin) for _ in range(workers)]
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="npu-vendor")

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        for session in self.sessions:
            session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _column_chunks(n_total, workers, m, k_total):
        starts = list(range(0, n_total, 64))
        widths = [min(64, n_total - start) for start in starts]
        groups = []
        index = 0
        while index < len(starts):
            count, _ = VendorSession._select_gemm_group(m, widths[index:], k_total)
            groups.append((starts[index], starts[index + count - 1] +
                           widths[index + count - 1]))
            index += count
        active = min(workers, len(groups))
        if active == 1:
            return [(0, n_total)]
        base, extra = divmod(len(groups), active)
        chunks = []
        group_index = 0
        for worker in range(active):
            count = base + (worker < extra)
            selected = groups[group_index:group_index + count]
            chunks.append((selected[0][0], selected[-1][1]))
            group_index += count
        return chunks

    def gemm(self, lhs, rhs=None, *, rhs_loader=None, n=None):
        lhs = np.asarray(lhs, dtype=np.float16)
        if rhs is not None:
            rhs = np.asarray(rhs, dtype=np.float16)
            n_total = rhs.shape[1]
        else:
            if rhs_loader is None or n is None:
                raise ValueError("gemm requires rhs or rhs_loader plus n")
            n_total = int(n)
        chunks = ([(0, n_total)] if lhs.shape[0] > 64 else
                  self._column_chunks(
                      n_total, len(self.sessions), lhs.shape[0], lhs.shape[1]))
        if len(chunks) == 1:
            return self.sessions[0].gemm(lhs, rhs, rhs_loader=rhs_loader, n=n)

        def run(index, start, end):
            if rhs is not None:
                return self.sessions[index].gemm(lhs, rhs[:, start:end])

            def shifted_loader(k_slice, n_slice):
                return rhs_loader(
                    k_slice, slice(n_slice.start + start, n_slice.stop + start))
            return self.sessions[index].gemm(
                lhs, rhs_loader=shifted_loader, n=end - start)

        futures = [self.executor.submit(run, index, start, end)
                   for index, (start, end) in enumerate(chunks)]
        return np.concatenate([future.result() for future in futures], axis=1)

    def binary(self, *args, **kwargs):
        return self.sessions[0].binary(*args, **kwargs)

    def unary(self, *args, **kwargs):
        return self.sessions[0].unary(*args, **kwargs)

    def broadcast_to(self, *args, **kwargs):
        return self.sessions[0].broadcast_to(*args, **kwargs)

    def transpose2d(self, *args, **kwargs):
        return self.sessions[0].transpose2d(*args, **kwargs)

    def reduce_sum_last(self, *args, **kwargs):
        return self.sessions[0].reduce_sum_last(*args, **kwargs)

    def reduce_max_last(self, *args, **kwargs):
        return self.sessions[0].reduce_max_last(*args, **kwargs)

    def stats(self):
        child = [session.stats() for session in self.sessions]
        return {
            "invocations": sum(item["invocations"] for item in child),
            "program_words": sum(item["program_words"] for item in child),
            "elapsed_seconds": sum(item["elapsed_seconds"] for item in child),
            "cached_programs": sum(item["cached_programs"] for item in child),
            "workers": len(self.sessions),
        }
