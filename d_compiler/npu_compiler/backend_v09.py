"""Relax-to-v09 backend: ver.08 kernel emission over a staged SRAM.

Reuses the row-major ver.08 memory plan unchanged (same offsets as the 0818
backend, so goldens compare element-for-element) and mirrors the 0818 kernel
loop orders exactly; the only addition is DMA staging:

  * Every binding stages its operands from global memory into SRAM, runs the
    ver.08-encoded kernel against SRAM nibble addresses, and stages the result
    back.  The SRAM cursor resets per binding (no cross-binding reuse yet).
  * Cell parity is handled by *superset staging*: the staged window is the
    smallest 32-bit-cell-aligned range containing the tensor, so odd element
    offsets/sizes need no memplan alignment.  A destination whose borders
    fall mid-cell is staged in first, so the border neighbours round-trip
    with their current values.
  * Tensors above ``whole_limit`` elements stream through 2D panels/tiles
    (matmul weights, wide activations); their row stride must be even, which
    every 64-multiple model dimension satisfies.
  * Vector instructions carry full ver.08-range vlen (the 256-lane datapath
    strip-mines internally), so vector emission matches the 0818 backend and
    the flat accumulation order -- the basis of the bit-exactness proof --
    is preserved by construction.
"""
from __future__ import annotations

from contextlib import contextmanager

from tvm import relax

from .backend_0818 import CodegenError, _numel, _opname, plan
from .isa_0818 import ACT_GELU, ACT_SILU, DST, IMM, SRC1, SRC2, VECTOR, Asm
from .isa_v09 import (
    DT_FP16, DT_INT8, SRAM_NIBBLES, enc_ascale, enc_gload, enc_gstore,
    enc_halt, enc_mcols, enc_mrows, enc_wscale)
from .quantize import packed_layout

VLEN = 0xFFFF          # full 16-bit vlen; the 256-lane datapath strip-mines
DMA_MAX_CELLS = 0xFFFF


class V09Asm(Asm):
    format_version = "v09"
    execution_target = "v09"

    def gload(self, g_addr, g_stride, sram, rows, cols):
        for word in enc_gload(g_addr, g_stride, sram, rows, cols):
            self._emit(word)
        return self

    def gstore(self, g_addr, g_stride, sram, rows, cols):
        for word in enc_gstore(g_addr, g_stride, sram, rows, cols):
            self._emit(word)
        return self

    def halt(self):
        return self._emit(enc_halt())

    def shape_dt(self, operand, rows, cols, partial, dtype):
        self._emit(enc_mrows(operand, rows, partial, dtype))
        return self._emit(enc_mcols(operand, cols, partial, dtype))

    def wscale(self, nibble):
        for word in enc_wscale(nibble):
            self._emit(word)
        return self


class _Stager:
    """Per-binding SRAM bump allocator emitting superset-staging DMA."""

    def __init__(self, asm):
        self.a = asm
        self.cursor = 0
        self.peak = 0

    def reset(self):
        self.cursor = 0

    @contextmanager
    def scope(self):
        mark = self.cursor
        try:
            yield
        finally:
            self.cursor = mark

    def alloc(self, nibbles):
        start = (self.cursor + 7) // 8 * 8
        self.cursor = start + nibbles
        self.peak = max(self.peak, self.cursor)
        if self.cursor > SRAM_NIBBLES:
            raise CodegenError(
                f"v09 SRAM overflow: binding needs {self.cursor} nibbles "
                f"(capacity {SRAM_NIBBLES})")
        return start

    # -------- contiguous element ranges (any parity, via cell superset)

    @staticmethod
    def _span(elem0, count):
        lo = elem0 // 2
        return lo, (elem0 + count + 1) // 2 - lo

    def _dma_1d(self, store, lo_cell, cells, slot):
        for base in range(0, cells, DMA_MAX_CELLS):
            chunk = min(DMA_MAX_CELLS, cells - base)
            (self.a.gstore if store else self.a.gload)(
                lo_cell + base, chunk, slot + base * 8, 1, chunk)

    def range_in(self, elem0, count):
        """Stage global elements [elem0, elem0+count) -> nibble of elem0."""
        lo, cells = self._span(elem0, count)
        slot = self.alloc(cells * 8)
        self._dma_1d(False, lo, cells, slot)
        return slot + (elem0 - lo * 2) * 4

    def range_dst(self, elem0, count):
        """Allocate a destination range; preload only if borders are mid-cell."""
        lo, cells = self._span(elem0, count)
        slot = self.alloc(cells * 8)
        if elem0 % 2 or (elem0 + count) % 2:
            self._dma_1d(False, lo, cells, slot)
        return slot + (elem0 - lo * 2) * 4

    def range_out(self, elem0, count, nib0):
        lo, cells = self._span(elem0, count)
        self._dma_1d(True, lo, cells, nib0 - (elem0 - lo * 2) * 4)

    # -------- 2D windows (row stride must be even; row parity then uniform)

    def _panel(self, elem0, rows, cols, stride, *, load):
        if stride % 2:
            raise CodegenError(
                f"v09 2D staging needs an even row stride, got {stride}")
        parity = elem0 % 2
        lo = (elem0 - parity) // 2
        row_cells = (parity + cols + 1) // 2
        if row_cells > DMA_MAX_CELLS or rows > DMA_MAX_CELLS:
            raise CodegenError("v09 2D staging window exceeds DMA field range")
        slot = self.alloc(rows * row_cells * 8)
        if load:
            self.a.gload(lo, stride // 2, slot, rows, row_cells)
        return slot + parity * 4, row_cells * 2

    def panel_in(self, elem0, rows, cols, stride):
        """Stage a [rows, cols] window -> (nibble of (0,0), sram row stride)."""
        return self._panel(elem0, rows, cols, stride, load=True)

    def panel_dst(self, elem0, rows, cols, stride):
        parity = elem0 % 2
        need = parity != 0 or (parity + cols) % 2 != 0
        return self._panel(elem0, rows, cols, stride, load=need)

    def panel_out(self, elem0, rows, cols, stride, nib0):
        parity = elem0 % 2
        lo = (elem0 - parity) // 2
        row_cells = (parity + cols + 1) // 2
        self.a.gstore(lo, stride // 2, nib0 - parity * 4, rows, row_cells)


def compile_module(mod, func_name="main", *, tile=64, reuse=False,
                   whole_limit=1 << 20, chunk=1 << 16, quant_int8=None):
    func = mod[func_name]
    mp = plan(func, reuse=reuse)
    return compile_func(func, mp, tile=tile, whole_limit=whole_limit,
                        chunk=chunk, quant_int8=quant_int8), mp


def compile_func(func, mp, *, tile=64, whole_limit=1 << 20, chunk=1 << 16,
                 quant_int8=None, emit_log=None):
    """Compile a planned Relax function into a v09 program (ends with HALT)."""
    if tile <= 0 or tile > 64:
        raise ValueError(f"PE tile must be in 1..64, got {tile}")

    a = V09Asm()
    st = _Stager(a)
    off = mp.offset
    quant_names = frozenset(quant_int8 or ())
    quant_vars = {p for p in mp.params if p.name_hint in quant_names}
    for p in quant_vars:
        k, n = mp.shape[p]
        if k % tile or n % tile:
            raise CodegenError(
                f"INT8 weight '{p.name_hint}' [{k},{n}] must be {tile}-multiple")
    a.quant_int8_params = {p.name_hint: tuple(mp.shape[p]) for p in quant_vars}

    def fits(count):
        return count <= whole_limit

    def sram_region(operand, nib0, stride, rows, cols, row, col,
                    part_rows, part_cols, dtype=DT_FP16, width=4):
        """MAIN/PARTIAL descriptors for a staged window (addresses in nibbles,
        shapes in elements — the ver.08 field values are unchanged; the operand
        dtype rides in the shape words' spare bits)."""
        a.addr(operand, nib0, 0).shape_dt(operand, rows, stride, 0, dtype)
        a.addr(operand, nib0 + (row * stride + col) * width, 1)
        a.shape_dt(operand, part_rows, part_cols, 1, dtype)

    def sram_copy(dst_nib, src_nib, count):
        for base in range(0, count, VLEN):
            length = min(VLEN, count - base)
            a.vlen(length)
            a.addr(SRC1, src_nib + base * 4).load(0, SRC1).v_copy()
            a.addr(DST, dst_nib + base * 4).save(0)

    def reduce_row(kind, src_nib, count, dst_nib):
        """One reduce instruction per row (ver.08 form; flat FP32 order)."""
        if count > VLEN:
            raise CodegenError(f"row of {count} elements exceeds the 16-bit vlen")
        a.vlen(count)
        a.addr(SRC1, src_nib).load(0, SRC1)
        (a.v_reduce_sum if kind == "sum" else a.v_reduce_max)()
        a.addr(DST, dst_nib).save(0)

    # ------------------------------------------------------------- kernels

    def emit_matmul(dst, lhs, rhs):
        try:
            rows, inner = mp.shape[lhs]
            inner2, cols = mp.shape[rhs]
        except ValueError as exc:
            raise CodegenError("matmul expects two rank-2 tensors") from exc
        if inner != inner2:
            raise CodegenError(f"matmul K mismatch {inner} vs {inner2}")
        rhs_int8 = rhs in quant_vars
        lhs_whole = fits(rows * inner)
        # packed INT8 occupies half the FP16-element footprint
        rhs_elems = inner * cols // 2 if rhs_int8 else inner * cols
        rhs_whole = fits(rhs_elems)
        dst_whole = fits(rows * cols)
        if rhs_int8:
            data_elem, scale_elem = packed_layout(off[rhs], inner, cols)
            scale_nib = st.range_in(scale_elem, 2 * cols)
        if lhs_whole:
            lhs_nib = st.range_in(off[lhs], rows * inner)
            lhs_stride = inner
        if rhs_whole:
            rhs_nib = st.range_in(data_elem if rhs_int8 else off[rhs], rhs_elems)
        if dst_whole:
            dst_nib = st.range_dst(off[dst], rows * cols)
        for row in range(0, rows, tile):
            part_rows = min(tile, rows - row)
            with st.scope():
                if not lhs_whole:
                    # contiguous row block (stride == inner): 1D staging
                    panel_nib = st.range_in(off[lhs] + row * inner,
                                            part_rows * inner)
                    lhs_stride = inner
                for col in range(0, cols, tile):
                    part_cols = min(tile, cols - col)
                    with st.scope():
                        if rhs_int8:
                            # scale vector indexed tile-locally by the unit:
                            # aim 0x8B at this column tile's first scale
                            a.wscale(scale_nib + col * 8)
                        for k_index, k0 in enumerate(range(0, inner, tile)):
                            part_inner = min(tile, inner - k0)
                            if lhs_whole:
                                sram_region(SRC1, lhs_nib, lhs_stride,
                                            rows, inner, row, k0,
                                            part_rows, part_inner)
                            else:
                                sram_region(SRC1, panel_nib, lhs_stride,
                                            part_rows, inner, 0, k0,
                                            part_rows, part_inner)
                            a.load(1, SRC1)
                            with st.scope():
                                if rhs_whole and rhs_int8:
                                    sram_region(SRC2, rhs_nib, cols,
                                                inner, cols, k0, col,
                                                part_inner, part_cols,
                                                dtype=DT_INT8, width=2)
                                elif rhs_whole:
                                    sram_region(SRC2, rhs_nib, cols,
                                                inner, cols, k0, col,
                                                part_inner, part_cols)
                                elif rhs_int8:
                                    # int8 tile: window expressed in the
                                    # FP16-element footprint (2 values/elem)
                                    tile_nib, tile_stride = st.panel_in(
                                        data_elem + (k0 * cols + col) // 2,
                                        part_inner, part_cols // 2, cols // 2)
                                    sram_region(SRC2, tile_nib, tile_stride * 2,
                                                part_inner, part_cols, 0, 0,
                                                part_inner, part_cols,
                                                dtype=DT_INT8, width=2)
                                else:
                                    tile_nib, tile_stride = st.panel_in(
                                        off[rhs] + k0 * cols + col,
                                        part_inner, part_cols, cols)
                                    sram_region(SRC2, tile_nib, tile_stride,
                                                part_inner, part_cols, 0, 0,
                                                part_inner, part_cols)
                                a.load(1, SRC2)
                            a.m_mul(VECTOR, mac=k_index != 0)
                        if dst_whole:
                            sram_region(DST, dst_nib, cols, rows, cols,
                                        row, col, part_rows, part_cols)
                            a.save(1)
                        else:
                            out_nib, out_stride = st.panel_dst(
                                off[dst] + row * cols + col,
                                part_rows, part_cols, cols)
                            sram_region(DST, out_nib, out_stride,
                                        part_rows, part_cols, 0, 0,
                                        part_rows, part_cols)
                            a.save(1)
                            st.panel_out(off[dst] + row * cols + col,
                                         part_rows, part_cols, cols, out_nib)
        if rhs_int8:
            # descriptor dtype is sticky state: restore SRC2 to FP16 so a
            # following vector op (which sets only addresses) reads correctly
            a.shape_dt(SRC2, min(inner, 0xFFFF), min(cols, 0xFFFF), 1, DT_FP16)
        if dst_whole:
            st.range_out(off[dst], rows * cols, dst_nib)

    def emit_row_sum(dst, src):
        rows, cols = mp.shape[src]
        if mp.shape[dst][-1] != 1:
            raise CodegenError(f"sum output must keep the last axis, "
                               f"got {mp.shape[dst]}")
        dst_nib = st.range_dst(off[dst], rows)
        for row in range(rows):
            with st.scope():
                src_nib = st.range_in(off[src] + row * cols, cols)
                reduce_row("sum", src_nib, cols, dst_nib + row * 4)
        st.range_out(off[dst], rows, dst_nib)

    def emit_row_max(dst, src):
        """0818 column-fold order preserved (vlen = part_rows <= 64)."""
        rows, cols = mp.shape[src]
        if cols < 1 or mp.shape[dst][-1] != 1:
            raise CodegenError(f"max expects non-empty [R,C] -> [R,1], "
                               f"got {mp.shape[src]}")
        dst_nib = st.range_dst(off[dst], rows)
        for row in range(0, rows, tile):
            part_rows = min(tile, rows - row)
            acc_nib = dst_nib + row * 4
            with st.scope():
                # a row window with stride == cols is contiguous in global
                # memory, so 1D superset staging covers any parity
                panel_nib = st.range_in(off[src] + row * cols,
                                        part_rows * cols)
                stride = cols
                sram_region(SRC1, panel_nib, stride, part_rows, cols,
                            0, 0, part_rows, 1)
                a.load(1, SRC1, strided=1, ncols=1, start=0)
                a.vlen(part_rows).v_copy().addr(DST, acc_nib).save(0)
                for col in range(1, cols):
                    sram_region(SRC1, panel_nib, stride, part_rows, cols,
                                0, col, part_rows, 1)
                    a.load(1, SRC1, strided=1, ncols=1, start=0)
                    a.vlen(part_rows).addr(SRC2, acc_nib).load(0, SRC2)
                    a.v_max(VECTOR).addr(DST, acc_nib).save(0)
        st.range_out(off[dst], rows, dst_nib)

    def emit_broadcast(dst, src):
        dst_shape = mp.shape[dst]
        src_shape = mp.shape[src]
        total = _numel(dst_shape)
        src_total = _numel(src_shape)
        if src_total == total:
            for base in range(0, total, chunk):
                length = min(chunk, total - base)
                with st.scope():
                    src_nib = st.range_in(off[src] + base, length)
                    dst_nib = st.range_dst(off[dst] + base, length)
                    sram_copy(dst_nib, src_nib, length)
                    st.range_out(off[dst] + base, length, dst_nib)
            return
        if src_total == 1:
            src_nib = st.range_in(off[src], 1)
            for base in range(0, total, chunk):
                length = min(chunk, total - base)
                with st.scope():
                    dst_nib = st.range_dst(off[dst] + base, length)
                    for vb in range(0, length, VLEN):
                        vl = min(VLEN, length - vb)
                        a.vlen(vl).v_broadcast_addr(src_nib)
                        a.addr(DST, dst_nib + vb * 4).save(0)
                    st.range_out(off[dst] + base, length, dst_nib)
            return
        if len(dst_shape) != 2:
            raise CodegenError(f"broadcast {src_shape} -> {dst_shape} "
                               "unsupported")
        rows, cols = dst_shape
        row_source = ((len(src_shape) == 1 and src_shape[0] == cols) or
                      (len(src_shape) == 2 and src_shape == [1, cols]))
        col_source = len(src_shape) == 2 and src_shape == [rows, 1]
        if row_source:
            src_nib = st.range_in(off[src], cols)
            for row in range(rows):
                with st.scope():
                    dst_nib = st.range_dst(off[dst] + row * cols, cols)
                    sram_copy(dst_nib, src_nib, cols)
                    st.range_out(off[dst] + row * cols, cols, dst_nib)
            return
        if col_source:
            src_nib = st.range_in(off[src], rows)
            for row in range(rows):
                with st.scope():
                    dst_nib = st.range_dst(off[dst] + row * cols, cols)
                    for vb in range(0, cols, VLEN):
                        vl = min(VLEN, cols - vb)
                        a.vlen(vl).v_broadcast_addr(src_nib + row * 4)
                        a.addr(DST, dst_nib + vb * 4).save(0)
                    st.range_out(off[dst] + row * cols, cols, dst_nib)
            return
        raise CodegenError(f"broadcast {src_shape} -> {dst_shape} unsupported")

    def emit_transpose(dst, src):
        if len(mp.shape[src]) != 2:
            raise CodegenError(f"permute_dims expects rank 2, "
                               f"got {mp.shape[src]}")
        rows, cols = mp.shape[src]
        if not (fits(rows * cols) and fits(rows * cols)):
            raise CodegenError("v09 transpose currently requires the tensor "
                               "to fit in SRAM whole")
        src_nib = st.range_in(off[src], rows * cols)
        dst_nib = st.range_dst(off[dst], rows * cols)
        for col in range(0, cols, tile):
            part_cols = min(tile, cols - col)
            for row in range(0, rows, tile):
                part_rows = min(tile, rows - row)
                sram_region(SRC1, src_nib, cols, rows, cols, row, col,
                            part_rows, part_cols)
                a.load(1, SRC1, strided=1, ncols=part_cols, start=0)
                a.shape(SRC1, part_cols, part_rows).m_add(IMM, 0)
                sram_region(DST, dst_nib, rows, cols, rows, col, row,
                            part_cols, part_rows)
                a.save(1)
        st.range_out(off[dst], rows * cols, dst_nib)

    def emit_strided_slice(dst, call):
        src = call.args[0]
        axes = [int(value.value) for value in call.args[1]]
        begin = [int(value.value) for value in call.args[2]]
        end = [int(value.value) for value in call.args[3]]
        if len(mp.shape[src]) != 2 or axes != [1] and axes != [-1]:
            raise CodegenError(f"strided_slice only supports a rank-2 last "
                               f"axis, axes={axes}")
        rows, cols = mp.shape[src]
        first = begin[0] + (cols if begin[0] < 0 else 0)
        last = min(end[0] + (cols if end[0] < 0 else 0), cols)
        width = last - first
        if width != mp.shape[dst][1]:
            raise CodegenError("strided_slice steps other than one are "
                               "unsupported")
        for row in range(rows):
            with st.scope():
                src_nib = st.range_in(off[src] + row * cols + first, width)
                dst_nib = st.range_dst(off[dst] + row * width, width)
                sram_copy(dst_nib, src_nib, width)
                st.range_out(off[dst] + row * width, width, dst_nib)

    def emit_concat(dst, call):
        sources = list(call.args[0].fields)
        if int(call.attrs.axis) not in (1, -1):
            raise CodegenError("concat only supports the last axis of "
                               "rank-2 tensors")
        rows, dst_cols = mp.shape[dst]
        dst_col = 0
        for src in sources:
            src_rows, src_cols = mp.shape[src]
            if src_rows != rows:
                raise CodegenError("concat row mismatch")
            for row in range(rows):
                with st.scope():
                    src_nib = st.range_in(off[src] + row * src_cols, src_cols)
                    base = off[dst] + row * dst_cols + dst_col
                    dst_nib = st.range_dst(base, src_cols)
                    sram_copy(dst_nib, src_nib, src_cols)
                    st.range_out(base, src_cols, dst_nib)
            dst_col += src_cols

    def emit_vector_elementwise(dst, method, args):
        count = _numel(mp.shape[dst])
        for arg in args:
            if _numel(mp.shape[arg]) != count:
                raise CodegenError(
                    f"elementwise {mp.shape[arg]} -> {mp.shape[dst]} requires "
                    "an explicit broadcast_to before lowering")
        for base in range(0, count, chunk):
            length = min(chunk, count - base)
            with st.scope():
                nibs = [st.range_in(off[arg] + base, length) for arg in args]
                dst_nib = st.range_dst(off[dst] + base, length)
                for vb in range(0, length, VLEN):
                    vl = min(VLEN, length - vb)
                    a.vlen(vl)
                    a.addr(SRC1, nibs[0] + vb * 4).load(0, SRC1)
                    if len(args) == 2:
                        a.addr(SRC2, nibs[1] + vb * 4).load(0, SRC2)
                        method(VECTOR)
                    else:
                        method()
                    a.addr(DST, dst_nib + vb * 4).save(0)
                st.range_out(off[dst] + base, length, dst_nib)

    def emit_activation(dst, src, activation):
        shape = mp.shape[src]
        rows = _numel(shape[:-1]) if len(shape) > 1 else 1
        cols = shape[-1]
        for row in range(0, rows, tile):
            part_rows = min(tile, rows - row)
            with st.scope():
                src_nib = st.range_in(off[src] + row * cols, part_rows * cols)
                dst_nib = st.range_dst(off[dst] + row * cols, part_rows * cols)
                for col in range(0, cols, tile):
                    part_cols = min(tile, cols - col)
                    sram_region(SRC1, src_nib, cols, part_rows, cols,
                                0, col, part_rows, part_cols)
                    a.load(1, SRC1).m_add(IMM, 0, activation)
                    sram_region(DST, dst_nib, cols, part_rows, cols,
                                0, col, part_rows, part_cols)
                    a.save(1)
                st.range_out(off[dst] + row * cols, part_rows * cols, dst_nib)

    vector_binary = {
        "relax.add": a.v_add,
        "relax.subtract": a.v_sub,
        "relax.multiply": a.v_mul,
        "relax.divide": a.v_div,
    }
    vector_unary = {
        "relax.sqrt": a.v_sqrt,
        "relax.exp": a.v_exp,
        "relax.negative": a.v_sign_inv,
        "relax.cos": a.v_cos,
        "relax.sin": a.v_sin,
    }

    seq = func.body
    for block in seq.blocks:
        for binding in block.bindings:
            dst = binding.var
            call = binding.value
            if isinstance(call, (relax.Var, relax.Tuple)):
                continue
            if not isinstance(call, relax.Call):
                raise CodegenError(f"unsupported binding value {type(call)}")
            name = _opname(call)
            start = len(a.words)
            st.reset()
            if name == "relax.matmul":
                with a.role("matmul"):
                    emit_matmul(dst, call.args[0], call.args[1])
            elif name in ("relax.sum", "relax.max"):
                src = call.args[0]
                axis = [int(value) % len(mp.shape[src])
                        for value in call.attrs.axis]
                if axis != [len(mp.shape[src]) - 1]:
                    raise CodegenError(f"{name} only supports last-axis "
                                       f"keepdims, axis={axis}")
                with a.role("reduce"):
                    (emit_row_sum if name == "relax.sum" else emit_row_max)(
                        dst, src)
            elif name == "relax.broadcast_to":
                with a.role("broadcast"):
                    emit_broadcast(dst, call.args[0])
            elif name == "relax.permute_dims":
                with a.role("transpose"):
                    emit_transpose(dst, call.args[0])
            elif name == "relax.strided_slice":
                with a.role("layout"):
                    emit_strided_slice(dst, call)
            elif name == "relax.concat":
                with a.role("layout"):
                    emit_concat(dst, call)
            elif name in vector_binary:
                with a.role("elementwise"):
                    emit_vector_elementwise(
                        dst, vector_binary[name], [call.args[0], call.args[1]])
            elif name in vector_unary:
                with a.role("elementwise"):
                    emit_vector_elementwise(dst, vector_unary[name],
                                            [call.args[0]])
            elif name == "relax.nn.silu":
                with a.role("activation"):
                    emit_activation(dst, call.args[0], ACT_SILU)
            elif name == "relax.nn.gelu":
                with a.role("activation"):
                    emit_activation(dst, call.args[0], ACT_GELU)
            else:
                raise CodegenError(f"unsupported op for v09 backend: {name}")
            if emit_log is not None:
                arg_shapes = [mp.shape.get(arg) for arg in call.args
                              if arg in mp.shape]
                emit_log.append((name, mp.shape.get(dst), arg_shapes, start,
                                 len(a.words)))

    a.halt()
    return a
