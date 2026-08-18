"""Relax-to-ver.08 backend targeting the supplied 0818 vendor C-model.

All tensors stay row-major in G-buffer.  The new MAIN/PARTIAL descriptors make
matrix tiles directly addressable, so matmul no longer gathers 64x64 blocks into
a tile-blocked global layout.  The PE operation is still tiled to at most 64x64.
"""
from __future__ import annotations

from tvm import relax

from . import memplan as _memplan
from .isa_0818 import (
    ACT_GELU,
    ACT_SILU,
    DST,
    IMM,
    SCALAR,
    SRC1,
    SRC2,
    VECTOR,
    Asm,
)
from .runtime_0818 import GBUF_CAPACITY


class CodegenError(Exception):
    pass


def _opname(call):
    return call.op.name if hasattr(call.op, "name") else str(call.op)


def _numel(shape):
    result = 1
    for dim in shape:
        result *= dim
    return result


def plan(func, *, reuse=False):
    """Create the ver.08 physical plan: row-major only, with no weight packing."""
    return _memplan.plan(func, pack=False, pack_params=False, layouts=False, reuse=reuse,
                         fuse_oproj=False)


def compile_module(mod, func_name="main", *, tile=64, reuse=False, validate=True):
    func = mod[func_name]
    mp = plan(func, reuse=reuse)
    return compile_func(func, mp, tile=tile, validate=validate), mp


def compile_func(func, mp, *, tile=64, validate=True, emit_log=None):
    """Compile a planned Relax function into vendor ver.08 instructions.

    The vendor binary has a fixed 8192-entry G-buffer.  Its program file is
    dynamically allocated, so ``validate`` only enforces the real G-buffer cap.
    """
    if tile <= 0 or tile > 64:
        raise ValueError(f"ver.08 PE tile must be in 1..64, got {tile}")
    if validate and mp.top > GBUF_CAPACITY:
        raise CodegenError(
            f"ver.08 vendor G-buffer overflow: plan needs {mp.top} FP16 values, "
            f"capacity is {GBUF_CAPACITY}"
        )

    a = Asm()
    off = mp.offset

    def vector_copy(dst, src, count):
        for base in range(0, count, GBUF_CAPACITY):
            length = min(GBUF_CAPACITY, count - base)
            a.vlen(length)
            a.addr(SRC1, src + base).load(0, SRC1).v_copy()
            a.addr(DST, dst + base).save(0)

    def copy2d(dst, dst_stride, src, src_stride, rows, cols):
        for row in range(rows):
            vector_copy(dst + row * dst_stride, src + row * src_stride, cols)

    def region(operand, base, rows, cols, row, col, part_rows, part_cols):
        a.matrix_region(
            operand,
            base,
            rows,
            cols,
            base + row * cols + col,
            part_rows,
            part_cols,
        )

    def emit_matmul(dst, lhs, rhs):
        try:
            rows, inner = mp.shape[lhs]
            inner2, cols = mp.shape[rhs]
        except ValueError as exc:
            raise CodegenError("ver.08 matmul expects two rank-2 tensors") from exc
        if inner != inner2:
            raise CodegenError(f"matmul K mismatch {inner} vs {inner2}")
        lhs_base, rhs_base, dst_base = off[lhs], off[rhs], off[dst]
        for row in range(0, rows, tile):
            part_rows = min(tile, rows - row)
            for col in range(0, cols, tile):
                part_cols = min(tile, cols - col)
                for k_index, inner_start in enumerate(range(0, inner, tile)):
                    part_inner = min(tile, inner - inner_start)
                    region(SRC1, lhs_base, rows, inner, row, inner_start,
                           part_rows, part_inner)
                    a.load(1, SRC1)
                    region(SRC2, rhs_base, inner, cols, inner_start, col,
                           part_inner, part_cols)
                    a.load(1, SRC2)
                    a.m_mul(VECTOR, mac=k_index != 0)
                region(DST, dst_base, rows, cols, row, col, part_rows, part_cols)
                a.save(1)

    def emit_row_sum(dst, src):
        rows, cols = mp.shape[src]
        if mp.shape[dst][-1] != 1:
            raise CodegenError(f"sum output must keep the last axis, got {mp.shape[dst]}")
        for row in range(rows):
            a.vlen(cols).addr(SRC1, off[src] + row * cols).load(0, SRC1)
            a.v_reduce_sum().vlen(1).addr(DST, off[dst] + row).save(0)

    def emit_row_max(dst, src):
        """Correct row max despite the vendor 0-initialized native reduce-max.

        Load one column at a time over up to 64 rows and fold with vector max.
        Starting from the actual first column preserves all-negative rows.
        """
        rows, cols = mp.shape[src]
        if cols < 1 or mp.shape[dst][-1] != 1:
            raise CodegenError(f"max expects non-empty [R,C] -> [R,1], got {mp.shape[src]}")
        for row in range(0, rows, tile):
            part_rows = min(tile, rows - row)
            accumulator = off[dst] + row
            region(SRC1, off[src], rows, cols, row, 0, part_rows, 1)
            a.load(1, SRC1, strided=1, ncols=1, start=0)
            a.vlen(part_rows).v_copy().addr(DST, accumulator).save(0)
            for col in range(1, cols):
                region(SRC1, off[src], rows, cols, row, col, part_rows, 1)
                a.load(1, SRC1, strided=1, ncols=1, start=0)
                a.vlen(part_rows).addr(SRC2, accumulator).load(0, SRC2)
                a.v_max(VECTOR).addr(DST, accumulator).save(0)

    def emit_broadcast(dst, src):
        dst_shape = mp.shape[dst]
        src_shape = mp.shape[src]
        total = _numel(dst_shape)
        src_total = _numel(src_shape)
        if src_total == total:
            vector_copy(off[dst], off[src], total)
            return
        if src_total == 1:
            for base in range(0, total, GBUF_CAPACITY):
                length = min(GBUF_CAPACITY, total - base)
                a.vlen(length).v_broadcast(SCALAR, off[src])
                a.addr(DST, off[dst] + base).save(0)
            return
        if len(dst_shape) != 2:
            raise CodegenError(f"broadcast {src_shape} -> {dst_shape} unsupported")
        rows, cols = dst_shape
        row_source = ((len(src_shape) == 1 and src_shape[0] == cols) or
                      (len(src_shape) == 2 and src_shape == [1, cols]))
        col_source = len(src_shape) == 2 and src_shape == [rows, 1]
        if row_source:
            for row in range(rows):
                vector_copy(off[dst] + row * cols, off[src], cols)
            return
        if col_source:
            for row in range(rows):
                for col in range(0, cols, GBUF_CAPACITY):
                    length = min(GBUF_CAPACITY, cols - col)
                    a.vlen(length).v_broadcast(SCALAR, off[src] + row)
                    a.addr(DST, off[dst] + row * cols + col).save(0)
            return
        raise CodegenError(f"broadcast {src_shape} -> {dst_shape} unsupported")

    def emit_transpose(dst, src):
        if len(mp.shape[src]) != 2:
            raise CodegenError(f"permute_dims expects rank 2, got {mp.shape[src]}")
        rows, cols = mp.shape[src]
        for col in range(0, cols, tile):
            part_cols = min(tile, cols - col)
            for row in range(0, rows, tile):
                part_rows = min(tile, rows - row)
                region(SRC1, off[src], rows, cols, row, col, part_rows, part_cols)
                a.load(1, SRC1, strided=1, ncols=part_cols, start=0)
                # The strided load already packs the tile in transposed row-major
                # order.  Re-describe that PE buffer as [part_cols, part_rows],
                # run +0 to establish matrix output dimensions, and save it directly
                # into the row-major destination sub-region.
                a.shape(SRC1, part_cols, part_rows).m_add(IMM, 0)
                region(DST, off[dst], cols, rows, col, row, part_cols, part_rows)
                a.save(1)

    def emit_strided_slice(dst, call):
        src = call.args[0]
        axes = [int(value.value) for value in call.args[1]]
        begin = [int(value.value) for value in call.args[2]]
        end = [int(value.value) for value in call.args[3]]
        if len(mp.shape[src]) != 2 or axes != [1] and axes != [-1]:
            raise CodegenError(f"strided_slice only supports a rank-2 last axis, axes={axes}")
        rows, cols = mp.shape[src]
        first = begin[0] + (cols if begin[0] < 0 else 0)
        last = min(end[0] + (cols if end[0] < 0 else 0), cols)
        width = last - first
        if width != mp.shape[dst][1]:
            raise CodegenError("strided_slice steps other than one are unsupported")
        copy2d(off[dst], width, off[src] + first, cols, rows, width)

    def emit_concat(dst, call):
        sources = list(call.args[0].fields)
        if int(call.attrs.axis) not in (1, -1):
            raise CodegenError("concat only supports the last axis of rank-2 tensors")
        rows, dst_cols = mp.shape[dst]
        dst_col = 0
        for src in sources:
            src_rows, src_cols = mp.shape[src]
            if src_rows != rows:
                raise CodegenError("concat row mismatch")
            copy2d(off[dst] + dst_col, dst_cols, off[src], src_cols, rows, src_cols)
            dst_col += src_cols

    def emit_vector_elementwise(dst, method, args):
        count = _numel(mp.shape[dst])
        for arg in args:
            if _numel(mp.shape[arg]) != count:
                raise CodegenError(
                    f"elementwise {mp.shape[arg]} -> {mp.shape[dst]} requires an explicit "
                    "broadcast_to before ver.08 lowering"
                )
        for base in range(0, count, GBUF_CAPACITY):
            length = min(GBUF_CAPACITY, count - base)
            a.vlen(length).addr(SRC1, off[args[0]] + base).load(0, SRC1)
            if len(args) == 2:
                a.addr(SRC2, off[args[1]] + base).load(0, SRC2)
                method(VECTOR)
            else:
                method()
            a.addr(DST, off[dst] + base).save(0)

    def emit_activation(dst, src, activation):
        shape = mp.shape[src]
        rows = _numel(shape[:-1]) if len(shape) > 1 else 1
        cols = shape[-1]
        for row in range(0, rows, tile):
            part_rows = min(tile, rows - row)
            for col in range(0, cols, tile):
                part_cols = min(tile, cols - col)
                region(SRC1, off[src], rows, cols, row, col, part_rows, part_cols)
                a.load(1, SRC1).m_add(IMM, 0, activation)
                region(DST, off[dst], rows, cols, row, col, part_rows, part_cols)
                a.save(1)

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
            if name == "relax.matmul":
                with a.role("matmul"):
                    emit_matmul(dst, call.args[0], call.args[1])
            elif name in ("relax.sum", "relax.max"):
                src = call.args[0]
                axis = [int(value) % len(mp.shape[src]) for value in call.attrs.axis]
                if axis != [len(mp.shape[src]) - 1]:
                    raise CodegenError(f"{name} only supports last-axis keepdims, axis={axis}")
                with a.role("reduce"):
                    (emit_row_sum if name == "relax.sum" else emit_row_max)(dst, src)
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
                    emit_vector_elementwise(dst, vector_binary[name], [call.args[0], call.args[1]])
            elif name in vector_unary:
                with a.role("elementwise"):
                    emit_vector_elementwise(dst, vector_unary[name], [call.args[0]])
            elif name == "relax.nn.silu":
                with a.role("activation"):
                    emit_activation(dst, call.args[0], ACT_SILU)
            elif name in ("relax.nn.gelu", "relax.nn.gelu_tanh"):
                with a.role("activation"):
                    emit_activation(dst, call.args[0], ACT_GELU)
            else:
                raise CodegenError(f"unsupported op for ver.08 backend: {name}")
            if emit_log is not None:
                arg_shapes = [mp.shape.get(arg) for arg in call.args if arg in mp.shape]
                emit_log.append((name, mp.shape.get(dst), arg_shapes, start, len(a.words)))

    a.finish()
    return a
