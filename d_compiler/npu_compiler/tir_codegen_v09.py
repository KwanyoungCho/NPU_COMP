"""Scheduled TIR -> v09 ISA.

Retargets the compiler-v2 TIR walker (tir_backend.py / v2_backend.py) to the
v09 machine.  The reusable parts are kept verbatim in spirit -- intrinsic
declarations, the tensorize schedule, loop unrolling and affine index
evaluation -- while every emission goes through :mod:`isa_v09`.

The v09 target simplifies the 0710 walker considerably: MAIN/PARTIAL
descriptors address strided sub-tiles directly, so the gather/scatter of tiles
into contiguous scratch that 0710 required disappears.  A K-tile chain becomes
descriptor setup plus ``m_mul(mac=...)`` and a single ``save``, which is the
same instruction sequence the Relax backend emits -- that is what makes the
kernel-level bit-exactness gate meaningful.
"""
from __future__ import annotations

import tvm
import tvm.tir as tir
from tvm import arith

from .isa_0818 import DST, IMM, SRC1, SRC2, VECTOR
from .npu_intrin import VLEN
from .tir_backend import TILE, TirBackendError, _TIR_BINOPS


class V09TirError(RuntimeError):
    pass


class Walker:
    """Interpret scheduled TIR and emit v09 instructions.

    ``bases`` maps a buffer's data Var to its element offset in the flat global
    buffer; emission assumes operands are already staged (S4 adds the DMA).
    """

    def __init__(self, asm, bases, stage):
        self.a = asm
        self.bases = dict(bases)
        self.stage = stage              # _Stager for SRAM residence
        self.env = {}                   # tir Var -> int
        self.ana = arith.Analyzer()
        self.pending = None             # (c, sc, k_index) chain state
        self.zeroed = set()

    # ---- index evaluation (ported from tir_backend) ----

    def ev(self, expr):
        value = self._ev_fast(expr)
        if value is not None:
            return value
        env = {k: tir.IntImm(k.dtype or "int64", v) for k, v in self.env.items()}
        folded = self.ana.simplify(tir.stmt_functor.substitute(expr, env))
        if not isinstance(folded, tir.IntImm):
            raise V09TirError(f"cannot const-evaluate: {expr}")
        return int(folded.value)

    def _ev_fast(self, e):
        kind = type(e)
        if kind is tir.Var or kind is tir.SizeVar:
            return self.env.get(e)
        if kind is tir.IntImm:
            return int(e.value)
        op = _TIR_BINOPS.get(kind)
        if op is not None:
            left = self._ev_fast(e.a)
            if left is None:
                return None
            right = self._ev_fast(e.b)
            return None if right is None else op(left, right)
        return None

    def ptr(self, access):
        """access_ptr call -> element offset of the accessed tile."""
        if not (isinstance(access, tir.Call)
                and access.op.name == "tir.tvm_access_ptr"):
            raise V09TirError(f"expected access_ptr, got {access}")
        data, offset = access.args[1], access.args[2]
        if data not in self.bases:
            raise V09TirError(f"unknown buffer {data}")
        return self.bases[data] + self.ev(offset)

    # ---- statement walk ----

    def visit(self, stmt):
        kind = type(stmt)
        if kind is tir.For:
            if self._match_nest(stmt):
                return
            begin, extent = self.ev(stmt.min), self.ev(stmt.extent)
            for value in range(begin, begin + extent):
                self.env[stmt.loop_var] = value
                self.visit(stmt.body)
            self.env.pop(stmt.loop_var, None)
        elif kind is tir.SeqStmt:
            for sub in stmt:
                self.visit(sub)
        elif kind is tir.BlockRealize:
            block = stmt.block
            for var, value in zip(block.iter_vars, stmt.iter_values):
                self.env[var.var] = self.ev(value)
            for match in block.match_buffers:
                self._bind_match(match)
            if block.init is not None:
                self.visit(block.init)
            self.visit(block.body)
        elif kind is tir.Evaluate:
            self.call(stmt.value)
        elif kind is tir.BufferStore or kind is tir.LetStmt:
            raise V09TirError(f"un-tensorized statement reached codegen: {kind}")
        elif kind is tir.Block:
            self.visit(stmt.body)
        elif kind is tir.AttrStmt or kind is tir.AllocateConst:
            self.visit(stmt.body)
        else:
            raise V09TirError(f"unhandled TIR node {kind}")

    # ---- whole-loop-nest patterns (movement and reduction) ----------------
    # Ops that tensorize poorly because their extent varies (row reductions,
    # broadcast, transpose, slice) are recognised as a complete loop nest and
    # emitted as one vector instruction per output row, which is what the ISA
    # actually provides.

    def _collect_nest(self, stmt):
        loops, node = [], stmt
        while isinstance(node, tir.For):
            loops.append(node)
            node = node.body
        return (loops, node) if isinstance(node, tir.BlockRealize) else None

    def _match_nest(self, stmt):
        collected = self._collect_nest(stmt)
        if collected is None:
            return False
        loops, realize = collected
        block = realize.block
        if block.match_buffers:
            return False                       # tensorized: handled elsewhere
        body = block.body
        if not isinstance(body, tir.BufferStore):
            return False
        for iter_var, value in zip(block.iter_vars, realize.iter_values):
            if not isinstance(value, tir.Var):
                return False
            self.env[iter_var.var] = 0
        spatial = [v.var for v in block.iter_vars if v.iter_type == 0]
        reduce_axes = [v.var for v in block.iter_vars if v.iter_type == 2]
        extents = {}
        for loop, iter_var in zip(loops, block.iter_vars):
            extents[iter_var.var] = self.ev(loop.extent)
        try:
            if block.init is not None and len(reduce_axes) == 1:
                self._emit_reduction(body, block.init, spatial, reduce_axes[0],
                                     extents)
            elif not reduce_axes:
                self._emit_movement(body, spatial, extents)
            else:
                return False
        finally:
            for iter_var in block.iter_vars:
                self.env.pop(iter_var.var, None)
        return True

    def _affine(self, expr, axis):
        """(base, stride) of an index expression along one symbolic axis."""
        self.env[axis] = 0
        base = self.ev(expr)
        self.env[axis] = 1
        stride = self.ev(expr) - base
        self.env[axis] = 0
        return base, stride

    @staticmethod
    def _flat(buffer, indices, walker):
        """Absolute element offset of a buffer access, given the current env."""
        if buffer.data not in walker.bases:
            raise V09TirError(f"unplaced buffer {buffer.name}")
        shape = [int(d) for d in buffer.shape]
        offset, scale = 0, 1
        for dim, index in reversed(list(zip(shape, indices))):
            offset += walker.ev(index) * scale
            scale *= dim
        return walker.bases[buffer.data] + offset

    def _outer_positions(self, axes, extents):
        """Iterate concrete values for every axis except the innermost."""
        if not axes:
            yield ()
            return
        counts = [extents[a] for a in axes]
        total = 1
        for c in counts:
            total *= c
        for flat in range(total):
            values, rest = [], flat
            for c in reversed(counts):
                values.append(rest % c)
                rest //= c
            yield tuple(reversed(values))

    def _emit_reduction(self, store, init, spatial, axis, extents):
        value = store.value
        if isinstance(value, tir.Add):
            method, kind = "v_reduce_sum", "sum"
        elif isinstance(value, tir.Max):
            method, kind = "v_reduce_max", "max"
        else:
            raise V09TirError(f"unsupported reduction {type(value)}")
        source = value.b if isinstance(value.b, tir.BufferLoad) else value.a
        if not isinstance(source, tir.BufferLoad):
            raise V09TirError("reduction source is not a buffer load")
        length = extents[axis]
        self.flush()
        for position in self._outer_positions(spatial, extents):
            for var, val in zip(spatial, position):
                self.env[var] = val
            base, stride = self._affine_flat(source, axis)
            if stride != 1:
                raise V09TirError(f"reduction needs a contiguous axis, got {stride}")
            dst = self._flat(store.buffer, store.indices, self)
            asm = self.a
            asm.vlen(length)
            self.stage.vector(SRC1, base)
            asm.load(0, SRC1)
            getattr(asm, method)()
            self.stage.vector(DST, dst)
            asm.save(0)

    def _affine_flat(self, load, axis):
        """(base element offset, stride) of a buffer load along one axis."""
        self.env[axis] = 0
        base = self._flat(load.buffer, load.indices, self)
        self.env[axis] = 1
        stride = self._flat(load.buffer, load.indices, self) - base
        self.env[axis] = 0
        return base, stride

    def _emit_movement(self, store, spatial, extents):
        value = store.value
        if not isinstance(value, tir.BufferLoad):
            raise V09TirError(f"unsupported movement value {type(value)}")
        inner = spatial[-1]
        outer = spatial[:-1]
        length = extents[inner]
        self.flush()
        for position in self._outer_positions(outer, extents):
            for var, val in zip(outer, position):
                self.env[var] = val
            src_base, src_stride = self._affine_flat(value, inner)
            dst_base, dst_stride = self._affine_flat_store(store, inner)
            if dst_stride != 1:
                raise V09TirError(f"movement destination stride {dst_stride}")
            asm = self.a
            if src_stride == 1:                      # contiguous copy / slice
                asm.vlen(length)
                self.stage.vector(SRC1, src_base)
                asm.load(0, SRC1)
                asm.v_copy()
            elif src_stride == 0:                    # broadcast along the row
                asm.vlen(length)
                self.stage.broadcast(src_base)
            else:
                # non-unit source stride (transpose): the vector load only
                # reads contiguously, so use the matrix strided load, which
                # gathers one column, then copy it out as a vector
                self.stage.strided(SRC1, src_base, src_stride, length)
                asm.load(1, SRC1, strided=1, ncols=1, start=0)
                asm.vlen(length)
                asm.v_copy()
            self.stage.vector(DST, dst_base)
            asm.save(0)

    def _affine_flat_store(self, store, axis):
        self.env[axis] = 0
        base = self._flat(store.buffer, store.indices, self)
        self.env[axis] = 1
        stride = self._flat(store.buffer, store.indices, self) - base
        self.env[axis] = 0
        return base, stride

    def _bind_match(self, match):
        """A tensorized block views a tile of a root buffer through
        match_buffer; bind the view's data Var to the tile's element offset and
        its symbolic stride to the parent row width."""
        source, view = match.source, match.buffer
        parent = source.buffer.data
        if parent not in self.bases:
            raise V09TirError(f"match_buffer source is not a known buffer: "
                              f"{source.buffer.name}")
        row, col = source.region[0].min, source.region[1].min
        stride = int(source.buffer.shape[1])
        # a view shares the parent's data pointer; its position is carried by
        # the symbolic elem_offset that access_ptr adds
        self.bases[view.data] = self.bases[parent]
        if isinstance(view.elem_offset, tir.Var):
            self.env[view.elem_offset] = self.ev(row) * stride + self.ev(col)
        for symbol, value in zip(view.strides, (stride, 1)):
            if isinstance(symbol, tir.Var):
                self.env[symbol] = value

    def call(self, expr):
        if not (isinstance(expr, tir.Call)
                and expr.op.name == "tir.call_extern"):
            raise V09TirError(f"unhandled call {expr}")
        name = expr.args[0].value
        if name == "npu_fill_zero":
            self.zeroed.add(self.ptr(expr.args[1]))
        elif name == "npu_gemm_acc":
            c, sc = self.ptr(expr.args[1]), self.ev(expr.args[2])
            a, sa = self.ptr(expr.args[3]), self.ev(expr.args[4])
            b, sb = self.ptr(expr.args[5]), self.ev(expr.args[6])
            self.emit_gemm(c, sc, a, sa, b, sb)
        elif name.startswith("npu_ew2_"):
            self.flush()
            self.emit_binary(name[len("npu_ew2_"):], self.ptr(expr.args[1]),
                             self.ptr(expr.args[2]), self.ptr(expr.args[3]),
                             self.ev(expr.args[4]))
        elif name.startswith("npu_ew1_"):
            self.flush()
            self.emit_unary(name[len("npu_ew1_"):], self.ptr(expr.args[1]),
                            self.ptr(expr.args[2]), self.ev(expr.args[3]))
        else:
            raise V09TirError(f"unknown intrinsic {name}")

    # ---- v09 emission ----

    def emit_gemm(self, c, sc, a, sa, b, sb):
        """One 64x64x64 tile of C += A @ B.

        Consecutive calls to the same C tile chain through the PE accumulator
        with the MAC bit and store once, matching the Relax backend's sequence.
        """
        first = c in self.zeroed
        if first:
            self.zeroed.discard(c)
            self.flush()
            self.pending = (c, sc)
        elif self.pending is None or self.pending[0] != c:
            raise V09TirError(f"accumulate into unseen C tile @{c}")
        asm = self.a
        self.stage.region(SRC1, a, sa, TILE, TILE)
        asm.load(1, SRC1)
        self.stage.region(SRC2, b, sb, TILE, TILE)
        asm.load(1, SRC2)
        asm.m_mul(VECTOR, mac=not first)

    _BINARY = {"add": "v_add", "subtract": "v_sub",
               "multiply": "v_mul", "divide": "v_div"}
    _UNARY = {"sqrt": "v_sqrt", "exp": "v_exp", "negative": "v_sign_inv",
              "cos": "v_cos", "sin": "v_sin"}

    def emit_binary(self, op, dst, lhs, rhs, count):
        method = self._BINARY.get(op)
        if method is None:
            raise V09TirError(f"unsupported binary op {op}")
        asm = self.a
        asm.vlen(count)
        self.stage.vector(SRC1, lhs)
        asm.load(0, SRC1)
        self.stage.vector(SRC2, rhs)
        asm.load(0, SRC2)
        getattr(asm, method)(VECTOR)
        self.stage.vector(DST, dst)
        asm.save(0)

    def emit_unary(self, op, dst, src, count):
        method = self._UNARY.get(op)
        if method is None:
            raise V09TirError(f"unsupported unary op {op}")
        asm = self.a
        asm.vlen(count)
        self.stage.vector(SRC1, src)
        asm.load(0, SRC1)
        getattr(asm, method)()
        self.stage.vector(DST, dst)
        asm.save(0)

    def flush(self):
        """Store the finished accumulator tile."""
        if self.pending is None:
            return
        c, sc = self.pending
        self.stage.region(DST, c, sc, TILE, TILE)
        self.a.save(1)
        self.pending = None

    def run(self, prim_func, buffer_bases):
        self.bases.update(buffer_bases)
        self.visit(prim_func.body)
        self.flush()
        return self.a
