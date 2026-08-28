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
from .npu_intrin import SRAM_SCOPE, VLEN
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
        self.stage = stage              # descriptor/DMA emitter
        self.env = {}                   # tir Var -> int
        self.ana = arith.Analyzer()
        self.pending = None             # C tile currently accumulating
        self.zeroed = set()
        self.scopes = {}                # buffer data Var -> "global" | SRAM_SCOPE
        self.sram = {}                  # buffer data Var -> SRAM nibble base
        self.constants = {}             # float value -> SRAM nibble of that scalar
        self.scratch_slots = ()         # SRAM nibbles for expression temporaries
        self.depth = 0
        self.inner_base = 0             # offset applied to the innermost axis
        self.stored = {}                # C tile address -> stride, once stored

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
        if data in self.sram:
            return self.sram[data] + self.ev(offset)
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
        elif kind is tir.IfThenElse:
            # the program is fully unrolled, so every guard is decidable here
            if self.ev(stmt.condition):
                self.visit(stmt.then_case)
            elif stmt.else_case is not None:
                self.visit(stmt.else_case)
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
        if not isinstance(block.body, tir.BufferStore):
            return False
        # after compute_at the block's iter values are affine in the enclosing
        # loops, so substitute them away and work in terms of the local loops
        subst = {iv.var: value for iv, value
                 in zip(block.iter_vars, realize.iter_values)}
        body = tir.stmt_functor.substitute(block.body, subst)
        init = (tir.stmt_functor.substitute(block.init, subst)
                if block.init is not None else None)
        reduce_vars = set()
        for iter_var, value in zip(block.iter_vars, realize.iter_values):
            if iter_var.iter_type == 2:
                reduce_vars |= set(tir.analysis.undefined_vars(value))
        local = [loop.loop_var for loop in loops]
        extents = {loop.loop_var: self.ev(loop.extent) for loop in loops}
        spatial = [v for v in local if v not in reduce_vars]
        reduce_axes = [v for v in local if v in reduce_vars]
        saved = {v: self.env.get(v) for v in local}
        for v in local:
            self.env[v] = 0
        try:
            if init is not None and len(reduce_axes) == 1:
                self._emit_reduction(body, init, spatial, reduce_axes[0], extents)
            elif not reduce_axes:
                self._emit_pointwise(body, spatial, extents)
            else:
                return False
        finally:
            for v, value in saved.items():
                if value is None:
                    self.env.pop(v, None)
                else:
                    self.env[v] = value
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
        """Absolute address of a buffer access in that space's natural unit:
        elements for global memory, nibbles for SRAM."""
        shape = [int(d) for d in buffer.shape]
        offset, scale = 0, 1
        for dim, index in reversed(list(zip(shape, indices))):
            offset += walker.ev(index) * scale
            scale *= dim
        if buffer.data in walker.sram:
            return walker.sram[buffer.data] + offset * 4
        if buffer.data not in walker.bases:
            raise V09TirError(f"unplaced buffer {buffer.name}")
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
        accumulator = store.buffer
        summand = None
        for side in (value.a, value.b):
            if not (isinstance(side, tir.BufferLoad)
                    and side.buffer.data is accumulator.data):
                summand = side
        if summand is None:
            raise V09TirError("reduction has no summand")
        length = extents[axis]
        self.flush()
        for position in self._outer_positions(spatial, extents):
            for var, val in zip(spatial, position):
                self.env[var] = val
            asm = self.a
            if isinstance(summand, tir.BufferLoad):
                base, stride = self._affine_flat(summand, axis)
                unit = 4 if summand.buffer.data in self.sram else 1
                if stride != unit:
                    raise V09TirError(
                        f"reduction needs a contiguous axis, got {stride}")
                asm.vlen(length)
                self.stage.vector(SRC1, base)
                asm.load(0, SRC1)
            else:
                # compound summand (e.g. x*x): materialize it, then reduce
                self.depth = 0
                nibble = self._materialize(summand, axis, length)
                asm.vlen(length)
                self.stage.vector(SRC1, nibble)
                asm.load(0, SRC1)
            getattr(asm, method)()
            dst = self._flat(store.buffer, store.indices, self)
            self.stage.vector(DST, dst)
            asm.save(0)

    def _pointwise_into(self, dst_nibble, expr, inner, length):
        """Emit a pointwise expression over `length` elements into SRAM."""
        asm = self.a
        method = self._POINTWISE.get(type(expr))
        if method is not None:
            asm.vlen(length)
            self._operand(expr.a, SRC1, inner, length)
            self._operand(expr.b, SRC2, inner, length)
            asm.vlen(length)
            getattr(asm, method)(VECTOR)
        elif isinstance(expr, tir.Call):
            name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
            call = self._CALLS.get(name)
            if call is None:
                raise V09TirError(f"unsupported intrinsic {name}")
            asm.vlen(length)
            self._operand(expr.args[0], SRC1, inner, length)
            asm.vlen(length)
            getattr(asm, call)()
        else:
            raise V09TirError(f"unsupported expression {type(expr)}")
        self.stage.vector(DST, dst_nibble)
        asm.save(0)

    def _affine_flat(self, load, axis):
        """(base address, stride) of a buffer load along one axis, taken from
        ``inner_base`` so a row can be emitted in pieces."""
        self.env[axis] = self.inner_base
        base = self._flat(load.buffer, load.indices, self)
        self.env[axis] = self.inner_base + 1
        stride = self._flat(load.buffer, load.indices, self) - base
        self.env[axis] = self.inner_base
        return base, stride

    _POINTWISE = {tir.Add: "v_add", tir.Sub: "v_sub",
                  tir.Mul: "v_mul", tir.Div: "v_div"}
    _CALLS = {"tir.exp": "v_exp", "tir.sqrt": "v_sqrt",
              "tir.cos": "v_cos", "tir.sin": "v_sin"}

    def _emit_pointwise(self, store, spatial, extents):
        """A per-element block: copy, or an arbitrary pointwise expression."""
        value = store.value
        if isinstance(value, tir.BufferLoad):
            return self._emit_movement(store, spatial, extents)
        inner = spatial[-1]
        outer = spatial[:-1]
        length = extents[inner]
        self.flush()
        for position in self._outer_positions(outer, extents):
            for var, val in zip(outer, position):
                self.env[var] = val
            unit = 4 if store.buffer.data in self.sram else 1
            for expr, base, count in self._pieces(value, inner, length):
                if count <= 0:
                    continue
                self.inner_base = base
                dst, dst_stride = self._affine_flat_store(store, inner)
                if dst_stride != unit:
                    raise V09TirError(
                        f"pointwise destination stride {dst_stride}")
                self.depth = 0
                self._materialize(expr, inner, count, into=dst)
            self.inner_base = 0

    def _pieces(self, value, inner, length):
        """Split a row into (expression, start, count) pieces.

        A conditional select (``if_then_else`` from concat or from
        ``pad_einsum``'s fill) becomes one piece per run of the condition; the
        condition is not assumed to hold on a prefix, since concat's holds on
        the tail.
        """
        name = (value.op.name if isinstance(value, tir.Call)
                and hasattr(value.op, "name") else None)
        if name != "tir.if_then_else":
            return [(value, 0, length)]
        condition, taken, other = value.args
        saved = self.env.get(inner)
        flags = []
        for index in range(length):
            self.env[inner] = index
            flags.append(bool(self.ev(condition)))
        if saved is None:
            self.env.pop(inner, None)
        else:
            self.env[inner] = saved
        pieces, start = [], 0
        for index in range(1, length + 1):
            if index == length or flags[index] != flags[start]:
                pieces.append((taken if flags[start] else other,
                               start, index - start))
                start = index
        return pieces

    # ---- expression materialization -------------------------------------
    # The vector unit applies one operation to a whole vector, so an
    # expression tree is serialized into vector steps with SRAM temporaries.

    def _slot(self):
        if self.depth >= len(self.scratch_slots):
            raise V09TirError("expression nesting exceeds the scratch slots")
        nibble = self.scratch_slots[self.depth]
        self.depth += 1
        return nibble

    def _materialize(self, expr, inner, length, into=None):
        """Emit `expr` over `length` elements; return the SRAM address holding it."""
        asm = self.a
        if isinstance(expr, tir.BufferLoad):
            base, stride = self._affine_flat(expr, inner)
            unit = 4 if expr.buffer.data in self.sram else 1
            if stride == 0:
                # a per-row scalar (e.g. the reciprocal norm): broadcast it
                target = into if into is not None else self._slot()
                asm.vlen(length)
                self.stage.broadcast(base)
                self.stage.vector(DST, target)
                asm.save(0)
                return target
            if stride != unit:
                raise V09TirError(f"operand stride {stride} is not contiguous")
            if into is None:
                return base
            asm.vlen(length)
            self.stage.vector(SRC1, base)
            asm.load(0, SRC1)
            asm.v_copy()
            self.stage.vector(DST, into)
            asm.save(0)
            return into
        if isinstance(expr, (tir.FloatImm, tir.IntImm)):
            value = float(expr.value)
            if value not in self.constants:
                raise V09TirError(f"constant {value} not in the pool")
            target = into if into is not None else self._slot()
            asm.vlen(length)
            self.stage.broadcast(self.constants[value])
            self.stage.vector(DST, target)
            asm.save(0)
            return target
        method = self._POINTWISE.get(type(expr))
        if method is not None:
            depth = self.depth
            left = self._materialize(expr.a, inner, length)
            right = self._materialize(expr.b, inner, length)
            self.depth = depth
            target = into if into is not None else self._slot()
            asm.vlen(length)
            self.stage.vector(SRC1, left)
            asm.load(0, SRC1)
            self.stage.vector(SRC2, right)
            asm.load(0, SRC2)
            getattr(asm, method)(VECTOR)
            self.stage.vector(DST, target)
            asm.save(0)
            return target
        if isinstance(expr, tir.Call):
            name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
            call = self._CALLS.get(name)
            if call is None and name == "tir.rsqrt":
                return self._rsqrt(expr, inner, length, into)
            if call is None and name == "tir.sigmoid":
                return self._sigmoid(expr, inner, length, into)
            if call is None:
                raise V09TirError(f"unsupported intrinsic {name}")
            depth = self.depth
            operand = self._materialize(expr.args[0], inner, length)
            self.depth = depth
            target = into if into is not None else self._slot()
            asm.vlen(length)
            self.stage.vector(SRC1, operand)
            asm.load(0, SRC1)
            getattr(asm, call)()
            self.stage.vector(DST, target)
            asm.save(0)
            return target
        raise V09TirError(f"unsupported expression {type(expr)}")

    def _sigmoid(self, expr, inner, length, into):
        """1/(1+exp(-x)) from the primitives the unit provides."""
        asm = self.a
        depth = self.depth
        operand = self._materialize(expr.args[0], inner, length)
        self.depth = depth
        negated = self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, operand)
        asm.load(0, SRC1)
        asm.v_sign_inv()
        self.stage.vector(DST, negated)
        asm.save(0)
        exponent = self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, negated)
        asm.load(0, SRC1)
        asm.v_exp()
        self.stage.vector(DST, exponent)
        asm.save(0)
        one = self._materialize(tir.FloatImm("float16", 1.0), inner, length)
        denominator = self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, exponent)
        asm.load(0, SRC1)
        self.stage.vector(SRC2, one)
        asm.load(0, SRC2)
        asm.v_add(VECTOR)
        self.stage.vector(DST, denominator)
        asm.save(0)
        target = into if into is not None else self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, one)
        asm.load(0, SRC1)
        self.stage.vector(SRC2, denominator)
        asm.load(0, SRC2)
        asm.v_div(VECTOR)
        self.stage.vector(DST, target)
        asm.save(0)
        return target

    def _rsqrt(self, expr, inner, length, into):
        """1/sqrt(x) via the sqrt and divide the unit provides."""
        asm = self.a
        depth = self.depth
        operand = self._materialize(expr.args[0], inner, length)
        self.depth = depth
        root = self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, operand)
        asm.load(0, SRC1)
        asm.v_sqrt()
        self.stage.vector(DST, root)
        asm.save(0)
        one = self._materialize(tir.FloatImm("float16", 1.0), inner, length)
        target = into if into is not None else self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, one)
        asm.load(0, SRC1)
        self.stage.vector(SRC2, root)
        asm.load(0, SRC2)
        asm.v_div(VECTOR)
        self.stage.vector(DST, target)
        asm.save(0)
        return target

    def _emit_movement(self, store, spatial, extents):
        value = store.value
        if not isinstance(value, tir.BufferLoad):
            raise V09TirError("movement expects a buffer load")
        dst_space = self._space(store.buffer)
        src_space = self._space(value.buffer)
        if dst_space != src_space:
            # a cache_read / cache_write stage: this is the DMA the hand-written
            # backend used to insert by hand
            self._emit_dma(store, value, spatial, extents,
                           to_sram=dst_space == SRAM_SCOPE)
            return
        inner = spatial[-1]
        outer = spatial[:-1]
        length = extents[inner]
        self.flush()
        for position in self._outer_positions(outer, extents):
            for var, val in zip(outer, position):
                self.env[var] = val
            src_base, src_stride = self._affine_flat(value, inner)
            dst_base, dst_stride = self._affine_flat_store(store, inner)
            src_unit = 4 if value.buffer.data in self.sram else 1
            dst_unit = 4 if store.buffer.data in self.sram else 1
            if dst_stride != dst_unit:
                raise V09TirError(f"movement destination stride {dst_stride}")
            asm = self.a
            if src_stride == src_unit:               # contiguous copy / slice
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
                self.stage.strided(SRC1, src_base, src_stride // src_unit, length)
                asm.load(1, SRC1, strided=1, ncols=1, start=0)
                asm.vlen(length)
                asm.v_copy()
            self.stage.vector(DST, dst_base)
            asm.save(0)

    def _emit_dma(self, store, load, spatial, extents, to_sram):
        """Emit one GLOAD/GSTORE per staged row of a cache block."""
        inner = spatial[-1]
        outer = spatial[:-1]
        length = extents[inner]
        if not to_sram:
            # a write-back reads SRAM the accumulator still owns; staging an
            # input does not touch the PE, so it must not break a MAC chain
            self.flush()
        for position in self._outer_positions(outer, extents):
            for var, val in zip(outer, position):
                self.env[var] = val
            src_base, src_stride = self._affine_flat(load, inner)
            dst_base, dst_stride = self._affine_flat_store(store, inner)
            src_unit = 4 if load.buffer.data in self.sram else 1
            dst_unit = 4 if store.buffer.data in self.sram else 1
            if src_stride != src_unit or dst_stride != dst_unit:
                raise V09TirError("DMA stage needs contiguous rows")
            if to_sram:
                self.stage.dma_in(src_base, dst_base, length)
            else:
                self.stage.dma_out(dst_base, src_base, length)

    def _affine_flat_store(self, store, axis):
        self.env[axis] = self.inner_base
        base = self._flat(store.buffer, store.indices, self)
        self.env[axis] = self.inner_base + 1
        stride = self._flat(store.buffer, store.indices, self) - base
        self.env[axis] = self.inner_base
        return base, stride

    def _bind_match(self, match):
        """A tensorized block views a tile of a root buffer through
        match_buffer; bind the view's data Var to the tile's element offset and
        its symbolic stride to the parent row width."""
        source, view = match.source, match.buffer
        parent = source.buffer.data
        # a tile lives in the last two axes; leading axes select the batch
        shape = [int(d) for d in source.buffer.shape]
        stride = shape[-1]
        offset, scale = 0, 1
        for dim, region in reversed(list(zip(shape, source.region))):
            offset += self.ev(region.min) * scale
            scale *= dim
        # a view shares the parent's data pointer; its position is carried by
        # the symbolic elem_offset that access_ptr adds
        if parent in self.sram:
            self.sram[view.data] = self.sram[parent]
            self.scopes[view.data] = self.scopes.get(parent)
        elif parent in self.bases:
            self.bases[view.data] = self.bases[parent]
        else:
            raise V09TirError(f"match_buffer source is not a known buffer: "
                              f"{source.buffer.name}")
        unit = 4 if parent in self.sram else 1
        if isinstance(view.elem_offset, tir.Var):
            self.env[view.elem_offset] = offset * unit
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
        asm = self.a
        first = c in self.zeroed
        if first:
            self.zeroed.discard(c)
            self.flush()
            self.pending = (c, sc)
        elif self.pending is None or self.pending[0] != c:
            # the chain was interrupted (a vector op owns the PE output in
            # between), so reload the partial sum before continuing -- the
            # same trick the 0710 walker used
            if c not in self.stored:
                raise V09TirError(f"accumulate into unseen C tile @{c}")
            self.flush()
            # bring the stored partial back into the PE output register
            self.stage.region(SRC1, c, self.stored[c], TILE, TILE)
            asm.load(1, SRC1)
            asm.m_add(IMM, 0)
            self.pending = (c, sc)
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
        self.stored[c] = sc
        self.pending = None

    def declare_sram(self, buffer, nibble):
        """Place a cache-stage buffer in SRAM (addresses are nibbles there)."""
        self.sram[buffer.data] = nibble
        self.scopes[buffer.data] = SRAM_SCOPE

    def _space(self, buffer):
        return self.scopes.get(buffer.data, "global")

    def run(self, prim_func, buffer_bases):
        self.bases.update(buffer_bases)
        self.visit(prim_func.body)
        self.flush()
        return self.a


class SramEmitter:
    """Descriptor and DMA emission for the SRAM-staged schedule.

    Addresses arrive already in the right unit: SRAM operands as nibbles
    (the walker converts), global operands as element offsets, which the DMA
    turns into 32-bit cell addresses.
    """

    def __init__(self, asm):
        self.asm = asm

    # -- compute-side descriptors (SRAM nibble addresses)

    def vector(self, operand, nibble):
        self.asm.addr(operand, nibble, 1)

    def broadcast(self, nibble):
        self.asm.v_broadcast_addr(nibble)

    def region(self, operand, nibble, stride, rows, cols):
        self.asm.addr(operand, nibble, 0)
        self.asm.shape(operand, rows, stride, 0)
        self.asm.addr(operand, nibble, 1)
        self.asm.shape(operand, rows, cols, 1)

    def strided(self, operand, nibble, stride, count):
        self.asm.addr(operand, nibble, 0)
        self.asm.shape(operand, count, stride, 0)
        self.asm.addr(operand, nibble, 1)
        self.asm.shape(operand, count, 1, 1)

    # -- DMA (global element offsets <-> SRAM nibbles)

    def dma_in(self, global_elem, sram_nibble, count):
        cells, nib = self._cells(global_elem, count, sram_nibble)
        self.asm.gload(global_elem // 2, cells, nib, 1, cells)

    def dma_out(self, global_elem, sram_nibble, count):
        cells, nib = self._cells(global_elem, count, sram_nibble)
        self.asm.gstore(global_elem // 2, cells, nib, 1, cells)

    @staticmethod
    def _cells(global_elem, count, sram_nibble):
        # allocations are cell-rounded, so an odd count only reaches into the
        # tensor's own padding; an odd start would shift the SRAM image and is
        # rejected instead
        if global_elem % 2:
            raise V09TirError("DMA row must start on a 32-bit cell")
        if sram_nibble % 8:
            raise V09TirError("DMA SRAM address must be 8-nibble aligned")
        return (count + 1) // 2, sram_nibble
