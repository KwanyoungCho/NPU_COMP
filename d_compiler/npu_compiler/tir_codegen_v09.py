"""Reference Python emitter for scheduled TIR -> v09 ISA.

Unrolls static loops, evaluates indices, and emits matrix/vector operations
and DMA through SramEmitter. The native walker implements the same contract
and is checked against this emitter word-for-word.
"""
from __future__ import annotations

import tvm.tir as tir
from tvm import arith

from .isa_compute import DST, SRC1, SRC2, VECTOR
from .isa_v09 import DT_FP16, DT_INT8
from .npu_intrin import SRAM_SCOPE, TILE
from .v09_contract import CONTRACT, MAX_VLEN


_TIR_BINOPS = {
    tir.Add: lambda a, b: a + b,
    tir.Sub: lambda a, b: a - b,
    tir.Mul: lambda a, b: a * b,
    tir.FloorDiv: lambda a, b: a // b,
    tir.FloorMod: lambda a, b: a % b,
    tir.Min: min,
    tir.Max: max,
}


class V09TirError(RuntimeError):
    pass


# guards -- padding predicates and concat conditions -- are scanned per element,
# so they get the same compiled evaluation as an address
_COMPARISONS = {
    tir.LT: lambda a, b: a < b, tir.LE: lambda a, b: a <= b,
    tir.GT: lambda a, b: a > b, tir.GE: lambda a, b: a >= b,
    tir.EQ: lambda a, b: a == b, tir.NE: lambda a, b: a != b,
    tir.And: lambda a, b: a and b, tir.Or: lambda a, b: a or b,
}


class Walker:
    """Interpret scheduled TIR and emit v09 instructions.

    ``bases`` maps a buffer's data Var to its element offset in the flat global
    buffer. Explicit SRAM cache copies become DMA through ``stage``.
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
        self._match_plans = {}          # match_buffer node -> what it binds
        self._flat_cache = {}           # buffer access node -> its addresser
        self.dtypes = {}                # buffer data Var -> dtype string
        self.one_fp32 = None            # SRAM nibble of a literal fp32 1.0
        self.stored = {}                # C tile address -> stride, once stored

    # ---- index evaluation ----

    _SRAM_WIDTH = {dtype: units[0] for dtype, units in CONTRACT["storage"].items()}
    _ITEMSIZE = {dtype: units[1] for dtype, units in CONTRACT["storage"].items()}

    def width(self, data):
        """SRAM nibbles per element of the buffer behind ``data``."""
        return self._SRAM_WIDTH[self.dtypes.get(data, "float16")]

    def itemsize(self, data):
        """Bytes per element in global memory."""
        return self._ITEMSIZE[self.dtypes.get(data, "float16")]

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
        # comparisons belong here too: guards are evaluated per element, and
        # leaving them to the substitute/simplify path made them 68% of a real
        # layer's link time
        op = _TIR_BINOPS.get(kind) or _COMPARISONS.get(kind)
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
        # bytes, like every other global address (only the S3-scope harness
        # ever reaches this branch -- staged kernels compute out of SRAM)
        return self.bases[data] * 2 + self.ev(offset) * self.itemsize(data)

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
            if self._match_nest(stmt):
                return
            if not self.ev(stmt.predicate):
                return
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
        elif kind is tir.BufferStore:
            self._emit_pointwise(stmt, [], {})
        elif kind is tir.LetStmt:
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
        body, init, guard = block.body, block.init, None
        # LowerInitBlock rewrites a reduction into a guarded initial store
        # followed by the accumulating store, and ConvertBlocksToOpaque then
        # strips the iter vars, so the reduce axis has to come from the guard.
        if (isinstance(body, tir.SeqStmt) and len(body) == 2
                and isinstance(body[0], tir.IfThenElse)
                and body[0].else_case is None
                and isinstance(body[0].then_case, tir.BufferStore)
                and isinstance(body[1], tir.BufferStore)):
            init, guard, body = body[0].then_case, body[0].condition, body[1]
        if not isinstance(body, tir.BufferStore):
            return False
        # after compute_at the block's iter values are affine in the enclosing
        # loops, so substitute them away and work in terms of the local loops
        subst = {iv.var: value for iv, value
                 in zip(block.iter_vars, realize.iter_values)}
        body = tir.stmt_functor.substitute(body, subst)
        is_dma = (isinstance(body.value, tir.BufferLoad)
                  and ((body.buffer.data in self.sram)
                       != (body.value.buffer.data in self.sram)))
        if not is_dma and str(body.buffer.dtype) not in CONTRACT["generic_store"]:
            raise V09TirError("v09 generic vector lowering requires FP16 storage; "
                              f"got {body.buffer.dtype}")
        init = (tir.stmt_functor.substitute(init, subst)
                if init is not None else None)
        reduce_vars = set()
        if guard is not None:
            guard = tir.stmt_functor.substitute(guard, subst)
            reduce_vars = set(tir.analysis.undefined_vars(guard))
        for iter_var, value in zip(block.iter_vars, realize.iter_values):
            if iter_var.iter_type == 2:
                reduce_vars |= set(tir.analysis.undefined_vars(value))
        local = [loop.loop_var for loop in loops]
        extents = {loop.loop_var: self.ev(loop.extent) for loop in loops}
        spatial = [v for v in local if v not in reduce_vars]
        reduce_axes = [v for v in local if v in reduce_vars]
        if not (isinstance(realize.predicate, tir.IntImm)
                and int(realize.predicate.value) == 1):
            extents, origins = self._narrow(realize.predicate, spatial, extents)
            if any(extent == 0 for extent in extents.values()):
                return True
            shifted = {var: var + offset for var, offset in origins.items() if offset}
            if shifted:
                body = tir.stmt_functor.substitute(body, shifted)
                if init is not None:
                    init = tir.stmt_functor.substitute(init, shifted)
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
            return walker.sram[buffer.data] + offset * walker.width(buffer.data)
        if buffer.data not in walker.bases:
            raise V09TirError(f"unplaced buffer {buffer.name}")
        # global addresses are byte-based: the DMA is their only consumer, and
        # bytes are the one unit int8, fp16 and fp32 tensors share
        return walker.bases[buffer.data] * 2 \
            + offset * walker.itemsize(buffer.data)

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
        explicit_sum = (isinstance(value, tir.Call) and value.op.name == "tir.call_pure_extern"
                        and str(value.args[0].value) == CONTRACT["reductions"]["sum_update"])
        if isinstance(value, tir.Add) or explicit_sum:
            method, kind = "v_reduce_sum", "sum"
        elif isinstance(value, tir.Max):
            method, kind = "v_reduce_max", "max"
        else:
            raise V09TirError(f"unsupported reduction {type(value)}")
        accumulator = store.buffer
        summand = None
        for side in (value.args[1:3] if explicit_sum else (value.a, value.b)):
            if not (isinstance(side, tir.BufferLoad)
                    and side.buffer.data is accumulator.data):
                summand = side
        if summand is None:
            raise V09TirError("reduction has no summand")
        length = extents[axis]
        if not 0 < length <= MAX_VLEN:
            raise V09TirError(f"v09 reduction length {length} requires a segmented reduction")
        self.flush()
        for position in self._outer_positions(spatial, extents):
            for var, val in zip(spatial, position):
                self.env[var] = val
            asm = self.a
            if isinstance(summand, tir.BufferLoad):
                base, stride = self._affine_flat(summand, axis)
                data = summand.buffer.data
                unit = (self.width(data) if data in self.sram
                        else self.itemsize(data))
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

    _POINTWISE = {getattr(tir, name): method
                  for name, method in CONTRACT["binary"].items()}
    _CALLS = CONTRACT["unary"]

    def _emit_pointwise(self, store, spatial, extents):
        """A per-element block: copy, or an arbitrary pointwise expression."""
        if not spatial:
            # compute_at legitimately removes every unit loop. A scalar
            # movement/vector operation still has one element to execute.
            unit = tir.Var("npu_unit", "int32")
            spatial, extents = [unit], {unit: 1}
        value = store.value
        if (self.pending is not None and isinstance(value,tir.BufferLoad)
                and len(spatial)==2 and all(extents[v]==TILE for v in spatial)
                and self._try_matrix_copy(store,spatial,extents)):
            return
        if self._try_bulk_copy_fill(store, spatial, extents):
            return
        if isinstance(value, tir.BufferLoad):
            return self._emit_movement(store, spatial, extents)
        if str(store.buffer.dtype) not in CONTRACT["generic_store"]:
            raise V09TirError(f"v09 pointwise operation requires FP16 storage, got {store.buffer.dtype}")
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
                for offset in range(0, count, MAX_VLEN):
                    self.inner_base = base + offset
                    dst, dst_stride = self._affine_flat_store(store, inner)
                    if dst_stride != unit and count != 1:
                        raise V09TirError(
                            f"pointwise destination stride {dst_stride}")
                    self.depth = 0
                    self._materialize(expr, inner, min(count - offset, MAX_VLEN), into=dst)
            self.inner_base = 0

    def _try_bulk_copy_fill(self, store, spatial, extents):
        """Coalesce adjacent SRAM copies/fills, including padding branches.

        Only a copy or literal can cross row boundaries this way: general
        expressions may have row broadcasts and scratch-sized intermediates.
        Preflight every piece before emitting, and never merge overlapping
        source/destination regions where row execution order would matter.
        """
        def eligible(expr):
            if isinstance(expr, (tir.BufferLoad, tir.FloatImm, tir.IntImm)):
                return True
            return (isinstance(expr, tir.Call)
                    and expr.op.name == "tir.if_then_else"
                    and eligible(expr.args[1]) and eligible(expr.args[2]))

        if (not spatial or store.buffer.data not in self.sram
                or str(store.buffer.dtype) != "float16" or not eligible(store.value)):
            return False
        inner, outer = spatial[-1], spatial[:-1]
        runs = []
        try:
            for position in self._outer_positions(outer, extents):
                self.env.update(zip(outer, position))
                for expr, base, count in self._pieces(store.value, inner, extents[inner]):
                    if count <= 0:
                        continue
                    self.inner_base = base
                    dst, stride = self._affine_flat_store(store, inner)
                    if stride != 4:
                        return False
                    if isinstance(expr, tir.BufferLoad):
                        if expr.buffer.data not in self.sram or str(expr.buffer.dtype) != "float16":
                            return False
                        src, stride = self._affine_flat(expr, inner)
                        if stride != 4:
                            return False
                        kind, value = "copy", src
                    else:
                        kind, value = "fill", float(expr.value)
                    if runs and runs[-1][0] == kind:
                        prev = runs[-1]
                        adjacent = dst == prev[2] + prev[3] * 4
                        same = (value == prev[1] if kind == "fill" else
                                value == prev[1] + prev[3] * 4)
                        length = prev[3] + count
                        disjoint = (kind == "fill" or prev[1] + length * 4 <= prev[2]
                                    or prev[2] + length * 4 <= prev[1])
                        if adjacent and same and disjoint:
                            prev[3] = length
                            continue
                    runs.append([kind, value, dst, count])
        finally:
            self.inner_base = 0
        self.flush()
        for kind, value, dst, count in runs:
            offset = 0
            while offset < count:
                part = min(MAX_VLEN, count - offset)
                self.a.vlen(part)
                if kind == "copy":
                    self.stage.vector(SRC1, value + offset * 4)
                    self.a.load(0, SRC1)
                    self.a.v_copy()
                else:
                    if value not in self.constants:
                        raise V09TirError(f"constant {value} not in the pool")
                    self.stage.broadcast(self.constants[value])
                self.stage.vector(DST, dst + offset * 4)
                self.a.save(0)
                offset += part
        return True

    def _narrow(self, predicate, spatial, extents):
        """Shrink the loop extents to the part a block predicate allows.

        ``pad_einsum``'s copy-back carries a predicate clipping the padded
        tile to the real output.  The allowed region has to be a box anchored
        at the origin -- which is the shape a padding predicate always has --
        and anything else is rejected rather than silently truncated.
        """
        guard = self._compile(predicate)
        if guard is None:
            guard = lambda env: bool(self.ev(predicate))
        saved = {axis: self.env.get(axis) for axis in spatial}
        limits, minima, allowed = [0] * len(spatial), [extents[a] for a in spatial], 0
        for position in self._outer_positions(spatial, extents):
            for axis, value in zip(spatial, position):
                self.env[axis] = value
            if guard(self.env):
                allowed += 1
                limits = [max(limit, value + 1)
                          for limit, value in zip(limits, position)]
                minima = [min(low, value) for low, value in zip(minima, position)]
        for axis, value in saved.items():
            if value is None:
                self.env.pop(axis, None)
            else:
                self.env[axis] = value
        box = 1
        lengths = [max(0, high-low) for high, low in zip(limits, minima)]
        for length in lengths:
            box *= length
        if box != allowed:
            raise V09TirError("block predicate is not a rectangular region")
        narrowed = dict(extents)
        narrowed.update(zip(spatial, lengths))
        return narrowed, dict(zip(spatial, minima))

    def _pieces(self, value, inner, length):
        """Split a row into (expression, start, count) pieces.

        A conditional select (``if_then_else`` from concat or from
        ``pad_einsum``'s fill) becomes one piece per run of the condition; the
        condition is not assumed to hold on a prefix, since concat's holds on
        the tail.  The split recurses, because an N-way concat lowers to
        N-1 NESTED selects -- a 28-layer cache stack is a 27-deep chain, and
        each surviving branch may itself be another select.
        """
        saved = self.env.get(inner)
        try:
            pieces = self._split_row(value, inner, 0, length)
        finally:
            if saved is None:
                self.env.pop(inner, None)
            else:
                self.env[inner] = saved
        return pieces

    def _scanner(self, expr, var):
        """Compile ``expr`` into a plain function of one loop variable's value.

        A select's condition is tested once per element of the row, so reading
        its TIR nodes there dominated link time.  Every other variable is fixed
        while one row is scanned, so it folds to a constant now and nothing but
        integer arithmetic is left in the scan -- no FFI, and no dictionary
        keyed by TIR objects (whose hash is itself an FFI call).
        """
        kind = type(expr)
        if kind is tir.IntImm:
            value = int(expr.value)
            return lambda _: value
        if kind is tir.Var or kind is tir.SizeVar:
            if expr.same_as(var):
                return lambda index: index
            fixed = self.env.get(expr)
            return None if fixed is None else (lambda _: fixed)
        operator = _TIR_BINOPS.get(kind) or _COMPARISONS.get(kind)
        if operator is None:
            return None
        left = self._scanner(expr.a, var)
        right = self._scanner(expr.b, var)
        if left is None or right is None:
            return None
        return lambda index: operator(left(index), right(index))

    def _split_row(self, value, inner, lo, count):
        name = (value.op.name if isinstance(value, tir.Call)
                and hasattr(value.op, "name") else None)
        if isinstance(value, tir.Select):
            condition, taken, other = value.condition, value.true_value, value.false_value
        elif name == "tir.if_then_else":
            condition, taken, other = value.args
        else:
            return [(value, lo, count)]
        guard = self._scanner(condition, inner)
        if guard is None:
            def guard(index, _condition=condition):
                self.env[inner] = index
                return self.ev(_condition)
        flags = [bool(guard(index)) for index in range(lo, lo + count)]
        pieces, start = [], 0
        for index in range(1, count + 1):
            if index == count or flags[index] != flags[start]:
                branch = taken if flags[start] else other
                pieces.extend(self._split_row(branch, inner, lo + start,
                                              index - start))
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
            if str(expr.buffer.dtype) not in CONTRACT["vector_load"]:
                raise V09TirError(f"v09 vector load does not support {expr.buffer.dtype}")
            base, stride = self._affine_flat(expr, inner)
            data = expr.buffer.data
            unit = (self.width(data) if data in self.sram
                    else self.itemsize(data))
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
        if isinstance(expr, tir.Cast):
            return self._cast(expr, inner, length, into)
        if isinstance(expr, tir.Call):
            name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
            call = self._CALLS.get(name)
            composite = CONTRACT["composite"].get(name)
            if call is None and composite is not None:
                return getattr(self, composite[0])(expr, inner, length, into)
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

    def _cast(self, expr, inner, length, into):
        """int8 -> float16 via VDEQUANT; anything else has no instruction."""
        source = expr.value
        if not (expr.dtype == "float16" and isinstance(source, tir.BufferLoad)
                and source.buffer.dtype == "int8"):
            raise V09TirError(f"unsupported cast {source} -> {expr.dtype}")
        base, stride = self._affine_flat(source, inner)
        if stride != self.width(source.buffer.data):
            raise V09TirError(f"vdequant needs a contiguous int8 row, "
                              f"got stride {stride}")
        if self.one_fp32 is None:
            raise V09TirError("no fp32 1.0 was staged for vdequant")
        asm = self.a
        target = into if into is not None else self._slot()
        asm.vlen(length)
        # VDEQUANT reads SRC1's partial address with the descriptor dtype and
        # multiplies by the one scalar at the activation-scale address; a
        # literal 1.0 there leaves the conversion pure, and the per-channel
        # scale is applied by the vector multiply that follows in the TIR
        asm.addr(SRC1, base, 1)
        asm.shape_dt(SRC1, 1, length, 1, DT_INT8)
        asm.ascale(self.one_fp32)
        asm.vdequant()
        self.stage.vector(DST, target)
        asm.save(0)
        # the INT8 dtype is sticky descriptor state: restore FP16 so the next
        # consumer of SRC1 is not misread
        asm.shape_dt(SRC1, 1, length, 1, DT_FP16)
        return target

    def _tanh(self, expr, inner, length, into):
        """1 - 2/(exp(2x)+1), from the primitives the unit provides.

        Written this way rather than (e^2x - 1)/(e^2x + 1) because it
        saturates correctly: once exp(2x) overflows FP16 the quotient is 0 and
        the result is exactly 1, where the difference form would divide
        infinity by infinity.  Large negative x needs no special case either --
        exp(2x) underflows to 0 and the result is -1.
        """
        asm = self.a
        depth = self.depth
        operand = self._materialize(expr.args[0], inner, length)
        self.depth = depth
        two = self._materialize(tir.FloatImm("float16", 2.0), inner, length)
        doubled = self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, operand)
        asm.load(0, SRC1)
        self.stage.vector(SRC2, two)
        asm.load(0, SRC2)
        asm.v_mul(VECTOR)
        self.stage.vector(DST, doubled)
        asm.save(0)
        exponent = self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, doubled)
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
        quotient = self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, two)
        asm.load(0, SRC1)
        self.stage.vector(SRC2, denominator)
        asm.load(0, SRC2)
        asm.v_div(VECTOR)
        self.stage.vector(DST, quotient)
        asm.save(0)
        target = into if into is not None else self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, one)
        asm.load(0, SRC1)
        self.stage.vector(SRC2, quotient)
        asm.load(0, SRC2)
        asm.v_sub(VECTOR)
        self.stage.vector(DST, target)
        asm.save(0)
        return target

    def _fabs(self, expr, inner, length, into):
        """|x| as max(x, -x) -- the machine has no absolute-value instruction.

        This is how the hand-written W8A8 sequence did it, and the activation
        quantization needs it: the row scale is the row's absolute maximum.
        """
        asm = self.a
        operand = self._materialize(expr.args[0], inner, length)
        # max(x, -x) reads x twice. Keep a computed operand's slot live
        # while writing -x; resetting depth here aliases and destroys x.
        negated = self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, operand)
        asm.load(0, SRC1)
        asm.v_sign_inv()
        self.stage.vector(DST, negated)
        asm.save(0)
        target = into if into is not None else self._slot()
        asm.vlen(length)
        self.stage.vector(SRC1, operand)
        asm.load(0, SRC1)
        self.stage.vector(SRC2, negated)
        asm.load(0, SRC2)
        asm.v_max(VECTOR)
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
        source = self._addresser(value, inner, outer)
        target = self._addresser(store, inner, outer, store=True)
        self.flush()
        src_unit = (self.width(value.buffer.data)
                    if value.buffer.data in self.sram
                    else self.itemsize(value.buffer.data))
        dst_unit = (self.width(store.buffer.data)
                    if store.buffer.data in self.sram
                    else self.itemsize(store.buffer.data))
        for position in self._outer_positions(outer, extents):
            src_base, src_stride = source(position)
            dst_base, dst_stride = target(position)
            if dst_stride != dst_unit and length != 1:
                raise V09TirError(f"movement destination stride {dst_stride}")
            asm = self.a
            if src_stride == src_unit or length == 1:  # contiguous copy / slice
                offset = 0
                while length - offset > MAX_VLEN:
                    asm.vlen(MAX_VLEN)
                    self.stage.vector(SRC1, src_base + offset * src_unit)
                    asm.load(0, SRC1)
                    asm.v_copy()
                    self.stage.vector(DST, dst_base + offset * dst_unit)
                    asm.save(0)
                    offset += MAX_VLEN
                asm.vlen(length - offset)
                self.stage.vector(SRC1, src_base + offset * src_unit)
                asm.load(0, SRC1)
                asm.v_copy()
                self.stage.vector(DST, dst_base + offset * dst_unit)
                asm.save(0)
                continue
            elif src_stride == 0:
                asm.vlen(length)
                self.stage.broadcast(src_base)
            else:
                self.stage.strided(SRC1, src_base, src_stride // src_unit, length)
                asm.load(1, SRC1, strided=1, ncols=1, start=0)
                asm.vlen(length)
                asm.v_copy()
            self.stage.vector(DST, dst_base)
            asm.save(0)

    def _try_matrix_copy(self,store,spatial,extents):
        """Save the unchanged FP32 result twice, preserving both FP16 stores."""
        value=store.value
        src_space,dst_space=self._space(value.buffer),self._space(store.buffer)
        inner,outer=spatial[-1],spatial[:-1]
        length=extents[inner]
        source=self._addresser(value,inner,outer)
        target=self._addresser(store,inner,outer,store=True)
        if (self.pending is not None and src_space == SRAM_SCOPE and dst_space == SRAM_SCOPE
                and str(value.buffer.dtype) == str(store.buffer.dtype) == 'float16'
                and len(outer) == 1 and extents[outer[0]] == TILE and length == TILE):
            c, sc = self.pending
            dest, unit = target((0,))
            step = target((1,))[0] - dest
            if unit == 4 and step >= TILE*4 and step % 4 == 0:
                same = dest == c and step == sc*4
                disjoint = (dest+(TILE-1)*step+TILE*4 <= c or
                            c+(TILE-1)*sc*4+TILE*4 <= dest)
                if (same or disjoint) and all(
                        source((row,)) == (c+row*sc*4,4) and
                        target((row,)) == (dest+row*step,4) for row in range(TILE)):
                    self.flush()
                    if not same:
                        self.stage.region(DST,dest,step//4,TILE,TILE)
                        self.a.save(1)
                    return True
        return False

    def _emit_dma(self, store, load, spatial, extents, to_sram):
        """Emit GLOAD/GSTORE for a cache block, one per contiguous run.

        Rows that are adjacent on both sides are merged into a single
        transfer.  Merging is what makes odd row lengths work at all: a
        transfer moves whole 32-bit cells, so a row of odd length would leave
        the next row starting mid-cell, while the whole staged region starts
        at the tensor's own (cell-aligned) base.
        """
        inner = spatial[-1]
        outer = spatial[:-1]
        length = extents[inner]
        if not to_sram:
            # a write-back reads SRAM the accumulator still owns; staging an
            # input does not touch the PE, so it must not break a MAC chain
            self.flush()
        source = self._addresser(load, inner, outer)
        target = self._addresser(store, inner, outer, store=True)
        # DMA: the global side counts bytes, the SRAM side counts nibbles
        src_unit = (self.width(load.buffer.data)
                    if load.buffer.data in self.sram
                    else self.itemsize(load.buffer.data))
        dst_unit = (self.width(store.buffer.data)
                    if store.buffer.data in self.sram
                    else self.itemsize(store.buffer.data))
        # every row transfers the same number of bytes, and finding that out
        # costs two FFI attribute reads plus a hash of a TIR object -- which a
        # weight stage would pay tens of thousands of times over
        nbytes = length * self.itemsize(
            (load if to_sram else store).buffer.data)
        segments = []
        for position in self._outer_positions(outer, extents):
            src_base, src_stride = source(position)
            dst_base, dst_stride = target(position)
            if length > 1 and (src_stride != src_unit or dst_stride != dst_unit):
                # Unit output columns can leave a strided M loop after TVM
                # removes N=1. Transfer individual elements with the existing
                # cell-preserving bounce implementation.
                for index in range(length):
                    src, dst = src_base + index * src_stride, dst_base + index * dst_stride
                    glob, sram = (src, dst) if to_sram else (dst, src)
                    segments.append([glob, sram, nbytes // length])
                continue
            glob, sram = ((src_base, dst_base) if to_sram
                          else (dst_base, src_base))
            # segment lengths are bytes; every dtype packs SRAM at two nibbles
            # per byte, so the SRAM side advances at exactly twice the bytes
            if segments and glob == segments[-1][0] + segments[-1][2] \
                    and sram == segments[-1][1] + segments[-1][2] * 2:
                segments[-1][2] += nbytes     # one contiguous run on both sides
            else:
                segments.append([glob, sram, nbytes])
        self._emit_segments(segments, to_sram)

    def _compile(self, expr):
        """Turn an index expression into a Python function of the environment.

        Reading a TIR node's children crosses the FFI, so evaluating the same
        expression once per staged row dominated link time.  Compiling it once
        leaves plain arithmetic in the loop.  Returns None for anything the
        walker's general evaluator has to handle.
        """
        kind = type(expr)
        if kind is tir.IntImm:
            value = int(expr.value)
            return lambda env: value
        if kind is tir.Var or kind is tir.SizeVar:
            return lambda env: env[expr]
        operator = _TIR_BINOPS.get(kind) or _COMPARISONS.get(kind)
        if operator is None:
            return None
        a, b = expr.a, expr.b
        if kind is not tir.Add and kind is not tir.Sub:
            # a product of two varying factors, or a floordiv of one, makes the
            # address non-affine; noting it here saves a second traversal
            if kind is not tir.Mul or not (isinstance(a, tir.IntImm)
                                           or isinstance(b, tir.IntImm)):
                self._linear = False
        left, right = self._compile(a), self._compile(b)
        if left is None or right is None:
            return None
        return lambda env: operator(left(env), right(env))

    def _flattener(self, access):
        """(address function of the environment, is it affine in the loops).

        The address function is None when some index needs the walker's
        general evaluator.

        Building one reads the access node's whole structure across the FFI,
        and a tiled loop nest reaches the same node once per tile -- a staged
        weight does so tens of thousands of times -- so the result is cached
        by node address.  The entry holds the node itself, which is what stops
        that address from being reused; the same trick keys ``_match_plans``.
        The cached closure reads the environment when called, so only the
        buffer's base is frozen here, and a buffer's base does not move once
        the kernel's SRAM layout is declared.
        """
        cached = self._flat_cache.get(access.handle.value)
        if cached is not None:
            return cached[1], cached[2]
        buffer = access.buffer
        shape = [int(dim) for dim in buffer.shape]
        scales, scale = [1] * len(shape), 1
        for position in range(len(shape) - 1, -1, -1):
            scales[position] = scale
            scale *= shape[position]
        self._linear = True
        indices = [self._compile(index) for index in access.indices]
        linear = self._linear
        if any(index is None for index in indices):
            self._flat_cache[access.handle.value] = (access, None, False)
            return None, False
        data = buffer.data
        if data in self.sram:
            base, unit = self.sram[data], self.width(data)
        elif data in self.bases:
            base, unit = self.bases[data] * 2, self.itemsize(data)
        else:
            raise V09TirError(f"unplaced buffer {buffer.name}")
        pairs = list(zip(indices, scales))
        function = lambda env: base + unit * sum(
            index(env) * scale for index, scale in pairs)
        self._flat_cache[access.handle.value] = (access, function, linear)
        return function, linear

    def _addresser(self, access, inner, outer, store=False):
        """A function from an outer loop position to (address, inner stride).

        Most index expressions are affine in the loop variables, and then the
        address is one multiply-add per axis instead of an evaluation of TIR
        per row -- which matters at real dimensions, where a single matmul
        stages tens of thousands of rows.  ``repeat`` and friends index
        through a floordiv, so those fall back to evaluating each row.
        """
        compiled, linear = self._flattener(access)
        if compiled is None:
            flat = self._affine_flat_store if store else self._affine_flat
        else:
            def flat(_access, axis):
                self.env[axis] = self.inner_base
                base = compiled(self.env)
                self.env[axis] = self.inner_base + 1
                stride = compiled(self.env) - base
                self.env[axis] = self.inner_base
                return base, stride

        def evaluate(position):
            for var, value in zip(outer, position):
                self.env[var] = value
            return flat(access, inner)

        if compiled is None or not linear:
            return evaluate
        saved = {var: self.env.get(var) for var in outer}
        for var in outer:
            self.env[var] = 0
        origin, stride = flat(access, inner)
        steps = []
        for var in outer:
            self.env[var] = 1
            steps.append(flat(access, inner)[0] - origin)
            self.env[var] = 0
        for var, value in saved.items():
            if value is None:
                self.env.pop(var, None)
            else:
                self.env[var] = value
        return lambda position: (
            origin + sum(map(int.__mul__, steps, position)), stride)

    def _emit_segments(self, segments, to_sram):
        index, group, aligned = 0, 0, False
        while index < len(segments):
            rows, stride = self._regular_block(segments, index)
            glob, sram, count = segments[index]
            if rows > 1:
                # a tile of a wider tensor: exactly what the 2D transfer is for
                self.stage.dma_2d(glob, stride, sram, rows, count, to_sram)
                index += rows
                group = 0
                continue
            if index >= group:
                group = self._contiguous_group(segments, index)
                aligned = all(g % 4 == 0 and n % 4 == 0
                              for g, _, n in segments[index:group])
            if (aligned or group == index + 1) and glob % 4 == 0 and count % 4 == 0:
                if to_sram:
                    self.stage.dma_in(glob, sram, count)
                else:
                    self.stage.dma_out(glob, sram, count)
                index += 1
                continue
            if group == index + 1:           # single mis-aligned row: gather
                self._bounce_dma(glob, [(sram, count)], to_sram)
                index += 1
                continue
            self._bounce_dma(glob, [(s, n) for _, s, n in segments[index:group]],
                             to_sram)
            index = group

    @staticmethod
    def _contiguous_group(segments, index):
        """End of the run of segments that are adjacent in global memory."""
        end = index + 1
        while end < len(segments) and \
                segments[end][0] == segments[end - 1][0] + segments[end - 1][2]:
            end += 1
        return end

    @staticmethod
    def _regular_block(segments, index):
        """Length and global row stride of a 2D block starting at ``index``.

        A transfer addresses global memory in cells and packs SRAM rows, so a
        block needs base, row length and stride on whole 4-byte cells.
        """
        glob, sram, count = segments[index]
        if index + 1 >= len(segments) or glob % 4 or count % 4:
            return 1, 0
        stride = segments[index + 1][0] - glob
        if stride % 4 or stride < count:
            return 1, 0
        rows = 1
        while index + rows < len(segments):
            next_glob, next_sram, next_count = segments[index + rows]
            if next_count != count or next_glob != glob + rows * stride \
                    or next_sram != sram + rows * count * 2:
                break
            rows += 1
        return rows, stride

    def _bounce_dma(self, glob, pieces, to_sram):
        """Move a globally contiguous range whose SRAM image is scattered.

        A transfer moves whole 32-bit cells, so a row that begins mid-cell
        cannot be moved on its own; a vector copy has element granularity.
        The rows are therefore gathered into a contiguous scratch region and
        moved in aligned chunks.
        """
        self.flush()                    # the copies below take the PE output
        if len(self.scratch_slots) < 2:
            raise V09TirError("bounce DMA requires a bounded scratch region")
        scratch = self.scratch_slots[0]
        slot_nibbles = self.scratch_slots[1] - scratch
        if slot_nibbles <= 0 or any(
                slot != scratch + i * slot_nibbles
                for i, slot in enumerate(self.scratch_slots)):
            raise V09TirError("bounce DMA requires contiguous uniform scratch slots")
        # DMA runs between vector expressions, so it owns the whole scratch
        # region.  Bound the staging chunks by its actual allocated size.
        capacity = min(2 * self.BOUNCE_BYTES,
                       len(self.scratch_slots) * slot_nibbles // 2) // 4 * 4
        if capacity < 4:
            raise V09TirError("bounce DMA scratch cannot hold one cell")
        lead = glob % 4       # bytes sharing the first cell with someone else
        glob -= lead
        chunk, total = [], 0
        for sram, nbytes in pieces:
            while nbytes:
                part = min(nbytes, capacity - lead - total)
                chunk.append((sram, part))
                total += part
                sram += part * 2
                nbytes -= part
                if total + lead == capacity:
                    glob = self._bounce_chunk(glob, chunk, total, scratch,
                                              to_sram, lead)
                    chunk, total, lead = [], 0, 0
        if chunk:
            self._bounce_chunk(glob, chunk, total, scratch, to_sram, lead)

    BOUNCE_BYTES = 8192

    def _bounce_chunk(self, glob, pieces, total, scratch, to_sram, lead):
        span = total + lead
        if span > 2 * self.BOUNCE_BYTES:
            raise V09TirError(f"bounce chunk of {span} exceeds the scratch slot")
        if to_sram:
            self.stage.dma_in(glob, scratch, span)
        elif lead:
            # the bytes sharing the first cell belong to someone else, so read
            # them back before the cell is written
            self.stage.dma_in(glob, scratch, lead)
        if not to_sram and span % 4:
            # The final partial cell also shares bytes with another tensor
            # region (e.g. the next row's first element after an N=65 tail).
            tail = span // 4 * 4
            if tail or not lead:
                self.stage.dma_in(glob + tail, scratch + tail * 2, 4)
        offset = lead
        for sram, nbytes in pieces:
            if nbytes % 2 or offset % 2:
                raise V09TirError("bounce pieces must be fp16-aligned")
            src, dst = ((scratch + offset * 2, sram) if to_sram
                        else (sram, scratch + offset * 2))
            self.a.vlen(nbytes // 2)      # the gather copies move fp16 lanes
            self.stage.vector(SRC1, src)
            self.a.load(0, SRC1)
            self.a.v_copy()
            self.stage.vector(DST, dst)
            self.a.save(0)
            offset += nbytes
        if not to_sram:
            self.stage.dma_out(glob, scratch, span)
        return glob + span

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
        its symbolic stride to the parent row width.

        Every tensorized tile visits the same match_buffer node, so its shape
        and origin expressions are read from TIR once and evaluated per visit;
        the node is kept alive by the cache, which is what makes its address
        usable as the key.
        """
        plan = self._match_plans.get(match.handle.value)
        if plan is None:
            source, view = match.source, match.buffer
            # a tile lives in the last two axes; leading axes select the batch
            shape = [int(d) for d in source.buffer.shape]
            elem_offset = view.elem_offset
            plan = (match, source.buffer.data, view.data, source.buffer.name,
                    elem_offset if isinstance(elem_offset, tir.Var) else None,
                    [symbol if isinstance(symbol, tir.Var) else None
                     for symbol in view.strides],
                    shape, [(self._compile(r.min), r.min) for r in source.region])
            self._match_plans[match.handle.value] = plan
        _, parent, data, name, elem_offset, strides, shape, origins = plan
        stride = shape[-1]
        offset, scale = 0, 1
        for dim, (compiled, expr) in zip(reversed(shape), reversed(origins)):
            offset += (compiled(self.env) if compiled is not None
                       else self.ev(expr)) * scale
            scale *= dim
        # a view shares the parent's data pointer; its position is carried by
        # the symbolic elem_offset that access_ptr adds
        if parent in self.sram:
            self.sram[data] = self.sram[parent]
            self.scopes[data] = self.scopes.get(parent)
        elif parent in self.bases:
            self.bases[data] = self.bases[parent]
        else:
            raise V09TirError(f"match_buffer source is not a known buffer: {name}")
        # SRAM views carry their position in nibbles; global views stay in
        # elements, because ptr() applies the byte scaling itself
        unit = self.width(parent) if parent in self.sram else 1
        if elem_offset is not None:
            self.env[elem_offset] = offset * unit
        for symbol, value in zip(strides, (stride, 1)):
            if symbol is not None:
                self.env[symbol] = value

    def call(self, expr):
        if not (isinstance(expr, tir.Call)
                and expr.op.name == "tir.call_extern"):
            raise V09TirError(f"unhandled call {expr}")
        name = expr.args[0].value
        if name in ("npu_quantize_row", "npu_scaled_mac", "npu_store_scaled",
                    "npu_reset_quant_descriptors"):
            self._w8a8_call(name, expr.args[1:])
        elif name in ("npu_fill_zero_fp32", "npu_gemm_acc_int8"):
            raise V09TirError("FP32 matrix SRAM storage is unsupported; use the scaled W8A8 register contract")
        elif name == "npu_fill_zero":
            self.zeroed.add(self.ptr(expr.args[1]))
        elif name == "npu_gemm_acc":
            c, sc = self.ptr(expr.args[1]), self.ev(expr.args[2])
            a, sa = self.ptr(expr.args[3]), self.ev(expr.args[4])
            b, sb = self.ptr(expr.args[5]), self.ev(expr.args[6])
            self.emit_gemm(c, sc, a, sa, b, sb)
        else:
            raise V09TirError(f"unknown intrinsic {name}")

    def _w8a8_call(self, name, args):
        # These intrinsics use standard TIR element offsets, after
        # LowerMatchBuffer. Older matrix intrinsics use nibble-offset views.
        def ptr(access):
            data, offset = access.args[1:3]
            return self.sram[data] + self.ev(offset) * self.width(data)
        asm = self.a
        if name == "npu_quantize_row":
            from .npu_w8a8_intrin import quantize_row
            self.flush()
            if getattr(self, "scaled_pending", None) is not None:
                raise RuntimeError("quantization interrupts a scaled MAC chain")
            quantize_row(asm, *[ptr(a) for a in args[:5]], self.ev(args[5]))
        elif name == "npu_scaled_mac":
            c, a, stride, b, sa, sw, rows, first = args
            a_access, b_access = a, b
            c = c.args[1]  # local.accumulator is a register token, not SRAM
            a, b, sa, sw = (ptr(v) for v in (a, b, sa, sw))
            rows, stride = self.ev(rows), self.ev(stride)
            if self.ev(first):
                self.flush()
                if getattr(self, "scaled_pending", None) is not None:
                    raise RuntimeError("scaled accumulator overwritten before save")
                self.scaled_pending = (c, rows)
            elif getattr(self, "scaled_pending", None) != (c, rows):
                raise RuntimeError("scaled MAC has no matching accumulator")
            asm.ascale(sa)
            asm.wscale(sw)
            self.stage.region_dt(SRC1, a, stride, rows, 64, DT_INT8,
                main_base=self.sram[a_access.args[1]],
                main_rows=self.ev(a_access.args[2] + a_access.args[3]) // stride)
            asm.load(1, SRC1)
            self.stage.region_dt(SRC2, b, 64, 64, 64, DT_INT8,
                main_base=self.sram[b_access.args[1]],
                main_rows=self.ev(b_access.args[2] + b_access.args[3]) // 64)
            asm.load(1, SRC2)
            asm.m_mul(VECTOR, mac=not self.ev(first))
        elif name == "npu_store_scaled":
            c, dst, rows = args[0].args[1], ptr(args[1]), self.ev(args[2])
            if getattr(self, "scaled_pending", None) != (c, rows):
                raise RuntimeError("scaled save has no matching accumulator")
            self.stage.region_dt(DST, dst, 64, rows, 64, DT_FP16)
            asm.save(1)
            self.scaled_pending = None
        else:
            if getattr(self, "scaled_pending", None) is not None:
                raise RuntimeError("scaled accumulator was not saved")
            for operand in (SRC1, SRC2, DST):
                asm.shape_dt(operand, 1, 1, 1, DT_FP16)

    # ---- v09 emission ----

    def emit_gemm(self, c, sc, a, sa, b, sb):
        """FP16 operands, FP32 register MAC chain, one final FP16 store."""
        asm = self.a
        first = c in self.zeroed
        if first:
            self.zeroed.discard(c)
            self.flush()
            self.pending = (c, sc)
        elif self.pending is None or self.pending[0] != c:
            raise V09TirError("FP32 matrix chain interrupted: resuming from an FP16 store would change the reduction")
        self.stage.region(SRC1, a, sa, TILE, TILE)
        asm.load(1, SRC1)
        self.stage.region(SRC2, b, sb, TILE, TILE)
        asm.load(1, SRC2)
        asm.m_mul(VECTOR, mac=not first)

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
        self.dtypes[buffer.data] = str(buffer.dtype)

    def _space(self, buffer):
        return self.scopes.get(buffer.data, "global")

    def run(self, prim_func, buffer_bases):
        self.bases.update(buffer_bases)
        self.visit(prim_func.body)
        self.flush()
        if getattr(self, "scaled_pending", None) is not None:
            raise RuntimeError("kernel ended with an unsaved scaled accumulator")
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

    def region_dt(self, operand, nibble, stride, rows, cols, dtype,
                  main_base=None, main_rows=None):
        """``region`` with the descriptor's dtype bits set explicitly.

        Kept separate rather than adding a default argument to ``region`` so
        the FP16 path emits exactly the words it did before.
        """
        self.asm.addr(operand, nibble if main_base is None else main_base, 0)
        self.asm.shape_dt(operand, rows if main_rows is None else main_rows, stride, 0, dtype)
        self.asm.addr(operand, nibble, 1)
        self.asm.shape_dt(operand, rows, cols, 1, dtype)

    def strided(self, operand, nibble, stride, count):
        self.asm.addr(operand, nibble, 0)
        self.asm.shape(operand, count, stride, 0)
        self.asm.addr(operand, nibble, 1)
        self.asm.shape(operand, count, 1, 1)

    # -- DMA (global byte offsets <-> SRAM nibbles)

    # rows and cols are 16-bit instruction fields; anything larger is split
    MAX_CELLS = 0xFFFF

    def dma_in(self, global_byte, sram_nibble, nbytes):
        self._dma_1d(self.asm.gload, global_byte, sram_nibble, nbytes)

    def dma_out(self, global_byte, sram_nibble, nbytes):
        self._dma_1d(self.asm.gstore, global_byte, sram_nibble, nbytes)

    def _dma_1d(self, move, global_byte, sram_nibble, nbytes):
        while nbytes > 0:
            piece = min(nbytes, self.MAX_CELLS * 4)
            cells, nib = self._cells(global_byte, piece, sram_nibble)
            move(global_byte // 4, cells, nib, 1, cells)
            global_byte += piece
            sram_nibble += piece * 2
            nbytes -= piece

    def dma_2d(self, global_byte, global_stride, sram_nibble, rows, nbytes,
               to_sram):
        """Move ``rows`` rows of ``nbytes`` bytes spaced ``global_stride``
        bytes apart in global memory to or from a packed SRAM block."""
        move = self.asm.gload if to_sram else self.asm.gstore
        if nbytes // 4 > self.MAX_CELLS:       # a row alone overflows the field
            for _ in range(rows):
                self._dma_1d(move, global_byte, sram_nibble, nbytes)
                global_byte += global_stride
                sram_nibble += nbytes * 2
            return
        while rows > 0:
            batch = min(rows, self.MAX_CELLS)
            cells, nib = self._cells(global_byte, nbytes, sram_nibble)
            move(global_byte // 4, global_stride // 4, nib, batch, cells)
            global_byte += batch * global_stride
            sram_nibble += batch * nbytes * 2
            rows -= batch

    @staticmethod
    def _cells(global_byte, nbytes, sram_nibble):
        # allocations are cell-rounded, so a short tail only reaches into the
        # tensor's own padding; an unaligned start would shift the SRAM image
        # and is rejected instead
        if global_byte % 4:
            raise V09TirError("DMA row must start on a 32-bit cell")
        if sram_nibble % 8:
            raise V09TirError("DMA SRAM address must be 8-nibble aligned")
        return (nbytes + 3) // 4, sram_nibble
