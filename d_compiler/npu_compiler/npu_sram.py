"""Kernel-local SRAM reuse using TVM USMP conflict analysis and placement.

The executable TIR is not rewritten. LowerMatchBuffer/LowerOpaqueBlock run
on an analysis copy, then USMP byte offsets are mapped back by buffer data
Var identity. Constants and expression scratch stay outside this pool.
"""
from dataclasses import dataclass
from math import prod

import tvm
from tvm import tir
from tvm.ir.memory_pools import WorkspacePoolInfo
from tvm.tir.usmp.analysis import extract_buffer_info
from tvm.tir.usmp.utils import BufferInfo, CANDIDATE_MEMORY_POOL_ATTR

from .npu_errors import LinkError
from .device import npu_target
from .v09_contract import CONTRACT


def align(value, unit=8):
    """Align nibble offsets to a 32-bit DMA cell."""
    return (value + unit - 1) // unit * unit


def size_nibbles(buffer):
    dtype = str(buffer.dtype)
    if dtype not in CONTRACT["storage"]:
        raise LinkError(f"no SRAM storage contract for {dtype}")
    if buffer.strides or int(buffer.elem_offset) != 0:
        raise LinkError(f"SRAM allocation requires a compact buffer: {buffer.name}")
    dims = [int(d) for d in buffer.shape]
    if any(d <= 0 for d in dims):
        raise LinkError(f"SRAM allocation requires positive static extents: {buffer.name}")
    return align(prod(dims) * CONTRACT["storage"][dtype][0])


@dataclass(frozen=True)
class SramLayout:
    buffers: tuple
    sizes: tuple                  # allocated/padded nibbles
    offsets: tuple                # relative nibbles
    conflicts: tuple              # original allocation indices
    pinned: tuple                 # retained for pending matrix state / opaque calls
    algorithm: str
    reason: str
    baseline_nibbles: int
    pressure_bytes: int | None = None

    def __iter__(self):
        return iter(self.buffers)

    def __len__(self):
        return len(self.buffers)

    @property
    def peak_nibbles(self):
        return max((a + b for a, b in zip(self.offsets, self.sizes)), default=0)

    def validate(self):
        count = len(self.buffers)
        if not (len(self.offsets) == len(self.sizes) == len(self.conflicts) == count):
            raise LinkError("incomplete SRAM placement")
        for i in range(count):
            if self.offsets[i] < 0 or self.offsets[i] % 8 or self.sizes[i] <= 0:
                raise LinkError(f"invalid SRAM placement for buffer {i}")
            if self.sizes[i] != size_nibbles(self.buffers[i]):
                raise LinkError(f"invalid SRAM storage extent for buffer {i}")
            for j in range(i):
                overlap = (self.offsets[i] < self.offsets[j] + self.sizes[j]
                           and self.offsets[j] < self.offsets[i] + self.sizes[i])
                if overlap and (j in self.conflicts[i] or i in self.conflicts[j]
                                or i in self.pinned or j in self.pinned):
                    raise LinkError(f"SRAM lifetime conflict: buffers {j} and {i}")
        if self.peak_nibbles > self.baseline_nibbles:
            raise LinkError("SRAM reuse exceeds the bump-allocation baseline")

    def place(self, cursor=0, capacity=None):
        self.validate()
        if cursor < 0 or (capacity is not None and capacity < 0):
            raise LinkError("SRAM pool bounds must be nonnegative")
        base = align(cursor)
        top = base + self.peak_nibbles
        if capacity is not None and top > capacity:
            raise LinkError(f"kernel exceeds SRAM capacity: {top / 2**21:.2f} MiB "
                            f"of {capacity / 2**21:.2f} MiB")
        return dict(zip(self.buffers, (base + o for o in self.offsets))), top

    def describe(self, cursor=0):
        base = align(cursor)
        return dict(algorithm=self.algorithm, reason=self.reason,
                    pool_base_nibbles=base, peak_nibbles=base+self.peak_nibbles,
                    baseline_peak_nibbles=base+self.baseline_nibbles,
                    saved_bytes=(self.baseline_nibbles-self.peak_nibbles)//2,
                    usmp_pressure_bytes=self.pressure_bytes,
                    buffers=[dict(id=i, name=b.name, shape=[int(d) for d in b.shape],
                                  dtype=str(b.dtype), offset_nibbles=self.offsets[i],
                                  address_nibbles=base+self.offsets[i],
                                  size_nibbles=self.sizes[i], conflicts=list(self.conflicts[i]),
                                  pinned=i in self.pinned)
                             for i, b in enumerate(self.buffers)])


def plan_sram(prim, buffers, reuse=True):
    """Analyze once per scheduled candidate; return a relocatable layout.

    USMP is conservative across repeated loop bodies. We additionally pin
    matrix destinations: their final store may be deferred past the TIR
    call by the walker's accumulator cache. Unknown external calls pin every
    local pointer they receive. No reuse relies on block or model names.
    """
    buffers = tuple(buffers)
    sizes = tuple(size_nibbles(b) for b in buffers)
    offsets, end = [], 0
    for size in sizes:
        offsets.append(end)
        end += size
    baseline = SramLayout(buffers, sizes, tuple(offsets), tuple(() for _ in buffers),
                          (), "bump", "reuse disabled" if not reuse else "no smaller layout", end)
    if not reuse or not buffers:
        return baseline
    by_data = {b.data: i for i, b in enumerate(buffers)}
    if len(by_data) != len(buffers):
        raise LinkError("duplicate SRAM allocation data Vars")
    pool = WorkspacePoolInfo("npu_sram", [npu_target()])
    analysis_mod = tvm.transform.Sequential([
        tir.transform.LowerMatchBuffer(), tir.transform.LowerOpaqueBlock(),
    ])(tvm.IRModule({"main": prim}))
    pinned = set()
    nonserial = []

    def visit(node):
        if isinstance(node, tir.For) and node.kind != tir.ForKind.SERIAL:
            nonserial.append(node)
        if not (isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op)
                and node.op.name == "tir.call_extern"):
            return
        name = str(node.args[0].value)
        if name in ("npu_gemm_acc", "npu_gemm_acc_int8",
                    "npu_fill_zero", "npu_fill_zero_fp32"):
            args = [node.args[1]]
        elif name in (
                "npu_quantize_row", "npu_scaled_mac", "npu_store_scaled",
                "npu_reset_quant_descriptors"):
            # These versioned intrinsics expose every SRAM access through
            # access_ptr. Scaled matrix state lives in local.accumulator and
            # store_scaled saves it immediately. USMP can therefore use the
            # pointer lifetimes, including disjoint quantization/store phases.
            args = []
        else:
            args = node.args[1:]
        for arg in args:
            tir.stmt_functor.post_order_visit(arg, lambda v: pinned.add(by_data[v])
                if isinstance(v, tir.Var) and v in by_data else None)

    tir.stmt_functor.post_order_visit(analysis_mod["main"].body, visit)
    if nonserial:
        return SramLayout(**{**baseline.__dict__, "reason": "USMP requires serial loops"})

    def annotate(node):
        if isinstance(node, tir.Allocate):
            annotations = dict(node.annotations)
            annotations[CANDIDATE_MEMORY_POOL_ATTR] = [pool]
            return tir.Allocate(node.buffer_var, node.dtype, node.extents,
                                node.condition, node.body, annotations)
        return None

    fn = analysis_mod["main"]
    analysis_mod["main"] = fn.with_body(tir.stmt_functor.ir_transform(
        fn.body, None, annotate, ["tir.Allocate"]))
    result = extract_buffer_info(analysis_mod["main"], analysis_mod)
    raw = {}
    for info, alloc in result.buffer_info_stmts.items():
        if not isinstance(alloc, tir.Allocate) or alloc.buffer_var not in by_data:
            raise LinkError("USMP returned an unbound SRAM allocation")
        raw[by_data[alloc.buffer_var]] = info
    if len(raw) != len(buffers):
        raise LinkError("USMP did not account for every SRAM allocation")
    index = {info: i for i, info in raw.items()}
    conflicts = [set(index[c] for c in raw[i].conflicts) - {i} for i in range(len(buffers))]
    for i in pinned:
        for j in range(len(buffers)):
            if i != j:
                conflicts[i].add(j)
                conflicts[j].add(i)
    infos = []
    for i in range(len(buffers)):
        if align(int(raw[i].size_bytes) * 2) != sizes[i]:
            raise LinkError(f"USMP/codegen storage size mismatch: {buffers[i].name}")
        # Reserve the entire DMA cell, with the target's four-byte alignment.
        # Stable unique identifiers also make USMP's equal-size tie-break deterministic.
        infos.append(BufferInfo(f"buffer_{i:06d}", sizes[i]//2, [pool], 4))
    for i, info in enumerate(infos):
        info.set_conflicts([infos[j] for j in sorted(conflicts[i])])
    common = dict(buffers=buffers, sizes=sizes,
                  conflicts=tuple(tuple(sorted(c)) for c in conflicts),
                  pinned=tuple(sorted(pinned)), baseline_nibbles=end,
                  pressure_bytes=int(result.memory_pressure))
    best = SramLayout(**common, offsets=tuple(offsets), algorithm="bump",
                      reason="USMP did not reduce the footprint")
    for algorithm in ("greedy_by_size", "greedy_by_conflicts"):
        allocation = tvm.get_global_func("tir.usmp.algo." + algorithm)(infos, result.memory_pressure)
        candidate = SramLayout(**common,
            offsets=tuple(int(allocation[b].byte_offset)*2 for b in infos),
            algorithm="usmp."+algorithm, reason="validated lifetime reuse")
        if candidate.peak_nibbles < best.peak_nibbles:
            candidate.validate()
            best = candidate
    best.validate()
    return best
