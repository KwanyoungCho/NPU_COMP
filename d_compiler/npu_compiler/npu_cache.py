"""Versioned KV append ABI and relocatable DMA programs for static decode.

Cache effects are declared on the Relax function, outside its pure tensor
calculation. The program's tail updates only returned token slices. Cell RMW
preserves neighbours for odd capacity/position; the full cache is never copied.
"""
import json
from dataclasses import dataclass
from math import prod

import numpy as np
from tvm import relax

from .v09_asm import V09Asm
from .isa_compute import SRC1, DST, IMM
from .isa_v09 import DT_FP16
from .tir_codegen_v09 import SramEmitter
from .v09_stats import packets, analyze

ATTRIBUTE = "npu.cache_append"


def declare(function, entries, mask="mask"):
    """entries: (cache parameter name, returned tuple index, time axis)."""
    contract = dict(version=1, mask=mask, entries=[dict(parameter=p, output=o, axis=a)
                                                for p, o, a in entries])
    return function.with_attr(ATTRIBUTE, json.dumps(contract, sort_keys=True))


@dataclass(frozen=True)
class Slice:
    parameter: str
    source_bytes: int
    destination_bytes: int
    rows: int
    capacity: int
    width: int


class AppendPlan:
    def __init__(self, func, plan, sram_bytes=8 * 1024 * 1024):
        contract = json.loads(str(func.attrs[ATTRIBUTE]))
        if contract["version"] != 1:
            raise ValueError("unsupported cache append ABI version")
        self.mask = contract["mask"]
        self.sram_bytes = sram_bytes
        self.slices = []
        params = {p.name_hint: p for p in func.params}
        result = func.body.body
        members = ([field.name_hint for field in result.fields]
                   if isinstance(result, relax.Tuple) else plan.tuples.get(result.name_hint))
        if members is None or not hasattr(result.struct_info, "fields"):
            raise ValueError("cache append requires a tuple output")
        for entry in contract["entries"]:
            name, output, axis = entry["parameter"], entry["output"], entry["axis"]
            info = params[name].struct_info
            shape = tuple(map(int, info.shape))
            if not 0 <= axis < len(shape) or min(shape) <= 0:
                raise ValueError("invalid static cache shape/axis")
            out_info = result.struct_info.fields[output]
            expected = shape[:axis] + (1,) + shape[axis + 1:]
            if str(info.dtype) != "float16" or str(out_info.dtype) != "float16" or tuple(map(int, out_info.shape)) != expected:
                raise ValueError("cache output shape/dtype does not match an append slice")
            source = plan.address[members[output]] * 2
            destination = plan.address[name] * 2
            size, out_size = prod(shape) * 2, prod(expected) * 2
            if source < destination + size and destination < source + out_size:
                raise ValueError("cache append source aliases its destination")
            self.slices.append(Slice(name, source, destination, prod(shape[:axis]),
                                     shape[axis], prod(shape[axis + 1:])))
        if not self.slices or len({s.capacity for s in self.slices}) != 1:
            raise ValueError("cache capacities must agree")
        if len({s.parameter for s in self.slices}) != len(self.slices):
            raise ValueError("duplicate cache append destination")
        self.capacity = self.slices[0].capacity
        self.workspace_bytes = 0
        mask_info = params[self.mask].struct_info
        if tuple(map(int, mask_info.shape)) != (1, 1, self.capacity + 1) or str(mask_info.dtype) != "float16":
            raise ValueError("cache valid-length mask must be FP16 [1,1,capacity+1]")
        self.templates = [self._template(parity) for parity in (0, 1)]
        self.logical_update_bytes = sum(s.rows * s.width * 2 for s in self.slices)

    def position(self, mask):
        mask = np.asarray(mask)
        if mask.shape != (1, 1, self.capacity + 1):
            raise ValueError("cache mask shape mismatch")
        values = mask.reshape(-1)
        valid = values[:-1] == 0
        length = int(valid.sum())
        if (not np.all(valid[:length]) or np.any(valid[length:]) or values[-1] != 0
                or not np.all(values[length:-1] == -65504)):
            raise ValueError("cache mask must describe one valid prefix and the current token")
        if length >= self.capacity:
            raise ValueError("cache is full")
        return length

    def words(self, position):
        if not isinstance(position, (int, np.integer)) or not 0 <= position < self.capacity:
            raise ValueError("cache append position outside capacity")
        words, fixups = self.templates[position % 2]
        words = list(words)
        for index, coefficient in fixups:
            words[index] += (position // 2) * coefficient
        return words

    def _template(self, position):
        asm = V09Asm()
        stage = SramEmitter(asm)
        fixups = []

        def cache_dma(slice_, global_byte, nibble, rows, row_bytes, load, stride=None):
            begin = len(asm.words)
            stage.dma_2d(global_byte, stride or slice_.capacity * slice_.width * 2,
                         nibble, rows, row_bytes, load)
            for offset, _, _ in packets(asm.words[begin:]):
                fixups.append((begin + offset + 1, slice_.width))

        for s in self.slices:
            row_bytes = s.width * 2
            if row_bytes % 4 == 0:
                count = max(1, min(65535, self.sram_bytes // row_bytes))
                if row_bytes > self.sram_bytes:
                    raise ValueError("cache token row exceeds SRAM capacity")
                for base in range(0, s.rows, count):
                    rows = min(count, s.rows - base)
                    self.workspace_bytes = max(self.workspace_bytes, rows * row_bytes)
                    stage.dma_in(s.source_bytes + base * row_bytes, 0, rows * row_bytes)
                    cache_dma(s, s.destination_bytes + (base * s.capacity + position) * row_bytes,
                              0, rows, row_bytes, False)
            elif s.width == 1:
                # Group alternating rows when C is odd: every DMA stride is
                # then cell-aligned, including C=1 where two groups share cells.
                step = 2 if s.capacity % 2 else 1
                # This RMW uses a matrix transfer. Bound its PARTIAL rows by
                # the physical 64-row tile, even though the field has 16 bits.
                count = min(64, (self.sram_bytes - 16) // 8)
                if count < 1:
                    raise ValueError("cache append scratch does not fit")
                for group in range(step):
                    total = (s.rows - group + step - 1) // step
                    for base in range(0, total, count):
                        rows = min(count, total - base)
                        first = group + base * step
                        source = s.source_bytes + first * 2
                        aligned = source // 4 * 4
                        src_bytes = ((source % 4 + ((rows-1) * step + 1) * 2 + 3) // 4) * 4
                        stage.dma_in(aligned, 0, src_bytes)
                        tmp = src_bytes * 2
                        self.workspace_bytes = max(self.workspace_bytes, src_bytes + rows * 4)
                        address = s.destination_bytes + (first * s.capacity + position) * 2
                        cache_dma(s, address // 4 * 4, tmp, rows, 4, True,
                                  stride=step * s.capacity * 2)
                        stage.region_dt(SRC1, (source % 4) * 2, step, rows, 1, DT_FP16)
                        asm.load(1, SRC1)
                        asm.m_mul(IMM, 1)
                        stage.region_dt(DST, tmp, 2, rows, 1, DT_FP16)
                        asm.save(1, strided=1, ncols=1, start=(address % 4) // 2)
                        cache_dma(s, address // 4 * 4, tmp, rows, 4, False,
                                  stride=step * s.capacity * 2)
            else:
                # Odd-width value rows need both endpoint cells preserved.
                # Their work remains proportional to the appended row width.
                for row in range(s.rows):
                    source = s.source_bytes + row * row_bytes
                    target = s.destination_bytes + (row * s.capacity + position) * row_bytes
                    src_bytes = ((source % 4 + row_bytes + 3) // 4) * 4
                    dst_bytes = ((target % 4 + row_bytes + 3) // 4) * 4
                    if src_bytes + dst_bytes > self.sram_bytes:
                        raise ValueError("cache append scratch does not fit")
                    self.workspace_bytes = max(self.workspace_bytes, src_bytes + dst_bytes)
                    stage.dma_in(source // 4 * 4, 0, src_bytes)
                    tmp = src_bytes * 2
                    cache_dma(s, target // 4 * 4, tmp, 1, dst_bytes, True, stride=dst_bytes)
                    for begin in range(0, s.width, 65535):
                        size = min(65535, s.width - begin)
                        asm.vlen(size)
                        asm.addr(SRC1, (source % 4) * 2 + begin * 4)
                        asm.shape_dt(SRC1, 1, size, 1, DT_FP16)
                        asm.load(0, SRC1)
                        asm.v_copy()
                        asm.addr(DST, tmp + (target % 4) * 2 + begin * 4)
                        asm.shape_dt(DST, 1, size, 1, DT_FP16)
                        asm.save(0)
                    cache_dma(s, target // 4 * 4, tmp, 1, dst_bytes, False, stride=dst_bytes)
        return asm.words, fixups

    def describe(self):
        return dict(version=1, capacity=self.capacity,
                    sram_peak_bytes=self.workspace_bytes,
                    slices=[s.__dict__ for s in self.slices],
                    logical_update_bytes=self.logical_update_bytes,
                    templates=[analyze(words) for words, _ in self.templates],
                    fixups=[len(f) for _, f in self.templates])
