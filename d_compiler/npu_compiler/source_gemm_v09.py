"""Large-stride GEMM lowering for the v09 C-model (SRAM-staged panels).

Mirrors :class:`source_gemm_0818.PackedRhsGemm` — same packed-RHS buffer
layout and the same per-panel K-tile accumulation order, so results are
bit-exact with the ver.08 lowering — with GLOAD/GSTORE staging: the LHS row
is staged once, each full-K column panel is staged before use, and each
output segment is stored back after its save.
"""
from __future__ import annotations

import numpy as np

from .backend_v09 import V09Asm
from .isa_0818 import DST, SRC1, SRC2, VECTOR

DMA_MAX_CELLS = 0xFFFF


def _dma_1d(emit, cell0, cells, nibble):
    for base in range(0, cells, DMA_MAX_CELLS):
        chunk = min(DMA_MAX_CELLS, cells - base)
        emit(cell0 + base, chunk, nibble + base * 8, 1, chunk)


class V09PackedRhsGemm:
    """One-row GEMM over consecutive full-K RHS panels, staged through SRAM."""

    def __init__(self, inner, columns, panel=64, tile=64):
        self.inner = int(inner)
        self.columns = int(columns)
        self.panel = int(panel)
        self.tile = int(tile)
        if self.inner < 1 or self.columns < 1:
            raise ValueError("GEMM dimensions must be positive")
        if not 1 <= self.panel <= 64 or not 1 <= self.tile <= 64:
            raise ValueError("panel and tile must be in 1..64")
        if self.inner % 2:
            raise ValueError("v09 packed GEMM requires an even inner dimension")
        self.a_offset = 0
        self.b_offset = self.inner
        self.c_offset = self.b_offset + self.inner * self.columns
        self.gbuffer_entries = self.c_offset + self.columns
        self.asm = self._compile()

    def _compile(self):
        asm = V09Asm()
        nib_a = 0
        nib_b = (self.inner * 4 + 7) // 8 * 8
        nib_c = nib_b + self.inner * self.panel * 4
        _dma_1d(asm.gload, self.a_offset // 2, self.inner // 2, nib_a)
        panel_index = 0
        for column in range(0, self.columns, self.panel):
            width = min(self.panel, self.columns - column)
            if width % 2:
                raise ValueError("v09 packed GEMM requires even panel widths")
            rhs = self.b_offset + panel_index * self.inner * self.panel
            _dma_1d(asm.gload, rhs // 2, self.inner * width // 2, nib_b)
            for index, k0 in enumerate(range(0, self.inner, self.tile)):
                kt = min(self.tile, self.inner - k0)
                asm.addr(SRC1, nib_a, 0).shape(SRC1, 1, self.inner, 0)
                asm.addr(SRC1, nib_a + k0 * 4, 1).shape(SRC1, 1, kt, 1)
                asm.load(1, SRC1)
                asm.addr(SRC2, nib_b, 0).shape(SRC2, self.inner, width, 0)
                asm.addr(SRC2, nib_b + k0 * width * 4, 1).shape(SRC2, kt, width, 1)
                asm.load(1, SRC2)
                asm.m_mul(VECTOR, mac=index != 0)
            asm.addr(DST, nib_c, 0).shape(DST, 1, width, 0)
            asm.addr(DST, nib_c, 1).shape(DST, 1, width, 1)
            asm.save(1)
            _dma_1d(asm.gstore, (self.c_offset + column) // 2, width // 2, nib_c)
            panel_index += 1
        asm.halt()
        return asm

    def __len__(self):
        return len(self.asm.words)

    def run(self, lhs, packed_rhs):
        from .v09_runtime import run as run_v09
        lhs = np.asarray(lhs, dtype=np.float16).reshape(-1)
        packed_rhs = np.asarray(packed_rhs, dtype=np.float16).reshape(-1)
        if lhs.size != self.inner:
            raise ValueError(f"lhs has {lhs.size} entries, expected {self.inner}")
        expected = self.inner * self.columns
        if packed_rhs.size != expected:
            raise ValueError(
                f"packed RHS has {packed_rhs.size} entries, expected {expected}")
        gbuffer = np.empty(self.gbuffer_entries, dtype="<f2")
        gbuffer[self.a_offset:self.b_offset] = lhs
        gbuffer[self.b_offset:self.c_offset] = packed_rhs
        gbuffer[self.c_offset:] = 0
        if gbuffer.size % 2:
            gbuffer = np.concatenate([gbuffer, np.zeros(1, dtype="<f2")])
        images, _ = run_v09(self.asm.words, gbuffer.view("<u4"))
        full = np.ascontiguousarray(images[-1]).view("<f2")
        return full[self.c_offset:self.c_offset + self.columns].reshape(
            1, self.columns).copy()
