"""Large-stride GEMM lowering for the extended ver.08 source C-model."""
from __future__ import annotations

import numpy as np

from .isa_0818 import DST, SRC1, SRC2, VECTOR, Asm
from .source_runtime_0818 import run


class PackedRhsGemm:
    """One-row GEMM whose logical RHS width exceeds the 16-bit stride field.

    RHS columns are stored as consecutive full-K panels.  Each panel has a legal
    local stride while all panels and their output execute in one program.
    """

    def __init__(self, inner, columns, panel=64, tile=64):
        self.inner = int(inner)
        self.columns = int(columns)
        self.panel = int(panel)
        self.tile = int(tile)
        if self.inner < 1 or self.columns < 1:
            raise ValueError("GEMM dimensions must be positive")
        if not 1 <= self.panel <= 64 or not 1 <= self.tile <= 64:
            raise ValueError("panel and tile must be in 1..64")
        self.a_offset = 0
        self.b_offset = self.inner
        self.c_offset = self.b_offset + self.inner * self.columns
        self.gbuffer_entries = self.c_offset + self.columns
        self.asm = self._compile()

    def _compile(self):
        asm = Asm()
        panel_index = 0
        for column in range(0, self.columns, self.panel):
            width = min(self.panel, self.columns - column)
            rhs = self.b_offset + panel_index * self.inner * self.panel
            for index, k0 in enumerate(range(0, self.inner, self.tile)):
                kt = min(self.tile, self.inner - k0)
                asm.matrix_region(
                    SRC1, self.a_offset, 1, self.inner,
                    self.a_offset + k0, 1, kt).load(1, SRC1)
                asm.matrix_region(
                    SRC2, rhs, self.inner, width,
                    rhs + k0 * width, kt, width).load(1, SRC2)
                asm.m_mul(VECTOR, mac=index != 0)
            asm.matrix_region(
                DST, self.c_offset + column, 1, width,
                self.c_offset + column, 1, width).save(1)
            panel_index += 1
        return asm.finish()

    def __len__(self):
        return len(self.asm.words)

    def run(self, lhs, packed_rhs):
        lhs = np.asarray(lhs, dtype=np.float16).reshape(-1)
        packed_rhs = np.asarray(packed_rhs, dtype=np.float16).reshape(-1)
        if lhs.size != self.inner:
            raise ValueError(f"lhs has {lhs.size} entries, expected {self.inner}")
        expected = self.inner * self.columns
        if packed_rhs.size != expected:
            raise ValueError(f"packed RHS has {packed_rhs.size} entries, expected {expected}")
        gbuffer = np.empty(self.gbuffer_entries, dtype=np.float16)
        gbuffer[self.a_offset:self.b_offset] = lhs
        gbuffer[self.b_offset:self.c_offset] = packed_rhs
        gbuffer[self.c_offset:] = 0
        output = run(
            self.asm, gbuffer, output_dtype=np.float16,
            output_range=(self.c_offset, self.columns))
        return output.reshape(1, self.columns)
