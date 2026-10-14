"""v09 assembler independent of graph backends."""
from .isa_compute import Asm
from .isa_v09 import (V09EncodeError, enc_gload, enc_gstore, enc_halt,
    enc_snapshot, enc_mrows, enc_mcols, enc_wscale, enc_ascale, enc_vquant, enc_vdequant)
from .v09_contract import MAX_VLEN


class V09Asm(Asm):
    format_version = "v09"
    execution_target = "v09"

    def vlen(self, n):
        # the field is 16-bit and the base encoder masks silently; a longer
        # vector would quietly process only the low bits' worth of lanes
        if not 0 < int(n) <= MAX_VLEN:
            raise V09EncodeError(f"vlen={n} does not fit in 16 bits")
        return super().vlen(n)

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

    def snapshot(self):
        """Append the whole global image to the snapshot file mid-run."""
        return self._emit(enc_snapshot())

    def shape_dt(self, operand, rows, cols, partial, dtype):
        self._emit(enc_mrows(operand, rows, partial, dtype))
        return self._emit(enc_mcols(operand, cols, partial, dtype))

    def wscale(self, nibble):
        for word in enc_wscale(nibble):
            self._emit(word)
        return self

    def ascale(self, nibble):
        for word in enc_ascale(nibble):
            self._emit(word)
        return self

    def vquant(self):
        return self._emit(enc_vquant())

    def vdequant(self):
        return self._emit(enc_vdequant())

