"""Conservative SRAM write bounds for the v09 straight-line instruction stream.

Tracks descriptor halves, loaded register lengths and *result* dimensions.
SAVE does not use the current vlen. Matrix/strided bounds include untouched gaps;
unknown instructions poison the analysis rather than invent a write footprint.
All bounds are in nibbles, including packed INT4 writes.
"""


class WriteEffects:
    def __init__(self):
        self.address = [[0, 0] for _ in range(3)]  # main, partial
        self.rows = [[0, 0] for _ in range(3)]
        self.cols = [[0, 0] for _ in range(3)]
        self.dtype = [0, 0, 0]
        self.vlen = 0
        self.inputs = [0, 0]
        self.length = self.output_rows = self.output_cols = 0
        self.unknown = False

    def step(self, word):
        """Return () for no SRAM write, (start, size), or None if unproved."""
        op = word & 255
        if self.unknown:
            return None
        operand = (word >> 30) & 3
        value = (word >> 8) & 65535
        if op == 0x80:
            if operand < 3:
                partial, high = (word >> 28) & 1, (word >> 29) & 1
                old = self.address[operand][partial]
                self.address[operand][partial] = ((old & 65535) | (value << 16)
                    if high else (old & 0xffff0000) | value)
        elif op == 0x82:
            self.vlen = value
        elif op in (0x88, 0x89):
            if operand < 3:
                shape = self.rows if op == 0x88 else self.cols
                shape[operand][(word >> 29) & 1] = value
                self.dtype[operand] = (word >> 25) & 3
        elif op == 0x90:
            source = (word >> 30) & 1
            self.inputs[source] = (self.rows[source][1] *
                (((word >> 16) & 255) if word & (1 << 29) else self.cols[source][1])
                if word & (1 << 31) else self.vlen)
        elif op == 0x98:
            start = self.address[2][1]
            if not (word & (1 << 31)):
                return start, self.length * (4, 8, 2, 1)[self.dtype[2]]
            if not self.length:
                return ()
            if word & (1 << 29):
                rows = self.output_rows or self.rows[0][1]
                cols = (word >> 16) & 255
                start += ((word >> 8) & 255) * 4
                stride = self.cols[2][0]
            else:
                rows = self.output_rows or self.rows[2][1]
                cols = self.output_cols or self.cols[2][1]
                stride = self.cols[2][0] or cols
            return (start, ((rows - 1) * stride + cols) * 4) if rows and cols else ()
        elif op == 0x1a:
            return self.address[2][1], self.vlen * (4, 8, 2, 1)[self.dtype[2]]
        elif op in (0x14, 0x19):
            self.length = self.output_rows = self.output_cols = 1
        elif op == 0x18:
            self.length, self.output_rows, self.output_cols = self.inputs[0], 1, self.inputs[0]
        elif op in (0x15, 0x1b) or op in (1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 22, 23):
            self.length, self.output_rows, self.output_cols = self.vlen, 1, self.vlen
        elif 0x40 <= op <= 0x43:
            self.output_rows = self.rows[0][1]
            self.output_cols = self.cols[1][1] if op == 0x42 and operand == 2 else self.cols[0][1]
            self.length = (self.output_rows * self.output_cols
                           if op == 0x42 and operand == 2 else self.inputs[0])
        elif op not in (0, 0x8a, 0x8b, 0xa0, 0xa8, 0xf0, 0xff):
            self.unknown = True
            return None
        return ()
