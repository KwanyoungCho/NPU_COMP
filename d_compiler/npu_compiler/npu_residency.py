"""Conservative SRAM value forwarding between linked kernel calls.

Global stores remain visible (including snapshots). A cached store is a value
version, invalidated by any overlapping global write or SRAM DMA write. Compute
stores invalidate their proven SRAM bounds. At the next kernel entry,
before any register LOAD/compute, FP16 global reads may use a disjoint SRAM copy.
No capacity reservation or reduction in a matrix panel is needed.
"""
from .v09_asm import V09Asm
from .isa_compute import SRC1, DST
from .isa_v09 import OP_GLOAD, OP_GSTORE, decode_dma, DT_FP16
from .v09_stats import packets
from .v09_effects import WriteEffects


def _overlap(a, size, b, other):
    return a < b + other and b < a + size


def _rows(dma):
    for row in range(dma["rows"]):
        yield (dma["g_addr"] * 4 + row * dma["g_stride"] * 4,
               dma["sram_addr"] + row * dma["cols"] * 8,
               dma["cols"] * 4)


def forward(words, records):
    starts = {r["span"][0]: r for r in records}
    boundaries = {v for r in records for v in r["span"]}
    cache, descriptors, shape_dtype = [], {}, {}
    initial = V09Asm()
    initial.words.append(0x82)  # hardware reset state; never used for compute
    for operand in range(3):
        for partial in (0, 1):
            initial.addr(operand, 0, partial)
            initial.shape_dt(operand, 0, 0, partial, DT_FP16)
    for word in initial.words:
        op = word & 255
        if op == 0x80:
            descriptors[(op, (word >> 28) & 15)] = word
        elif op == 0x82:
            descriptors[(op,)] = word
        else:
            operand = (word >> 30) & 3
            descriptors[(op, operand, (word >> 29) & 1)] = word
            shape_dtype[operand] = word
    out, remap, events = [], {}, []
    version = 0
    entry = False
    current = None
    effects = WriteEffects()
    peak_intervals = 0
    precise_writes = unknown_writes = 0

    def invalidate(axis, addr, size):
        # Retain disjoint DMA cells on both sides of a write. Rounding outwards
        # also handles FP16/INT8/INT4 writes into one half of a cached cell.
        unit = 1 if axis == 0 else 2
        kept = []
        for c in cache:
            start = c[axis]
            if not _overlap(addr, size, start, c[2] * unit):
                kept.append(c)
                continue
            left = max(0, (addr - start) // (4 * unit) * 4)
            right = min(c[2], (addr + size - start + 4 * unit - 1) // (4 * unit) * 4)
            if left:
                kept.append((c[0], c[1], left, *c[3:]))
            if right < c[2]:
                kept.append((c[0] + right, c[1] + right * 2, c[2] - right, *c[3:]))
        cache[:] = kept

    def typed_fp16(global_address, size):
        matches = [a["dtype"] for a in current.get("arguments", [])
                   if a["address_bytes"] <= global_address
                   and global_address + size <= a["address_bytes"] + a["size_bytes"]]
        return bool(matches) and all(t == "float16" for t in matches)

    def copy(source, destination, size):
        asm = V09Asm()
        for first in range(0, size // 2, 65535):
            count = min(65535, size // 2 - first)
            asm.vlen(count)
            asm.addr(SRC1, source + first * 4)
            asm.shape_dt(SRC1, 1, count, 1, DT_FP16)
            asm.load(0, SRC1)
            asm.v_copy()
            asm.addr(DST, destination + first * 4)
            asm.shape_dt(DST, 1, count, 1, DT_FP16)
            asm.save(0)
        # Restore descriptor state before the original instructions continue.
        # A shape write also writes dtype, so restore that field last.
        for key, word in descriptors.items():
            if key[0] == 0x82 or (key[0] == 0x80 and ((key[1] >> 2) & 3) in (SRC1, DST)
                                  and key[1] & 1):
                asm.words.append(word)
            elif key[0] in (0x88, 0x89) and key[1] in (SRC1, DST) and key[2] == 1:
                asm.words.append(word)
        asm.words.extend(shape_dtype[operand] for operand in (SRC1, DST))
        return asm.words

    for index, op, packet in packets(words):
        if index in boundaries:
            remap[index] = len(out)
        if index in starts:
            current, entry = starts[index], True
        word = int(packet[0])
        if op in (OP_GLOAD, OP_GSTORE):
            dma = decode_dma(packet)
            g = dma['g_addr'] * 4
            dest = dma['sram_addr']
            size = dma['rows'] * dma['cols'] * 4
            replacement = None
            # A complete packet must be one contiguous global interval; partial
            # matches and strided reads remain DMA. This also bounds proof cost.
            if op == OP_GLOAD and current is not None and (dma["rows"] == 1 or dma["g_stride"] == dma["cols"]):
                for cg, cs, cn, cv, producer, canonical in reversed(cache):
                    if cg <= g and g + size <= cg + cn and producer < current["index"]:
                        source = cs + (g - cg) * 2
                        if source == dest:
                            replacement = []
                        elif canonical and entry and typed_fp16(g, size) and not _overlap(source, size * 2, dest, size * 2):
                            replacement = copy(source, dest, size)
                        if replacement is not None:
                            events.append(dict(kernel=current["index"], producer=producer,
                                               global_address_bytes=g, version=cv,
                                               source_nibbles=source, destination_nibbles=dest,
                                               saved_load_bytes=size,
                                               kind="same_address" if source == dest else "sram_copy"))
                            break
            if op == OP_GSTORE:
                version += 1
                invalidate(0, g, ((dma['rows'] - 1) * dma['g_stride'] + dma['cols']) * 4)
                # Coalesce a contiguous DMA into one proof interval.
                if dma["rows"] == 1 or dma["g_stride"] == dma["cols"]:
                    rows = [(g, dest, size)]
                elif dma['g_stride'] >= dma['cols']:
                    rows = _rows(dma)
                else:
                    # Overlapping rows write different versions inside one
                    # packet. Keep only the last row, which is fully current.
                    last = dma['rows'] - 1
                    rows = [(g + last * dma['g_stride'] * 4,
                             dest + last * dma['cols'] * 8, dma['cols'] * 4)]
                cache.extend((g, s, n, version, current["index"] if current else -1,
                              current.get("canonical_fp16_output", False) if current else False)
                             for g, s, n in rows)
            else:
                invalidate(1, dest, size * 2)
            if replacement:
                for injected in replacement:
                    effects.step(injected)
            peak_intervals = max(peak_intervals, len(cache))
            out.extend(packet if replacement is None else replacement)
            continue
        if op == 0x80:
            descriptors[(op, (word >> 28) & 15)] = word
        elif op == 0x82:
            descriptors[(op,)] = word
        elif op in (0x88, 0x89):
            operand = (word >> 30) & 3
            descriptors[(op, operand, (word >> 29) & 1)] = word
            shape_dtype[operand] = word
        write = effects.step(word)
        if write is None:
            cache.clear()
            unknown_writes += 1
        elif write and write[1]:
            invalidate(1, *write)
            precise_writes += 1
        if op not in (0, 0x80, 0x82, 0x88, 0x89, 0x8a, 0x8b, 0xf0):
            entry = False
        out.extend(packet)
    remap[len(words)] = len(out)
    return out, remap, dict(policy="write-through, versioned kernel-entry forwarding",
                           events=events,
                           peak_intervals=peak_intervals,
                           precise_writes=precise_writes, unknown_effects=unknown_writes,
                           saved_load_bytes=sum(e["saved_load_bytes"] for e in events))
