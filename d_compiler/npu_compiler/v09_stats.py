"""Decode v09 packets for optimization costs and hardware validation exports.

DMA payload words are data, never instructions. Counts are exact for the
straight-line stream; they are deliberately not a latency model.
"""
from collections import Counter

from .isa_v09 import decode_dma, OP_GLOAD, OP_GSTORE


def packets(words):
    index = 0
    while index < len(words):
        op = int(words[index]) & 255
        size = 4 if op in (OP_GLOAD, OP_GSTORE) else 1
        if index + size > len(words):
            raise ValueError(f"truncated v09 packet at word {index}")
        yield index, op, words[index:index + size]
        index += size


def analyze(words, plan=None, *, detail=True):
    counts = Counter()
    load = store = 0
    for _, op, packet in packets(words):
        counts[op] += 1
        if op in (OP_GLOAD, OP_GSTORE):
            dma = decode_dma(packet)
            size = dma["rows"] * dma["cols"] * 4
            if op == OP_GLOAD:
                load += size
            else:
                store += size
    result = dict(words=len(words), instructions=sum(counts.values()),
                  dma_packets=counts[OP_GLOAD]+counts[OP_GSTORE],
                  dma_load_bytes=load, dma_store_bytes=store,
                  dma_bytes=load + store,
                  opcodes={f"0x{op:02x}": n for op, n in sorted(counts.items())},
                  measurement="static instruction stream; no cycle estimate")
    if plan is not None:
        result["sram_peak_bytes"] = max(plan.sram_peak.values(), default=0) // 2
        result["global_image_bytes"] = plan.top * plan.unit_bytes
        if detail:
            result["kernels"] = [dict(index=i, name=name,
                                  **analyze(words[first:last]))
                             for i, (name, first, last) in enumerate(plan.kernel_span)]
    return result


def cost(stats):
    """Explicit lexicographic policy: traffic, code size, then local SRAM."""
    return (stats["dma_bytes"], stats["words"], stats.get("sram_peak_bytes", 0))


def check_counters(words, counters):
    """Compare independent functional-model counters to decoded packets."""
    stats = analyze(words)
    expected = dict(words_executed=stats["instructions"],
                    dma_cells_loaded=stats["dma_load_bytes"] // 4,
                    dma_cells_stored=stats["dma_store_bytes"] // 4,
                    gload=stats["opcodes"].get("0xa0", 0),
                    gstore=stats["opcodes"].get("0xa8", 0),
                    vquant=stats["opcodes"].get("0x1a", 0),
                    vdequant=stats["opcodes"].get("0x1b", 0))
    for key, value in expected.items():
        if counters.get(key) != value:
            raise ValueError(f"{key}: decoded {value}, measured {counters.get(key)}")
    return expected
