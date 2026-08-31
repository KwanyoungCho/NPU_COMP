"""Drop descriptor writes that cannot change what the machine does.

A v09 program is a single straight-line stream with no branch or loop
instruction, so the value of every descriptor register at every point is
decidable by walking the stream once.  A write that stores the value a
register already holds is therefore provably dead.

The state modelled here is exactly the state the C-model keeps (see the
dispatch in ``_poc/mysim_v09.cpp``):

  0x80        one half of a descriptor's main or partial address
  0x82        the vector length
  0x88/0x89   one descriptor's rows or cols, and always its dtype
  0x8A/0x8B   one half of the activation or weight scale address

Nothing else writes any of that.  0x15 also updates an address half, but it
executes at the same time, so it is never a candidate.  Addresses are tracked
per half, which is what the hardware updates, so a half write is judged
without needing to know the other half.
"""
from __future__ import annotations

from .isa_v09 import DMA_WORDS, OP_GLOAD, OP_GSTORE

_ADDR, _VLEN, _SHAPE, _SCALE = 0x80, 0x82, (0x88, 0x89), (0x8A, 0x8B)


def _writes(state, key, value):
    """Record a write; True if it changes the register (or it was unknown)."""
    if key in state and state[key] == value:
        return False
    state[key] = value
    return True


def eliminate_dead_stores(words):
    """-> a shorter word stream that executes identically."""
    out = []
    state = {}
    index, total = 0, len(words)
    while index < total:
        word = words[index]
        opcode = word & 0xFF
        if opcode in (OP_GLOAD, OP_GSTORE):
            out.extend(words[index:index + DMA_WORDS])
            index += DMA_WORDS
            continue
        keep = True
        if opcode == _ADDR:
            operand = (word >> 30) & 3
            if operand < 3:                      # operand 3 is ignored anyway
                keep = _writes(
                    state,
                    ("addr", operand, (word >> 28) & 1, (word >> 29) & 1),
                    (word >> 8) & 0xFFFF)
        elif opcode == _VLEN:
            keep = _writes(state, ("vlen",), (word >> 8) & 0xFFFF)
        elif opcode in _SHAPE:
            operand = (word >> 30) & 3
            if operand < 3:
                extent = _writes(
                    state, (opcode, operand, (word >> 29) & 1),
                    (word >> 8) & 0xFFFF)
                dtype = _writes(state, ("dtype", operand), (word >> 25) & 3)
                keep = extent or dtype           # one word writes both
        elif opcode in _SCALE:
            keep = _writes(state, ("scale", opcode, (word >> 29) & 1),
                           (word >> 8) & 0xFFFF)
        if keep:
            out.append(word)
        index += 1
    return out
