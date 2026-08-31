"""The NPU as a TVM target, and the machine numbers that go with it.

``npu_target()`` is a real ``tvm.target.Target`` of kind ``ext_dev`` keyed
``npu``, so passes and schedules can dispatch on it the way they do for any
other backend.  ``ext_dev`` only accepts TVM's own attribute set, so the
machine's numbers -- SRAM size, tile, lanes -- sit in an :class:`NpuProfile`
selected by the target's ``model``, rather than being constants scattered
through the linker and the schedules.
"""
from __future__ import annotations

import dataclasses

import tvm

NPU_KEY = "npu"


@dataclasses.dataclass(frozen=True)
class NpuProfile:
    """What the compiler needs to know about the machine it is targeting."""

    model: str = "v09"
    sram_bytes: int = 8 * 1024 * 1024
    tile: int = 64                 # the matrix unit's square block
    lanes: int = 256               # vector lanes (longer vectors strip-mine)
    scratch_slots: int = 6         # temporaries for serializing expressions
    cell_bits: int = 32            # global memory addressing unit

    @property
    def sram_nibbles(self):
        return self.sram_bytes * 2


PROFILES = {"v09": NpuProfile()}


def npu_target(model="v09"):
    """The NPU target; ``model`` selects the machine profile."""
    if model not in PROFILES:
        raise ValueError(f"unknown NPU model {model!r}; have {sorted(PROFILES)}")
    return tvm.target.Target(
        {"kind": "ext_dev", "keys": [NPU_KEY], "model": model})


def profile_of(target=None):
    """The profile a target selects (the default machine when given None)."""
    if target is None:
        return PROFILES["v09"]
    target = tvm.target.Target(target)
    if NPU_KEY not in [str(key) for key in target.keys]:
        raise ValueError(f"{target} is not an NPU target")
    return PROFILES[str(target.attrs.get("model", "v09"))]
