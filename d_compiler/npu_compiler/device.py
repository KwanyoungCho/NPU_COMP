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
from .v09_contract import NUMERIC_POLICY

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
    numeric_policy: str = NUMERIC_POLICY
    reuse_sram: bool = True       # TVM USMP lifetime reuse within each kernel
    schedule_candidates: int = 2
    schedule_probe_word_budget: int = 100000
    schedule_search: str = "space"  # space, meta_schedule (ReplayFunc), or legacy
    schedule_seed: int = 0
    schedule_space_trials: int = 8
    # Count failed and over-capacity candidates too. None permits exhaustive
    # traversal until the existing legal-candidate / emitted-word limits.
    schedule_max_attempts: int | None = 64
    graph_search_seconds: float | None = 120.0  # additional search AFTER a legal module
    joint_memory: bool = True
    memory_beam_width: int = 4

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
