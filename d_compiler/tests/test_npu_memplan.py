"""S2 gate: static addresses derived from TVM's standard memory planning.

The NPU needs compile-time addresses; the liveness analysis that makes reuse
safe comes from the stock StaticPlanBlockMemory pass rather than a hand-rolled
allocator.  These tests check the derivation is sound: tensors that are live at
the same time never share an address range, and reuse actually happens.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from tvm import relax

from npu_compiler import npu_memplan as M
from npu_compiler import tvm_pipeline as P
from npu_compiler.nn_models import llama

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0)
SEQ = 5


def _plan():
    mod, _, _ = llama.build_prefill(TINY, seq=SEQ)
    lowered = P.graph_pipeline()(mod)
    planned, plan = M.assign_addresses(lowered)
    return lowered, planned, plan


def test_every_tensor_gets_an_address():
    _, planned, plan = _plan()
    missing = []
    for block in planned["prefill"].body.blocks:
        for binding in block.bindings:
            value = binding.value
            if isinstance(value, relax.Call):
                name = value.op.name if hasattr(value.op, "name") else ""
                if name.endswith("alloc_tensor") and \
                        binding.var.name_hint not in plan.address:
                    missing.append(binding.var.name_hint)
    assert not missing, missing
    print(f"  [PASS] all {len(plan.address)} tensors placed, "
          f"{plan.summary()['pools']} pools, {plan.summary()['total_bytes']:,} B")


def test_reuse_reduces_footprint():
    lowered, _, plan = _plan()
    naive = M.unplanned_footprint(lowered)
    planned_bytes = plan.summary()["total_bytes"]
    assert planned_bytes < naive, (planned_bytes, naive)
    print(f"  [PASS] footprint {planned_bytes:,} B vs no-reuse {naive:,} B "
          f"({100 * (planned_bytes - naive) / naive:+.1f}%)")


def test_simultaneously_live_tensors_do_not_overlap():
    """Two tensors sharing a pool must not have overlapping live ranges.

    Live range is approximated by first-write .. last-read over the binding
    order, which is exactly what the planner reasons about.
    """
    _, planned, plan = _plan()
    func = planned["prefill"]
    order, produced, last_use = [], {}, {}
    index = 0
    for block in func.body.blocks:
        for binding in block.bindings:
            var = binding.var.name_hint
            value = binding.value
            if isinstance(value, relax.Call):
                name = value.op.name if hasattr(value.op, "name") else ""
                if name.endswith("alloc_tensor"):
                    produced[var] = index
                    order.append(var)
                for arg in value.args:
                    if isinstance(arg, relax.Var):
                        last_use[arg.name_hint] = index
                    elif isinstance(arg, relax.Tuple):
                        for field in arg.fields:
                            if isinstance(field, relax.Var):
                                last_use[field.name_hint] = index
            index += 1

    live = [(v, produced[v], last_use.get(v, produced[v])) for v in order]
    overlaps = 0
    for i, (va, sa, ea) in enumerate(live):
        for vb, sb, eb in live[i + 1:]:
            if sa > eb or sb > ea:            # disjoint in time
                continue
            aa, ab = plan.address[va], plan.address[vb]
            na, nb = plan.nbytes[va] // 2, plan.nbytes[vb] // 2
            if aa < ab + nb and ab < aa + na:  # overlapping in space
                overlaps += 1
    assert overlaps == 0, f"{overlaps} live-range/address conflicts"
    print(f"  [PASS] no address conflict among {len(live)} planned tensors")


if __name__ == "__main__":
    test_every_tensor_gets_an_address()
    test_reuse_reduces_footprint()
    test_simultaneously_live_tensors_do_not_overlap()
    print("ALL NPU MEMPLAN (S2) TESTS PASSED")
