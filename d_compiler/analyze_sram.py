#!/usr/bin/env python3
"""How well is the 8 MiB actually used?

The traffic sweep says how many bytes cross the bus; this says why.  For each
kernel it reports what it occupies, what it moves, and what it would move if
every byte crossed once -- so the gap between the two can be attributed to a
kernel and to a reason rather than to the program as a whole.

Three things it is meant to answer:

  * how much of the 8 MiB the busiest kernel takes, and how much sits idle
  * which kernels are above their floor, and by how much
  * whether the occupancy is capacity-bound (the planner took all it could)
    or shape-bound (the kernel had nothing more to put there)
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tvm import relax, tir

from npu_compiler import npu_legalize, npu_link, npu_quantize
from npu_compiler import tvm_pipeline as pipeline
from npu_compiler.isa_v09 import DMA_WORDS, OP_GLOAD, OP_GSTORE
from npu_compiler.nn_models import llama
from npu_compiler.device import PROFILES

CELL = 4
CAPACITY = 8 * 1024 * 1024


def layer_config(layers, vocab):
    return llama.benchmark_config(layers, vocab)


def _floors(planned, plan):
    """Per call, the bytes its distinct operands and result weigh."""
    func = planned["prefill"]
    alias = {}

    def canon(name):
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def names(arg):
        if isinstance(arg, relax.Tuple):
            return [canon(f.name_hint) for f in arg.fields
                    if isinstance(f, relax.Var)]
        return [canon(arg.name_hint)] if isinstance(arg, relax.Var) else []

    out = []
    for block in func.body.blocks:
        for binding in block.bindings:
            value = binding.value
            if isinstance(value, relax.Var):
                alias[binding.var.name_hint] = canon(value.name_hint)
                continue
            if isinstance(value, relax.TupleGetItem):
                members = plan.tuples.get(value.tuple_value.name_hint)
                if members and value.index < len(members):
                    alias[binding.var.name_hint] = canon(members[value.index])
                continue
            if not (isinstance(value, relax.Call)
                    and isinstance(value.op, relax.GlobalVar)):
                continue
            if not isinstance(planned[value.op], tir.PrimFunc):
                continue
            seen, floor = set(), 0
            for arg in value.args:
                for name in names(arg):
                    if name not in seen:
                        seen.add(name)
                        floor += plan.nbytes.get(name, 0)
            out.append(floor)
    return out


def measure(seq, layers, vocab, quantize, reuse_sram=True):
    mod, _, _ = llama.build_prefill(layer_config(layers, vocab), seq=seq)
    if quantize:
        mod = npu_quantize.QuantizeW8A8()(mod)
    lowered = pipeline.graph_pipeline(
        custom_legalize=npu_legalize.legalize_map(),
        fuse=False, lift_params=True)(mod)
    asm, plan = npu_link.compile_program(
        lowered, "prefill", peephole=False,
        profile=replace(PROFILES["v09"], reuse_sram=reuse_sram))
    planned, _ = npu_link.npu_memplan.assign_addresses(lowered, "prefill")
    floors = _floors(planned, plan)
    if not (len(floors) == len(plan.kernel_span) == asm.kernel_count):
        raise RuntimeError("kernel attribution is incomplete: "
                           f"{asm.kernel_count} calls, {len(plan.kernel_span)} spans, "
                           f"{len(floors)} operand floors")

    rows = {}
    for (name, start, end), floor in zip(plan.kernel_span, floors):
        moved, index = 0, start
        while index < end:
            if asm.words[index] & 0xFF in (OP_GLOAD, OP_GSTORE):
                moved += ((int(asm.words[index + 3]) >> 16)
                          * (int(asm.words[index + 3]) & 0xFFFF) * CELL)
                index += DMA_WORDS
            else:
                index += 1
        entry = rows.setdefault(name, {"moved": 0, "floor": 0, "calls": 0})
        entry["moved"] += moved
        entry["floor"] += floor
        entry["calls"] += 1
    for name, entry in rows.items():
        entry["sram_bytes"] = plan.sram_peak.get(name, 0) // 2
    return {"seq": seq, "layers": layers, "vocab": vocab,
            "reuse_sram": reuse_sram,
            "quantize": quantize, "words": len(asm.words),
            "kernels": rows, "schedules": plan.schedules,
            "kernel_records": plan.kernel_records,
            "sram_layouts": plan.sram_layouts,
            "native_diagnostics": plan.native_diagnostics,
            "numeric_policy": plan.numeric_policy,
            "frontend_precision": plan.frontend_precision,
            "codegen_contract": plan.codegen_contract}


def report(result):
    rows = result["kernels"]
    total_moved = sum(e["moved"] for e in rows.values())
    total_floor = sum(e["floor"] for e in rows.values())
    peak = max(e["sram_bytes"] for e in rows.values())
    print(f"\n=== seq {result['seq']}"
          f"{' W8A8' if result['quantize'] else ''} — "
          f"{total_moved/1e6:.1f} MB, 이론상 최소 {total_floor/1e6:.1f} MB "
          f"({total_moved/max(total_floor,1):.2f}배) ===")
    print(f"{'커널':<34}{'SRAM':>9}{'/8MiB':>7}"
          f"{'이동량':>11}{'최소':>10}{'배수':>7}")
    for name, entry in sorted(rows.items(),
                              key=lambda kv: -kv[1]["sram_bytes"])[:10]:
        ratio = entry["moved"] / entry["floor"] if entry["floor"] else 0
        print(f"{name[:32]:<34}{entry['sram_bytes']/2**20:>8.2f}M"
              f"{100*entry['sram_bytes']/CAPACITY:>6.0f}%"
              f"{entry['moved']/1e6:>10.1f}M{entry['floor']/1e6:>9.1f}M"
              f"{ratio:>7.2f}")
    print(f"  최대 사용 {peak/2**20:.2f} MiB / 8 MiB "
          f"({100*peak/CAPACITY:.0f}%), 유휴 {(CAPACITY-peak)/2**20:.2f} MiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, nargs="+", default=[512])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--no-sram-reuse", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    results = []
    for seq in args.seq:
        result = measure(seq, args.layers, args.vocab, args.quantize, not args.no_sram_reuse)
        report(result)
        results.append(result)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
