#!/usr/bin/env python3
"""Is keeping tensors in SRAM across kernel boundaries worth it?

SRAM staging resets to zero at every kernel, so a kernel's result goes out to
global and the next kernel reads it back.  At seq 512 that round trip is half
the traffic, which makes an inter-op resident region look like the obvious
next optimization.  This measures whether it actually is, and the answer is
no: the round trip is large but it sits in tensors too big to be resident,
and the space a resident region would take is space the matmul panels are
already using to better effect.

Two halves, both static -- neither runs the C-model:

  candidates()  walks the planned function and records, per intermediate,
                (first writer, last reader, bytes, uses).  Keeping it
                resident saves the store plus every load.

  matmul_cost() replays _matmul_mapping's own arithmetic at a reduced budget,
                which is what a reserved region leaves the kernels.

Reading the two against each other gives the net, which is what decides it.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tvm import relax, tir

from npu_compiler import model, npu_intrin, npu_legalize, npu_memplan
from npu_compiler import tvm_pipeline as pipeline
from npu_compiler.nn_models import llama

TILE = 64
ITEMSIZE = 2


def layer_config(layers=1, vocab=256):
    cfg = model.LLAMA_3_2_3B
    return dict(hidden_size=cfg.D, intermediate_size=cfg.F,
                num_hidden_layers=layers, num_attention_heads=cfg.H,
                num_key_value_heads=cfg.KV, head_dim=cfg.HD,
                vocab_size=vocab, rms_norm_eps=1e-5, rope_theta=500000.0)


def lower(seq, layers=1, vocab=256):
    mod, _, _ = llama.build_prefill(layer_config(layers, vocab), seq=seq)
    mod = pipeline.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                                  fuse=False, lift_params=True)(mod)
    return npu_memplan.assign_addresses(mod, "prefill")


def candidates(planned, plan):
    """[(name, first, last, bytes, uses)] for every intermediate read back.

    After CallTIRRewrite a kernel's result is its LAST argument, an ``alloc``
    var, and a following ``name = alloc`` binding gives it the name consumers
    use.  Those aliases have to be resolved or nothing appears to be read back
    at all.
    """
    func = planned["prefill"]
    params = {p.name_hint for p in func.params}
    alias = {}

    def canon(name):
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def names_of(arg):
        if isinstance(arg, relax.Tuple):
            return [canon(f.name_hint) for f in arg.fields
                    if isinstance(f, relax.Var)]
        return [canon(arg.name_hint)] if isinstance(arg, relax.Var) else []

    calls, index = [], 0
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
            args = list(value.args)
            calls.append(([n for a in args[:-1] for n in names_of(a)],
                          names_of(args[-1])))
            index += 1

    first, last, uses = {}, {}, {}
    for i, (ins, outs) in enumerate(calls):
        for name in outs:
            if name not in params:
                first.setdefault(name, i)
        for name in ins:
            if name in params or name not in first:
                continue
            last[name] = i
            uses[name] = uses.get(name, 0) + 1

    rows = [(name, first[name], last[name], plan.nbytes.get(name, 0),
             uses[name]) for name in first if name in last]
    rows.sort(key=lambda row: -row[3] * (1 + row[4]))
    return len(calls), rows


def select(rows, kernels, capacity):
    """Greedy by benefit/byte; a tensor is refused if it would overflow the
    reserved region at any kernel it is live across."""
    load = [0] * (kernels + 1)
    chosen, saved = [], 0
    for row in sorted(rows, key=lambda r: -(1 + r[4]) / max(r[3], 1)):
        _, first, last, size, use = row
        if size > capacity or any(load[i] + size > capacity
                                  for i in range(first, last + 1)):
            continue
        for i in range(first, last + 1):
            load[i] += size
        chosen.append(row)
        saved += size * (1 + use)
    return chosen, saved


def matmul_cost(shapes, budget_bytes):
    """Bytes the layer's matmuls move when _matmul_mapping runs at this budget.

    Mirrors npu_link._matmul_mapping rather than calling it: the point is to
    sweep the budget, and the linker takes it from the profile.
    """
    total = 0
    for _, (rows, cols, depth) in shapes:
        base = (rows * depth * math.ceil(cols / TILE)
                + depth * cols * math.ceil(rows / TILE))
        if rows % TILE:
            total += base * ITEMSIZE      # mapping declines; pad_einsum path
            continue
        room = budget_bytes - (depth * TILE + TILE * TILE) * ITEMSIZE
        if room <= 0:
            total += base * ITEMSIZE
            continue
        span = room // (depth * ITEMSIZE)

        def panel_for(extent):
            panel = extent
            while panel > span and panel % 2 == 0 and panel // 2 >= TILE:
                panel //= 2
            return panel if panel <= span else None

        best = base
        panel = panel_for(rows)
        if panel:
            best = min(best, rows * depth
                       + depth * cols * math.ceil(rows / panel))
        panel = panel_for(cols)
        if panel:
            best = min(best, depth * cols
                       + rows * depth * math.ceil(cols / panel))
        total += best * ITEMSIZE
    return total


def report(seq, layers, peak_mib):
    planned, plan = lower(seq, layers)
    kernels, rows = candidates(planned, plan)
    shapes = [(gv.name_hint, npu_intrin.matmul_extents(planned, gv.name_hint))
              for gv, fn in planned.functions.items()
              if isinstance(fn, tir.PrimFunc)
              and npu_intrin.matmul_extents(planned, gv.name_hint)]

    stores = sum(size for *_, size, _ in rows)
    round_trip = sum(size * (1 + use) for *_, size, use in rows)
    print(f"\n=== seq {seq}, {layers} layer(s): {kernels} kernels, "
          f"{len(rows)} intermediates read back ===")
    print(f"kernel-boundary round trip: {round_trip/1e6:.1f} MB "
          f"(stores {stores/1e6:.1f} + re-reads {(round_trip-stores)/1e6:.1f})")

    print(f"\n{'tensor':<12}{'k':>4}{'->':>4}{'MiB':>8}{'uses':>6}{'saves MB':>10}")
    for name, first, last, size, use in rows[:8]:
        print(f"{name:<12}{first:>4}{last:>4}{size/2**20:>8.2f}{use:>6}"
              f"{size*(1+use)/1e6:>10.1f}")

    print(f"\nreserving R MiB for residency (kernels then get {peak_mib} - R):")
    print(f"{'R':>6} {'resident':>9} {'saves':>10} {'matmuls lose':>14} {'net':>10}")
    base = matmul_cost(shapes, int(peak_mib * 2**20))
    for reserve in range(0, 7):
        budget = int((peak_mib - reserve) * 2**20)
        if budget <= 0:
            break
        _, saved = select(rows, kernels, reserve * 2**20) if reserve else ([], 0)
        lost = matmul_cost(shapes, budget) - base
        print(f"{reserve:>4} MiB {len(_ ) if reserve else 0:>9} "
              f"{saved/1e6:>9.1f}M {-lost/1e6:>13.1f}M "
              f"{(saved-lost)/1e6:>+9.1f}M")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--peak", type=float, default=7.5,
                        help="SRAM a kernel may use today, MiB")
    args = parser.parse_args()
    for seq in args.seq:
        report(seq, args.layers, args.peak)


if __name__ == "__main__":
    main()
