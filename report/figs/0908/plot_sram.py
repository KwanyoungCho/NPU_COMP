"""How the 8 MiB is used, and where the traffic above the minimum sits.

Reads report/figs/0908/sram_fp16.json, written by d_compiler/analyze_sram.py
(one link per prompt length; traffic summed from the DMA instructions in the
linked program, no execution).

The pair answers two questions the earlier figures could not.  The traffic
plot showed bytes moved but not whether that was good; here every kernel's
floor -- its distinct operands plus its result, each crossing once -- is
computed too, so "how far from the best possible" has a number.  And the
occupancy plot shows the planner filling more of the 8 MiB as the prompt
grows, which is the intended behaviour: idle SRAM is traffic not avoided.

Colours: blue for what is moved, grey for the floor it is measured against,
red for the capacity limit, green where the two coincide.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                     "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 11})

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = json.load(open(os.path.join(HERE, "sram_fp16.json")))

MOVED_C, FLOOR_C, LIMIT_C, GOOD_C = "#1f77b4", "#bdbdbd", "#d62728", "#2ca02c"
MB, MiB, CAPACITY = 1e6, 2 ** 20, 8.0


def totals(entry):
    rows = entry["kernels"].values()
    return (sum(r["moved"] for r in rows) / MB,
            sum(r["floor"] for r in rows) / MB,
            max(r["sram_bytes"] for r in rows) / MiB)


figure, (left, right) = plt.subplots(1, 2, figsize=(15.5, 5.6))

seqs = [e["seq"] for e in RAW]
moved = [totals(e)[0] for e in RAW]
floor = [totals(e)[1] for e in RAW]
peak = [totals(e)[2] for e in RAW]
spots = np.arange(len(seqs))

left.bar(spots - 0.19, floor, 0.38, color=FLOOR_C,
         label="floor: every byte crosses once")
left.bar(spots + 0.19, moved, 0.38, color=MOVED_C, label="measured")
for x, (m, f) in enumerate(zip(moved, floor)):
    ratio = m / f
    left.text(x, max(m, f) * 1.04, f"{ratio:.2f}x",
              ha="center", fontsize=12,
              color=GOOD_C if ratio < 1.02 else "black",
              fontweight="bold" if ratio < 1.02 else "normal")
left.set_xticks(spots)
left.set_xticklabels(seqs)
left.set_xlabel("prompt length (tokens)")
left.set_ylabel("global <-> SRAM traffic per layer (MB)")
left.set_title("(a) traffic sits on the floor until the prompt is long")
left.set_ylim(0, max(moved) * 1.22)
left.legend(loc="upper left")
left.grid(axis="y", alpha=0.3)

right.bar(spots, peak, 0.5, color=MOVED_C)
right.axhline(CAPACITY, color=LIMIT_C, ls="--", lw=2)
right.text(len(seqs) - 0.5, CAPACITY + 0.12, "SRAM capacity 8 MiB",
           color=LIMIT_C, ha="right", fontsize=12)
for x, value in enumerate(peak):
    right.text(x, value + 0.12, f"{value:.2f}\n({100*value/CAPACITY:.0f}%)",
               ha="center", fontsize=11)
right.set_xticks(spots)
right.set_xticklabels(seqs)
right.set_xlabel("prompt length (tokens)")
right.set_ylabel("SRAM the busiest kernel takes (MiB)")
right.set_title("(b) a longer prompt fills more of the 8 MiB")
right.set_ylim(0, CAPACITY * 1.16)
right.grid(axis="y", alpha=0.3)

figure.suptitle("What the memory planning achieves -- measured, one link per point",
                fontsize=16, y=1.00)
figure.tight_layout()
figure.savefig(os.path.join(HERE, "g_sram_efficiency.png"), dpi=150,
               bbox_inches="tight")

# ---- where the excess is, at the longest prompt --------------------------
longest = max(RAW, key=lambda e: e["seq"])
rows = [(name, r["moved"] / MB, r["floor"] / MB, r["sram_bytes"] / MiB)
        for name, r in longest["kernels"].items() if r["floor"]]
rows.sort(key=lambda r: -(r[1] - r[2]))
top = rows[:8]

figure2, axis = plt.subplots(figsize=(11.5, 6.0))
spots = np.arange(len(top))
axis.barh(spots, [r[2] for r in top], 0.6, color=FLOOR_C, label="floor")
axis.barh(spots, [r[1] - r[2] for r in top], 0.6,
          left=[r[2] for r in top], color=MOVED_C,
          label="excess: operands re-read because they do not fit 8 MiB")
for y, r in enumerate(top):
    axis.text(r[1] * 1.01, y, f"{r[1]/max(r[2],1e-9):.2f}x  |  {r[3]:.2f} MiB of 8",
              va="center", fontsize=11)
axis.set_yticks(spots)
axis.set_yticklabels([r[0][:30] for r in top])
axis.invert_yaxis()
axis.set_xlabel("traffic (MB)")
axis.set_xlim(0, max(r[1] for r in top) * 1.45)
axis.set_title(f"the excess is only in kernels that hit the 8 MiB wall "
               f"(seq {longest['seq']}, one layer)")
axis.legend(loc="lower right")
axis.grid(axis="x", alpha=0.3)
figure2.tight_layout()
figure2.savefig(os.path.join(HERE, "g_sram_excess.png"), dpi=150,
                bbox_inches="tight")
print("wrote g_sram_efficiency.png, g_sram_excess.png")
