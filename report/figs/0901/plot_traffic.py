"""Global<->SRAM traffic and SRAM occupancy vs prompt length — MEASURED.

Reads report/figs/0901/traffic.json, written by d_compiler/analyze_traffic.py
(one link per point; traffic is summed from the DMA instructions in the linked
program, and the total was checked against the C-model's own counter -- both
say 1,619,976,421 cells for the 28-layer program).

The point of these two figures is that every gate we run is seq=7, and at that
one point the compiler looks optimal: weights are read exactly once and
activation round trips are 1% of traffic.  Both of those stop being true as
the prompt grows, and SRAM stops fitting altogether.

Colours: categorical blue/green for the two traffic kinds that behave
differently, red reserved for the capacity violation, grey for recessive
detail.
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
RAW = json.load(open(os.path.join(HERE, "traffic.json")))

WEIGHT_C, ACT_C, STORE_C = "#1f77b4", "#2ca02c", "#98df8a"
FAIL_C, GREY = "#d62728", "#7f7f7f"
MB = 1e6
CAPACITY = 8.0                      # MiB of SRAM
TILE = 64                           # the PE tile, and so the M-tile

rows = [r for r in RAW if r.get("label") != "verify"]
seqs = [r["seq"] for r in rows]
fits = [r.get("fits", False) for r in rows]
labels = [str(s) for s in seqs]
x = np.arange(len(rows))

weight = [r["load"].get("weight", 0) / MB if f else 0
          for r, f in zip(rows, fits)]
act_in = [r["load"].get("activation", 0) / MB if f else 0
          for r, f in zip(rows, fits)]
act_out = [sum(r["store"].values()) / MB if f else 0
           for r, f in zip(rows, fits)]
peak = [r["sram_peak_bytes"] / 1024 / 1024 if f else None
        for r, f in zip(rows, fits)]


def _needed(row):
    """How much a kernel asked for when the link refused it.

    The linker's message carries the figure ("takes it to 9.04 MiB of 8.00"),
    so the failing bars are drawn at what was actually required rather than at
    an arbitrary height above the line.
    """
    text = row.get("error", "")
    marker = "takes it to "
    if marker not in text:
        return CAPACITY * 1.15
    return float(text.split(marker)[1].split(" MiB")[0])


over = [None if f else _needed(r) for r, f in zip(rows, fits)]


def _bar_labels(ax, xs, tops, texts, colour="black", dy=0.02):
    span = ax.get_ylim()[1]
    for xi, top, text in zip(xs, tops, texts):
        ax.text(xi, top + span * dy, text, ha="center", va="bottom",
                fontsize=10, color=colour)


# ===== figure 1: where the bytes go =========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))

ax1.bar(x, weight, 0.62, color=WEIGHT_C, label="weight read")
ax1.bar(x, act_in, 0.62, bottom=weight, color=ACT_C,
        label="activation read back")
ax1.bar(x, act_out, 0.62, bottom=np.add(weight, act_in), color=STORE_C,
        label="activation written out")
totals = np.add(np.add(weight, act_in), act_out)
ax1.set_ylim(0, max(totals) * 1.35)
for xi, fit in zip(x, fits):
    if not fit:
        ax1.text(xi, max(totals) * 0.45, "does not fit in SRAM",
                 ha="center", va="center", color=FAIL_C, fontsize=12,
                 fontweight="bold", rotation=90)
_bar_labels(ax1, [xi for xi, f in zip(x, fits) if f],
            [t for t, f in zip(totals, fits) if f],
            [f"{t:.0f} MB" for t, f in zip(totals, fits) if f])
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel("prompt length (tokens)")
ax1.set_ylabel("global <-> SRAM traffic per layer (MB)")
ax1.set_title("(a) traffic for one real Llama 3.2 3B layer")
ax1.legend(loc="upper left")
ax1.grid(axis="y", alpha=0.25)

# per token: the number that should fall as the prompt grows, and does not
per_token = [t / s if f else None for t, s, f in zip(totals, seqs, fits)]
good = [(xi, v) for xi, v, f in zip(x, per_token, fits) if f]
ax2.plot([g[0] for g in good], [g[1] for g in good], "o-", color=WEIGHT_C,
         linewidth=2, markersize=8)
for xi, value in good:
    ax2.annotate(f"{value:.1f}", (xi, value), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=10)
ideal = weight[0] / np.array(seqs, dtype=float)
ax2.plot(x, ideal, "--", color=GREY, linewidth=1.8,
         label="ideal: each weight read once,\nno activation round trip")
ax2.set_yscale("log")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_xlabel("prompt length (tokens)")
ax2.set_ylabel("MB moved per token  (log)")
ax2.set_title("(b) cost per token — the gap is what SRAM planning would close")
ax2.legend()
ax2.grid(alpha=0.25)

fig.suptitle("Measured: one link per point, DMA instructions summed. "
             "Total checked against the C-model counter.",
             y=0.005, fontsize=10, color=GREY)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "g_traffic_by_seq.png"), dpi=150,
            bbox_inches="tight")

# ===== figure 2: the two causes =============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))

heights = [p if p is not None else o for p, o in zip(peak, over)]
colours = [WEIGHT_C if p is not None else FAIL_C for p in peak]
bars = ax1.bar(x, heights, 0.62, color=colours)
for bar, value in zip(bars, peak):
    if value is None:
        bar.set_hatch("//")
ax1.axhline(CAPACITY, color=FAIL_C, linestyle="--", linewidth=2)
ax1.text(-0.45, CAPACITY + 0.12, "SRAM capacity 8 MiB", ha="left",
         va="bottom", color=FAIL_C, fontsize=12)
for xi, value, height in zip(x, peak, heights):
    text = f"{value:.2f}" if value is not None else f"{height:.2f}\nlink fails"
    ax1.text(xi, height + 0.15, text, ha="center", va="bottom", fontsize=10,
             color="black" if value is not None else FAIL_C)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0, max(heights) * 1.32)
ax1.set_xlabel("prompt length (tokens)")
ax1.set_ylabel("peak SRAM a kernel needs (MiB)")
ax1.set_title("(a) the wall: elementwise kernels stage whole tensors")
ax1.grid(axis="y", alpha=0.25)

# weight re-reads: the closed-form prediction, and what was measured
one_pass = weight[0]
measured = [w / one_pass if f else None for w, f in zip(weight, fits)]
predicted = [np.ceil(s / TILE) for s in seqs]
ax2.plot(x, predicted, "s--", color=GREY, linewidth=1.8, markersize=8,
         label=r"predicted  $\lceil S/64 \rceil$")
good = [(xi, v) for xi, v, f in zip(x, measured, fits) if f]
ax2.plot([g[0] for g in good], [g[1] for g in good], "o-", color=WEIGHT_C,
         linewidth=2, markersize=9, label="measured")
for xi, value in good:
    ax2.annotate(f"{value:.2f}x", (xi, value), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=10, color=WEIGHT_C)
ax2.axhline(1.0, color=ACT_C, linestyle=":", linewidth=2)
ax2.text(len(x) - 0.15, 1.12, "floor: every weight read once", color=ACT_C,
         fontsize=11, ha="right")
ax2.text(len(x) - 0.15, predicted[-1] * 0.72,
         "measured stops at 128:\nlonger prompts do not link", color=GREY,
         fontsize=10, ha="right")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_xlabel("prompt length (tokens)")
ax2.set_ylabel("weight bytes moved / weight bytes")
ax2.set_title("(b) weights get re-read once the prompt exceeds one M-tile")
ax2.legend(loc="upper left")
ax2.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "g_sram_wall.png"), dpi=150,
            bbox_inches="tight")

print("wrote g_traffic_by_seq.png, g_sram_wall.png")
