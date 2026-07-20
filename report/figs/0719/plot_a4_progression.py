"""A4 tile-blocked layout progression — 3B prefill layer, MEASURED command counts.
Reads report/figs/0719/measurements.json (no re-compilation). Edit the JSON to update.

Stacked bars: each stage's total = useful/other (neutral) + gather + scatter. As A4
propagates the tile-blocked layout (FFN -> RMSNorm/residual -> attention), the coloured
gather+scatter shrinks to ZERO while the total drops -52.7% — i.e. the ~48% overhead that
report_0710 concluded needs a row-major strided HW mode is removed in SOFTWARE.

Colours are the dataviz reference categorical slots 1 (blue) / 2 (green), which the
palette reference certifies (worst adjacent CVD ΔE 9.1 light / 8.4 dark). 'other' is a
recessive neutral, not a categorical hue."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "measurements.json")))
S = M["stages"]
T0 = M["baseline_total"]

labels = [s["label"] for s in S]
gather = [s["gather"] for s in S]
scatter = [s["scatter"] for s in S]
other = [s["total"] - s["gather"] - s["scatter"] for s in S]
total = [s["total"] for s in S]
x = list(range(len(S)))

C_OTHER = "#d8d8d5"   # recessive neutral (useful compute + remaining overhead)
C_GATHER = "#2a78d6"  # categorical slot 1 (blue)
C_SCATTER = "#008300"  # categorical slot 2 (green)
INK = "#0b0b0b"
MUTED = "#52514e"

fig, ax = plt.subplots(figsize=(10.5, 6.6))
bw = 0.62
# stacked: other (bottom) -> gather -> scatter ; 2px white gap between fills via edgecolor
b1 = ax.bar(x, other, bw, color=C_OTHER, edgecolor="white", linewidth=2,
            label="useful / other")
b2 = ax.bar(x, gather, bw, bottom=other, color=C_GATHER, edgecolor="white", linewidth=2,
            label="gather (input)")
b3 = ax.bar(x, scatter, bw, bottom=[o + g for o, g in zip(other, gather)],
            color=C_SCATTER, edgecolor="white", linewidth=2, label="scatter (output)")

# direct labels: gather / scatter values (only when non-zero) centred in their segment
for i in range(len(S)):
    if gather[i] > 0:
        ax.text(x[i], other[i] + gather[i] / 2, f"{gather[i]:,}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    if scatter[i] > 0:
        ax.text(x[i], other[i] + gather[i] + scatter[i] / 2, f"{scatter[i]:,}", ha="center",
                va="center", fontsize=9, color="white", fontweight="bold")
    # total above each bar with delta vs baseline
    d = 100 * (total[i] - T0) / T0
    tag = "baseline" if i == 0 else f"{d:+.1f}%"
    ax.text(x[i], total[i] + 34000, f"{total[i]:,}\n({tag})", ha="center", va="bottom",
            fontsize=10, color=INK, fontweight="bold")
    # gather+scatter == 0 callout on the last stage
    if gather[i] == 0 and scatter[i] == 0:
        ax.annotate("gather = scatter = 0\n(fully eliminated in SW)",
                    xy=(x[i], other[i]), xytext=(x[i] - 0.02, other[i] * 0.62),
                    ha="center", va="center", fontsize=9.5, color=C_SCATTER, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("commands per 3B prefill layer", fontsize=11, color=MUTED)
ax.set_ylim(0, 2_500_000)
ax.set_title("A4 tile-blocked layout — gather/scatter eliminated in software\n"
             "3B prefill layer (S=128):  total 2,235,194 → 1,057,758  (−52.7%),  "
             "gather+scatter 950,272 → 0", fontsize=12.5)
ax.legend(loc="upper right", fontsize=10, frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(colors=MUTED)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v else "0"))
plt.tight_layout()
out = os.path.join(HERE, "g_a4_progression.png")
plt.savefig(out, dpi=130)
print("wrote", out)
