"""G-buffer memory of the 3B prefill layer: weights (always-live) vs activations
(A1 liveness reuse). MEASURED (memplan.plan footprints). Reads measurements_detail.json.

The point: weights (the packed static params + tiled input) dominate at ~202 MB and are
irreducible (every layer needs them resident), so the OVERALL saving is only -18%. The
activation working set, however, is the part A1 actually attacks: its PEAK drops -68%
(73.9 -> 23.8 MB) once liveness reuse frees each binding var's slot at its last read."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "measurements_detail.json")))["memory_elems"]
mb = lambda e: e * 2 / 1e6                       # fp16 elements -> MB
W, C = mb(M["weights"]), mb(M["constants"])
A_bump, A_peak = mb(M["act_bump"]), mb(M["act_peak"])
tot_bump, tot_reuse = mb(M["top_bump"]), mb(M["top_reuse"])

C_W = "#d8d8d5"    # weights: recessive neutral (irreducible)
C_C = "#b9b9b3"    # constants
C_A = "#2a78d6"    # activations: categorical slot 1 (the part A1 optimizes)
INK, MUTED = "#0b0b0b", "#52514e"

fig, ax = plt.subplots(figsize=(7.6, 6.6))
x = [0, 1]
labels = ["bump\n(no reuse)", "A1 reuse\n(liveness)"]
acts = [A_bump, A_peak]
bw = 0.55
for i in x:
    ax.bar(i, W, bw, color=C_W, edgecolor="white", linewidth=2, label="weights (+tiled input)" if i == 0 else None)
    ax.bar(i, C, bw, bottom=W, color=C_C, edgecolor="white", linewidth=2, label="constants" if i == 0 else None)
    ax.bar(i, acts[i], bw, bottom=W + C, color=C_A, edgecolor="white", linewidth=2, label="activations" if i == 0 else None)
    ax.text(i, W / 2, f"weights\n{W:.0f} MB", ha="center", va="center", fontsize=9.5, color=MUTED, fontweight="bold")
    ax.text(i, W + C + acts[i] / 2, f"act\n{acts[i]:.1f}", ha="center", va="center", fontsize=9.5,
            color="white", fontweight="bold")
    tot = W + C + acts[i]
    ax.text(i, tot + 4, f"{tot:.1f} MB", ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
# activation-reduction callout
ax.annotate(f"activations −68%\n{A_bump:.1f} → {A_peak:.1f} MB",
            xy=(1, W + C + A_peak), xytext=(1.42, W + 45), ha="center", va="center",
            fontsize=10, color=C_A, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_A, lw=1.6))

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("G-buffer (fp16, MB)", fontsize=11, color=MUTED)
ax.set_ylim(0, 320); ax.set_xlim(-0.6, 2.0)
ax.set_title("3B prefill-layer memory — A1 liveness reuse\n"
             f"weights dominate (irreducible) → total {tot_bump:.0f} → {tot_reuse:.0f} MB (−18%),\n"
             f"but the activation working set peak drops −68%", fontsize=11.5)
ax.legend(loc="upper right", fontsize=9.5, frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(colors=MUTED)
plt.tight_layout()
out = os.path.join(HERE, "g_memory.png")
plt.savefig(out, dpi=130)
print("wrote", out)
