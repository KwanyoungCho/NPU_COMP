"""G-buffer memory of the 3B prefill layer: weights (always-live) vs activations,
across three scenarios — no reuse / liveness reuse (current) / theoretical ideal.
MEASURED (memplan.plan footprints). Reads measurements_detail.json.

Point 1: weights (packed static params + tiled input) dominate at ~202 MB and are
irreducible (every layer needs them resident), so the OVERALL saving is only modest.
Point 2: the activation working set is what reuse attacks — 73.1 -> 23.0 MB. The
'ideal' bar is the theoretical floor = max simultaneously-live activation footprint
(7.6 MB, what a perfect allocator would need); the gap between reuse (23.0) and ideal
(7.6) is fragmentation the exact-size free-list leaves (constants always bump; freed
slots of one size don't fill another size's need)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "measurements_detail.json")))["memory_elems"]
mb = lambda e: e * 2 / 1e6                       # fp16 elements -> MB
W, C = mb(M["weights"]), mb(M["constants"])
acts = [mb(M["act_noreuse"]), mb(M["act_reuse"]), mb(M["act_ideal"])]
tots = [mb(M["top_noreuse"]), mb(M["top_reuse"]), mb(M["top_ideal"])]

C_W = "#d8d8d5"    # weights: recessive neutral (irreducible)
C_C = "#b9b9b3"    # constants
C_A = "#2a78d6"    # activations: categorical slot 1 (the part reuse optimizes)
INK, MUTED = "#0b0b0b", "#52514e"

fig, ax = plt.subplots(figsize=(8.4, 6.6))
x = [0, 1, 2]
labels = ["no reuse", "reuse\n(current)", "ideal\n(theoretical min)"]
bw = 0.56
for i in x:
    ax.bar(i, W, bw, color=C_W, edgecolor="white", linewidth=2, label="weights (+tiled input)" if i == 0 else None)
    ax.bar(i, C, bw, bottom=W, color=C_C, edgecolor="white", linewidth=2, label="constants" if i == 0 else None)
    # the ideal bar is hypothetical -> hatch it so it reads as "not what we allocate today"
    hatch = "///" if i == 2 else None
    ax.bar(i, acts[i], bw, bottom=W + C, color=C_A, edgecolor="white", linewidth=2, hatch=hatch,
           label="activations" if i == 0 else None)
    ax.text(i, W / 2, f"weights\n{W:.0f} MB", ha="center", va="center", fontsize=9.5, color=MUTED, fontweight="bold")
    ax.text(i, W + C + acts[i] + 5, f"act {acts[i]:.1f} MB", ha="center", va="bottom", fontsize=10,
            color=C_A, fontweight="bold")
    tl = f"{tots[i]:.0f} MB" if i == 0 else f"{tots[i]:.0f} MB  (−{100*(1-tots[i]/tots[0]):.0f}%)"
    ax.text(i, tots[i] + 20, tl, ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("G-buffer (fp16, MB)", fontsize=11, color=MUTED)
ax.set_ylim(0, 330); ax.set_xlim(-0.6, 2.6)
ax.set_title("3B prefill layer memory", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5, frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(colors=MUTED)
plt.tight_layout()
out = os.path.join(HERE, "g_memory.png")
plt.savefig(out, dpi=130)
print("wrote", out)
