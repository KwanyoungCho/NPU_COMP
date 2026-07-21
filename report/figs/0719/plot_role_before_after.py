"""Per-role BEFORE (pre-A4 row-major) vs AFTER (A4 5d-2 tile) — 3B prefill layer,
MEASURED command counts by role (asm.tags). Companion to g_a4_progression.png: that
one shows the stage-by-stage total; this shows WHERE the -52.7% came from and what the
tile-native ops cost. Reads measurements_detail.json (no re-compilation).

Same before/after encoding as figs/0710/g_before_after_prefill.png (before=neutral gray,
after=green), so the two reports read consistently."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "measurements_detail.json")))
R, T = D["roles"]["row"], D["roles"]["tile"]
TB, TA = D["total"]["row"], D["total"]["tile"]

# merge mmul+accum into "matmul core" (as in the 0710 figure)
def merged(r):
    return {"matmul core (mmul+accum)": r.get("mmul", 0) + r.get("accum", 0),
            "scatter (output)": r.get("scatter", 0), "gather (input)": r.get("gather", 0),
            "broadcast": r.get("broadcast", 0), "layout (RoPE slice/concat)": r.get("layout", 0),
            "reduce (norm/softmax)": r.get("reduce", 0), "transpose (Kᵀ)": r.get("transpose", 0),
            "elementwise (SiLU)": r.get("elementwise", 0)}
mb, ma = merged(R), merged(T)
labels = list(mb.keys())
before = np.array([mb[k] for k in labels], float)
after = np.array([ma[k] for k in labels], float)
y = np.arange(len(labels))[::-1]; h = 0.38

fig, ax = plt.subplots(figsize=(12, 6.8))
ax.barh(y + h/2, before, h, color="#9e9e9e", label=f"Before  (pre-A4 row-major, total {TB:,})")
ax.barh(y - h/2, after, h, color="#2ca02c", label=f"After  (A4 tile 5d-2, total {TA:,})")
for i, (bf, af) in enumerate(zip(before, after)):
    yy = y[i]
    ax.text(bf + 9000, yy + h/2, f"{bf:,.0f}", va="center", fontsize=9, color="#555")
    d = 0 if bf == 0 else 100 * (af - bf) / bf
    tag = "0%" if abs(d) < 0.5 else f"{d:+.0f}%"
    col = "#1a7a1a" if d < -1 else ("#b8860b" if d > 1 else "#888")
    lab = f"{af:,.0f}  ({tag})" if af or bf else "0"
    ax.text(af + 9000, yy - h/2, lab, va="center", fontsize=9, color=col)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("commands per 3B prefill layer")
ax.set_title("A4 tile-blocked layout — per-role before/after  (3B prefill layer, S=128)\n"
             f"total {TB:,} → {TA:,}  (−{100*(1-TA/TB):.1f}%):  gather+scatter → 0,  "
             "transpose/broadcast/RoPE cheaper,  reduce +6%", fontsize=11.5)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0, 950_000); ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
out = os.path.join(HERE, "g_role_before_after.png")
plt.savefig(out, dpi=130)
print("wrote", out)
