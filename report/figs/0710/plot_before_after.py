"""BEFORE vs AFTER the 0710 ISA update — 3B prefill layer, MEASURED command counts.
Reads report/figs/0710/measurements.json (no re-compilation). Edit the JSON to update.

Uses the MANUAL build_prefill_layer_module (the real generation path) for BOTH bars, so
the comparison is consistent AND reflects the RoPE change (permutation-matmul -> sign-inv +
on-device cos/sin): see the 'layout (RoPE)' bar jump and 'elementwise' (native SiLU) drop."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = json.load(open(os.path.join(HERE, "measurements.json")))["datasets"]
B = DS["manual_old_isa"]["roles"]; A = DS["manual_signinv_rope"]["roles"]
TB = DS["manual_old_isa"]["total"]; TA = DS["manual_signinv_rope"]["total"]

# merge mmul+accum into "matmul core" (MAC shifts the tile-save tag between them)
def merged(r):
    return {"matmul core (mmul+accum)": r.get("mmul", 0) + r.get("accum", 0),
            "transpose (Kᵀ)": r.get("transpose", 0), "scatter (output)": r.get("scatter", 0),
            "gather (input)": r.get("gather", 0), "broadcast": r.get("broadcast", 0),
            "reduce (norm/softmax)": r.get("reduce", 0), "layout (RoPE)": r.get("layout", 0),
            "elementwise (SiLU)": r.get("elementwise", 0)}
mb, ma = merged(B), merged(A)
labels = list(mb.keys())
before = np.array([mb[k] for k in labels], float)
after = np.array([ma[k] for k in labels], float)
y = np.arange(len(labels))[::-1]; h = 0.38

fig, ax = plt.subplots(figsize=(12, 6.8))
ax.barh(y + h/2, before, h, color="#9e9e9e", label=f"Before  (total {TB:,})")
ax.barh(y - h/2, after, h, color="#2ca02c", label=f"After  (total {TA:,})")
for i, (bf, af) in enumerate(zip(before, after)):
    yy = y[i]
    ax.text(bf + 12000, yy + h/2, f"{bf:,.0f}", va="center", fontsize=9, color="#555")
    d = 0 if bf == 0 else 100*(af-bf)/bf
    tag = "0%" if abs(d) < 0.5 else f"{d:+.0f}%"
    col = "#1a7a1a" if d < -1 else ("#b8860b" if d > 1 else "#888")
    ax.text(af + 12000, yy - h/2, f"{af:,.0f}  ({tag})", va="center", fontsize=9, color=col)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("commands per 3B prefill layer")
ax.set_title(f"3B prefill layer (S=128)\n"
             f"total {TB:,} → {TA:,}  (−{100*(1-TA/TB):.1f}%)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0, 1_200_000); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
out = os.path.join(HERE, "g_before_after_prefill.png")
plt.savefig(out, dpi=130)
print("wrote", out)
