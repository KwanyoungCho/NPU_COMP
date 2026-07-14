"""Measured BEFORE (old ISA) vs AFTER (0710 retarget) — HF 3B prefill layer, best mode.
Both bars are REAL compiled command counts (no projection)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (role, BEFORE, AFTER) — measured command counts. mmul+accum merged (tagging-invariant).
rows = [
    ("matmul core (mmul+accum)", 1127712, 836832),
    ("transpose (Kᵀ)",           1048576,  16672),
    ("scatter (output)",          540672, 540672),
    ("gather (input)",            409600, 409600),
    ("reduce (norm/softmax)",     153716,  26624),
    ("layout (RoPE)",             131072, 131072),
    ("broadcast",                  50496,  50424),
    ("elementwise",                16802,  16802),
]
TB, TA = 3478647, 2028699
labels = [r[0] for r in rows]
before = np.array([r[1] for r in rows], float)
after  = np.array([r[2] for r in rows], float)
y = np.arange(len(rows))[::-1]      # top-to-bottom by magnitude
h = 0.38

fig, ax = plt.subplots(figsize=(12, 6.6))
b1 = ax.barh(y + h/2, before, h, color="#9e9e9e", label=f"BEFORE (old ISA)  total {TB:,}")
b2 = ax.barh(y - h/2, after,  h, color="#2ca02c", label=f"AFTER (0710 retarget)  total {TA:,}")

for i, (bf, af) in enumerate(zip(before, after)):
    yy = y[i]
    ax.text(bf + 12000, yy + h/2, f"{bf:,.0f}", va="center", fontsize=9, color="#555")
    d = 0 if bf == 0 else 100*(af-bf)/bf
    tag = "0%" if abs(d) < 0.5 else f"{d:+.0f}%"
    col = "#c0392b" if d < -1 else ("#888" if abs(d) < 1 else "#c0392b")
    ax.text(af + 12000, yy - h/2, f"{af:,.0f}  ({tag})", va="center", fontsize=9,
            color=("#1a7a1a" if d < -1 else "#888"))

ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("commands per 3B prefill layer (measured)")
ax.set_title(f"HF 3B prefill layer — measured BEFORE vs AFTER 0710 retarget\n"
             f"total {TB:,} → {TA:,}  (−{100*(1-TA/TB):.1f}%)  |  useful 32.4% → 41.2%",
             fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim(0, 1_200_000)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
out = "/home/chokwans99/NPU_cmodel/report/figs/g_before_after_hf_prefill.png"
plt.savefig(out, dpi=130)
print("wrote", out)
