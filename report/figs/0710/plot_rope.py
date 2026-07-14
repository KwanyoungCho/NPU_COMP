"""RoPE: permutation-matmul vs sign-inversion + on-device cos/sin.
Manual 3B prefill layer (packed), MEASURED role distribution. Total is ~unchanged
(instruction-neutral); the story is the role SHIFT (matmul gather/scatter -> layout)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (role, BEFORE perm-matmul, AFTER sign-inv+on-device cos/sin) — measured
rows = [
    ("matmul (mmul)",     597504, 594432),
    ("scatter",           606208, 540672),
    ("gather",            475136, 409600),
    ("accum",             243040, 242400),
    ("broadcast",         203514, 206639),
    ("layout (RoPE)",      17408, 148480),
    ("reduce",             63608,  63608),
    ("transpose",          16672,  16672),
    ("elementwise",        12380,  12690),
]
TB, TA = 2235471, 2235194
labels = [r[0] for r in rows]
before = np.array([r[1] for r in rows], float)
after  = np.array([r[2] for r in rows], float)
y = np.arange(len(rows))[::-1]
h = 0.38

fig, ax = plt.subplots(figsize=(12, 6.8))
ax.barh(y + h/2, before, h, color="#9e9e9e", label=f"BEFORE (permutation-matmul)  total {TB:,}")
ax.barh(y - h/2, after,  h, color="#7b52ab", label=f"AFTER (sign-inv + on-device cos/sin)  total {TA:,}")
for i, (bf, af) in enumerate(zip(before, after)):
    yy = y[i]
    ax.text(bf + 7000, yy + h/2, f"{bf:,.0f}", va="center", fontsize=9, color="#555")
    d = 0 if bf == 0 else 100*(af-bf)/bf
    tag = "0%" if abs(d) < 0.5 else f"{d:+.0f}%"
    col = "#1a7a1a" if d < -1 else ("#b8860b" if d > 1 else "#888")
    ax.text(af + 7000, yy - h/2, f"{af:,.0f}  ({tag})", va="center", fontsize=9, color=col)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("commands per 3B prefill layer (measured, packed)")
ax.set_title("RoPE retarget — MEASURED, 3B prefill layer\n"
             f"total {TB:,} -> {TA:,}  (-0.01%, instruction-neutral): "
             "matmul gather/scatter (131k) -> slice/concat layout;\n"
             "gain = Rot matrix removed from memory + cos/sin now on-device from position",
             fontsize=12)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0, 700000)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
out = "/home/chokwans99/NPU_cmodel/report/figs/0710/g_rope_before_after.png"
plt.savefig(out, dpi=130)
print("wrote", out)
