"""Per-role BEFORE (pre-decode-opt) vs AFTER (M=1 gather/scatter skip) for the 3B
DECODE step (kv_proj + attn_ffn kernels, M=1). MEASURED command counts by role
(asm.tags). Companion to g_role_before_after.png (that one is the PREFILL layer);
this is the decode counterpart. Reads measurements_decode.json (no re-compilation).

Same before/after encoding as g_role_before_after.png (before=neutral gray,
after=green) so the two read consistently. The point: decode is M=1 (row-major,
NOT tile-blocked like prefill), yet the same 'only row 0 matters' insight removes
the M-padding gather (activation) and scatter (output) — the KV-cache reads are
the only gather left."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "measurements_decode.json")))
R = D["roles_combined"]["before"]
T = D["roles_combined"]["after"]
TB, TA = D["total_words"]["before"], D["total_words"]["after"]

# display order + labels (mirror the prefill figure)
order = [("matmul", "matmul core (mmul+accum)"),
         ("scatter", "scatter (output)"),
         ("gather", "gather (input)"),
         ("reduce", "reduce (norm/softmax)"),
         ("broadcast", "broadcast"),
         ("elementwise", "elementwise (SiLU)"),
         ("layout", "layout (RoPE slice/concat)"),
         ("pad", "M-pad + row-0 extract")]
labels = [lab for _, lab in order]
before = np.array([R[k] for k, _ in order], float)
after = np.array([T[k] for k, _ in order], float)
y = np.arange(len(labels))[::-1]; h = 0.38

fig, ax = plt.subplots(figsize=(12, 6.8))
ax.barh(y + h/2, before, h, color="#9e9e9e", label=f"Before  (pre-decode-opt, total {TB:,})")
ax.barh(y - h/2, after, h, color="#2ca02c", label=f"After  (M=1 gather/scatter skip, total {TA:,})")
for i, (bf, af) in enumerate(zip(before, after)):
    yy = y[i]
    ax.text(bf + 3000, yy + h/2, f"{bf:,.0f}", va="center", fontsize=9, color="#555")
    d = 0 if bf == 0 else 100 * (af - bf) / bf
    tag = "0%" if abs(d) < 0.5 else f"{d:+.0f}%"
    col = "#1a7a1a" if d < -1 else ("#b8860b" if d > 1 else "#888")
    lab = f"{af:,.0f}  ({tag})" if af or bf else "0"
    ax.text(af + 3000, yy - h/2, lab, va="center", fontsize=9, color=col)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("commands per 3B decode step  (kv_proj + attn_ffn, M=1, MAX=128)")
ax.set_title("Decode (M=1) — per-role before/after  (3B, one token)\n"
             f"total {TB:,} → {TA:,}  (−{100*(1-TA/TB):.1f}%):  scatter → 0,  "
             "gather −87% (only KV-cache Kᵀ/V left),  matmul/reduce unchanged", fontsize=11.5)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0, 470_000); ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
out = os.path.join(HERE, "g_decode_before_after.png")
plt.savefig(out, dpi=130)
print("wrote", out)
