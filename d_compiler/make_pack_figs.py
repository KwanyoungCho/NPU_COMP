"""Graphs for the input-weight-packing measurement (report_0703 §7).
Reads /tmp/pack_results.json (from analyze_pack.py); writes report/figs/g8_*, g9_*.
Reuses the report's ROLE_COLORS so the figures match g1..g23."""
import os, sys, json
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from analyze_layer import ROLE_COLORS, ROLE_EN, ROLE_ORDER

plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report", "figs")
OFF_C, ON_C = "#d62728", "#2ca02c"      # unpacked (gather-heavy) / packed (lean)

res = json.load(open("/tmp/pack_results.json"))
KS = ["kv_proj", "attn_ffn", "prefill", "lm_head"]
LAB = {"kv_proj": "decode\nkv_proj", "attn_ffn": "decode\nattn_ffn",
       "prefill": "prefill\nlayer", "lm_head": "lm_head"}


def _m(v):
    return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"


# ===== G8: packing effect — per-kernel totals + token cost =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
x = np.arange(len(KS)); w = 0.38
off = [res[k]["off"]["total"] for k in KS]
on = [res[k]["on"]["total"] for k in KS]
ax1.bar(x - w/2, off, w, color=OFF_C, label="packing OFF (weights = param, gathered)")
ax1.bar(x + w/2, on, w, color=ON_C, label="packing ON (input weight packing)")
ax1.set_yscale("log")
for i in range(len(KS)):
    ax1.text(x[i] - w/2, off[i], _m(off[i]), ha="center", va="bottom", fontsize=11)
    ax1.text(x[i] + w/2, on[i], f"{_m(on[i])}\n({100*(on[i]-off[i])/off[i]:+.0f}%)",
             ha="center", va="bottom", fontsize=11, color=ON_C)
ax1.set_xticks(x); ax1.set_xticklabels([LAB[k] for k in KS])
ax1.set_ylabel("total commands (log)"); ax1.set_ylim(top=max(off) * 3)
ax1.set_title("G8a. Per-kernel commands — input weight packing")
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1)

tok_off, tok_on = res["_token"]["off"], res["_token"]["on"]
ax2.bar(["OFF", "ON"], [tok_off, tok_on], color=[OFF_C, ON_C], width=0.55)
ax2.text(0, tok_off, "  " + _m(tok_off), ha="center", va="bottom", fontsize=14)
ax2.text(1, tok_on, "  " + _m(tok_on), ha="center", va="bottom", fontsize=14)
ax2.annotate(f"{100*(tok_on-tok_off)/tok_off:+.1f}%", xy=(1, tok_off*0.5), ha="center",
             va="center", fontsize=22, color=ON_C, fontweight="bold")
ax2.set_ylabel("commands / token"); ax2.set_ylim(0, tok_off * 1.18)
ax2.set_title("G8b. Cost per generated token\n(28 decode layers + lm_head)")
plt.tight_layout(); plt.savefig(f"{FIG}/g8_pack_effect.png", dpi=120, bbox_inches="tight"); plt.close()

# ===== G9: role composition OFF vs ON (100% stacked) — gather collapse =====
fig, ax = plt.subplots(figsize=(14, 6.5))
present = []
for r in ROLE_ORDER:
    if any(res[k][s]["roles"].get(r, 0) for k in KS for s in ("off", "on")):
        present.append(r)
xs = np.arange(len(KS)); w = 0.38
for st, dx, tag in [("off", -w/2, "OFF"), ("on", w/2, "ON")]:
    bottoms = np.zeros(len(KS))
    for r in present:
        frac = np.array([100 * res[k][st]["roles"].get(r, 0) / res[k][st]["total"] for k in KS])
        ax.bar(xs + dx, frac, w, bottom=bottoms, color=ROLE_COLORS.get(r, "#888"))
        bottoms += frac
    for i in range(len(KS)):                              # OFF/ON tag + gather% callout
        g = 100 * res[KS[i]][st]["roles"].get("gather", 0) / res[KS[i]][st]["total"]
        ax.text(xs[i] + dx, 101, tag, ha="center", va="bottom", fontsize=10, color="#333")
        ax.text(xs[i] + dx, g/2, f"{g:.0f}%", ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([LAB[k] for k in KS]); ax.set_ylim(0, 108)
ax.set_ylabel("% of kernel commands")
ax.set_title("G9. Command composition per kernel — packing OFF vs ON (white % = gather share)")
ax.legend(handles=[Patch(color=ROLE_COLORS.get(r), label=ROLE_EN.get(r, r)) for r in present],
          loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(len(present), 6))
plt.tight_layout(); plt.savefig(f"{FIG}/g9_pack_roles.png", dpi=120, bbox_inches="tight"); plt.close()

print("wrote", f"{FIG}/g8_pack_effect.png", "and g9_pack_roles.png")
print(f"token cost {tok_off:,} -> {tok_on:,} ({100*(tok_on-tok_off)/tok_off:+.1f}%)")
