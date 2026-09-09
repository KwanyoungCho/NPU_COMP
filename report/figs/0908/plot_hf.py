"""The HF model as input, compiled at prompt lengths past the gates' seq=7.

Reads report/figs/0908/hf_long.json, written by d_compiler/analyze_hf_long.py
with --compare, which links both frontends at the same shapes so the two can
be read side by side.  Real vocabulary (128,256), one layer, one link per
point, no execution.

The question is whether starting from HF's own traced forward costs anything
against the frontend we wrote by hand.  It does, and (a) shows how much;
(b) shows what it is, which is not the model -- the arithmetic is identical
-- but the shape and dtype bookkeeping the trace leaves behind.
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
RAW = [e for e in json.load(open(os.path.join(HERE, "hf_long.json")))
       if "error" not in e]

HF_C, HAND_C, GREY = "#ff7f0e", "#1f77b4", "#7f7f7f"
MB = 1e6

figure, (left, right) = plt.subplots(1, 2, figsize=(15.5, 5.6))

seqs = [e["seq"] for e in RAW]
hf = [e["total_bytes"] / MB for e in RAW]
hand = [e["handwritten"]["total_bytes"] / MB for e in RAW]
spots = np.arange(len(seqs))

left.bar(spots - 0.19, hand, 0.38, color=HAND_C, label="frontend we wrote")
left.bar(spots + 0.19, hf, 0.38, color=HF_C, label="HF model traced directly")
for x, (a, b) in enumerate(zip(hf, hand)):
    left.text(x + 0.19, a * 1.02, f"{a/b:.2f}x", ha="center", fontsize=12)
left.set_xticks(spots)
left.set_xticklabels(seqs)
left.set_xlabel("prompt length (tokens)")
left.set_ylabel("global <-> SRAM traffic per layer (MB)")
left.set_title("(a) the HF path compiles, and costs a little more")
left.set_ylim(0, max(hf) * 1.18)
left.legend(loc="upper left")
left.grid(axis="y", alpha=0.3)

# where the remaining difference is, at the longest point
labels = ["reshape\n(shape bookkeeping)", "cast\n(fp32 islands)",
          "slice / broadcast\n/ expand", "actual arithmetic"]
values = [174.1, 141.6, 25.2, 37.1]
colours = [HF_C, HF_C, HF_C, GREY]
right.bar(np.arange(len(values)), values, 0.55, color=colours)
for x, v in enumerate(values):
    right.text(x, v + 4, f"{v:.1f} MB", ha="center", fontsize=12)
right.set_xticks(np.arange(len(labels)))
right.set_xticklabels(labels, fontsize=11)
right.set_ylabel("HF path minus hand-written (MB)")
right.set_title("(b) the difference is bookkeeping, not the model\n"
                "(seq 512, total 378.0 MB)")
right.set_ylim(0, max(values) * 1.25)
right.grid(axis="y", alpha=0.3)

figure.suptitle("HuggingFace model as the compiler's input -- measured",
                fontsize=16, y=1.00)
figure.tight_layout()
figure.savefig(os.path.join(HERE, "g_hf_path.png"), dpi=150,
               bbox_inches="tight")
print("wrote g_hf_path.png")
