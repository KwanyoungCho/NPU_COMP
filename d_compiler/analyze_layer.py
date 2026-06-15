"""Llama 3.2 3B 한 레이어 '커맨드 오버헤드' 정적 분석 (prefill + decode).

측정 = 정적 명령 수(프로그램 크기 ≈ fetch/issue 부하). latency 아님(cost.py 참고).

두 시나리오:
  - prefill : Mq=SEQ=128 토큰을 한 번에. (square attention 128x128)
  - decode  : Mq=1 토큰 생성, KV 캐시 길이 L=128. (M=1 -> 64로 패딩됨)

각 모드에서: 구성요소 단독 컴파일 -> role별 명령 수 -> 출현횟수로 레이어 합
 -> 미지원 ISA 추가 시 절감(상한/현실) -> matplotlib 그래프(report/figs) + 콘솔 요약.

실행:  /home/chokwans99/anaconda3/envs/npu-tvm/bin/python d_compiler/analyze_layer.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from tvm import relax
from npu_compiler import codegen, memplan, cost, legalize, model

cfg = model.LLAMA_3_2_3B
D, HD, F = cfg.D, cfg.HD, cfg.F
FIGDIR = os.path.join(os.path.dirname(HERE), "report", "figs")   # 상위 NPU_cmodel/report/figs
os.makedirs(FIGDIR, exist_ok=True)


def _c(a):
    return relax.const(np.asarray(a, dtype="float16"))


def compile_op(param_shapes, build):
    bb = relax.BlockBuilder()
    pvars = [relax.Var(n, relax.TensorStructInfo(list(s), "float16")) for n, s in param_shapes]
    with bb.function("main", pvars):
        with bb.dataflow():
            out = build(bb, pvars); gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    mod = bb.finalize()
    mp = memplan.plan(mod["main"])
    asm = codegen.compile_func(mod["main"], mp, tile=64, mm_backend="tir")
    return cost.analyze(asm, mp)


def mm(a, b):
    return lambda bb, v: bb.emit(relax.op.matmul(v[0], v[1]))


def build_components(Mq, Lctx):
    """Mq=query rows(prefill=SEQ, decode=1), Lctx=attention context(=KV length)."""
    cos, sin, rot = legalize.rope_tables(Mq, HD, base=cfg.rope_base, llama3_scaling=cfg.rope_scale)
    mask = legalize.causal_mask(Mq) if Mq == Lctx else np.zeros((Mq, Lctx))
    return [
        ("RMSNorm",      [("x", (Mq, D)), ("w", (1, D))],
            lambda bb, v: legalize.rms_norm(bb, v[0], v[1], Mq, D, eps=cfg.eps), 2, "norm"),
        ("Q/K/V proj",   [("x", (Mq, D)), ("W", (D, HD))], mm(0, 1), 24 + 8 + 8, "attn"),
        ("RoPE",         [("q", (Mq, HD))],
            lambda bb, v: legalize.rope(bb, v[0], _c(cos), _c(sin), _c(rot)), 24 + 8, "attn"),
        ("K^T transpose", [("k", (Lctx, HD))],
            lambda bb, v: bb.emit(relax.op.permute_dims(v[0], axes=[1, 0])), 8, "attn"),
        ("scores Q@Kt",  [("q", (Mq, HD)), ("kt", (HD, Lctx))], mm(0, 1), 24, "attn"),
        ("scale+mask",   [("s", (Mq, Lctx))],
            lambda bb, v: bb.emit(relax.op.add(
                bb.emit(relax.op.multiply(v[0], _c(np.full((Mq, Lctx), 0.088)))), _c(mask))), 24, "attn"),
        ("softmax",      [("s", (Mq, Lctx))],
            lambda bb, v: legalize.softmax_lastdim(bb, v[0], Mq, Lctx), 24, "attn"),
        ("ctx P@V",      [("p", (Mq, Lctx)), ("vv", (Lctx, HD))], mm(0, 1), 24, "attn"),
        ("O proj",       [("c", (Mq, HD)), ("Wo", (HD, D))], mm(0, 1), 24, "attn"),
        ("attn resid +", [("a", (Mq, D)), ("b", (Mq, D))],
            lambda bb, v: bb.emit(relax.op.add(v[0], v[1])), 25, "attn"),
        ("gate/up proj", [("x", (Mq, D)), ("W", (D, F))], mm(0, 1), 2, "ffn"),
        ("SiLU",         [("z", (Mq, F))], lambda bb, v: legalize.silu(bb, v[0], Mq, F), 1, "ffn"),
        ("SwiGLU mul",   [("a", (Mq, F)), ("b", (Mq, F))],
            lambda bb, v: bb.emit(relax.op.multiply(v[0], v[1])), 1, "ffn"),
        ("down proj",    [("h", (Mq, F)), ("Wd", (F, D))], mm(0, 1), 1, "ffn"),
    ]


ROLE_ORDER = ["mmul", "gather", "scatter", "accum", "transpose",
              "reduce", "broadcast", "elementwise", "layout", "pad", "untagged"]
ROLE_EN = {"mmul": "matmul (useful)", "gather": "gather (input)", "scatter": "scatter (output)",
           "accum": "K-accumulate", "transpose": "transpose", "reduce": "reduce (ones-mm)",
           "broadcast": "broadcast (ones-mm)", "elementwise": "elementwise", "layout": "layout",
           "pad": "pad", "untagged": "other"}
ROLE_COLORS = {"mmul": "#2ca02c", "gather": "#d62728", "scatter": "#ff7f0e", "accum": "#9467bd",
               "transpose": "#8c564b", "reduce": "#e377c2", "broadcast": "#bcbd22",
               "elementwise": "#17becf", "layout": "#7f7f7f", "pad": "#c7c7c7", "untagged": "#ccc"}


def run_mode(tag, Mq, Lctx, ntokens):
    print(f"\n========== {tag}  (Mq={Mq}, Lctx={Lctx}) ==========")
    rows, silu_total = [], 0
    for label, ps, build, mult, cls in build_components(Mq, Lctx):
        st = compile_op(ps, build)
        per = st["total"]; roles = {k: v * mult for k, v in st["by_role"].items()}
        rows.append((label, mult, per, per * mult, roles, cls))
        if label == "SiLU":
            silu_total = per * mult
    layer_total = sum(r[3] for r in rows)
    by_role = {}
    for r in rows:
        for k, v in r[4].items():
            by_role[k] = by_role.get(k, 0) + v
    g = lambda k: by_role.get(k, 0)
    useful = g("mmul") + g("accum")
    n_tr = g("transpose") / 32768.0
    n_gs = (g("gather") + g("scatter") + g("pad")) / 512.0
    feats = [
        ("strided load/save", g("gather") + g("scatter") + g("pad"), n_gs * 2),
        ("transpose unit",    g("transpose"),                        n_tr * 7),
        ("row-reduce(sum)",   g("reduce"),                           g("reduce") * 0.5),
        ("broadcast",         g("broadcast"),                        g("broadcast") * 0.5),
        ("native activation", silu_total,                            silu_total * 0.2),
    ]
    savings = [(n, up, max(0.0, up - rep)) for (n, up, rep) in feats]
    attn = sum(r[3] for r in rows if r[5] in ("attn", "norm"))
    ffn = sum(r[3] for r in rows if r[5] == "ffn")
    print(f"  레이어 총 명령 = {layer_total:,}   (토큰당 {layer_total/ntokens:,.0f})")
    print(f"  유효(matmul+accum) {useful:,} ({100*useful/layer_total:.1f}%)  "
          f"gather {g('gather'):,} ({100*g('gather')/layer_total:.1f}%)")
    real_total = sum(s[2] for s in savings)
    print(f"  ISA 5종 현실 절감 누적 {100*real_total/layer_total:.1f}% -> 남음 {layer_total-int(real_total):,}")
    _graphs(rows, by_role, savings, layer_total, useful, tag)
    return dict(tag=tag, Mq=Mq, Lctx=Lctx, ntokens=ntokens, layer_total=layer_total,
                by_role=by_role, rows=rows, savings=savings, useful=useful, attn=attn, ffn=ffn,
                per_token=layer_total / ntokens, real_saved=real_total)


# ============================ 그래프 ============================
def _graphs(rows, by_role, savings, layer_total, useful, tag):
    # G1: 크기 tier별 LINEAR subplot
    tiers = [(">1M", 1e6, float("inf")), ("10K-1M", 1e4, 1e6), ("<10K", 0, 1e4)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, (tname, lo, hi) in zip(axes, tiers):
        grp = sorted([r for r in rows if lo <= r[3] < hi], key=lambda r: -r[3])
        for ax_set, _ in [(ax, 0)]:
            bottoms = np.zeros(len(grp)); labels = [r[0] for r in grp]
            for role in ROLE_ORDER:
                vals = np.array([r[4].get(role, 0) for r in grp], dtype=float)
                if vals.sum() == 0:
                    continue
                ax.bar(labels, vals, bottom=bottoms, color=ROLE_COLORS.get(role))
                bottoms += vals
        ax.set_title(f"{tname} cmds/OP"); ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("commands (linear)")
    present = [r for r in ROLE_ORDER if any(row[4].get(r, 0) for row in rows)]
    fig.legend(handles=[Patch(color=ROLE_COLORS.get(r), label=ROLE_EN.get(r, r)) for r in present],
               loc="upper center", ncol=len(present), fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"G1[{tag}]. Commands per OP — LINEAR by magnitude (role proportions honest)", y=0.96)
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g1_per_op_{tag}.png", dpi=120, bbox_inches="tight"); plt.close()

    # G2: role 분포
    items = sorted(by_role.items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([ROLE_EN.get(k, k) for k, _ in items], [v for _, v in items],
            color=[ROLE_COLORS.get(k, "#888") for k, _ in items])
    for i, (k, v) in enumerate(items):
        ax.text(v, i, f" {v:,} ({100*v/layer_total:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("commands"); ax.set_title(f"G2[{tag}]. Command-type (role) distribution")
    ax.set_xscale("log"); plt.tight_layout(); plt.savefig(f"{FIGDIR}/g2_role_dist_{tag}.png", dpi=120); plt.close()

    # G3: realistic 절감 워터폴
    stages = ["baseline"] + [s[0] for s in savings]
    vals = [float(layer_total)]; cur = float(layer_total)
    for _n, _up, real in savings:
        cur -= real; vals.append(cur)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(range(len(stages)), vals, color=["#1f77b4"] + ["#2ca02c"] * len(savings))
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        if i > 0:
            ax.text(i, v + layer_total * 0.04, f"-{100*(vals[i-1]-v)/layer_total:.1f}%",
                    ha="center", va="bottom", fontsize=8, color="#2ca02c")
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_ylabel("remaining commands"); ax.set_ylim(0, layer_total * 1.12)
    ax.set_title(f"G3[{tag}]. Realistic ISA savings (replacement cost incl.): "
                 f"{layer_total:,} -> {vals[-1]:,.0f} (-{100*(1-vals[-1]/layer_total):.1f}%)")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g3_isa_waterfall_{tag}.png", dpi=120); plt.close()

    # G4: 유효 vs 오버헤드
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    overhead = layer_total - useful
    ax1.pie([useful, overhead], labels=[f"useful (matmul+accum)\n{useful:,}", f"overhead\n{overhead:,}"],
            autopct="%1.1f%%", colors=["#2ca02c", "#d62728"], startangle=90)
    ax1.set_title(f"G4a[{tag}]. Useful vs overhead")
    oh = {ROLE_EN.get(k, k): v for k, v in by_role.items() if k not in ("mmul", "accum")}
    oh = dict(sorted(oh.items(), key=lambda x: -x[1]))
    ax2.bar(list(oh.keys()), list(oh.values()), color="#d62728"); ax2.set_yscale("log")
    ax2.set_title(f"G4b[{tag}]. Overhead breakdown")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/g4_useful_vs_overhead_{tag}.png", dpi=120); plt.close()

    # G5: OP 비중 + role 구성(100%)
    rr = list(reversed(sorted(rows, key=lambda r: -r[3]))); labels = [r[0] for r in rr]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    shares = [100 * r[3] / layer_total for r in rr]
    ax1.barh(labels, shares, color="#1f77b4")
    for i, v in enumerate(shares):
        ax1.text(v, i, f" {v:.1f}% ({rr[i][3]:,})", va="center", fontsize=8)
    ax1.set_xlabel("% of total layer"); ax1.set_title(f"G5a[{tag}]. Each OP's share of layer")
    ax1.set_xlim(0, max(shares) * 1.3)
    bottoms = np.zeros(len(rr))
    for role in ROLE_ORDER:
        fr = np.array([100 * r[4].get(role, 0) / r[3] for r in rr], dtype=float)
        if fr.sum() == 0:
            continue
        ax2.barh(labels, fr, left=bottoms, color=ROLE_COLORS.get(role), label=ROLE_EN.get(role, role))
        bottoms += fr
    ax2.set_xlim(0, 100); ax2.set_xlabel("% within OP")
    ax2.set_title(f"G5b[{tag}]. Role mix within each OP (100%-normalized)")
    ax2.legend(fontsize=7, ncol=2, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g5_share_and_mix_{tag}.png", dpi=120); plt.close()


def compare_graph(pf, dc):
    # G6: prefill vs decode — 토큰당 명령 수 + 유효 비율
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    names = ["prefill (SEQ=128)", "decode (M=1)"]
    pt = [pf["per_token"], dc["per_token"]]
    ax1.bar(names, pt, color=["#1f77b4", "#d62728"])
    for i, v in enumerate(pt):
        ax1.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("commands PER TOKEN"); ax1.set_yscale("log")
    ax1.set_title(f"G6a. Commands per token ({pt[1]/pt[0]:.0f}x worse for decode)")
    uf = [100 * pf["useful"] / pf["layer_total"], 100 * dc["useful"] / dc["layer_total"]]
    ax2.bar(names, uf, color=["#2ca02c", "#2ca02c"])
    for i, v in enumerate(uf):
        ax2.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("useful (matmul+accum) %"); ax2.set_title("G6b. Useful-compute share")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g6_prefill_vs_decode.png", dpi=120); plt.close()


def main():
    pf = run_mode("prefill", cfg.SEQ, cfg.SEQ, ntokens=cfg.SEQ)
    dc = run_mode("decode", 1, cfg.SEQ, ntokens=1)
    compare_graph(pf, dc)
    print(f"\n== prefill vs decode ==")
    print(f"  per-token commands: prefill {pf['per_token']:,.0f}  decode {dc['per_token']:,.0f}  "
          f"({dc['per_token']/pf['per_token']:.0f}x)")
    print(f"  useful%: prefill {100*pf['useful']/pf['layer_total']:.1f}%  "
          f"decode {100*dc['useful']/dc['layer_total']:.1f}%")
    print(f"\n  figs -> {FIGDIR}")


if __name__ == "__main__":
    main()
