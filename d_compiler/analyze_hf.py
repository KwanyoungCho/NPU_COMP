"""실제 HF Llama 3.2 3B 연산을 frontend로 import한 '진짜 그래프' 기준 오버헤드 분석.

analyze_layer.py(수동 빌더 단독 컴파일)와 달리, 여기서는:
  PyTorch LlamaLayer(HF 수식 그대로) → frontend.import_torch(torch.export→import_legalize)
  → codegen(emit_log로 per-binding 귀속) → role/OP별 명령 수 → ISA 절감 → 그래프.

HF 충실도: RMSNorm/GQA/causal-softmax/SwiGLU 동일, RoPE는 HF식 rotate_half=slice+neg+concat,
Kᵀ 전치는 KV-head당 1회 재사용(HF 배치 동작과 동일). 우리 codegen이 2D이므로 head는 펼침.

실행:  /home/chokwans99/anaconda3/envs/npu-tvm/bin/python d_compiler/analyze_hf.py
"""
import os, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch, torch.nn as nn

from npu_compiler import frontend, codegen, memplan, cost, legalize as LG
from analyze_layer import ROLE_ORDER, ROLE_EN, ROLE_COLORS

D, H, KV, HD, F = 3072, 24, 8, 128, 8192
GPK = H // KV
FIGDIR = os.path.join(os.path.dirname(HERE), "report", "figs")
os.makedirs(FIGDIR, exist_ok=True)


def _f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


class RMSNorm(nn.Module):
    def __init__(s, d):
        super().__init__(); s.w = nn.Parameter(torch.ones(d))
    def forward(s, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * s.w


def rope(x, cos, sin):                              # HF rotate_half = slice+neg+concat
    h = x.shape[-1] // 2
    rh = torch.cat((-x[..., h:], x[..., :h]), dim=-1)
    return x * cos + rh * sin


class PrefillLayer(nn.Module):
    """HF Llama 3.2 3B decoder layer (prefill, per-head 2D, Kᵀ reused per KV head)."""
    def __init__(s):
        super().__init__()
        mk = lambda i, o: nn.Linear(i, o, bias=False)
        s.n1 = RMSNorm(D); s.n2 = RMSNorm(D)
        s.wq = nn.ModuleList([mk(D, HD) for _ in range(H)])
        s.wo = nn.ModuleList([mk(HD, D) for _ in range(H)])
        s.wk = nn.ModuleList([mk(D, HD) for _ in range(KV)])
        s.wv = nn.ModuleList([mk(D, HD) for _ in range(KV)])
        s.g = mk(D, F); s.u = mk(D, F); s.dn = mk(F, D); s.act = nn.SiLU()
        for p in s.parameters():
            with torch.no_grad():
                p.mul_(0.02)

    def forward(s, x, cos, sin, mask):
        xn = s.n1(x)
        Kt, V = [], []
        for k in range(KV):
            kk = rope(s.wk[k](xn), cos, sin)
            Kt.append(kk.transpose(0, 1))           # [HD,S] — transpose once per KV head (×8)
            V.append(s.wv[k](xn))
        parts = []
        for hh in range(H):
            kv = hh // GPK
            q = rope(s.wq[hh](xn), cos, sin)
            sc = (q @ Kt[kv]) * (1.0 / math.sqrt(HD)) + mask
            parts.append(s.wo[hh](torch.softmax(sc, dim=-1) @ V[kv]))
        attn = parts[0]
        for p in parts[1:]:
            attn = attn + p
        hres = x + attn
        hn = s.n2(hres)
        return hres + s.dn(s.act(s.g(hn)) * s.u(hn))


class DecodeLayer(nn.Module):
    """1 query token + KV cache(length L). Kc/Vc: tuple of KV tensors [L,HD]."""
    def __init__(s):
        super().__init__()
        mk = lambda i, o: nn.Linear(i, o, bias=False)
        s.n1 = RMSNorm(D); s.n2 = RMSNorm(D)
        s.wq = nn.ModuleList([mk(D, HD) for _ in range(H)])
        s.wo = nn.ModuleList([mk(HD, D) for _ in range(H)])
        s.wk = nn.ModuleList([mk(D, HD) for _ in range(KV)])
        s.wv = nn.ModuleList([mk(D, HD) for _ in range(KV)])
        s.g = mk(D, F); s.u = mk(D, F); s.dn = mk(F, D); s.act = nn.SiLU()
        for p in s.parameters():
            with torch.no_grad():
                p.mul_(0.02)

    def forward(s, x, cos, sin, Kc, Vc):            # x[1,D], cos/sin[1,HD], Kc/Vc: tuple([L,HD])
        xn = s.n1(x)
        Kt = [Kc[k].transpose(0, 1) for k in range(KV)]   # [HD,L] — transpose cache once per KV head (×8)
        for k in range(KV):                          # new token's K/V (appended to cache)
            rope(s.wk[k](xn), cos, sin); s.wv[k](xn)
        parts = []
        for hh in range(H):
            kv = hh // GPK
            q = rope(s.wq[hh](xn), cos, sin)         # [1,HD]
            sc = (q @ Kt[kv]) * (1.0 / math.sqrt(HD))   # [1,L]
            parts.append(s.wo[hh](torch.softmax(sc, dim=-1) @ Vc[kv]))   # [1,HD]@... -> [1,D]
        attn = parts[0]
        for p in parts[1:]:
            attn = attn + p
        hres = x + attn
        hn = s.n2(hres)
        return hres + s.dn(s.act(s.g(hn)) * s.u(hn))


def mm_bucket(K, N):
    return {(3072, 128): "Q/K/V proj", (128, 3072): "O proj", (3072, 8192): "gate/up proj",
            (8192, 3072): "down proj", (128, 128): "attn matmul (scores+ctx)"}.get((K, N), f"matmul {K}x{N}")


def op_bucket(name, dshape, ashapes):
    if name == "relax.matmul":
        K = ashapes[0][1]; N = ashapes[1][1]
        return mm_bucket(K, N), "attn" if (K, N) in [(128, 128), (128, 3072)] else (
            "ffn" if N in (8192,) or (ashapes[0][1] == 8192) else "attn")
    if name == "relax.permute_dims":
        return "K^T transpose", "attn"
    if name in ("relax.strided_slice", "relax.concat"):
        return "RoPE (slice/concat)", "attn"
    if name == "relax.sum":
        return "reduce (norm/softmax)", "norm"
    if name == "relax.broadcast_to":
        return "broadcast (norm/softmax)", "norm"
    # elementwise: bucket by last-dim width
    w = dshape[-1] if dshape else 0
    if w == D:
        return "elementwise D (norm/resid)", "norm"
    if w == F:
        return "elementwise F (SiLU/FFN)", "ffn"
    return "elementwise (attn/softmax)", "attn"


def run_mode(tag, module, example_inputs, ntokens):
    print(f"\n========== HF {tag} ==========")
    mod = frontend.import_torch(module, example_inputs)
    mp = memplan.plan(mod["main"])
    log = []
    asm = codegen.compile_func(mod["main"], mp, tile=64, mm_backend="tir", emit_log=log)
    st = cost.analyze(asm, mp)
    tags = asm.tags
    # per-OP 집계 (op_bucket으로 그룹)
    ops = {}
    for name, dshape, ashapes, s0, s1 in log:
        label, cls = op_bucket(name, dshape, ashapes)
        roles = {}
        for t in tags[s0:s1]:
            roles[t or "untagged"] = roles.get(t or "untagged", 0) + 1
        e = ops.setdefault(label, dict(label=label, cls=cls, count=0, total=0, roles={}))
        e["count"] += 1; e["total"] += (s1 - s0)
        for k, v in roles.items():
            e["roles"][k] = e["roles"].get(k, 0) + v
    layer_total = st["total"]
    by_role = st["by_role"]; g = lambda k: by_role.get(k, 0)
    useful = g("mmul") + g("accum")
    silu = ops.get("elementwise F (SiLU/FFN)", {}).get("total", 0)
    n_tr = g("transpose") / 32768.0
    n_gs = (g("gather") + g("scatter") + g("pad")) / 512.0
    feats = [("strided load/save", g("gather") + g("scatter") + g("pad"), n_gs * 2),
             ("transpose unit", g("transpose"), n_tr * 7),
             ("row-reduce(sum)", g("reduce"), g("reduce") * 0.5),
             ("broadcast", g("broadcast"), g("broadcast") * 0.5),
             ("native activation", silu, silu * 0.2)]
    savings = [(n, up, max(0.0, up - rep)) for (n, up, rep) in feats]
    real_total = sum(s[2] for s in savings)
    print(f"  총 명령 = {layer_total:,}  (토큰당 {layer_total/ntokens:,.0f})")
    print(f"  유효(matmul+accum) {useful:,} ({100*useful/layer_total:.1f}%)  gather {g('gather'):,} ({100*g('gather')/layer_total:.1f}%)")
    print("  role:", {k: f"{100*v/layer_total:.1f}%" for k, v in sorted(by_role.items(), key=lambda x: -x[1])})
    print(f"  ISA 5종 현실 절감 {100*real_total/layer_total:.1f}% -> 남음 {layer_total-int(real_total):,}")
    rows = [(o["label"], o["count"], o["total"] // max(o["count"], 1), o["total"], o["roles"], o["cls"])
            for o in ops.values()]
    _graphs(rows, by_role, savings, layer_total, useful, f"hf_{tag}")
    return dict(tag=tag, layer_total=layer_total, by_role=by_role, useful=useful,
                per_token=layer_total / ntokens, rows=rows, savings=savings)


def _graphs(rows, by_role, savings, layer_total, useful, tag):
    # G1 tier
    tiers = [(">1M", 1e6, float("inf")), ("10K-1M", 1e4, 1e6), ("<10K", 0, 1e4)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, (tn, lo, hi) in zip(axes, tiers):
        grp = sorted([r for r in rows if lo <= r[3] < hi], key=lambda r: -r[3])
        bottoms = np.zeros(len(grp)); labels = [r[0] for r in grp]
        for role in ROLE_ORDER:
            vals = np.array([r[4].get(role, 0) for r in grp], float)
            if vals.sum() == 0:
                continue
            ax.bar(labels, vals, bottom=bottoms, color=ROLE_COLORS.get(role)); bottoms += vals
        ax.set_title(f"{tn} cmds/OP"); ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
    axes[0].set_ylabel("commands (linear)")
    present = [r for r in ROLE_ORDER if any(row[4].get(r, 0) for row in rows)]
    fig.legend(handles=[Patch(color=ROLE_COLORS.get(r), label=ROLE_EN.get(r, r)) for r in present],
               loc="upper center", ncol=len(present), fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"G1[{tag}]. Commands per OP — LINEAR by magnitude (real HF graph)", y=0.96)
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g1_per_op_{tag}.png", dpi=120, bbox_inches="tight"); plt.close()

    # G2 role dist
    items = sorted(by_role.items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([ROLE_EN.get(k, k) for k, _ in items], [v for _, v in items],
            color=[ROLE_COLORS.get(k, "#888") for k, _ in items])
    for i, (k, v) in enumerate(items):
        ax.text(v, i, f" {v:,} ({100*v/layer_total:.1f}%)", va="center", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("commands"); ax.set_title(f"G2[{tag}]. Role distribution (real HF graph)")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g2_role_dist_{tag}.png", dpi=120); plt.close()

    # G3 realistic waterfall
    stages = ["baseline"] + [s[0] for s in savings]; vals = [float(layer_total)]; cur = float(layer_total)
    for _n, _u, real in savings:
        cur -= real; vals.append(cur)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(range(len(stages)), vals, color=["#1f77b4"] + ["#2ca02c"] * len(savings))
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        if i > 0:
            ax.text(i, v + layer_total * 0.04, f"-{100*(vals[i-1]-v)/layer_total:.1f}%", ha="center", va="bottom", fontsize=8, color="#2ca02c")
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_ylabel("remaining commands"); ax.set_ylim(0, layer_total * 1.12)
    ax.set_title(f"G3[{tag}]. Realistic ISA savings: {layer_total:,} -> {vals[-1]:,.0f} (-{100*(1-vals[-1]/layer_total):.1f}%)")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g3_isa_waterfall_{tag}.png", dpi=120); plt.close()

    # G4 useful vs overhead
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    oh_tot = layer_total - useful
    ax1.pie([useful, oh_tot], labels=[f"useful\n{useful:,}", f"overhead\n{oh_tot:,}"], autopct="%1.1f%%",
            colors=["#2ca02c", "#d62728"], startangle=90); ax1.set_title(f"G4a[{tag}]. Useful vs overhead")
    oh = {ROLE_EN.get(k, k): v for k, v in by_role.items() if k not in ("mmul", "accum")}
    oh = dict(sorted(oh.items(), key=lambda x: -x[1]))
    ax2.bar(list(oh.keys()), list(oh.values()), color="#d62728"); ax2.set_yscale("log")
    ax2.set_title(f"G4b[{tag}]. Overhead breakdown"); plt.xticks(rotation=35, ha="right")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g4_useful_vs_overhead_{tag}.png", dpi=120); plt.close()

    # G5 share + mix
    rr = list(reversed(sorted(rows, key=lambda r: -r[3]))); labels = [r[0] for r in rr]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    shares = [100 * r[3] / layer_total for r in rr]
    ax1.barh(labels, shares, color="#1f77b4")
    for i, v in enumerate(shares):
        ax1.text(v, i, f" {v:.1f}% ({rr[i][3]:,})", va="center", fontsize=8)
    ax1.set_xlabel("% of layer"); ax1.set_title(f"G5a[{tag}]. Each OP share"); ax1.set_xlim(0, max(shares) * 1.35)
    bottoms = np.zeros(len(rr))
    for role in ROLE_ORDER:
        fr = np.array([100 * r[4].get(role, 0) / r[3] for r in rr], float)
        if fr.sum() == 0:
            continue
        ax2.barh(labels, fr, left=bottoms, color=ROLE_COLORS.get(role), label=ROLE_EN.get(role, role)); bottoms += fr
    ax2.set_xlim(0, 100); ax2.set_xlabel("% within OP"); ax2.set_title(f"G5b[{tag}]. Role mix per OP (100%)")
    ax2.legend(fontsize=7, ncol=2, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g5_share_and_mix_{tag}.png", dpi=120); plt.close()


def compare_graph(pf, dc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    names = ["prefill (S=128)", "decode (M=1)"]; pt = [pf["per_token"], dc["per_token"]]
    ax1.bar(names, pt, color=["#1f77b4", "#d62728"]); ax1.set_yscale("log")
    for i, v in enumerate(pt):
        ax1.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("commands PER TOKEN"); ax1.set_title(f"G6. Commands/token ({pt[1]/pt[0]:.0f}x for decode) [real HF]")
    uf = [100 * pf["useful"] / pf["layer_total"], 100 * dc["useful"] / dc["layer_total"]]
    ax2.bar(names, uf, color="#2ca02c")
    for i, v in enumerate(uf):
        ax2.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("useful %"); ax2.set_title("G6b. Useful-compute share")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g6_prefill_vs_decode_hf.png", dpi=120); plt.close()


def main():
    S, L = 128, 128
    cos, sin, _ = LG.rope_tables(S, HD, base=500000.0, llama3_scaling=True)
    # prefill
    xt = torch.randn(S, D) * 0.3
    ct = torch.tensor(cos, dtype=torch.float32); stt = torch.tensor(sin, dtype=torch.float32)
    mk = torch.tensor(LG.causal_mask(S), dtype=torch.float32)
    pf = run_mode("prefill", PrefillLayer().eval(), (xt, ct, stt, mk), ntokens=S)
    # decode
    cos1, sin1, _ = LG.rope_tables(1, HD, base=500000.0, llama3_scaling=True)
    xd = torch.randn(1, D) * 0.3
    c1 = torch.tensor(cos1, dtype=torch.float32); s1 = torch.tensor(sin1, dtype=torch.float32)
    Kc = tuple(torch.randn(L, HD) * 0.1 for _ in range(KV))
    Vc = tuple(torch.randn(L, HD) * 0.1 for _ in range(KV))
    dc = run_mode("decode", DecodeLayer().eval(), (xd, c1, s1, Kc, Vc), ntokens=1)
    compare_graph(pf, dc)
    print(f"\n== HF prefill vs decode ==  per-token: {pf['per_token']:,.0f} vs {dc['per_token']:,.0f} "
          f"({dc['per_token']/pf['per_token']:.0f}x);  useful {100*pf['useful']/pf['layer_total']:.1f}% vs "
          f"{100*dc['useful']/dc['layer_total']:.1f}%")
    print(f"figs -> {FIGDIR}")


if __name__ == "__main__":
    main()
