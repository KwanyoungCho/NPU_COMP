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
plt.rcParams.update({                          # 글씨 최소 크기 키우기
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 13,
})
import torch, torch.nn as nn

from npu_compiler import frontend, codegen, memplan, cost, legalize as LG
from analyze_layer import ROLE_ORDER, ROLE_EN, ROLE_COLORS

D, H, KV, HD, F = 3072, 24, 8, 128, 8192
GPK = H // KV
FIGDIR = os.path.join(os.path.dirname(HERE), "report", "figs")
os.makedirs(FIGDIR, exist_ok=True)
ROLES_VIS = [r for r in ROLE_ORDER if r != "untagged"]   # "other"(halt 1개) 그래프에서 제외


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


def analyze_once(mod, ntokens, pack, reuse=True, fuse=True):
    mp = memplan.plan(mod["main"], pack=pack)
    log = []
    asm = codegen.compile_func(mod["main"], mp, tile=64, mm_backend="tir", emit_log=log,
                               reuse_act=reuse, fuse_oproj=fuse)
    st = cost.analyze(asm, mp)
    tags = asm.tags
    ops = {}
    for name, dshape, ashapes, s0, s1 in log:
        label, cls = op_bucket(name, dshape, ashapes)
        e = ops.setdefault(label, dict(label=label, cls=cls, count=0, total=0, roles={}))
        e["count"] += 1; e["total"] += (s1 - s0)
        for t in tags[s0:s1]:
            e["roles"][t or "untagged"] = e["roles"].get(t or "untagged", 0) + 1
    layer_total = st["total"]; by_role = st["by_role"]; g = lambda k: by_role.get(k, 0)
    useful = g("mmul") + g("accum")
    silu = ops.get("elementwise F (SiLU/FFN)", {}).get("total", 0)
    feats = [("strided load/save", g("gather") + g("scatter") + g("pad"), (g("gather")+g("scatter")+g("pad"))/512.0*2),
             ("transpose unit", g("transpose"), g("transpose")/32768.0*7),
             ("m_mul accumulate", g("accum"), 0.0),       # C+=A@B 누산기로 K-accum 흡수(완전 제거)
             ("row-reduce(sum)", g("reduce"), g("reduce")*0.5),
             ("broadcast", g("broadcast"), g("broadcast")*0.5),
             ("native activation", silu, silu*0.2)]
    savings = [(n, up, max(0.0, up - rep)) for (n, up, rep) in feats]
    rows = [(o["label"], o["count"], o["total"]//max(o["count"],1), o["total"], o["roles"], o["cls"])
            for o in ops.values()]
    return dict(layer_total=layer_total, by_role=by_role, useful=useful,
                per_token=layer_total/ntokens, rows=rows, savings=savings)


def run_mode(tag, module, example_inputs, ntokens):
    print(f"\n========== HF {tag} ==========")
    mod = frontend.import_torch(module, example_inputs)
    base = analyze_once(mod, ntokens, pack=False, reuse=False, fuse=False)  # 진짜 baseline
    packed = analyze_once(mod, ntokens, pack=True, reuse=False, fuse=False)  # +가중치 패킹
    reused = analyze_once(mod, ntokens, pack=True, reuse=True, fuse=False)   # +활성화 gather 재사용
    best = analyze_once(mod, ntokens, pack=True, reuse=True, fuse=True)      # +O-proj head 융합 (현재 기본)
    for nm, st in [("baseline   ", base), ("+pack      ", packed),
                   ("+pack+reuse", reused), ("+oproj fuse", best)]:
        g = lambda k: st["by_role"].get(k, 0)
        print(f"  [{nm}] 총={st['layer_total']:>10,} (토큰당 {st['per_token']:>12,.0f})  "
              f"유효 {100*st['useful']/st['layer_total']:4.1f}%  gather {100*g('gather')/st['layer_total']:4.1f}%"
              f"  scatter {100*g('scatter')/st['layer_total']:4.1f}%")
    # 상세 그래프는 현재 기본(=+pack+reuse+fuse) 기준
    _graphs(best["rows"], best["by_role"], best["savings"], best["layer_total"], best["useful"], f"hf_{tag}")
    return base, packed, reused, best


def _graphs(rows, by_role, savings, layer_total, useful, tag):
    # G1 tier (transpose(>1M)을 두 번째 tier와 합쳐 ≥10K 한 묶음으로)
    tiers = [("≥10K", 1e4, float("inf")), ("<10K", 0, 1e4)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (tn, lo, hi) in zip(axes, tiers):
        grp = sorted([r for r in rows if lo <= r[3] < hi], key=lambda r: -r[3])
        bottoms = np.zeros(len(grp)); labels = [r[0] for r in grp]
        for role in ROLES_VIS:
            vals = np.array([r[4].get(role, 0) for r in grp], float)
            if vals.sum() == 0:
                continue
            ax.bar(labels, vals, bottom=bottoms, color=ROLE_COLORS.get(role)); bottoms += vals
        ax.set_title(f"{tn} cmds/OP"); ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=12)
    axes[0].set_ylabel("commands (linear)")
    present = [r for r in ROLES_VIS if any(row[4].get(r, 0) for row in rows)]
    fig.legend(handles=[Patch(color=ROLE_COLORS.get(r), label=ROLE_EN.get(r, r)) for r in present],
               loc="lower center", ncol=min(len(present), 6), fontsize=13, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"G1[{tag}]. Commands per OP — LINEAR by magnitude (real HF graph)", y=0.99)
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g1_per_op_{tag}.png", dpi=120, bbox_inches="tight"); plt.close()

    # G2+G3 통합: (좌) role 분포 — 무엇이 오버헤드인가, (우) ISA waterfall — 그 role을 없애는 절감
    items = sorted([(k, v) for k, v in by_role.items() if k != "untagged"], key=lambda x: x[1])
    stages = ["baseline"] + [s[0] for s in savings]; wf = [float(layer_total)]; cur = float(layer_total)
    for _n, _u, real in savings:
        cur -= real; wf.append(cur)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
    # 좌: G2 role distribution (linear, 값+% 라벨이라 작은 role도 가독)
    ax1.barh([ROLE_EN.get(k, k) for k, _ in items], [v for _, v in items],
             color=[ROLE_COLORS.get(k, "#888") for k, _ in items])
    for i, (k, v) in enumerate(items):
        ax1.text(v, i, f" {v:,} ({100*v/layer_total:.1f}%)", va="center", fontsize=12)
    ax1.set_xlabel("commands"); ax1.set_xlim(0, max(v for _, v in items) * 1.22)
    ax1.set_title(f"G2[{tag}]. Role distribution (after SW opt)")
    # 우: G3 realistic ISA waterfall
    ax2.bar(range(len(stages)), wf, color=["#1f77b4"] + ["#2ca02c"] * len(savings))
    for i, v in enumerate(wf):
        ax2.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=11)
        if i > 0:
            ax2.text(i, v + layer_total * 0.04, f"-{100*(wf[i-1]-v)/layer_total:.1f}%",
                     ha="center", va="bottom", fontsize=11, color="#2ca02c")
    ax2.set_xticks(range(len(stages))); ax2.set_xticklabels(stages, rotation=20, ha="right")
    ax2.set_ylabel("remaining commands"); ax2.set_ylim(0, layer_total * 1.12)
    ax2.set_title(f"G3[{tag}]. Realistic ISA savings: {layer_total:,} -> {wf[-1]:,.0f} "
                  f"(-{100*(1-wf[-1]/layer_total):.1f}%)")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g23_role_and_isa_{tag}.png", dpi=120, bbox_inches="tight"); plt.close()

    # G4 useful vs overhead ("other" 제외) — G4b를 log + linear 두 버전
    oh_tot = layer_total - useful
    oh = {ROLE_EN.get(k, k): v for k, v in by_role.items() if k not in ("mmul", "accum", "untagged")}
    oh = dict(sorted(oh.items(), key=lambda x: -x[1]))
    for scale in ("log", "linear"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        ax1.pie([useful, oh_tot], labels=[f"useful\n{useful:,}", f"overhead\n{oh_tot:,}"], autopct="%1.1f%%",
                colors=["#2ca02c", "#d62728"], startangle=90); ax1.set_title(f"G4a[{tag}]. Useful vs overhead")
        ax2.bar(list(oh.keys()), list(oh.values()), color="#d62728"); ax2.set_yscale(scale)
        ax2.set_title(f"G4b[{tag}]. Overhead breakdown ({scale})"); plt.setp(ax2.get_xticklabels(), rotation=35, ha="right")
        sfx = "" if scale == "log" else "_linear"
        plt.tight_layout(); plt.savefig(f"{FIGDIR}/g4_useful_vs_overhead_{tag}{sfx}.png", dpi=120); plt.close()

    # G5 share + mix (보고서는 prefill만 사용 → prefill에서만 생성)
    if "prefill" in tag:
        rr = list(reversed(sorted(rows, key=lambda r: -r[3]))); labels = [r[0] for r in rr]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        shares = [100 * r[3] / layer_total for r in rr]
        ax1.barh(labels, shares, color="#1f77b4")
        for i, v in enumerate(shares):
            ax1.text(v, i, f" {v:.1f}% ({rr[i][3]:,})", va="center", fontsize=12)
        ax1.set_xlabel("% of layer"); ax1.set_title(f"G5a[{tag}]. Each OP share"); ax1.set_xlim(0, max(shares) * 1.35)
        bottoms = np.zeros(len(rr))
        for role in ROLES_VIS:
            fr = np.array([100 * r[4].get(role, 0) / r[3] for r in rr], float)
            if fr.sum() == 0:
                continue
            ax2.barh(labels, fr, left=bottoms, color=ROLE_COLORS.get(role), label=ROLE_EN.get(role, role)); bottoms += fr
        ax2.set_xlim(0, 100); ax2.set_xlabel("% within OP"); ax2.set_title(f"G5b[{tag}]. Role mix per OP (100%)")
        ax2.legend(fontsize=12, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10))
        plt.tight_layout(); plt.savefig(f"{FIGDIR}/g5_share_and_mix_{tag}.png", dpi=120, bbox_inches="tight"); plt.close()


def compare_graph(pf, dc):
    names = ["prefill (S=128)", "decode (M=1)"]; pt = [pf["per_token"], dc["per_token"]]
    uf = [100 * pf["useful"] / pf["layer_total"], 100 * dc["useful"] / dc["layer_total"]]
    for scale in ("log", "linear"):                       # G6a를 log + linear 두 버전
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        ax1.bar(names, pt, color=["#1f77b4", "#d62728"]); ax1.set_yscale(scale)
        for i, v in enumerate(pt):
            ax1.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=13)
        ax1.set_ylabel("commands PER TOKEN")
        ax1.set_title(f"G6. Commands/token ({pt[1]/pt[0]:.0f}x for decode, {scale}) [real HF]")
        ax2.bar(names, uf, color="#2ca02c")
        for i, v in enumerate(uf):
            ax2.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=13)
        ax2.set_ylabel("useful %"); ax2.set_title("G6b. Useful-compute share")
        sfx = "" if scale == "log" else "_linear"
        plt.tight_layout(); plt.savefig(f"{FIGDIR}/g6_prefill_vs_decode_hf{sfx}.png", dpi=120); plt.close()


def pack_compare_graph(pf, dc):
    """G7: SW 최적화 단계별 효과 — baseline → +pack → +act reuse → +o_proj fuse.
    pf/dc = (base, packed, reused, best)."""
    stages = ["baseline", "+weight\npack", "+act\nreuse", "+oproj\nfuse"]
    n = len(stages)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(n); w = 0.38
    for off, (lab, series, c) in [(-w/2, ("prefill", pf, "#1f77b4")), (w/2, ("decode", dc, "#d62728"))]:
        tot = [s["layer_total"] for s in series]
        ax1.bar(x + off, tot, w, label=lab, color=c)
        for i in range(n):
            ax1.text(x[i] + off, tot[i], f"{tot[i]/1e6:.1f}M", ha="center", va="bottom", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(stages); ax1.set_ylabel("total commands")
    ax1.set_title("G7a. SW optimization steps (total commands)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    for off, (lab, series, c) in [(-w/2, ("prefill", pf, "#1f77b4")), (w/2, ("decode", dc, "#d62728"))]:
        uf = [100*s["useful"]/s["layer_total"] for s in series]
        ax2.bar(x + off, uf, w, label=lab, color=c)
        for i in range(n):
            ax2.text(x[i] + off, uf[i], f"{uf[i]:.1f}%", ha="center", va="bottom", fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(stages); ax2.set_ylabel("useful (matmul+accum) %")
    ax2.set_title("G7b. Useful-compute share")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/g7_packing_effect.png", dpi=120, bbox_inches="tight"); plt.close()


def main():
    S, L = 128, 128
    cos, sin, _ = LG.rope_tables(S, HD, base=500000.0, llama3_scaling=True)
    xt = torch.randn(S, D) * 0.3
    ct = torch.tensor(cos, dtype=torch.float32); stt = torch.tensor(sin, dtype=torch.float32)
    mk = torch.tensor(LG.causal_mask(S), dtype=torch.float32)
    pf = run_mode("prefill", PrefillLayer().eval(), (xt, ct, stt, mk), ntokens=S)   # (base,pack,reuse,best)
    cos1, sin1, _ = LG.rope_tables(1, HD, base=500000.0, llama3_scaling=True)
    xd = torch.randn(1, D) * 0.3
    c1 = torch.tensor(cos1, dtype=torch.float32); s1 = torch.tensor(sin1, dtype=torch.float32)
    Kc = tuple(torch.randn(L, HD) * 0.1 for _ in range(KV))
    Vc = tuple(torch.randn(L, HD) * 0.1 for _ in range(KV))
    dc = run_mode("decode", DecodeLayer().eval(), (xd, c1, s1, Kc, Vc), ntokens=1)
    compare_graph(pf[-1], dc[-1])         # G6: final(=+pack+reuse+fuse)
    pack_compare_graph(pf, dc)            # G7: baseline→+pack→+reuse→+oproj fuse 4단계
    print("\n== SW 최적화 단계별 (baseline → +pack → +reuse → +oproj fuse) ==")
    for nm, t in [("prefill", pf), ("decode", dc)]:
        b, p, r, f = t
        tot = lambda s: s['layer_total']
        print(f"  {nm}: {tot(b):,} → +pack {tot(p):,} (-{100*(1-tot(p)/tot(b)):.1f}%)"
              f" → +reuse {tot(r):,} (-{100*(1-tot(r)/tot(b)):.1f}%)"
              f" → +fuse {tot(f):,} (전체 -{100*(1-tot(f)/tot(b)):.1f}%); 유효 "
              f"{100*b['useful']/tot(b):.1f}%→{100*f['useful']/tot(f):.1f}%")
    print("\n== final(+oproj fuse) per-OP (prefill) ==")
    for r in sorted(pf[-1]["rows"], key=lambda r: -r[3]):
        print(f"  {r[0]:<26} {r[3]:>10,}  ({100*r[3]/pf[-1]['layer_total']:.1f}%)")
    print("\n== final(+oproj fuse) per-OP (decode) ==")
    for r in sorted(dc[-1]["rows"], key=lambda r: -r[3]):
        print(f"  {r[0]:<26} {r[3]:>10,}  ({100*r[3]/dc[-1]['layer_total']:.1f}%)")
    print(f"\nfigs -> {FIGDIR}")


if __name__ == "__main__":
    main()
