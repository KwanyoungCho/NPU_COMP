"""Fine-grained overhead analysis of the ACTUAL implemented kernels at 3B dims.

Unlike analyze_hf.py (which analyzes a naive per-step-transpose decode), this
compiles the real generation kernels (kv_proj / attn_ffn / prefill layer / lm_head)
exactly as driver.generate_tokens runs them (packed, fused, reuse, transposed
cache) and breaks down commands by role and by op. Compile-only (no mysim run).
"""
import sys, time
sys.path.insert(0, ".")
from collections import Counter
import numpy as np
from npu_compiler import model, memplan, codegen, cost

cfg = model.LLAMA_3_2_3B
MAX, S, VOCAB = 128, 128, 128256          # context, prompt len, real Llama vocab

USEFUL = {"mmul", "accum"}
ROLE_ORDER = ["mmul", "accum", "gather", "scatter", "transpose", "reduce",
              "broadcast", "layout", "pad", "elementwise", "untagged"]


def op_bucket(name, dsh, ash):
    if name == "relax.matmul":
        K, N = int(ash[0][1]), int(ash[1][1])
        m = {(3072, 128): "Q/K/V proj", (128, 3072): "O proj",
             (3072, 8192): "gate/up proj", (8192, 3072): "down proj",
             (3072, VOCAB): "lm_head"}
        if (K, N) in m:
            return m[(K, N)]
        if K == 128 or N == 128:
            return "attn score/ctx"
        return f"matmul {K}x{N}"
    return {"relax.permute_dims": "Kᵀ transpose", "relax.sum": "reduce",
            "relax.broadcast_to": "broadcast", "relax.concat": "concat"}.get(
        name, f"ew:{name.split('.')[-1]}")


def analyze_kernel(tag, mod):
    log = []
    t = time.time()
    mp = memplan.plan(mod["main"])                     # pack=True
    asm = codegen.compile_func(mod["main"], mp, tile=64, mm_backend="tir",
                               emit_log=log, fuse_oproj=True)
    st = cost.analyze(asm, mp)
    dt = time.time() - t
    total = st["total"]; roles = st["by_role"]
    tags = asm.tags
    ops = {}
    for name, dsh, ash, s0, s1 in log:
        lab = op_bucket(name, dsh, ash)
        e = ops.setdefault(lab, [0, Counter()])
        e[0] += (s1 - s0)
        for tg in tags[s0:s1]:
            e[1][tg or "untagged"] += 1
    useful = sum(roles.get(r, 0) for r in USEFUL)
    print(f"\n===== {tag}  (total={total:,} cmds, compile {dt:.0f}s, "
          f"G-buf {st['gbuffer_elems']*2/1e6:.0f}MB) =====")
    print(f"  useful(mmul+accum) {100*useful/total:4.1f}%   overhead {100*(total-useful)/total:4.1f}%")
    print("  by ROLE:")
    for r in ROLE_ORDER:
        v = roles.get(r, 0)
        if v:
            print(f"    {r:12s} {v:>10,}  {100*v/total:5.1f}%")
    print("  by OP (top):")
    for lab, (c, rc) in sorted(ops.items(), key=lambda kv: -kv[1][0])[:9]:
        rb = " ".join(f"{k}={v//1000}k" if v >= 1000 else f"{k}={v}"
                      for k, v in rc.most_common(3))
        print(f"    {lab:22s} {c:>10,}  {100*c/total:5.1f}%   [{rb}]")
    return total, useful, roles


if __name__ == "__main__":
    print(f"Llama 3.2 3B kernels: D={cfg.D} H={cfg.H} KV={cfg.KV} HD={cfg.HD} "
          f"F={cfg.F} | MAX={MAX} S={S} vocab={VOCAB}")
    kv = analyze_kernel("decode kv_proj", model.build_kv_proj_module(cfg))
    an = analyze_kernel(f"decode attn_ffn (MAX={MAX})", model.build_attn_ffn_module(cfg, MAX))
    pf = analyze_kernel(f"prefill layer (S={S})", model.build_prefill_layer_module(cfg, S))
    lm = analyze_kernel(f"lm_head (vocab={VOCAB})", model.build_lm_head_module(cfg, VOCAB))

    step = kv[0] + an[0]
    step_u = kv[1] + an[1]
    print("\n" + "=" * 64)
    print(f"DECODE STEP / layer (kv_proj+attn_ffn) = {step:,} cmds, useful {100*step_u/step:.1f}%")
    print(f"  x{28} layers/token = {28*step:,} cmds  (+ lm_head {lm[0]:,} once/token)")
    print(f"PREFILL layer (batched, S={S}) = {pf[0]:,} cmds, useful {100*pf[1]/pf[0]:.1f}%")
