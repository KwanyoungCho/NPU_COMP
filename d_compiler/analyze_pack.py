"""Measure the effect of INPUT WEIGHT PACKING on the real 3B generation kernels.

Compiles each generation kernel (kv_proj / attn_ffn / prefill layer / lm_head) at
3B dims with pack_params OFF vs ON and reports total commands + role breakdown.
Compile-only (no mysim). Dumps JSON for the graph script (make_pack_figs.py).
"""
import sys, json, time
sys.path.insert(0, ".")
from npu_compiler import model, memplan, codegen, cost

cfg = model.LLAMA_3_2_3B
MAX, S, VOCAB = 128, 128, 128256


def measure(mod, pack):
    mp = memplan.plan(mod["main"], pack_params=pack)
    asm = codegen.compile_func(mod["main"], mp, tile=64, mm_backend="tir", fuse_oproj=True)
    st = cost.analyze(asm, mp)
    return st["total"], dict(st["by_role"])


KERNELS = [("kv_proj", lambda: model.build_kv_proj_module(cfg)),
           ("attn_ffn", lambda: model.build_attn_ffn_module(cfg, MAX)),
           ("prefill", lambda: model.build_prefill_layer_module(cfg, S)),
           ("lm_head", lambda: model.build_lm_head_module(cfg, VOCAB))]

if __name__ == "__main__":
    print(f"3B kernels: D={cfg.D} H={cfg.H} KV={cfg.KV} HD={cfg.HD} F={cfg.F} | MAX={MAX} S={S} vocab={VOCAB}")
    print("INPUT WEIGHT PACKING: OFF -> ON\n")
    res = {}
    u = lambda r: r.get("mmul", 0) + r.get("accum", 0)
    for name, build in KERNELS:
        t = time.time()
        off_t, off_r = measure(build(), False)
        on_t, on_r = measure(build(), True)
        res[name] = {"off": {"total": off_t, "roles": off_r}, "on": {"total": on_t, "roles": on_r}}
        print(f"{name:9s} total {off_t:>12,} -> {on_t:>12,} ({100*(on_t-off_t)/off_t:+5.1f}%)  "
              f"gather {100*off_r.get('gather',0)/off_t:3.0f}%->{100*on_r.get('gather',0)/on_t:2.0f}%  "
              f"useful {100*u(off_r)/off_t:3.0f}%->{100*u(on_r)/on_t:2.0f}%   [{time.time()-t:.0f}s]")

    # token cost (28 decode layers + 1 lm_head)
    def step(res, k):  # kv_proj + attn_ffn
        return res["kv_proj"][k]["total"] + res["attn_ffn"][k]["total"]
    for k, lab in [("off", "OFF"), ("on", "ON ")]:
        tok = 28 * step(res, k) + res["lm_head"][k]["total"]
        print(f"\n[{lab}] token cost = 28*(kv+attn) + lm_head = {tok:,}")
    tok_off = 28 * step(res, "off") + res["lm_head"]["off"]["total"]
    tok_on = 28 * step(res, "on") + res["lm_head"]["on"]["total"]
    print(f"=> token cost {tok_off:,} -> {tok_on:,} ({100*(tok_on-tok_off)/tok_off:+.1f}%)")
    res["_token"] = {"off": tok_off, "on": tok_on}
    json.dump(res, open("/tmp/pack_results.json", "w"))
    print("\nsaved /tmp/pack_results.json")
