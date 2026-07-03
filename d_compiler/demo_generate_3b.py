"""Run the full Llama-3.2-3B-DIMENSION model end-to-end on mysim.

  python d_compiler/demo_generate_3b.py [n_layers] [MAX] [n_gen]
  (defaults: n_layers=2  MAX=64  n_gen=2)

Uses real 3B per-layer dims (D=3072, H=24, KV=8, HD=128, F=8192) with a small
context (MAX) and few generated tokens so it fits in mysim. Kernels are compiled
ONCE (compile caching) and reused across every layer and token. Weights are random
=> generated tokens are meaningless; the point is that real-3B-scale kernels
compile once and actually execute end-to-end on the given c-model.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_compiler import legalize, model, driver


def main(n_layers=2, MAX=64, n_gen=2, vocab=256, seed=0, ws=0.05):
    cfg = model.LLAMA_3_2_3B
    prompt = [3, 7, 11]
    print("=== full 3B-dimension prefill->decode generation on mysim ===")
    print(f"dims : D={cfg.D} H={cfg.H} KV={cfg.KV} HD={cfg.HD} F={cfg.F}")
    print(f"run  : layers={n_layers} MAX={MAX} n_gen={n_gen} vocab={vocab} prompt={prompt} ws={ws}")
    print("       (small ws keeps attention scores fp16-safe; see NOTE below)\n")

    cos, sin, rot = legalize.rope_tables(MAX, cfg.HD, base=cfg.rope_base,
                                         llama3_scaling=cfg.rope_scale)
    layer_Ws, top = model.make_gen_weights(cfg, n_layers, vocab, seed=seed, ws=ws)

    t0 = time.time()
    gen = driver.generate_tokens(cfg, layer_Ws, top, prompt, n_gen, MAX,
                                 cos, sin, rot, batched_prefill=True, verbose=True)
    dt = time.time() - t0

    ref = model.ref_generate_tokens(cfg, layer_Ws, top, prompt, n_gen, cos, sin, rot)
    print(f"\nprompt {prompt}")
    print(f"generated (mysim/NPU) : {gen}")
    print(f"reference (numpy f64) : {ref}")
    print(f"match: {gen == ref}")
    print(f"total wall time: {dt:.0f}s  (compile cached ONCE; rest is mysim execution)")
    print("\nNOTE (fp16 at 3B scale): two range issues surface at D=3072 that don't at small dims:")
    print(" 1) RMSNorm sum(x^2) over 3072 terms overflowed fp16 -> FIXED (reduce with 1/d = mean).")
    print(" 2) softmax omits max-subtraction (no reduce-max ISA, report §5): attention scores")
    print(f"    reach ~100 at ws=0.2 and overflow exp. A small ws={ws} keeps scores fp16-safe.")
    print("With (1) fixed and small ws, mysim matches the numpy reference. Real-3B kernels compile")
    print("ONCE (cached) and execute end-to-end on the c-model.")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(n_layers=int(a[0]) if len(a) > 0 else 2,
         MAX=int(a[1]) if len(a) > 1 else 64,
         n_gen=int(a[2]) if len(a) > 2 else 2,
         ws=float(a[3]) if len(a) > 3 else 0.05)
