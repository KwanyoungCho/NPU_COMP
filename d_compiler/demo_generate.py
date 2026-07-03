"""Run the whole prefill->decode pipeline end-to-end and print generated tokens.

  python d_compiler/demo_generate.py

Small config (MEDIUM/REDUCED) so it actually runs in mysim. This exercises the
full path: [CPU] embedding -> [NPU] batched prefill (seeds KV cache) -> decode loop
(kv_proj + attn_ffn per layer) + lm_head -> [CPU] argmax -> repeat.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_compiler import legalize, model, driver


def run(cfg, MAX, n_layers, vocab, prompt, n_gen, batched_prefill=True, seed=0):
    cos, sin, rot = legalize.rope_tables(MAX, cfg.HD, base=cfg.rope_base,
                                         llama3_scaling=cfg.rope_scale)
    layer_Ws, top = model.make_gen_weights(cfg, n_layers, vocab, seed=seed)

    # compiled generation on mysim
    gen = driver.generate_tokens(cfg, layer_Ws, top, prompt, n_gen, MAX,
                                 cos, sin, rot, batched_prefill=batched_prefill)
    # numpy float64 reference (ground truth) for comparison
    ref = model.ref_generate_tokens(cfg, layer_Ws, top, prompt, n_gen, cos, sin, rot)

    print(f"[{cfg.name}] layers={n_layers} vocab={vocab} MAX={MAX} "
          f"batched_prefill={batched_prefill}")
    print(f"  prompt        : {prompt}")
    print(f"  generated(NPU): {gen}")
    print(f"  reference(np) : {ref}")
    print(f"  match         : {'YES' if gen == ref else 'NO'}\n")


if __name__ == "__main__":
    print("=== full prefill->decode token generation on mysim ===\n")
    run(model.MEDIUM, MAX=64, n_layers=2, vocab=64, prompt=[8, 8, 51], n_gen=5)
    run(model.REDUCED, MAX=16, n_layers=2, vocab=32, prompt=[4, 4, 25], n_gen=4)
