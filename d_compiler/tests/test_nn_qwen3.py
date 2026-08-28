"""S7 groundwork: the Qwen3 frontend must match a numpy reference, and its
one structural difference from Llama -- per-head q/k normalization -- must
survive the NPU path.
"""
import sys
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler import npu_legalize, npu_link, npu_memplan as M
from npu_compiler import tvm_pipeline as P
from npu_compiler.nn_models import qwen3

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=32, rms_norm_eps=1e-6, rope_theta=1000000.0)
SEQ = 5


def _reference(cfg, weights, x, cos, sin, mask):
    """Plain float32 numpy Qwen3 prefill, independent of TVM."""
    def rms(v, w):
        v = v.astype(np.float32)
        return v / np.sqrt((v * v).mean(-1, keepdims=True) + cfg.rms_eps) \
            * w.astype(np.float32)

    def rot_half(v, hd):
        half = hd // 2
        return np.concatenate([-v[..., half:], v[..., :half]], axis=-1)

    seq = x.shape[0]
    hd, heads, kv = cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    cos32, sin32 = cos.astype(np.float32), sin.astype(np.float32)
    hidden = x.astype(np.float32)
    for layer in range(cfg.num_layers):
        w = lambda k: weights[f"layers.{layer}.{k}"].astype(np.float32)
        n1 = rms(hidden, weights[f"layers.{layer}.input_layernorm.weight"])
        q = (n1 @ w("self_attn.q_proj.weight").T).reshape(seq, heads, hd)
        k = (n1 @ w("self_attn.k_proj.weight").T).reshape(seq, kv, hd)
        v = (n1 @ w("self_attn.v_proj.weight").T).reshape(seq, kv, hd)
        q = rms(q, w("self_attn.q_norm.weight"))
        k = rms(k, w("self_attn.k_norm.weight"))
        q = q * cos32 + rot_half(q, hd) * sin32
        k = k * cos32 + rot_half(k, hd) * sin32
        k = np.repeat(k, heads // kv, axis=1)
        v = np.repeat(v, heads // kv, axis=1)
        q, k, v = (t.transpose(1, 0, 2) for t in (q, k, v))
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask.astype(np.float32)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        attn = (probs @ v).transpose(1, 0, 2).reshape(seq, heads * hd)
        hidden = hidden + attn @ w("self_attn.o_proj.weight").T
        n2 = rms(hidden, weights[f"layers.{layer}.post_attention_layernorm.weight"])
        gate = n2 @ w("mlp.gate_proj.weight").T
        up = n2 @ w("mlp.up_proj.weight").T
        hidden = hidden + (gate / (1 + np.exp(-gate)) * up) \
            @ w("mlp.down_proj.weight").T
    final = rms(hidden, weights["norm.weight"])
    return final @ weights["lm_head.weight"].astype(np.float32).T


def _inputs(cfg, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.5, (SEQ, cfg.hidden_size)).astype(np.float16)
    cos, sin = qwen3.rope_inputs(cfg, np.arange(SEQ))
    return rng, x, cos, sin, qwen3.causal_mask(cfg.num_heads, SEQ)


def test_qwen3_frontend_matches_numpy_reference():
    mod, params, cfg = qwen3.build_prefill(TINY, seq=SEQ)
    names = [name for name, _ in params]
    assert "layers.0.self_attn.q_norm.weight" in names
    assert "layers.0.self_attn.k_norm.weight" in names

    rng, x, cos, sin, mask = _inputs(cfg, 0)
    weights = {n: rng.normal(0, 0.15, p.shape).astype(np.float16)
               for n, p in params}
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    got = vm["prefill"](*[tvm.nd.array(v) for v in (x, cos, sin, mask)],
                        [tvm.nd.array(weights[n]) for n, _ in params]).numpy()

    ref = _reference(cfg, weights, x, cos, sin, mask)
    a, b = got.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine > 0.9999, cosine
    assert int(np.argmax(a)) == int(np.argmax(b))
    print(f"  [PASS] qwen3 frontend vs numpy reference: cosine {cosine:.6f}")


def test_qwen3_layer_runs_on_the_cmodel():
    cfg_dict = dict(TINY, num_hidden_layers=1)
    mod, params, cfg = qwen3.build_prefill(cfg_dict, seq=SEQ)
    rng, x, cos, sin, mask = _inputs(cfg, 3)
    weights = {n: rng.normal(0, 0.15, p.shape).astype(np.float16)
               for n, p in params}
    exact = _reference(cfg, weights, x, cos, sin, mask)

    lowered = P.graph_pipeline(custom_legalize=npu_legalize.legalize_map(),
                              fuse=False, lift_params=True)(mod)
    vm = relax.VirtualMachine(relax.build(lowered, "llvm"), tvm.cpu())
    transformed = vm["prefill_transform_params"](
        [[tvm.nd.array(weights[n]) for n, _ in params]])

    asm, plan = npu_link.compile_program(lowered)
    planned, _ = M.assign_addresses(lowered)
    func = planned["prefill"]
    values = dict(zip([p.name_hint for p in func.params],
                      [x, cos, sin, mask] + [t.numpy() for t in transformed]))
    got, _ = npu_link.run_program(asm, plan, func, values, exact.shape)

    a, b = got.astype(np.float64).ravel(), exact.astype(np.float64).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine > 0.9999, cosine
    assert int(np.argmax(got[-1])) == int(np.argmax(exact[-1]))
    print(f"  [PASS] qwen3 layer on the c-model vs float32: cosine {cosine:.6f} "
          f"({len(asm.words):,} words)")


if __name__ == "__main__":
    test_qwen3_frontend_matches_numpy_reference()
    test_qwen3_layer_runs_on_the_cmodel()
    print("ALL QWEN3 FRONTEND TESTS PASSED")
