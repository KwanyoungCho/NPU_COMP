"""S0 gate (fast): the standard nn.Module frontend builds and computes correctly.

Uses a tiny config so the whole thing runs in seconds on llvm; the
real-checkpoint gate is run_nn_llama_cpu.py.
"""
import sys
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from npu_compiler.nn_models import llama

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0)
SEQ = 5


def _reference(cfg, weights, x, cos, sin, mask):
    """Plain float32 numpy Llama prefill, independent of TVM."""
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
    final = rms(hidden, weights["norm.weight"])[-1:]
    return final @ weights["lm_head.weight"].astype(np.float32).T


def test_nn_frontend_matches_numpy_reference():
    mod, params, cfg = llama.build_prefill(TINY, seq=SEQ)
    assert [gv.name_hint for gv in mod.functions] == ["prefill"]
    names = [name for name, _ in params]
    assert "layers.0.self_attn.q_proj.weight" in names
    assert "lm_head.weight" in names

    rng = np.random.default_rng(0)
    weights = {n: rng.normal(0, 0.15, p.shape).astype(np.float16) for n, p in params}
    x = rng.normal(0, 0.5, (SEQ, cfg.hidden_size)).astype(np.float16)
    cos, sin = llama.rope_inputs(cfg, np.arange(SEQ))
    mask = llama.causal_mask(cfg.num_heads, SEQ)

    # standard build path: relax.build -> VirtualMachine
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    got = vm["prefill"](*[tvm.nd.array(v) for v in (x, cos, sin, mask)],
                        [tvm.nd.array(weights[n]) for n, _ in params]).numpy()

    ref = _reference(cfg, weights, x, cos, sin, mask)
    a, b = got.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine > 0.9999, cosine
    assert int(np.argmax(a)) == int(np.argmax(b))
    print(f"  [PASS] nn frontend vs numpy reference: cosine {cosine:.6f}")


def test_rope_frequencies_match_existing_convention():
    """The nn model must use the same RoPE frequencies as the validated path."""
    from npu_compiler.legalize import rope_freqs_row
    for scaling in (False, True):
        ours = llama.rope_freqs(128, 500000.0, scaling)
        theirs = rope_freqs_row(128, 500000.0, scaling)[0]
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=0)
    print("  [PASS] RoPE frequencies identical to legalize.rope_freqs_row")


if __name__ == "__main__":
    test_rope_frequencies_match_existing_convention()
    test_nn_frontend_matches_numpy_reference()
    print("ALL NN FRONTEND (S0) TESTS PASSED")
