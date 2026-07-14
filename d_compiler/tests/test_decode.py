"""Prefill -> decode with a KV cache: correctness on small dims.

The real 3B can't run in mysim, so we validate the decode path on MEDIUM (all-64,
TIR path) and REDUCED (GQA, direct/padded path). Builders + numpy reference live in
npu_compiler.model; the host generation loop is npu_compiler.driver.generate.

  M1  numpy self-consistency : one big prefill == prefill(prompt)+decode(rest)
  M2  compiled kernels 1-step: kv_proj / attn_ffn each == numpy piece (via mysim)
  M3  compiled generation    : host prefill+decode loop == numpy layer (via mysim)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler import legalize, model, driver


def _f16(x):
    return np.asarray(x, np.float16)


def _tables(cfg, n):
    return legalize.rope_tables(n, cfg.HD, base=cfg.rope_base, llama3_scaling=cfg.rope_scale)


def test_m1_self_consistency():
    for cfg in (model.MEDIUM, model.REDUCED):
        err, rel = model.decode_self_consistency(cfg, n_prompt=cfg.SEQ, n_gen=5)
        assert rel < 1e-9, f"{cfg.name}: self-consistency rel={rel}"
    return "prefill-all == prefill+decode (float64)"


def _one_step(cfg, MAX, seed=0):
    cos, sin, rot = _tables(cfg, MAX)
    W = model.make_weights(cfg, seed=seed)
    rng = np.random.default_rng(seed + 1)
    X = _f16(rng.standard_normal((cfg.SEQ + 1, cfg.D))).astype(np.float64)
    cache = model.new_cache(cfg)
    model.ref_layer_cache(cfg, W, X[:cfg.SEQ], cache, 0, cos, sin, rot)   # cache len SEQ
    pos = cfg.SEQ; x_new = X[pos:pos + 1]; HD = cfg.HD

    # M2a kv_proj
    Ks, Vs = model.ref_kv_proj(cfg, W, x_new, np.array([pos]), cos, sin, rot)
    ins = {"x": _f16(x_new), "Wn1": W["Wn1"], "pos": _f16([[pos]])}
    for k in range(cfg.KV):
        ins[f"Wk{k}"] = W[f"Wk{k}"]; ins[f"Wv{k}"] = W[f"Wv{k}"]
    o1 = driver.run_module(model.build_kv_proj_module(cfg), ins, backend="hybrid")
    for k in range(cfg.KV):
        Kc = o1[:, 2 * k * HD:(2 * k + 1) * HD]; Vv = o1[:, (2 * k + 1) * HD:(2 * k + 2) * HD]
        sc = np.max(np.abs(Ks[k])) + np.max(np.abs(Vs[k])) + 1e-6
        assert np.max(np.abs(Kc - Ks[k])) < 0.05 * sc + 0.05, f"{cfg.name} kv_proj K head{k}"
        assert np.max(np.abs(Vv - Vs[k])) < 0.05 * sc + 0.05, f"{cfg.name} kv_proj V head{k}"

    # M2b attn_ffn (append new token, fp16-cast + pad cache, build mask)
    model.cache_append(cache, Ks, Vs)
    cache_f16 = model.new_cache(cfg); KtP, VcP = [], []
    for k in range(cfg.KV):
        Ktk = _f16(cache["Kt"][k]).astype(np.float64); Vck = _f16(cache["V"][k]).astype(np.float64)
        cache_f16["Kt"][k] = Ktk; cache_f16["V"][k] = Vck
        Kp = np.zeros((HD, MAX)); Kp[:, :pos + 1] = Ktk
        Vp = np.zeros((MAX, HD)); Vp[:pos + 1, :] = Vck
        KtP.append(_f16(Kp)); VcP.append(_f16(Vp))
    mask = np.zeros((1, MAX), np.float16); mask[0, pos + 1:] = driver.NEG
    y_ref = model.ref_attn_ffn(cfg, W, x_new, cache_f16, np.array([pos]), cos, sin, rot, stable_softmax=False)
    ins = {"x": _f16(x_new), "Wn1": W["Wn1"], "Wn2": W["Wn2"],
           "pos": _f16([[pos]]), "mask": mask,
           "Wg": W["Wg"], "Wu": W["Wu"], "Wd": W["Wd"]}
    for h in range(cfg.H):
        ins[f"Wq{h}"] = W[f"Wq{h}"]; ins[f"Wo{h}"] = W[f"Wo{h}"]
    for k in range(cfg.KV):
        ins[f"Kt{k}"] = KtP[k]; ins[f"Vc{k}"] = VcP[k]
    y = driver.run_module(model.build_attn_ffn_module(cfg, MAX), ins, backend="hybrid")
    sc = np.max(np.abs(y_ref)) + 1e-6
    assert np.max(np.abs(y - y_ref)) < 0.05 * sc + 0.05, f"{cfg.name} attn_ffn"


def test_m2_kernels_one_step():
    _one_step(model.MEDIUM, MAX=128)
    _one_step(model.REDUCED, MAX=16)
    return "kv_proj + attn_ffn == numpy (1 step, mysim)"


def _gen(cfg, MAX, n_prompt, n_gen, seed=0):
    N = n_prompt + n_gen
    cos, sin, rot = _tables(cfg, MAX)
    W = model.make_weights(cfg, seed=seed)
    rng = np.random.default_rng(seed + 1)
    X = _f16(rng.standard_normal((N, cfg.D))).astype(np.float64)
    cache = model.new_cache(cfg)
    Y_ref, _ = model.ref_layer_cache(cfg, W, X, cache, 0, cos, sin, rot)
    Y = driver.generate(cfg, W, X, MAX, cos, sin, rot)
    rel = np.max(np.abs(Y - Y_ref)) / (np.max(np.abs(Y_ref)) + 1e-6)
    assert rel < 0.08, f"{cfg.name}: compiled generate rel={rel}"
    return rel


def test_m3_compiled_generation():
    r1 = _gen(model.MEDIUM, MAX=64, n_prompt=6, n_gen=4)
    r2 = _gen(model.REDUCED, MAX=16, n_prompt=5, n_gen=4)
    return f"compiled prefill+decode == numpy (MEDIUM rel={r1:.1e}, REDUCED rel={r2:.1e})"


def _gen_tokens(cfg, MAX, n_layers, vocab, n_prompt, n_gen, batched_prefill=False, seed=0):
    cos, sin, rot = _tables(cfg, MAX)
    layer_Ws, top = model.make_gen_weights(cfg, n_layers, vocab, seed=seed)
    rng = np.random.default_rng(seed + 11)
    prompt = [int(t) for t in rng.integers(0, vocab, size=n_prompt)]
    ref = model.ref_generate_tokens(cfg, layer_Ws, top, prompt, n_gen, cos, sin, rot)
    got = driver.generate_tokens(cfg, layer_Ws, top, prompt, n_gen, MAX, cos, sin, rot,
                                 batched_prefill=batched_prefill)
    assert ref == got, f"{cfg.name}: token seq ref={ref} npu={got} (batched={batched_prefill})"
    return ref


def test_m4_token_generation():
    """Full LM (embed[CPU] -> N layers + lm_head[NPU] -> argmax[CPU]) greedy gen:
    compiled token sequence must equal the numpy reference (token-by-token prefill)."""
    t1 = _gen_tokens(model.MEDIUM, MAX=64, n_layers=2, vocab=64, n_prompt=3, n_gen=3)
    t2 = _gen_tokens(model.REDUCED, MAX=16, n_layers=2, vocab=32, n_prompt=3, n_gen=2)
    return f"multi-layer greedy gen == numpy (MEDIUM {t1}, REDUCED {t2})"


def test_m6_batched_prefill_generation():
    """Same generation but the prompt is prefilled in ONE kernel/layer (batched);
    tokens must still equal the numpy reference (batched prefill seeds the cache
    that decode continues from)."""
    t1 = _gen_tokens(model.MEDIUM, MAX=64, n_layers=2, vocab=64, n_prompt=3, n_gen=3, batched_prefill=True)
    t2 = _gen_tokens(model.REDUCED, MAX=16, n_layers=2, vocab=32, n_prompt=3, n_gen=2, batched_prefill=True)
    return f"batched-prefill + decode gen == numpy (MEDIUM {t1}, REDUCED {t2})"


def _batched_prefill_kv(cfg, S, MAX, seed=0):
    cos, sin, rot = _tables(cfg, MAX)
    W = model.make_weights(cfg, seed=seed)
    rng = np.random.default_rng(seed + 3)
    X = _f16(rng.standard_normal((S, cfg.D))).astype(np.float64)
    Ks, Vs = model.ref_kv_proj(cfg, W, X, np.arange(S), cos, sin, rot)   # numpy, positions 0..S-1
    ins = {"x": _f16(X), "Wn1": W["Wn1"]}      # RoPE positions 0..S-1 baked into the kernel
    for k in range(cfg.KV):
        ins[f"Wk{k}"] = W[f"Wk{k}"]; ins[f"Wv{k}"] = W[f"Wv{k}"]
    out = driver.run_module(model.build_kv_proj_batched(cfg, S), ins, backend="hybrid")  # [S,2*KV*HD]
    HD = cfg.HD
    for k in range(cfg.KV):
        Kc = out[:, 2 * k * HD:(2 * k + 1) * HD]; Vv = out[:, (2 * k + 1) * HD:(2 * k + 2) * HD]
        sc = np.max(np.abs(Ks[k])) + np.max(np.abs(Vs[k])) + 1e-6
        assert np.max(np.abs(Kc - Ks[k])) < 0.05 * sc + 0.05, f"{cfg.name} batched K head{k}"
        assert np.max(np.abs(Vv - Vs[k])) < 0.05 * sc + 0.05, f"{cfg.name} batched V head{k}"


def test_m5_batched_prefill():
    """Batched prefill K/V == numpy (same values decode's cache expects), so a
    single batched kernel can seed the cache the decode loop continues from."""
    _batched_prefill_kv(model.MEDIUM, S=8, MAX=64)
    _batched_prefill_kv(model.REDUCED, S=8, MAX=16)
    return "batched-prefill K/V == numpy (decode-compatible cache layout)"


if __name__ == "__main__":
    print("[PASS] M1:", test_m1_self_consistency())
    print("[PASS] M2:", test_m2_kernels_one_step())
    print("[PASS] M3:", test_m3_compiled_generation())
    print("[PASS] M4:", test_m4_token_generation())
    print("[PASS] M5:", test_m5_batched_prefill())
    print("[PASS] M6:", test_m6_batched_prefill_generation())
    print("ALL DECODE (prefill->decode + KV cache + full LM gen) TESTS PASSED")
