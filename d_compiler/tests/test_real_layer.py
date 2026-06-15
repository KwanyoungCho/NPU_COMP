"""Llama 3.2 3B single-layer prefill: compile via the hybrid backend.

Validation strategy (real 3B can't run in mysim — millions of instrs, GBs):
  - REDUCED proxy : hybrid == direct, byte-exact (router + integration intact)
  - MEDIUM (64x)  : hybrid == float reference (exercises the TIR matmul path in
                    a full layer; small enough to actually run in mysim)
  - LLAMA_3_2_3B  : COMPILE each real matmul shape via TIR + cost estimate
                    (same dim-agnostic code paths validated above; here we only
                    confirm it compiles hardware-legal and report instruction cost)
"""
import os, sys, gc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

from npu_compiler import model, driver, memplan, isa, tir_backend


def test_reduced_hybrid_eq_direct():
    cfg = model.REDUCED
    cos, sin, rot = model.rope_tables(cfg)
    W = model.make_weights(cfg, seed=7)
    mod = model.build_layer_module(cfg, cos, sin, rot)
    gd = driver.run_module(mod, W, tile=64)
    gh = driver.run_module(mod, W, backend="hybrid")
    assert np.array_equal(gd, gh), "reduced: hybrid != direct"
    exp = model.ref_layer(cfg, W, cos, sin)
    rel = float(np.max(np.abs(gh - exp))) / (float(np.max(np.abs(exp))) + 1e-9)
    assert rel < 0.02, f"reduced rel={rel}"
    return rel


def test_medium_hybrid_correct():
    """All-64-multiple full layer through hybrid (TIR matmul path), vs float ref."""
    cfg = model.MEDIUM
    cos, sin, rot = model.rope_tables(cfg)
    W = model.make_weights(cfg, seed=3)
    mod = model.build_layer_module(cfg, cos, sin, rot)
    gh = driver.run_module(mod, W, backend="hybrid")
    exp = model.ref_layer(cfg, W, cos, sin)
    maxabs = float(np.max(np.abs(exp)))
    maxerr = float(np.max(np.abs(gh - exp)))
    assert maxerr < 0.05 * maxabs + 0.05, f"medium maxerr={maxerr}"
    return maxerr / (maxabs + 1e-9)


def _mm_instr(M, K, N):
    """Compile one matmul via the TIR(reuse) path; return instruction count."""
    asm = isa.Asm(); mp = memplan.MemPlan()
    a = mp.scratch_alloc(M * K); b = mp.scratch_alloc(K * N); c = mp.scratch_alloc(M * N)
    tir_backend.emit_matmul_into(asm, mp, c, a, b, M, K, N)
    n = len(asm.words); del asm; gc.collect()
    return n


def test_3b_matmul_compiles():
    """A real-3B-dimension matmul compiles hardware-legal via the TIR path."""
    M, K, N = 128, 3072, 128                       # q/k/v projection
    asm = isa.Asm(); mp = memplan.MemPlan()
    a = mp.scratch_alloc(M * K); b = mp.scratch_alloc(K * N); c = mp.scratch_alloc(M * N)
    tir_backend.emit_matmul_into(asm, mp, c, a, b, M, K, N)
    bad = [(d1, d2) for w in asm.words if (w & 0xFF) == 0x88
           for d1, d2 in [((w >> 8) & 0xFF, (w >> 16) & 0xFF)] if d1 > 64 or d2 > 64]
    assert not bad, f"illegal tiles {bad}"
    assert len(asm.words) > 0
    return len(asm.words)


def report_3b_cost():
    cfg = model.LLAMA_3_2_3B
    S, D, H, KV, HD, F = cfg.SEQ, cfg.D, cfg.H, cfg.KV, cfg.HD, cfg.F
    cos, sin, rot = model.rope_tables(cfg)
    mp_full = memplan.plan(model.build_layer_module(cfg, cos, sin, rot)["main"])
    roles = [("q/k/v proj", S, D, HD, H + 2 * KV), ("scores", S, HD, S, H),
             ("ctx", S, S, HD, H), ("o proj", S, HD, D, H),
             ("gate/up", S, D, F, 2), ("down", S, F, D, 1)]
    print(f"=== {cfg.name} single layer prefill (SEQ={S},D={D},H={H},KV={KV},HD={HD},F={F}) ===")
    print(f"G-buffer footprint (weights+activations): {mp_full.top:,} FP16 = {mp_full.top*2/1e9:.2f} GB")
    print(f"{'role':<12}{'shape':<22}{'instr/each':>13}{'x':>5}{'subtotal':>15}")
    total = 0
    for name, M, K, N, cnt in roles:
        per = _mm_instr(M, K, N); sub = per * cnt; total += sub
        print(f"{name:<12}{f'[{M},{K}]@[{K},{N}]':<22}{per:>13,}{cnt:>5}{sub:>15,}")
    tr = S * HD * 8 * KV
    print(f"{'matmul sum':<12}{'':<22}{'':>13}{'':>5}{total:>15,}")
    print(f"transpose(K^T): {tr:,}   |   layer total ~= {total + tr:,} instructions")
    return total + tr


if __name__ == "__main__":
    print("[PASS] REDUCED hybrid==direct, rel=%.4g" % test_reduced_hybrid_eq_direct())
    print("[PASS] MEDIUM hybrid vs float ref, rel=%.4g" % test_medium_hybrid_correct())
    print("[PASS] 3B-dim matmul compiles, instrs=%d" % test_3b_matmul_compiles())
    print("-" * 60)
    report_3b_cost()
    print("-" * 60)
    print("ALL REAL-LAYER TESTS PASSED")
