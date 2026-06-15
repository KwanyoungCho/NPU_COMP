"""Real Llama 3.2 3B weights: compiled NPU output must match torch.

Weights: layer-0 tensors of meta-llama/Llama-3.2-3B (range-read ~200MB to
d_compiler/build/llama32_3b_layer0.npz). Full D=3072 layer can't RUN in mysim
(it prints every element), so we verify the RUNNABLE pieces with REAL weights:
q/k/v/o projections (and the FFN in __main__), each vs the torch reference.

If the weights file is absent, the test self-skips (it's a ~200MB gated download).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

_WPATH = os.path.join(ROOT, "d_compiler", "build", "llama32_3b_layer0.npz")
HAVE = os.path.exists(_WPATH)

D, HD, S = 3072, 128, 64


def _f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def _piece(name, wt, in_dim, tol=0.02):
    import torch, torch.nn as nn
    from npu_compiler import frontend, driver
    lin = nn.Linear(wt.shape[1], wt.shape[0], bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(wt))
    xt = torch.randn(S, in_dim) * 0.5
    mod = frontend.import_torch(lin, (xt,))
    got = driver.run_module(mod, [_f16(xt.numpy())], backend="hybrid")
    ref = lin(xt).detach().numpy()
    rel = float(np.max(np.abs(got - ref))) / (float(np.max(np.abs(ref))) + 1e-6)
    assert rel < tol, f"{name} rel={rel}"
    return rel


def test_real_projections():
    if not HAVE:
        print("SKIP: real weights not downloaded (d_compiler/build/llama32_3b_layer0.npz)")
        return
    W = np.load(_WPATH)
    out = {}
    out["q_proj"] = _piece("q_proj head0", W["self_attn.q_proj.weight"][0:HD, :], D)
    out["k_proj"] = _piece("k_proj head0", W["self_attn.k_proj.weight"][0:HD, :], D)
    out["v_proj"] = _piece("v_proj head0", W["self_attn.v_proj.weight"][0:HD, :], D)
    out["o_proj"] = _piece("o_proj head0", W["self_attn.o_proj.weight"][:, 0:HD], HD)
    return out


def test_real_ffn():
    """Real gate/up/down + SiLU. Bigger (slow in mysim) — run explicitly."""
    if not HAVE:
        print("SKIP: real weights not downloaded")
        return
    import torch, torch.nn as nn
    from npu_compiler import frontend, driver
    W = np.load(_WPATH)
    F = 8192

    class SwiGLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.Linear(D, F, bias=False); self.u = nn.Linear(D, F, bias=False)
            self.dn = nn.Linear(F, D, bias=False); self.act = nn.SiLU()
            with torch.no_grad():
                self.g.weight.copy_(torch.tensor(W["mlp.gate_proj.weight"]))
                self.u.weight.copy_(torch.tensor(W["mlp.up_proj.weight"]))
                self.dn.weight.copy_(torch.tensor(W["mlp.down_proj.weight"]))

        def forward(self, x):
            return self.dn(self.act(self.g(x)) * self.u(x))

    m = SwiGLU().eval(); xt = torch.randn(S, D) * 0.3
    mod = frontend.import_torch(m, (xt,))
    got = driver.run_module(mod, [_f16(xt.numpy())], backend="hybrid")
    ref = m(xt).detach().numpy()
    rel = float(np.max(np.abs(got - ref))) / (float(np.max(np.abs(ref))) + 1e-6)
    assert rel < 0.03, f"ffn rel={rel}"
    return rel


if __name__ == "__main__":
    if not HAVE:
        print("SKIP (no real weights)"); sys.exit(0)
    out = test_real_projections()
    for k, v in out.items():
        print(f"[PASS] real 3B {k:<8} NPU vs torch rel={v:.4g}")
    print("[PASS] real 3B FFN (SwiGLU)  NPU vs torch rel=%.4g" % test_real_ffn())
    print("ALL REAL-WEIGHT TESTS PASSED")
