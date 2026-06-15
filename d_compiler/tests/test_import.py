"""Import real PyTorch modules (torch.export) -> Relax -> our NPU backend.

Proves the ingestion pipeline on growing op coverage:
  - Linear            (matmul + bias broadcast, tuple output, fp32->fp16 G-buffer)
  - MLP               (Linear -> SiLU -> Linear)
  - SwiGLU FFN        (the real Llama FFN: down(silu(gate(x)) * up(x)))
Compared against the torch reference (FP16 tolerance). Random weights — this
validates the COMPILE pipeline, not output quality.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "d_compiler"))

import math
import torch
import torch.nn as nn
from npu_compiler import frontend, driver, legalize as LG


def _f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def _run(model, xt, backend="hybrid"):
    mod = frontend.import_torch(model, (xt,))
    got = driver.run_module(mod, [_f16(xt.detach().numpy())], backend=backend)  # positional
    ref = model(xt).detach().numpy()
    return float(np.max(np.abs(got - ref))) / (float(np.max(np.abs(ref))) + 1e-6)


def test_linear():
    torch.manual_seed(0)
    m = nn.Linear(64, 64)
    rel = _run(m, torch.randn(64, 64) * 0.3)
    assert rel < 0.02, f"linear rel={rel}"
    return rel


def test_mlp():
    torch.manual_seed(1)
    m = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
    rel = _run(m, torch.randn(64, 64) * 0.3)
    assert rel < 0.03, f"mlp rel={rel}"
    return rel


class SwiGLU(nn.Module):
    """Real Llama FFN: down( SiLU(gate(x)) * up(x) )."""
    def __init__(self, d, f):
        super().__init__()
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


def test_swiglu_ffn():
    torch.manual_seed(2)
    m = SwiGLU(64, 128)
    # small weights so SiLU/FP16 stay well-conditioned
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(0.3)
    rel = _run(m, torch.randn(64, 64) * 0.3)
    assert rel < 0.03, f"swiglu rel={rel}"
    return rel


# ---- full Llama decoder block (RMSNorm + GQA + RoPE + causal softmax + SwiGLU) ----
_BD, _BH, _BKV, _BHD, _BF, _BS = 128, 2, 1, 64, 128, 64


class _RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.w


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class LlamaBlock(nn.Module):
    """Self-contained Llama decoder layer (eager, per-head; imports cleanly)."""
    def __init__(self):
        super().__init__()
        D, H, KV, HD, F = _BD, _BH, _BKV, _BHD, _BF
        self.n1 = _RMSNorm(D); self.n2 = _RMSNorm(D)
        mk = lambda i, o: nn.Linear(i, o, bias=False)
        self.wq = nn.ModuleList([mk(D, HD) for _ in range(H)])
        self.wo = nn.ModuleList([mk(HD, D) for _ in range(H)])
        self.wk = nn.ModuleList([mk(D, HD) for _ in range(KV)])
        self.wv = nn.ModuleList([mk(D, HD) for _ in range(KV)])
        self.g = mk(D, F); self.u = mk(D, F); self.dn = mk(F, D); self.act = nn.SiLU()
        for p in self.parameters():
            with torch.no_grad():
                p.mul_(0.3)

    def forward(self, x, cos, sin, mask):
        H, KV, HD = _BH, _BKV, _BHD
        gpk = H // KV
        xn = self.n1(x)
        K, V = [], []
        for k in range(KV):
            kk = self.wk[k](xn)
            K.append(_rot_half(kk) * sin + kk * cos)
            V.append(self.wv[k](xn))
        parts = []
        for h in range(H):
            kv = h // gpk
            q = self.wq[h](xn); q = _rot_half(q) * sin + q * cos
            sc = (q @ K[kv].transpose(0, 1)) / math.sqrt(HD) + mask
            parts.append(self.wo[h](torch.softmax(sc, dim=-1) @ V[kv]))
        attn = parts[0]
        for p in parts[1:]:
            attn = attn + p
        hh = x + attn
        hn = self.n2(hh)
        return hh + self.dn(self.act(self.g(hn)) * self.u(hn))


def test_llama_block():
    torch.manual_seed(4)
    m = LlamaBlock().eval()
    cos, sin, _ = LG.rope_tables(_BS, _BHD, base=10000.0)
    xt = torch.randn(_BS, _BD) * 0.3
    ct = torch.tensor(cos, dtype=torch.float32); st = torch.tensor(sin, dtype=torch.float32)
    mk = torch.tensor(LG.causal_mask(_BS), dtype=torch.float32)
    mod = frontend.import_torch(m, (xt, ct, st, mk))
    got = driver.run_module(mod, [_f16(xt.numpy()), _f16(cos), _f16(sin), _f16(mk.numpy())],
                            backend="hybrid")
    ref = m(xt, ct, st, mk).detach().numpy()
    rel = float(np.max(np.abs(got - ref))) / (float(np.max(np.abs(ref))) + 1e-6)
    assert rel < 0.05, f"llama block rel={rel}"
    return rel


if __name__ == "__main__":
    print("[PASS] Linear (torch->NPU)      rel=%.4g" % test_linear())
    print("[PASS] MLP (Linear-SiLU-Linear) rel=%.4g" % test_mlp())
    print("[PASS] SwiGLU FFN (real Llama)  rel=%.4g" % test_swiglu_ffn())
    print("[PASS] FULL Llama decoder block rel=%.4g" % test_llama_block())
    print("ALL IMPORT TESTS PASSED")
