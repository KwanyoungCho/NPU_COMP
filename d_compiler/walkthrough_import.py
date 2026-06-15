"""PyTorch 모델 import를 '눈으로' 따라가는 학습 스크립트.

실제 PyTorch nn.Module을 받아 우리 NPU 백엔드로 컴파일하는 전 과정을 단계별로 보여준다:
  torch.export → from_exported_program(Relax) → FoldConstant → import_legalize → compile → run
그리고 각 '갭'(고수준 op, 가중치 전치, tuple 출력, bias broadcast, fp16)이 어디서 처리되는지.

실행:  /home/chokwans99/anaconda3/envs/npu-tvm/bin/python d_compiler/walkthrough_import.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from npu_compiler import import_legalize, driver
from study_util import banner


def _f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def _ops(mod):
    c = {}
    for gv, fn in mod.functions_items():
        if isinstance(fn, relax.Function):
            for blk in fn.body.blocks:
                for b in blk.bindings:
                    if isinstance(b.value, relax.Call):
                        nm = getattr(b.value.op, "name", str(b.value.op))
                        c[nm] = c.get(nm, 0) + 1
    return dict(sorted(c.items()))


# 작은 예: Linear -> SiLU -> Linear  (matmul/bias/silu 가 다 등장)
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(64, 128); s.act = nn.SiLU(); s.b = nn.Linear(128, 64)
    def forward(s, x):
        return s.b(s.act(s.a(x)))


m = MLP().eval()
xt = torch.randn(64, 64) * 0.3

# ---------------------------------------------------------------------------
banner(1, "PyTorch 모델 → torch.export → Relax (from_exported_program)")
ex = torch.export.export(m, (xt,))
mod = from_exported_program(ex)
print("import된 op 종류:", _ops(mod))
print("\n특징: 가중치가 permute_dims(상수)로 전치됨(Linear W^T), 출력이 (out,) tuple,")
print("      활성화가 relax.nn.silu(고수준), dtype은 float32.")

# ---------------------------------------------------------------------------
banner(2, "FoldConstant: 가중치 전치를 상수로 미리 계산 (런타임 전치 제거)")
mod = relax.transform.FoldConstant()(mod)
print("op 종류 (permute_dims 사라짐):", _ops(mod))

# ---------------------------------------------------------------------------
banner(3, "import_legalize: 고수준 op → 우리 primitive")
mod = import_legalize.legalize(mod)
print("op 종류 (nn.silu → exp/add/divide/multiply/subtract 로 분해):", _ops(mod))
print("\n→ 이제 그래프가 우리 codegen이 아는 연산(matmul/add/sub/mul/div/exp/...)만 남음.")

# ---------------------------------------------------------------------------
banner(4, "나머지 갭은 memplan/driver가 처리")
print("  • tuple 출력 (out,)  → memplan이 언래핑")
print("  • bias broadcast [64]→[64,128] → memplan이 상수를 호스트에서 확장")
print("  • fp32 → fp16        → G-buffer가 fp16이라 데이터만 변환")
print("  • 입력 param명이 torch 인자명 → driver가 위치(list) 입력 허용")

# ---------------------------------------------------------------------------
banner(5, "컴파일(하이브리드) → mysim 실행 → torch와 비교")
got = driver.run_module(mod, [_f16(xt.numpy())], backend="hybrid")
ref = m(xt).detach().numpy()
rel = float(np.max(np.abs(got - ref))) / (float(np.max(np.abs(ref))) + 1e-6)
print(f"  NPU vs torch  rel = {rel:.4g}  (FP16 수준이면 일치)")

print("\n" + "="*72)
print("요약: torch.export→Relax는 TVM이, '고수준 op→우리 primitive'는 import_legalize가,")
print("      자잘한 정규화(tuple/bias/fp16/입력)는 memplan/driver가 처리. 그 뒤 우리 백엔드로 컴파일.")
print("      실제 Llama 블록도 같은 경로(legalize에 mean/softmax/rsqrt/slice/concat 추가됨).")
print("="*72)
