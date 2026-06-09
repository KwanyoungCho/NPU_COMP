"""legalize(미지원 연산 우회)를 '눈으로' 보는 학습 스크립트 — RMSNorm.

RMSNorm은 NPU에 전용 명령이 없다. 그래서 legalize.rms_norm()이 이걸
NPU가 할 수 있는 연산들(곱셈/행렬곱/제곱근/나눗셈)의 조합으로 '분해'한다.
특히 '합(reduce)'과 '복제(broadcast)'를 'ones와의 행렬곱'으로 바꾸는 트릭이 핵심.

실행:  /home/chokwans99/anaconda3/envs/npu-tvm/bin/python d_compiler/walkthrough_rmsnorm.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tvm import relax
from npu_compiler import memplan, codegen, driver, legalize
from study_util import banner, print_program

SEQ, D = 4, 8        # 작게 (읽기 쉽게)


def arg_str(a, mp):
    if isinstance(a, relax.Constant):
        arr = a.data.numpy().reshape(-1)
        tag = "ones" if np.all(arr == 1) else (f"{np.round(arr[:3],3)}…" if arr.size > 3 else str(np.round(arr, 3)))
        return f"상수{list(a.data.numpy().shape)}={tag}"
    if isinstance(a, relax.Var):
        return f"{a.name_hint}{mp.shape.get(a, '')}"
    return type(a).__name__


# ---------------------------------------------------------------------------
banner(0, "RMSNorm이 수학적으로 하는 일")
print("RMSNorm(x) = x / sqrt(mean(x²)) · w     (한 행(토큰)마다 정규화)")
print(f"입력 x[{SEQ},{D}], 가중치 w[1,{D}]")
print("NPU엔 'mean(=합)', '브로드캐스트', '음수화' 같은 전용 명령이 없음 → 우회 필요")


# ---------------------------------------------------------------------------
banner(1, "legalize.rms_norm(): 한 번 호출 → 여러 primitive 연산으로 분해")
bb = relax.BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo([SEQ, D], "float16"))
w = relax.Var("w", relax.TensorStructInfo([1, D], "float16"))
with bb.function("main", [x, w]):
    with bb.dataflow():
        y = legalize.rms_norm(bb, x, w, SEQ, D)   # ← 이 한 줄이 아래 그래프 전체를 만든다
        gv = bb.emit_output(y)
    bb.emit_func_output(gv)
mod = bb.finalize()
func = mod["main"]
print("분해된 Relax 그래프:")
mod.show()


# ---------------------------------------------------------------------------
banner(2, "각 연산이 무엇이고, '우회 트릭'이 어디 있나")
mp = memplan.plan(func)
for b in func.body.blocks[0].bindings:
    v = b.value
    if isinstance(v, relax.Call):
        args = ", ".join(arg_str(a, mp) for a in v.args)
        opname = v.op.name.replace("relax.", "")
        note = ""
        if opname == "matmul":
            # ones와의 행렬곱이면 reduce 또는 broadcast 트릭
            shapes = [mp.shape.get(a) if isinstance(a, relax.Var) else list(a.data.numpy().shape)
                      for a in v.args]
            if shapes[1] == [D, 1]:
                note = "   ← 합(reduce): x⊙x 를 ones[D,1]와 곱해 행마다 더함"
            elif shapes[1] == [1, D]:
                note = "   ← 복제(broadcast): [.,1]을 ones[1,D]와 곱해 D칸으로 펼침"
        print(f"  {b.var.name_hint:<5} = {opname}({args}){note}")
    elif isinstance(v, relax.Var):
        print(f"  {b.var.name_hint:<5} = {v.name_hint}   (별칭/출력)")
print(f"\n→ rms_norm() 한 줄이 {len(func.body.blocks[0].bindings)}개 바인딩으로 펼쳐짐.")
print("  전용 명령 없는 reduce/broadcast가 전부 'ones 행렬곱'으로 바뀐 게 핵심.")


# ---------------------------------------------------------------------------
banner(3, "상수(ones 등)는 G-buffer에 미리 박힌다")
print("memplan이 잡은 상수들:")
for c in mp.constants:
    arr = c.data.numpy()
    flat = arr.reshape(-1)
    tag = "전부 1" if np.all(arr == 1) else f"{np.round(flat[:4],4)}…"
    print(f"  주소 {mp.offset[c]:<4} 모양 {list(arr.shape)}  값: {tag}")
print("→ 이 상수 데이터는 driver가 실행 전에 초기 G-buffer에 써넣는다 (입력 데이터와 함께).")


# ---------------------------------------------------------------------------
banner(4, "생성된 NPU 명령어 (앞부분만)")
asm = codegen.compile_func(func, mp)
print_program(asm, limit=22)
print(f"\n→ matmul 하나 = TILE/LOAD/MATMUL/SAVE 묶음. elementwise 하나 = VLEN/LOAD/연산/SAVE 묶음.")


# ---------------------------------------------------------------------------
banner(5, "실행하고 numpy 정답과 비교")
rng = np.random.default_rng(0)
xv = np.asarray(rng.standard_normal((SEQ, D)), dtype=np.float16).astype(np.float32)
wv = np.asarray(rng.uniform(0.8, 1.2, (1, D)), dtype=np.float16).astype(np.float32)
got = driver.run_module(mod, {"x": xv, "w": wv})

ms = np.mean(xv ** 2, axis=1, keepdims=True)
ref = xv / np.sqrt(ms) * wv

maxerr = float(np.max(np.abs(got - ref)))
print("NPU 결과 [0행]:", np.round(got[0], 4))
print("정답   [0행]:", np.round(ref[0], 4))
print(f"최대오차 = {maxerr:.5f}  (FP16 반올림 수준이면 정상)")
print("\n" + "="*72)
print("요약: 전용명령 없는 연산(RMSNorm)을 legalize가 곱/행렬곱/제곱근/나눗셈으로 분해.")
print("      reduce=ones곱, broadcast=ones곱. 상수는 G-buffer에 미리 적재.")
print("="*72)
