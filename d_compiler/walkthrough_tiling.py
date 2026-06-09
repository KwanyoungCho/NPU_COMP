"""타일링(B0.5)을 '눈으로' 보는 학습 스크립트.

같은 행렬곱을 두 방식으로 컴파일해 비교한다:
  (A) tile=None : 한 방에 (logical). m_mul 명령 1개.  ← B0
  (B) tile=64   : K를 64씩 쪼개 부분곱을 누적 (hardware-legal). ← B0.5
NPU PE는 64×64라서 K가 64를 넘으면 (B)처럼 쪼개야 한다.
그리고 (B)는 조각마다 저장(save)→FP16 반올림이 끼어들어 (A)와 결과가 '정상적으로' 달라진다.

실행:  /home/chokwans99/anaconda3/envs/npu-tvm/bin/python d_compiler/walkthrough_tiling.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tvm import relax
from npu_compiler import driver
from npu_compiler.driver import compile_func
from study_util import banner, print_program

M, K, N = 4, 128, 4        # K=128 > 64 이므로 타일 2개 필요. M,N은 작게(읽기 쉽게)


def _fp16(a):
    return np.asarray(a, dtype=np.float16).astype(np.float32)


def make_matmul(M, K, N):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([M, K], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([K, N], "float16"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x, w)); gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


mod = make_matmul(M, K, N)

# ---------------------------------------------------------------------------
banner(0, f"문제: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}],  PE는 64×64인데 K={K}>64")
print("→ K 방향으로 64씩 두 조각으로 쪼개, 각 조각의 행렬곱을 '누적'해야 한다.")
print(f"   조각: K[0:64], K[64:128]  → 부분곱 2개를 더함")


# ---------------------------------------------------------------------------
banner(1, "(A) tile=None : 한 방에 (logical, B0)")
asmA, _ = compile_func(mod["main"], tile=None)
print_program(asmA)
print(f"\n→ 명령어 {len(asmA.words)}개. MATMUL 1개로 끝. (mysim이 K=128도 받아주니 가능 = '논리적')")
print("   하지만 실제 64×64 PE에선 불법(한 번에 128은 못 함).")


# ---------------------------------------------------------------------------
banner(2, "(B) tile=64 : K를 쪼개 누적 (hardware-legal, B0.5)")
asmB, _ = compile_func(mod["main"], tile=64)
print(f"명령어 총 {len(asmB.words)}개. 구조를 보면:")
print_program(asmB, limit=30)
print("...")
print("\n구조 해설:")
print("  • VLEN/ADDR/LOAD/VADD(복사)/SAVE 묶음이 여러 번 = A의 한 조각을 '연속 메모리로 모으기(gather)'")
print("    (A의 K조각은 띄엄띄엄 있어서 NPU가 못 읽음 → 행마다 복사해 모음)")
print("  • TILE/LOAD/MATMUL/SAVE = 64×64 이하 부분곱 1개")
print("  • 둘째 조각부터는 VADD로 '이전 합 + 이번 부분곱' = 누적")
print(f"  • 모든 MATMUL 타일이 64 이하인지 확인 ↓")
maxtile = 0
for word in asmB.words:
    if (word & 0xFF) == 0x88:
        d1 = (word >> 8) & 0xFF; d2 = (word >> 16) & 0xFF
        maxtile = max(maxtile, d1, d2)
print(f"    가장 큰 타일 차원 = {maxtile}  ({'OK ≤64, hardware-legal' if maxtile <= 64 else '64 초과!'})")


# ---------------------------------------------------------------------------
banner(3, "두 방식 실행 + 'one-shot vs 타일링' 결과 비교")
rng = np.random.default_rng(1)
A = _fp16(rng.standard_normal((M, K)) * 0.3)
B = _fp16(rng.standard_normal((K, N)) * 0.3)

outA = driver.run_module(mod, {"x": A, "w": B}, tile=None)
outB = driver.run_module(mod, {"x": A, "w": B}, tile=64)


# 타일링이 내부적으로 하는 FP16 반올림을 그대로 흉내낸 참조
def tiled_fp16_ref(A, B, T=64):
    C = None
    for kk in range(0, K, T):
        part = _fp16(A[:, kk:kk + T] @ B[kk:kk + T, :])   # 조각마다 반올림
        C = part if C is None else _fp16(C + part)        # 누적마다 반올림
    return C


ref_true = (A.astype(np.float64) @ B.astype(np.float64))   # 진짜 정답
tref = tiled_fp16_ref(A, B)                                 # 타일링 반올림 흉내 (모양 (M,N))
# run_module은 결과를 이미 (M,N) 모양으로 돌려준다
print("(A) one-shot 결과:\n", np.round(outA, 4))
print("(B) 타일링  결과:\n", np.round(outB, 4))
print()
print("타일링(B) == 'tiled_fp16_ref'(같은 반올림 흉내) ?",
      "예 (byte-exact)" if np.array_equal(outB, tref) else "아니오")
ndiff = int(np.sum(outA != outB))
print(f"타일링(B) vs one-shot(A) 다른 원소 = {ndiff}/{M*N}개  ← 조각마다 FP16 반올림 때문 (버그 아님!)")
relB = float(np.max(np.abs(outB - ref_true))) / (float(np.max(np.abs(ref_true))) + 1e-6)
print(f"둘 다 진짜 정답(float64)엔 가까움: 타일링 상대오차 = {relB:.2e}")

print("\n" + "="*72)
print("요약: K>64면 64조각으로 쪼개 누적(save→load→add). 조각마다 FP16 반올림이라")
print("      one-shot과 결과가 다른 게 정상 → 검증은 같은 반올림을 흉내낸 tiled_fp16_ref와.")
print("="*72)
