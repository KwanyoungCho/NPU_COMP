"""전체 동작을 '눈으로' 따라가는 학습 스크립트.

작은 행렬곱 C[2,2] = A[2,3] @ B[3,2] 하나를 우리 컴파일러 파이프라인의
모든 단계에 통과시키며, 각 단계에서 실제로 무엇이 만들어지는지 출력한다.

실행:  /home/chokwans99/anaconda3/envs/npu-tvm/bin/python d_compiler/walkthrough_matmul.py
파이프라인:  Relax 그래프 → memplan(주소배치) → codegen(명령어) → G-buffer채우기 → mysim실행 → 결과
"""
import os, sys, warnings
warnings.filterwarnings("ignore")          # pygments 색칠 경고 등 숨김 (동작과 무관)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tvm import relax
from npu_compiler import memplan, codegen, runtime
from npu_compiler.driver import _numel
from study_util import banner, disasm


# ---------------------------------------------------------------------------
banner(0, "입력 준비: 작은 행렬 A[2,3], B[3,2]")
A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float16)
B = np.array([[1, 0],
              [0, 1],
              [1, 1]], dtype=np.float16)
print("A =\n", A)
print("B =\n", B)
print("우리가 원하는 답 A@B =\n", (A.astype(np.float32) @ B.astype(np.float32)))


# ---------------------------------------------------------------------------
banner(1, "Relax 그래프 만들기 (신경망을 IR로 표현)")
bb = relax.BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo([2, 3], "float16"))   # A가 들어갈 입력
w = relax.Var("w", relax.TensorStructInfo([3, 2], "float16"))   # B가 들어갈 입력
with bb.function("main", [x, w]):
    with bb.dataflow():
        y = bb.emit(relax.op.matmul(x, w))   # y = x @ w
        gv = bb.emit_output(y)
    bb.emit_func_output(gv)
mod = bb.finalize()
func = mod["main"]
print("Relax 함수 구조:")
mod.show()
print("\n→ 'lv = R.matmul(x, w)' 한 줄짜리 그래프. 아직 명령어가 아니라 '무엇을 할지'만 있음.")


# ---------------------------------------------------------------------------
banner(2, "memplan: 각 텐서를 G-buffer(1차원 메모리) 어디에 둘지 정하기")
mp = memplan.plan(func)
print(f"{'텐서':<8}{'주소(offset)':<14}{'모양':<10}{'원소수'}")
for v in list(mp.offset.keys()):
    name = getattr(v, "name_hint", type(v).__name__)
    print(f"{name:<8}{mp.offset[v]:<14}{str(mp.shape[v]):<10}{_numel(mp.shape[v])}")
print(f"\nG-buffer 총 크기: {mp.top} 개")
print("→ x는 주소 0부터, w는 주소 6부터, 결과는 주소 12부터. (행 우선으로 펼쳐 저장)")
print("  'lv'(matmul 결과)와 'gv'(함수 출력)는 같은 주소 = alias (gv는 lv의 복사라 자리 안 줌).")


# ---------------------------------------------------------------------------
banner(3, "codegen: Relax 연산 → NPU 명령어")
asm = codegen.compile_func(func, mp)
print(f"{'#':<4}{'hex':<14}{'사람이 읽는 의미'}")
for i, word in enumerate(asm.words):
    print(f"{i:<4}0x{word:08x}   {disasm(word)}")
print("\n→ matmul 하나가 'TILE/ADDR/LOAD/MATMUL/SAVE' 명령들로 펼쳐짐. 이게 NPU가 실제 실행할 프로그램.")
print("  ADDR이 주소를 정하면, 바로 다음 LOAD/SAVE가 그 주소를 사용함 (상태를 세팅→사용).")


# ---------------------------------------------------------------------------
banner(4, "G-buffer 초기값 채우기: 입력 데이터를 정해진 주소에 써넣기")
gbuf = np.zeros(mp.top, dtype=np.float32)
gbuf[mp.offset[func.params[0]]:][:A.size] = A.reshape(-1)   # x ← A at 0
gbuf[mp.offset[func.params[1]]:][:B.size] = B.reshape(-1)   # w ← B at 6
print("G-buffer (실행 전):")
print(" 주소 0~5  (x=A 평탄화):", gbuf[0:6])
print(" 주소 6~11 (w=B 평탄화):", gbuf[6:12])
print(" 주소 12~15(결과 자리, 아직 0):", gbuf[12:16])


# ---------------------------------------------------------------------------
banner(5, "mysim 실행: 주어진 NPU 시뮬레이터가 프로그램을 돌림")
out = runtime.run(asm, gbuf, gn=mp.top)
print("실행 후 G-buffer 주소 12~15 (결과가 채워짐):", out[12:16])
result = out[12:16].reshape(2, 2)
print("→ 결과를 2x2로 복원:\n", result)


# ---------------------------------------------------------------------------
banner(6, "검증: numpy 정답과 비교")
ref = (A.astype(np.float32) @ B.astype(np.float32))
print("NPU 결과:\n", result)
print("numpy 정답:\n", ref)
print("\n일치?", "예 (byte-exact)" if np.array_equal(result, ref) else "아니오")

print("\n" + "="*70)
print("요약: Relax(무엇)→memplan(어디에)→codegen(어떤 명령)→G-buffer(데이터)→mysim(실행)→검증")
print("="*70)
