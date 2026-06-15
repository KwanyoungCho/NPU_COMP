"""TIR+tensorize 백엔드를 '눈으로' 따라가는 학습 스크립트.

direct 백엔드(Relax 연산→명령 직접)와 달리, 이 경로는 정석 TVM처럼:
  Relax matmul → LegalizeOps(TIR 스칼라 루프) → tir.Schedule(64³ split + tensorize)
  → 스케줄된 TIR(바깥 루프 + intrinsic 호출) → _Walker(TIR 해석) → NPU 명령어
로 간다. 그리고 input reuse(같은 입력 타일 한 번만 gather)로 명령 수가 줄어든다.

실행:  /home/chokwans99/anaconda3/envs/npu-tvm/bin/python d_compiler/walkthrough_tir.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tvm
from tvm import relax
from npu_compiler import tir_backend, driver, cost
from study_util import banner, print_program

M = K = N = 128                      # 2x2x2 = 8 tiles of 64x64


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
banner(0, f"문제: C[{M},{N}] = A @ B  — 모두 64배수라 64×64 타일로 쪼갬")
print("direct 백엔드는 이걸 손으로 타일링했고, 여기선 TVM의 TIR+tensorize로 한다.")

# ---------------------------------------------------------------------------
banner(1, "Relax matmul (무엇을 할지)")
mod["main"].show()

# ---------------------------------------------------------------------------
banner(2, "LegalizeOps → TIR: 스칼라 3중 루프 (어떻게)")
modL = relax.transform.LegalizeOps()(mod)
gvar = None
for gv, fn in modL.functions_items():
    import tvm.tir as tir
    if isinstance(fn, tir.PrimFunc):
        gvar = gv
print("matmul PrimFunc (TIR) — 숫자 하나씩 곱·더하는 루프:")
modL[gvar].show()

# ---------------------------------------------------------------------------
banner(3, "tir.Schedule: 64³ split + reorder + decompose + tensorize")
sched = tir_backend.schedule_matmul(modL, gvar.name_hint)
print("스케줄된 TIR — 바깥 타일 루프 + 64×64 intrinsic 호출(npu_gemm_acc / npu_fill_zero):")
print("(아래 call_extern 들이 우리 명령으로 바뀔 자리)")
sched[gvar.name_hint].show()

# ---------------------------------------------------------------------------
banner(4, "_Walker: TIR을 해석해 NPU 명령어로 (루프 펼침 + 주소계산 + intrinsic→명령)")
asm_tir, mp_tir = tir_backend.compile_func(mod)
print_program(asm_tir, limit=24)
st = cost.analyze(asm_tir, mp_tir)
print(f"\n→ 총 {st['total']} 명령, MATMUL 타일 {st['matmul_tiles']}개(=2·2·2), "
      f"gather복사 {st['copy_ops']}, 누적 {st['accum_adds']}")

# ---------------------------------------------------------------------------
banner(5, "input reuse 효과: direct vs TIR(reuse) 명령 수")
asm_d, mp_d = driver.compile_func(mod["main"], tile=64)
sd = cost.analyze(asm_d, mp_d)
print(f"  direct      : {sd['total']:>7} 명령  (gather복사 {sd['copy_ops']})")
print(f"  tir+reuse   : {st['total']:>7} 명령  (gather복사 {st['copy_ops']})")
print("  → 같은 입력 타일을 '한 번만' gather(메모이제이션)해서 복사가 줄어든다.")
print("    차원이 클수록 절감이 커짐(중복 gather가 차원에 비례).")

# ---------------------------------------------------------------------------
banner(6, "정확성: TIR 경로 == tiled_fp16_reference (byte-exact)")
rng = np.random.default_rng(0)
A = _fp16(rng.standard_normal((M, K)) * 0.3); B = _fp16(rng.standard_normal((K, N)) * 0.3)
got = driver.run_module(mod, {"x": A, "w": B}, backend="tir")
def tiled_ref(A, B, T=64):
    C = None
    for kk in range(0, K, T):
        part = _fp16(A[:, kk:kk+T] @ B[kk:kk+T, :])
        C = part if C is None else _fp16(C + part)
    return C
print("  TIR 결과 == tiled_fp16_ref ?", "예 (byte-exact)" if np.array_equal(got, tiled_ref(A, B)) else "아니오")

print("\n" + "="*72)
print("요약: Relax→TIR(스칼라루프)→스케줄(타일+tensorize)→walker(해석→명령). reuse로 gather 절감.")
print("      walker가 곧 'TIR→우리 ISA codegen'. 정확성은 tiled_fp16_ref로 보증.")
print("="*72)
