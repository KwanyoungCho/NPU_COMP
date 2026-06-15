"""M·N·K 세 축 타일링을 '눈으로' 확인하는 학습 스크립트.

큰 행렬곱을 64×64 조각으로 어떻게 쪼개는지:
  - 출력 C[M,N]을 (M/64)×(N/64)개의 64×64 '출력 타일'로 나눔   ← M축, N축
  - 각 출력 타일 = K방향 (K/64)개 부분곱의 합(누적)            ← K축
  - 즉 m_mul 호출 수 = ⌈M/64⌉·⌈N/64⌉·⌈K/64⌉, 전부 ≤64×64

여기서 ① 타일링의 '수학'(numpy)  ② TIR 백엔드가 만든 루프 구조  ③ 실제 명령어 증거
④ 두 백엔드(direct/tir) 정확성 을 모두 보여준다.

실행:  /home/chokwans99/anaconda3/envs/npu-tvm/bin/python d_compiler/walkthrough_mnk.py
"""
import os, sys, warnings, math
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tvm.tir as tir
from tvm import relax
from npu_compiler import driver, tir_backend
from study_util import banner

T = 64
M, K, N = 128, 192, 128                 # 2 M-타일 × 2 N-타일 × 3 K-타일


def _f16(a):
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


# ---------------------------------------------------------------------------
banner(0, f"문제: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}],  PE는 64×64")
mt, nt, kt = math.ceil(M/T), math.ceil(N/T), math.ceil(K/T)
print(f"  M축 → {mt}조각,  N축 → {nt}조각,  K축 → {kt}조각")
print(f"  출력 타일 = {mt}×{nt} = {mt*nt}개,  각 타일은 K방향 {kt}개 부분곱 누적")
print(f"  ⇒ 64×64 행렬곱(m_mul) 호출 = {mt}·{nt}·{kt} = {mt*nt*kt}회")

# ---------------------------------------------------------------------------
banner(1, "① 타일링의 '수학' — numpy로 직접 (이게 정답의 정의)")
rng = np.random.default_rng(0)
A = _f16(rng.standard_normal((M, K)) * 0.3)
B = _f16(rng.standard_normal((K, N)) * 0.3)
C = np.zeros((M, N), np.float32)
log = []
for mi in range(0, M, T):                       # M축 출력 타일
    for nj in range(0, N, T):                   # N축 출력 타일
        for kk in range(0, K, T):               # K축 누적
            C[mi:mi+T, nj:nj+T] += A[mi:mi+T, kk:kk+T] @ B[kk:kk+T, nj:nj+T]
            log.append((mi//T, nj//T, kk//T))
print("  (출력타일 mi,nj / K조각 kk) 별 64×64 부분곱 순서:")
for t in log:
    print(f"     출력타일(M{t[0]},N{t[1]})  += A블록[M{t[0]},K{t[2]}] @ B블록[K{t[2]},N{t[1]}]")
full = A.astype(np.float32) @ B.astype(np.float32)
rel = float(np.max(np.abs(C - full))) / (float(np.max(np.abs(full))) + 1e-9)
print(f"  타일 합 ≈ 전체 행렬곱?  상대오차 {rel:.1e} (합산 순서 차이 = float 재배열 수준)  · 총 {len(log)}개 64×64 곱")

# ---------------------------------------------------------------------------
banner(2, "② TIR 백엔드가 만든 루프 구조 — M·N·K 축이 그대로 보임")
modL = relax.transform.LegalizeOps()(make_matmul(M, K, N))
gv = [g for g, f in modL.functions_items() if isinstance(f, tir.PrimFunc)][0]
sched = tir_backend.schedule_matmul(modL, gv.name_hint)
txt = sched[gv.name_hint].script()
for line in txt.splitlines():
    s = line.strip()
    if s.startswith("for ") or "block(" in s or "call_extern" in s:
        print("   " + s[:96])
print(f"\n  → 바깥 'for i0_0, i1_0 in T.grid({mt}, {nt})' = M·N 출력 타일,  'for k_0 in range({kt})' = K 누적.")
print("     안쪽 64블록은 npu_gemm_acc(=우리 m_mul 명령)로 tensorize됨.")

# ---------------------------------------------------------------------------
banner(3, "③ 실제 명령어 증거 — m_mul 호출 수와 타일 크기 (두 백엔드)")
def stats(asm):
    n_mm = sum(1 for w in asm.words if (w & 0xFF) == 0x42 and (w >> 30) & 3 == 2)
    dims = [(((w>>8)&0xFF), ((w>>16)&0xFF)) for w in asm.words if (w & 0xFF) == 0x88]
    return n_mm, max((max(d) for d in dims), default=0)
mod = make_matmul(M, K, N)
ad, _ = driver.compile_func(mod["main"], tile=64)
at, _ = tir_backend.compile_func(mod)
for name, asm in [("direct", ad), ("tir", at)]:
    nmm, mx = stats(asm)
    print(f"  {name:<7} m_mul(64×64) 호출 = {nmm}  (기대 {mt*nt*kt})   가장 큰 타일변 = {mx} ({'≤64 OK' if mx<=64 else '64초과!'})")

# ---------------------------------------------------------------------------
banner(4, "④ 정확성 — 두 백엔드 결과 == tiled_fp16_reference (byte-exact)")
def tiled_ref(A, B):
    C = None
    for kk in range(0, K, T):
        p = _f16(A[:, kk:kk+T] @ B[kk:kk+T, :])
        C = p if C is None else _f16(C + p)
    return C
ref = tiled_ref(A, B)
gd = driver.run_module(mod, {"x": A, "w": B}, tile=64)
gt = driver.run_module(mod, {"x": A, "w": B}, backend="tir")
print(f"  direct == ref ? {np.array_equal(gd, ref)}    tir == ref ? {np.array_equal(gt, ref)}")

# ---------------------------------------------------------------------------
banner(5, "⑤ 여러 차원: 어느 축이 쪼개지나 (K만 / M만 / N만 / 셋다 / 비64배수)")
print(f"  {'M×K×N':<16}{'M타일':>6}{'N타일':>6}{'K타일':>6}{'m_mul수':>9}{'타일≤64':>9}{'정확':>7}")
for (m, k, n, tag) in [(64,192,64,"K만"),(192,64,64,"M만"),(64,64,192,"N만"),
                       (128,192,128,"셋다"),(130,67,200,"비64배수")]:
    mm = make_matmul(m, k, n)
    A2 = _f16(np.random.default_rng(m+k+n).standard_normal((m, k))*0.3)
    B2 = _f16(np.random.default_rng(k+n).standard_normal((k, n))*0.1)
    a2, _ = tir_backend.compile_func(mm)
    nmm, mx = stats(a2)
    g2 = driver.run_module(mm, {"x": A2, "w": B2}, backend="tir")
    def tref(A, B, K):
        C = None
        for kk in range(0, K, T):
            p = _f16(A[:, kk:kk+T] @ B[kk:kk+T, :]); C = p if C is None else _f16(C+p)
        return C
    ok = np.array_equal(g2, tref(A2, B2, k))
    print(f"  {f'{m}×{k}×{n} ({tag})':<16}{math.ceil(m/T):>6}{math.ceil(n/T):>6}{math.ceil(k/T):>6}"
          f"{nmm:>9}{('OK' if mx<=64 else 'X'):>9}{('OK' if ok else 'X'):>7}")

print("\n" + "="*78)
print("요약: 큰 행렬곱 = (M/64)×(N/64) 출력타일, 각 타일은 K/64 부분곱 누적. m_mul은 전부 ≤64×64.")
print("      TIR은 이 M·N·K 루프를 자동 생성(tensorize), direct는 손으로 같은 구조를 emit. 결과 동일.")
print("="*78)
