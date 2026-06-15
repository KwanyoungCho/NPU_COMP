"""matmul의 전체 lowering을 '아주 자세히' 따라가는 학습 스크립트 (TVM 입문 동반 설명).

relax.matmul 한 줄이 NPU 명령이 되기까지 6단계를 TVM 개념과 함께 본다:

  [1] Relax(그래프 IR)        무엇을 계산할지            relax.op.matmul
  [2] LegalizeOps → TIR        숫자 단위 스칼라 루프      TVM 패스
  [3] tir.Schedule             64×64 타일링 + tensorize  split/reorder/decompose/tensorize
  [4] match_buffer/access_ptr  '심볼릭 타일 뷰'           스케줄 결과 해부
  [5] _Walker                  심볼 → 실제 주소 → 명령    우리 codegen (verbose 추적)
  [6] reuse + 정확성           중복 gather 제거, byte-exact

TVM을 모른다는 가정으로, 각 단계마다 개념을 먼저 풀어 설명한다.
실행:  /home/chokwans99/anaconda3/envs/npu-tvm/bin/python d_compiler/walkthrough_tir.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tvm.tir as tir
from tvm import relax
from npu_compiler import tir_backend, driver, cost, memplan
from npu_compiler.isa import Asm
from study_util import banner, print_program

M = K = N = 128                      # 2×2×2 = 8 개의 64×64 타일 (작아서 추적이 짧음)


def _fp16(a):
    return np.asarray(a, dtype=np.float16).astype(np.float32)


def note(*lines):
    """개념 설명 블록 (들여쓰기 + 말머리)."""
    for ln in lines:
        print("   " + ln)
    print()


def make_matmul(M, K, N):
    """Relax 함수 하나를 만든다: main(x[M,K], w[K,N]) -> matmul(x,w)."""
    bb = relax.BlockBuilder()                                 # IR을 쌓는 빌더
    x = relax.Var("x", relax.TensorStructInfo([M, K], "float16"))
    w = relax.Var("w", relax.TensorStructInfo([K, N], "float16"))
    with bb.function("main", [x, w]):
        with bb.dataflow():                                  # 부수효과 없는 계산 블록
            y = bb.emit(relax.op.matmul(x, w)); gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


mod = make_matmul(M, K, N)

# ===========================================================================
banner(0, "큰 그림 — TVM의 IR은 2층이다 (Relax / TIR)")
note("TVM에서 모델은 두 단계의 IR(중간표현)로 표현된다:",
     "  • Relax = '그래프 IR'. 텐서와 연산(op)의 그래프. '무엇을' 계산하는지. (PyTorch 그래프와 비슷)",
     "  • TIR   = 'Tensor IR'. for 루프와 버퍼로 된 저수준 IR. '어떻게' 계산하는지. (C 루프와 비슷)",
     "IRModule = 이 함수들을 담는 컨테이너. Relax 함수가 TIR 커널(PrimFunc)을 호출하는 형태가 된다.",
     "우리 목표: relax.matmul → (Relax) → (TIR) → 64×64 타일 → NPU 명령. 아래에서 한 층씩 내려간다.")

# ===========================================================================
banner(1, "[1] Relax — '무엇을' 계산하나 (그래프 IR)")
note("Relax 함수는 입력 텐서(x,w)를 받아 dataflow 블록 안에서 op들을 emit한다.",
     "여기선 단 하나, R.matmul(x, w). 아직 루프도 타일도 없다 — 그냥 '행렬곱을 한다'는 선언.")
mod["main"].show()
note("읽는 법:",
     "  • x: R.Tensor((128,128), 'float16')  →  shape/타입이 붙은 텐서 인자",
     "  • lv = R.matmul(x, w)                →  중간결과(local var) lv",
     "  • R.output(gv)                       →  이 블록의 출력",
     "이게 '무엇을'이다. 이제 '어떻게'(루프)로 내린다.")

# ===========================================================================
banner(2, "[2] LegalizeOps → TIR — 행렬곱을 '숫자 단위 3중 루프'로")
note("개념: '패스(pass)'는 IR→IR 변환이다. relax.transform.LegalizeOps()는 Relax op을",
     "      그에 해당하는 TIR 커널(PrimFunc) + 그 커널을 부르는 call_tir 로 바꾼다.",
     "개념: PrimFunc = TIR 커널(하나의 함수). T.Buffer = 메모리 위의 텐서.",
     "      T.grid(a,b,c) = 3중 for 루프. block = 한 번의 계산 단위(반복변수+읽기/쓰기 영역).",
     "      T.axis.remap('SSR', ...) = 이 반복축들의 종류 선언: S=공간축(독립), R=리덕션축(합쳐짐).",
     "      with T.init(): = 리덕션 시작 시 누산기 초기화(C=0).")
modL = relax.transform.LegalizeOps()(mod)
gvar = [gv for gv, fn in modL.functions_items() if isinstance(fn, tir.PrimFunc)][0]
modL[gvar].show()
note("읽는 법 (핵심 두 줄):",
     "  with T.init(): matmul[v_i0,v_i1] = 0.0          ← C[i,j] = 0  (리덕션 초기화)",
     "  matmul[v_i0,v_i1] += x[v_i0,v_k]*w[v_k,v_i1]    ← C[i,j] += A[i,k]·B[k,j]  (정의 그대로)",
     "i0,i1는 S(공간), k는 R(리덕션). 즉 'k에 대해 합쳐서 C[i,j]를 만든다'.",
     "아직 타일링은 없다(128×128×128 전체 루프). 다음 단계가 이걸 64×64로 쪼갠다.")

# ===========================================================================
banner(3, "[3] tir.Schedule — 64×64 타일링 + tensorize (lowering의 심장)")
note("개념: tir.Schedule(mod)는 IR을 들고 '의미는 그대로, 루프 구조만 바꾸는' 변환을 적용하는 객체다.",
     "      (split/reorder/... 어떤 순서로 계산해도 수학 결과는 같다 — FP 반올림 빼고.)",
     "아래에서 변환을 '하나씩' 적용하며 TIR이 어떻게 바뀌는지 본다.")
name = gvar.name_hint
sch = tir.Schedule(modL)
blk = sch.get_block("matmul", func_name=name)     # 'matmul' 이라는 block을 잡는다
i, j, k = sch.get_loops(blk)                       # 그 block의 세 루프 i,j,k

print("---- (3a) split: 각 루프를 [바깥, 64]로 쪼갬 ----")
note("split(i,[None,64]) = i(128)를 io(2)×ii(64)로 분할. j,k도 동일.",
     "이제 ii,ji,ki = 64짜리 안쪽 루프(=한 타일), io,jo,ko = 타일 인덱스.")
io, ii = sch.split(i, [None, 64]); jo, ji = sch.split(j, [None, 64]); ko, ki = sch.split(k, [None, 64])
sch.mod[name].show()

print("---- (3b) reorder: 타일 인덱스를 바깥으로, 64블록을 안으로 ----")
note("reorder(io,jo,ko, ii,ji,ki): 'for 타일(io,jo,ko) { 64×64 블록(ii,ji,ki) }' 형태로.",
     "바깥 3중 = 어느 타일을 계산할지, 안쪽 3중 = 그 64×64 타일 내부 계산.")
sch.reorder(io, jo, ko, ii, ji, ki)
sch.mod[name].show()

print("---- (3c) decompose_reduction: 'C=0 초기화'를 k루프 밖으로 분리 ----")
note("개념: 누적 C[i,j]+=...는 'C=0' 다음 'k마다 더하기'다. decompose_reduction은",
     "      C=0 부분을 k루프 *밖*의 별도 블록(matmul_init_o)으로 떼어낸다.",
     "      → 타일마다 '한 번 0으로' + 'k타일마다 누적'으로 깔끔히 갈린다.")
init_blk = sch.decompose_reduction(blk, ko)
sch.mod[name].show()

print("---- (3d) tensorize: 64×64 블록을 'NPU 명령 하나'로 치환 ----")
note("개념: TensorIntrin = (desc, impl) 한 쌍을 이름으로 등록한 것.",
     "      desc = '이런 계산 패턴'(아래), impl = '그 자리에 넣을 대체물'(우리는 call_extern 마커).",
     "      tensorize(loop, name) = 그 loop 아래 64×64 sub-루프가 desc와 일치하면 통째로 impl로 바꾼다.",
     "      → 안쪽 64×64 루프가 사라지고 'npu_gemm_acc(이 타일)' 호출 하나가 남는다. 이게 PE 명령이 될 자리.")
print("   [desc] npu_gemm_acc 가 매칭하는 패턴 (64×64 C+=A·B):")
tir_backend._gemm_desc.show()
sch.tensorize(sch.get_loops(blk)[3], "npu_gemm_acc")          # 업데이트 블록 tensorize
sch.tensorize(sch.get_loops(init_blk)[2], "npu_fill_zero")    # 초기화 블록 tensorize
print("\n   [결과] 스케줄된 TIR — 바깥 타일 루프 + call_extern(npu_fill_zero/npu_gemm_acc):")
sch.mod[name].show()
note("읽는 법:",
     "  for i0_0,i1_0 in grid(2,2)         ← M타일 × N타일",
     "    block matmul_init_o + npu_fill_zero(C타일)   ← 이 출력타일 0초기화",
     "    for k_0 in range(2)               ← K타일(누적)",
     "      block matmul_update_o + npu_gemm_acc(C,A,B)  ← 64×64 누적 = PE 명령 1개",
     "이제 call_extern 8개(=2·2·2)가 우리 명령으로 바뀌면 된다.")

# ===========================================================================
banner(4, "[4] match_buffer / access_ptr — '심볼릭 타일 뷰' 해부")
note("위 결과의 각 block 안에 이런 줄이 있었다:",
     "  A = T.match_buffer(x[i0_0*64:+64, k_0*64:+64], (64,64), strides=(A_s0, 1))",
     "개념: match_buffer = 큰 버퍼 x의 64×64 '부분뷰'를 선언. 단 시작위치(elem_offset)와",
     "      행간격(A_s0)을 *심볼*로 둔다 (아직 i0_0,k_0가 숫자가 아니므로).",
     "  npu_gemm_acc(..., A.access_ptr, A_s0, ...) 처럼 호출에 넘긴다.",
     "개념: access_ptr = (버퍼데이터, elem_offset)로 만든 '그 타일 시작 포인터'.",
     "핵심: 여기 심볼 = { i0_0, i1_0, k_0(루프), A_s0/B_s0/C_s0(행간격), elem_offset(시작) }.",
     "      이 심볼들을 '실제 숫자'로 바꾸는 게 다음 단계(walker)다.")

# ===========================================================================
banner(5, "[5] _Walker — 심볼을 실제 G-buffer 주소로 풀고 명령을 emit (verbose 추적)")
note("개념: NPU ISA엔 루프가 없다. walker가 TIR을 '해석'하며:",
     "  ① for를 펼친다(unroll): for i0_0 in range(2) → i0_0=0, i0_0=1 두 번.",
     "  ② block 진입 시 반복변수(i0_0..)와 match_buffer 심볼을 *현재 숫자*로 바인딩.",
     "  ③ ev(expr): 심볼 대입 후 단순화 → 상수. 예) i0_0*64, i0_0=1 → 64.",
     "  ④ _bind_match: off = (행)·(행간격) + (열).  ptr = base + off = 실제 G-buffer 주소.",
     "  ⑤ call_extern을 만나면 gather + m_mul + 누적 명령을 emit.",
     "아래는 walker를 verbose로 돌린 '실시간 추적'. 각 줄의 주소가 ④에서 계산된 값이다.")

# _emit_tir_gemm 의 꼬리를 그대로 재현하되, 주소를 보기 쉽게 고정하고 verbose=True 로 walker 실행
a_off, b_off, c_off = 0, M * K, M * K + K * N            # x, w, 출력의 (가짜) G-buffer 시작
pf = sch.mod[name]
data_base = {pf.buffer_map[pf.params[0]].data: a_off,    # x.data  -> a_off
             pf.buffer_map[pf.params[1]].data: b_off,    # w.data  -> b_off
             pf.buffer_map[pf.params[2]].data: c_off}    # 출력.data-> c_off
print(f"   data_base(버퍼→실주소):  x@{a_off}  w@{b_off}  out@{c_off}\n")
asm = Asm()
mp = memplan.MemPlan(); mp.top = c_off + M * N           # 스크래치는 이 뒤부터 할당
wk = tir_backend._Walker(asm, mp, data_base, verbose=True)
wk.walk(pf.body); wk.flush()
note("추적 읽는 법:",
     "  gemm_acc C@..(s128) += A@..(s128)·B@..(s128)  ← match_buffer가 푼 실제 타일 주소들",
     "    gather NEW   = 처음 보는 (주소,stride) 타일 → 연속 스크래치로 복사",
     "    gather REUSE = 이미 모은 타일 → 재복사 안 함 (input reuse!)",
     "  A@? 가 i0_0,k_0로, B@? 가 k_0,i1_0로 정해지는 것을 주소 변화로 확인할 수 있다.")

print("---- 위 추적의 첫 gather(A@0)가 실제로 emit한 명령 (0~11번) ----")
print_program(asm, limit=12)
note("명령으로 보는 gather (한 행 복사 = VLEN→ADDR src→LOAD→VADD(a+0)→ADDR dst→SAVE):",
     "  입력1 주소: 0 → 128 → 256 ...  = 원본을 stride 128로 '띄엄띄엄' 읽음",
     "  출력 주소 : 연속(+64씩)         = 스크래치에 '촘촘히' 모음",
     "이게 strided→contiguous gather. m_mul은 연속 타일만 읽으니 필요하다.")

# ===========================================================================
banner(6, "[6] input reuse 효과 + 정확성(byte-exact)")
asm_tir, mp_tir = tir_backend.compile_func(mod)          # 정식 경로(verbose 없음)
asm_d, mp_d = driver.compile_func(mod["main"], tile=64)  # direct 오라클
st = cost.analyze(asm_tir, mp_tir); sd = cost.analyze(asm_d, mp_d)
print(f"  direct    : {sd['total']:>6} 명령  (gather복사 {sd['copy_ops']}, m_mul {sd['matmul_tiles']})")
print(f"  tir+reuse : {st['total']:>6} 명령  (gather복사 {st['copy_ops']}, m_mul {st['matmul_tiles']})")
note("같은 입력 타일을 '한 번만' gather(메모이제이션)해서 복사가 줄어든다. 차원이 클수록 절감↑.")
rng = np.random.default_rng(0)
A = _fp16(rng.standard_normal((M, K)) * 0.3); B = _fp16(rng.standard_normal((K, N)) * 0.3)
got = driver.run_module(mod, {"x": A, "w": B}, backend="tir")
def tiled_ref(A, B, T=64):
    C = None
    for kk in range(0, K, T):
        part = _fp16(A[:, kk:kk+T] @ B[kk:kk+T, :]); C = part if C is None else _fp16(C + part)
    return C
print(f"\n  TIR == tiled_fp16_ref ? {'예 (byte-exact)' if np.array_equal(got, tiled_ref(A,B)) else '아니오'}")

print("\n" + "=" * 74)
print("요약: Relax(무엇) → LegalizeOps(스칼라 루프) → Schedule(타일+tensorize, call_extern)")
print("      → Walker(루프 펼침 + match_buffer 심볼을 실주소로 + gather/m_mul/누적 명령) → NPU ISA.")
print("      주소는 _bind_match의 'off=행·행간격+열', reuse는 _gather_cached 메모이제이션에서 나온다.")
print("=" * 74)
