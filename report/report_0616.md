# TVM 기반 NPU 컴파일러 — 구현 상세 및 커맨드 오버헤드 분석 (실제 HF 모델 기준)

> 작성일: 2026-06-16
> 대상 모델: **Llama 3.2 3B** (HuggingFace `transformers`) / 대상 하드웨어: 본 레포 NPU c-model(`_poc/mysim`)
> 본 문서는 처음 보는 사람도 이해하도록 개념부터 서술한다. 코드: `d_compiler/`.
> **오버헤드 분석은 실제 HF `LlamaDecoderLayer`를 frontend로 import한 그래프를 기준으로 한다(아래 §6.0).**

---

## 0. 한 문장 요약

> 신경망(Llama 레이어)을 **TVM Relax 그래프**로 받아 → NPU가 못 하는 연산을 가능한 연산 조합으로 바꾸고 → 행렬곱을 **64×64 타일**로 쪼개 → **NPU 명령어**로 생성 → `mysim`에서 돌려 정답과 대조하는 **컴파일러**다.

핵심 결과(실제 HF Llama 3.2 3B 레이어 1개, frontend import 그래프 기준):
- 한 레이어 명령의 **유효 행렬곱은 prefill 5.7% / decode 3.4%뿐**, 나머지는 데이터 이동·우회 오버헤드.
- **gather(입력 타일을 연속으로 모으는 복사)가 prefill 78.2% / decode 84.1%** 로 압도적.
- 미지원 ISA(특히 strided load/save) 추가 시 명령을 **현실적으로 prefill −92.6% / decode −96.2%** 감축 가능.
- **decode는 토큰당 명령이 prefill의 100배** — GEMV(M=1)+가중치 재적재의 구조적 비효율.

---

## 1. 대상 하드웨어와 시뮬레이터 (배경)

### 1.1 NPU c-model
- **64×64 PE 행렬 배열**: 한 번에 64×64 타일 행렬곱.
- **G-buffer**: 평탄한 1D 메모리. FP16 저장, float32 계산, **저장(save) 시에만 FP16 반올림**.
- **명령은 contiguous(연속) 메모리만** load/save (stride 접근 없음 — 오버헤드의 핵심 원인).
- **루프·분기 없음.** → 컴파일러가 모든 반복을 펼쳐(unroll) 명령을 나열.

### 1.2 mysim (주어진 실행기)
`_poc/mysim.cpp`는 동작 시뮬레이터로 모든 원소를 stdout 출력한다. 기능 검증엔 좋지만 **실제 3072차원 전체 레이어를 끝까지 돌리는 건 비현실적**(출력량 폭발). 그래서 검증은 "작은 차원 전체 + 실가중치 조각". **mysim은 수정 불가**(주어진 ISA가 타깃). 사이클 모델이 없으므로 본 분석의 "오버헤드"는 **정적 명령 수**(프로그램 크기 ≈ fetch/issue 부하)이며 latency가 아니다.

---

## 2. 전체 컴파일러 구조

### 2.1 두 층의 IR
- **Relax** = 그래프 IR(텐서·op, "무엇을"). **TIR** = 루프 IR("어떻게"). 흐름: `Relax → TIR → NPU 명령(ISA)`.

### 2.2 모듈 지도 (`d_compiler/npu_compiler/`)
| 모듈 | 역할 |
|---|---|
| `frontend.py` | **PyTorch(HF) → Relax** (torch.export → from_exported_program → FoldConstant → import_legalize) |
| `import_legalize.py` | 고수준 op(silu/softmax/mean/rsqrt…) → 우리 primitive |
| `legalize.py` | 수동 빌더(rms_norm/rope/softmax/silu/swiglu) |
| `model.py` | Llama 레이어 Relax 조립 + numpy 참조 |
| `memplan.py` | 정적 메모리 배치(모든 텐서의 G-buffer 오프셋 고정) |
| `codegen.py` | Relax → NPU 명령. matmul은 TIR 백엔드로 위임, 나머지는 직접 emit. role 태깅·per-OP 기록(`emit_log`) |
| `tir_backend.py` | matmul 전용 TIR+tensorize + walker(TIR→ISA) |
| `isa.py` | 32비트 명령 인코더(mysim과 바이트 대조 검증) |
| `driver.py`/`runtime.py` | 묶기 + mysim 빌드·실행 |
| `cost.py` | 정적 분석기(명령 수, role별·OP별 집계) |

### 2.3 두 백엔드와 라우팅
- **direct**(`codegen.py`): 비-matmul op(elementwise/transpose/reduce/broadcast/slice/concat) + 행렬곱 오라클.
- **TIR**(`tir_backend.py`): 행렬곱 전용(타일+tensorize, input reuse).
- 라우팅: 모든 `relax.matmul` → TIR, 나머지 → direct.

---

## 3. 행렬곱의 TIR lowering (핵심) — 단계별 IR 예시

`relax.matmul` 한 줄이 NPU 명령이 되기까지를 **실제 IR**과 함께 따라간다. 예시는 `C[128,128] = A[128,128] @ B[128,128]` (64×64 타일이 M·N·K 각 2개 → 2·2·2 = **8개**). 아래 IR은 가독성을 위해 `T.int64(64)`→`64`, `T.` 접두사를 생략한 것이며, `python d_compiler/walkthrough_tir.py`로 원본을 직접 볼 수 있다.

> 큰 그림: TVM IR은 2층이다 — **Relax**(그래프, "무엇을") → **TIR**(루프, "어떻게"). 행렬곱은 Relax→TIR로 내린 뒤 **타일링(schedule)** 하고, 마지막에 **walker**가 NPU 명령으로 바꾼다.

### [1] Relax — "무엇을" 계산하나
```python
@R.function
def main(x: R.Tensor((128,128),"float16"), w: R.Tensor((128,128),"float16")):
    with R.dataflow():
        lv = R.matmul(x, w)        # 아직 루프·타일 없음 — "행렬곱을 한다"는 선언
        R.output(lv)
    return lv
```

### [2] LegalizeOps → TIR — 숫자 단위 3중 루프
TVM 패스가 행렬곱을 **정의 그대로의 스칼라 루프**로 내린다(`matmul`이 출력 버퍼 = C):
```python
@T.prim_func
def main(x: Buffer((128,128)), w: Buffer((128,128)), matmul: Buffer((128,128))):
    for i0, i1, k in grid(128, 128, 128):       # i0=M(행), i1=N(열), k=K(누적)
        with block("matmul"):
            v_i0, v_i1, v_k = axis.remap("SSR", [i0, i1, k])   # S=공간축, R=리덕션축
            with init():
                matmul[v_i0, v_i1] = 0.0                       # C = 0
            matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + x[v_i0, v_k] * w[v_k, v_i1]
```

### [3] tir.Schedule — 64×64 타일링 + tensorize (lowering의 심장)
`schedule_matmul`이 변환 4개를 차례로 적용한다.

**(3a) split** — 각 축 128을 `타일(2)×64블록`으로:
```python
for i0_0, i0_1, i1_0, i1_1, k_0, k_1 in grid(2,64, 2,64, 2,64):
    with block("matmul"):
        v_i0 = axis.spatial(128, i0_0*64 + i0_1)    # 진짜 행 좌표 = 타일·64 + 블록내
        v_i1 = axis.spatial(128, i1_0*64 + i1_1)
        v_k  = axis.reduce (128, k_0*64 + k_1)
        ...  # 계산은 [2]와 동일, 루프만 쪼갬
```

**(3b) reorder** — 타일 인덱스(io,jo,ko)를 바깥, 64블록(ii,ji,ki)을 안으로:
```python
for i0_0, i1_0, k_0,  i0_1, i1_1, k_1 in grid(2,2,2,  64,64,64):
    ...   # 바깥 3중 = 어느 타일을 계산할지(M,N,K), 안쪽 3중 = 그 64×64 타일 내부
```
→ 안쪽 64×64×64가 정확히 PE 하나가 하는 일이 됨. 이게 M-N-K 루프 순서를 확정.

**(3c) decompose_reduction** — "C=0 초기화"를 K루프 밖의 별도 블록으로 분리:
```python
for i0_0, i1_0 in grid(2, 2):                        # 출력 타일 (M,N)
    for i0_1_init, i1_1_init in grid(64, 64):        # ── 블록1: 초기화
        with block("matmul_init"):  matmul[v_i0,v_i1] = 0.0
    for k_0, i0_1, i1_1, k_1 in grid(2, 64, 64, 64): # ── 블록2: 누적 (k_0 = K타일)
        with block("matmul_update"): matmul[v_i0,v_i1] += x[v_i0,v_k]*w[v_k,v_i1]
```
→ "출력 타일마다 한 번 0으로 + K타일마다 누적" 구조(우리 NPU 흐름: fill 1회 + gemm 여러 번).

**(3d) tensorize** — 안쪽 64×64×64 루프를 **명령 1개로 치환**. 먼저 매칭할 패턴(desc):
```python
@T.prim_func
def _gemm_desc(a, b, c):                              # "64×64 C += A·B" 패턴
    A = match_buffer(a, (64,64), "float16", strides=("A_s0", 1))
    B = match_buffer(b, (64,64), "float16", strides=("B_s0", 1))
    C = match_buffer(c, (64,64), "float16", strides=("C_s0", 1))
    for i, j, k in grid(64, 64, 64):
        C[vi,vj] = C[vi,vj] + A[vi,vk]*B[vk,vj]
```
`tensorize`가 일치하는 안쪽 루프를 desc의 대체물(impl=`call_extern` 마커)로 바꾼 **최종 스케줄 TIR**:
```python
for i0_0, i1_0 in grid(2, 2):                         # 출력 타일 (M×N = 2×2)
    with block("matmul_init_o"):
        C = match_buffer(matmul[i0_0*64 : i0_0*64+64,  i1_0*64 : i1_0*64+64],
                         (64,64), strides=("C_s0", 1))
        call_extern("npu_fill_zero", C.access_ptr, C.strides[0])   # 이 타일 0초기화
    for k_0 in range(2):                              # K 타일 (누적)
        with block("matmul_update_o"):
            A = match_buffer(x[i0_0*64:+64, k_0*64:+64], (64,64), strides=("A_s0",1))
            B = match_buffer(w[k_0*64:+64, i1_0*64:+64], (64,64), strides=("B_s0",1))
            C = match_buffer(matmul[i0_0*64:+64, i1_0*64:+64], (64,64), strides=("C_s0",1))
            call_extern("npu_gemm_acc", C.access_ptr,C_s0, A.access_ptr,A_s0, B.access_ptr,B_s0)
```
→ 안쪽 64³ 루프가 사라지고 **`call_extern` 8개(=2·2·2)** 가 남았다. 이게 PE 명령이 될 자리.

### [4] match_buffer / access_ptr — '심볼릭 타일 뷰'
위 결과의 `A = match_buffer(x[i0_0*64:+64, k_0*64:+64], (64,64), strides=("A_s0",1))`는 큰 x의 **64×64 부분뷰**를 선언하되, **시작 위치(elem_offset)와 행간격(`A_s0`)을 심볼**로 둔다(아직 `i0_0,k_0`가 숫자가 아니므로). `call_extern(..., A.access_ptr, A_s0, ...)`이 그 포인터·stride를 호출에 넘긴다. **이 심볼을 실제 숫자로 푸는 게 다음 단계.**

### [5] _Walker — 심볼을 실주소로 풀고 명령 emit ("TIR→ISA codegen")
walker는 스케줄 TIR을 **컴파일 타임에 해석**하며 명령을 받아 적는다. NPU엔 루프가 없으니 **모든 `for`를 펼친다**.
- **루프 펼침**: `for i0_0 in range(2)` → i0_0=0,1 각각 처리.
- **주소 공식**: `off = (행 시작) × (행 폭) + (열 시작)`, 절대주소 = 버퍼 시작 + off.

`x@0, w@16384, out@32768`에 배치된 경우의 **실제 walker 추적**(verbose):
```
[walker] gemm_acc  C@32768(s128) += A@0(s128)     · B@16384(s128)   # i0_0=0,i1_0=0,k_0=0
   gather NEW   @0(s128)     -> scratch@53248 (copy 64 rows)        # A타일을 연속으로 모음
   gather NEW   @16384(s128) -> scratch@57344
[walker] gemm_acc  C@32768   += A@64(s128)         · B@24576(s128)  # k_0=1: A off=0·128+64=64
[walker] gemm_acc  C@32832   += A@0                · B@16448        # i1_0=1: C off=+64
   gather REUSE @0           -> scratch@53248 (memoized)            # A@0 재사용 (input reuse!)
```
- `A@8192`처럼 보이는 주소는 `off = i0_0·64·128 + k_0·64` (예: i0_0=1,k_0=0 → 64·128 = 8192).
- **gather**가 필요한 이유: 행렬 폭이 128이라 64×64 타일은 메모리에서 행간격 128로 흩어져 있는데, NPU `load`는 연속만 읽으므로 연속 스크래치로 복사한다. 같은 타일은 **한 번만**(REUSE).

그 첫 gather가 실제로 만든 **명령어**(한 행 복사):
```
0  VLEN 64                  # 64개 복사
1  ADDR 입력1 = 0           # 원본 행 시작 (r=0)
3  LOAD
4  VADD a+0  (복사)
5  ADDR 출력 = 53248        # 스크래치
7  SAVE
8  VLEN 64
9  ADDR 입력1 = 128         # r=1 → 원본을 stride 128로 띄엄띄엄 읽음
...                         # 출력은 +64씩 연속 (strided→contiguous)
```

### [6] 결과 = NPU ISA
`relax.matmul` → (스칼라 루프) → (타일+tensorize, `call_extern` 8개) → (walker: 펼침 + `off=행·폭+열` 실주소화 + gather/m_mul/누적) → **평탄한 NPU 명령 목록**. 주소는 walker의 `_bind_match`, 재사용은 `_gather_cached`(메모이제이션)에서 나온다.

---

## 4. 컴파일러가 적용한 최적화

| # | 최적화 | 내용 | 효과 |
|---|---|---|---|
| 1 | **input reuse** | 같은 `(주소,stride)` 입력 타일을 한 번만 gather 후 재사용 | 중복 gather 제거(3B matmul −72%) |
| 2 | **fill fusion** | C 0초기화를 표시만, 첫 누적이 덮어씀 | 0초기화 명령 제거 |
| 3 | **contiguous-skip** | 이미 연속(차원=64)이면 gather 생략 | 불필요 복사 제거 |
| 4 | **reduce/broadcast 전용 op 분리** | mean/softmax의 reduce·broadcast를 `relax.sum`/`relax.broadcast_to`로 → 모든 `relax.matmul`은 진짜 GEMM → TIR 단독 | matmul 경로 단일화 |
| 5 | **비-64 패딩(fallback 제거)** | 비-64배수 차원을 TIR 경로가 직접 패딩(byte-exact) | direct fallback 제거 |

미구현(최대 잠재 절감): **가중치 사전패킹** — 가중치는 상수이므로 tile-blocked 연속 배치로 미리 깔면 §6의 gather 대부분을 ISA 없이 제거 가능.

---

## 5. 정확성 검증 (실제 HF 포함)

- **실제 `transformers.LlamaDecoderLayer`(Llama 3.2 3B)와 수치 동등성**: 우리 2D 레이어에 실제 HF 가중치를 복사해 출력 비교 → **최대 상대오차 4.3e-7 (동일).** 구성요소별로도 RMSNorm 0, MLP 0, attention 5e-7로 일치.
- frontend로 import한 전체 Llama 블록(작은 차원) end-to-end: torch 대비 ~2.5%.
- 실제 Llama 3.2 3B 가중치 조각(q/k/v/o proj, FFN): ≤0.36%.
- 행렬곱: 임의 차원에서 TIR=direct=tiled_fp16_reference (byte-exact).
- (제외) softmax의 **reduce-max(max 빼기)** — ISA에 없어 생략.

---

## 6. 커맨드 오버헤드 분석 (실제 HF 그래프 기준)

### 6.0 분석 기반 — "진짜 HF"를 쓰는 방법과 그 한계

**literal HF 레이어는 우리 컴파일러로 그대로 들어오지 않는다.** 실제 `transformers.LlamaDecoderLayer`는 ① `torch.export`가 `_assert_tensor_metadata` 같은 우리 import 미지원 노드를 내고, ② 어텐션이 **4D 배치 텐서(batch,head,seq,dim)** 인데 **우리 NPU codegen은 2D 전용**(NPU 타일 모델이 2D)이라 4D matmul/transpose/softmax를 처리할 수 없다. (실측: import 시 `AssertionError: Unsupported function type _assert_tensor_metadata.default`.)

그래서 분석은 **HF Llama 3.2 3B의 연산을 head 루프로 펼친 2D 레이어**를 쓰되, **그 레이어가 실제 HF와 수치적으로 동일함을 증명**(§5, rel 4.3e-7)하고 **실제 frontend(torch.export→import_legalize)로 import한 그래프**를 측정한다. 즉:
- 연산(수식)은 **실제 transformers와 검증된 동일**.
- 그래프는 **실제 import 경로**로 생성(수동 빌더 아님). per-OP는 codegen의 `emit_log`로 binding별 귀속.
- HF 충실 포인트: RoPE는 **실제 HF식 slice+neg+concat**(→ layout), Kᵀ 전치는 **KV-head당 1회**(HF 배치 동작과 동일).

두 시나리오: **prefill**(S=128 토큰 동시), **decode**(M=1 토큰 생성, KV 캐시 길이 L=128; NPU 특성상 M=1은 64로 패딩).

### 6.1 Prefill (S=128) — 구성요소별 (총 19,211,527 명령)

| OP | 레이어 합 | % | 비고 |
|---|---:|---:|---|
| gate/up proj | 7,211,520 | 37.5% | 거의 gather |
| Q/K/V proj | 4,188,960 | 21.8% | 거의 gather |
| down proj | 3,607,520 | 18.8% | 거의 gather |
| O proj | 2,489,088 | 13.0% | gather+scatter |
| K^T transpose | 1,048,576 | 5.5% | 전치 |
| attn matmul (scores+ctx) | 301,632 | 1.6% | gather+scatter |
| reduce (norm/softmax) | 153,716 | 0.8% | ones-mm |
| RoPE (slice/concat) | 131,072 | 0.7% | layout(HF식) |
| broadcast (norm/softmax) | 50,496 | 0.3% | ones-mm |
| elementwise (norm/resid/SiLU/softmax) | ~28,946 | 0.2% | |

**role 분포**: gather **78.2%**, scatter 8.7%, transpose 5.5%, K-accumulate 3.4%, **matmul(유효) 2.4%**, reduce 0.8%, layout 0.7%, broadcast 0.3%, elementwise 0.2%.
**유효 연산(matmul+누적) = 5.7%**, 나머지 94.3%가 오버헤드.

![G1 prefill](figs/g1_per_op_hf_prefill.png)
*G1[HF prefill]: 구성요소별 명령 수(크기 tier별 선형). 큰 projection은 막대 대부분이 빨강(gather), 초록(matmul)은 바닥의 얇은 띠.*

![G5 prefill](figs/g5_share_and_mix_hf_prefill.png)
*G5[HF prefill]: (좌) OP별 레이어 비중 — 상위 4개 projection이 ~91%. (우) OP별 role 구성(100%) — projection ~90% gather, O proj 절반 scatter, K^T 100% transpose, RoPE 100% layout, reduce/broadcast/elementwise 각각.*

![G2 prefill](figs/g2_role_dist_hf_prefill.png)
*G2[HF prefill]: gather 78.2% 압도, 유효 matmul 2.4%.*

![G4 prefill](figs/g4_useful_vs_overhead_hf_prefill.png)
*G4[HF prefill]: 유효 5.7% vs 오버헤드 94.3%.*

### 6.2 Decode (M=1, L=128) — 구성요소별 (총 15,066,897 명령)

| OP | 레이어 합 | % |
|---|---:|---:|
| gate/up proj | 6,751,504 | 44.8% |
| down proj | 3,376,632 | 22.4% |
| Q/K/V proj | 1,846,704 | 12.3% |
| O proj | 1,834,560 | 12.2% |
| K^T transpose | 1,048,576 | 7.0% |
| attn matmul (scores+ctx) | 200,352 | 1.3% |
| reduce/broadcast/elementwise | ~8,000 | 0.0% |

**role 분포**: gather **84.1%**, transpose 7.0%, scatter 5.4%, K-accumulate 2.0%, **matmul(유효) 1.4%**, 나머지 ~0%.
**유효 연산 = 3.4%**(prefill보다 더 낮음).

**decode 특징**: M에 의존하는 부분(RMSNorm/softmax/A-gather)은 급감하지만 **가중치 gather는 M과 무관**해 거의 그대로 → gather 비중이 84.1%로 더 커진다. K^T transpose(캐시 [L,HD] 전치)는 prefill과 동일(1.05M)해 상대 비중이 7.0%로 상승.

![G1 decode](figs/g1_per_op_hf_decode.png)
*G1[HF decode]: projection의 gather가 더욱 지배. softmax/RMSNorm은 거의 소멸(M=1).*

![G4 decode](figs/g4_useful_vs_overhead_hf_decode.png)
*G4[HF decode]: 유효 3.4% vs 오버헤드 96.6%, gather가 prefill보다 더 큼.*

### 6.3 Prefill vs Decode — 토큰당 비용

![G6](figs/g6_prefill_vs_decode_hf.png)
*G6[HF]: (좌) 토큰당 명령 — decode가 prefill의 **100배**. (우) 유효 비율 — decode 3.4% < prefill 5.7%.*

| 지표 | prefill (128토큰 동시) | decode (1토큰) |
|---|---:|---:|
| 레이어 총 명령 | 19,211,527 | 15,066,897 |
| **토큰당 명령** | **150,090** | **15,066,897 (100×)** |
| 유효 연산 비율 | 5.7% | 3.4% |
| gather 비율 | 78.2% | 84.1% |

**왜 decode가 토큰당 100배인가**: 토큰 1개를 만들려고 **가중치 전체를 한 번 읽어야** 한다(projection의 gather는 M과 무관). 게다가 M=1이 **64로 패딩**되어 64×64 PE의 **1/64 행만 유효**. GEMV(행렬×벡터)를 행렬 엔진에 태우는 구조적 비효율. (배칭·KV캐시·양자화로 완화하는 영역. 우리 컴파일러엔 decode 전용 경로가 없어 동일 primitive로 추정한 per-token 비용이다.)

### 6.4 미지원 ISA 추가 시 절감 (상한 vs 현실)

각 미지원 연산을 전용 ISA로 대체할 때 줄어드는 명령. **상한** = 우회 role을 0으로 가정. **현실** = 대체 ISA가 새로 내는 명령(replacement)을 차감.

| 추가 ISA | 없애는 우회 | prefill 현실 절감 |
|---|---|---:|
| **strided load/save** | gather/scatter 복사 | **−86.6%** |
| transpose unit | Kᵀ 원소복사 | −5.5% |
| row-reduce(sum) | reduce=ones-mm | −0.4% |
| broadcast | broadcast=ones-mm | −0.15% |
| native activation | SiLU 체인 | −0.0% |
| **누적(현실)** | | **−92.6%** (남음 1,424,538) |

![G3 prefill](figs/g3_isa_waterfall_hf_prefill.png)
*G3[HF prefill]: 현실 절감 워터폴 — strided load/save 하나로 절벽(19.2M→2.6M).*

![G3 decode](figs/g3_isa_waterfall_hf_decode.png)
*G3[HF decode]: gather 비중이 더 커 현실 절감 누적 **−96.2%**(→575,417).*

**실행 시사점**: gather의 대부분은 **가중치 타일 gather**다. 가중치는 컴파일 타임 상수이므로 **호스트 사전패킹(소프트웨어, ISA 불필요)** 만으로도 이 86%의 큰 몫을 제거할 수 있다 — "−86%"는 ISA 없이도 상당 부분 달성 가능한 목표.

> reduce-max ISA는 절감이 아니라 안정 softmax(정확성)용이며 커맨드는 오히려 증가한다.

---

## 7. 결론 및 다음 단계

### 결론 (실제 HF 그래프 기준)
1. 명령의 **~94%(decode ~97%)가 데이터 이동·우회 오버헤드**, 유효 행렬곱은 prefill 5.7% / decode 3.4%뿐.
2. 근본 원인은 **NPU의 contiguous 전용 load/save** — 넓은 행렬 타일을 들어올 때(gather)·나갈 때(scatter) 복사. **strided load/save가 단일 최대 개선(−86%)**.
3. gather의 대부분이 **가중치**이므로 **가중치 사전패킹(소프트웨어)** 으로 ISA 없이도 큰 폭 절감 가능.
4. **decode는 토큰당 100배** — GEMV(M=1)+가중치 재적재. 배칭·KV캐시·사전패킹·양자화로 완화 대상.
5. 한계: literal 4D HF는 우리 2D 컴파일러로 import 불가 → **검증된 동등 2D 그래프**로 분석.

### 다음 단계
- **(SW) 가중치 사전패킹** 구현 → "ISA 없는 절감" 막대 추가.
- **(기능) 2D-batched 어텐션 지원** 또는 4D import 경로 → literal HF 직접 컴파일.
- **(SW) input reuse를 schedule(cache_read)로** + weight-stationary dataflow(PE 입력버퍼 재사용).
- **(분석) SEQ 스윕**(128→2048→4096) → attention(SEQ²) vs projection(SEQ) 역전 지점.
- **(기능) decode 경로**(KV 캐시 재사용, 배칭) 구현 후 재측정.
- **(정확성) reduce-max** 지원 시 안정 softmax.

---

### 부록 A. 재현
```bash
# 실제 HF 그래프 기준 분석 + 그래프 (prefill/decode)
python d_compiler/analyze_hf.py          # report/figs/*_hf_*.png
# transformers 동등성 검증은 analyze_hf.py의 PrefillLayer를 HF 가중치로 비교(§5)
# matmul lowering 단계별 IR
python d_compiler/walkthrough_tir.py
```

### 부록 B. 측정 한계
- 정적 명령 수만 측정(사이클 latency·메모리 대역폭 미반영) → 상대·구조 분석에 유효.
- literal 4D HF는 우리 2D codegen 미지원 → transformers와 **rel 4.3e-7로 검증된 2D 동등 그래프** 사용.
- decode는 컴파일러 전용 경로가 없어 동일 primitive로 추정(KV 길이 L=128 가정).
- ISA 절감 "현실" 모델의 replacement 가정은 `analyze_hf.py`의 `feats`에 명시.
