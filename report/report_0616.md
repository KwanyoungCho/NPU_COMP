# TVM 기반 NPU 컴파일러 — 구현 상세 및 커맨드 오버헤드 분석 (실제 HF 모델 기준)

> 작성일: 2026-06-16
> 대상 모델: **Llama 3.2 3B** (HuggingFace `transformers`) / 대상 하드웨어: 본 레포 NPU c-model(`_poc/mysim`)
> 본 문서는 처음 보는 사람도 이해하도록 개념부터 서술한다. 코드: `d_compiler/`.
> **오버헤드 분석은 실제 HF `LlamaDecoderLayer`를 frontend로 import한 그래프를 기준으로 한다(아래 §6.0).**

---

## 0. 한 문장 요약

> 신경망(Llama 레이어)을 **TVM Relax 그래프**로 받아 → NPU가 못 하는 연산을 가능한 연산 조합으로 바꾸고 → 행렬곱을 **64×64 타일**로 쪼개 → **NPU 명령어**로 생성 → `mysim`에서 돌려 정답과 대조하는 **컴파일러**다.

핵심 결과(실제 HF Llama 3.2 3B 레이어 1개, frontend import 그래프 기준):
- **패킹 전**: 유효 행렬곱이 prefill 5.7% / decode 3.4%뿐, **gather가 78.2% / 84.1%** 로 압도(나머지는 데이터이동·우회 오버헤드).
- **가중치 사전패킹(소프트웨어, ISA 불필요)** 으로 가중치 gather 제거 → **총 명령 prefill −65.5%(19.2M→6.6M) / decode −78.3%(15.1M→3.3M)**, 유효 비율 **5.7%→16.6% / 3.4%→15.8%** (§6.1, byte-exact).
- 패킹 후 남은 오버헤드는 **활성화 gather + scatter + K^T transpose**. 특히 **K^T transpose가 새 주요 비용**(prefill 15.8%, decode 32.1%).
- 추가 ISA(strided load/save, transpose unit) 시 **추가로 prefill −79.3% / decode −83.8%** 감축 가능.
- **decode는 토큰당 명령이 prefill의 63배**(패킹 전 100배에서 개선) — GEMV(M=1)+가중치 통과의 구조적 비효율.

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
| 6 | **가중치 사전패킹(weight pre-packing)** | matmul 상수 가중치를 **tile-blocked `[Kt,Nt,64,64]`** 로 호스트에서 미리 배치(memplan) → walker가 가중치 타일을 연속으로 직접 read | **가중치 gather 제거: prefill −65.5%, decode −78.3%** (§6.1, ISA 불필요, byte-exact) |

남은 잠재 최적화: 활성화 gather/scatter(→ strided load/save 또는 활성화 타일레이아웃), **K^T 전치**(패킹 후 최대 비용 → transpose ISA / Kᵀ-전치 캐시), input reuse를 schedule(cache_read)로 이전.

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

> **가중치 사전패킹이 기본 적용됨**: matmul의 상수 가중치는 memplan이 **tile-blocked `[Kt,Nt,64,64]`** 로 호스트에서 미리 배치하므로(§4), walker가 가중치 타일을 **연속으로 직접 read(가중치 gather 없음)**. 아래 §6.1이 그 전/후를, §6.2~6.5는 **패킹 적용 후(현재 기본)** 수치를 보여준다.

### 6.1 가중치 사전패킹 효과 (전/후) — ISA 없는 소프트웨어 최적화

가중치 gather를 호스트 사전패킹으로 제거한 결과(연산·정확성 불변, byte-exact):

| 지표 | prefill 전 | **prefill 후** | decode 전 | **decode 후** |
|---|---:|---:|---:|---:|
| 총 명령 | 19,211,527 | **6,628,615 (−65.5%)** | 15,066,897 | **3,270,417 (−78.3%)** |
| gather | 78.2% (15.0M) | **36.8% (2.44M)** | 84.1% (12.7M) | **26.8% (0.88M)** |
| 유효(matmul+누적) | 5.7% | **16.6%** | 3.4% | **15.8%** |

![G7](figs/g7_packing_effect.png)
*G7: 가중치 사전패킹 전/후 — (좌) 총 명령·gather 급감, (우) 유효 연산 비율 3배 상승. ISA 변경 없이 소프트웨어(memplan)만으로 달성.*

→ **가중치 gather는 ISA 없이 사라졌다.** 남은 오버헤드는 **활성화 gather(A) + scatter(출력) + transpose**이며, 이들이 다음 최적화 대상이다.

### 6.2 Prefill (S=128, 패킹 후) — 총 6,628,615 명령

| OP | 레이어 합 | % | 비고 |
|---|---:|---:|---|
| Q/K/V proj | 2,222,880 | 33.5% | A-gather+scatter+accum (B-gather 제거됨) |
| O proj | 1,309,440 | 19.8% | gather+scatter |
| **K^T transpose** | 1,048,576 | **15.8%** | 전치(패킹 무관) → 새 주요 비용 |
| gate/up proj | 920,064 | 13.9% | |
| down proj | 461,792 | 7.0% | |
| attn matmul (scores+ctx) | 301,632 | 4.6% | 활성화@활성화(미패킹) |
| reduce (norm/softmax) | 153,716 | 2.3% | |
| RoPE (slice/concat) | 131,072 | 2.0% | layout(HF식) |
| broadcast / elementwise | ~80,000 | 1.2% | |

**role 분포(후)**: gather **36.8%**, scatter 25.2%, transpose 15.8%, K-accumulate 9.8%, **matmul(유효) 6.9%**, reduce 2.3%, layout 2.0%, broadcast 0.8%. **유효 16.6%**.

![G1 prefill](figs/g1_per_op_hf_prefill.png)
*G1[HF prefill, 패킹 후]: projection의 빨강(gather)이 크게 줄고, K^T transpose(갈색)·scatter(주황)·accum(보라)이 상대적으로 커짐.*

![G5 prefill](figs/g5_share_and_mix_hf_prefill.png)
*G5[HF prefill, 패킹 후]: (좌) OP 비중 — Q/K/V·O proj·K^T transpose가 상위. (우) OP별 role 구성.*

![G4 prefill](figs/g4_useful_vs_overhead_hf_prefill.png)
*G4[HF prefill, 패킹 후]: 유효 16.6% vs 오버헤드 83.4%.*

### 6.3 Decode (M=1, L=128, 패킹 후) — 총 3,270,417 명령

| OP | 레이어 합 | % |
|---|---:|---:|
| **K^T transpose** | 1,048,576 | **32.1%** |
| Q/K/V proj | 667,056 | 20.4% |
| O proj | 654,912 | 20.0% |
| gate/up proj | 460,048 | 14.1% |
| down proj | 230,904 | 7.1% |
| attn matmul (scores+ctx) | 200,352 | 6.1% |
| reduce/broadcast/RoPE/elementwise | ~8,500 | 0.3% |

**role 분포(후)**: **transpose 32.1%**, gather 26.8%, scatter 25.0%, K-accumulate 9.3%, **matmul(유효) 6.5%**. **유효 15.8%**.
**decode 핵심**: 가중치 gather 제거 후 **K^T transpose(캐시 [L,HD] 전치)가 단일 최대 비용(32.1%)** 으로 부상 — 패킹이 손대지 못하는 부분. 다음 타깃은 transpose(전용 ISA 또는 Kᵀ-전치 캐시).

![G1 decode](figs/g1_per_op_hf_decode.png)
![G4 decode](figs/g4_useful_vs_overhead_hf_decode.png)
*G1/G4[HF decode, 패킹 후]: K^T transpose가 지배, 유효 15.8%.*

### 6.4 Prefill vs Decode — 토큰당 비용 (패킹 후)

![G6](figs/g6_prefill_vs_decode_hf.png)
*G6[HF, 패킹 후]: (좌) 토큰당 명령 — decode가 prefill의 **63배**(패킹 전 100배에서 개선). (우) 유효 비율 16.6% vs 15.8%.*

| 지표 | prefill (128토큰) | decode (1토큰) |
|---|---:|---:|
| 레이어 총 명령 | 6,628,615 | 3,270,417 |
| **토큰당 명령** | **51,786** | **3,270,417 (63×)** |
| 유효 연산 비율 | 16.6% | 15.8% |

**decode가 여전히 토큰당 63배인 이유**: 가중치 gather는 사라졌지만 ① 토큰 1개에도 **가중치 행렬 전체를 m_mul에 통과**시켜야 하고(이제 contiguous read지만 여전히 전부 읽음), ② M=1이 **64로 패딩**되어 PE의 1/64만 유효. → 근본 완화는 **배칭(M=B)** + KV캐시 + 양자화.

### 6.5 미지원 ISA 추가 시 절감 (패킹 후 기준, 상한 vs 현실)

패킹으로 가중치 gather가 이미 빠졌으므로, ISA의 추가 절감은 **남은 활성화 gather + scatter + transpose** 대상이다.

| 추가 ISA | 없애는 우회 | prefill 현실 절감 |
|---|---|---:|
| **strided load/save** | 남은 gather(A)/scatter 복사 | 큼(남은 62%의 데이터이동) |
| **transpose unit** | K^T 전치 | −15.8% (패킹 후 비중 큼) |
| row-reduce / broadcast / activation | 우회 연산 | 소폭 |
| **누적(현실)** | | **−79.3%** (남음 1,375,386) |

![G3 prefill](figs/g3_isa_waterfall_hf_prefill.png)
*G3[HF prefill, 패킹 후]: 남은 명령 6.63M에 대한 ISA 절감 워터폴(−79.3%).*

![G3 decode](figs/g3_isa_waterfall_hf_decode.png)
*G3[HF decode, 패킹 후]: −83.8%(→529,337). transpose 비중이 커 transpose ISA의 가치 상승.*

**정리**: 가중치 사전패킹(SW)으로 **prefill −65.5% / decode −78.3%** 를 ISA 없이 달성했고, 남은 오버헤드(활성화 gather·scatter·transpose)는 strided load/save·transpose ISA로 추가 제거 가능하다.

> reduce-max ISA는 절감이 아니라 안정 softmax(정확성)용이며 커맨드는 오히려 증가한다.

---

## 7. 결론 및 다음 단계

### 결론 (실제 HF 그래프 기준)
1. **패킹 전**: 명령의 ~94%(decode ~97%)가 데이터이동·우회 오버헤드, 유효 행렬곱 5.7%/3.4%. 근본 원인은 **NPU의 contiguous 전용 load/save**(넓은 행렬 타일을 gather/scatter 복사).
2. **가중치 사전패킹(소프트웨어, ISA 불필요)으로 가중치 gather를 제거**해 **prefill −65.5% / decode −78.3%** 달성, 유효 비율 ~3배 상승(5.7%→16.6% / 3.4%→15.8%). byte-exact.
3. 패킹 후 남은 오버헤드는 **활성화 gather + scatter + K^T transpose**. 특히 **K^T transpose가 새 최대 비용**(decode 32.1%) → 다음 타깃은 **transpose ISA / Kᵀ-전치 캐시**와 **strided load/save**(추가 −79~84%).
4. **decode는 토큰당 63배**(패킹 전 100배) — GEMV(M=1) 1/64 활용 + 가중치 통과. 배칭·KV캐시·양자화로 완화 대상.
5. 한계: literal 4D HF는 우리 2D 컴파일러로 import 불가 → **transformers와 rel 4.3e-7로 검증된 동등 2D 그래프**로 분석.

### 다음 단계
- ~~(SW) 가중치 사전패킹~~ → **완료**(§4 #6, §6.1: prefill −65.5% / decode −78.3%).
- **(SW/HW) K^T transpose 제거** — 패킹 후 최대 비용(decode 32.1%). Kᵀ-전치 캐시(SW) 또는 transpose ISA.
- **(SW) 활성화 gather/scatter 제거** — 활성화 타일레이아웃 전파(SW) 또는 strided load/save(HW).
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
