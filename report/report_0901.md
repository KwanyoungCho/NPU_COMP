# NPU 컴파일러 종합 보고서 — 2026-09-01

이 문서는 현재의 **표준 TVM 기반 LLM 컴파일러**를 처음 보는 사람을 위해 쓴 것이다.
컴파일러가 무엇으로 구성되어 있고, 각 부분이 무슨 일을 하며, 모델 하나가 어떤
단계를 거쳐 NPU 명령어가 되는지("lowering")를 동작 중심으로 설명한다.
코드 위치는 참조용으로만 적는다. branch는 `cmodel-v09`.

---

## 0. 한 장 요약

```
HF 체크포인트           우리가 정의한 모델 구조
     │                        │
     ▼                        ▼
[1] 프론트엔드   relax.frontend.nn.Module  ──export──▶  Relax IRModule
[2] 그래프 pass  (전부 TVM 표준; 양자화 pass는 우리 것 하나)
[3] 메모리 계획  StaticPlanBlockMemory ──▶ 평면 정적 주소
[4] 스케줄       tir.Schedule (타일링 + SRAM staging + tensorize)
[5] codegen      TIR을 걸어가며 v09 명령어 word 방출
[6] 링크         커널들을 하나의 직선 명령 스트림으로 연결 + peephole
[7] 실행         v09 C-model (시뮬레이터)
```

핵심 설계 원칙 두 가지:

1. **표준 TVM 경로를 기본으로 한다.** 모델 정의부터 메모리 계획까지 TVM이
   제공하는 표준 단계를 그대로 쓰고, 우리 것은 표준이 마련해 둔 확장 지점
   (커스텀 legalize map, 커스텀 pass, tensorize 인트린식)에 끼워 넣는다.
   손으로 짠 기존 컴파일러(`backend_v09`)는 **oracle**(정답 비교용)로만 남는다.
2. **타깃 기계가 특이하다는 사실을 마지막 단계에만 가둔다.** v09는 분기·루프
   명령이 없는 직선(straight-line) 기계이고, 런타임도 할당기도 없다. 프로그램
   전체가 "명령어 word 목록 하나 + 초기 메모리 이미지 하나"다. 이 특성은
   [5]~[6]에서만 나타나고, 그 앞은 전부 보통의 TVM이다.

### 최종 검증 상태 (전체 깊이, 실제 체크포인트, C-model 실행)

| 모델 | 결과 | 규모 |
|---|---|---|
| Llama 3.2 3B (28층) | **HF golden token 358 일치 + HF logits cosine 0.9999927** (기존 golden 0.9999881 상회) | A1 적용 **15.3M word** (전 31.2M, −51.0%) · 6.1 GiB image |
| Qwen3-4B (36층) | **HF golden token 358 일치 + HF logits cosine 0.9999922** | A1 적용 **21.5M word** (전 42.9M, −49.8%) · 7.7 GiB image |
| Gemma 4 E2B (35층) | **HF golden token 108 일치 + HF logits cosine 0.9997845** | 13.7M word · 4.3 GiB image · 실행 247s |
| W8A16 양자화 | 타일 규모에서 **numpy mirror와 bit-exact**, tiny 모델 cosine 0.999998 | — |

---

## 1. 타깃 기계 (v09) — 컴파일러가 상대하는 것

컴파일러를 이해하려면 기계의 다섯 가지 특징을 먼저 알아야 한다.

**(a) 직선 프로그램.** 분기도 루프도 없다. 컴파일 시점에 모든 루프가 완전히
펼쳐지고, 모든 주소가 상수로 박힌다. "for 64번"은 명령 64벌이다. 이것이 이
컴파일러의 거의 모든 특이한 결정(전량 펼침, dead-store 제거의 완전성,
스냅샷 디버깅)의 근원이다.

**(b) 두 층 메모리.** 전역 메모리는 32-bit 셀 단위 주소의 16 GiB 공간이고
**연산 유닛이 직접 읽을 수 없다**. 연산은 8 MiB SRAM(4-bit nibble 주소)에서만
일어난다. 전역↔SRAM 이동은 DMA(GLOAD/GSTORE) 뿐이며, DMA는 **dtype을 모르고**
셀 단위로만 옮긴다. 2차원 전송(행 수 + 전역 행 간격)을 지원해서 큰 행렬의
타일 하나를 명령 하나로 가져올 수 있다.

**(c) 행렬 유닛.** 64×64 타일 단위 행렬곱. 내부 누적은 **FP32**, 저장 시 FP16
반올림. MAC 비트로 연속 호출을 누산기에 이어붙일 수 있어서 K축 타일들을
저장 없이 합산한다. 이 "FP32 내부 누적"이 뒤에 나올 정확도 판정 기준의
근거가 된다.

**(d) 벡터 유닛.** 256-lane, 한 명령이 **연산 하나**를 벡터 전체에 적용한다
(add/mul/exp/sqrt/…). 복합식(예: silu의 `x·σ(x)`)은 명령 여러 개로 풀어야
하고, 중간값은 SRAM에 놓아야 한다. INT8→FP16 변환(VDEQUANT), FP16→INT8
양자화(VQUANT)도 여기 있다.

**(e) 서술자(descriptor) 상태.** 피연산자 주소·모양·dtype은 명령 인자가 아니라
**별도 설정 명령으로 채워 두는 레지스터**다(0x80 주소, 0x82 벡터 길이,
0x88/0x89 행·열+dtype, 0x8A/0x8B scale 주소). 상태는 sticky라서 안 바꾸면
유지된다 — 이 성질이 최적화(A1)와 버그(INT8 dtype leak)의 양면이 된다.

---

## 2. [1] 프론트엔드 — 모델을 어떻게 정의하나

**표준 `relax.frontend.nn.Module`로 모델을 짠다.** PyTorch의 `nn.Module`과 거의
같은 감각이다: `nn.Linear`, `nn.RMSNorm`, `nn.ModuleList`를 조립해 `prefill`
메서드를 쓰면, `export_tvm()`이 그것을 **Relax IRModule**(TVM의 그래프 IR)로
바꿔 준다. 이 시점의 그래프는 `matmul`, `softmax`, `reshape` 같은 고수준
연산의 나열이고, 아직 기계와 아무 상관이 없다 — 실제로 **같은 모듈을 그대로
llvm으로 빌드해 CPU에서 돌릴 수 있고**, 우리는 이것을 검증에 계속 쓴다.

세 family가 있다 (`npu_compiler/nn_models/`):

- **Llama** (`llama.py`) — 기준 구현. GQA 어텐션을 `[seq, head, dim]` 3차원
  텐서로 쓰고, RoPE는 half-duplicate 주파수 + rotate_half 관례(기존 검증
  경로와 동일 수치)를 따른다. Llama-3 주파수 스케일링 포함.
- **Qwen3** (`qwen3.py`) — Llama + **head별 Q/K RMSNorm**(RoPE 전에 각 head를
  head_dim 축으로 정규화). 나머지는 Llama의 MLP/RoPE/mask를 그대로 재사용.
- **Gemma 4 E2B** (`gemma.py`) — 델타가 크다. 층마다 어텐션이 두 종류
  (sliding-window / full; head_dim 256 vs 512, RoPE theta 다름, full은
  proportional RoPE — 각도의 앞 1/4만 비영이고 나머지는 cos=1/sin=0이라
  통과), 마지막 20개 층은 K/V를 자기가 만들지 않고 **앞선 owner 층의 K/V를
  공유**, 층마다 per-layer embedding 행을 gate→projection으로 주입, 층당
  norm 5개, weight 없는 V-norm, scale 1.0 어텐션, tanh-GELU MLP, 층별 출력
  스칼라.

**호스트가 미리 만들어 주는 입력**이 몇 개 있다. embedding lookup(토큰 id →
행), RoPE cos/sin 표, causal/banded mask, (Gemma) per-layer embedding 표 행.
이들은 전부 **데이터 의존 gather**라서 — 주소가 토큰 값에 달려 있다 — 모든
주소가 컴파일 시점에 정해져야 하는 이 기계에서는 프로그램 안에 넣을 수 없다.
그래서 그래프의 입력 텐서로 들어온다.

각 family는 같은 모양의 인터페이스를 내놓는다: `model_config`(체크포인트
config → 빌드 인자), `build_prefill`(IRModule 생성), `load_params`(체크포인트
텐서를 파라미터 순서대로), `runtime_inputs`(위의 호스트 입력들). 덕분에 실행
스크립트 `run_nn_npu.py --model {llama,qwen3,gemma,hf}` 하나가 전부를 다룬다.

### 2.5 네 번째 프론트엔드 — HF 모델 자체가 입력 (`nn_models/hf.py`)

위의 세 frontend는 구조를 우리가 다시 쓰고 가중치만 HF에서 가져온다. 네 번째는
**HF 모델 자체를 입력으로 쓴다**: `transformers`가 모델을 만들고, HF의
`forward`를 `torch.export`가 그대로 추적하고, TVM 표준 torch 프론트엔드
(`from_exported_program`)가 그 추적을 Relax로 바꾼다. 구조도 수치도 HF가
원본이며, 아키텍처를 손으로 다시 쓰는 부분이 없다.

추적과 표준 파이프라인 사이에 필요한 처리(전부 이유가 있다):

1. **mask·position을 입력으로 뺀 뒤 상수로 bind** — HF는 forward 안에서
   mask/position을 만드는데 그 코드는 데이터 의존이라 기계가 못 돌린다.
   4D additive mask를 넘기면 HF가 그대로 통과시키고, position을 상수로
   bind하면 `FoldConstant`가 **rotary cos/sin 표 전체를 컴파일타임에 계산**해
   int64가 기계에 들어가지 않는다.
2. **가중치도 상수로 bind** — 어차피 프로그램 이미지에 박히는 기계다.
   덕분에 nn.Linear의 전치도 컴파일타임에 접힌다(LiftTransformParams가 주던
   이득을, 그 pass의 파라미터 장부 정리와 싸우지 않고 얻는다).
3. **추적된 RMSNorm을 op으로 복원** — HF는 fp32에서 "제곱→평균"으로
   정규화하는데 이를 fp16으로 내리면 실모델 폭(3072)에서 합이 넘친다.
   패턴 재작성으로 `relax.nn.rms_norm`을 복원하면 우리 legalize의 안전 전개
   (제곱 전에 1/√D)가 그대로 적용된다.
4. **잔여 fp32 섬 강등** — softmax/rotary의 fp32 upcast를 fp16으로. 손으로
   모델을 쓸 때 내렸던 결정을 pass로 옮긴 것이다.
5. **matmul의 unit batch 제거** — 추적본은 모든 텐서에 선두 1 차원을 달고
   다니는데, matmul 스케줄은 마지막 세 루프를 타일링하므로 벗겨 준다
   (reshape는 view가 되어 무비용).

검증: tiny HF Llama가 **HF torch 자신의 출력과** llvm 0.999999 / NPU
1.000000으로 일치, 실 체크포인트 1층이 llvm 대비 0.99856(손작성 1층 수치와
동일), 그리고 **전체 28층이 C-model에서 golden token 358 + HF logits cosine
0.9999927** — 손작성 frontend의 최종 수치와 완전히 같다
(16,021,618 word · 2,134 kernel · 링크 7,033s · 실행 275s).

---

## 3. [2] 그래프 pass — 그래프를 다듬는 표준 단계들

`npu_compiler/tvm_pipeline.py`의 `graph_pipeline`이 표준 pass들을 순서대로
돌린다. 각각이 하는 일:

- **CanonicalizeBindings / EliminateCommonSubexpr / FoldConstant** — 정리.
  같은 부분식 합치기, 상수 접기(예: RoPE 관련 상수 계산이 컴파일 시점에 끝남).
- **RewriteDataflowReshape** — reshape를 실제 복사가 아니라 **view**로 바꾼다.
  4차원 어텐션이 reshape/permute를 많이 쓰므로 필수.
- **LiftTransformParams** — **파라미터에만 의존하는 계산을 그래프에서 떼어**
  별도의 `prefill_transform_params` 함수로 옮긴다. 대표적으로 `nn.Linear`의
  weight 전치: 매 토큰마다 기계에서 전치하는 대신, 호스트가 모델 로드 때 한 번
  실행한다. 활성 메모리 풀이 1.6 GiB → 0.7 MiB로 줄었던 pass다. 양자화(§5)도
  이 pass에 올라탄다.
- **LegalizeOps (+ 커스텀 map)** — 고수준 연산을 **TIR PrimFunc**(구체적 루프
  프로그램)로 낮춘다. `matmul`은 3중 루프가, `softmax`는 max/빼기/exp/합/나눔
  루프들이 된다. 여기가 표준이 마련한 첫 확장 지점이다:
  `npu_legalize.legalize_map()`이 **RMSNorm 하나만** 교체한다 — TVM 기본
  lowering은 float32 중간 버퍼를 만드는데 우리 벡터 유닛은 FP16 저장이라 담을
  수 없다. 교체본은 제곱 **전에** 1/√D를 곱해(합이 FP16 범위에 머물게; 기존
  검증 경로의 V3-020 트릭) 텐서 dtype 그대로 계산한다.
- **AnnotateTIROpPattern** — 융합 가능성 분류. 단, **융합(FuseOps/FuseTIR)은
  끈다**. 측정해 봤다: 켜면 링크·실행은 되지만 결과가 틀리고(cosine 0.17),
  맞더라도 이득이 없다 — 벡터 유닛은 한 번에 연산 하나라 융합 본문을 다시
  단계별로 풀면 같은 일이 되기 때문(word 수 29,970 vs 29,946).
- **DeadCodeElimination** — 정리.
- **ToNonDataflow / RemovePurityChecking / CallTIRRewrite** — 빌드 배관.
  이후 그래프는 "PrimFunc를 순서대로 호출하는 바인딩 목록"이 된다.
- **StaticPlanBlockMemory** — 버퍼 **생존구간 분석** 기반의 정적 메모리 계획.
  수명이 겹치지 않는 텐서들이 같은 storage를 재사용한다.

이 파이프라인은 `relax.register_pipeline("npu")`로 등록되어 있어
`relax.get_pipeline("npu")`로도 얻는다. 중요한 성질: **여기까지의 결과물은
llvm으로도 그대로 빌드된다.** NPU 없이 "모델 정의 + 그래프 pass"만 먼저
검증할 수 있는 이유다.

---

## 4. [3] 메모리 계획 — 주소가 상수가 되는 곳

기계에는 할당기가 없으므로 모든 텐서가 **하나의 평면 전역 이미지 안의 고정
주소**를 가져야 한다. `npu_memplan.py`가 `StaticPlanBlockMemory`의 결과
(storage 객체와 그 안의 offset)를 받아 평면 주소로 편다:

- 함수 파라미터(입력, lift된 가중치)들을 앞에서부터 배치.
- storage들을 그 뒤에 배치 — StaticPlanBlockMemory가 재사용을 이미 결정했으니
  여기서는 자리만 정하면 된다.
- 그래프 상수는 **내용으로 키를 만들어**(dtype+shape+bytes) 중복 배치를 막는다
  (TVM이 객체를 재포장해서 `id()`가 불안정하기 때문).
- 모든 할당을 32-bit 셀 경계로 올림 — DMA가 셀 단위라서.

결과는 `StaticPlan`: 이름→주소 표, 총 크기, 호스트가 채워야 할 상수 목록.
호스트는 이 표대로 `build_image()`에서 초기 이미지를 만든다(가중치·입력을
제 주소에 복사).

---

## 5. 양자화 pass (W8A16) — 유일한 우리 그래프 pass

`npu_quantize.QuantizeWeightsW8A16`는 **LegalizeOps보다 앞**, 즉 아직 고수준
`matmul`이 보일 때 동작한다. 파라미터에서 온 weight의 matmul을 찾아 세 단계로
재작성한다:

1. `w_scale` — 출력 채널별 scale: `max|W[:,j]| / 127` (FP32).
2. `w_quantize` — `round(W/scale)`를 INT8로 (round-half-to-even, 하드웨어와
   동일).
3. `qmatmul` — **dequant와 matmul을 한 PrimFunc에 담은** 커널:
   `W_fp16[k,j] = int8[k,j] × scale_fp16[j]` 를 만든 뒤 보통의 FP16 matmul.

배치가 요점이다. 1·2는 파라미터만의 함수이므로 **LiftTransformParams가 알아서
호스트로 hoist**한다 → 토큰당 추가 비용 0, 전역 메모리에는 INT8 weight와
FP16 scale만 남는다(→ **DMA 트래픽 절반**, 이게 양자화의 실질 이득이다).
3은 활성값 x에 의존하므로 기계에 남는데, dequant와 matmul이 **한 커널**이라
lift가 둘을 갈라 FP16 weight를 전역에 되돌려 놓는 사고가 원리적으로 없다.

scale이 출력 채널별인 이유: scale은 **합산 축(K)에서 상수**여야 한다. 그래야
내적 밖으로 빠져나올 수 있고, 기계가 부분합이 누산기에 들어갈 때 곱하는 것과
수학적으로 같아진다.

기계 매핑(§7에서 codegen이 하는 일): INT8 타일을 dtype 모르는 DMA로 SRAM에
올리고 → **VDEQUANT**로 행 단위 INT8→FP16 변환(스칼라 scale 레지스터에 fp32
상수 1.0을 가리켜 순수 변환으로 사용) → 벡터곱 한 번으로 채널별 FP16 scale
적용 → 이미 검증된 FP16 gemm 경로 그대로.

C-model 실측: 단일 양자화 matmul이 같은 산술의 numpy mirror와 **bit-exact**
(64³과 패딩 7×128×96 모두), tiny-Llama 전체 경로 float32 대비 cosine 0.999998.

---

## 6. [4] 스케줄 — 루프 프로그램을 기계 모양으로

여기서부터 기계가 보이기 시작한다. 링커가 커널(PrimFunc)마다 `tir.Schedule`을
적용한다 (`npu_intrin.py`).

### matmul의 스케줄 (`schedule_matmul_sram`)

1. **producer 처리** — matmul보다 앞서 뭔가를 계산하는 블록(pad_einsum의
   패딩 채움, 양자화의 w_dequant)이 있으면: 그 입력들을 `cache_read`로 SRAM에
   올리고, 출력 버퍼를 `set_scope`로 SRAM에 둔다. 연산 유닛이 전역을 못 읽기
   때문이고, 이렇게 해 두면 matmul은 그 피연산자를 다시 staging하지 않는다
   (SRAM→SRAM 복사는 PE 출력 레지스터를 뺏어 MAC 사슬을 끊는다).
2. **패딩** — M/N/K가 64의 배수가 아니면 `pad_einsum`으로 반복 공간을 64
   배수로 키운다. 표준 프리미티브가 경계 조건(`if_then_else`)이 든 producer/
   consumer 블록을 만들어 주고, 그 경계는 codegen이 행을 in-bounds/패딩 두
   조각으로 나눠 처리한다.
3. **타일링** — i/j/k 각각을 64로 `split`하고 `reorder`로
   `i_o, j_o, k_o, i_i, j_i, k_i` 순서를 만든다. 바깥 세 루프가 "타일 격자",
   안 세 루프가 "타일 하나"다.
4. **staging** — 두 피연산자를 `cache_read("global.sram")`하고 `compute_at`으로
   k_o 루프에 붙인다: 각 K타일이 소비 직전에 DMA로 올라온다. 결과는
   `cache_write`로 SRAM에 두고 j_o에서 되쓴다. 패딩된 결과의 되쓰기(unpad)는
   `reverse_compute_at`으로 타일 루프 **안**에 넣는다 — 안 그러면 패딩된 결과
   전체(예: lm_head는 [64, 128256] = 16 MiB)가 N축 내내 살아 있어 SRAM을
   초과한다.
5. **tensorize** — `decompose_reduction`으로 초기화(0 채움)와 누적을 분리한
   뒤, 타일 몸통을 `npu_gemm_acc_sram`/`npu_fill_zero_sram` 인트린식으로
   바꾼다. 인트린식은 "이 블록은 64×64 gemm이다"라는 **표식**(call_extern)일
   뿐이고, 실제 명령은 codegen이 만든다. SRAM 스코프 전용 인트린식이 따로
   있는 이유는 tensorize가 스코프까지 일치를 요구하기 때문이다.

### 그 밖의 커널 (`schedule_generic_sram`)

softmax·norm·elementwise 등은 내부 버퍼를 SRAM으로 `set_scope`하고 입출력을
`cache_read`/`cache_write`로 staging하는 정도로 충분하다. 루프 구조 자체는
codegen이 통째로 읽는다(§7).

### 표준 압축 접미 — 실모델이 들어가게 만든 것

`cache_read`는 타일 하나만 staging해도 **생산자의 전체 shape**로 버퍼를
만든다. [3072,3072] weight면 18 MiB — 8 MiB SRAM에 안 들어간다. 표준 해법이
`CompactBufferAllocation`(실제 접근 영역으로 버퍼 축소)인데, 선행 pass들이
"모든 블록의 init이 lower된 상태"를 요구한다. 그래서 스케줄 직후에 표준 4종을
돌린다: `LowerInitBlock` → `PlanAndUpdateBufferAllocationLocation` →
`ConvertBlocksToOpaque` → `CompactBufferAllocation`. 이 과정이 리덕션을
"가드된 초기 store + 누적 store" 형태로 바꾸고 iter var를 없애는데, 우리
codegen의 매처가 그 형태까지 읽도록 확장되어 있다(감축 축은 가드 조건에
나타나는 루프 변수로 판별). 이게 풀리면서 실모델 차원의 커널당 SRAM이
1~2.5 MiB 수준으로 내려왔다.

---

## 7. [5] codegen — TIR을 걸어가며 v09 word를 만든다

`tir_codegen_v09.py`의 **Walker**가 스케줄된 TIR을 위에서 아래로 해석하며
명령을 방출한다. 분기 없는 기계라서 Walker는 사실상 **TIR 인터프리터**다:
루프를 만나면 파이썬에서 그 횟수만큼 돌고, 몸통이 요구하는 명령을 그때그때
쏟아낸다. 모든 인덱스 식은 즉시 상수로 평가된다.

동작 별로:

**(a) 루프-패턴 인식.** 루프 중첩 + 블록 하나 꼴을 만나면 세 패턴으로
분류한다 — *이동*(복사·slice·transpose·broadcast: 행마다 벡터 load/save 또는
strided load), *리덕션*(행마다 reduce_sum/reduce_max 한 방), *pointwise*
(아래 (b)). 어느 것도 아니면 보통 순회로 내려간다. 패딩 predicate(블록에 붙은
`T.where`)는 반복 공간을 실제 영역으로 좁혀 해석한다.

**(b) 표현식 직렬화 (`_materialize`).** pointwise 블록의 우변이 복합식이면
트리를 후위 순회로 풀어 벡터 명령 여러 개로 만든다. 중간값은 커널별로 잡은
**scratch slot**(6개, 폭은 그 커널이 다루는 가장 긴 행)에 둔다. 자연 지원이
없는 함수는 조합으로 전개한다: `sigmoid = 1/(1+exp(−x))`,
`rsqrt = 1/sqrt(x)`, `tanh = 1 − 2/(exp(2x)+1)` — tanh를 이 꼴로 쓰는 이유는
지수가 FP16을 넘쳐도 몫이 0이 되어 결과가 정확히 ±1로 포화하기 때문이다
(차분 꼴은 ∞/∞가 된다). `Cast(int8→fp16)`은 VDEQUANT 명령이 된다.

**(c) gemm 방출.** tensorize 표식(`npu_gemm_acc`)을 만나면 행렬 서술자
(MAIN=부모 행 폭, PARTIAL=타일 위치)를 채우고 `m_mul(mac=…)`을 낸다. 같은
C타일로의 연속 호출은 **MAC 비트로 누산기에 체인**되고, 다른 데서 PE 출력을
써야 할 때만 `flush`(save)한다. 사슬이 끊겼다 이어지면 저장해 둔 부분합을
행렬 load로 되불러 잇는다.

**(d) DMA 방출.** cache 블록(전역↔SRAM 이동)은 행 단위로 걷되, 방출 전에
행들을 분석해 최소의 전송으로 묶는다: 양쪽 다 연속인 행들은 **1D 병합**,
전역 쪽이 일정 간격의 같은 길이 행들이면 **2D 전송 하나**(타일 staging의
전형 — 이걸 안 쓰던 시절 대비 staged 128³ matmul이 5,313→273 word),
전역만 연속이고 SRAM이 흩어졌으면 scratch에 모아 정렬된 덩어리로 보내는
**bounce**(홀수 길이 행이 셀 중간에서 시작하는 문제의 해법; 셀을 공유하는
남의 반쪽은 먼저 읽어 보존한다). 주소 단위는 전역=**byte**(int8/fp16/fp32가
공유하는 유일한 단위; DMA만 전역 주소를 소비하므로 안전), SRAM=**nibble**
(버퍼 dtype별 폭: fp16=4, int8=2, fp32=8).

**(e) 속도.** 인덱스 식을 행마다 TVM FFI로 재평가하면 링크가 수 시간
단위로 느려진다. 그래서 각 버퍼 접근을 **파이썬 함수로 한 번 컴파일**해 두고
(affine이면 origin+step 곱셈-덧셈으로, 아니면 행마다 평가) 루프에서는 순수
산술만 돈다. match_buffer 노드도 처음 본 것만 FFI로 읽고 캐시한다.

---

## 8. [6] 링크 — 하나의 프로그램으로

`npu_link.compile_program`이 계획된 Relax 함수의 바인딩(=커널 호출)을 순서대로
걸으며, 커널마다 [스케줄 → 주소 바인딩(파라미터 순서!) → SRAM 배치 → Walker]
를 돌려 **하나의 어셈블러에 이어붙인다**. 그 밖에:

- **상수 풀** — pointwise 커널이 쓰는 스칼라 리터럴들(1.0, 2.0, softmax의
  상수 등)을 모아 이미지에 두고 시작 시 SRAM으로 한 번 DMA. VDEQUANT용 fp32
  1.0도 여기 얹는다. 수집 대상은 **기계가 실제 호출하는 커널만** — 호스트로
  lift된 PrimFunc의 리터럴(리듀스 항등원 −3.4e38 같은 것)은 FP16에 안 들어간다.
- **SRAM 배치** — 커널 로컬 버퍼(cache 버퍼, 패딩 버퍼)를 nibble 단위로 bump
  할당. 용량 검사는 target profile(§9)의 SRAM 크기로.
- **peephole (A1)** — 완성된 word 스트림에서 "이미 그 값인 서술자 레지스터에
  다시 쓰는 명령"을 지운다. 직선 스트림이라 상태 추적이 **완전**해서 증명
  가능하게 안전하고, 실측 결과도 전부 bit-exact였다. 효과: 1층 검증 프로그램
  **29,946 → 19,321 word (−35.5%)**.
- **SNAPSHOT 계측** — `snapshot_at={커널 번호}`를 주면 해당 커널 뒤에 전체
  이미지를 떨구는 명령을 끼워 준다. 커널별 llvm 참조값과 대조해 "처음
  어긋나는 커널"을 집는 디버깅 도구다(§10의 버그들을 이걸로 잡았다).

마지막으로 HALT를 붙이면 프로그램이 완성된다.

## 9. target과 build 진입점 (S6)

`npu_target.py`가 이 흐름을 TVM의 관례대로 포장한다. `npu_target()`은 진짜
`tvm.target.Target`(`ext_dev -keys=npu -model=v09`)이고, 기계 상수(SRAM 8 MiB,
타일 64, lane 256, scratch slot 6)는 target의 model이 고르는 `NpuProfile`에
모여 있어 링커와 스케줄이 거기서 읽는다. `build(mod, target)` 한 번이
[그래프 pass → 링크]를 다 하고 `NpuExecutable`(word 스트림 + 메모리 계획,
`.run()`/`.save()`)을 돌려준다.

`relax.build`를 종점으로 삼지 **않는** 것은 의도다: 그 끝은 할당기와 함수
호출 기제를 가진 런타임이 로드하는 `runtime.Module`인데, 이 기계엔 둘 다
없다. 미완성이 아니라 구조가 그렇다.

## 10. 실행과 판정 — 무엇을 기준으로 "맞다"고 하나

실행은 `v09_runtime`이 C-model 시뮬레이터에 [프로그램, 초기 이미지]를 넘기고
최종 이미지에서 출력 주소를 읽는 것이다.

판정 기준에 위계가 있다 (이번에 확립된 중요한 사실):

1. **커널 수준** — 손작성 oracle(backend_v09) 또는 numpy mirror와 bit-exact.
2. **모델 수준** — HF의 첫 생성 token 일치 + logits cosine.
3. **중간 검증** — 같은 lowered IR의 llvm 빌드. 단, **llvm은 우리보다 느슨한
   기준이다**: TVM의 float16 matmul은 float16으로 누적하는데 우리 기계는 내부
   FP32 누적이라, 실모델 1층 실측에서 float32 기준 우리가 cosine 1.000000,
   llvm이 0.999965였고 argmax도 우리만 기준과 일치했다. **둘이 어긋나면
   float32 numpy로 판정한다.**

## 11. 대표 측정치

- **전체 Llama 28층**: A1 적용 **15,285,893 word**(적용 전 31,222,473 · −51.0%) ·
  1,206 kernel · image 6,130 MiB · 실행 376s. DMA 적재 1.62G cell(≈6.5 GB —
  S=7이라 weight를 사실상 한 번씩만 읽음). token 358 = HF golden,
  **HF logits cosine 0.9999927** — 손작성 경로의 golden(0.9999881)을 표준
  경로가 상회한다.
- **전체 Qwen3 36층**: A1 적용 **21,540,580 word**(적용 전 42,899,225 · −49.8%) ·
  1,622 kernel · 7,675 MiB · 실행 397s. token 358 = HF golden,
  **HF logits cosine 0.9999922**(NPU logits 기준 — 기존 golden 수준을 표준
  경로가 재현).
- **A1 peephole**: matmul −46.6%, softmax −32.7%, 1층 전체 −35.5%, 전부
  bit-exact.
- **2D DMA**: staged 128³ matmul 5,313 → 273 word (19.5×).
- **W8A16**: mirror와 bit-exact(64³·패딩), tiny 모델 cosine 0.999998.
- 손작성 oracle 대비 전체 word 수: A1 이전 약 1.8×였으나 **A1 적용 후 약
  0.89×** — 표준 경로가 손작성 경로보다 짧아졌다 (oracle 층당 615,462 word
  × 28층 ≈ 17.2M vs 우리 15.3M). backlog에 후속 최적화 항목들 정리됨.

## 12. 겪은 버그와 교훈 (같은 함정을 다시 밟지 않기 위해)

| 버그 | 원인 | 교훈 |
|---|---|---|
| Qwen3 전체 깊이에서 token 0 | 표현식 scratch slot이 고정 8192 원소, Qwen3 FFN은 9728 → 옆 slot 덮어씀. Llama는 FFN이 정확히 8192라 **우연히** 통과 | 타일 규모 테스트로는 원리적으로 못 잡는 버그가 있다 → 실차원 테스트 유지. slot은 커널별 최장 행으로 |
| 실모델 1층 argmax 불일치 | llvm이 fp16으로 누적 — **우리가 맞고 llvm이 틀림** | llvm을 최종 기준으로 쓰지 말 것 (§10) |
| qmatmul에서 A 타일을 scale 주소에서 읽음 | 커널 버퍼를 `buffer_map` **순회 순서**로 바인딩 (4-파라미터 커널에서 처음 어긋남) | 반드시 파라미터 순서로 바인딩 |
| 상수 풀 fp16 overflow | 호스트로 lift된 PrimFunc의 리터럴(−3.4e38)까지 수집 | 기계가 호출하는 커널만 수집 |
| Gemma 층 대조가 cosine 0.018 | 참조 파일의 `hidden_NN`은 층 NN에 **들어가는** 상태 (off-by-one) | 참조 데이터의 인덱스 관례를 코드에 주석으로 박음 |
| 융합 켜면 결과 오류 | 융합 본문 직렬화 미검증 + 이득 없음 | 융합은 정확성부터; 현재 기본 off |
| HF 추적 경로가 실차원에서 전부 0 | 추적 모듈이 `lv` 같은 **바인딩 이름을 재사용** — 계획/링커가 name_hint로 키를 잡아 마지막 것이 덮어씀 | 계획 전에 이름 유일화 (tiny 모델도 조용히 영향: 0.9995→1.0000) |
| [7,128256] 복사가 45,696개만 이동 | vlen 필드가 16-bit인데 인코더가 **조용히 마스킹** | movement 경로에서 긴 행 분할 + 인코더는 초과 시 즉시 에러 |
| lm_head 전체 DMA가 인코딩 실패 | rows/cols 16-bit 필드 초과 | emitter가 필드 한계로 자동 분할 |
| INT8 dtype이 다음 연산에 누출 | 서술자 dtype은 sticky | 변환 후 FP16 복원을 방출 |

디버깅의 표준 수순도 정립됐다: (1) 같은 lowered IR을 llvm으로 돌려 그래프/
codegen을 가른다 → (2) 깊이·차원 이분으로 최소 재현을 만든다 → (3) SNAPSHOT
+ 커널별 llvm 참조 대조로 "처음 어긋나는 커널"을 집는다 → (4) 그 커널을
단독 링크해 단위 재현을 만들고 테스트로 고정한다.

## 13. 남은 것

- **실모델 양자화 실행** — `w_dequant` 버퍼가 아직 weight 전체 크기로 SRAM에
  잡힘(타일 규모까지만 안전). 타일 루프로의 `compute_at`이 필요 — B1(weight
  재적재)과 같은 성질.
- **백로그** (`d_compiler/OPTIMIZATION_BACKLOG.md`) — 커널 경계 넘는 DMA 병합,
  weight 재적재(B1), 상수 index `take`→slice(lm_head 낭비 S배), transpose의
  matmul 흡수, layout 전파, MetaSchedule 튜닝, 비동기 DMA 등.
- ~~decode 경로~~ → **완료 (§18)**. 모델당 프로그램 2개(prefill_cache +
  고정 capacity decode)로 구현, tiny에서 float32 전체 재계산과 일치.
  전체 28층 golden 게이트는 §18.3 참조.

## 14. 전체 깊이 게이트 최종 결과 (2026-09-01 확정)

세 family 모두, 실제 체크포인트 전체 깊이가 C-model에서 HF golden token과
일치하고, **NPU logits 기준** HF logits cosine이 golden 수준에 도달했다.

| 모델 | word 수 (A1 후) | token | HF logits cosine | 실행 |
|---|---|---|---|---|
| Llama 3.2 3B (28층) | 15,285,893 (−51.0%) | **358 ✓** | **0.9999927** (기존 golden 0.9999881 상회) | 376s |
| Qwen3-4B (36층) | 21,540,580 (−49.8%) | **358 ✓** | **0.9999922** | 397s |
| Gemma 4 E2B (35층) | 13,665,154 | **108 ✓** | **0.9997845** | 247s |
| **Llama 3.2 3B — HF 추적 경로** (§2.5) | 16,021,618 | **358 ✓** | **0.9999927** (손작성과 동일) | 275s |

Gemma 이야기는 §12의 수순이 실제로 작동한 사례다. 첫 전체 실행이 token
236761로 실패했을 때: (1) 같은 lowered IR을 llvm으로 돌려 token 108 확인 →
frontend 무죄. (2) 깊이 이분 — 수정된 HEAD에서 2/5/16층이 HF hidden state와
cosine 0.999999/0.999999/0.999996 → full-attention과 공유 KV 포함 층 본체
무죄. (3) 원인은 실패 실행이 **buffer binding-order 버그 수정 전 코드**로
돌았던 것 (같은 word 수 13,665,154 — 프로그램 크기는 같고 주소만 틀렸었다).
HEAD 재실행으로 통과.

이로써 PLAN_TVM.md의 S0~S8 전 단계가 완료 상태다. 남은 것은 §13의 백로그
(성능 최적화)와 decode 경로의 표준화다.


---

# 부록: 메모리 계층·양자화 관점의 최적화 (2026-09-01 실측 포함)

## 15. 한정된 SRAM(8 MiB)을 어떻게 쓰고 있나

### 15.1 원칙: "상주"가 아니라 "타일 통과"

가중치 하나([3072,3072] = 18 MiB)가 SRAM보다 크므로, 이 컴파일러의 SRAM 전략은
**무엇을 상주시킬지 고르는 문제가 아니라, 모든 것을 타일 단위로 통과시키는
문제**다. 세 장치가 이를 만든다:

1. **`cache_read` + `compute_at`(k_o)** — matmul의 각 피연산자 타일이
   **소비 직전** K-타일 루프에서 DMA로 올라오고, 다음 타일이 같은 자리를
   재사용한다. 결과 타일은 `cache_write` 후 j_o에서 즉시 되쓴다.
2. **`CompactBufferAllocation`(표준 pass)** — `cache_read`가 잡는 버퍼를
   "생산자 전체 shape"에서 **실제 접근 영역**으로 줄인다. 이게 없으면
   실모델 커널당 18~48 MiB를 요구해 링크가 불가능했다(§6).
3. **커널 단위 bump 할당 + 함수 단위 storage 재사용** — 커널 로컬 버퍼는
   커널마다 0부터 bump 할당(커널 사이 재사용은 자동), 커널 결과 텐서들은
   `StaticPlanBlockMemory`가 생존구간으로 전역 storage를 재사용한다
   (소형 모델 실측 −6.7%; lift와 결합해 활성 풀 1,599.7 MiB → 0.7 MiB).

### 15.2 실측: 실모델 차원(3B, seq=7)에서 커널별 SRAM 사용량

기계가 실제 호출하는 32개 커널 중 상위:

| 커널 | SRAM | 내용 |
|---|---|---|
| `matmul5` (FFN down [7,8192]×[8192,3072]) | **1,160 KiB** | 최대 — A타일+B타일+C타일+패딩 버퍼 |
| `matmul2` (QK^T batched) | 868 KiB | |
| `matmul3` (probs×V) | 636 KiB | |
| q/k/v/o/gate/up proj | 450 KiB | |
| `silu` / `multiply` [7,8192] | 336 KiB | 행 단위 + scratch slot 6개 |
| `npu_rms_norm` | 174 KiB | |

**peak 1.13 MiB / 8 MiB = 14%.** 정확성 관점에서는 충분히 안전하고, 성능
관점에서는 **86%가 노는 공간**이다 — §17의 B1(weight 상주/재적재 제거)과
double-buffering이 정확히 이 공간을 쓰라고 있는 항목이다.

### 15.3 DMA를 아끼는 세 가지 emitter 최적화 (모두 실측)

| 기법 | 무엇 | 효과 |
|---|---|---|
| **2D 전송** | 넓은 텐서의 타일(행 n개, 일정 간격)을 명령 하나로 | staged 128³ matmul **5,313 → 273 word (19.5×)** |
| **byte 단위 인접 병합** | 전역·SRAM 양쪽이 이어지는 행들을 한 전송으로 | 홀수 행 테스트 5,057 → 3,474 word |
| **bounce(산집합 모음)** | 전역만 연속이고 SRAM이 흩어진/비정렬 행을 scratch에 모아 셀 정렬 전송 | 홀수 길이·비정렬 시작을 정확히 처리 (KV cache 행 10바이트 간격이 이 경로) |

여기에 **A1 peephole**(§8)이 서술자 중복 설정을 지워 전체 프로그램을
−35~51% 줄인다(전부 bit-exact 확인).

## 16. 양자화(W8A16): Q/DQ 그래프는 어떻게 처리되나

### 16.1 구조로 해결한 Q/DQ — "fusion pass"가 필요 없게 설계

일반적인 양자화 그래프는 weight마다 Q(quantize)/DQ(dequantize) 노드가 붙어
커널 수가 불어나고, 이를 fusion pass로 이웃 연산에 접는 것이 통례다.
이 컴파일러는 **배치(placement)로 같은 결과를 얻는다**:

1. **Q는 기계에 아예 없다.** `scale`·`quantize` 계산은 파라미터만의 함수라서
   `LiftTransformParams`가 **호스트 1회 실행**으로 hoist한다.
2. **DQ는 matmul과 한 PrimFunc다.** dequant(VDEQUANT + scale row 곱)와
   matmul을 한 커널로 내보내므로, lift가 둘을 갈라 FP16 weight를 전역에
   되돌릴 수 없고(→ DMA 절감 유지), 그래프에 독립 DQ 노드가 생기지 않는다.

**실측 census (tiny Llama 1층, W8A16):**

```
DEVICE(매 토큰): 45 calls — npu_qmatmul×5, matmul×5(좁은 k/v proj·어텐션·lm_head는 dense 유지),
                 나머지는 비양자화와 동일.  독립 Q/DQ 커널: 0개
HOST(1회):       npu_w_quantize×5, npu_w_scale(fp32+fp16)×10, transpose×8
```

즉 **일반 FuseOps 없이도 Q/DQ가 전부 접혀 있다.** 참고로 일반 융합은 측정
결과 이 기계에서 무익하다(§3): 벡터 유닛이 한 번에 연산 하나라 융합 본문을
다시 풀면 같은 일이 되고(word 29,970 vs 29,946), 현재 codegen으로는 결과도
틀린다(cosine 0.1749) — 그래서 기본 off다.

### 16.2 실측: 양자화의 실제 이득 (같은 모델, 같은 입력, C-model 카운터)

| | FP16 | W8A16 | Δ |
|---|---|---|---|
| DMA 적재 (cells) | 25,701 | **17,733** | **−31.0%** |
| weight bytes (lifted) | 78,208 | 46,336 | −40.8% |
| 프로그램 word | 19,321 | 24,525 | +26.9% (in-SRAM dequant 작업) |
| float32 대비 cosine | 1.000000 | 0.999998 | 양자화 오차뿐 |

weight가 정확히 절반이 안 되는 이유: norm/좁은 proj는 FP16 유지 +
채널당 FP16 scale이 추가되기 때문. word 증가는 **SRAM 내부 작업**(VDEQUANT
+ 곱)이라 외부 메모리 트래픽과 무관하다 — 이 기계의 병목 가정(DMA)에서
올바른 교환이다. 커널 수준으로는 numpy mirror와 **bit-exact**(64³·패딩
7×128×96)까지 확인되어 있다.

### 16.3 실측: 양자화가 지금 막히는 지점 (실모델 차원)

```
quantized [7,3072]×[3072,3072] 링크 시도:
  LinkError: kernel exceeds SRAM capacity
    w_dequant_global.sram  18.00 MiB (float16)   <- 원인
    lv1_global.sram         9.00 MiB (int8)
```

`w_dequant` 버퍼가 **weight 전체 크기**로 SRAM에 잡혀 있었다 — dequant가
타일 루프 안으로 `compute_at` 되지 않아서였다.

**해결(2026-09-01)**: `w_dequant`와 그 뒤의 int8/scale stage를
`compute_at(k_o)`로 K-타일 루프에 넣었다. dequant된 weight가 타일 하나씩만
존재하고 `CompactBufferAllocation`이 타일 크기로 줄인다. 실측:

```
quantized [7,3072]×[3072,3072]  → 링크 성공, C-model 실행
  fp32 dense 대비 cosine 0.999964   (순수 INT8 오차 수준)
  numpy mirror 대비 max|diff| 0.04
```

트레이드오프는 행 타일마다 재-dequant(B1의 재적재와 동형). 남은 것은 전체
모델 W8A16 게이트(oracle의 cosine 0.9994~0.9998 재현)뿐이다.

## 17. 남은 이슈와 최적화 항목 (우선순위순, 측정치 포함)

1. ~~양자화 dequant의 타일화~~ → **완료 (§16.3)**. 남은 것은 전체 모델
   W8A16 게이트 실행뿐.
2. **B1: weight 재적재** — 실측 행 128에서 1.89×, 192에서 2.68× 외부 트래픽.
   15.2의 **86% 노는 SRAM**이 해법 공간: 열 타일을 바깥으로 돌리거나
   weight 패널 상주. 양자화와 결합하면(§16.2의 −31%와 곱) 효과가 복리다.
3. **링크 시간** — 전체 모델 2~3시간. decode의 context별 재링크는 §18의
   고정-capacity 설계로 **해소**(모델당 링크 2회, step 재링크 0회). 남는
   것은 최초 링크 자체의 시간 — 커널 인스턴스 간 word가 거의 같으므로
   "커널 템플릿 + 주소 패치" 캐시로 크게 줄일 수 있다.
4. **A2 잔여: 커널 경계를 넘는 DMA 병합** — 층 출력 store 직후 다음 커널이
   같은 텐서를 다시 load하는 왕복이 남아 있다.
5. **C2: 상수 index take → slice** — prefill lm_head가 전체 seq에 대해
   계산됨(마지막 행만 필요) → **×seq 낭비**(seq=7이면 7배). decode 경로는
   1행이라 이미 낭비가 없다.
6. **비동기 DMA/double buffering** — ISA에 barrier가 없어 전송·연산 중첩
   불가(ISA 확장 보류 항목). 15.2의 SRAM 여유가 이중 버퍼 공간이다.
7. **정밀도 관찰** — 실데이터 RMSNorm에서 `(x/√D)²`가 fp16 subnormal에
   닿아 상대오차 0.55%(스케일 10배 시 0.10%) 실측. 현재 cosine 게이트에는
   무해하나, scale-aware 전개(입력 크기에 따라 사전 스케일 선택)가 후보.
8. **MetaSchedule 타일 튜닝 / layout 전파(C4) / transpose 흡수(C3)** —
   미측정 백로그 유지.


## 18. decode와 생성 런타임 — 모델당 프로그램 2개 (2026-09-01)

### 18.1 설계: 고정 capacity + 길이 mask

동적 shape이 없는 기계에서 "step마다 context가 자라는" decode를 프로그램
하나로 만드는 표준 해법을 썼다:

- **cache는 고정 capacity**로 잡는다: K는 `[L, kv, hd, C]`(decode의 score
  matmul이 바로 읽는 **전치** 배치), V는 `[L, kv, C, hd]`.
- decode는 **항상 `C+1` slot을 어텐션**한다: 이번 토큰의 K/V를 그래프 안에서
  concat으로 붙이고, **additive mask 입력 `[1,1,C+1]`** 이 현재 길이를
  지정한다 — 유효 slot과 자기 자신은 0, 빈 slot은 fp16 바닥값(softmax가 0으로).
- decode는 `(logits, 이번 step의 K행, V행)`을 반환하고, **호스트가 그 행을
  cache의 다음 slot에 써넣는다**. shape이 step마다 불변이므로 컴파일된
  decode 프로그램 하나가 전체 생성을 담당한다.

따라서 배포 단위는 정확히 **프로그램 2개**다: `prefill_cache`(prompt 길이,
logits + 초기 cache 반환)와 `decode`(capacity 고정). 런타임
(`npu_generate.py`)은 둘을 **한 번 링크하고 재사용**한다 — step별 재링크가
없다. 같은 루프를 llvm runner로도 돌릴 수 있어 step 단위 교차검증이 된다.

### 18.2 양자화와의 관계 (사용자 질문 정리)

- **weight 양자화(W8)는 오프라인이 맞다**: scale/quantize는 파라미터만의
  함수라 lift된 transform에서 **모델 로드 시 1회** 실행된다(토큰마다 아님).
  `run_nn_npu.py --params-cache`가 그 결과를 npz로 저장해 이후 실행에서
  재사용한다 — 진짜 "사전 양자화 보관".
- **vector unit의 VQUANT는 activation 양자화(A8)용**이다: 행마다 동적
  max가 필요해 기계 안에서만 가능하다.

### 18.2b W8A8 — activation 양자화 완료 (2026-09-01)

`QuantizeW8A8` pass가 파라미터-weight matmul을 W8A8 커널 하나로 재작성한다.
기계 매핑은 oracle의 검증된 시퀀스를 그대로 방출한다(전용 emitter,
`npu_w8a8.py`):

- 행마다: `|x| = max(x, −x)` → seeded reduce-max → ÷127을 **FP32로 저장**
  (scale 레지스터가 읽는 폭) → **VQUANT**로 INT8 행 생성 — 전부 SRAM 안.
- **INT8×INT8 gemm**: 서술자 dtype=INT8, `ascale`/`wscale` 레지스터가
  타일-로컬 scale 벡터를 가리키고, 기계가 부분합이 FP32 누산기에 들어갈 때
  `w_scale[col]·a_scale[row]`을 곱한다(scale이 K에서 상수라 수학적으로 동일).
- 이 커널만은 TIR을 걷지 않고 직접 방출한다 — TIR 본문은 llvm 교차검증용
  의미 정의로 남는다(커널 단위 tensorize에 해당).

**실측 (전부 C-model):**

| 검증 | 결과 |
|---|---|
| mirror(`w8a8_reference`, lift된 q_w/w_scale 입력) 대비 | **bit-exact**: 64³, [7,128]×[128,192], [7,3072]×[3072,3072], [7,8192]×[8192,3072], [64,3072]×[3072,8192] |
| fp32 dense 대비 cosine (실모델 폭) | 0.999917~0.999967 — 순수 W8A8 오차 |
| tiny Llama 전체 그래프 | float32 대비 **0.999998**, argmax 일치, vquant 카운터 20(=양자화 matmul 5×행 4) |
| 전체 28층 실 체크포인트 (`--quantize w8a8`) | **게이트 진행 중** — 완료 시 추기 |

도중에 잡은 함정 둘: (a) gemm이 SRC1/SRC2 dtype을 INT8로 남겨 다음 커널의
벡터 load가 거부 → emitter 끝에서 FP16 복원(sticky dtype 규칙 §12 재확인),
(b) 호스트 transform의 w_scale이 numpy와 1 ULP 달라 경계값 725개가 반대로
반올림 — mirror는 기계의 **실제 입력**(lift된 텐서)을 받아야 한다.

### 18.3 검증 상태

| 수준 | 결과 |
|---|---|
| tiny 3-token 생성 (mask 경로 포함) | **NPU == llvm == float32 전체 재계산** (`[0,1,27]`) — cache가 의미 변경 없는 최적화임을 매 step 증명 |
| 실 체크포인트 2층 절단 | NPU == llvm (`[0,0,0]`) |
| 전체 28층 golden `[358,1184,311]` | **게이트 진행 중** (프로그램 2개 링크, ~4h) — 완료 시 여기에 추기 |
