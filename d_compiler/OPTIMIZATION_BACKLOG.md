# 최적화 백로그 — 나중에 pass로 한 번에 추가할 항목

> 2026-08-28 · 방침: 지금은 **naive하게 동작**시키고, 최적화는 여기에 모아
> 별도로 검토·구현한다. 각 항목은 **무엇을 / 왜 / 어떻게 / 측정치**로 적는다.
> 측정된 항목은 숫자를, 미측정 항목은 "미측정"을 명시한다.

## A. ISA·codegen 계층

### A1. 서술자 dead-store 제거 (peephole) — **측정 완료, 최우선**
- **무엇**: 이미 그 값인 서술자(주소·shape·vlen·scale)를 다시 설정하는 명령 제거
- **왜**: llama prefill 층 615,462 word 중 **386,449 word(62.8%)** 가 중복 설정.
  proxy 층에 적용 실험 결과 **−31.4%, 결과 bit-exact 확인**
- **어떻게**: 직선 코드이므로 서술자 상태를 컴파일러가 정확히 추적할 수 있다 →
  선형 dead-store 제거로 증명 가능하게 안전. word 스트림 대상이라 신·구 codegen 공용
- **주의**: "두 half 항상 emit" 관례는 하드웨어 요구가 아니라 방어 규칙이었음

### A2. 인접 DMA 병합 + 2D 전송 활용 — **완료 (2026-08-28)**, C6 참조
- **무엇**: 연속된 GLOAD/GSTORE 병합, 그리고 **넓은 텐서의 타일을 2D 전송
  한 번으로** (기존 codegen은 rows=1만 써서 타일 한 행마다 명령을 냈다)
- **측정**: staged matmul `[128,128]x[128,128]` **5,313 → 273 word (19.5배)**
- **남은 것**: 아직 **커널 하나 안**에서만 병합한다. 커널 경계를 넘는 병합
  (같은 텐서를 연속 커널이 staging하는 경우)은 미구현
- **2026-09-01 측정**: SRAM 배치가 커널 단위로 닫혀 있어(커널마다 0부터 bump)
  모든 커널 경계가 global 왕복이다. global에 **쓰이는 것은 activation뿐**
  이므로 양이 정확히 갈린다 — 28층 1회 실행에서 `gstore` **48.3 MB**,
  되읽기 약 54 MB로 **총 6.53 GB의 1.6%**. 지금은 작지만 **seq에 비례**하고
  weight 재적재(B1)를 고치면 긴 프롬프트에서 최대 항이 된다
- **어떻게**: 커널 단위 bump를 버리고 그래프 레벨에서 생존구간·사용횟수를
  보고 상주 집합을 고른다. global에 대해 `StaticPlanBlockMemory`가 하는 일을
  SRAM에 대해 하되 용량이 빡빡하므로 **선택**이 추가된다. B0·B1과 **같은
  8 MiB 예산을 두고 경쟁**하므로 하나의 배치 문제로 풀어야 한다
  (`report_0901.md` §15.5)

### A5. 링크 시간 (컴파일러 자신의 속도) — **1차 완료 (2026-09-01), 5.96배**
- **무엇**: 링크가 전체 모델 2~3시간이라 디버깅 사이클 전체를 막고 있었다
- **원인(프로파일 실측)**: 언롤 자체가 아니라 **루프 안에서 TVM 객체를 만지는
  비용**. 손작성 `backend_v09`가 같은 파이썬으로 같은 완전 언롤을 **613,558 word
  / 0.8초(816k word/s)** 에 하므로, 파이썬도 언롤도 원인이 아님이 증명된다.
  TIR 노드는 속성 읽기(`e.a`)도 해시도 전부 FFI 호출이다
- **고친 것** (Llama 1층 실차원 495,386 word 기준, 전 구간 bit-exact):

  | 조치 | 링크 시간 |
  |---|---|
  | (기준) | 246.1s · 2,013 word/s |
  | `_split_row`의 select 조건을 원소마다 TIR로 평가 → 한 번 컴파일 (`_scanner`) | 77.3s (3.18배) |
  | `_emit_dma` 행 루프의 불변식(`.buffer.data`, dtype 조회) 밖으로 | 57.8s (4.26배) |
  | `_flattener` 결과를 접근 노드 주소로 캐시 (타일마다 재컴파일하던 것) | **41.3s (5.96배)** |

  진단은 `ev()`를 감싸 느린 경로의 **노드 종류와 호출자**를 세는 식으로 했다 —
  결과가 "전부 `_split_row`, 전부 비교 연산"이라 곧바로 지목됐다
- **전체 모델에서는 병목이 옮겨갔다**: A6 적용 후 28층 링크 342.0s의 **64%가
  `_schedule`** 이었다. 28층 모델은 **커널 종류 32개 / 호출 1,206번**이라 같은
  PrimFunc을 37번씩 다시 스케줄하고 있었다. 스케줄은 (PrimFunc, 타일)의 순수
  함수이므로 GlobalVar로 캐시했다 — `_sram_layout`도 수집(본문 전체 순회)과
  배치(덧셈 몇 번)로 갈라 수집만 캐시한다. **219.0s → 6.3s**

  | 전체 Llama 28층 (15,285,893 word) | 링크 |
  |---|---|
  | (개선 전) | 8,147s |
  | A5 + A6 | 342.0s |
  | **+ 스케줄 캐시** | **127.0s** |

  **전체 모델 64.1배.** 현재 배분: codegen 102.1s(80%) · peephole 11.7s(9%) ·
  schedule 6.3s(5%)
- **남은 것 (파이썬 walker 한정)**: `self.env`가 **TIR Var 객체로 키잉**되어 조회마다
  FFI 해시가 난다(층당 약 380만 회). native 경로에는 해당되지 않으므로, 파이썬
  walker를 다시 주력으로 쓸 일이 생길 때만 의미가 있다

### A6. codegen을 C++ TVM 백엔드로 — **구현 완료 (2026-09-01), 8.9배**
- **무엇**: TIR→v09 방출을 파이썬 Walker가 아니라 **C++ 방문자**로 구현
- **왜**: TVM의 실제 codegen(LLVM/CUDA/C)은 전부 `src/target/` 밑의 C++이고
  **FFI가 0회**다. 프로덕션 codegen을 파이썬으로 TIR 노드 단위로 걷는 사례는 없다.
  A5에서 확인했듯 링크 시간의 대부분은 방출 자체가 아니라 FFI 왕복이며, 이건
  파이썬에 남아 있는 한 상수배로만 줄어든다
- **어디에**: `d_compiler/npu_codegen/` — `v09_isa.h`(인코더), `v09_asm.h`
  (어셈블러 + SRAM/DMA emitter), `v09_walker.cc`(walker + TVM 등록),
  `build.sh`. **out-of-tree**로 `libtvm.so`에 링크하고 TVM 전역 레지스트리에
  등록하므로 **TVM 자체를 다시 빌드하지 않는다**
- **안전 장치**: 파이썬 Walker가 **동작 정의(reference)** 로 남는다. C++이
  모르는 것을 만나면 예외를 던지고 그 커널만 파이썬으로 폴백하므로, **커버리지
  부족은 속도 문제일 뿐 정확성 문제가 아니다**
- **측정** (실차원 Llama 1층, 495,386 word):

  | | 링크 | 처리율 |
  |---|---|---|
  | 파이썬 walker (A5 적용 후) | 40.6s | 12,193 word/s |
  | **native walker** | **4.6s** | **108,749 word/s** |

  **8.9배**, 커널 종류 **32/32 전부 native**, 프로그램 **word 단위 완전 일치**.
  A5와 합치면 246.1s → 4.6s = **53.5배**
- **게이트** (`tests/test_native_codegen.py`): (1) 모든 인코더·서술자·DMA 형태를
  같은 스크립트로 돌려 word 대조, (2) llama prefill / qwen3 prefill /
  prefill_cache / decode 네 그래프에서 커널별 word 대조, (3) 프로그램 전체가
  파이썬 단독 결과와 동일. Gemma는 체크포인트가 필요해 여기 대신
  `test_nn_gemma`가 native 켜진 채로 전체 링크·실행하며 덮는다
- **기본값**: 라이브러리가 있으면 켜짐. `NPU_NATIVE=0`으로 끄고,
  `compile_program(native_mode="python"|"use"|"compare")`로 직접 지정
- **남은 것**: W8A8 커널은 아직 파이썬 직접 방출 경로(`npu_w8a8.py`)를 쓴다.
  파이썬 Walker도 유지 대상이다 — 정의이자 폴백이므로 지우지 않는다

### A3. 표현식 임시 버퍼 재사용
- **무엇**: `_materialize`가 표현식 트리를 풀 때 쓰는 scratch slot을 생존구간
  기반으로 재사용 (현재는 depth 기반 고정 슬롯)
- **왜**: SRAM 점유와 저장/적재 왕복을 줄인다
- **측정**: 미측정

### A4. 결과가 이미 SRAM에 있을 때 중복 복사 제거
- **무엇**: `_materialize`의 최상위 결과를 목적지에 직접 쓰도록 (현재 일부 경로에서
  scratch → 목적지 복사가 한 번 더 발생)
- **측정**: 미측정

## B. 스케줄 계층

### B0. 긴 프롬프트가 컴파일되지 않는다 — **최우선 (2026-09-01 측정)**
- **무엇**: 원소별·정규화 커널이 텐서 **전체**를 SRAM에 올린다. matmul은
  `compute_at`으로 타일 단위인데 `schedule_generic_sram`에는 그 단계가 없다
- **증상**: 실 Llama 한 층 기준 **seq 256에서 링크 실패** —
  `mask_global.sram [24,256,256]`이 9.00 MiB, seq 512는
  `rms_norm_global.sram [512,3072]`가 9.04 MiB (용량 8 MiB)
- **성능이 아니라 동작 한계다.** 지금 게이트가 전부 seq=7이라 안 드러났다
- **어떻게**: `schedule_generic_sram`에 행 축 `split` + `compute_at`을 넣어
  "SRAM에 들어가는 행 수"만 스테이징
- **측정**: `d_compiler/analyze_traffic.py` (링크만 하고 DMA 명령을 합산;
  28층에서 C-model 카운터와 1,619,976,421 cell로 일치 확인). 그래프는
  `report/figs/0901/`, 서술은 `report/report_0901.md` §15.4~15.5

### B1. weight·activation 재적재 제거 (staging 단위 분리) — **측정 완료**
- **2026-09-01 갱신**: 원인이 "루프 순서"만이 아니라 **staging 단위가 PE
  타일(64)에 묶여 있는 것**임이 실측으로 드러났다. 행렬 유닛이 64×64를 먹는
  것과 SRAM에 얼마를 둘지는 독립인데 지금 둘 다 64다
- **실측 (실 Llama 한 층)**:
  - weight: seq 64까지 1.00×, **seq 128에서 2.01×** — 예측식 `⌈S/64⌉` 그대로
  - activation 피연산자: seq 32에서 9.0 MB인데 **seq 64에서 217.6 MB (24배)**.
    seq가 64의 배수가 **아니면** `pad_einsum`의 패딩 버퍼가 SRAM에 상주해
    (`set_scope`) 1회로 끝나고, 배수면 일반 `cache_read` 경로로 **N/64회**
    재적재된다 — **패딩이 필요한 경우가 우연히 더 빠르다**
  - 토큰당 비용이 seq 32 이후 6.9 MB/token에서 **전혀 개선되지 않는다**
- **어떻게**: `schedule_matmul_sram(..., panel=(Tm,Tn,Tk))`로 열고, 용량 제약
  하에 트래픽 최소화로 패널을 고른다 (§15.5의 식). 목적함수가 바이트라
  **cycle 모델이 필요 없다**

### B1b. weight 재적재 (구 측정, 참고) — **측정 완료**
- **무엇**: SRAM에 한 번에 안 들어가는 weight를 행 타일마다 다시 읽는 문제
- **왜**: 행 7·64에서 1.00×인데 **128에서 1.89×, 192에서 2.68×** (실측).
  지금 검증이 S=7이라 안 드러났을 뿐, 실제 prefill 길이에서 외부 메모리 트래픽이 2~3배
- **어떻게**: `sch.reorder`로 열 타일을 바깥으로 보내거나 weight를 SRAM에 상주시키는
  스케줄. TIR 경로에서는 선언적 선택이 된다

### B2. 이중 버퍼링 (비동기 DMA)
- **무엇**: 전송과 연산 중첩. `sch.rolling_buffer` 또는 명시적 double buffer
- **전제**: v09 ISA에 비동기 DMA + barrier 추가 (ISA_V09.md 보류 목록)
- **측정**: 미측정

### B3. 타일 크기 자동 튜닝 (MetaSchedule)
- **무엇**: 타일 크기·staging 위치를 탐색
- **비용 모델**: word 수 · DMA bytes · SRAM 점유 (`analyze_isa_stats.py`가 이미 측정)
- **측정**: 미측정

## C. 그래프 계층

### C1. 융합(FuseOps/FuseTIR) 활성화 — **현재 꺼져 있음**
- **왜 껐나**: 융합 커널은 원소마다 복합 표현식을 계산하는데, vector 유닛은
  벡터 전체에 연산 하나를 적용한다. 그래서 융합 본문을 다시 벡터 단계로
  풀어야 하는데(=임시 버퍼 필요), 그 직렬화가 CPU/GPU에서 융합이 주는 이득
  (커널 내 메모리 왕복 제거)을 자동으로 주지 않는다
- **측정 완료 (2026-08-28)**: 1층(D=64) 기준으로 켜서 링크·실행까지 해봤다.
  `fuse=False` 34 PrimFunc / 29,946 word / **cosine 1.000000**,
  `fuse=True` 18 PrimFunc / 29,970 word / **cosine 0.174900**
- **결론 두 가지**: ① 융합 커널이 링크는 되지만 **결과가 틀린다** — codegen이
  융합 본문을 잘못 직렬화한다(디버깅 필요). ② 설령 고쳐도 **word 수가 줄지 않는다**
  (오히려 +24). 벡터 유닛은 한 번에 연산 하나라 융합 본문을 다시 단계로 풀면
  같은 일이 되기 때문. 커널 수만 줄고 이득은 없다
- **따라서 우선순위 낮음**. 켜려면 먼저 정확성부터 고쳐야 한다

### C2. 상수 인덱스 `take` → slice 재작성
- **무엇**: `take(x, 상수 인덱스)`를 `strided_slice`로 바꾸는 작은 Relax pass
- **왜**: 현재는 데이터 의존 gather를 정적 codegen이 평가할 수 없어,
  **lm_head를 전 위치에 적용**하고 호스트가 마지막 행을 고르는 naive 우회를 쓴다.
  실모델(S=7, vocab 128k)에서는 lm_head 작업이 **7배** 낭비
- **측정**: 낭비 배수 = 시퀀스 길이

### C3. transpose를 matmul 서술자로 흡수
- **무엇**: 어텐션의 Kᵀ를 별도 transpose 커널 대신 matmul의 주소 지정으로
- **왜**: transpose는 행마다 strided load가 필요해 비싸다 (단독 커널 909 word)
- **측정**: 미측정

### C4. 4D 어텐션의 layout 재배치
- **무엇**: `[seq, head, dim]` ↔ `[head, seq, dim]` 전치를 layout 선택으로 없애기
  (`sch.transform_layout` 또는 그래프 수준 layout 전파)
- **왜**: 손작성 경로에서 층당 binding의 24%가 strided_slice/concat이었던 것과
  같은 성질의 비용
- **측정**: 미측정

## C5. SRAM 캐시 버퍼 압축 — **해결됨 (2026-08-28)**
- **무엇이었나**: `cache_read`는 타일 하나만 staging해도 **생산자의 전체 shape**로
  버퍼를 할당한다. 실제 weight([3072,3072] = 18 MiB)에서는 8 MiB SRAM을 초과해
  링크가 실패했다 (`LinkError: kernel exceeds SRAM capacity`)
- **표준 해법 그대로 적용**: `npu_link._schedule`이 스케줄 뒤에
  `LowerInitBlock` → `PlanAndUpdateBufferAllocationLocation` →
  `ConvertBlocksToOpaque` → `CompactBufferAllocation`을 돌린다
- **막혔던 지점과 해법**: `LowerInitBlock`은 리덕션을 "가드된 초기 store +
  누적 store"로 바꾸고 `ConvertBlocksToOpaque`는 iter var를 없앤다. 매처가
  `block.init`과 `iter_type == 2`로 리덕션을 판별했기 때문에 그 형태를 못 읽었다
  → **선택지 ①**대로 `_match_nest`가 그 형태를 읽고 감축 축을 **가드 조건에
  나타나는 루프 변수**로 잡도록 확장했다 (`tir_codegen_v09.py`)
- **효과**: 실모델 차원에서 커널당 SRAM이 8 MiB 안에 들어온다
  (예: `matmul` 1.18 MiB, `matmul4` 2.43 MiB — 압축 전에는 18~48 MiB)
- **실측 (3B 차원 1층, hidden 3072 / ffn 8192 / head 24·8×128, seq 7)**:
  링크 1,032,615 word · 45 kernel · image 194.4 MiB (`run_real_layer_npu.py`).
  float32 numpy 기준으로 채점하면 **NPU가 cosine 1.000000 / max|diff| 0.00008**,
  llvm 빌드가 0.999965 / 0.00260. 마지막 행 argmax도 NPU만 기준과 일치(243).
  → 두 빌드가 갈리는 이유는 **TVM의 float16 matmul이 float16으로 누적**하는 반면
  우리 기계는 내부 누적이 FP32이기 때문이다. **llvm 빌드는 우리보다 느슨한
  기준**이므로, 둘이 어긋나면 float32 numpy로 판정해야 한다

## C6. DMA 셀 정렬 — 홀수 길이 행 (2026-08-28 해결)
- **무엇이었나**: 전송은 32-bit 셀 단위이므로 길이가 홀수인 행은 다음 행이
  셀 중간에서 시작한다. `seq=7`의 어텐션 점수([24,7,7])를 되쓸 때
  `DMA row must start on a 32-bit cell`로 실패했다
- **해법**: `_emit_dma`가 (a) 전역·SRAM 양쪽이 이어지는 행들을 **하나의 전송으로
  병합**하고, (b) 전역만 이어지고 SRAM이 흩어져 있으면 **scratch에 모아서**
  셀 정렬된 덩어리로 한 번에 전송한다(`_bounce_dma`). 벡터 복사는 원소 단위라
  정렬 제약이 없다
- **부수 효과**: 백로그 A2(인접 DMA 병합)의 대부분이 여기서 해결됐다.
  1층 검증 프로그램이 **44,826 → 39,438 word (−12.0%)**,
  `matmul[64,64]x[64,64]` 단독은 801 → 45 word

## D. naive로 둔 정확성 우회 (성능이 아니라 단순화)

| 항목 | 현재 naive 방식 | 나중에 |
|---|---|---|
| 64 배수 미만 matmul | `pad_einsum`으로 64 배수까지 패딩 | 소형 인트린식 또는 패딩 최소화 |
| 패딩 경계 조건(`if_then_else`) | codegen이 행을 in-bounds/패딩 두 조각으로 분할 | `decompose_padding`이 다중 writer를 만들지 않게 스케줄 조정 |
| MAC 사슬 중간의 벡터 연산 | 부분합을 저장했다가 재적재 (0710 walker 방식) | 스케줄에서 staging을 k 루프 밖으로 hoist |
| `sigmoid`/`rsqrt` | exp·나눗셈·sqrt 조합으로 전개 | 네이티브 activation(코드 2 SiLU) 활용 |
| 홀수 길이 DMA | 할당을 32-bit 칸 단위로 올림해 여유분에 기록 | 필요 시 부분 칸 처리 |
| 커널 로컬 임시 | 전부 SRAM에 bump 할당 | 생존구간 기반 재사용 |
| 표현식 임시 슬롯 폭 | 커널이 다루는 **가장 긴 행**으로 슬롯 6개를 잡는다 (`_scratch_row`) | 실제 `_materialize` 최대 길이만 계산해 더 좁게 |

## F. S8 양자화 codegen — **해결됨 (2026-08-31)**

세 가지 차단 항목이 모두 풀렸고 방식이 예상과 조금 달랐다:

1. **스케줄** — dequant를 matmul 뒤가 아니라 **앞**으로 옮겼다: weight를 SRAM에서
   FP16으로 풀고(VDEQUANT + scale row 곱) 검증된 FP16 gemm을 그대로 쓴다.
   `schedule_matmul_sram`의 producer 처리(pad 블록용)를 일반화해서 `w_dequant`도
   같은 취급을 받는다 → 스케줄 수정 자체가 거의 없었다
2. **혼재 폭 SRAM** — 전역 주소를 **byte 단위**로 통일했다(DMA만 전역 주소를
   소비하므로 안전). SRAM은 버퍼별 nibble 폭(fp16=4, int8=2, fp32=8).
   부산물: byte 단위 병합이 더 잘 되어 홀수 행 테스트가 5,057→3,474 word
3. **서술자** — wscale 불필요(위 방식이라서). VDEQUANT는 SRC1에 dtype=INT8을
   싣고 fp32 상수 1.0을 ascale로 가리켜 **순수 변환**으로 쓰고, per-channel
   scale은 뒤따르는 벡터곱이 적용한다. 변환 후 dtype을 FP16으로 복원(sticky)

**후속 해결(2026-09-01)**: `w_dequant`(+ int8/scale stage)를 `compute_at(k_o)`로
타일 루프에 넣어 실모델 폭이 링크된다. 실측: quantized [7,3072]×[3072,3072]가
C-model에서 **fp32 dense 대비 cosine 0.999964**(순수 INT8 오차 수준),
mirror 대비 max|diff| 0.04. 트레이드오프는 행 타일마다 재-dequant(B1과 동형).
남은 것: 전체 모델 W8A16 게이트(oracle cosine 0.9994~0.9998 재현).
**W8A8도 완료(2026-09-01)**: `QuantizeW8A8` + 전용 emitter(`npu_w8a8.py`) —
mirror와 bit-exact(실모델 폭 포함), tiny 모델 float32 대비 0.999998.
전체 28층 게이트 진행 중

## E. 참고: "표준 TVM으로 안 되는 것"의 정확한 구분

| 구분 | 내용 |
|---|---|
| 표준이 그대로 해결 | 메모리 계획(`StaticPlanBlockMemory`), 가중치 전치 제거(`LiftTransformParams`), 타일 패딩(`pad_einsum`), 융합, 스케줄 프리미티브 |
| 표준 결과가 우리 기계에 안 맞아 **표준 훅으로 교체** | RMSNorm의 FP32 중간값 → `LegalizeOps`의 per-op 커스텀 map. tensorize 스코프 → 전용 인트린식 |
| 표준에 원래 없는 것 (우리가 쓸 수밖에 없음) | v09 명령어 생성 자체, VM 런타임 미사용(우리 기계엔 할당기·호출 기제가 없음) |

→ **"표준 경로가 못 한다"는 항목은 사실상 없다.** 남는 것은 (a) 표준 훅으로
바꿔 끼우는 지점과 (b) 아직 안 쓴 배관뿐이다.
