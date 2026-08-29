# NPU Compiler 작업 인계 문서 — 2026-08-28

표준 TVM 파이프라인 전환(`d_compiler/PLAN_TVM.md`)의 S5 완료와 S7 착수 시점 handoff다.
새 세션은 이 문서와 `d_compiler/PLAN_TVM.md`, `d_compiler/OPTIMIZATION_BACKLOG.md`를
읽고 시작하면 된다. branch는 `cmodel-v09`.

---

## 1. 오늘 무엇이 바뀌었나

시작 시점의 상태는 "표준 파이프라인이 **타일 규모(1층, D=64)** 까지만 링크된다"였다.
실모델 차원은 `LinkError: kernel exceeds SRAM capacity`로 막혀 있었다.
지금은 **실 체크포인트 전체 28층이 C-model에서 golden token과 일치한다.**

### 1.1 막고 있던 것 두 가지 (백로그 C5, C6)

**C5 — SRAM 캐시 버퍼가 생산자 전체 shape로 잡힌다.**
`cache_read`는 타일 하나만 staging해도 원본 텐서 크기로 버퍼를 만든다.
[3072,3072] weight면 18 MiB로 8 MiB SRAM을 넘는다. 표준 해법은
`CompactBufferAllocation`인데, 선행 pass들이 **모든 블록의 init이 lower된 상태**를
요구하고, `LowerInitBlock`은 리덕션을 "가드된 초기 store + 누적 store"로 바꾼다.
우리 loop-nest 매처는 `block.init`과 `iter_type == 2`로 리덕션을 판별했으므로
그 형태를 못 읽었다.

→ `Walker._match_nest`가 그 형태를 읽도록 확장했다(감축 축은 **가드 조건의 루프 변수**).
`npu_link._schedule`이 스케줄 뒤에 `LowerInitBlock` →
`PlanAndUpdateBufferAllocationLocation` → `ConvertBlocksToOpaque` →
`CompactBufferAllocation`을 돌린다. 커널당 SRAM이 8 MiB 안에 들어온다.

**C6 — 홀수 길이 행은 셀 경계에서 시작하지 않는다.**
전송은 32-bit 셀 단위다. `seq=7`의 어텐션 점수 [24,7,7]을 되쓸 때 실패했다.
→ `_emit_dma`가 양쪽 다 이어지는 행을 병합하고, 전역만 이어지고 SRAM이 흩어진
경우에는 scratch에 모아 셀 정렬된 덩어리로 보낸다(`_bounce_dma`). 시작이 셀 중간이면
그 셀의 **다른 반쪽을 먼저 읽어 보존**한다.

### 1.2 그 과정에서 발견해 고친 것들

| 문제 | 조치 | 효과 |
|---|---|---|
| DMA가 `rows=1`만 써서 타일 한 행마다 명령 하나 | 2D 전송 사용(`_regular_block` → `dma_2d`) | staged `[128,128]³` matmul **5,313 → 273 word** |
| 인접 행 전송이 안 합쳐짐 | 양쪽 연속 구간 병합 | 1층 프로그램 44,826 → 39,438 word |
| 행마다 TIR 인덱스를 재평가(FFI 왕복) | `_flattener`가 접근을 Python 함수로 **한 번** 컴파일 | 같은 링크 66.7s → 33.5s |
| `pad_einsum` 결과가 N축 전체에 걸쳐 살아있음 (lm_head면 [64,128256]=16 MiB) | copy-back을 `reverse_compute_at`로 N-타일 안에 배치 | 실 체크포인트 링크 가능, 1층 38,130 → 29,946 word |
| 그로 인해 생긴 **블록 predicate**를 매처가 무시 | `_narrow`가 predicate를 평가해 extent를 줄임(원점 box가 아니면 거부) | 범위 밖 쓰기 제거 |

### 1.2b Qwen3가 전체 깊이에서 터진 원인 (2026-08-29, 수정 완료)

전체 36층 Qwen3가 C-model에서 token 0을 냈다(golden 358). 같은 lowered IR을
llvm으로 돌리면 358이 나왔으므로 legalize가 아니라 **링커/codegen** 문제였다.

원인: 링커가 커널마다 **고정 8192 원소짜리 임시 슬롯 6개**를 SRAM에 연속 배치했다.
`_materialize`는 표현식을 **행 단위**로 펼치므로, 행 길이가 8192를 넘으면 자기 슬롯을
넘어 **다음 슬롯을 덮어썼다**.

- Llama는 `intermediate_size`가 정확히 **8192** → 딱 맞아서 28층이 멀쩡했다.
- Qwen3는 **9728** → silu의 `exp(-x)` 임시값이 8192번째 열부터 오염되고,
  x가 −10 근처인 자리에서 분모가 inf가 되어 결과가 non-finite가 됐다.
  거기서 FFN → residual → 전체 logit이 무너졌다.

수정: `_scratch_row`가 **그 커널이 실제로 다루는 가장 긴 행**을 구하고 슬롯을
커널별로 배치한다. 최소 재현(테스트로 고정): silu 행 길이 8192/8193/9728.
수정 전 8193은 8192열 뒤로 최대 오차 1780, 9728은 non-finite였고,
지금은 셋 다 0.0048(평범한 fp16 오차)이다.

**찾은 방법**(다음에도 쓸 것): `V09Asm.snapshot()` +
`compile_program(snapshot_at=...)`로 커널마다 메모리 이미지를 떠서,
binding별 llvm 참조값과 대조해 **처음 어긋나는 커널**을 집었다.

### 1.3 정확도 판정 기준이 바뀌었다 — 중요

실모델 차원 1층에서 C-model과 llvm 빌드가 마지막 행 argmax에서 갈렸다.
**float32 numpy로 채점하니 우리가 맞고 llvm이 틀렸다.**

| 대상 | cosine | max\|diff\| | argmax(마지막 행) |
|---|---|---|---|
| NPU (C-model) | **1.000000** | 0.00008 | 243 (기준과 일치) |
| llvm 빌드 | 0.999965 | 0.00260 | 159 |

이유는 **TVM의 float16 matmul이 float16으로 누적**하는 반면 우리 기계는 내부 누적이
FP32이기 때문이다. 즉 **llvm 빌드는 우리보다 느슨한 기준**이다. 둘이 어긋나면
float32 numpy로 판정한다. 재현: `d_compiler/run_real_layer_npu.py`.

---

## 2. 지금 어디까지 검증됐나

| 범위 | 결과 |
|---|---|
| 타일 규모 op 16종 (S5 스위트) | 전부 통과, 1층 llvm 대비 cosine 0.999999 |
| **실모델 차원 1층** (hidden 3072 / ffn 8192 / head 24·8×128, seq 7) | 링크 1,005,891 word · 45 kernel · image 194.4 MiB, **float32 기준 cosine 1.000000** |
| **실 체크포인트 1층** (Llama 3.2 3B, vocab 128256) | 링크 4,263,891 word · image 946 MiB, C-model 실행 64s, llvm과 같은 token |
| Qwen3 프론트엔드 (신규) | numpy 기준 cosine 0.999999(llvm), C-model 1층 **cosine 1.000000** |
| **Qwen3-4B 전체 36층 (llvm)** | **token 358 — golden 일치**, HF logits 대비 cosine 0.9852 |
| **Qwen3-4B 전체 36층 (C-model)** | **token 358 — golden 일치**. 42,899,225 word · 1,622 kernel · image 7,675 MiB, 링크 8,589초, 실행 505초 (§1.2b 수정 후) |
| Qwen3-4B 1층 (C-model, 실 체크포인트) | 링크 4,527,605 word · 47 kernel · image 937 MiB, 실행 61초 |
| 전체 28층 Llama (llvm) | **token 358 — golden 일치** |
| **전체 28층 Llama (C-model)** | **token 358 — golden 일치**. 31,222,473 word · 1,206 kernel · image 6,130 MiB, 링크 10,268초(2.85h), 실행 462초 |

전체 모델 실행 카운터(참고): 실행 word 28,793,514 / gload 786,312 · gstore 23,341 /
DMA 적재 1,619,976,420 cell(≈6.5 GB — 가중치를 사실상 한 번씩만 읽는다. `seq=7`이라
행 타일이 하나여서 재적재가 없다) / matrix 787,008 · vector 1,826,591.
손작성 oracle이 층당 615,462 word였으므로 28층 환산 대비 **약 1.8배**인데,
백로그 A1(서술자 dead-store 62.8%)이 아직 안 들어간 상태다.

---

## 3. 진행 중 / 다음 할 일

1. **Qwen3 NPU 결과의 logits cosine 측정** — token은 맞췄고(358), 남은 건 HF logits와의
   cosine을 **NPU 결과로** 재는 것이다. llvm의 0.9852는 §1.3대로 llvm이 fp16으로
   누적해서 나온 값이라 게이트로 쓸 수 없다.
2. **Gemma 프론트엔드** — S7의 남은 한 family. Llama/Qwen3와 달리 델타가 크다
   (PLE, 공유 KV, sliding window, 추가 norm 5종). `npu_compiler/gemma4_graph.py`가
   검증된 참조 구현이므로 그대로 옮기면 된다.
3. **S6** (custom target + `relax.build` 통합), **S8** (양자화를 Relax pass로).
4. 백로그 최우선은 여전히 **A1 서술자 dead-store 제거**(측정 62.8%).

---

## 4. 하지 말아야 할 것

- **융합을 켜지 말 것.** 오늘 측정했다: 링크·실행은 되지만 **결과가 틀리고**
  (cosine 0.1749 vs 1.000000) word 수도 안 줄어든다(29,970 vs 29,946).
  벡터 유닛은 한 번에 연산 하나라 융합 본문을 다시 풀면 같은 일이 된다.
- **vendor C-model 산술·quirk를 고치지 말 것** (기존 원칙 그대로).
- llvm 빌드를 정확도의 최종 기준으로 삼지 말 것 (§1.3).
