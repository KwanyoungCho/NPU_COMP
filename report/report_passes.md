# TVM Pass 파이프라인 전환 보고서

> 일자: 2026-08-21 ~ 2026-08-22 (main 직접 작업)
>
> 목표: 0818 compiler 경로가 **표준 TVM Relax pass 파이프라인**을 타도록 전환한다.
> custom하게 우회하는 부분을 줄이되, **비효율(instruction 증가) 0**과
> **세 모델 golden 결과 보존**을 필수 gate로 한다.

## 1. 배경 — 전환 전 상태의 감사 결과

- Relax IR/BlockBuilder/`relax.op.*`는 표준 TVM을 사용 중이었으나,
  **`relax.transform.*` pass는 0818 경로에서 하나도 실행되지 않았다.**
- lowering은 custom walker(`backend_0818`)가 IR binding을 직접 순회,
  RMSNorm/softmax/GELU 분해는 graph 구성 시점에 하드코딩(`legalize.py` builder).
- 표준 pass 적용 실험: 우리 graph에 CanonicalizeBindings/CSE/FoldConstant/DCE가
  그대로 적용되며 결과 byte-exact — "붙일 수 있는데 안 붙인 상태"임을 확인.

## 2. 적용 내용 (commit 단위 추적)

| Commit | 내용 |
|---|---|
| `11dbd4b` | **Step 1**: `passes.npu_pipeline()` — 표준 4-pass(Canonicalize/CSE/FoldConstant/DCE)를 driver의 0818 분기에 삽입. FoldConstant가 prefill RoPE cos/sin(정적 position ramp)을 초기 G-buffer 상수로 굽는다 |
| `6f7ced7` | **Step 2**: `LowerToNPUPrimitives` (PyExprMutator module pass) — family builder들이 `nn.rms_norm`/`nn.softmax`/`nn.gelu_tanh` 고수준 op을 emit하고, pass가 `legalize.py`의 단일 소스 분해(V3-020 순서, V3-003 회피, V3-004 회피)로 확장. all-ones weight(Gemma V-norm)는 multiply 생략. backend의 `nn.gelu_tanh`→native ACT_GELU 암묵 매핑 제거(잘못된 수식으로의 silent fallback 차단). streaming oracle(v3_executor)은 lowering만 적용해 역사적 semantics 유지 |
| `6c2149e` | (별건) Gemma `stats()`의 memmap ambiguous-truth 잠복 버그 수정 — PLE 테이블 완성 후 첫 table-mode 실행이 노출 |

파이프라인: `LowerToNPUPrimitives → CanonicalizeBindings → EliminateCommonSubexpr
→ FoldConstant → DeadCodeElimination` (0818/source-0818 전용, legacy 경로 불변).

## 3. Instruction 수 평가 (모델별 전수, `analyze_isa_stats.py`)

canonical set = 각 family의 prefill layer/decode(ctx8)/final norm, S=7.

| program | 전환 전 | Step 1 | Step 2 | 최종 delta |
|---|---:|---:|---:|---:|
| llama/prefill_layer | 497,563 | 497,438 | 497,438 | −0.03% |
| llama/decode_ctx8 | 517,781 | 517,773 | 517,773 | −0.00% |
| gemma/prefill_sliding_owner | 182,753 | 182,236 | 182,236 | −0.28% |
| gemma/prefill_full_shared | 342,665 | 342,148 | 342,148 | −0.15% |
| gemma/decode_s-owner_ctx8 | 184,701 | 184,637 | 184,637 | −0.03% |
| qwen/prefill_layer | 515,449 | 513,196 | 513,196 | −0.44% |
| qwen/decode_ctx8 | 526,872 | 526,560 | 526,560 | −0.06% |
| final_norm ×3 | 98 | 98 | 98 | ±0 |
| **TOTAL** | **2,768,078** | **2,764,282** | **2,764,282** | **−0.14%** |

- **증가한 프로그램 0개** — "비효율 금지" 충족
- Step 2는 Step 1과 word 단위 완전 동일(+0.00% 전 항목) — 고수준 추상화의 비용 0을
  증명 (lowering이 legalize 분해를 재사용하고 ones-weight 곱을 생략하므로)

## 4. 정확성 gate (전부 통과)

- `test_npu_passes.py`(신규): 파이프라인의 값 보존, 고수준↔legalize byte-exact
  (word 수 동일 포함), ones-weight 생략, 고수준 op 잔존 0 확인
- 전체 sweep **36 entrypoint PASS** (vendor parity, Gemma/Qwen vendor closure 포함)
- **세 모델 golden clean 재실행: token과 logits 지표가 전환 전 golden과
  byte 단위 완전 동일**
  - Llama `[358,2846,4560]`, Gemma `[108,236777,236789]`, Qwen `[358,1184,311]`
- 부수 확인: Gemma가 사전계산 PLE 테이블 모드(`ple_source: table`)로 처음 완주
  (source 실행 452→357초)

## 5. 남은 custom 영역 (의도적 유지)

- `backend_0818`(Relax→ver.08 word)과 `memplan`(정적 G-buffer 배치)은 여전히
  custom — ver.08의 stateful descriptor·compile-time 고정 주소 모델이 TIR 머신
  모델과 맞지 않아, BYOC/tensorize 정규화는 이득 대비 비용이 불확실하다고 판단.
  이 경계는 "TVM pass 세계(IR→IR)"와 "target codegen"의 표준적 분리이기도 하다.
- 이후 `LowerAttentionToStreaming`(long-context) 등 새 변환은 이제
  `passes.py`에 pass로 추가하는 것이 기본 경로다.
