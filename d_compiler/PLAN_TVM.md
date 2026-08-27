# PLAN — TVM 표준 파이프라인 전환 및 최적화 pass 구축

> 작성 2026-08-27 · 대상 브랜치(제안) `tvm-pipeline` (현재 `cmodel-v09`에서 분기)
> 전제: v09 ISA·C-model·양자화는 확정 완료(PLAN_V09.md N0~N7). **타깃은 바뀌지 않는다.**
> 목표: Relax **아래 절반**(TIR 변환·융합·스케줄링·메모리 계획·빌드)을 TVM 표준으로
> 교체하고, 각 계층의 최적화 pass를 갖춘다.

## 0. 현황 진단 (2026-08-27 측정)

| 항목 | 현재 | 근거 |
|---|---|---|
| 모델 정의 | family별 손작성 Relax 빌더 3벌 | `relax.frontend.nn` 미사용 |
| 그래프 pass | 커스텀 1 + stock 4 | `passes.py` — **이 부분만 표준** |
| TIR 변환 | **없음** (Relax binding을 Python으로 순회해 명령어 생성) | `backend_v09.py` |
| 융합 | **없음** | `FuseOps` 미적용 |
| 스케줄링 | **없음** (타일링·루프 순서 하드코딩) | `emit_matmul` 3중 루프 |
| 메모리 계획 | 자체 bump/free-list | `memplan.py`, `StaticPlanBlockMemory` 미사용 |
| 빌드·런타임 | `relax.build`/VM 미사용, subprocess | `v09_runtime.py` |

**측정으로 확인된 손실 (모두 실측)**

1. **중복 서술자 62.8%** — llama prefill 층 615,462 word 중 386,449 word가
   "이미 그 값인 서술자를 다시 설정". proxy 층에 제거 실험 적용 시
   **−31.4%, 결과 bit-exact 확인**
2. **weight 재적재 최대 2.68×** — SRAM에 한 번에 안 들어가는 weight를
   행 타일마다 다시 읽는다 (행 7·64에서 1.00×, 128에서 1.89×, 192에서 2.68×).
   루프 순서가 코드에 박혀 있어 생기는 문제
3. **layout 연산이 binding의 24%** — 층당 569 binding 중 strided_slice 104 +
   concat 34. 어텐션 헤드를 그래프에서 잘라 붙이는 방식의 부산물
4. **명령어의 92.7%가 matmul 타일 체인** — 즉 전체 비용이 사실상 타일링 전략
   하나에 달려 있는데, 그것이 스케줄이 아니라 하드코딩

## 1. 설계 원칙

1. **기존 경로는 oracle로 동결.** `backend_0818`/`backend_v09`와 3-모델 golden은
   손대지 않는다. 새 경로는 전부 신규 파일
2. **매 단계 bit-exact gate.** v09 전환 때 검증된 방식을 그대로 사용 —
   단계마다 기존 결과와 비트 단위 비교. 통과 못 하면 다음 단계로 가지 않는다
3. **수치 순서를 먼저 고정한다.** 스케줄이 리덕션 순서를 바꾸면 결과가 달라지므로,
   *현재 순서를 그대로 재현하는 기준 스케줄*을 먼저 만들고 bit-exact를 확보한 뒤에야
   대안 스케줄을 탐색한다 (그 시점부터는 품질 기준으로 평가)
4. **ISA 제약을 인정한다.** v09에는 분기·반복 명령이 없다 → TIR을 거쳐도 **모든
   루프는 완전히 펼쳐진다**. 따라서 TIR 도입 자체는 프로그램 크기를 줄이지 않으며,
   크기는 §4의 ISA peephole과 (후속) 반복 명령이 담당한다

## 2. 목표 아키텍처

```
[모델 정의]  relax.frontend.nn.Module  --export_tvm()-->  IRModule + param spec
                     |
[그래프]     NPU 커스텀 pass + stock 최적화 pass            (§4.1)
                     |  LegalizeOps(커스텀 legalize map)
[커널]       TIR PrimFunc  --FuseOps/FuseTIR-->  융합 커널   (§4.2)
                     |  tir.Schedule: split(64) / cache_read("sram") / tensorize
[스케줄]     NPU 인트린식으로 텐서화된 TIR
                     |
[링크]       정적 주소 배정  ->  TIR->v09 ISA codegen         (§4.3)
                     |  ISA peephole
[산출물]     v09 명령어 스트림  (relax.build 산출물로 포장)
```

핵심 대응 관계 — **지금 손으로 하는 일이 TVM 어휘에 1:1로 있다**:

| 현재 (손작성) | TVM 표준 |
|---|---|
| `_Stager`의 GLOAD/GSTORE 삽입 | `cache_read` / `cache_write` (스토리지 스코프 `"sram"`) |
| `emit_matmul`의 64 타일 3중 루프 | `sch.split` / `sch.reorder` |
| 64×64 matrix unit 호출 | `sch.tensorize` + 인트린식 선언 |
| 256-lane vector 처리 | `sch.vectorize` (또는 인트린식) |
| `memplan.py` | `StaticPlanBlockMemory` + 주소 배정 pass |
| binding 순회 codegen | TIR 순회 codegen (`tir_backend` 방식, 이미 전례 있음) |

## 3. 단계 계획

각 단계 = 커밋 + gate. **P0는 즉시 이득이면서 새 codegen에서도 재사용**되므로 먼저 한다.

| 단계 | 내용 | Gate |
|---|---|---|
| **P0** | **ISA peephole** — 서술자 dead-store 제거 (word 스트림 대상이라 신·구 backend 공용) | 3-모델 golden bit-exact + word 감소 측정 |
| **P1** | NPU 인트린식 선언(`npu_gemm_64x64`, vector) + 단일 matmul을 TIR 스케줄→v09 ISA로 | 현재 backend 출력과 **bit-exact** |
| **P2** | vector/elementwise 커널 TIR화 + 커스텀 legalize map (rms_norm/softmax/gelu는 `legalize.py` 재사용) | op별 bit-exact |
| **P3** | `AnnotateTIROpPattern`+`FuseOps`+`FuseTIR` 활성화, 융합 커널 스케줄(중간값 SRAM 잔류) | bit-exact + word/DMA 비교 |
| **P4** | 층 전체를 새 파이프라인으로 + `StaticPlanBlockMemory` 기반 주소 배정 | proxy 층·실제 층 bit-exact |
| **P5** | **3-모델 golden을 새 파이프라인으로 재현** | token·hidden·KV·logits **bit-exact** (최종 관문) |
| **P6** | 프론트엔드를 `nn.Module`로 교체 (3 family, 고차원 텐서 유지) | 3-모델 golden 재현 (bit 또는 품질 기준) |
| **P7** | 양자화를 **Relax pass로 이관** (현재는 실행 시 driver가 처리) | W8A16/W8A8 기존 결과와 동일 |
| **P8** | 커스텀 target 등록 + `relax.build(pipeline=…)` 산출물화 | 산출물에서 실행 → 동일 결과 |
| **P9** | (선택) MetaSchedule 자동 튜닝 — 비용 모델 = word 수·DMA bytes·SRAM 점유 | 품질 기준 (bit 아님) |

**P5까지가 "동작 동일 증명", P6부터가 구조 개선.** P0~P5는 기존 프론트엔드를
유지하므로 언제든 되돌릴 수 있다.

## 4. 최적화 pass 목록

### 4.1 그래프 계층 (Relax)

| pass | 종류 | 노리는 것 |
|---|---|---|
| `LowerToNPUPrimitives` | 기존 커스텀 | rms_norm/softmax/gelu의 NPU 안전 분해 (V3-003/004/020 회피) |
| `CanonicalizeBindings`, `EliminateCommonSubexpr`, `FoldConstant`, `DeadCodeElimination` | stock (기존) | 정리·상수 folding (prefill RoPE 상수화) |
| `RewriteDataflowReshape` | stock (**신규**) | reshape를 view로 — §0의 layout 24% 문제 완화 |
| `AnnotateTIROpPattern` → `FuseOps` → `FuseTIR` | stock (**신규**) | 융합 커널 경계 결정. 중간값이 SRAM에 머무름 |
| `FoldTransposeIntoMatmul` | **신규 커스텀** | 어텐션의 Kᵀ를 matmul 서술자로 흡수 (transpose 명령 제거) |
| `QuantizeWeights` (W8A16/W8A8) | **신규 커스텀** | 양자화를 IR 변환으로 승격 (P7). 현재는 실행 시 driver 처리라 IR에 안 보임 |
| `StaticPlanBlockMemory` | stock (**신규**) | 버퍼 수명 기반 재사용 → G-buffer 사용량 감소 |

### 4.2 커널 계층 (TIR 스케줄)

| 규칙 | 노리는 것 |
|---|---|
| `TileForPE` — split/reorder로 64×64×64 | 현재 하드코딩된 타일링을 스케줄로 |
| `StageToSRAM` — `cache_read`/`cache_write("sram")` | DMA 삽입의 표준화. **루프 순서를 바꾸면 §0-2의 2.68× 재적재가 1.0×로** |
| `TensorizeMatmul` — 64×64 MAC 인트린식 | matrix unit 매핑 (0710에서 전례 있음) |
| `TensorizeVector` — 256-lane | vector unit 매핑 |
| `DoubleBuffer` (후속) | 비동기 DMA 도입 시 전송·연산 중첩 |

### 4.3 링크·codegen 계층 (ISA)

| pass | 노리는 것 | 측정된 효과 |
|---|---|---|
| **`EliminateRedundantDescriptors`** | 이미 그 값인 서술자 설정 제거 | **상한 62.8%**, proxy 층 실측 **−31.4% (bit-exact)** |
| `AssignStaticAddresses` | 버퍼 → 평면 주소 배정 (memplan 대체) | — |
| `MergeAdjacentDMA` | 인접·연속 GLOAD/GSTORE 병합 | 미측정 |
| `HoistLoopInvariantSetup` | 반복되는 타일 체인의 불변 서술자 선행 배치 | peephole과 부분 중복 |

> 안전성 근거: v09 프로그램은 **분기·반복이 없는 직선 코드**이므로
> 서술자 상태를 컴파일러가 정확히 추적할 수 있다 → dead-store 제거가
> 증명 가능하게 안전하다. (기존의 "두 half 항상 emit" 관례는 하드웨어 요구가
> 아니라 컴파일러 버그에 대한 방어였고, 상태 추적이 정확하면 불필요하다.)

## 5. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| 스케줄이 리덕션 순서를 바꿔 bit-exact 붕괴 | 기준 스케줄을 현재 순서에 맞춰 고정(원칙 3). 0710 TIR backend가 matmul byte-exact를 달성한 전례 있음 |
| `nn.Module` 전환 시 rank>2 텐서 등장 → 기존 backend 불가 | P6를 TIR 경로 완성(P5) 이후로 배치. 그 전까지 기존 프론트엔드 유지 |
| 컴파일 시간 증가 (타일 완전 전개 + TIR 왕복) | P1에서 조기 측정. 필요 시 인덱스 계산을 Python으로 우회(0710 walker가 쓰던 기법) |
| 작업량 과다 | P0·P5·P8이 각각 독립적 가치를 갖도록 분할. P5에서 중단해도 "표준 하위 경로 확보"라는 성과는 남음 |

## 6. 신규 파일 배치 (제안)

```
npu_compiler/
  isa_opt.py          # §4.3 peephole (P0, 신·구 공용)
  npu_intrin.py       # tensorize 인트린식 선언
  npu_legalize.py     # 커스텀 legalize map (Relax op -> TIR)
  npu_schedule.py     # 스케줄 규칙 (tile/cache_read/tensorize)
  tir_codegen_v09.py  # 스케줄된 TIR -> v09 ISA
  tvm_pipeline.py     # 전체 Sequential 파이프라인
  npu_target.py       # target 등록 + relax.build 연동 (P8)
  nn_models/          # nn.Module 모델 정의 (P6)
```

## 7. 즉시 착수 항목

**P0 (ISA peephole)** — 구현·검증이 하루 단위이고, 새 codegen에서도 그대로 쓰이며,
효과가 이미 측정되어 있다. 여기서 시작해 3-모델 golden으로 검증한 뒤 P1로 넘어간다.
