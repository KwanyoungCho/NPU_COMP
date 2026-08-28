# PLAN — TVM 표준 파이프라인 전환 (개정 2: 표준 우선)

> 개정 2026-08-27 · 대상 브랜치(제안) `tvm-pipeline`
> 전제: v09 ISA·C-model·양자화 확정 완료(PLAN_V09.md N0~N7). **타깃은 바뀌지 않는다.**

## 0. 방침 (사용자 지시, 2026-08-27)

1. **표준 TVM 프론트엔드를 항상 기준으로 한다** — 모델 정의는
   `relax.frontend.nn.Module` + `export_tvm()`. 손으로 짠 Relax 그래프는
   기준이 아니라 *과거 자산*으로 강등한다
2. **표준 파이프라인을 기본적으로 전부 따라간다** — `LegalizeOps` →
   `AnnotateTIROpPattern` → `FuseOps` → `FuseTIR` → (스케줄) → 빌드까지,
   TVM이 정의한 단계를 우회하지 않는다. 우리 것은 *끼워 넣는다*, 대체하지 않는다
3. **NPU 특화 최적화는 나중에 별도 검토** — 서술자 peephole, transpose 흡수,
   DMA 병합 등은 §6에 보류 목록으로 두고 이번 전환에서 구현하지 않는다.
   목표는 "빠른 파이프라인"이 아니라 **"올바른 구조의 파이프라인"**이다
4. **표준 pass가 default, 기존 구현은 검증용** (2026-08-28 확정) — 자체 layout
   배정·메모리 계획(`memplan.py`, `v2_backend._assign_layouts/_plan_memory`)은
   표준 pass로 **교체**한다. 기존 backend는 oracle로만 남긴다

## 1. 검증 전략의 변경 (중요)

기존 계획은 매 단계 3-모델 golden과 **bit-exact**를 gate로 삼았다. 그러나 표준
프론트엔드를 쓰면 그래프 구조(4D 어텐션, reshape/permute 기반)가 달라져
연산 순서가 바뀌므로 **모델 수준 bit-exact는 정의상 불가능**하다. 따라서 gate를
두 층으로 나눈다.

| 층위 | Gate | 근거 |
|---|---|---|
| **커널(연산) 수준** | 기존 backend와 **bit-exact** | 같은 shape의 matmul/vector 커널은 순서를 맞출 수 있다 — 산술 구현의 정확성을 여기서 못 박는다 |
| **모델 수준** | **HF와 token 일치 + logits cosine** | 원래 golden을 만들 때 쓴 바로 그 기준. 기존 golden 값(예: Llama cos 0.9999881)이 목표치가 된다 |
| **중간 검증** | `relax.build(target="llvm")` + VM으로 CPU 실행 | NPU codegen 없이 **프론트엔드와 pass만** 먼저 검증할 수 있다 |

세 번째 항목이 이번 전환의 핵심 장치다: 표준 파이프라인은 CPU에서도 그대로
돌아가므로, NPU backend를 만들기 전에 "모델 정의 + 표준 pass"가 옳은지를
독립적으로 확인할 수 있다.

## 2. 목표 구조

```
[프론트엔드]  nn.Module 모델 정의  --export_tvm()-->  IRModule + param spec
                    |                                    ↑ HF 가중치는 param 이름으로 매핑
[그래프 pass] 표준 파이프라인 (get_pipeline "zero" 기반)
                    |  + 커스텀 legalize map (NPU 안전 분해가 필요한 3개 op)
[커널]        LegalizeOps -> TIR PrimFunc -> AnnotateTIROpPattern -> FuseOps -> FuseTIR
                    |
[스케줄]      tir.Schedule: split(64) / cache_read("sram") / tensorize(NPU 인트린식)
                    |
[빌드]        relax.build(mod, target="npu", pipeline=...)  -> 산출물
                    |                              \
[실행]        v09 C-model                           llvm 타깃으로도 빌드 가능(검증용)
```

**핵심 대응**: 지금 손으로 하던 일이 표준 어휘에 그대로 있다 —
`_Stager`(DMA 삽입) = `cache_read`/`cache_write("sram")`,
`emit_matmul` 3중 루프 = `split`/`reorder`, 64×64 유닛 = `tensorize`,
`memplan.py` = `StaticPlanBlockMemory` + 주소 배정.

## 3. 단계 계획

| 단계 | 내용 | Gate |
|---|---|---|
| **S0** | **표준 프론트엔드**: `nn.Module`로 Llama 3.2 3B 정의, `export_tvm()`, HF 가중치 이름 매핑 | ✅ (2026-08-27) 전체 28층 실제 체크포인트 → llvm 빌드 → **첫 token 358 일치**. 소형 config는 numpy 참조와 cosine 0.999999 |
| **S1** | **표준 파이프라인 골격**: 표준 단계를 명시적으로 구성(`tvm_pipeline.graph_pipeline`), 커스텀 legalize map 삽입 지점 확보 | ✅ (2026-08-27) stock 빌드와 **bit-identical**(융합 on/off 모두), 융합으로 40→21 PrimFunc, 전체 모델 4.9초·22 PrimFunc(층 간 커널 공유) |
| **S2** | **메모리 계획을 표준 pass로** — 표준 시퀀스 뒤 storage/offset을 평면 정적 주소로 배정 (`npu_memplan.py`) | ✅ (2026-08-28) 생존구간 충돌 0, 재사용으로 footprint 감소. **`LiftTransformParams` 발견 적용** — 런타임 가중치 전치 제거로 활성 pool 1,599.7→0.7 MiB (값 보존 확인) |
| **S3** | **인트린식·walker 재타깃** — compiler-v2의 인트린식·tensorize 스케줄·TIR walker를 **v09 ISA로 이식** | ✅ (2026-08-28, matmul) 64³·128³ 모두 backend_v09와 **bit-exact**. MAIN/PARTIAL 서술자 덕에 0710의 gather/scatter 불필요 → walker 대폭 단순화. **커널 13종 전부 bit-exact 완료** (matmul 64³·128³, binary 4, unary 3, sum/max, broadcast, transpose, slice) |
| **S4** | **SRAM staging** — `cache_read/cache_write("sram")`으로 DMA 표현 (0710엔 없던 신규 작업) | 커널 bit-exact 유지 + word·DMA 측정 |
| **S5** | **링크** — 커널 인스턴스를 하나의 명령 스트림으로 연결 | 층 전체 실행, HF 대비 수치 |
| **S6** | **target 등록 + `relax.build` 통합** | `relax.build(mod, target="npu")` 산출물로 실행 |
| **S7** | **3-모델 end-to-end** (Gemma·Qwen3 nn.Module 추가) | **HF token 일치 + logits cosine ≥ 기존 golden 수준** |
| **S8** | **양자화를 Relax pass로** (현재 driver 실행 시 처리) | W8A16/W8A8 기존 측정치 재현 |

**S0~S1은 NPU와 무관하게 CPU에서 검증**되므로 리스크가 낮고, 여기서 프론트엔드
기준선이 확정된다. S3부터 NPU backend가 붙는다.

## 4. 표준 pass 채택 목록

이번 전환에서 **기본으로 켜는** 것들 (TVM 정의 순서 그대로):

| pass | 역할 |
|---|---|
| `LegalizeOps` (커스텀 map 포함) | Relax op → TIR PrimFunc. NPU 안전 분해 3건은 여기에 등록 |
| `AnnotateTIROpPattern` | 융합 가능성 분류 |
| `FuseOps` → `FuseTIR` | 연산 융합 → 융합 커널을 하나의 PrimFunc로 |
| `FoldConstant` | prefill RoPE 상수 등 컴파일타임 계산 |
| `CanonicalizeBindings`, `EliminateCommonSubexpr`, `DeadCodeElimination` | 정리 |
| `RewriteDataflowReshape` | reshape를 view로 (4D 어텐션에서 필수) |
| `ToNonDataflow`, `RemovePurityChecking`, `CallTIRRewrite` | 빌드 배관 (표준) |
| `StaticPlanBlockMemory` | 버퍼 수명 기반 정적 메모리 계획 |

## 5. 기존 자산의 처리

| 자산 | 처리 |
|---|---|
| v09 ISA·C-model·시뮬레이터 | **그대로 유지** — 타깃은 안 바뀐다 |
| `legalize.py`의 NPU 안전 분해 | **유지** — 커스텀 legalize map으로 재등록 (V3-003/004/020 회피는 여전히 필요) |
| `backend_0818`/`backend_v09` | **oracle로 동결** — 커널 bit-exact 비교용 (default 자리에서 물러남) |
| **compiler-v2의 TIR backend** (`tir_backend.py`, `v2_backend.py`) | **재타깃 대상** — 인트린식 21종(gemm/elementwise/unary/silu/copy/transpose/slice/reduce/broadcast), tensorize 스케줄, TIR walker 골격·인덱스 평가를 v09로 이식. 0710용 emit 함수만 교체 |
| `memplan.py`, v2의 layout/메모리 계획 | **표준 pass로 교체** (방침 4). 검증 비교용으로만 유지 |
| 3-모델 golden | **참조값으로 유지** — 모델 수준은 HF 기준으로 비교 |
| 손작성 graph builder 3벌 | 과거 자산. S7 완료 시 사용 중단 (삭제는 별도 판단) |
| `tir_backend.py`(0710) | **참고 구현** — TIR 순회 codegen의 전례 |

## 6. 보류: NPU 특화 최적화 (사용자 검토 후 별도 진행)

측정은 되어 있으나 이번 전환 범위에서 제외한다.

| 항목 | 측정된 기회 |
|---|---|
| 서술자 dead-store 제거 (peephole) | llama 층 상한 **62.8%**, proxy 층 실측 −31.4% (bit-exact 확인) |
| 루프 순서 조정으로 weight 재적재 제거 | 행 192에서 **2.68× → 1.0×** |
| transpose를 matmul 서술자로 흡수 | 미측정 |
| 인접 DMA 병합 | 미측정 |
| MetaSchedule 자동 튜닝 (비용 모델 = word·DMA·SRAM) | 미측정 |

> 참고: v09 ISA에는 분기·반복 명령이 없어 **모든 루프가 완전히 펼쳐진다.**
> 따라서 TIR 도입만으로 프로그램 크기는 줄지 않으며, 크기 문제는 위 보류
> 항목(특히 peephole)과 후속 반복 명령이 담당한다.

## 7. 리스크

| 리스크 | 완화 |
|---|---|
| 표준 프론트엔드로 바꾸면 모델 수준 bit-exact를 잃는다 | 의도된 것(§1). 커널 수준 bit-exact + 모델 수준 HF 기준으로 이원화 |
| 4D 텐서 → 기존 backend 불가 | 새 codegen은 TIR을 소비하므로 차원 무관. 기존 backend는 커널 비교용으로만 사용 |
| 융합 커널의 스케줄 복잡도 | S4에서 단순 규칙부터. 튜닝은 §6 보류 |
| 컴파일 시간 | S3에서 조기 측정. 필요 시 0710 walker의 인덱스 평가 기법 재사용 |
