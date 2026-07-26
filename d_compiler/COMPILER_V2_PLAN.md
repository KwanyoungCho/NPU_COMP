# NPU 컴파일러 v2 설계 — TVM-native (Path A)

> 목표: 현재의 수제(hand-written) 백엔드를 **TVM 정석 파이프라인**으로 재구성한다.
> compute→TIR, 스케줄→MetaSchedule(tensorize + 제약 space), 메모리→TVM memory 패스,
> codegen→TIR→NPU-ISA 단일 lowering. 최종적으로 **C-model cycle 기반 auto-scheduling**을 얹는다.
>
> 선행 문서: `REFACTOR_PLAN.md`(v1, 완료), `report/report_0719.md`(v1 결과).
> 이 문서는 v2(대규모 재구성)의 **설계·패스 구조·단계별 계획**을 담는다. **아직 구현 전 — 설계 리뷰용.**

---

## 0. 목적 · 범위 · 원칙

- **최종 목표(Path A)**: TVM MetaSchedule에 올라탄 정석 가속기 컴파일러.
  - 가속기 정석 = **free-form Ansor가 아니라** `tensorize`(HW 연산=TIR intrinsic) + **제약된 schedule space** + **커스텀 TIR→ISA codegen** + cost model. (TVM VTA가 그 표준 예.)
- **auto-scheduling 범위**: **타깃형 cost 기반 선택**(타일/fusion/layout 후보 소수를 cycle로 비교). 넓은 search 아님.
- **cost model 시점**: 한참 뒤/미정 → search(Phase 4)는 **유보**, 그 밑의 기반부터.
- **원칙**:
  1. **오라클 주도**: 현재 v1 경로를 numerical 오라클로 삼아 op 단위로 대조 이행. 빅뱅 금지.
  2. **gate 유지**: 매 단계 회귀 게이트 GREEN. 단 auto-sched 도입 시 **byte-exact→tolerance** 전환(합의 필요).
  3. **요구 주도**: "hacky해서"가 아니라 "auto-sched가 요구하는 seam"만 판다.
  4. **하이브리드 현실**: 100% MetaSchedule은 목표 아님. matmul은 tensorize, 메모리이동은 스케줄 primitive, 특이 op은 커스텀 lowering 병존.

---

## 1. 전체 파이프라인 (패스 순서 한눈에)

```
                                   IR 레벨      현재 v1                     v2 (this doc)
─────────────────────────────────────────────────────────────────────────────────────────
[F1] Import (torch/HF → Relax)     Relax        import_legalize             relax.frontend  (재사용/정리)
[F2] Op 정규화 (통합 legalize)      Relax        legalize/import_legalize     Relax transform (통합)
[F3] Fusion                        Relax        codegen이 O-proj 암묵 융합    relax FuseOps/FuseTIR (명시 패스)
[F4] Layout planning (tile-blocked) Relax       memplan.assign_layouts       Relax layout 패스 (packed layout)
[F5] LegalizeOps (Relax→TIR)       Relax→TIR    schedule_matmul(부분)         relax.LegalizeOps → call_tir
─────────────────────────────────────────────────────────────────────────────────────────
[S1] Tensorize + 스케줄             TIR          schedule_matmul(수제)         tir.Schedule + tensorize(NPU intrin)
[S2] (cost 기반 선택)               TIR          없음                         MetaSchedule + C-model cost  ← 유보
─────────────────────────────────────────────────────────────────────────────────────────
[M1] Memory planning               Relax+TIR    memplan.plan (수제 liveness)  StaticPlanBlockMemory/StorageRewrite/USMP
─────────────────────────────────────────────────────────────────────────────────────────
[C1] TIR→NPU ISA codegen           TIR→ISA      codegen.py + _Walker/emit_gemm 일반화된 단일 codegen (target 등록)
[C2] 직렬화 → 런타임 → mysim         ISA          runtime/driver               재사용
```

핵심: **F(그래프)·M(메모리)·C(codegen)는 대부분 TVM 표준 패스로 흡수**, **S(스케줄)가 auto-sched의 무대**.

---

## 2. Relax 단계 (그래프 레벨) — "무엇을 계산하나 + 어떤 layout으로"

### F1. Import — torch/HF → Relax IRModule
- `relax.frontend.torch.from_exported_program`(또는 fx)로 실제 HF Llama를 Relax로.
- **산출**: high-level Relax op(matmul/add/silu/rms_norm/softmax/rope…)로 된 IRModule.
- **현재 대비**: `import_legalize`의 수제 매핑 → TVM frontend + 얇은 어댑터. (v1의 A3가 이걸 준비해 둠.)

### F2. Op 정규화 (통합 legalize) — Relax transform
- 모델별 변형(SiLU 분해, softmax 안정화, RoPE rotate-half)을 **하나의 canonical Relax op 집합**으로 정규화.
- **왜 Relax에서**: 이후 fusion·layout·TIR화가 전부 이 정규형 위에서 돎.
- **현재 대비**: v1의 `legalize`/`import_legalize` 두 경로 통합(A3) → Relax `transform.Pass` 하나로.

### F3. Fusion — `FuseOps` / `FuseTIR` (Relax)
- elementwise 체인(silu·mul), matmul+bias, **O-proj per-head 합** 등을 **명시적 fusion 패스**로.
- **효과(중요)**: v1의 "codegen이 O-proj를 암묵 융합 → A1 liveness가 그걸 특별 처리"라는 hack이 **사라짐**. 파이프라인이 `FuseOps → 그 다음 memory planning`이라, 메모리 패스가 **이미 융합된 그래프**를 보므로 fusion-인지가 공짜.

### F4. Layout planning (tile-blocked = **packed layout**) — Relax 커스텀 패스 ★
- tile-blocked `[R,N]→[⌈R/64⌉,⌈N/64⌉,64,64]`는 TVM의 **NCHWc식 packed layout**과 정확히 같은 개념.
- **왜 packing을 하나**: 내부 64×64가 연속이어야 **tensorize(S1)** 가 먹기 때문. 즉 **layout과 tensorize는 한 쌍** — packing은 intrinsic을 먹이려고 존재.
- 패스가 하는 일: 텐서마다 layout∈{row, tile} 결정(v1 `assign_layouts` fixpoint 로직 계승) → **경계에 `relax.op.layout_transform` 삽입** → op shape을 packed로 rewrite.
- **현재 대비**: `memplan.assign_layouts`의 **아이디어는 계승**하되 Relax `layout_transform` 기반 표준 패스로. (완전 자동 ConvertLayout은 Relax에 성숙하지 않아 **커스텀 패스**로 두는 게 현실적.)

### F5. LegalizeOps (Relax → TIR) — `relax.transform.LegalizeOps`
- 각 Relax op → `R.call_tir(prim_func, ...)`. 즉 **연산마다 TIR PrimFunc** 생성(packed 버퍼 위에서 동작).
- matmul/rms_norm/softmax 등의 TIR 정의를 제공(일부는 TVM 기본, NPU 특화는 커스텀 TIR).
- **산출**: TIR PrimFunc 다수 + 이들을 부르는 Relax 그래프. 이제 스케줄 대상.

---

## 3. TIR 단계 (PrimFunc 레벨) — "어떻게 계산하나(스케줄/tensorize)"

### S1. NPU tensor intrinsics + 스케줄
**(a) NPU intrinsic 정의**(`tvm.tir.TensorIntrin.register`, (desc, impl) 쌍):

| intrinsic | 의미(desc) | impl(NPU ISA) | tensorize? |
|---|---|---|---|
| `npu.mma_64x64` | `C[64,64] += A[64,64]@B[64,64]` | `m_mul(mode=VECTOR, mac)` | **예 (핵심)** |
| `npu.reduce_sum_64` | 행 reduce | native reduce(0x14) | 예 |
| `npu.vmax_64` / `npu.vadd` 등 | 원소별 | v_max/v_add | intrin lowering(굳이 tensorize 아님) |
| `npu.load_T_64` | 64×64 전치 load | strided column-major load | 메모리op — 스케줄/패턴으로 |

- **matmul**: block을 64×64로 `split`+`reorder` → **`sch.tensorize(inner, "npu.mma_64x64")`**. 이게 v1 `schedule_matmul`+`emit_gemm`을 대체.
- **elementwise/reduce**: TIR 그대로 두고 `LowerIntrin`에서 NPU 벡터 op(call_extern)로 내림.
- **transpose/layout**: F4의 packing + `cache_read` 접근패턴 → codegen이 strided-load로 인식.
- **메모리 scope**: scratch 타일에 storage scope 부여(`sch.cache_read(..., "npu.scratch")`), G-buffer는 M1에서 pool로.

**(b) 스케줄 파라미터화**: 타일 크기·루프 순서·compute_at(fusion)·scope를 **하드코딩이 아닌 스케줄 선택**으로. (지금은 전부 64 고정 → 파라미터.)

### S2. MetaSchedule (cost 기반 선택) — **유보(Phase 4)**
- MetaSchedule의 **schedule rule**로 space 정의: tensorize rule(`MultiLevelTilingWithIntrin`류) + 제약(tile=64 배수, scope 규칙).
- **cost model = C-model cycle**: custom `PyCostModel`(cycle 예측) 또는 C-model을 **runner**로(실측 cycle). 타깃형이라 후보 소수만 평가.
- cost model 도착 전엔 **고정/휴리스틱 스케줄**로 파이프라인만 통과.

---

## 4. Memory planning 단계 — **TVM 패스로 대체 (A1 은퇴)**

- **전제**: G-buffer를 수동 offset이 아니라 **TVM storage pool / scope**로 선언(VTA on-chip 메모리 방식). codegen이 TVM이 배치한 버퍼를 읽음.
- **패스**:
  - `relax.transform.StaticPlanBlockMemory` — 그래프 레벨 중간 텐서 storage 공유(= v1 A1 활성화 재사용).
  - `tir.transform.StorageRewrite` — PrimFunc 내부 scratch 타일 재사용.
  - **USMP**(pool 기반) — whole-program G-buffer 정적 할당.
- **효과**: v1의 수제 liveness/free-list(A1) + fusion-인지 특별처리가 **표준 패스로 흡수**. 손으로 짤 필요 없음.
- **주의**: flat G-buffer라 표준 플래너가 잘 맞을 것(특이 뱅킹 없음). fragmentation도 TVM 플래너가 v1보다 나을 여지.

---

## 5. Codegen 단계 (TIR → NPU ISA)

- **정체**: scheduled+tensorized TIR + 배치된 버퍼(scope/offset) → `isa.Asm.words`.
- **출발점**: v1 `_Walker`가 **이미 scheduled TIR을 걷어 ISA를 emit**함 → **이걸 일반화**(임의 스케줄·intrinsic 소화). `emit_gemm`은 canonical fast-path로만 잔존(속도).
- **매핑**: `tensorize`된 `npu.mma_64x64` block → `m_mul(mac)` 시퀀스. 벡터 intrin → v_add 등. strided 접근 → strided load/save.
- **TVM 등록**: 커스텀 target(`target.build.npu`)로 codegen 등록 → `tvm.build`/relax pipeline이 자동 호출.
- **현재 대비**: `codegen.py`의 op별 수제 emitter(row/tile 분기, 오버플로우 가드 등 ~1000줄)가 **단일 lowering으로 대체**.

---

## 6. 현재 코드 → v2 마이그레이션 맵

| v1 파일/함수 | v2 운명 | 비고 |
|---|---|---|
| `import_legalize` / `legalize` | **재사용→정리** (Relax transform 통합) | A3가 준비 |
| `assign_layouts` (fixpoint) | **아이디어 계승** → Relax layout 패스(packed) | layout_transform 기반 |
| `memplan.plan` liveness/free-list | **은퇴** → StaticPlanBlockMemory/StorageRewrite/USMP | 전제: G-buffer=TVM pool |
| `schedule_matmul` | **교체** → tir.Schedule + tensorize | |
| `codegen.py` op emitter | **교체** → 단일 TIR→ISA codegen | 특수케이스 소멸 |
| `_Walker` | **일반화 → 핵심 자산** | 이미 TIR→ISA |
| `emit_gemm` | fast-path로만 잔존 | 속도 캐시 |
| `tir_backend` matmul 스케줄 | tensorize intrinsic으로 | |
| `runtime`/`driver`/`isa` | **재사용** | 하위 emit/직렬화 |
| `direct` 백엔드 | **오라클로 유지**(+numpy `tiled_fp16_ref`) | 이행 대조용 |

**핵심 메시지**: frontend·layout·메모리·fusion의 수제 코드가 **대거 TVM 표준 패스로 흡수되어 사라진다**. 남는 NPU-고유 자산은 **intrinsic 정의 + TIR→ISA codegen** 둘로 수렴.

---

## 7. 검증 전략

- **오라클**: v1 hybrid(현재) + numpy `tiled_fp16_ref`. v2 각 op을 세우면 **오라클과 tolerance 대조** 후 교체.
- **게이트 진화**: Phase 2에서 **벤더 byte-exact → numpy tolerance**로 전환(스케줄 유연화 = FP16 순서 변동 → 비트일치 불가). **의식적 결정 사항.**
- **단계 독립**: 각 Phase는 중간에 멈춰도 동작(v1 방법론 유지).
- **회귀**: 새 경로용 단위 테스트(intrinsic lowering, layout 패스, memory 패스 각각).

---

## 8. 단계별 구현 계획 (go/no-go)

### Phase 0 — 타당성 spike ★ 최우선 (코드 재작성 없음)
검증할 가정 두 개:
1. **`npu.mma_64x64`를 TIR tensorize로 표현 → matmul 하나를 tensorize→codegen** 했을 때 v1 `emit_gemm`과 tolerance 일치?
2. **G-buffer를 storage pool로 선언 → StaticPlanBlockMemory/USMP가 우리 레이어를 플래닝** 가능?
- 부수: strided-전치/reduce/tile-layout이 tensorize vs 스케줄 vs 커스텀 중 무엇인지 판정.
- **go/no-go**: (1)(2) 통과 → Path A 확정. 일부 실패 → 하이브리드 비중 조정 or Path B.

### Phase 1 — path-무관 안전 정리 (병행, 저위험)
- `plan()`을 layout/alloc/liveness 별도 패스로 분리, 64/fusion/layout 파라미터화, codegen 특수케이스 정리.
- **DoD**: 매 커밋 gate GREEN, 출력 불변.

### Phase 2 — NPU ISA를 TIR intrinsic으로 + `_Walker` 일반화 (Phase 0 go 후, 핵심)
- intrinsic 등록 + codegen 일반화 + op 단위 점진 이행(v1=오라클, tolerance 대조).
- **게이트 tolerance 전환 여기서.**

### Phase 3 — Relax 파이프라인 완성 + 메모리 TVM화
- F1~F5 + M1을 TVM 패스로 조립. 고정 스케줄로 end-to-end 통과.

### Phase 4 — cost 기반 타깃 선택 (cycle 정보 도착 후) ★ 유보
- C-model cycle을 cost hook으로 → 타일/fusion/layout 후보 소수 cycle 비교 선택.

---

## 9. 열린 질문 · 리스크 · 필요 결정

1. **byte-exact 포기 합의** — auto-sched는 FP16 순서를 바꿈. 벤더 비트일치를 tolerance로 낮추는 것 확정 필요.
2. **NPU ISA의 tensorize 적합성** — matmul은 유력, strided-전치/native-reduce는 불확실 → **Phase 0가 판정**.
3. **컴파일 속도** — 임의 TIR walk 복귀로 A2(12.4×) 회귀 위험. canonical fast-path 병존으로 완화.
4. **G-buffer=TVM pool 표현** — 수동 offset 모델을 storage scope/pool로 옮기는 비용.
5. **TVM 버전 고정** — MetaSchedule/Relax API가 버전마다 변동. 사용 중 `tvm-src` 버전에 API 존재 확인 필요(Phase 0에서 같이).
6. **범위 재확인** — 타깃형이면 full MetaSchedule search가 과할 수 있음. Phase 0 후 "MetaSchedule search vs 소형 cost 셀렉터" 재결정.

---

## 부록 — 왜 이게 v1보다 "체계적"인가 (한 줄씩)
- 스케줄이 **탐색 가능한 대상**이 됨(하드코딩 64×64 → tensorize+스케줄) → auto-sched 가능.
- 메모리·fusion·layout hack이 **표준 패스로 흡수** → 수제 코드·특수케이스 급감.
- codegen이 **단일 TIR→ISA lowering** → op마다 emitter 짜는 구조 소멸.
- 남는 NPU 자산이 **intrinsic + codegen 둘**로 수렴 → 유지보수·확장 명확.
