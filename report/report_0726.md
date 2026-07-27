# NPU 컴파일러 v2 — TVM-native 재구성 (진행 추적 보고서, 2026-07-26~)

> **목적**: 이 문서는 v2(대규모 재구성, Path A = TVM MetaSchedule)의 **살아있는 추적 문서**다.
> 이슈 트래킹 · 각 단계 구현 내용 · 멀티 에이전트 작업 로그 · 검증 결과를 **모두** 여기서 확인한다.
> 설계 근거·패스 구조: [`d_compiler/COMPILER_V2_PLAN.md`](../d_compiler/COMPILER_V2_PLAN.md).
> v1(완료) 결과: [`report/report_0719.md`](report_0719.md).

- **브랜치**: `compiler-v2` (검증 완벽 시에만 `main`에 merge)
- **방법론**: 오라클 주도(v1=numerical 오라클) · 매 단계 gate 유지 · 빅뱅 금지 · 요구 주도
- **관리**: 멀티 에이전트로 병렬 조사/구현, 결과는 이 문서의 로그·이슈로 집계

---

## 0. 환경 (Phase 0에서 확정)

| 항목 | 값 |
|---|---|
| TVM | **0.19.0** (`/home/chokwans99/tvm-src`) |
| python | `/home/chokwans99/anaconda3/envs/npu-tvm/bin/python` (conda `npu-tvm`) |
| PYTHONPATH | `/home/chokwans99/NPU_cmodel/d_compiler` |
| **필수 API 가용성** | ✅ 전부 존재: `relax.frontend.torch`·`LegalizeOps`·`FuseOps`·`FuseTIR`·`StaticPlanBlockMemory`·`tir.TensorIntrin`·`StorageRewrite`·`meta_schedule`·`tir.usmp`·`relax.op.layout_transform` |

→ **Path A 파이프라인의 모든 TVM 패스가 설치본에 존재** — 인프라 관점 최대 리스크 해소.

---

## 1. Phase 상태 보드

| Phase | 내용 | 상태 | go/no-go |
|---|---|---|---|
| **0** | 타당성 spike (tensorize/memory/codegen 검증) | 🟢 **GO** | 3/3 probe 통과 (A GO · B GO-caveats · C MEDIUM/GO) |
| **1** | path-무관 안전 정리 | ⚪ 선택적/후순위 | layout·liveness는 이미 별도 함수 → ROI 낮음, 필요시만 |
| **2** | NPU ISA→TIR intrinsic + `_Walker` 일반화 | 🟡 진행 | matmul=이미 완료(v1). 2-A.1 elementwise ✅ |
| **3** | Relax 파이프라인 완성 + 메모리 TVM화 | ⚪ 대기 | — |
| **4** | cost 기반 타깃 선택 (cycle 도착 후) | ⚪ 유보 | cost model 필요 |

범례: ⚪대기 🟡진행 🟢완료 🔴블록

---

## 2. 이슈 트래커

| ID | 상태 | 심각도 | 제목 | 관련 |
|---|---|---|---|---|
| V2-001 | open | 설계 | **fill-vs-MAC 비트(ko==0)는 tensorize로 표현 불가** — 어디서 lower할지 결정 필요(TIR 패스 vs walker 상태머신). Probe A 발견 | Phase 2 |
| V2-002 | open | 설계 | init 블록(C=0) 처리 — 별도 fill intrinsic vs 첫 MAC에 fold(v1 방식: fp16(0+x)==fp16(x)) | Phase 2 |
| V2-003 | open | **중** | `_bind_match` 확장 — `cache_read/write` scratch 스코프 버퍼 지원(scratch 할당+copy emit). Probe C가 "진짜 새 작업(R1)"으로 지목 | Phase 2 |
| V2-004 | open | 중 | **canonical-schedule detector** — 고정 64³ 스케줄은 `emit_gemm` fast-path 유지, 나머지만 walker. 12.4× 컴파일 이득 보호 | Phase 2 |
| V2-005 | open | 설계 | 게이트 분리 — 고정 canonical 스케줄=byte-exact, auto-scheduled=numpy tolerance. (결정 로그의 byte-exact→tolerance를 2-tier 테스트로 구체화) | Phase 2 |
| V2-006 | open | **중** | **tile-blocked footprint vs TVM logical-size 정합** — TVM는 논리 byte-size로 플래닝, v1은 padded `tiled_numel`. offset 충돌 방지 위해 sinfo에 padded shape 반영 필요. Probe B 카베아트(E) | Phase 3 |
| V2-007 | open | 소 | **offset post-pass** — StaticPlanBlockMemory는 N개 storage 객체 반환(플랫 offset 아님) → survivor에 base offset bump(~30줄). 또는 USMP로 진짜 offset. | Phase 3 |
| V2-008 | open | 소 | decode 동적 shape는 `tir_var_upper_bound` func_attr 필요(플래너 sizing). Probe B 카베아트(F) | Phase 3 |

> 규칙: 새 이슈는 `V2-NNN`. 상태 ∈ {open, in-progress, resolved, wontfix}. resolve 시 커밋 해시 기록.

---

## 3. 결정 로그 (Decisions)

| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-07-26 | 최종 목표 = **Path A (TVM MetaSchedule)**, 범위 = 타깃형 cost 기반 | 정석 가속기 경로, 사용자 선택 |
| 2026-07-26 | **브랜치 작업 후 검증 완료 시 merge** | 안전 |
| 2026-07-26 | (예정) auto-sched 도입 시 **벤더 byte-exact → tolerance 전환** | schedule 유연화 = FP16 순서 변동 (Probe C가 재확인, V2-005) |
| 2026-07-26 | **Phase 0 = GO → Path A 확정** | 3/3 probe 통과: matmul tensorize 성립(A), TVM 메모리 플래닝 동작(B), codegen 일반화 MEDIUM(C). 모두 TVM 0.19.0 실측 |

### Phase 0 결론 (go/no-go)
**GO.** Path A(TVM MetaSchedule)의 세 load-bearing 가정이 실측으로 검증됨:
1. **tensorize**(A): NPU 64×64 MAC → TIR TensorIntrin, matmul tensorize 성립. 생성 call이 v1 walker 인자와 동일 → codegen 재사용 큼.
2. **memory**(B): `StaticPlanBlockMemory`가 liveness+재사용 수행(v1 reuse 대체 가능), best-fit이라 fragmentation도 개선 여지. offset은 얇은 post-pass로.
3. **codegen**(C): v1 walker가 이미 tensorized matmul을 프로덕션 lower 중 → 재작성이 아니라 MEDIUM 수준 확장(cache stage + emit_* 재배치). fast-path 유지로 12.4× 컴파일 보호.
**리스크는 전부 관리 가능**(연구 리스크 없음). 다음: Phase 1(안전 정리) 착수 가능, Phase 2는 V2-001~005 순서로.

---

## 4. 멀티 에이전트 작업 로그 (Work Log)

| 날짜 | 에이전트/작업 | Phase | 결과 요약 | 산출 |
|---|---|---|---|---|
| 2026-07-26 | 환경 API 서베이 | 0 | TVM 0.19.0, 필수 API 전부 존재 | §0 |
| 2026-07-26 | Probe A (matmul tensorize) | 0 | **GO** — 64×64 MAC이 TensorIntrin으로 tensorize됨(단 `decompose_reduction` 필수). 생성 call `(C_ptr,C_s,A_ptr,A_s,B_ptr,B_s)` = v1 `npu_gemm_acc`와 동일 → walker 재사용 가능. multi-K/N OK. → V2-001/002 | 스크립트 spike_tensorize.py |
| 2026-07-26 | Probe B (memory planning) | 0 | **GO-caveats** — `StaticPlanBlockMemory`가 실제 liveness+재사용(5중간→2 storage, residual 최적). best-fit(match_range 16)로 v1 exact-size보다 유연(fragmentation↓ 여지). 파이프라인 prefix(→CallTIRRewrite→plan) 필수. N객체→offset post-pass(V2-007), tile footprint 정합(V2-006), decode 동적(V2-008) | exp2_plan.py, exp3_residual.py |
| 2026-07-26 | **Phase 2-A.1 — elementwise → 통합 walker** | 2 | ✅ `v2_backend.V2Walker`가 elementwise 8종(add/sub/mul/div/sqrt/exp/neg/cos/sin/silu)을 `npu_ew*` marker로 lower. **v1 emit_ew와 ISA byte-exact + mysim 수치 일치**. v1 무변경(오라클). **gate GREEN(15/15+vendor)**. test_v2 게이트 편입 | v2_backend.py, tests/test_v2.py |
| 2026-07-27 | **Phase 2-A.2 — reduce → 통합 walker** | 2 | ✅ `npu_rsum/rmax_row/tile` 4경로(row·tile × sum·max, scratch 사용)를 marker로 lower. **v1 emit_row_sum/max와 ISA byte-exact + mysim==numpy**. gate GREEN | v2_backend.py, tests/test_v2.py |
| 2026-07-26 | v1 코드 확인 (tir_backend.py:44-126) | 0/2 | **★ v1은 이미 tensorize matmul 컴파일러** — `npu_gemm_acc`/`npu_fill_zero` TensorIntrin 등록 + `schedule_matmul`(canonical recipe) + walker lowering, byte-exact. → Phase 2 "matmul tensorize" step 사실상 완료. 남은 건 non-matmul op 일반화 | tir_backend.py |
| 2026-07-26 | Probe C (codegen 일반화 평가) | 0 | **MEDIUM/GO** — walker가 이미 tensorized matmul TIR을 프로덕션 lower(O-proj group, byte-exact 검증). ~90% 재사용. 새 작업=cache stage(V2-003). 전략=fast-path 유지+walker fallback(V2-004). byte-exact는 schedule 변경 시 상실→tolerance(V2-005) | — |

---

## 5. 검증 로그 (Verification)

| 날짜 | 대상 | 기준 | 결과 |
|---|---|---|---|
| — | (Phase별 DoD) | gate/tolerance | — |

---

## 6. 커밋 로그 (v2 branch)

| 커밋 | 내용 |
|---|---|
| (this) | v2 setup — COMPILER_V2_PLAN.md + report_0726.md, compiler-v2 branch |
