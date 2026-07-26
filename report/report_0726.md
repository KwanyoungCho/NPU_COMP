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
| **0** | 타당성 spike (tensorize/memory/codegen 검증) | 🟡 진행 | — |
| **1** | path-무관 안전 정리 (plan() 패스 분리·파라미터화) | ⚪ 대기 | — |
| **2** | NPU ISA→TIR intrinsic + `_Walker` 일반화 | ⚪ 대기 | Phase 0 go 필요 |
| **3** | Relax 파이프라인 완성 + 메모리 TVM화 | ⚪ 대기 | — |
| **4** | cost 기반 타깃 선택 (cycle 도착 후) | ⚪ 유보 | cost model 필요 |

범례: ⚪대기 🟡진행 🟢완료 🔴블록

---

## 2. 이슈 트래커

| ID | 상태 | 심각도 | 제목 | 관련 |
|---|---|---|---|---|
| — | — | — | (Phase 0 probe 결과로 채워짐) | — |

> 규칙: 새 이슈는 `V2-NNN`. 상태 ∈ {open, in-progress, resolved, wontfix}. resolve 시 커밋 해시 기록.

---

## 3. 결정 로그 (Decisions)

| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-07-26 | 최종 목표 = **Path A (TVM MetaSchedule)**, 범위 = 타깃형 cost 기반 | 정석 가속기 경로, 사용자 선택 |
| 2026-07-26 | **브랜치 작업 후 검증 완료 시 merge** | 안전 |
| 2026-07-26 | (예정) auto-sched 도입 시 **벤더 byte-exact → tolerance 전환** | schedule 유연화 = FP16 순서 변동 |

---

## 4. 멀티 에이전트 작업 로그 (Work Log)

| 날짜 | 에이전트/작업 | Phase | 결과 요약 | 산출 |
|---|---|---|---|---|
| 2026-07-26 | 환경 API 서베이 | 0 | TVM 0.19.0, 필수 API 전부 존재 | §0 |
| 2026-07-26 | Probe A (matmul tensorize) | 0 | (진행) | — |
| 2026-07-26 | Probe B (memory planning) | 0 | (진행) | — |
| 2026-07-26 | Probe C (codegen 일반화 평가) | 0 | (진행) | — |

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
