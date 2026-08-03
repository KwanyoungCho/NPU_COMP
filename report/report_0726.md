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
| **2** | NPU ISA→TIR intrinsic + `_Walker` 일반화 | 🟢 대부분 | matmul + elementwise + reduce + broadcast + transpose/slice/concat 전부 walker로 |
| **3** | v2.compile() 파이프라인 조립 | 🟢 **완전한 레이어 컴파일** | `build_layer_module` → v2.compile_module → mysim, **ref_layer 대비 rel=0.0011** |
| **3-R** | **compile_module 리팩토링 → 명시적 pass 파이프라인 + v1 최적화 이식** | 🟡 진행 | Stage 1 완료(파이프라인 분해 `_parse`/`_plan_memory`/`_emit` + **A1 liveness 재사용**, act −47~63%, byte-exact). Stage 2=A4 tile, Stage 3=fusion |
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
| V2-009 | open | **중** | **tile-native 검증** — packed `[Rt,Ct,64,64]` 위 op(특히 reduce=tile+inner축 리덕션)이 스케줄+walker로 **효율적 native ISA fold**로 lower되는지. (ii) 확정용 | Phase 3 |
| V2-010 | **resolved** | **높** | **reduce-max ≥256이 8-bit 필드 wrap → softmax 오답/SIGSEGV**(SEQ≥256). guard 복원(loud fail). 진짜 지원은 tile-blocked 필요 | `_emit_rmax_row` assert |
| V2-011 | resolved | 중 | transpose C≥256 wrap(HD≥256) → guard 복원 | `_emit_transpose_row` assert |
| V2-012 | resolved | **높** | EW broadcast operand(작은 크기) → 할당 초과 read. operand==output 크기 assert | ew2/ew1 dispatch |
| V2-013 | resolved | **높** | non-last-axis strided_slice → 잘못된 행 + OOB write. last-axis assert | strided_slice dispatch |
| V2-014 | resolved | 중 | concatenate가 axis 무시 → non-last-axis 오답. last-axis(row 동일) assert | concatenate dispatch |
| V2-015 | resolved | 중 | broadcast_to `[1,1]` scalar 오분류 → source 초과 read. col=`sc==1 and sr==Rd` + valid assert | broadcast_to dispatch |
| V2-016 | resolved | 소 | sum/max axis/rank 미검사 → non-last-axis 오답. last-axis 2D assert | sum/max dispatch |
| V2-017 | open | 소 | strided_slice stride≠1 → contiguous 복사(v1 공유 한계). 현 경로 미사용 | later |

> 이슈 V2-010~017은 **adversarial-review workflow(15 agents, 8 distinct 확정 버그)** 가 발견. 공통 원인: v2가 v1의 **guard(assert/CodegenError)와 tile-path fallback을 떨어뜨림** → 범위 밖 값이 silent 오답. **모두 guard 복원으로 loud-fail 처리**(≥256 진짜 지원은 tile-blocked 레이아웃 대기). 현 5-config(SEQ≤128 등)는 전부 안전, 검증 rel≤0.0043 유지.

> 규칙: 새 이슈는 `V2-NNN`. 상태 ∈ {open, in-progress, resolved, wontfix}. resolve 시 커밋 해시 기록.

---

## 3. 결정 로그 (Decisions)

| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-07-26 | 최종 목표 = **Path A (TVM MetaSchedule)**, 범위 = 타깃형 cost 기반 | 정석 가속기 경로, 사용자 선택 |
| 2026-07-26 | **브랜치 작업 후 검증 완료 시 merge** | 안전 |
| 2026-07-26 | (예정) auto-sched 도입 시 **벤더 byte-exact → tolerance 전환** | schedule 유연화 = FP16 순서 변동 (Probe C가 재확인, V2-005) |
| 2026-07-26 | **Phase 0 = GO → Path A 확정** | 3/3 probe 통과: matmul tensorize 성립(A), TVM 메모리 플래닝 동작(B), codegen 일반화 MEDIUM(C). 모두 TVM 0.19.0 실측 |
| 2026-07-27 | **★ codegen 교정 — v1 emitter를 통째로 옮기지 않는다** | ISA(isa.py)가 elementwise·reduce_sum(0x14)·copy(0x17)·strided-load(0x90)·matmul mac/act(0x42)를 **네이티브 지원**. v1의 tile-fold·ones-matmul·relayout 복잡함은 **ISA가 아니라 tile-blocked 레이아웃 산물**. → v2 codegen = **"op→TIR→네이티브 ISA 1명령" 얇은 매핑 + 레이아웃은 Relax `layout_transform` 패스**. broadcast/transpose/concat의 tile 시퀀스는 포팅 안 함(패스로 흡수). col-broadcast의 0x15 주소지정 한계만 진짜 특별 처리. (사용자 지적) |
| 2026-07-27 | 남은 fork: tile-native (i)경계 relayout vs (ii)tile-native 생성 | (ii)가 A4 gather=0 보존이나 TVM 생성 가능성 확인 필요 → 레이아웃 spike |
| 2026-07-27 | **레이아웃 표현은 `layout_transform`로 확정.** fork는 **(ii) 지향** | spike: tile-blocked가 표준 op로 표현·legalize됨. (ii)는 packed `[Rt,Ct,64,64]` 위에서 op을 돌림(elementwise=layout-transparent 자명, reduce=tile+inner축 리덕션=fold를 **TVM이 생성**). A4 gather=0 보존. 단 "op-on-packed가 효율적 native fold로 lower되나"는 V2-009로 검증 |

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
| 2026-07-27 | **★★★ v2가 완전한 레이어 컴파일** | 3 | ✅ full-layer 스코핑 결과 남은 op은 딱 3개(strided_slice·concatenate·transpose) → walker에 추가 → `model.build_layer_module`(RMSNorm+RoPE/softmax attention+FFN) 전체를 v2.compile_module로 컴파일 → mysim, **v1 `ref_layer` 대비 rel=0.0011**. 모든 op 계열(matmul·ew·reduce·broadcast·transpose·slice·concat·상수) 커버. gate GREEN | v2_backend, test_v2.test_v2_compile_full_layer |
| 2026-07-27 | **compile_module 확장 → SwiGLU FFN + RMSNorm** | 3 | ✅ op-name dispatch(+`tir_` 접두사)로 unary ew(silu/exp/sqrt/...) 추가 → **SwiGLU FFN**. 그다음 **상수(full-size, id 아닌 (bind,arg) 위치로 키잉)** + reduce(sum/max) + broadcast(col=ones-matmul/row=copy) 추가 → **RMSNorm**(v1 legalize.rms_norm) mysim==numpy(maxdiff 0.008). gate GREEN | v2_backend.compile_module, test_v2 |
| 2026-07-27 | **★★ v2.compile_module 동작 (working 컴파일러)** | 3 | ✅ legalized Relax 멀티-op 모듈 → NPU ISA → mysim==numpy. `(x@w1+b)@w2`(matmul→add→matmul, 중간값 G-buffer 흐름) maxdiff=0.000. dispatch: matmul=v1 emit_matmul_into(tensorize 유도), binary ew=통합 walker marker. bump 할당. gate GREEN. **v2가 실제 서브그래프를 컴파일** | v2_backend.compile_module, test_v2.py |
| 2026-07-27 | **★ REAL 파이프라인 e2e 증명** | 3 | ✅ `Relax add → LegalizeOps → tir.Schedule(split CH + tensorize `npu_vadd`) → V2Walker → mysim == numpy`. **hand-marker가 아닌 진짜 TVM 흐름**이 non-matmul op에서 동작. enabler = `_bind_match` **N-D 일반화**(Probe C R4 해소: 2D matmul 전용 → 임의 rank). gate GREEN | v2_backend.py(_vadd intrin, schedule_ew, _bind_match override), test_v2.py |
| 2026-07-27 | **레이아웃 spike — Relax layout_transform** | 2 | ✅ `layout_transform(index_map (r,c)->(r//64,c//64,r%64,c%64))`가 tile-blocked `[Rt,Ct,64,64]` 표현 → **표준 TIR copy-reindex로 legalize**(순수 copy, //·%는 인덱스만). v1 수제 pack/relayout을 **TVM 표준 op로 대체 가능** 확정. → V2-009 | spike_layout.py |
| 2026-07-26 | v1 코드 확인 (tir_backend.py:44-126) | 0/2 | **★ v1은 이미 tensorize matmul 컴파일러** — `npu_gemm_acc`/`npu_fill_zero` TensorIntrin 등록 + `schedule_matmul`(canonical recipe) + walker lowering, byte-exact. → Phase 2 "matmul tensorize" step 사실상 완료. 남은 건 non-matmul op 일반화 | tir_backend.py |
| 2026-08-03 | **자기검토 — compile_module이 계획대로인가?** | 3-R | ❗ 진단: `compile_module`이 계획한 pass 파이프라인이 아니라 **모놀리식 op-dispatcher**(자체 bump 메모리·op별 손 marker·layout/fusion 패스 없음)로 흐름 = **v1 "복잡한 단일 pass" 재현**. v1 `plan()`의 A1/A4/packing을 **전혀 안 씀**. 사용자 지적 타당 → 리팩토링 착수 | — |
| 2026-08-03 | **Stage 1 — 파이프라인 분해 + A1 메모리 재사용** | 3-R | ✅ `compile_module`을 3-pass로 분해: `_parse`(F5-read: legalized Relax→ops+tensor table+consts) → `_plan_memory`(M1: **liveness last-read + exact-size free-list**, params/consts/output persist) → `_emit`(C1: walker/matmul). **A1이 legalized 그래프에서 이식됨**(high-level op 이름 불필요). 측정: activation 영역 **−47~63%**(REDUCED 47.9·MEDIUM 63.2·GQA 50.0·wide 52.8·HD32 47.1%), reuse⟺bump **비트 동일 5/5**. gate GREEN, `test_v2_memory_reuse` 편입 | v2_backend.py(_Op/_parse/_plan_memory/_emit), test_v2.py |
| 2026-07-26 | Probe C (codegen 일반화 평가) | 0 | **MEDIUM/GO** — walker가 이미 tensorized matmul TIR을 프로덕션 lower(O-proj group, byte-exact 검증). ~90% 재사용. 새 작업=cache stage(V2-003). 전략=fast-path 유지+walker fallback(V2-004). byte-exact는 schedule 변경 시 상실→tolerance(V2-005) | — |

---

## 5. 검증 로그 (Verification)

| 날짜 | 대상 | 기준 | 결과 |
|---|---|---|---|
| 2026-07-27 | v2 op별(elementwise/reduce/primitive) | v1 ISA byte-exact + mysim | ✅ 통과 |
| 2026-07-27 | v2.compile_module 서브그래프(matmul chain, SwiGLU, RMSNorm) | mysim==numpy tolerance | ✅ maxdiff≤0.008 |
| 2026-07-27 | **v2.compile_module 완전한 레이어 (multi-config)** | v1 `ref_layer` 대비 rel<0.05 | ✅ **5/5 PASS**: REDUCED 0.0011 · MEDIUM 0.0012 · GQA(H4/KV2) 0.0009 · wide(D192/F384) 0.0043 · HD32 0.0024 |
| 2026-07-27 | v2 adversarial 코드리뷰 (workflow, 15 agents) | 확인된 correctness 버그 | **8 distinct 버그 발견·전부 검증** → guard 복원으로 loud-fail 처리(V2-010~017). SEQ=256이 이제 silent 오답 대신 AssertionError |
| 2026-08-03 | **Stage 1 리팩토링 — A1 메모리 재사용** | reuse⟺bump 비트 동일 + peak 감소 | ✅ **5/5 비트 동일**(REDUCED/MEDIUM/GQA/wide/HD32 maxdiff=0.0), activation −47~63%. 전체 gate GREEN(15/15+vendor byte-exact) |

---

## 6. 커밋 로그 (v2 branch)

| 커밋 | 내용 |
|---|---|
| `8e75fe3` | v2 setup — COMPILER_V2_PLAN.md + report_0726.md, compiler-v2 branch |
| `63f2fa2` | Phase 0 GO — 3 probe(tensorize/memory/codegen) 통과, TVM 0.19.0 |
| `12c349c` | Phase 2-A.1 — elementwise 8종 → 통합 walker (byte-exact + mysim) |
| `71f1098` | Phase 2-A.2 — reduce(rsum/rmax × row/tile) → 통합 walker |
| `dc154fa` | **course-correct** — thin native-ISA 매핑 + copy/ttile primitive (v1 워크어라운드 안 옮김) |
| `10c5942` | 레이아웃 spike — tile-blocked = Relax layout_transform 확정, fork→(ii) |
| `cea3547` | **★ REAL 파이프라인 e2e** — Relax→LegalizeOps→tensorize→walker→mysim + `_bind_match` N-D |
| `82340b9`~`5ef22ca` | Phase 3 — compile_module e2e → unary ew → consts/reduce/broadcast → **FULL LAYER** → 5-config 검증 → correctness hardening(guard 복원) |
| *(this)* | **Stage 1 리팩토링** — compile_module을 `_parse`/`_plan_memory`/`_emit` 3-pass로 분해 + **A1 liveness 메모리 재사용 이식**(act −47~63%, byte-exact) |
