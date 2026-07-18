# 컴파일러 최적화 리팩토링 계획 (완결형)

목표: 현재 컴파일러를 **최적화된 최종 형태**로 마무리한다. 각 Stage는 **독립적으로
테스트 통과 + 측정 기록**을 만족해(= 중간에 멈춰도 동작), 순서대로 끝내면 최적화 완료.

측정은 `report/figs/0710/measurements.json` 및 아래 지표를 매 Stage 갱신:
`mp.top`(버퍼), 컴파일 시간, gather/scatter %, 명령 수, 전체 테스트 green.

---

## 0. 현재 구조 분석 (grounded)

파이프라인:
```
torch import(frontend) ─┬─ import_legalize (HF)          ← (A3) 경로 분기
                        └─ legalize (manual builders)
   → memplan.plan  (bump allocator, 재사용 0)             ← (A1)
   → codegen.compile_func  (op 디스패치)
        ├─ emit_matmul: direct 타일링 (오라클/폴백)        ← (A5) 중복
        └─ mm_backend="tir": tir_backend._Walker (전부 unroll)
   → isa.Asm.words (Python list) → runtime._program_bytes (per-word pack) ← (A2)
   → mysim  ← driver (레이어/스텝 오케스트레이션)
```

핵심 사실(파일 확인):
- **memplan**: `self.top`만 증가, 주석 "no reuse yet". binding var·scratch 모두 fresh offset → 3B 버퍼 494MB~1.6GB.
- **scratch_alloc 12곳**(codegen 6 + tir_backend 6): 대부분 op-transient(sA/sB/sC, ones/sP, scr, acc, apad/bpad/cpad). 지속: `gather_cache` dst, `cbuf`(flush까지).
- **unroll**: 명령 수 = 타일 수(3B gate/up 3.5M). `_program_bytes`는 워드마다 `struct.pack`.
- **legalize 분기**: manual = stable softmax + native SiLU + sign-inv RoPE; import = non-stable + 5-op silu + slice/concat.
- **direct 백엔드**: `test_real_layer`·`test_tiling`이 byte-exact 오라클로 사용.
- **레이어 op 시퀀스**(legalize): RMSNorm(mul·mul·reduce·add·sqrt·div·bcast·mul·bcast·mul) → Q/K/V proj(matmul) → RoPE(slice/neg/concat·mul·add) → QKᵀ(transpose+matmul) → softmax(max·bcast·sub·exp·sum·bcast·div) → PV(matmul) → Oproj(matmul)+resid → RMSNorm → FFN(matmul·silu·mul·matmul)+resid.

레이아웃 경계(= gather/scatter 발생점): **matmul 입출력**(현재 row-major↔tile gather/scatter),
**reduce**(RMSNorm mean, softmax max/sum), **transpose**(Kᵀ), **broadcast**(scale/denom).

---

## 1. 목표 아키텍처 (End State)

- **단일 legalization**(import는 공통 legalize로 위임하는 얇은 어댑터).
- **tir 단일 matmul 백엔드**(direct 제거, 오라클은 numpy `tiled_fp16_ref`).
- **빠른 컴파일**: 벡터화 직렬화 + 타일 주소 증분(no re-simplify).
- **liveness 기반 메모리 재사용**: 버퍼 = peak-live(합계 아님).
- **tile-blocked 레이아웃 전파**: matmul 체인의 gather/scatter 소멸, reduce/transpose/broadcast만 layout-aware(경계에서만 re-layout).

---

## 2. Stage 별 계획 (각 Stage = 완결 증분)

### Stage 0 — 기준 고정 (0.5d)
- 전체 테스트 green 확인, 현재 지표를 measurements.json에 baseline으로 기록.
- **DoD**: 회귀 게이트(전체 테스트 + byte-exact 63/63) 스크립트 1개로 정리.

### Stage 1 — (A5) direct 백엔드 제거 [소]
- `codegen.emit_matmul`의 direct 타일링 분기 삭제(tir로 일원화).
- `test_tiling`·`test_real_layer`의 오라클을 `direct==tir` → `numpy tiled_fp16_ref==tir`로 교체(이미 ref 존재).
- **DoD**: 전체 테스트 green. codegen LOC 감소. 의미 변화 0.
- **위험**: 낮음(오라클 교체만).

### Stage 2 — (A2) 컴파일 속도 [소~중, SW]
- 2a: `runtime._program_bytes` → `np.asarray(words, np.uint32).tobytes()`.
- 2b: 프로파일(cProfile) → hot path 확정. `_Walker.ev()`의 `arith.simplify`가 hot이면
  타일 주소를 **증분/캐시**(k-loop에서 base+delta) 로 계산해 재-simplify 제거.
- **DoD**: 3B 커널 컴파일 시간 대폭↓(목표 100s→10s대), **byte-exact 유지**(속도만).
- **위험**: 낮음(수치 불변).
- 주의: 최종 **명령 수는 HW 루프 부재로 불변**(문서에 명시, HW 요청 별도).

### Stage 3 — (A3) legalization 통합 [중]
- `import_legalize`의 softmax/silu/rope를 `legalize.py` 공통 빌더로 위임
  (stable softmax, native SiLU, sign-inv+on-device RoPE 공유).
- import 경로 회귀 테스트 추가(HF 레이어 == manual 레이어 값 일치).
- **DoD**: import·manual 두 경로가 **동일 코드·정확도**. 전체 테스트 green.
- **위험**: 중(HF 프론트엔드 op 형태 차이) → import 어댑터에서 shape/attr 정규화.

### Stage 4 — (A1) liveness 기반 메모리 재사용 [대] ★
- 4a: **graph-var liveness** — SSA use-def로 각 binding var의 [def, last-use] 구간 계산.
  topo 순서 **linear-scan** + free-list로 offset 재사용(같은 크기 우선, 아니면 best-fit).
- 4b: **scratch 풀링** — op-transient scratch를 op 종료 시 반환하는 풀(`scratch_scope`).
  `gather_cache`/`cbuf`는 layer/flush 스코프로 명시적 관리.
- 불변식: **출력 값 불변**(데이터 위치만 바뀜) → 테스트는 값 비교라 green.
- **DoD**: 전체 테스트 green + byte-exact 유지. `mp.top` 대폭↓(목표 3B 레이어 수십× 감소).
- **위험**: 중(오프셋 충돌 시 잘못된 재사용 → 오답). liveness 정확성 단위테스트 필수.
- 검증: 재사용 on/off 출력 **완전 일치** 자동 비교.

### Stage 5 — (A4) tile-blocked 레이아웃 전파 [대] ★
- 5a: **레이아웃 배정 패스** — 텐서별 layout∈{row_major, tile_blocked} 결정.
  matmul in/out·elementwise·residual = tile_blocked 전파; 경계(입력·최종출력·reduce·transpose·broadcast) = row_major, 경계에만 **re-layout(=현 gather/scatter)** 삽입.
- 5b: matmul이 tile_blocked를 직접 read/write → **체인 내부 gather/scatter 소멸**.
- 5c: **layout-aware reduce/broadcast/transpose**(RMSNorm·softmax): tile_blocked에서 논리 행을
  다루도록 재구현하거나 그 지점에서 국소 re-layout.
- 단계적: (i) projection→projection 체인만 → (ii) FFN 체인 → (iii) RMSNorm/softmax 경계.
- 불변식: 출력 tolerance 유지. gather/scatter %가 체인에서 ~0.
- **DoD**: 전체 테스트 green + 3B 레이어 gather/scatter 대폭↓ 측정. 각 (i)(ii)(iii) 독립 green.
- **위험**: 높음 → 서브스테이지별로 격리·측정.

---

## 3. 순서 & 종료 조건

권장 순서: **Stage 0 → 1 → 2 → 3 → 4 → 5**.
(소·저위험부터 → 대공사 A1 → A4. 각 Stage green 전엔 다음으로 안 넘어감.)

**최종 종료 조건(End State 달성)**:
- 전체 테스트 + byte-exact 63/63 green
- direct 백엔드 제거, legalize 단일화
- 3B 레이어: **버퍼 수십× 감소(A1)**, **matmul 체인 gather/scatter ~0(A4)**, 컴파일 시간 대폭↓(A2)
- measurements.json에 Stage별 전/후 지표 전부 기록
- 남는 것은 **HW 의존 항목**(루프 → 명령 수, register-indirect → KV/가변길이)뿐 — 별도 벤더 요청서로 분리(SW로 마무리 가능한 것은 전부 완료)

---

## 4. 리스크 관리 (중간 결과물 방지)
- 매 Stage: 착수 전 브랜치, 완료 시 **전체 테스트 + byte-exact** 통과해야 커밋.
- 값-불변 Stage(1,2,4)는 **출력 완전일치** 자동 비교. 값-변경 Stage(3,5)는 tolerance + 참조 비교.
- 각 Stage 독립 커밋 → 언제 멈춰도 직전 Stage는 **완결·동작** 상태.
