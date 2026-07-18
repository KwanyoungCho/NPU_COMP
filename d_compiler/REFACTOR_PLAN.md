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

### Stage 2 — (A2) 컴파일 속도 [소~중, SW] ✅ phase-1 (커밋 3f3ef0b)
- 프로파일(cProfile)로 hot path 확정: `_Walker.ev()`가 인덱스식마다 TVM FFI로
  `substitute`+`simplify`(769k회, ~70s), `_bind_match`(ev 호출, ~109s), `_bind`의 IntImm 생성.
- **2b ✅**: `ev()`를 **순수 파이썬 affine 평가**(`_ev_fast`: Add/Sub/Mul/FloorDiv/FloorMod/Min/Max/Cast를
  int env로 평가)로 대체, 미지원 노드만 substitute+simplify로 fallback. env를 python int로 저장(IntImm FFI 제거).
- **2a ✅**: `runtime._program_bytes` per-word struct.pack 루프 → numpy 벡터화 uint32 pack(바이트 동일).
- **실측**: 3B prefill layer 컴파일 **106.3s → 58.5s (1.82x)**. gate GREEN, vendor byte-exact 유지.
- **phase-2(미착수, 10s대 목표)**: GEMM shape별 walk 결과(intrinsic 시퀀스)를 메모이즈해 동일 shape 재-walk 제거.
  더 큰 refactor·리스크라 보류. 컴파일은 kernel당 1회(토큰/레이어 재사용)라 우선순위 낮음.
- 주의: 최종 **명령 수는 HW 루프 부재로 불변**(속도만; HW 요청 별도).

### Stage 3 — (A3) legalization 통합 [중] ✅ (커밋 b34c723)
- `import_legalize`의 silu→`legalize.silu`(native), softmax→`legalize.softmax_lastdim`(stable),
  negative→native 유지(sign-inv, RoPE rotate_half 공유). import·manual 두 경로가 **동일 lowering**.
- 회귀: 기존 `test_import.test_llama_block`(full Llama decoder block import vs torch, rel=0.025<0.05)이
  통일된 lowering을 커버. rope는 import에서 이미 primitive(slice/concat/negative/mul/add)라 manual과 동일.
- **결과**: 전체 gate GREEN + vendor byte-exact 유지. 향후 실제 HF 모델도 동일 최적화 경로로 import됨.
- 미변경: `_power`/`_rsqrt`/`_mean`(RMSNorm 내부)은 manual과 같은 primitive라 유지.

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

## 2.5. ★ A1 실측 결과 & 재정렬 (2026-07-14)

A1(binding-var liveness 재사용)을 프로토타입해 **3B prefill layer로 실측**한 결과 재정렬 필요:

| | 메모리(mp.top) | 명령 수 |
|---|---:|---:|
| bump(현행) | 293MB | 2,235,194 |
| liveness 재사용 | 263MB (**−10%뿐**) | **4,236,090 (+90%!)** |

- **메모리 −10%뿐**: 버퍼는 **가중치(~200MB, 레이어 상주 불가피)** 가 지배. 활성화 재사용 효과 작음.
- **명령 +90%**: liveness 재사용은 **write-once 가정을 깨서** `gather_cache`(활성화 gather 재사용)와
  충돌 → 비활성화하면 xn/hn 재-gather로 명령 2배. (MEDIUM에선 출력 bit-exact 확인했으나 3B에서 net-손해)
- **O-proj 융합**이 leaf 입력을 root에서 지연-read → naive SSA liveness가 조기 free(0.51 오차). liveness가
  codegen의 **실제 스케줄(융합/reschedule)** 과 일치해야 함 = 추가 결합.
- **핵심**: 재사용 가능한 메모리 bulk는 **gather scratch(~80MB, 누적)** 인데 이는 **A4가 제거**한다.
  그리고 **A4 이후엔 gather가 없어 gather_cache도 없으니 binding 재사용이 안전**해진다.

→ **결론: A1은 A4에 포섭된다. A4를 먼저 하고, A1(binding+비-gather scratch 재사용)은 A4 이후
minor 단계로.** (A1 프로토타입은 net-손해라 default 미탑재, 되돌림.)

## 3. 순서 & 종료 조건 (실제 진행: Stage 0 → A4 → A3 → A2, A1 보류)

원래 순서는 A4 → A1 → A3 → A2였으나, A1의 전제(A4가 gather를 **전부** 제거 → gather_cache 불필요)가
A4를 **5c(FFN)에서 확정**하며 부분적으로만 성립(attention gather_cache 여전) → **A1 보류**, A3·A2 우선.

**최종 상태(2026-07-18)**:
- **A4 ✅ (5c, byte-exact)**: FFN 체인 tile-blocking. 3B prefill layer **−19.8%**(2,235,194→1,792,826),
  scatter −58%. `layouts=True/False` 토글로 A/B 비교. 5d(RMSNorm/attention)는 reduce 재정렬로 byte-exact
  불가라 보류(§3.5).
- **A3 ✅ (커밋 b34c723)**: import legalization을 manual 경로와 통일(native SiLU/stable softmax/sign-inv).
- **A2 ✅ phase-1 (커밋 3f3ef0b)**: 컴파일 106.3s→58.5s(1.82x, byte-exact).
- **A1 보류**: net-손해(§2.5) + A4 부분 완료로 attention gather_cache 여전 활성 → 리스크. 5d 완료 후 재평가.
- **전체 gate GREEN + vendor byte-exact 전 구간 유지.**

**남는 것(우선순위·리스크順)**: A2 phase-2(walk 메모이즈, 10s대) · 5d(tolerance 수용 시 attention/RMSNorm 타일링,
−~8%+) · A1(5d 후) · direct 백엔드는 **golden 오라클로 유지**(제거 안 함). HW 의존(루프→명령 수,
register-indirect→KV/가변길이)은 별도 벤더 요청서.

---

## 3.5. A4 상세 설계 (tile-blocked 레이아웃 전파)

**레이아웃 태그**: 각 텐서에 `layout ∈ {ROW, TILE}`.
- `TILE` = `[⌈R/64⌉, ⌈N/64⌉, 64, 64]`(타일 row-major, 타일 내부 row-major, 패딩 0). matmul 출력 타일을
  **연속 저장**(scatter 0), 다음 matmul이 **연속 읽기**(gather 0).
- `ROW` = 현행 `[R,C]`.

**레이아웃 배정 패스**(relax 그래프 위):
- matmul in/out → TILE 선호.
- elementwise/residual(add/mul/silu) → **레이아웃 투명**(두 피연산자 blocking 동일하면 원소별 그대로 동작) → TILE 전파.
- reduce(sum/max, last-dim), broadcast, transpose → 논리 행/열 필요 → **ROW**.
- 그래프 입력·최종 출력 → ROW.
- 불일치 지점에 **re-layout(ROW↔TILE)** 삽입 = 현행 gather/scatter를 **경계로 이동**.

**핵심 이득**: matmul **체인**(특히 FFN: gate/up→silu·mul→down)이 TILE로 이어지면 그 사이 gather/scatter 소멸.
- FFN 예: `hn(ROW)` →[relayout 1회]→ `TILE_hn` → gate/up(A-gather 0) → silu·mul(TILE) → down(A-gather 0) →[relayout 1회]→ ROW.
  현행 대비 **down의 A-gather + gate/up의 재-gather 제거**. FFN이 matmul 비용 최대라 큰 이득.

**구현 서브스테이지**(각각 독립 green + gather 측정):
- **5a ✅ (커밋 8282758)**: memplan `layout`/`alloc_tiled` + `pack_tiled/unpack_tiled/tiled_numel` + round-trip 테스트.
- **5b ✅ (커밋 4be9ccc)**: `emit_matmul_into`에 `a_tiled/c_tiled` — A/C를 `packed_src`에 등록 → tile-contiguous
  read/write로 gather/scatter skip. **실측 128×3072×128: 103,596 → 3,244 (32×↓), gather/scatter→0, byte-exact.**
- **5c ✅ (커밋 6ac0d5d)**: `memplan.assign_layouts`(fixpoint) — matmul out(64-mult)/투명 elementwise가
  모든 소비자 TILE-호환 시 TILE, 아니면 ROW(일관성 → relayout 불필요). TILE var는 `alloc_tiled`,
  codegen이 layout 따라 a_tiled/c_tiled + elementwise physical 크기. **3B prefill layer(gate GREEN, 출력 불변):
  total 2,235,194 → 1,792,826 (−19.8%), gather −32%, scatter −58%** (FFN 체인 tile-blocking). TILE var 28.
- **토글 ✅ (커밋 34d88be)**: `driver.compile_module/run_module`에 `layouts=True/False` — A4 on/off A/B 비교용.
- **5d (다음) — 타겟팅 실측(layouts=True, 5c 이후)**:

  | op | gather | scatter | 원인(ROW 경계) |
  |---|---|---|---|
  | attn score/ctx | **131,072** | 49,152 | Qr/Kr(RoPE)·P(softmax)·Kt(transpose)가 ROW |
  | Q/K/V proj | 49,152 | **81,920** | 출력이 RoPE(slice/concat)로 감 → ROW |
  | O proj | 49,152 | 49,152 | ctx(attn) 입력 ROW, residual add 출력 ROW |
  | gate/up | 49,152 | 0 | 입력 rms2(RMSNorm) ROW |
  | down | 0 | 49,152 | 출력 residual add ROW |

  **핵심 발견**: 남은 비용은 전부 **row↔tile 경계**(RMSNorm·RoPE·softmax·transpose·residual = 전부 ROW op)에 있음.
  경계에서 relayout은 데이터 볼륨이 같아 gather와 **비용 동일**(예: [128,3072] gather=49,152 = relayout scatter=49,152).
  → **경계를 relayout로 감싸는 건 이득 0**; 그 op들 자체를 **tile-native**(tile 레이아웃에서 직접 계산)로 만들어야 제거됨.

  **5d 결정: 보류(A4는 5c에서 확정). 이유 = byte-exact 불가.**
  - 5d는 reduce(RMSNorm의 sum, softmax의 sum/max)를 tile화해야 하는데, **tile reduce는 FP16 합산 순서를 바꿈**
    (3072개 일괄 reduce → 48개 64-col 타일 elementwise 누적 후 행 reduce). broadcast/const-pack은 정확한 순열이라
    무해하지만 **reduce만 산술 재정렬** → 비트 동일 불가. byte-exact를 유지하려면 reduce가 행을 gather해야 하고
    (parity), 그러면 tile화 이득이 소멸. **독립 matmul→RMSNorm→matmul로 실증: rel≈0.18%(maxdiff 9.77e-3), tolerance-valid.**
  - 5c는 **주소만 변경(같은 MAC 순서)** 이라 비트 동일이었음 → **A4의 byte-exact 실현 가능 구간은 5c가 끝**.
  - 사용자 결정(2026-07-18): **5c의 비트-동일(layouts=True==False, 깨끗한 A/B 비교)을 우선**, 5d 미진행.
  - **설계 보존(향후 full-model fusion 시 유효)**:
    - 5d-1 tile-native RMSNorm+residual: `emit_row_sum`(Nt 타일 elementwise 누적 후 행 reduce, gather 0) +
      `emit_broadcast` col/row tile-native + 2D-64mult elementwise 상수 tile-pack(`alloc_const_tiled`) +
      assign_layouts에 `sum`=TILE 입력 소비자·`broadcast_to`=TILE producer 규칙 + 입력 param tile-pack(host-side 무료)
      + O-proj group `c_tiled` 출력 + `emit_concat` tile-입력 relayout. per-layer 순이득 ~147K(−8%), 비용 ~1% FP16.
    - 5d-2 tile-native attention core: RoPE·softmax·transpose tile화(gather 131K, 최대). head_dim=64=1 타일폭,
      RoPE 타일 내부 반쪽 slice, softmax reduce도 tolerance-only.

**검증**: 5c까지 **출력 완전일치(byte-exact)** 확인 완료. 이후 최적화는 여기서 종료. golden 오라클 = direct 백엔드(row-major) 유지.

## 4. 리스크 관리 (중간 결과물 방지)
- 매 Stage: 착수 전 브랜치, 완료 시 **전체 테스트 + byte-exact** 통과해야 커밋.
- 값-불변 Stage(1,2,4)는 **출력 완전일치** 자동 비교. 값-변경 Stage(3,5)는 tolerance + 참조 비교.
- 각 Stage 독립 커밋 → 언제 멈춰도 직전 Stage는 **완결·동작** 상태.
