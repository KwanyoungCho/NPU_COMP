# NPU 컴파일러 v2 — TVM-native 재구성 (진행 보고서, 2026-07-26~)

> **이 문서 하나로** 프로젝트 배경 · 아키텍처 · 구현 · 측정 결과 · 현재 상태 · 다음 단계를 전부 파악할 수 있게 작성했습니다.
> **처음 보는 분**은 [§요약](#요약-tldr) → [§0 배경·핵심 개념](#0-배경--핵심-개념-용어집) → [§2 아키텍처](#2-아키텍처--컴파일러-pass-파이프라인) → [§3 구현](#3-구현한-최적화-무엇을-왜-결과) → [§4 결과](#4-측정-결과-성능정확성) 순으로 읽으면 됩니다.
> §6 이하는 상세 추적 로그(이슈·결정·작업·검증·커밋)입니다.
> 설계 근거: [`d_compiler/COMPILER_V2_PLAN.md`](../d_compiler/COMPILER_V2_PLAN.md) · v1(이전 세대) 결과: [`report/report_0719.md`](report_0719.md)

---

## 요약 (TL;DR)

**무엇을 만드나** — 커스텀 **NPU(하드웨어 c-model)** 위에서 **Llama 3.2 3B**를 실행하기 위한 **컴파일러**. 트랜스포머 레이어(Relax 그래프)를 받아 NPU의 명령어(ISA) 스트림으로 낮춘다(lowering).

**왜 v2인가** — 1세대(v1)는 동작하지만 "연산마다 손으로 짠" 구조라, 앞으로 **C-model의 cycle(지연) 정보가 들어오면 얹으려는 auto-scheduling**을 하기 어려웠다. v2는 각 최적화를 **명시적 컴파일러 pass**로 재구성해 확장·자동화가 가능한 **TVM 표준 파이프라인**으로 옮긴다.

**결과** (전부 `compiler-v2` 브랜치에 커밋·검증):
- v1의 모든 최적화를 이식(**메모리 재사용 · tile 레이아웃 · weight packing · fusion**)하고 더 나아감
- **실제 Llama-3B 레이어에서 relayout 오버헤드(gather) 명령 2.16M → 98K (−95%)**, 총 명령 3.5M → 1.29M (v1 최적치 1.06M에 근접)
- **컴파일 자체가 v1보다 1.5~2배 빨라짐**
- adversarial 리뷰 3회(누적 500+ config 실행 대조) **silent(조용한) 오답 0건**, v1 소스는 손대지 않음(정답 오라클로 유지)

**현재 상태** — Stage 1~3 + Stage 4의 Phase 4.0~4.3 완료. 남은 **Phase 4.4~4.5**는 "수제 codegen을 TVM 표준(`layout_transform` + LegalizeOps)으로 완전 대체"하는 **구조 개선**이며, 성능/정확성 이득은 이미 확보돼 있다. → [§5](#5-현재-상태--다음-단계-phase-4445)

---

## 0. 배경 & 핵심 개념 (용어집)

처음 보는 분을 위해 이 문서에서 계속 나오는 용어를 먼저 정리한다.

| 용어 | 뜻 |
|---|---|
| **NPU c-model** | 우리가 타깃하는 가속기의 소프트웨어 모델. 명령어(ISA)를 받아 실행하는 시뮬레이터(**mysim**)로 검증한다. |
| **ISA** | NPU의 명령어 집합. fp16 기반. 핵심: 64×64 **matmul MAC**(0x42), **strided load**(0x90, 열-우선 읽기로 64×64 transpose/gather 가능), **native reduce-sum**(0x14), **copy**(0x17), **broadcast**(0x15). 2026-07-10 업데이트로 상당수 연산이 네이티브 지원됨. |
| **mysim** | NPU c-model 시뮬레이터. 컴파일러가 낸 ISA "word" 스트림을 실행해 G-buffer(아래) 결과를 낸다. 정확성 검증의 실행 엔진. |
| **G-buffer** | NPU의 단일 평면 메모리. 모든 텐서가 컴파일 시점에 정해진 **offset**에 놓인다(동적 할당 없음). |
| **64×64 tile** | NPU matmul의 네이티브 단위. 모든 행렬 연산은 64×64 블록으로 쪼개져 처리된다. |
| **tile-blocked (packed) 레이아웃** | `[R,N]` 텐서를 `[⌈R/64⌉, ⌈N/64⌉, 64, 64]`로 저장. 각 64×64 tile이 **연속 메모리**라 matmul이 바로 읽는다(**gather 불필요**). 대비되는 게 일반 **row-major** 저장. |
| **gather / scatter** | row-major로 저장된 텐서에서 64×64 tile을 꺼내려면 **strided 복사**(gather)가 필요하고, 결과를 다시 흩어 쓰는 게 scatter. **순수 오버헤드** — tile-blocked면 사라진다. |
| **Relax / TIR** | TVM의 IR. **Relax**=그래프 레벨(op 단위), **TIR**=루프 레벨(연산 내부). `LegalizeOps`가 Relax op을 TIR로 낮춘다. |
| **tensorize** | TIR의 특정 블록을 하드웨어 intrinsic(예: NPU 64×64 MAC = `npu_gemm_acc`)으로 치환하는 스케줄 연산. |
| **build_layer_module** | 트랜스포머 레이어 1개(RMSNorm→attention→FFN)를 Relax 그래프로 만드는 테스트 하니스(`npu_compiler/model.py`). |
| **ref_layer** | 같은 레이어의 numpy(f64) 참조 구현. 정확성의 **수치 정답**. |
| **오라클 주도** | v1 백엔드와 numpy ref를 "정답"으로 고정하고, v2를 그에 대조해 만든다. **v1 소스는 절대 수정하지 않는다**(회귀 시 즉시 드러나도록). |
| **gate** | `run_gate.sh` — 전체 테스트(15개) + 벤더 예제 byte-exact 대조. 매 커밋 통과(GREEN) 유지. |
| **rel** | 상대 오차 `max|out-exp| / max|exp|`. fp16이라 rel < 0.05면 정확으로 본다. |

**대상 모델 (실 Llama 3.2 3B, prefill 1레이어)**: SEQ=128, D=3072, H=24 heads, KV=8(GQA), HD=128, F(FFN)=8192. — 이 크기가 최종 타깃이며, 성능은 이 레이어 기준으로 측정한다.

---

## 1. 왜 v2인가 (문제 · 목표 · 방법론)

**문제**: v1은 `codegen.py`가 op마다 손으로 ISA를 찍어내는 ~1000줄짜리 백엔드였다. 동작하고 최적화(메모리 재사용·tile 레이아웃 등)도 있었지만, 그 최적화들이 **한 함수에 뒤엉켜** 있어서:
- 새 최적화나 auto-scheduling을 얹기 어렵고,
- 레이아웃·메모리·fusion이 "표준 컴파일러 pass"가 아니라 특수 케이스 코드였다.

**목표 (Path A)**: TVM의 정석 가속기 컴파일 경로에 올라탄다 — **op=TIR intrinsic(tensorize) + 제약된 schedule space + 커스텀 TIR→ISA codegen + (미래) cost model**. 최종적으로 C-model cycle 기반 **auto-scheduling**을 얹는다.

**방법론** (이 문서 전체에 관통):
1. **오라클 주도** — v1 + numpy ref를 정답으로. v1 소스 무변경. 빅뱅 금지.
2. **매 단계 gate GREEN** — 각 커밋이 전체 테스트 + 벤더 byte-exact 통과.
3. **멀티 에이전트 검증** — 큰 조사/구현/adversarial 리뷰는 병렬 에이전트 워크플로우로, 결과는 이 문서에 집계.

---

## 2. 아키텍처 — 컴파일러 pass 파이프라인

### 2.1 큰 그림 (before → after)

v1(및 리팩토링 전 v2 초안)은 **모놀리식 dispatcher**였다. v2 리팩토링의 핵심은 이를 **명시적 5-pass 파이프라인**으로 분해한 것이다:

```
 [입력] build_layer_module → 고수준 Relax 그래프
          │
          ▼  relax.transform.LegalizeOps()   (Relax op → call_tir)
          │
  ┌───────┴─────────────────────────────────────────────────────────┐
  │  compile_module(mod, reuse=True, tile=True, fuse=True)           │
  │                                                                   │
  │  _parse            (F5)  legalized Relax → op 리스트 + 텐서표      │
  │  _assign_layouts   (F4)  텐서별 row/tile 결정(fixpoint) + weight   │
  │  _detect_oproj_groups(F3) per-head o-proj add-tree → fusion 그룹  │
  │  _plan_memory      (M1)  liveness + free-list → G-buffer offset    │
  │  _emit             (C1)  각 op → NPU ISA (walker / packed matmul)  │
  └───────┬───────────────────────────────────────────────────────────┘
          ▼
   (asm, off, shp, top, out_name, const_inits, tiled_feed, layout)
          │
          ▼  mysim 실행 → 결과 (host가 tile 출력은 unpack)
```

**핵심 자산 2개** (NPU 고유, 나머지는 TVM 표준으로 흡수):
- **intrinsic 정의** — `npu_gemm_acc`/`npu_fill_zero` (64×64 MAC / zero-fill TensorIntrin, `tir_backend.py`).
- **walker (`V2Walker`)** — tensorized TIR을 걸어 ISA를 emit. v1의 `_Walker`를 상속·일반화.

### 2.2 각 pass가 하는 일

| Pass | 함수 | 하는 일 | 담은 최적화 |
|---|---|---|---|
| **F5** parse | `_parse` | LegalizeOps된 Relax를 읽어 **내부 op 리스트**(opname·입력·출력·shape) + 상수(위치 키잉)로 변환 | — |
| **F4** layout | `_assign_layouts` | 텐서별 **row/tile 레이아웃을 fixpoint로 결정**("tile 텐서는 tile 소비자만 먹인다" 불변식으로 경계 relayout 최소화) + 64-mult matmul weight를 packing 대상으로 표시 | **A4 tile-blocked, weight packing** |
| **F3** fusion | `_detect_oproj_groups` | `attn = Σ_h ctx_h@Wo_h` 같은 **per-head matmul add-tree**를 찾아 1개 그룹으로 | **F3 O-proj fusion** |
| **M1** memory | `_plan_memory` | **liveness**(각 텐서의 마지막 read) + **exact-size free-list**로 G-buffer offset 배정. 죽은 슬롯을 같은 크기 후속 텐서에 재사용. fusion-aware(folded 노드 미할당) | **A1 메모리 재사용** |
| **C1** codegen | `_emit` | 각 op을 NPU ISA로 lower. matmul=`packed_matmul`(tile) 또는 `emit_matmul_into`(mixed); ew/reduce/broadcast=walker 마커 | **A2 fast-path, packed matmul** |

### 2.3 tile-blocked 레이아웃이 핵심인 이유

NPU matmul은 64×64 tile 단위다. 텐서가 **row-major**면 64×64 tile을 꺼낼 때마다 **strided 복사(gather)** 가 필요하다. **tile-blocked**로 저장하면 tile이 연속이라 바로 읽는다(gather=0). 그래서 A4(레이아웃)는 성능의 핵심이고, Stage 4는 이 tile 레이아웃을 "TVM `layout_transform` op"으로 표현하는 정석화 작업이다.

> **parity 통찰**: 경계에서의 relayout 1회 = gather 1회. 이득은 **내부 tile 체인이 gather를 아예 안 하는 것**에서 온다. 그래서 "residual 스트림을 끝까지 tile로 유지"(Phase 4.1)하면 projection들이 activation을 gather하지 않게 되어 큰 이득이 난다.

---

## 3. 구현한 최적화 (무엇을·왜·결과)

각 최적화는 커밋되어 있고 gate GREEN이며, v1 소스는 무변경이다.

### A1 — liveness 기반 메모리 재사용 (`_plan_memory`)
- **무엇**: 각 텐서가 그래프에서 **마지막으로 읽히는 지점**을 계산해, 그 이후 G-buffer 슬롯을 반납하고 **같은 크기의 후속 텐서에 재사용**(exact-size free-list). params/constants/output은 상주.
- **왜**: NPU엔 동적 할당이 없어 모든 텐서가 고정 offset. 재사용 안 하면 activation 메모리가 크게 낭비.
- **결과**: activation 영역 **−47~63%**(config별). 데이터 위치만 바뀌므로 재사용 on/off가 **비트 동일**(5/5 config).

### A4 — tile-blocked 레이아웃 + weight packing (`_assign_layouts` + `_emit`)
- **무엇**: 텐서를 tile-blocked `[Rt,Nt,64,64]`로 저장하도록 fixpoint로 결정. matmul은 tile 입력을 바로 읽고(gather 스킵), weight는 host에서 미리 packing.
- **왜**: gather/scatter 오버헤드 제거 + 8-bit ISA 필드 한계(≥256) 우회.
- **결과**: 멀티-타일 config에서 명령어 **−42~52%**; **SEQ≥256 · HD=128(실 Llama head dim) 해금**(row 경로는 8-bit 필드 wrap으로 크래시). tile==row 수치 일치.

### F3 — O-proj accumulate-group fusion (`_detect_oproj_groups`)
- **무엇**: `attn = Σ_h (ctx_h @ Wo_h)` (H개 head-projection + add-tree)을 **1개 in-place accumulate 그룹**으로 융합. H번의 scatter가 1번으로.
- **왜**: 실 Llama는 H=24 → matmul 24 + add 23개를 따로 내면 낭비.
- **결과**: 명령어 −3%(H2)~−19%(H8), 실 H=24는 더 큼. fuse rel==nofuse. (Phase 4.1로 C가 tile이 된 뒤엔 scatter가 이미 0이라 이득이 shared-accumulator로 이동.)

### A2 — 컴파일 속도 (마커 lru_cache)
- **무엇**: op을 walker에 넘기려고 만드는 TIR 마커 PrimFunc을 op마다 **TVMScript로 재파싱**하던 게 `_emit` 시간의 절반이었음(프로파일). 마커 빌더 9개에 `lru_cache`.
- **결과**: **v2 컴파일이 v1보다 1.5~2배 빨라짐**(MEDIUM 108→49ms, HD128 619→331ms).

### Stage 4 — layout_transform-native 정석화 (진행 중, 4.0~4.3 완료)
- **4.0/4.2a**: packed matmul을 `einsum` nest로 표현해 기존 `npu_gemm_acc`로 tensorize → **v1과 byte-exact**. 실 경로 투입.
- **4.1**: **fixpoint 한 줄 수정**(`v2_backend.py:706`)으로 64-mult 출력을 tile로 유지 → **residual/RMSNorm/FFN 스트림이 tile end-to-end** → 모든 projection이 activation을 gather 안 함. **실 3B gather 2.16M→98K (−95%)**.
- **4.3**: 고수준 Relax를 packed 4D로 rewrite하는 **F4 pre-legalize 패스**(`npu_compiler/packed.py`). relax-VM에서 row 그래프와 **bit-identical** 검증.
- **남은 4.4~4.5**: §5 참조.

---

## 4. 측정 결과 (성능·정확성)

### 4.1 실 Llama-3B prefill 레이어 — 명령 수(role별)

`report/figs/0719/measurements_detail.json`(v1) + v2 실측 대조. gather/scatter가 순수 relayout 오버헤드.

| | total 명령 | gather | scatter |
|---|---|---|---|
| **v1 before** (row-major, 최적화 전) | 2,235,194 | 409,600 | 540,672 |
| **v1 after** (v1의 A4 완전 적용, 목표치) | **1,057,758** | **0** | **0** |
| **v2 (Phase 4.1)** | **1,287,093** | **98,304** | 65,536 |

→ v2가 v1 "after"에 **근접**(1.29M vs 1.06M). 남은 gather 98K = RoPE의 Q/K가 아직 row인 부분(Phase 4.4에서 0으로).

### 4.2 메모리 (activation 영역, A1)
config별 −47~63% (REDUCED 47.9 · MEDIUM 63.2 · GQA 50.0 · wide 52.8 · HD32 47.1%). reuse on/off **비트 동일**.

### 4.3 명령 수 감소 (A4 tile, 멀티-타일)
SEQ128 D128: tile 33,610w vs row 100,844w (**−66%**) · HD128: 58,159w vs 225,625w · SEQ256(≥256 해금): row는 크래시.

### 4.4 컴파일 속도 (마커 캐시 후, v1 hybrid 대비)
| config | v1 | v2 |
|---|---|---|
| MEDIUM | 108ms | **49ms** |
| SEQ128 H2 | 202ms | **101ms** |
| SEQ256 H8 | 1163ms | **754ms** |
| HD128 H4 | 619ms | **331ms** |

### 4.5 정확성 & adversarial 검증
- 완전한 레이어: **5-config 전부 rel < 0.05** vs ref_layer (REDUCED 0.0011 · MEDIUM 0.0012 · GQA 0.0009 · wide 0.0043 · HD32 0.0024).
- adversarial 리뷰 **3회** (누적 500+ config 실행 대조): **silent 오답 0건**. 발견된 이슈는 전부 loud-fail 또는 수정(§7 이슈 트래커).
- F4 packed 패스: relax-VM에서 row 그래프와 **bit-identical**(6 config).

---

## 5. 현재 상태 & 다음 단계 (Phase 4.4–4.5)

### 지금 프로덕션에서 되는 것 (전부 커밋·gate GREEN)
- **파이프라인**: `_parse`(F5) → `_assign_layouts`(F4) → `_detect_oproj_groups`(F3) → `_plan_memory`(M1) → `_emit`(C1). 토글 `compile_module(mod, reuse=True, tile=True, fuse=True)` — 각각 A1 메모리 / A4 tile / F3 fusion을 켜고 끔(기존 row 레이아웃은 `tile=False`).
- **이식된 v1 최적화**: A1 · A4 · weight packing · F3 fusion · A2 fast-path. (T1 cross-matmul gather-cache는 v1도 reuse 경로에선 끄므로 실질 parity.)
- **성능**: §4 — 실 3B gather −95%, 컴파일 v1보다 빠름.

### Stage 4 세부 (4.0–4.3 완료 / 4.4–4.5 남음)
- ✅ **4.0** packed matmul(einsum→`npu_gemm_acc`) byte-exact + `_bind_match` 4D 수정 — `v2_backend.py`
- ✅ **4.1** residual tile(fixpoint 1줄, `v2_backend.py:706`) — 실 3B gather −95%
- ✅ **4.2a** fully-tile matmul을 `emit_packed_matmul`로(byte-exact)
- ✅ **4.3** F4 pre-legalize packed 패스 `npu_compiler/packed.py:_build_packed` — relax-VM에서 row와 **bit-identical**(6 config)
- ⏳ **4.4–4.5 (다음 세션)** — 아래

### 다음 세션 착수점 (Phase 4.4–4.5)
**목표**: `packed.py`의 F4 패스를 **실제 NPU-ISA codegen에 통합** → 수제 tile emitter 삭제 + 균일 codegen + 마지막 gather=0.

**정확한 갭(측정됨)**: `_build_packed`→LegalizeOps가 내는 op을 현 `_emit`이 못 lower함. 새로 처리할 op: `einsum`(→ 이미 있는 `emit_packed_matmul` 재사용, byte-exact), `te_layout_transform`(→ device reindex copy 또는 host 경계), `reshape`(→ no-op/copy), 그리고 4D shape의 `ew`/`sum`/`max`/`broadcast_to`.

**해야 할 일**:
1. **S1 tensorize 디스패처** — legalized packed PrimFunc별 tensorize: einsum→`npu_gemm_acc`(있음), ew loop→`npu_vadd`류(`schedule_ew_binary` 확장), reduce fold→새 schedule, layout_transform copy→`npu_copy64`(spike `s4_B2_reindex_copy.py`).
2. **packed-aware `_emit`/walker** — 위 tensorized PrimFunc을 walk. `_bind_match` 4D는 이미 됨.
3. **`compile_module`에 `packed=True` 경로** 신설(현 경로는 오라클로 유지) → 5-config mysim vs 현 tile 경로 대조(fp16 tolerance).
4. **Phase 4.4**: RoPE도 packed(slice/concat/transpose tile) → 마지막 98K gather=0. (transpose는 ISA 네이티브 지원 + v1에 이식 소스 있음.)
5. **Phase 4.5**: 수제 emitter 5개(`_emit_bcast_col_tile`/`_emit_bcast_row_tile`/`_emit_rsum_tile`/`_emit_rmax_tile`) + `_emit`의 op별 row/tile 분기 삭제 → codegen 단일 루프.

**자산 위치**: 검증된 spike = `scratchpad/s4_*.py`(다음 세션엔 없을 수 있음 — `packed.py`/이 문서에서 재생성), F4 패스 = `npu_compiler/packed.py`, 워크플로우 결과 = task `w5crt9bx4`(설계)·`w3xsm7u0y`(F4 빌드).

**주의**: correctness/gather 이득은 이미 확보(4.1). 4.4–4.5는 **구조(균일 codegen + auto-sched 기반)** 이득. 오라클-주도로 각 단계 gate GREEN.

---

## 6. Phase 상태 보드

| Phase | 내용 | 상태 |
|---|---|---|
| **0** | 타당성 spike (tensorize/memory/codegen 검증) | 🟢 **GO** (3/3 probe) |
| **1** | path-무관 안전 정리 | ⚪ 선택적 (ROI 낮음) |
| **2** | NPU ISA→TIR intrinsic + walker 일반화 | 🟢 완료 (전 op walker로) |
| **3** | v2.compile() 완전한 레이어 컴파일 | 🟢 완료 (rel=0.0011) |
| **3-R** | **리팩토링 → 명시적 pass + v1 최적화 이식** | 🟡 **거의 완료** (Stage 1~3 + Stage 4 4.0~4.3, 남은 4.4~4.5) |
| **4** | cost 기반 auto-scheduling | ⚪ 유보 (cycle 정보 도착 후) |

범례: ⚪대기 🟡진행 🟢완료 🔴블록

---

## 7. 이슈 트래커

| ID | 상태 | 심각도 | 제목 | 관련 |
|---|---|---|---|---|
| V2-001 | open | 설계 | fill-vs-MAC 비트(ko==0) tensorize 표현 불가 — 어디서 lower할지(TIR 패스 vs walker) | Phase 2 |
| V2-002 | open | 설계 | init 블록(C=0) — 별도 fill intrinsic vs 첫 MAC에 fold | Phase 2 |
| V2-003 | open | 중 | `_bind_match` cache_read/write scratch 스코프 지원 | Phase 2 |
| V2-004 | open | 중 | canonical-schedule detector — 고정 64³은 fast-path 유지 | Phase 2 |
| V2-005 | open | 설계 | 게이트 2-tier(canonical=byte-exact / auto-sched=tolerance) | Phase 2 |
| V2-006 | open | 중 | tile-blocked footprint vs TVM logical-size 정합 | Phase 3 |
| V2-007 | open | 소 | StaticPlanBlockMemory offset post-pass(N storage→flat) | Phase 3 |
| V2-008 | open | 소 | decode 동적 shape는 `tir_var_upper_bound` 필요 | Phase 3 |
| V2-009 | open | 중 | packed op이 효율적 native fold로 lower되나 (Stage 4가 검증 중) | Phase 3 |
| V2-010 | **resolved** | 높 | reduce-max ≥256 8-bit 필드 wrap → guard 복원(loud); 진짜 지원=A4 tile | `_emit_rmax_row` |
| V2-011 | resolved | 중 | transpose C≥256 wrap → guard 복원 | `_emit_transpose_row` |
| V2-012 | resolved | 높 | EW broadcast operand 초과 read → 크기 assert | ew dispatch |
| V2-013 | resolved | 높 | non-last-axis strided_slice OOB → last-axis assert | slice dispatch |
| V2-014 | resolved | 중 | concatenate axis 무시 → last-axis assert | concat dispatch |
| V2-015 | resolved | 중 | broadcast_to `[1,1]` 오분류 → valid assert | broadcast dispatch |
| V2-016 | resolved | 소 | sum/max axis/rank 미검사 → last-axis 2D assert | reduce dispatch |
| V2-017 | open | 소 | strided_slice stride≠1 (현 경로 미사용) | later |
| V2-018 | **resolved** | 높 | **tile ew 2D-64-mult 상수 미packing** → 멀티-타일(≥128) 오답. fixpoint 후 tile 마킹→pack_tiled | `_assign_layouts` |
| V2-019 | **resolved** | 높 | **HD≥128 tile 크래시** — RoPE half-slice(64-wide)가 tile로 seed되나 `_emit` row-only. HD=128=실 Llama head dim. 수정: transpose/slice/concat을 tile seeding에서 제외(RoPE 항상 row) | `_assign_layouts` |
| V2-020 | **resolved** | 중(perf) | **O-proj fusion 미구현**(H>1 적용, 실 H=24). parity 감사가 앞선 "N/A" 오판 정정 → Stage 3에서 구현 | `_detect_oproj_groups` |

> V2-010~017은 **adversarial 리뷰(workflow 15 agents, 8 distinct 버그)** 발견 — 공통 원인: v2가 v1의 guard/tile-fallback을 떨어뜨려 범위 밖 값이 silent 오답. **전부 guard 복원으로 loud-fail 처리**.
> 규칙: 새 이슈 `V2-NNN`, 상태 ∈ {open, in-progress, resolved, wontfix}, resolve 시 커밋 해시 기록.

---

## 8. 결정 로그 (Decisions)

| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-07-26 | 최종 목표 = **Path A (TVM MetaSchedule)** | 정석 가속기 경로 |
| 2026-07-26 | 브랜치 작업 후 검증 완료 시 merge | 안전 |
| 2026-07-26 | **Phase 0 = GO** | 3/3 probe: matmul tensorize · TVM memory · codegen 일반화 실측 |
| 2026-07-27 | **codegen 교정 — v1 emitter 통째로 안 옮김** | ISA가 대부분 네이티브 지원; v1 복잡함은 ISA가 아니라 **레이아웃 산물** → v2 = 얇은 매핑 + 레이아웃은 `layout_transform` 패스 |
| 2026-07-27 | **레이아웃 = `layout_transform`로 확정**, fork는 (ii) tile-native | spike: tile-blocked가 표준 op로 표현·legalize됨 |
| 2026-08-03 | **리팩토링 착수** (자기검토) | compile_module이 계획 pass가 아니라 모놀리식 dispatcher로 흐름 — 사용자 지적 타당 |
| 2026-08-03 | **Stage 4 = layout_transform-native로 진행** | 실 3B gather 갭(2.16M) 측정 → A4-5d "minor 보류" 오판 정정; feasibility spike GO |
| 2026-08-03 | **Phase 4.1 먼저(이득 확보) → 4.2~4.5 점진** | gather 닫는 데 전체 재아키텍처 불필요; 4.4~4.5는 구조 이득 |

---

## 9. 멀티 에이전트 작업 로그 (연대기)

| 날짜 | 작업 | 결과 요약 |
|---|---|---|
| 2026-07-26 | 환경 API 서베이 | TVM 0.19.0, 필수 API 전부 존재 |
| 2026-07-26 | Probe A/B/C (Phase 0) | GO — matmul tensorize 성립 · StaticPlanBlockMemory 동작 · walker 재사용 ~90% |
| 2026-07-26 | v1 코드 확인 | v1은 이미 tensorize matmul 컴파일러(`npu_gemm_acc`) — 남은 건 non-matmul 일반화 |
| 2026-07-26~27 | Phase 2 — elementwise/reduce → 통합 walker | v1과 ISA byte-exact + mysim |
| 2026-07-27 | **v2가 완전한 레이어 컴파일** | build_layer_module 전체 → mysim, ref_layer 대비 rel=0.0011 |
| 2026-07-27 | REAL 파이프라인 e2e | Relax→LegalizeOps→tensorize→walker→mysim (`_bind_match` N-D 일반화) |
| 2026-07-27 | 레이아웃 spike | `layout_transform`가 tile-blocked를 표준 copy로 legalize 확정 |
| 2026-08-03 | **자기검토** | compile_module이 모놀리식 dispatcher → 리팩토링 착수 |
| 2026-08-03 | **Stage 1 — 파이프라인 분해 + A1** | 3-pass 분해 + liveness 재사용, act −47~63%, reuse⟺bump 비트 동일 5/5 |
| 2026-08-03 | A4 tile 매핑 (workflow 4 agents) | v1 tile-blocked 계약 exhaustive 매핑 (w4dr2kft5) |
| 2026-08-03 | **Stage 2a — A4 tile-blocked** | 멀티-타일 −42~46%, ≥256·HD128 해금, tile==row 수치 일치 |
| 2026-08-03 | Stage 2a adversarial (workflow 12 agents) | 300+ config, silent 오답 0, V2-019 발견·수정 (wins8xf1c) |
| 2026-08-03 | parity 감사 (workflow 3 agents) | 40항목: 30 present·4 N/A·4 deferred·1 missing=O-proj fusion (w1xlpuxhf) |
| 2026-08-03 | **Stage 3 — O-proj fusion** | H matmul→1 group, fusion-aware liveness, −3~19% |
| 2026-08-03 | Stage 3 adversarial (workflow 5 finder) | ~240 config, correctness 버그 0 (w90kh3erx) |
| 2026-08-03 | 마커 캐싱 | 재파싱 제거 → v2 컴파일이 v1보다 1.5~2× 빠름 |
| 2026-08-03 | 실 3B overhead 측정 | v2 tile이 v1 after 못 미침(gather 2.16M 잔존) → A4-5d 갭 발견 → Stage 4 |
| 2026-08-03 | Stage 4 feasibility (workflow 5 agents) | GO — packed matmul tensorize + layout_transform-native 가능 (w5crt9bx4) |
| 2026-08-03 | **Stage 4 4.0/4.1/4.2a/4.3** | packed matmul byte-exact · residual tile(gather −95%) · F4 패스 relax-VM 검증 (w3xsm7u0y) |

---

## 10. 검증 로그 (Verification)

| 대상 | 기준 | 결과 |
|---|---|---|
| v2 op별 (ew/reduce/primitive) | v1 ISA byte-exact + mysim | ✅ 통과 |
| 완전한 레이어 (5-config) | ref_layer 대비 rel<0.05 | ✅ REDUCED 0.0011·MEDIUM 0.0012·GQA 0.0009·wide 0.0043·HD32 0.0024 |
| adversarial 코드리뷰 (15 agents) | correctness 버그 | 8 distinct 발견 → guard 복원 |
| Stage 1 A1 재사용 | reuse⟺bump 비트 동일 | ✅ 5/5, act −47~63% |
| Stage 2a A4 (멀티-타일 + ≥256) | tile rel<0.05, tile<row | ✅ SEQ128 −66%, HD128 해금, SEQ256 해금 |
| Stage 2a adversarial (12 agents, 300+ config) | 조합/레이아웃 miscompile | ✅ silent 0, V2-019 5건 CONFIRMED→수정 |
| parity 감사 (3 agents, 40항목) | v1 최적화 ↔ v2 | ✅ 1 missing(O-proj)→Stage 3 |
| Stage 3 O-proj fusion | fuse==nofuse, 명령↓ | ✅ H1~H8 rel<0.05, −3~19% |
| Stage 3 adversarial (5 finder, 240 config) | fusion liveness/detection | ✅ correctness 버그 0 |
| Stage 4 Phase 4.1 residual tile | gather↓, rel<0.05 | ✅ 실 3B gather −95%, rel 0.0047~0.028 |
| Stage 4 packed matmul | byte-exact vs v1 | ✅ 128/192/256 identical |
| Stage 4 F4 패스 (2 agents, 6 config) | packed vs row VM | ✅ bit-identical, CONFIRMED_EQUIVALENT |

---

## 11. 커밋 로그 (compiler-v2 브랜치)

| 커밋 | 내용 |
|---|---|
| `8e75fe3` | v2 setup — COMPILER_V2_PLAN.md + report_0726.md |
| `63f2fa2` | Phase 0 GO — 3 probe, TVM 0.19.0 |
| `12c349c`~`71f1098` | Phase 2 — elementwise/reduce → 통합 walker (byte-exact) |
| `dc154fa` | course-correct — thin native-ISA 매핑 |
| `10c5942` | 레이아웃 spike — tile-blocked = `layout_transform` 확정 |
| `cea3547` | REAL 파이프라인 e2e + `_bind_match` N-D |
| `82340b9`~`5ef22ca` | Phase 3 — compile_module e2e → FULL LAYER → 5-config → correctness hardening |
| `38fc9d6` | **Stage 1** — 3-pass 분해 + **A1 메모리 재사용** (act −47~63%) |
| `7e0a94a` | **Stage 2a** — F4 layout + **A4 tile-blocked** (−42~52%, ≥256 해금) |
| `f4427b4` | Stage 2a fix **V2-019** (HD≥128 tile 크래시) |
| `50f32fb` | **Stage 3** — F3 **O-proj fusion** |
| `b5db48e` | 마커 lru_cache → v2가 v1보다 1.5~2× 빠름 |
| `b8a6f46` | report — O-proj fusion adversarial |
| `788ce34` | **Stage 4 Phase 4.1** — residual tile → 실 3B gather −95% |
| `aa1be2d` | **Stage 4 Phase 4.0** — packed matmul foundation (byte-exact) |
| `dc9ca2b` | **Stage 4 Phase 4.2a** — fully-tile matmul을 packed nest로 |
| `987ba95` | **Stage 4 Phase 4.3** — F4 pre-legalize packed 패스 (`packed.py`) |
| `ff18cae` | report 정리 + Phase 4.4-4.5 재개 가이드 |

---

## 부록 A. 환경

| 항목 | 값 |
|---|---|
| TVM | 0.19.0 (`/home/chokwans99/tvm-src`) |
| python | `/home/chokwans99/anaconda3/envs/npu-tvm/bin/python` (conda `npu-tvm`) |
| PYTHONPATH | `/home/chokwans99/NPU_cmodel/d_compiler` |
| 필수 API | ✅ `relax.frontend.torch`·`LegalizeOps`·`FuseOps`·`StaticPlanBlockMemory`·`tir.TensorIntrin`·`meta_schedule`·`relax.op.layout_transform` 전부 존재 |
| gate | `bash d_compiler/run_gate.sh` (테스트 15 + 벤더 byte-exact) |

## 부록 B. 주요 파일

| 파일 | 역할 |
|---|---|
| `d_compiler/npu_compiler/v2_backend.py` | **v2 백엔드** — 5-pass `compile_module` + `V2Walker` + packed matmul |
| `d_compiler/npu_compiler/packed.py` | **F4 pre-legalize 패스** (`_build_packed`, Stage 4.3) |
| `d_compiler/npu_compiler/tir_backend.py` | v1 matmul 컴파일러 (`npu_gemm_acc` intrinsic, `_Walker`, `emit_matmul_into`) — v2가 상속·재사용 |
| `d_compiler/npu_compiler/model.py` | `build_layer_module`(테스트 레이어) + `ref_layer`(numpy 정답) |
| `d_compiler/npu_compiler/memplan.py` | v1 메모리 플래너 (A1/A4 원본, tile pack 유틸) |
| `d_compiler/tests/test_v2.py` | v2 테스트 (gate 편입) |
