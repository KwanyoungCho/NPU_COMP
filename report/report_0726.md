# NPU 컴파일러 v2 — TVM-native 재구성 (진행 보고서, 2026-07-26~)

> **이 문서 하나로** 프로젝트 배경 · 전체 아키텍처(두 경로) · **모듈별 상세** · 구현한 최적화 · 측정 결과 · 현재 상태 · 다음 단계를 전부 파악할 수 있게 작성했다.
> **처음 보는 분**은 [§요약](#요약-tldr) → [§0 용어](#0-배경--핵심-개념-용어집) → [§2 아키텍처(두 경로)](#2-아키텍처--두-경로-packed-기본--op-list-oracle) → [§3 모듈 레퍼런스](#3-모듈-레퍼런스-세부-구조) → [§4 최적화](#4-구현한-최적화) → [§5 결과](#5-측정-결과) 순으로 읽으면 된다.
> §6 이하는 상태·이슈·결정·검증·커밋 추적 로그다.

---

## 요약 (TL;DR)

**무엇을 만드나** — 커스텀 **NPU(하드웨어 c-model)** 위에서 **Llama 3.2 3B**를 실행하기 위한 **컴파일러**. 트랜스포머 레이어(TVM Relax 그래프)를 받아 NPU 명령어(ISA) 스트림으로 낮춘다(lowering).

**왜 v2인가** — 1세대(v1)는 동작하지만 "연산마다 손으로 짠" 모놀리식 백엔드라 확장·auto-scheduling이 어려웠다. v2는 각 최적화를 **명시적 컴파일러 pass**로 재구성하고, 더 나아가 레이아웃을 **TVM 표준 IR 변환(`layout_transform` + LegalizeOps)**으로 표현해 **임의 모델에 유연한 범용 경로**를 만든다.

**지금 상태 (2026-08-03, 전부 `compiler-v2` 브랜치 커밋 · gate GREEN · byte-exact)**:
- **두 경로 체계**로 정착:
  - **`compile_packed`** = **기본(default) 범용 경로**. Relax를 tile-blocked 4D로 rewrite하는 **진짜 IR→IR 패스**(`packed.py`) 위에서 돈다. model-agnostic(모르는 op는 row island로 흐름).
  - **`compile_module`** = **op-list 경로**. **수치 오라클** + sub-tile head-dim **fallback**로 유지(은퇴 아님).
- **실 Llama head_dim(HD=128)에서 packed가 손튜닝 op-list의 명령 수 절반**(95,013 → **52,453**, 0.55×). 모든 config에서 **gather=0 / scatter=0**.
- **v1 최적화 전부 packed에 반영**(9-에이전트 코드 검증): A1 메모리 재사용 · A4 tile 레이아웃 · weight/const host-packing · tile-RoPE · stable softmax · SiLU native · F3 O-proj fusion(opt-in). 놓친 것 없음.
- adversarial 리뷰·parity 감사(누적 수백 config 대조) **silent 오답 0건**. v1 소스 무변경(오라클).

**다음** — packed의 **decode/KV-cache 경로 검증**(현재 prefill 레이어만 검증). → [§6](#6-현재-상태--다음-단계)

---

## 0. 배경 & 핵심 개념 (용어집)

| 용어 | 뜻 |
|---|---|
| **NPU c-model / mysim** | 타깃 가속기의 SW 모델(`_poc/mysim.cpp`). ISA "word" 스트림을 실행해 G-buffer 결과를 낸다. 정확성 검증의 실행 엔진. |
| **ISA** | NPU 명령어(fp16). 핵심: 64×64 **matmul MAC**(0x42), **strided load**(0x90, 열-우선 읽기 = 64×64 transpose/gather), **native reduce**(0x14), **copy**(0x17), **broadcast**(0x15), **native SiLU**(m_add act-bit). |
| **G-buffer** | NPU 단일 평면 메모리. 모든 텐서가 컴파일 시점 고정 **offset**에 놓인다(동적 할당 없음). |
| **64×64 tile** | NPU matmul의 네이티브 단위. 모든 행렬 연산은 64×64 블록으로 처리. |
| **tile-blocked (packed) 레이아웃** | `[R,C]`를 `[⌈R/64⌉,⌈C/64⌉,64,64]`로 저장. 각 tile이 **연속 메모리**라 matmul이 바로 읽는다(**gather 불필요**). 대비: 일반 **row-major**. |
| **gather / scatter** | row-major에서 64×64 tile을 꺼내는 strided 복사(gather)·되쓰기(scatter). **순수 오버헤드** — tile-blocked면 사라진다. |
| **Relax / TIR** | TVM의 IR. **Relax**=그래프(op 단위), **TIR**=루프(연산 내부). `LegalizeOps`가 Relax op→TIR `call_tir`로 낮춘다. |
| **tensorize** | TIR 블록을 HW intrinsic(NPU 64×64 MAC=`npu_gemm_acc`)으로 치환하는 스케줄 연산. |
| **op-list 경로 / packed 경로** | v2의 두 경로. **op-list**=레이아웃을 side-channel dict로 들고 커스텀 op 리스트 위에서 pass 실행(`compile_module`). **packed**=레이아웃을 **4D shape + 그래프 내 `layout_transform` op**에 구워넣는 IR→IR(`compile_packed`, **기본**). |
| **build_layer_module / ref_layer** | 트랜스포머 레이어 1개 Relax 그래프(`model.py`) / 같은 레이어의 numpy(f64) **수치 정답**. |
| **오라클 주도** | v1 + numpy ref + `compile_module`을 "정답"으로 고정하고 packed를 대조. **v1 소스 무변경**. |
| **gate / rel** | `run_gate.sh`(테스트 15 + 벤더 byte-exact). / 상대오차 `max|out-exp|/max|exp|`, fp16이라 <0.05면 정확. |

**대상 모델 (실 Llama 3.2 3B, prefill 1레이어)**: SEQ=128, D=3072, H=24, KV=8(GQA), **HD=128**, F=8192. 성능은 이 크기(특히 HD=128) 기준으로 본다.

---

## 1. 왜 v2인가 (문제 · 목표)

**문제**: v1은 `codegen.py`가 op마다 손으로 ISA를 찍는 백엔드였다. 동작·최적화는 있었지만 레이아웃·메모리·fusion이 **한 함수에 뒤엉킨 특수 케이스 코드**라 (a) 새 최적화·auto-scheduling을 얹기 어렵고 (b) Llama 전용 op 집합에 묶여 있었다.

**목표**: TVM 정석 경로(op=TIR intrinsic + schedule + TIR→ISA codegen)로 옮기되, **레이아웃까지 IR 변환(`layout_transform`)으로 표현**해 — Llama 전용이 아니라 **임의 모델이 흐르는 범용 NPU 컴파일러**로. 최종적으로 C-model cycle 기반 **auto-scheduling**을 얹는다.

**방법론** (문서 전체 관통): ① 오라클 주도(v1+ref 무변경) ② 매 커밋 gate GREEN ③ 큰 조사·구현·리뷰는 병렬 멀티-에이전트 워크플로우.

---

## 2. 아키텍처 — 두 경로 (packed 기본 + op-list oracle)

### 2.1 큰 그림 — 하나의 프론트엔드, 두 경로

레이어 Relax 그래프(`build_layer_module`)가 **두 백엔드의 공통 입력**이다. 두 경로 모두 `_parse`(F5)로 시작하고, **차이는 "레이아웃이 어디에 사는가"**다.

```
                    build_layer_module → 고수준 Relax 그래프
                                 │
          ┌──────────────────────┴───────────────────────────┐
          │                                                    │
   [기본] packed 경로                              [오라클/fallback] op-list 경로
   compile_packed(mod)                            compile_module(mod, tile,fuse,reuse)
          │                                                    │
   packed._build_packed  ← Relax IR→IR:            LegalizeOps(mod)
     레이아웃을 4D shape +                                     │
     layout_transform op에 구워넣음                     _parse (F5)
          │                                                    │
   LegalizeOps(pk_mod)                             _assign_layouts (F4)  ← 레이아웃을
          │                                          side-channel dict로
   _parse (F5)  ────────── 공통 ──────────          _detect_oproj_groups (F3)
          │                                                    │
   _plan_memory_packed (M1, A1)                    _plan_memory (M1, A1)
          │                                                    │
   _emit_packed (C1)  ← shape RANK로               _emit (C1)  ← layout dict로
     tile(4D) vs row(2D) 디스패치                    row/tile 라우팅
          │                                                    │
        Asm.words ──────────→ runtime.run (mysim) ←────────── Asm.words
```

**핵심 차이**:
- **packed(기본)**: `_build_packed`가 tile-ness를 **4D shape**(`[Rt,Ct,64,64]`)와 **그래프 내 `layout_transform`/`reshape` op**에 직접 구워넣는다. 따라서 `_emit_packed`는 **shape RANK만 보고**(4D→tile emitter, 2D→row 마커) 디스패치한다. 레이아웃 dict가 없다. **모르는 op는 row island로 흘려** model-agnostic.
- **op-list(오라클)**: `_assign_layouts`가 텐서별 `row`/`tile`을 **side-channel dict**로 계산해 `_plan_memory`/`_emit`에 넘긴다. byte-exact 수치 기준이자, packed가 tile화 못 하는 sub-tile head-dim의 fallback.

### 2.2 왜 tile-blocked가 핵심인가 (parity 통찰)

NPU matmul은 64×64 tile 단위다. row-major면 tile을 꺼낼 때마다 **strided 복사(gather)**가 필요하다. tile-blocked면 tile이 연속이라 바로 읽는다(**gather=0**).

> **parity 통찰**: 경계 relayout 1회 = gather 1회. 이득은 **내부 tile 체인이 gather를 아예 안 하는 것**에서 온다. 그래서 "residual 스트림을 끝까지 tile로 유지"하면 모든 projection이 activation을 gather하지 않아 큰 이득이 난다. → packed 경로는 이걸 그래프 전체로 밀어 **모든 config에서 gather=0/scatter=0**을 달성한다.

### 2.3 NPU 고유 자산 2개 (나머지는 TVM 표준으로 흡수)

- **intrinsic** — `npu_gemm_acc`(64×64 MAC) / `npu_fill_zero`(zero-init) TensorIntrin (`tir_backend.py`).
- **walker (`V2Walker`)** — tensorized TIR / 마커 PrimFunc을 걸어 ISA를 emit. v1 `_Walker` 상속·일반화.

---

## 3. 모듈 레퍼런스 (세부 구조)

각 모듈이 무엇을·어디서 하는지, 처음 보는 사람이 코드를 열기 전에 알아야 할 것.

### 3.1 `npu_compiler/v2_backend.py` — 두 진입점 + 모든 pass
**한 줄**: legalized Relax → NPU ISA 백엔드. 두 진입점(`compile_packed` 기본 / `compile_module` 오라클)이 하나의 `_parse` 프론트엔드와 `V2Walker`를 공유.

| 함수 | 위치 | 역할 |
|---|---|---|
| `compile_packed` | :1297 | **기본 진입점**. `_build_packed`→Legalize→`_parse`→`_plan_memory_packed`→`_emit_packed`. **fuse 기본 OFF**. 반환 7-tuple(…, `pack_names`). |
| `compile_module` | :1042 | 오라클/fallback 진입점. 5-pass(`_parse`/`_assign_layouts`/`_detect_oproj_groups`/`_plan_memory`/`_emit`). fuse=True. 반환 8-tuple(…, `tiled_feed`, `layout`). |
| `_parse` | :636 | **공통 프론트엔드**. legalized Relax → (`params`, `ops[_Op]`, `shp`, `const_arrs`(위치 키잉), `out_name`). `strided_slice` begin은 `_slice_begins`(임의 축). |
| `_plan_memory_packed` | :1072 | packed M1: **A1 liveness + exact-size free-list**. `reshape`는 slot 별칭(zero-copy). fusion-aware(F3). |
| `_emit_packed` | :1203 | packed C1: op-family 디스패치. 4D→tile emitter, 2D→row 마커. einsum/layout_transform/ew/reduce/broadcast/transpose/slice/concat/reshape(alias)/F3. |
| `_emit_tile_slice`/`_concat`/`_transpose` | :1159/1171/1189 | **tile-native RoPE** emitter. slice=whole-tile 복사, concat=tile 배치, transpose=`permute_dims([1,0,3,2])`(grid+inner 64×64 strided 0x90). |
| `_emit_reindex`/`_emit_copy_run` | :1136/1151 | 경계 pack/unpack·연속 복사를 **TIR walk 없이 직접** native copy로(컴파일 hot-path 절감). |
| `_detect_oproj_groups` | :807 | F3: same-shape 64-mult matmul add-tree(head별 O-proj) 탐지. `leaf_op="einsum"`이 packed용. |
| `_assign_layouts` / `_plan_memory` / `_emit` | :683/877/945 | op-list 경로의 F4(레이아웃 fixpoint) / M1 / C1. |
| `V2Walker` / `walk_marker` | :31/:467 | v1 `_Walker` 상속(N-D `_bind_match`) + 마커 디스패치(ew/reduce/bcast/transpose/…). 미지 마커는 v1 matmul로 fall-through. |
| `packed_matmul`/`emit_packed_matmul` | :579/604 | tile C=A@B를 `npu_gemm_acc`로 tensorize(lru_cache) / 그 nest를 asm으로. |

**처음 보는 사람 주의**: ① 두 경로가 갈리는 지점은 "레이아웃이 4D shape에 구워졌나(packed) vs dict인가(op-list)". ② packed `fuse` 기본 OFF 이유는 §4-F3. ③ `reshape`는 packed에서 절대 emit/할당 안 됨(offset 별칭). ④ A1 재사용은 **값-불변**(offset만 재배치 → bump과 byte-exact, `top`만 축소). ⑤ op-list `_emit`에서 transpose/slice/concat은 row-only라, 그게 hd≥128에서 RoPE를 row로 유지시켰다(V2-019). packed는 그걸 tile-native로 해결(§4).

### 3.2 `npu_compiler/packed.py` — F4 IR→IR 레이아웃 패스 (packed 경로의 앞단)
**한 줄**: 고수준 레이어 그래프를 tile-blocked 4D로 rewrite하는 **model-agnostic Relax→Relax 변환**. `(packed_module, pack_names)` 반환.

| 함수 | 위치 | 역할 |
|---|---|---|
| `_build_packed` | :111 | 메인. ①matmul-operand weight 스캔→`pack_names` ②op별 레이아웃 정책으로 모든 binding 재구성. |
| `_PACK`/`_UNPACK`/`_t4`/`_is_full` | :31/35/44 | 인덱스맵 `(r,c)→(r//64,c//64,r%64,c%64)` / 4D shape / **full-2D 판별**(둘 다 >1일 때만 tile-transpose, 벡터는 reshape). |
| `_coerce` / `_emit_pack` / `_emit_unpack` | :90/73/84 | 원하는 레이아웃으로 materialize(1회 변환·캐시). **full-2D 상수는 `_emit_pack`이 컴파일 타임 host-pack**(`pack_tiled`→4D `relax.const`, on-device transform 제거). |
| `_tile_slice`/`_tile_concat`/`_tile_transpose` | :252/269/283 | **Phase 4.4 tile-native RoPE**. 64-tile 정렬이면 tile로(whole-tile-column slice, tile-axis concat, `permute_dims([1,0,3,2])`). 아니면 None→row island. |
| `_emit_following` | :298 | layout-following op(ew/unary/sum/max/broadcast) emit. tile일 때 reduce axis=`[1,3]`, row일 때 `[-1]`. |
| `validate` | :361 | 검증 하니스: packed VM을 ref_layer / row-VM / `compile_module`과 대조. |

**처음 보는 사람 주의**: ① **모듈 헤더 docstring(1–23줄)은 STALE**(Phase 4.3 시점 — slice/concat/transpose가 항상 row라고 적혀 있으나 4.4에서 tile-native가 됨). **코드를 봐라.** ② `_is_full` 분기가 핵심: 벡터에 %64 맵을 쓰면 size-1 축을 64로 패딩 = garbage. ③ layout vote: ew 출력은 중간변수 입력 중 하나라도 row면 row, 아니면 tile — 이게 sub-tile RoPE의 row island를 만든다. ④ model-agnostic fallback(:207–224): matmul도 tile-RoPE도 layout-following도 아닌 op은 **row-coerce 입력 위에 verbatim 재구성** → 다른 모델 op이 정확히 흐름. ⑤ weight host-packing: matmul 전용·64-mult 2D weight만 4D로 선언·pre-pack.

### 3.3 `npu_compiler/tir_backend.py` — intrinsic + walker + gemm (공유)
**한 줄**: 64×64 `npu_gemm_acc`/`npu_fill_zero` intrinsic 정의 + matmul을 64³ tensorize + scheduled TIR을 ISA로. **tile-blocked 피연산자면 gather/scatter=0**.

| 함수 | 위치 | 역할 |
|---|---|---|
| `npu_gemm_acc`/`npu_fill_zero` (desc/impl) | :68/45 | 64×64 `C+=A@B` / `C=0` TensorIntrin(walker가 인식하는 마커). |
| `schedule_matmul` | :114 | legalized matmul을 64³ tile로 split/reorder/tensorize. |
| `_Walker` | :130 | scheduled TIR 해석→ISA. `_bind_match`(:222, tile-view stride), `_gather_cached`(:289, **stride==64면 gather 스킵**), `emit_acc`(:316, MAC), `flush`(:344, scatter). |
| `emit_matmul_into` | :425 | 단일 matmul 진입점. `a_tiled`/`b_pack_nt`/`c_tiled`로 tile-blocking 전달. 비-64배수는 64로 pad(byte-exact) — **sub-tile fallback**. |
| `emit_gemm` | :356 | scheduled nest의 **순수 Python 재생**(walk와 byte-identical, ~1.8M FFI 절약) — packed einsum이 이걸 씀. |
| `emit_matmul_accumulate_group` | :531 | **F3**: `Σ_t A_t@B_t`를 공유 누적기에 모아 flush 1회(head별 O-proj scatter H→1). |

**처음 보는 사람 주의**: ① **byte-exact의 핵심**: 계산 순서 `C=0; C=fp16(C+fp16(partial_k))`가 오라클과 동일(fp16(0+x)==fp16(x)). ② **gather=0 원리**: `_gather_cached`가 stride==64면 원본 offset을 그대로 반환. tile-blocked면 tile당 stride가 64라 A/B gather 소멸, C tile이면 flush에 scatter 없음 → **완전 tile matmul = gather 0 + scatter 0**. ③ `emit_gemm`은 `schedule_matmul` nest와 lock-step으로 유지해야 byte-exact.

### 3.4 `npu_compiler/model.py` + `legalize.py` — 그래프 + numpy 오라클
**한 줄**: Llama 레이어(RMSNorm→GQA+RoPE+causal softmax→SwiGLU)를 NPU-legal 프리미티브로 조립(`model.py`) + f64 numpy 정답(`ref_layer`). `legalize.py`가 op 분해 빌더.

| 함수 | 위치 | 역할 |
|---|---|---|
| `build_layer_module` / `ref_layer` | model:107 / :149 | prefill 레이어 그래프(기본 경로 입력) / f64 오라클. |
| `_attn_head` / `_residual_ffn` | model:43 / :59 | prefill·decode 공유 attention head / residual+SwiGLU tail. |
| `build_kv_proj_module`/`build_attn_ffn_module` | model:198/:291 | **decode 커널 2개**(KV-cache). **아직 packed로 미검증** — op-list로만 검증. |
| `rope` / `rope_cos_sin` | legalize:116/:106 | RoPE `q*cos+rotate_half(q)*sin`(slice/negative/concat) / on-device 각도(0x18). |
| `softmax_lastdim` | legalize:130 | **stable softmax**(max-subtract → exp → rowsum → broadcast → divide). |
| `silu` / `rms_norm` | legalize:168/:35 | **native SiLU**(단일 0710 act) / RMSNorm(1/d 스케일을 reduce 전에 → fp16 overflow 회피). |

**처음 보는 사람 주의**: ① 전부 fp16, 모든 `ref_*`는 f64 upcast(수치 정답). ② `_attn_head`/`_residual_ffn`은 prefill·decode 공통 코드 → decode 수치가 prefill과 동일 구조(그래서 decode를 나중에 packed로 그대로 태울 수 있음). ③ RoPE는 그래프(on-device cos/sin+slice)와 ref(테이블+치환행렬) 두 표현이 일치해야 함. ④ decode는 정적 shape 2커널 + **host가 KV append**(NPU는 동적 주소 안 함).

### 3.5 지원 모듈 — `memplan.py` · `isa.py` · `runtime.py` · `driver.py`
**한 줄**: host tile 패킹 + v1 메모리 플래너(`memplan`), ISA 워드 인코더(`isa`), mysim 러너(`runtime`), **v1 프로덕션/decode 드라이버**(`driver`, v2와 별개).

| 함수 | 위치 | 역할 |
|---|---|---|
| `pack_tiled`/`unpack_tiled`/`tiled_numel` | memplan:32/41/48 | **공유** host tile-blocking 규약(`[R,C]↔[Rt,Ct,64,64]`). 두 경로·테스트가 공통 사용. |
| `Asm` / `enc_*` / `reencode` | isa:114/36/191 | **공유** ISA 워드 누적기 / 32-bit 인코더 / round-trip 검증기. |
| `runtime.run` | runtime:47 | **공유** mysim 실행기 — 두 경로가 word 리스트를 여기로 흘림. G-buffer는 fp16 저장(save마다 반올림). |
| `driver.compile_module`/`generate`/`_decode_layer` | driver:18/106/134 | **v1** op-list 컴파일 / decode 생성 루프. **v2_backend와 이름만 같고 다른 함수**(혼동 주의). |

**처음 보는 사람 주의**: ① `pack_tiled` 규약은 memplan·codegen·packed 경로·테스트에서 **byte-identical**해야 함. ② `isa.py`는 인코더일 뿐, 비트 레이아웃 진실은 `mysim.cpp`. ③ **packed(기본) 경로는 `memplan.plan()`을 안 쓴다** — bare `MemPlan`을 offset 홀더로만 쓰고 `_plan_memory_packed`가 직접 배치. `memplan.plan`/`assign_layouts`/`_liveness`와 `driver.py` 전체는 op-list/decode 전용.

### 3.6 테스트 표면 — `tests/test_v2.py` + `run_gate.sh`
**한 줄**: 3계층 수용 테스트 — (1) walker 프리미티브 byte-exact, (2) op-list 오라클 수치, (3) packed 기본 경로.

- **PACKED(기본)**: `test_v2_compile_packed`(:550) · `test_v2_tile_rope`(:580, HD128 row island=0 + packed<op-list) · `test_v2_packed_oproj_fusion_option`(:621, F3 opt-in) · `test_v2_packed_model_agnostic`(:527, tanh→row) · `test_v2_f4_packed_pass`(:690) · `test_v2_packed_matmul`(:704, **byte-exact 브리지**).
- **OP-LIST(오라클)**: `test_v2_compile_full_layer`(:369) · `test_v2_memory_reuse`(:422) · `test_v2_a4_*`(:467/510/750) · `test_v2_oproj_fusion`(:489) · `test_v2_guards_reject_unsupported`(:390).
- **WALKER**: `test_ew2/ew1/silu/reduce_byte_exact`(:73/93/113/179) · `test_native_primitives`(:234).
- `run_gate.sh`: 15개 테스트 파일 + 벤더 byte-exact. 두 브리지 테스트(`packed_matmul` byte-exact, `tile_rope` packed<op-list)가 마이그레이션의 핵심 주장.

---

## 4. 구현한 최적화

전부 커밋·gate GREEN·v1 무변경. **9-에이전트 코드 감사로 packed 경로에 v1 최적화 전부 반영 확인.**

### A1 — liveness 메모리 재사용 (`_plan_memory_packed`)
텐서의 마지막 read 이후 G-buffer slot을 반납해 **같은 크기 후속 텐서에 재사용**(exact-size free-list). packed 경로는 `reshape`를 alias로 처리(zero-copy)해 liveness를 관통. **값-불변**(offset만 이동 → bump과 byte-exact, `top`만 축소). 결과: activation −47~63%.

### A4 — tile-blocked 레이아웃 + weight/const host-packing (`packed.py` + `_emit_packed`)
그래프 전체를 tile-blocked 4D로 rewrite → 모든 matmul이 tile을 바로 읽어 **gather=0/scatter=0**. **weight**(matmul 전용·64-mult)와 **full-2D 상수**(scale/mask/RMS)는 **컴파일 타임 host-pack**(pre-tiled로 feed, on-device `layout_transform` 제거). 결과: HD128 `layout_transform` 31→**2**(입력 pack + 출력 unpack만).

### Phase 4.4 — tile-native RoPE (`_tile_slice`/`_tile_concat`/`_tile_transpose`)
head_dim/2가 64배수(실 Llama hd=128→h=64)면 rotate-half slice/concat과 K^T transpose를 **tile 레이아웃**으로 낮춤(whole-tile-column slice, `permute_dims([1,0,3,2])` = grid+inner 64×64 전치). → RoPE row island **완전 제거**(HD128). sub-tile hd(h=32)는 row island로 fallback(정확). **이것이 packed를 op-list보다 빠르게(0.55×) 만든 결정타.**

### F3 — O-proj accumulate-group fusion (`_detect_oproj_groups(leaf_op="einsum")`) — **opt-in, 기본 OFF**
head별 O-proj add-tree를 1개 누적 그룹으로. **단 packed 기본 OFF**: Phase 4.1/4.4로 residual/O-proj 출력이 이미 tile(scatter=0)이라 **F3의 주 이득(scatter H→1)이 이미 달성** → fusion은 head별 add(24~112 cmd, <0.1%)만 없애고 walk 비용을 추가. **tile-C에 의해 대체**되어 `compile_packed(fuse=True)` 옵션으로만 보존. op-list는 fuse=True 유지(O-proj가 항상 tile은 아님).

### A2 — 컴파일 속도
op-list 마커 `lru_cache` + packed 경계 복사를 **TIR walk 없이 직접 emit**(`_emit_reindex`) → packed HD128 컴파일 66s→2.3s.

---

## 5. 측정 결과

### 5.1 op-list vs packed — 명령 수 (동일 가중치, mysim 실행 대조)

| config | op-list | **packed(기본)** | 배율 | packed LT | gather | scatter | packed rel |
|---|---|---|---|---|---|---|---|
| MEDIUM | 8,573 | 13,197 | 1.54× | 9 | **0** | **0** | 0.0012 |
| SEQ128 hd64 | 55,844 | 76,880 | 1.38× | 21 | **0** | **0** | 0.0142 |
| **HD128 H4 (실 hd)** | 95,013 | **52,453** | **0.55×** | **2** | **0** | **0** | 0.0281 |
| **H8 KV4 HD128 GQA** | 204,277 | **121,229** | **0.59×** | **2** | **0** | **0** | 0.0400 |

→ **실 Llama head_dim(128)에서 packed가 손튜닝 op-list의 55%(GQA 59%) 명령**으로 실행. 모든 config gather/scatter=0. sub-tile hd64는 packed가 더 많음(1.38×, RoPE row island fallback) — 이 경우 op-list가 우위라 fallback으로 유지. 두 경로 출력 수치 동일(op-list rel과 ~일치).

### 5.2 수치 정확성 & 검증
- packed 전체 레이어 rel < 0.05 (MEDIUM 0.0012 ~ HD128 0.0281, HD128의 0.028은 config 고유 fp16 정밀도 — op-list도 동일).
- **parity 감사(9 에이전트, 코드 검증)**: v1 최적화 전부 packed 반영, 놓친 것 없음(§10).
- adversarial 리뷰·감사 누적 수백 config: silent 오답 0.
- byte-exact: `emit_packed_matmul` == v1 matmul emitter (test_v2_packed_matmul), 벤더 예제 byte-exact.

---

## 6. 현재 상태 & 다음 단계

### 완료 (전부 커밋·gate GREEN)
- **IR→IR 마이그레이션 완료**: `compile_packed`가 진짜 IRModule→IRModule(`packed.py`) 위에서 도는 **model-agnostic 기본 경로**. A1·A4·weight/const packing·tile-RoPE·F3(opt-in) 전부 반영.
- **성능**: 실 hd=128에서 op-list의 0.55×, gather=0. 컴파일 2.3s.
- **포지셔닝 확정(2026-08-03 사용자 결정)**: **packed = 기본**, **op-list(`compile_module`) = 오라클 + sub-tile-hd fallback로 유지**(은퇴 아님).

### 다음 (우선순위 순)
1. **packed의 decode/KV-cache 경로 검증** — 현재 packed는 prefill 레이어만 검증. decode 2커널(M=1, KV append)을 packed로 태워 `test_decode` 대조. **진짜 통합의 전제.**
2. sub-tile head-dim fallback 로직 정리(어느 경로 쓸지 자동 선택).
3. (유보) C-model cycle 정보 도착 시 cost 기반 auto-scheduling.

---

## 7. Phase 상태 보드

| Phase | 내용 | 상태 |
|---|---|---|
| 0 | 타당성 spike | 🟢 GO |
| 2 | ISA→TIR intrinsic + walker 일반화 | 🟢 완료 |
| 3 | 완전한 레이어 컴파일(op-list) | 🟢 완료 |
| 3-R | 명시적 pass 리팩토링 + v1 최적화 이식 | 🟢 완료 |
| 4.0–4.3 | packed 기반(matmul·residual tile·F4 패스) | 🟢 완료 |
| **4.4** | **tile-RoPE + const host-pack → packed 0.55×** | 🟢 **완료** |
| **IR→IR** | **compile_packed = model-agnostic 기본 경로 + parity 감사** | 🟢 **완료** |
| decode | **packed의 KV-cache 경로 검증** | ⚪ **다음** |
| auto-sched | cost 기반 스케줄링 | ⚪ 유보(cycle 정보 후) |

---

## 8. 이슈 트래커 (주요)

| ID | 상태 | 제목 |
|---|---|---|
| V2-010~017 | resolved | adversarial 리뷰 8버그(guard/tile-fallback 누락 → silent 오답) 전부 loud-fail 복원 |
| V2-018 | resolved | tile ew 2D-64-mult 상수 미packing → fixpoint 후 tile 마킹 |
| V2-019 | resolved(op-list) / **superseded(packed)** | HD≥128 RoPE tile 크래시. op-list=RoPE row 유지로 해결. **packed=Phase 4.4 tile-native RoPE로 근본 해결(row island 제거)** |
| V2-020 | resolved | O-proj fusion 미구현 → op-list Stage 3 구현, packed opt-in |
| V2-021 | **resolved** | packed에 F3 미반영(parity gap) → `_detect_oproj_groups(leaf_op="einsum")` 구현, tile-C에 대체되어 opt-in |
| V2-DECODE | open | packed의 decode/KV-cache 경로 미검증 → §6-1 |

---

## 9. 결정 로그 (주요)

| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-07-26 | 목표 = TVM 정석 경로(Path A) | 확장·auto-sched |
| 2026-07-27 | 레이아웃 = `layout_transform` op으로 | spike: tile-blocked가 표준 op로 legalize |
| 2026-08-03 | 미들 패스를 진짜 IR→IR로 이행 | Llama 전용 탈피, 모델 유연성 |
| 2026-08-03 | **Phase 4.4 tile-RoPE 진행** | command 수가 실 지표 — 44K가 RoPE 경계 copy로 확인, 실 hd=128은 tile 정렬 |
| 2026-08-03 | **F3 packed는 opt-in(기본 OFF)** | 실측 이득 <0.1%(tile-C가 scatter 이미 제거) — 대체됨 |
| 2026-08-03 | **packed=기본, op-list=오라클/fallback 유지** | packed가 hd=128 우위+범용이나, op-list는 오라클·sub-tile-hd·decode 커버 |

---

## 10. 멀티-에이전트 작업 로그 (이번 세션)

| 작업 | 결과 |
|---|---|
| IR→IR Step 0–4 (packed codegen·A1·model-agnostic) | compile_packed end-to-end 동작, gather=0 |
| Step 5 최적화 (weight/const host-pack·direct emit) | HD128 66s→2.3s, 52K 명령 |
| Phase 4.4 tile-RoPE + Stage C | HD128 packed 0.55× op-list, row island=0 |
| **parity 감사 (9 에이전트, 코드 검증)** | v1 최적화 전부 packed 반영 확인, gap=F3만 → 구현. reuse_act는 A4에 흡수 |
| **모듈 문서 (6 에이전트)** | 본 §3 모듈 레퍼런스의 근거 |

---

## 11. 검증 로그 (이번 세션)

| 대상 | 기준 | 결과 |
|---|---|---|
| packed 전체 레이어 (4 config) | ref_layer rel<0.05 | ✅ 0.0012~0.0400 |
| tile-RoPE HD128 | row island=0, packed<op-list | ✅ 52,453 < 95,013 |
| packed vs op-list 출력 | 수치 동일 | ✅ maxdiff ~0.002 |
| F3 opt-in | fuse fires, fuse≤nofuse | ✅ H=8 group, correct |
| parity 감사 | v1 최적화 ↔ packed | ✅ 전부 반영(F3 구현 후) |
| gate | 15 테스트 + 벤더 byte-exact | ✅ GREEN |

---

## 12. 커밋 로그 (이번 세션, `compiler-v2`)

| 커밋 | 내용 |
|---|---|
| `d0e0a27` | IR→IR step0 — npu_copy64 reindex-copy + feasibility |
| `36ed99f` | step2 — compile_packed layout_transform-native codegen 동작 |
| `79d1bab` | step3 — packed 메모리 패스 A1 liveness |
| `a7f2484` | step4 — model-agnostic F4(모르는 op→row island) |
| `d1cb909` | reindex-copy 스케줄 1함수 + shape 캐시 |
| `523e933` | step5a — weight host-packing |
| `239f51c` | einsum을 fast `emit_matmul_into`로(byte-exact) |
| `d1b8ada` | step5b — direct pack/unpack emit → 66s→2.3s |
| `97e6e8f` | **Phase 4.4 tile-RoPE → packed가 op-list 이김** |
| `fc8deac` | **Stage C const host-pack → HD128 packed 0.55×** |
| `01fae92` | **F3 packed opt-in(기본 OFF) + 9-에이전트 parity 감사** |

---

## 부록 A. 환경
| 항목 | 값 |
|---|---|
| TVM | 0.19.0 (`/home/chokwans99/tvm-src`) |
| python | `/home/chokwans99/anaconda3/envs/npu-tvm/bin/python` (conda `npu-tvm`) |
| PYTHONPATH | `/home/chokwans99/NPU_cmodel/d_compiler` |
| gate | `bash d_compiler/run_gate.sh` (테스트 15 + 벤더 byte-exact) |

## 부록 B. 주요 파일 (→ §3 모듈 레퍼런스)
| 파일 | 역할 | §3 |
|---|---|---|
| `npu_compiler/v2_backend.py` | 두 진입점 + 모든 pass + `V2Walker` | 3.1 |
| `npu_compiler/packed.py` | **F4 IR→IR 레이아웃 패스**(packed 기본 경로 앞단) | 3.2 |
| `npu_compiler/tir_backend.py` | intrinsic + walker + gemm(gather=0) | 3.3 |
| `npu_compiler/model.py` · `legalize.py` | 레이어 그래프 + numpy 오라클 | 3.4 |
| `npu_compiler/memplan.py`·`isa.py`·`runtime.py`·`driver.py` | 지원(tile pack·인코더·mysim·v1 드라이버) | 3.5 |
| `tests/test_v2.py` · `run_gate.sh` | v2 테스트 + 게이트 | 3.6 |
