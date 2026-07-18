# NPU 컴파일러 최적화 리팩터링 보고서 (2026-07-19) — tile-blocked 레이아웃 · legalize 통합 · 컴파일 12.4×

> 작성일: 2026-07-19
> 대상: `d_compiler` 컴파일러(0710 ISA 반영 이후의 SW 최적화 리팩터링)
> 선행 문서: `report/report_0710.md`(0710 ISA 반영 Phase 1–3), `d_compiler/REFACTOR_PLAN.md`(스테이지 계획)
> **이 문서는 처음 보는 사람도 이해하도록 배경부터 서술한다. 지난 리포트(0710) 이후 추가된 모든 사항을 담는다.**

---

## 0. 한 문장 요약

0710 리포트가 "**남은 최대 오버헤드는 gather+scatter(≈48%)이고 이건 행-major strided HW가
있어야 없앤다**"로 끝났는데, 이번 리팩터링에서 **그 gather/scatter를 하드웨어 변경 없이
소프트웨어(데이터 레이아웃)로 상당 부분 제거**했다(**A4 tile-blocked 레이아웃**, 3B prefill
layer **−19.8%**, scatter −58%, **byte-exact**). 더불어 **import 경로를 manual 경로와
동일 lowering으로 통일**(A3)하고, **컴파일 시간을 106.3s → 8.6s(12.4×)로 단축**(A2,
byte-exact)했다. 모든 단계가 **회귀 게이트(전체 테스트 + 벤더 byte-exact) 통과 상태로 커밋**됐다.

---

## 1. 배경 — 0710 리포트가 남긴 숙제

0710 리포트(§7.7·§7.8)의 결론은 다음과 같았다. 3B prefill layer를 **실제 생성 경로(가중치
패킹)** 로 측정하면 총 **2,235,194 cmd**, 그중:

| role | 비중 | 정체 |
|---|---:|---|
| scatter | 27.1% | 출력을 strided 위치로 per-row 기록 |
| gather | 21.3% | 입력(활성화 A) 타일을 연속으로 압축 |
| broadcast | 9.1% | col-broadcast(ones-matmul) |
| useful(matmul 등) | 37.6% | — |

그리고 **"0710의 strided load/save는 열-major(전치) 전용이라 행-major gather/scatter를 못
없앤다 → 행-major strided HW 모드가 있어야 48%가 사라진다"** 고 정량적으로 결론지었다.

**이번 리팩터링의 출발 질문**: *정말 HW가 있어야만 하나? 데이터를 애초에 "타일 단위"로
저장하면, 한 matmul이 낸 타일을 다음 matmul이 그대로 읽어 gather/scatter 자체가 안 생기지
않나?* — 이 아이디어가 **A4(tile-blocked 레이아웃)** 이고, 이 리포트의 핵심이다.

---

## 2. 전체 그림 — 무엇을, 왜 (스테이지 개요)

리팩터링은 `d_compiler/REFACTOR_PLAN.md`의 스테이지로 진행했다. **매 단계는 착수 전 브랜치가
아니라, 완료 시 회귀 게이트(아래 §3)를 통과해야만 커밋**하는 규칙을 지켰다.

| 스테이지 | 내용 | 상태 | 핵심 결과 |
|---|---|---|---|
| **Stage 0** | 회귀 게이트 + 측정 기준 고정 | ✅ | 전체 테스트 + 벤더 byte-exact를 한 번에 검사 |
| **A4** | tile-blocked 레이아웃 전파 | ✅ (5c에서 확정) | 3B layer **−19.8%**, scatter −58%, byte-exact |
| **A3** | import legalization을 manual과 통일 | ✅ | 향후 실제 HF 모델도 동일 최적화 경로 |
| **A2** | 컴파일 속도 | ✅ | **106.3s → 8.6s (12.4×)**, byte-exact |
| **A1** | liveness 기반 메모리 재사용 | ⏸ 보류 | 측정상 net-negative(§8) |

> **원래 계획 순서는 A4 → A1 → A3 → A2** 였다. 그런데 A1을 프로토타입해 3B로 측정하니
> **net-negative**(§8)였고, A1의 안전 전제(A4가 gather를 *전부* 없앰)가 A4를 **5c(FFN)에서
> 확정**하며 부분적으로만 성립(attention의 gather는 잔존) → **A1을 보류하고 A3·A2를 먼저**
> 끝냈다.

---

## 3. Stage 0 — 회귀 게이트 (안전장치) — 커밋 `c087be7`

리팩터링은 "동작을 바꾸지 않으면서 내부를 바꾸는" 일이라, **회귀 판정을 자동화**하는 게
먼저다. `d_compiler/run_gate.sh` 하나로:

1. 전체 단위 테스트 14종(`test_isa/layout/matmul/rmsnorm/swiglu/elementwise/tiling/runtime/
   layer/tir_backend/import/attention/real_layer/decode`),
2. **벤더 a.out 대비 byte-exact**(`validate_isa_0710.sh`, b_program 전 예제),

를 실행하고 마지막에 `gate result: GREEN/RED`를 출력한다. **이 문서의 모든 단계는 커밋 직전
이 게이트가 GREEN 임을 확인**했다.

- 값-불변 최적화(A4 5b/5c, A2)는 **출력 완전일치(byte-exact)** 를 요구.
- 값-변경(있었다면)은 tolerance + 참조 비교. (이번엔 값-변경 최적화는 채택하지 않음 → §5.6)

---

## 4. 배경 지식 — 컴파일 파이프라인과 gather/scatter의 정체

(처음 보는 사람을 위한 최소 배경. 이미 아는 사람은 §5로.)

**파이프라인**: `torch 모델 → import → legalize(우리 op으로 정규화) → memplan(G-buffer에 각
텐서의 고정 오프셋 배정) → codegen(명령 emit) → runtime → mysim(c-model)`.

- NPU는 **동적 할당이 없다**. 모든 텐서는 컴파일 타임에 정해진 **G-buffer 오프셋**에 산다
  (`memplan`이 bump 할당). 기본 레이아웃은 **row-major 연속**.
- matmul은 **64×64 타일** 단위로 계산한다(`tir_backend`). 큰 행렬 `[M,K]@[K,N]`을
  `Mt×Nt×Kt`개의 64³ 타일 곱으로 쪼갠다.

**gather/scatter가 왜 생기나** (핵심):
- 입력 `A[M,K]`가 row-major로 연속 저장돼 있으면, matmul이 필요로 하는 **64×64 타일**의 한
  행 64개는 연속이지만 **다음 행은 K칸 건너**에 있다. 즉 타일이 메모리에서 흩어져 있어,
  타일을 연속 버퍼로 **per-row 복사해 모으는 것 = gather**.
- 결과 타일 `C[64,64]`를 넓은 출력 `[M,N]`의 제자리에 쓰려면 다시 per-row로 흩어 쓰는 것
  **= scatter**.
- 0710 리포트가 밝혔듯, **strided load/save(0710)는 전치(열-major) 전용**이라 이 **행-major
  이동을 대체하지 못한다**. 그래서 gather/scatter가 최대 잔여 오버헤드로 남았다.

**핵심 관찰**: gather/scatter는 "**연속(row-major) 저장 ↔ 타일 계산**"의 **불일치** 때문에
생긴다. 그렇다면 **애초에 타일 단위로 저장**하면? → A4.

---

## 5. A4 — tile-blocked 레이아웃 전파 (이 리포트의 핵심)

### 5.0 아이디어 — 왜 이게 0710의 숙제를 SW로 푸나

**tile-blocked 레이아웃**: 논리적으로 `[R,N]`인 텐서를 물리적으로 `[⌈R/64⌉, ⌈N/64⌉, 64, 64]`
(64×64 타일들의 배열, 64배수로 zero-pad)로 저장한다. 이러면 **각 64×64 타일이 메모리에서
연속**이다.

- matmul이 낸 출력 타일을 **연속으로 그대로 저장**(scatter 없음),
- 다음 matmul이 그 입력 타일을 **연속으로 그대로 읽음**(gather 없음).

즉 **matmul → matmul로 이어지는 체인 내부에서는 gather/scatter가 원천적으로 안 생긴다**.
경계(row-major를 요구하는 지점)에만 필요하면 변환을 남긴다. 이건 **HW 변경 없이(=행-major
strided 없이) 데이터 배치만으로** gather/scatter를 제거하는 접근이다.

> 이미 하던 **가중치 패킹**(0710 §7.7, 가중치 상수를 타일로 미리 저장 → B-gather 제거)을
> **활성화(A)와 출력(C)로 확장**한 것으로 볼 수 있다.

### 5.1 (5a) 레이아웃 규약 — 커밋 `8282758`

`memplan.py`에 규약과 도구를 넣고 **round-trip 테스트**로 고정:
- `pack_tiled(arr)`: `[R,N]` row-major → 평평한 `[Rt,Nt,64,64]`(zero-pad).
- `unpack_tiled(flat,R,N)`: 역변환(패딩 제거) = `pack_tiled`의 역.
- `tiled_numel(shape)`: 타일 저장 시 물리 원소 수(64 패딩 반영).
- `MemPlan.layout[var] ∈ {'row','tile'}`, `alloc_tiled(var)`: 타일 레이아웃으로 할당.
- 검증: 64배수·비정형(ragged) shape 모두 `unpack(pack(x)) == x`, 각 64×64 타일이 실제로 연속.

### 5.2 (5b) TILE-mode matmul — 커밋 `4be9ccc`

matmul 백엔드(`tir_backend.emit_matmul_into`)에 **`a_tiled`/`c_tiled`** 를 추가. A/C가 tile-
blocked면 그 타일 오프셋을 `packed_src`에 등록 → 워커가 **stride==64(연속)로 판단해
gather/scatter를 skip**한다(가중치 패킹과 동일한 기계 재사용).

- ★**고립 실측(128×3072×128)**: **103,596 → 3,244 (32×↓)**, gather/scatter → 0,
  **row-major 결과와 byte-exact**(MAC 순서가 같고 레이아웃은 주소만 바꾸므로).
- 즉 **메커니즘이 옳고 무손실**임을 단일 matmul에서 먼저 증명(리스크 격리).

### 5.3 (5c) 레이아웃 배정 패스 — 커밋 `6ac0d5d`

어떤 텐서를 tile로 둘지는 **그래프 전역에서 일관**되게 정해야 한다(한 텐서가 tile인데 소비자가
row를 기대하면 relayout이 필요→무의미). `memplan.assign_layouts`(**fixpoint**)를 새로 구현:

- matmul 출력(M,K,N 모두 64배수)과 **레이아웃-투명 elementwise**(add/mul/silu/…)는 tile 후보.
- **모든 소비자가 tile-호환**(matmul의 A로 읽히거나, tile elementwise)일 때만 tile 유지,
  아니면 row로 **demote**. 이 일관성 덕분에 **경계에 명시적 relayout을 넣을 필요가 없다**
  (row A는 matmul이 자체 gather로 읽고, tile A는 그 gather를 skip).
- 함정 하나: **relax Var는 hash-equal이지만 identity(`is`)-equal이 아니다** → matmul의 A를
  식별하는 데 `is` 대신 hash-eq 집합(`mm_a`)을 써서 고쳤다.

이 패스가 3B prefill layer의 **FFN 체인(gate/up → silu·mul → down)** 을 tile로 묶는다
(TILE var 28개).

### 5.4 옵션 토글 (A/B 비교) — 커밋 `34d88be`

`driver.compile_module/run_module`에 **`layouts=True/False`** 를 노출했다. `True`가 A4(tiled,
기본), `False`가 이전 row-major. **같은 코드로 이전/현재를 A/B 비교**할 수 있게 한 것
(요청 사항). 5c는 값-불변이라 두 경로의 **출력이 비트 동일**이다.

### 5.5 실측 결과 (3B prefill layer, S=128, packed)

| | total | gather | scatter | mp.top |
|---|---:|---:|---:|---:|
| `layouts=False`(이전, row-major) | 2,235,194 | 409,600 | 540,672 | 292MB |
| **`layouts=True`(A4 5c)** | **1,792,826** | **278,528** | **229,376** | 285MB |
| Δ | **−19.8%** | **−32%** | **−58%** | −2.4% |

- gate GREEN, **출력 비트 동일**(byte-exact), TILE var 28.
- **0710이 "HW 없이는 못 없앤다"던 scatter를 −58%** 줄였다(FFN 체인 한정, HW 변경 없음).
- 시작점 2,235,194는 정확히 0710 리포트의 종료 상태(§7.8)와 같다 → **연속된 개선**.

### 5.6 (5d) 왜 여기서 멈췄나 — byte-exact 한계와 타겟팅 실측

**남은 gather/scatter가 어디 있나**를 op별로 실측(5c 이후, `layouts=True`):

| op | gather | scatter | 원인(전부 ROW 경계) |
|---|---:|---:|---|
| **attention score/ctx** | **131,072** | 49,152 | Qr/Kr(RoPE)·P(softmax)·Kᵀ(transpose)가 ROW |
| Q/K/V proj | 49,152 | **81,920** | 출력이 RoPE(slice/concat)로 감 → ROW |
| O proj | 49,152 | 49,152 | ctx(attention) 입력·residual add 출력이 ROW |
| gate/up | 49,152 | 0 | 입력 rms2(RMSNorm)가 ROW |
| down | 0 | 49,152 | 출력 residual add가 ROW |

**핵심 발견 1 — 남은 비용은 전부 "row↔tile 경계"에 있다**. 그 경계의 op(RMSNorm·RoPE·
softmax·transpose·residual)가 전부 ROW라서다. 그리고 **경계에서 relayout은 데이터 볼륨이
같아 gather와 비용이 정확히 동일**(예: `[128,3072]` gather 49,152 = relayout scatter 49,152).
→ **경계를 relayout로 감싸는 건 이득이 0**이고, 그 op들 자체를 **tile-native**(타일 레이아웃에서
직접 계산)로 만들어야 제거된다. (5c가 이긴 이유는 FFN 중간이 matmul→matmul이라 애초에 경계가
없어서였다.)

**핵심 발견 2 — 5d는 byte-exact가 원천적으로 불가능하다**. RMSNorm/attention을 tile화하려면
**reduce**(RMSNorm의 `sum`, softmax의 `sum`/`max`)를 타일 레이아웃에서 해야 한다. 그런데
tile reduce는 **FP16 합산 순서를 바꾼다**(예: 3072개를 한 번에 reduce → 48개 64-열 타일을
먼저 elementwise 누적 후 행 reduce). broadcast/상수-패킹은 정확한 순열이라 무해하지만,
**오직 reduce만 산술 재정렬**을 유발 → 비트 동일 불가. byte-exact를 지키려면 reduce가 행을
gather해야 하고(위 parity로 이득 소멸). **독립 `matmul→RMSNorm→matmul`로 실증: tile-native
RMSNorm은 row 경로와 rel ≈ 0.18%(maxdiff 9.77e-3), tolerance-valid이나 비트 동일 아님**.

**의사결정(2026-07-19)**: 5c는 **주소만 바꿔 비트 동일**이었으므로 **A4의 byte-exact 실현
가능 구간은 5c가 끝**이다. 5d(RMSNorm·attention 타일링, 예상 추가 −~8%)는 ~1% FP16 재정렬을
수용해야 하므로, **5c의 비트-동일 A/B 보장을 우선하여 5d는 보류**했다. 설계와 근거는
`REFACTOR_PLAN.md`(§3.5)에 보존 — 향후 full-model fusion에서 tolerance를 수용할 때 유효.

---

## 6. A3 — import legalization을 manual 경로와 통일 — 커밋 `b34c723`

**문제**: 우리는 레이어를 두 가지로 만든다. (1) **manual**(`model.py`가 BlockBuilder로 직접
구성), (2) **import**(실제 torch 모델을 `torch.export`→Relax로 들여옴). 0710 retarget은
manual 경로만 최신화돼 있었고, **import 경로(`import_legalize.py`)는 구식 lowering**을
쓰고 있었다:
- SiLU를 `z/(1+exp(-z))` **5-op으로 분해**(native SiLU 미사용),
- softmax를 **max 차감 없이**(비-stable),
- `negative`를 `subtract(0,x)`로(native sign-inversion 미사용).

**변경**: import 경로가 manual과 **동일한 공통 빌더(`legalize.py`)로 위임**하도록 통일:
- `relax.nn.silu` → `legalize.silu`(native HW 활성화 유지),
- `relax.nn.softmax` → `legalize.softmax_lastdim(stable=True)`(max 차감),
- `relax.negative` → 그대로 유지(codegen이 native sign-inversion 0x16으로 lower, RoPE
  rotate_half와 공유).
- (함정: 이 파일에 `legalize`라는 **동명의 진입 함수**가 있어 `from . import legalize`가
  가려졌다 → `import legalize as _lg`로 alias.)

**검증**: 기존 `test_import.test_llama_block`(전체 Llama decoder block을 import해 torch
레퍼런스와 대조)이 통일된 lowering을 그대로 커버 → **rel = 0.025 (< 0.05)**. rope는 import
경로에서도 이미 primitive(slice/concat/negative/mul/add)라 manual과 동일. 전체 gate GREEN,
벤더 byte-exact 유지.

**의의**: 이제 **import·manual 두 경로가 같은 그래프로 legalize** → 향후 실제 HF 3B 모델을
받아 import해도 **동일한 최적화(native SiLU, stable softmax, sign-inv)를 그대로** 받는다.

---

## 7. A2 — 컴파일 속도 12.4× (byte-exact) — 커밋 `3f3ef0b`(phase-1), `55b667d`(phase-2)

### 7.1 문제

3B prefill layer 한 커널을 컴파일하는 데 **106.3s**가 걸렸다(측정). 명령 수(런타임 비용)와
무관한 **컴파일-타임 병목**이며, 반복 개발·측정을 크게 느리게 했다. (참고: 명령 수는 HW 루프
부재로 어차피 불변이므로 A2는 **오직 컴파일 속도만** 바꾼다 — 값·명령 수 불변.)

### 7.2 프로파일 → hot path 확정 (추측 금지)

cProfile로 3B 컴파일을 분해하니, 병목은 **matmul을 emit하는 TIR 워커**였다:
- `tir_backend.ev()`가 **인덱스식마다 TVM FFI로 `substitute`+`simplify`**를 호출(769k회, ~70s),
- `_bind_match`(그 `ev`를 호출, ~109s), `_bind`의 IntImm 생성 등.

### 7.3 phase-1 — 파이썬 affine 평가 + 직렬화 벡터화 (`3f3ef0b`)

- **`ev()`를 순수 파이썬 affine 평가로 대체**: 스케줄된 GEMM의 인덱스식은 전부
  `Add/Sub/Mul/FloorDiv/FloorMod/Min/Max/Cast`뿐 → `_ev_fast`가 **int env로 파이썬에서 직접
  계산**하고, 미지원 노드 타입만 기존 `substitute+simplify`로 fallback. env를 **python int로
  저장**(IntImm FFI 생성 제거).
- `runtime._program_bytes`: per-word `struct.pack` 루프(~1.8M회) → **numpy 벡터화 uint32
  pack**(바이트 동일).
- **결과: 106.3s → 58.5s (1.82×)**, byte-exact.

### 7.4 phase-2 — canonical GEMM 직접 emit ★ (`55b667d`)

phase-1 이후 재-프로파일하니 병목이 **`_bind_match` + 재-walk마다 TVM FFI wrapper ~1.8M개
생성**으로 옮겨갔다(워커가 TIR을 매 출력 타일마다 다시 순회하며 새 파이썬 wrapper를 만든다).

**핵심 발견**: `schedule_matmul`이 내는 스케줄은 **항상 동일한 canonical nest**다:
```
for (io,jo) in grid(Mt,Nt):
    fill  C[io,jo]                       # 초기화(실제 ISA는 T1로 지연)
    for ko in range(Kt):
        gemm_acc  C[io,jo] += A[io,ko] @ B[ko,jo]
```
그리고 각 타일의 포인터/stride는 (io,jo,ko)의 **affine**이며, **정확히 pack_tiled 규약과 일치**
한다. → **TIR을 순회할 필요 없이 이 nest를 순수 파이썬 루프로 재생**할 수 있다.

- `_Walker.emit_gemm(...)`: 위 nest를 파이썬으로 직접 돌며 `emit_fill/emit_acc`를 호출.
  타일 순서·per-tile 포인터/stride를 **`_bind_match`와 동일하게 계산**(compact: row-major
  offset + row_stride; tile-blocked: `(r*Nt+c)*4096` + stride 64). gather_cache/누산 로직은
  기존 `emit_acc`를 그대로 재사용 → **walk와 byte-identical**, TIR 순회·wrapper 생성 소멸.
- 보조: `_bind_match`의 반복 FFI 속성 접근을 지역변수로 hoist, `_scheduled_gemm`에 lru_cache.
- **O-proj group 경로는 워커를 유지** → walker/`ev`/`_bind_match`가 **여전히 사용**되어 dead
  code가 없고, 직접 emit의 **검증 오라클** 역할을 겸한다.

### 7.5 실측

| 단계 | 컴파일 시간 | 배수 |
|---|---:|---:|
| baseline(리팩터 전) | 106.3s | 1× |
| phase-1(파이썬 affine 평가) | 58.5s | 1.82× |
| **phase-2(직접 GEMM emit)** | **8.6s** | **12.4×** |

- **명령 수 불변**(1,792,826 words 동일) — 순전히 컴파일 속도. gate GREEN, 벤더 byte-exact.
- 계획 DoD("100s→10s대")를 **초과 달성**(8.6s).

---

## 8. A1 — 왜 보류했나 (측정 기반 net-negative) — 커밋 `6cfe8c5`

계획엔 **liveness 기반 메모리 재사용**(binding var의 [def,last-use]로 offset 재활용)이 있었다.
프로토타입해 3B로 **실측**하니:

| | 메모리(mp.top) | 명령 수 |
|---|---:|---:|
| bump(현행) | 293MB | 2,235,194 |
| liveness 재사용 | 263MB (**−10%뿐**) | **4,236,090 (+90%!)** |

- **메모리 −10%뿐**: 버퍼는 **가중치(~200MB, 레이어 상주 불가피)** 가 지배 → 활성화 재사용
  효과가 작다.
- **명령 +90%**: liveness 재사용이 **write-once 가정을 깨서** `gather_cache`(활성화 gather
  재사용)와 충돌 → 비활성화하면 재-gather로 명령이 2배.
- **O-proj 융합**이 leaf 입력을 root에서 지연-read 하는데 naive SSA liveness가 조기 free →
  liveness가 codegen의 **실제 스케줄과 일치**해야 하는 추가 결합.

→ **net-negative라 되돌리고 보류**. A4가 gather를 **전부** 없애면 `gather_cache`가 불필요해져
A1이 안전해지는데, A4를 **5c(FFN)에서 확정**해 **attention의 gather는 잔존** → A1의 안전 전제가
아직 미성립. **5d 완료 후 재평가** 대상.

---

## 9. 변경/추가된 파일 & 커밋 (0710 리포트 이후 전부)

| 커밋 | 내용 |
|---|---|
| `c087be7` | Stage 0 — 스테이지 계획 + 회귀 게이트(`run_gate.sh`) |
| `6cfe8c5` | A1 net-negative 실측 → 순서 재정렬(A4 먼저) |
| `3097f12` | A4 구체 설계(tile-blocked 레이아웃) |
| `8282758` | **A4 5a** — memplan 레이아웃 규약(`pack_tiled`/`unpack_tiled`/`alloc_tiled`) + 테스트 |
| `4be9ccc` | **A4 5b** — matmul TILE-mode A/C(gather/scatter skip), 32× 고립 실측 |
| `650cdf5` | 계획 갱신(5a/5b done) |
| `6ac0d5d` | **A4 5c** — `assign_layouts` fixpoint → FFN 체인 tile화(−19.8%) |
| `97624a0` | 계획 갱신(5c done) |
| `34d88be` | **토글** — `driver`에 `layouts=True/False` 노출(A/B 비교) |
| `52fb98b` | 5d 타겟팅 실측(op별 gather/scatter + 경계-parity 발견) |
| `54f76c1` | **A4 확정** — 5c에서 종료(byte-exact), 5d 보류(reduce 재정렬) |
| `b34c723` | **A3** — import legalization 통일(native SiLU/stable softmax/sign-inv) |
| `1994e1d` | 계획 갱신(A3 done) |
| `3f3ef0b` | **A2 phase-1** — 파이썬 affine 평가 + 직렬화 벡터화(1.82×) |
| `1cdd48f` | 계획 갱신 + 최종 상태 |
| `55b667d` | **A2 phase-2** — canonical GEMM 직접 emit(12.4×) |

**주로 바뀐 소스 파일**:
- `npu_compiler/memplan.py` — 레이아웃 규약·`alloc_tiled`·`assign_layouts`(A4).
- `npu_compiler/tir_backend.py` — TILE-mode A/C(5b), `ev` 파이썬 평가·`emit_gemm` 직접 emit(A2).
- `npu_compiler/codegen.py` — 레이아웃에 따른 a_tiled/c_tiled·elementwise 물리 크기(A4 5c).
- `npu_compiler/driver.py` — `layouts` 토글.
- `npu_compiler/import_legalize.py` — manual 경로와 통일(A3).
- `npu_compiler/runtime.py` — program 직렬화 벡터화(A2).
- `tests/test_layout.py` — 레이아웃 규약·TILE-mode matmul 테스트(신규).
- `d_compiler/run_gate.sh`, `d_compiler/REFACTOR_PLAN.md` — 게이트·계획(신규).

---

## 10. 결론 & 남은 과제

**이번 리팩터링으로 확정된 것**(전부 커밋, gate GREEN, 벤더 byte-exact 유지):

1. **A4 tile-blocked 레이아웃(5c)** — 0710이 "HW가 있어야 없앤다"던 gather/scatter를,
   데이터를 타일 단위로 저장하는 **SW 접근으로 FFN 체인에서 제거**. 3B prefill layer
   **2,235,194 → 1,792,826 (−19.8%)**, scatter **−58%**, gather −32%, **비트 동일**.
   `layouts=True/False` 토글로 이전/현재를 A/B 비교 가능.
2. **A3 legalize 통일** — import·manual 두 경로가 동일 lowering → 향후 실제 HF 모델도 동일
   최적화 경로.
3. **A2 컴파일 속도** — canonical GEMM을 파이썬으로 직접 emit해 **106.3s → 8.6s (12.4×)**,
   명령 수 불변.

**핵심 교훈**:
- **row↔tile 경계의 relayout은 gather와 비용이 같다**(데이터 볼륨 동일). 그래서 A4의 이득은
  "경계를 감싸는" 게 아니라 **matmul→matmul처럼 경계가 없는 체인을 tile로 묶는 데서** 나온다.
- A4를 attention/RMSNorm까지 확장(5d)하려면 **reduce의 FP16 재정렬(비-byte-exact, ~1%)** 을
  수용해야 한다 — 5c는 주소만 바꿔 비트 동일이었던 것과 근본적으로 다르다.
- `schedule_matmul`의 출력이 **항상 동일 형태**라는 사실이 컴파일 12.4× 단축의 열쇠였다
  (TIR 재순회를 파이썬 루프로 대체).

**남은 과제(우선순위·근거와 함께)**:
- **5d**(attention/RMSNorm tile-native, 추가 −~8%+, 최대 항목은 attention gather 131k) —
  ~1% FP16 tolerance 수용 시 진행 가능. 최대·최난이도.
- **A1**(메모리 재사용) — 현재 측정상 net-negative. **5d로 attention gather까지 없앤 뒤**
  gather_cache가 불필요해지면 재평가.
- **HW 의존**: 여전히 **행-major strided 모드**(잔여 attention gather/scatter 직격),
  **register-indirect 주소**(KV append·가변길이 decode), **HW 루프**(명령 수 감소) — 별도
  벤더 요청서로 분리. (SW로 마무리 가능한 net-positive 최적화는 이번에 모두 완료.)
