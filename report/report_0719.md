# NPU 컴파일러 최적화 리팩터링 보고서 (2026-07-19) — tile-blocked 레이아웃(gather/scatter=0) · legalize 통합 · 컴파일 12.4× · decode(M=1) gather/scatter 제거

> 작성일: 2026-07-19 (§9 decode 최적화 추가: 2026-07-21)
> 대상: `d_compiler` 컴파일러(0710 ISA 반영 이후의 SW 최적화 리팩터링)
> 선행 문서: `report/report_0710.md`(0710 ISA 반영 Phase 1–3), `d_compiler/REFACTOR_PLAN.md`(스테이지 계획)
> **이 문서는 처음 보는 사람도 이해하도록 배경부터 서술한다. 지난 리포트(0710) 이후 추가된 모든 사항을 담는다.**
> **07-21 갱신**: prefill 중심의 A4~A1(§5~§8)에 더해, **decode(자기회귀 생성, M=1)** 경로의 gather/scatter를
> 제거한 A5(§9)를 추가했다. decode 스텝 명령 수 **981,664 → 501,824 (−48.9%)**, scatter 0, gather −87%.

---

## 0. 한 문장 요약

0710 리포트가 "**남은 최대 오버헤드는 gather+scatter(≈48%)이고 이건 행-major strided HW가
있어야 없앤다**"로 끝났는데, 이번 리팩터링에서 **그 gather/scatter를 하드웨어 변경 없이
소프트웨어(데이터 레이아웃)로 완전히 제거**했다(**A4 tile-blocked 레이아웃**, 3B prefill
layer **−52.7%**, **gather 0 / scatter 0**). FFN 체인(5c, −19.8%)은 **byte-exact**,
RMSNorm/residual(5d-1)·attention(5d-2)까지 확장하면 reduce의 FP16 재정렬로 **~0.1% tolerance**
(수용). 더불어 **import 경로를 manual 경로와 동일 lowering으로 통일**(A3)하고, **컴파일 시간을
106.3s → 8.6s(12.4×)로 단축**(A2, byte-exact)했다. 마지막으로 **decode(생성, M=1) 경로**에서도 같은
"M=1이면 0번 행만 산다" 관찰로 **gather/scatter를 제거**(A5, §9)해 3B decode 스텝 명령을
**981,664 → 501,824(−48.9%), scatter 0 / gather −87%(잔존은 KV 캐시뿐)** 로 줄였다(byte-exact).
모든 단계가 **회귀 게이트(전체 테스트 + 벤더 byte-exact) 통과 상태로 커밋**됐다.

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
| **A4** | tile-blocked 레이아웃 전파 | ✅ (5c→5d-1→5d-2) | 3B layer **−52.7%, gather 0 / scatter 0** (5c byte-exact, 5d ~0.1% tol) |
| **A3** | import legalization을 manual과 통일 | ✅ | 향후 실제 HF 모델도 동일 최적화 경로 |
| **A2** | 컴파일 속도 | ✅ | **106.3s → 8.6s (12.4×)**, byte-exact |
| **A1** | liveness 기반 메모리 재사용 | ✅ (5d-2 후 완료) | 3B mp.top **278MB → 228MB (−18%)**, 명령 수 불변, byte-exact(§8) |
| **A5** | decode(M=1) gather/scatter 제거 | ✅ (07-21) | 3B decode 스텝 **981,664 → 501,824 (−48.9%)**, scatter 0 / gather −87%, byte-exact(§9) |

> **원래 계획 순서는 A4 → A1 → A3 → A2** 였다. A1을 프로토타입해 3B로 측정하니 **net-negative**
> (§8)였고 그 원인이 `gather_cache`(활성화 gather 재사용)와의 충돌이었다. A1의 안전 전제는 "A4가
> gather를 **전부** 없앤다"인데, 당시엔 A4를 5c(FFN)까지만 하면 attention gather가 잔존해 성립하지
> 않았다 → **A1을 보류하고 A3·A2를 먼저** 끝낸 뒤, "끝까지" 요청에 따라 **A4를 5d-2(attention)까지
> 완주해 gather를 완전히 0으로** 만들었다. 그러면 gather_cache 자체가 불필요해져 A1의 net-negative
> 원인이 사라지므로, **마지막에 A1을 완료**(명령 수 불변 + mp.top −18%, byte-exact)했다.

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

### 5.6 (5d) byte-exact 한계와 타겟팅 실측 — 왜 5c 다음은 tolerance인가

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
가능 구간은 5c가 끝**이다. 5d(RMSNorm·attention 타일링)는 ~1% FP16 재정렬을 수용해야 한다.
이 ~1%는 **모든 프레임워크가 RMSNorm reduce를 제 나름 순서로 하는 표준적·무해한 차이**이고
모델 품질에 영향이 없으므로, **tolerance를 수용하고 5d를 진행**하기로 했다(§5.7). 검증 기준은
byte-exact → **참조 대비 tolerance**로 전환.

---

### 5.7 (5d-1) tile-native RMSNorm + residual stream — 커밋 `7042abd`

RMSNorm/residual 경계를 없애려면(발견 1) 그 op들을 **tile-native**로 만들고(발견 2의 reduce
재정렬 수용), residual 스트림 전체를 tile로 흘려야 한다.

- **5d-1a 이미터**: `emit_row_sum` tile 분기(Nt개 64-열 타일을 elementwise 누적 후 각 행 reduce
  — 전부 연속, gather 0), `emit_broadcast` tile 분기(col=ones-matmul 블록 생성 후 타일 복제,
  row=세그먼트 복제), `alloc_const_tiled`(RMSNorm의 [S,D] 스케일 상수를 tile로 host-pack). 그리고
  `assign_layouts`에 `broadcast_to`=TILE 생성자, `sum`=TILE 입력을 읽는 소비자, scalar/2D-64mult
  상수는 tile elementwise를 demote시키지 않음.
- **5d-1b 배선**: residual **입력 param x를 tile-blocked**로(호스트가 fed 데이터를 pack — 장치
  비용 0). 단 **`pack_params` 플래그 뒤로 게이팅** — 직접(direct) 백엔드 오라클은 tile 레이아웃을
  안 읽으므로, 입력을 pack하지 않는 비교 테스트에선 x를 row로 두어 **direct==tir byte-exact 유지**.
  O-proj accumulate group이 **tile C를 출력**(`c_tiled`), `emit_concat`이 tile 입력을 row 출력으로
  relayout(레이어 출력 concat), assign_layouts가 param을 fixpoint에 넣고 concat을 tile-수용 소비자로.

**★ 실측(3B prefill layer, packed, pre-A4 row-major 대비)**:

| | total | gather | scatter |
|---|---:|---:|---:|
| 5c (FFN만) | 1,792,826 (−19.8%) | 278,528 | 229,376 |
| **5d-1 (+RMSNorm/residual)** | **1,596,062 (−28.6%)** | **180,224 (−56%)** | **131,072 (−75.8%)** |

- gate GREEN: REDUCED(SEQ=8 → tile 안 됨 → hybrid==direct **byte-exact**), MEDIUM/3B tolerance,
  vendor byte-exact 유지. 회귀 테스트 `test_layout.test_tile_rmsnorm`.
- 남은 gather/scatter는 **전부 attention**(score/ctx gather 131k, Q/K/V scatter 82k, O-proj gather 49k)
  → **5d-2**(tile-native RoPE·transpose·softmax + matmul 활성화-B tile)에서. 6개 op를 조율해야 하는 큰 작업.

---

### 5.8 (5d-2) tile-native attention core — 커밋 `2020496` ★ gather/scatter = 0

attention 체인 전체(RoPE→scores→softmax→ctx→O-proj)를 tile로 흘려, **3B prefill layer에서
gather/scatter를 완전히 0으로** 만들었다. 서로 얽힌 6개 변경을 한 번에 조율:

1. **matmul이 tile 활성화 B를 읽음** — scores=Qr@Kt, ctx=P@V에서 Kt/V가 tile. 기존 가중치-패킹의
   `packed_src` 경로를 재사용(`emit_matmul`이 `mp.layout`으로 b_nt 설정) → B-gather 없음.
2. **tile transpose(Kt)** — 각 64×64 타일을 strided-load(0710 열-major=전치)로 읽어 swap된 타일
   위치에 저장. `[R,C]`tile → `[C,R]`tile.
3. **tile strided_slice(RoPE q1/q2)** — head_dim=128이라 반쪽 h=64 = **정확히 1 타일**. slice가
   타일-열 정렬이라 열-타일 부분집합 복사로 끝.
4. **tile→tile concat(RoPE rh)** — 각 입력의 열-타일을 출력 위치에 배치.
5. **tile reduce-max(stable softmax)** — 0710엔 native reduce-max가 없어, 열-타일을 vector-max로
   누적 후 열 fold. (reduce-sum은 5d-1a 재사용.)
6. **assign_layouts** — permute_dims/strided_slice/concat을 tile 생성자(입력 tile 필요), max를
   tile-reading reduce, matmul-B 소비자를 tile-호환으로.

**★ 핵심 버그(격리 테스트로 발견)**: 개별 이미터(transpose/slice/concat/max/matmul-B)는 전부
byte-exact인데 **전체 attention에서 rel=1.11**로 틀렸다. 원인은 **O-proj accumulate group이
tile ctx를 `a_tiled`로 읽지 않고 row로 읽어** garbage를 낸 것(5d-1까진 ctx가 row라 안 드러났음).
`emit_oproj_group`이 term별 a_tiled를 넘기도록 고쳐 rel=0.001로 정정.

**★ 실측(3B prefill layer, packed, pre-A4 대비)**:

| | total | gather | scatter |
|---|---:|---:|---:|
| 5c (FFN, byte-exact) | 1,792,826 (−19.8%) | 278,528 | 229,376 |
| 5d-1 (+RMSNorm/residual) | 1,596,062 (−28.6%) | 180,224 | 131,072 |
| **5d-2 (+attention)** | **1,057,758 (−52.7%)** | **0** | **0** |

![A4 진행: FFN(5c)→RMSNorm/residual(5d-1)→attention(5d-2)로 확장하며 gather(파랑)+scatter(초록)가
0으로 소멸, total −52.7%. 회색은 useful/기타.](figs/0719/g_a4_progression.png)

*(그래프: `report/figs/0719/plot_a4_progression.py` — 실측값 `measurements.json`에서 읽어 재생성.
색은 dataviz reference categorical slot 1(파랑)/2(초록), 검증 통과 ΔE 9.1.)*

**role별 분해 — −52.7%가 어디서 왔나 (0710 리포트의 before/after 그래프와 동일 양식)**:

![role별 before(row)/after(tile). gather/scatter→0, transpose −98%·broadcast −73%·RoPE −43%로
오히려 저렴, reduce만 +6%(tile fold 비용), matmul core 불변.](figs/0719/g_role_before_after.png)

| role | before(row) | after(tile) | Δ | 해석 |
|---|---:|---:|---:|---|
| matmul core (mmul+accum) | 836,832 | 836,832 | **0%** | 실제 MAC — 레이아웃 무관, 불변 |
| **scatter (출력)** | 540,672 | **0** | **−100%** | tile 저장 = 흩어 쓰기 소멸 |
| **gather (입력)** | 409,600 | **0** | **−100%** | tile 저장 = 긁어모으기 소멸 |
| broadcast | 206,639 | 56,327 | −73% | tile col/row broadcast가 per-tile로 더 저렴 |
| layout (RoPE slice/concat) | 148,480 | 83,968 | −43% | tile 열-타일 배치가 per-element보다 저렴 |
| **reduce (norm/softmax)** | 63,608 | 67,652 | **+6%** | **tile-native의 유일한 비용**(열-타일 fold) |
| transpose (Kᵀ) | 16,672 | 288 | −98% | tile당 strided-load 1회(부분타일 copy 소멸) |
| elementwise (SiLU) | 12,690 | 12,690 | 0% | 레이아웃 투명, 불변 |

→ **핵심**: gather/scatter가 사라진 것뿐 아니라 **transpose·broadcast·RoPE도 tile-native가 더 효율적**이다.
유일한 비용은 **reduce +6%**(FP16 합산 순서 재정렬을 유발하는 바로 그 부분). matmul MAC은 당연히 불변.

- **0710 리포트가 "행-major strided HW가 있어야 없앤다"던 gather+scatter(≈48%)를 하드웨어 변경
  없이 SW(tile-blocked 레이아웃)로 100% 제거.** 남은 것은 순수 useful 계산 + tile-native op 오버헤드.
- tolerance ~0.1%(tile reduce 재정렬). gate GREEN(REDUCED byte-exact, MEDIUM/attention/3B tolerance,
  vendor byte-exact). 회귀 `test_layout.test_tile_attention`.

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

## 8. A1 — liveness 기반 메모리 재사용 — 커밋 `6cfe8c5`(보류) → `c05e0b8`(완료) ★

계획엔 **liveness 기반 메모리 재사용**(binding var의 [def,last-read]로 offset 재활용)이 있었다.

**8.1 (당시) 프로토타입이 net-negative였던 이유** — 3B 실측:

| | 메모리(mp.top) | 명령 수 |
|---|---:|---:|
| bump(현행) | 293MB | 2,235,194 |
| liveness 재사용 | 263MB (**−10%뿐**) | **4,236,090 (+90%!)** |

- **명령 +90%**: 재사용이 **write-once 가정을 깨서** `gather_cache`(활성화 gather 재사용)와 충돌 →
  비활성화하면 재-gather로 명령이 2배. → **net-negative라 되돌리고 보류**.

**8.2 (5d-2 이후) 완료** — `reuse=True`(`memplan.plan`/`driver`):
- **exact-size free-list**: binding var의 slot을 마지막 read에서 free하고 이후 **같은 크기** var에 넘김.
- **융합-인지 liveness**: O-proj 그룹은 root에서 emit되며 리프 입력(ctx_h)을 거기서 읽으므로, naive
  그래프 liveness가 **한 스텝 일찍 free**하는 걸 막아 ctx_h를 root까지 live 유지.
- **★ 버그1(격리 bisect로 발견)**: chain/rmsnorm/attn-H1은 통과인데 H≥2(O-proj 융합)만 rel=0.5로 실패.
  원인은 **O-proj root가 `consumed`에도 속해** 죽은 dummy offset 0으로 잘못 배정(param x와 충돌). root는
  실제로 emit되어 residual이 읽으므로 **consumed면서 root가 아닌 것만** dummy로 수정.
- **★ 버그2(사후 코드리뷰로 발견, 커밋 별도)**: "gather=0이라 안전"은 **fully-tiled prefill에만** 참이었다.
  **decode 커널(M=1)** 은 RMSNorm 출력이 row로 남아 matmul-A가 stride=D>64로 **gather**하는데,
  `mm_gather_cache`가 `(off,stride)`로만 키잉되고 invalidate가 없어 **bump 할당의 write-once 불변식**에
  의존한다. reuse가 offset을 재사용하면(hn이 xnorm의 slot을 받음) 캐시가 **오래된 xnorm gather를 반환** →
  gate/up이 xnorm으로 계산됨(garbage, maxdiff 35). **수정: `reuse=True`면 gather 캐시를 끈다**
  (`driver`가 `reuse_act=not reuse`). tiled prefill은 gather=0이라 캐시 미사용 → **비용 0**; row gather만
  재-gather. liveness 자체는 캐시 off 시 30개 config byte-exact로 정확성 입증됨. 회귀 테스트에 **decode
  커널 reuse 검증 추가**(이전엔 prefill만 돌려 이 버그를 놓쳤음).
- **실측(3B prefill layer): mp.top 278MB → 228MB (−18%), 명령 수 완전 불변(+0)**(tiled prefill=gather 없음),
  byte-exact(prefill+decode 모두). gate GREEN. 회귀 `test_layout.test_reuse_memory`.

**8.3 메모리 분해 — weight vs peak activation (이번이 첫 메모리 최적화라 별도 실측)**

G-buffer를 **가중치·상수·활성화**로 쪼개 실측했다(`memplan.plan` footprint, fp16 MB):

| 구성 | 크기 | 성격 |
|---|---:|---|
| **weights (+ tiled 입력 x)** | **202.1 MB (73%)** | 레이어 상주 불가피 — **재사용 불가** |
| constants | 1.6 MB | 베이크된 상수 |
| **activations — bump(재사용 X)** | **73.9 MB** | 모든 binding var를 free 없이 누적한 총량 |
| **activations — A1 peak(재사용 O)** | **23.8 MB** | 동시 live 활성화의 **peak working set** |

![3B prefill layer 메모리: 가중치(회색)가 지배(202MB, 상주 불가피)라 total은 278→228MB(−18%)에 그치지만,
A1이 실제로 공략하는 **활성화 working set peak는 73.9→23.8MB(−68%)**.](figs/0719/g_memory.png)

→ **핵심(이 실험의 결론)**: A1의 진짜 효과는 **활성화 peak −68%**(73.9→23.8MB)인데, **전체로는 가중치가
73%를 차지해 −18%로 희석**된다. 즉 이 NPU에서 **레이어 메모리는 근본적으로 가중치가 지배**하고(모든
레이어가 가중치를 상주시켜야 함), 활성화 재사용은 "동시에 살아있는 활성화 집합(peak)"을 줄이는 것이라
비율 이득은 크지만 절대 총량 이득은 가중치에 가려진다. (가중치까지 줄이려면 양자화/오프로딩 등 별도 축이 필요.)

---

## 9. A5 — decode(M=1) gather/scatter 제거 — 커밋 `f50b82b`(gather)·`3279975`(scatter) ★

§5~§8은 전부 **prefill**(프롬프트 병렬 처리) 레이어를 최적화했다. 실제 서빙의 대부분은 그 뒤의
**decode**(자기회귀 생성)인데, decode는 성격이 정반대다 — 그래서 prefill 최적화(A4)가 **그대로는
적용되지 않는다**. 이 절은 decode에 맞는 최적화다.

### 9.0 동기 — decode는 왜 gather/scatter가 생기나 (prefill과 같은 코드, M만 다름)

decode는 매 스텝 **토큰 1개**만 처리한다 → 모든 matmul의 **M=1**. 코드 경로는 prefill과 **동일**하다
(같은 `emit_matmul`/`emit_gemm`, matrix unit MAC 그대로 사용). 다른 건 **행렬 크기(M)** 뿐이다.
그런데 A4의 tile-blocked 최적화(§5)는 **64배수 차원**에서만 발동(`is_mm64`)하는데 **M=1은 64배수가
아니므로**, decode는 tile-blocked이 아닌 **row-major fallback**으로 돈다. 이때 matmul은 M을 64로 pad해서:

- **입력 A**를 `[64, K]` 타일 격자로 **gather**(0번 행만 진짜, 1~63행은 padding garbage),
- **출력 C**를 `[64, N]`로 **scatter**(0번 행만 유효, 1~63행은 버림)

한다. 하지만 M=1이면 실제로 의미 있는 건 **A의 0번 행 하나, C의 0번 행 하나**뿐 — gather/scatter의
**63/64(≈98%)가 버려질 데이터를 위한 낭비**다. 3B decode 스텝에서 이 낭비가 gather 245,760 +
scatter 270,336 = **명령의 절반 이상**을 차지했다.

> **핵심 관찰**: A의 0번 행은 이미 메모리에 **연속**으로 있고, C도 0번 행만 쓰면 된다. "M=1이라 0번
> 행만 산다"는 사실 하나로 gather와 scatter를 **둘 다** 없앨 수 있다(prefill 경로는 전혀 안 건드림).

### 9.1 gather 제거 — 연속 단일 행 읽기 (`a_m1`) — 커밋 `f50b82b`

matmul이 A 타일 `(0, ko)`를 읽을 때 원래는 stride `K`(compact 2D)로 64행을 gather한다. **M=1이면
stride 64(연속)로** 바꾼다:

- 0번 행 = `A[0, ko·64:…]` (진짜 활성화),
- 1~63행 = 인접 메모리를 그대로 읽어 **버려질 C 1~63행**에만 기여(최종 결과 무관).

`_gather_cached`는 이미 `stride==64`면 gather를 **생략**(연속이라 복사 불필요)하므로, stride만 64로
주면 gather 명령이 사라진다.

- **단일 matmul 경로**(`emit_gemm`/`_emit_tir_gemm`의 `a_m1` 플래그): Q/K/V/gate/up/down proj,
  scores·ctx의 A쪽. Q/K/V/gate/up/down은 B가 **패킹된 가중치**라 B-gather도 없어 gather가 **완전 소멸**.
- **fused O-proj 그룹**(`_Walker.a_m1_src` + `_bind_gemm`/`emit_matmul_accumulate_group`): walker의
  `_bind_match`가 A를 compact offset + **stride 64**로 바인딩 → ctx gather 소멸.
- **부수 효과(안전성 ↑)**: pad된 A의 over-read 범위가 오히려 **줄어든다**(`a_off+K+4032` vs 기존 `a_off+64·K`).

**잔존 gather = KV 캐시(Kᵀ/V) B-read뿐**이다. scores(`Q@Kᵀ`)·ctx(`P@V`)는 B가 2D 캐시라 gather가
남는데, 이건 append 복잡성 트레이드오프로 **의도적으로 남긴다**(§9.4의 잔여 32,768). 반면 kv_proj
커널은 캐시를 **읽지 않고 K/V를 생산**하므로 gather가 **완전히 0**이 된다.

### 9.2 scatter 제거 — tile-blocked C 쓰기 + 0번 행 추출 (`c_tiled`) — 커밋 `3279975`

대칭적으로, C를 `[64, N]` compact(→ scatter)가 아니라 **tile-blocked**(`c_nt=N//64`)로 쓴다. 그러면
각 64×64 타일이 **연속**(stride 64)이라 `emit_acc`가 **in-place로 누적**(cbuf 없음, flush scatter
없음)한다. matmul 후 **logical 0번 행만** 추출:

- tile `jo`의 0번 행 = `cpad + jo·4096`의 연속 64개 → `dst + jo·64` (`_copy_row0_tiled`, Nt개 복사).

즉 **64행짜리 scatter → Nt개 작은 복사**로 대체. O-proj 그룹은 H개 term이 **타일별로 in-place
누적**(term 0가 fill, 이후 term은 `suppress_fill`로 같은 타일에 MAC 누적)한 뒤 한 번만 0번 행을 추출한다.

**byte-exact인 이유**: MAC 연산·누적 순서는 **완전히 동일**하고 **C 타일이 저장되는 위치만** 다르다
(strided cpad에 scatter vs tile-blocked cpad에 in-place). 유효 데이터(0번 행)의 fp16 값은 비트 동일하다.

### 9.3 정확성 검증

- **byte-exact(변경 격리)**: `hybrid(a_m1 ON)` vs `hybrid(a_m1 OFF)` **maxdiff=0**; `hybrid(scatter-opt
  ON)` vs `OFF` **maxdiff=0**. 두 최적화 모두 hybrid 출력을 **비트 불변**으로 유지(git-stash로 on/off 비교).
- **회귀**: `test_decode.py`(M1~M6: prefill=prefill+decode 일치, 컴파일 생성=numpy, 멀티레이어 greedy
  생성 토큰 일치) **전부 통과·결과 불변**(M3 rel 6.1e-04, M4/M6 토큰 [21,21,9]/[28,8] — 이전과 동일).
- **prefill 무영향**: `a_m1`/`c_tiled`(decode)는 **M==1일 때만** 발동. prefill은 M=64배수라 fast path
  → 전 게이트 GREEN + 벤더 byte-exact 유지.
- **주의(direct-vs-hybrid ≈1.5)**: M=1에서 `direct`(논리 one-shot, dims≤255) vs 타일 경로는 **원래부터**
  FP16 누적 순서가 달라 ~1.5 차이가 난다(이번 변경과 무관). 그래서 decode는 애초에 hybrid-vs-**numpy**로
  검증한다. 이번 변경의 정확성은 위 **hybrid on/off maxdiff=0**로 입증했다.

### 9.4 실측 — 3B decode 스텝 (kv_proj + attn_ffn, M=1, MAX=128)

`asm.tags` 실측. before=`90ce86d`(decode 최적화 전), after=`3279975`(현재). 두 상태는 `tir_backend.py`만
다르다(memplan/codegen 동일).

| 커널 | gather | scatter | 총 명령(words) |
|---|---:|---:|---:|
| **kv_proj** (K/V 생산) before | 24,576 | 16,384 | 68,888 |
| **kv_proj** after | **0** | **0** | **28,056 (−59%)** |
| **attn_ffn** (attention+FFN) before | 221,184 | 253,952 | 912,776 |
| **attn_ffn** after | **32,768** | **0** | **473,768 (−48%)** |
| **decode 스텝 합계** before | 245,760 | 270,336 | 981,664 |
| **decode 스텝 합계** after | **32,768 (−87%)** | **0 (−100%)** | **501,824 (−48.9%)** |

![decode 스텝 per-role before/after: scatter→0(−100%), gather −87%(잔존 32,768은 KV 캐시 Kᵀ/V read뿐),
matmul core·reduce·broadcast는 불변. 총 명령 981,664→501,824(−48.9%). M-pad+row-0 추출은 +474%지만
절대량 4,224로 무시가능.](figs/0719/g_decode_before_after.png)

→ **결론**: decode는 tile-blocked(A4)의 대상이 아닌 **row-major 경로**인데도, "M=1이면 0번 행만
산다"는 관찰만으로 **gather+scatter를 516,096 → 32,768(−93.7%)**, **decode 스텝 명령 수를 절반
(−48.9%)** 으로 줄였다. **matmul core(MAC)는 그대로**이므로 이는 유용한 계산은 손대지 않은 **순수
오버헤드 제거**다. 남은 gather 32,768은 전부 KV 캐시 read로, register-indirect 주소(가변 길이 append)
HW가 있어야 없어지는 유일한 잔여 항목이다(§11 남은 과제와 연결).

---

## 10. 변경/추가된 파일 & 커밋 (0710 리포트 이후 전부)

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
| `54f76c1` | A4 5c 확정 문서(당시엔 byte-exact 우선으로 5d 보류) |
| `b34c723` | **A3** — import legalization 통일(native SiLU/stable softmax/sign-inv) |
| `1994e1d` | 계획 갱신(A3 done) |
| `3f3ef0b` | **A2 phase-1** — 파이썬 affine 평가 + 직렬화 벡터화(1.82×) |
| `1cdd48f` | 계획 갱신 + 최종 상태 |
| `55b667d` | **A2 phase-2** — canonical GEMM 직접 emit(12.4×) |
| `d39f254` | **report_0719 최초 작성**(0710 이후 전체) |
| `7042abd` | **A4 5d-1** — tile-native RMSNorm + residual(입력 param tile-pack, O-proj tile 출력, concat relayout) → **−28.6%** |
| `4056f86` | 계획·리포트 갱신(5d-1 done, tolerance 수용 결정) |
| `2020496` | **A4 5d-2** — tile-native attention core(RoPE/transpose/softmax/matmul-B + O-proj a_tiled 버그 수정) → **−52.7%, gather/scatter 0** |
| `8d5affc` | 계획·리포트 갱신(5d-2 done) |
| `05162d7`·`b3cd9ea` | 리포트 전 섹션 5d 반영 + **A4 진행 그래프**(figs/0719/) |
| `c05e0b8`·`fc4f1a4` | **A1** — liveness 기반 활성화 offset 재사용(융합-인지, exact-size free-list) → **mp.top −18%, 명령 수 불변, byte-exact** |
| `efcf226` | 리뷰 수정 — reuse liveness가 `fuse_oproj`를 존중(잠재 결합 제거) |
| `709fee6` | 리뷰 수정 — 기존 row 경로 `emit_row_max`/`emit_transpose`의 **≥256 8-bit 필드 오버플로우** 가드 |
| `383313d` | 리뷰 수정 — **A1 reuse가 gather 캐시로 출력 손상**(decode) → reuse 시 캐시 off + decode 회귀 추가 |
| `1f72a43`·`90ce86d` | 리포트 갱신 — 리뷰 결과 기록 + **역할별 before/after·메모리(가중치 vs 활성화) 그래프**(figs/0719/) |
| `f50b82b` | **A5 gather**(§9.1) — decode(M=1) 활성화 gather 제거(`a_m1`: A 연속 읽기; 단일 matmul + O-proj group) |
| `3279975` | **A5 scatter**(§9.2) — decode(M=1) 출력 scatter 제거(tile-blocked C 쓰기 + 0번 행 추출 `_copy_row0_tiled`) |

**주로 바뀐 소스 파일**:
- `npu_compiler/memplan.py` — 레이아웃 규약·`alloc_tiled`/`alloc_const_tiled`·`assign_layouts`
  (fixpoint: matmul out/ew/broadcast/transpose/slice/concat producer, sum/max reduce, matmul-A/B
  consumer, 입력 param tile-pack)(A4 5a~5d) + **`_liveness`·`reuse=True` 융합-인지 offset 재사용(A1)**.
- `npu_compiler/codegen.py` — 레이아웃별 a/b/c_tiled matmul, `emit_row_sum`/`emit_row_max`/
  `emit_broadcast`/`emit_transpose`/`emit_strided_slice`/`emit_concat`의 tile 분기(5c~5d-2),
  O-proj group a_tiled/c_tiled(5d).
- `npu_compiler/tir_backend.py` — TILE-mode A/C(5b), `ev` 파이썬 평가·`emit_gemm` 직접 emit(A2),
  accumulate-group c_tiled/a_tiled(5d), **decode(M=1) `a_m1`(A 연속 읽기)·`c_tiled`+`_copy_row0_tiled`
  (tile-blocked C→scatter 제거)·walker `a_m1_src`(A5, §9)**.
- `npu_compiler/driver.py` — `layouts` 토글, tile 입력 param host-pack(5d-1b).
- `npu_compiler/import_legalize.py` — manual 경로와 통일(A3).
- `npu_compiler/runtime.py` — program 직렬화 벡터화(A2).
- `tests/test_layout.py` — 레이아웃 규약·TILE-mode matmul·**tile RMSNorm(5d-1)·tile attention(5d-2)** 테스트(신규).
- `d_compiler/run_gate.sh`, `d_compiler/REFACTOR_PLAN.md` — 게이트·계획(신규).

---

## 11. 결론 & 남은 과제

**이번 리팩터링으로 확정된 것**(전부 커밋, gate GREEN, 벤더 byte-exact 유지):

1. **A4 tile-blocked 레이아웃(5c/5d)** — 0710이 "HW가 있어야 없앤다"던 gather/scatter를,
   데이터를 타일 단위로 저장하는 **SW 접근으로 완전히 제거**. FFN(5c, byte-exact −19.8%) →
   RMSNorm/residual(5d-1, −28.6%) → attention(5d-2)로 확장해 3B prefill layer
   **2,235,194 → 1,057,758 (−52.7%), gather 0 / scatter 0**. 5c는 비트 동일, 5d는 ~0.1% tolerance
   (tile reduce 재정렬). `layouts=True/False` 토글로 A/B 비교 가능.
2. **A3 legalize 통일** — import·manual 두 경로가 동일 lowering → 향후 실제 HF 모델도 동일
   최적화 경로.
3. **A2 컴파일 속도** — canonical GEMM을 파이썬으로 직접 emit해 **106.3s → 8.6s (12.4×)**,
   명령 수 불변.
4. **A1 메모리 재사용** — liveness 기반 활성화 offset 재사용. A4가 gather를 0으로 만들어
   gather_cache가 사라진 덕에 **명령 수 불변(+0)**, byte-exact. **3B mp.top 278MB → 228MB (−18%)**.
5. **A5 decode(M=1) gather/scatter 제거** — prefill(A4)과 **같은 코드 경로**지만 M=1이라 tile-blocked이
   아닌 row-major로 도는 decode에서, "M=1이면 0번 행만 산다"는 관찰로 **활성화 gather(A 연속 읽기)와
   출력 scatter(tile-blocked C + 0번 행 추출)를 제거**. 3B decode 스텝 **981,664 → 501,824(−48.9%),
   scatter 0 / gather −87%**(잔존은 KV 캐시 read뿐), byte-exact.

**핵심 교훈**:
- **레이어 전체를 tile로 흘리면 gather/scatter가 원천적으로 사라진다.** row↔tile 경계의 relayout은
  gather와 비용이 같으므로(데이터 볼륨 동일), 부분 tile화는 경계만 이동시킬 뿐이다. **경계가 하나도
  없도록 전 op를 tile-native로** 만들어야(5d-2) 3B 레이어의 gather/scatter가 **완전히 0**이 된다.
  → 0710이 "HW가 있어야 한다"던 것을 **SW 레이아웃만으로** 해결.
- A4를 RMSNorm/attention까지 확장하는 대가는 **reduce의 FP16 재정렬(~0.1%, 비-byte-exact)** 뿐이다
  — 5c(주소만 변경 → 비트 동일)와 근본적으로 다르며, 이 ~0.1%는 모든 프레임워크가 겪는 표준 차이다.
- **격리 테스트가 결합 버그를 잡는다.** 5d-2에서 개별 이미터(transpose/slice/concat/max/matmul-B)는
  전부 byte-exact인데 전체 attention은 rel=1.11이었다. 각 이미터 + 조합을 하나씩 격리 테스트해,
  범인이 **O-proj group의 a_tiled 누락**(tile ctx를 row로 읽음)임을 특정했다. "개별 통과 ≠ 통합 통과".
- `schedule_matmul`의 출력이 **항상 동일 형태**라는 사실이 컴파일 12.4× 단축의 열쇠였다
  (TIR 재순회를 파이썬 루프로 대체).
- **한 최적화가 다른 최적화를 풀어준다.** A1은 원래 net-negative(gather_cache 충돌로 +90% 명령)라
  보류했는데, A4가 gather를 0으로 만들자 그 충돌 원인이 사라져 **A1이 명령 수 불변 순이득으로 전환**됐다.
- **같은 코드라도 shape가 최적화 축을 바꾼다.** prefill과 decode는 **동일한 matmul 코드**를 쓰지만,
  prefill(M=64배수)은 tile-blocked(A4)로, decode(M=1)는 그 대상이 아니라 **"0번 행만 산다"는 row-liveness**
  (A5)로 gather/scatter를 없앤다. 최적화는 연산이 아니라 **그 순간의 텐서 모양**에 맞춰야 한다.

**최종 종료 상태**: **A4(prefill −52.7%, gather/scatter 0) · A3(legalize 통일) · A2(컴파일 12.4×) ·
A1(mp.top −18%) · A5(decode −48.9%, scatter 0/gather −87%)** 를 모두 완료. 전 스테이지 커밋·gate
GREEN·벤더 byte-exact. **prefill·decode 두 경로 모두** gather/scatter 오버헤드를 SW로 제거했다
(decode의 잔존 gather는 KV 캐시 read 하나뿐).

**남은 것(전부 HW 의존 — SW로 net-positive한 최적화는 완료)**:
- **행-major strided 모드**는 tile-blocked 레이아웃으로 **SW에서 완전 대체**됨(더 이상 불필요).
- 남는 HW 항목은 **register-indirect 주소**뿐 — 구체적으로 **decode의 KV 캐시 gather(3B 스텝당
  32,768, §9.4)**. 캐시가 가변 길이 2D라 append/read에 간접 주소가 필요하고, 이것만 있으면 decode
  gather도 0이 된다. 그 외 **HW 루프**(명령 수 감소), 미세하게 A2의 tile-native op(전치/fold) 오버헤드.
