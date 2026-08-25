# ISA ver.09 명세 (N0 v4 — memory 편 + compute/양자화 편 확정)

> branch `cmodel-v09` · 2026-08-24 (v4 — 5차 결정(양자화 8건) 반영)
> 원칙: **ver.08 ISA 구조를 유지**하고 필요한 최소만 추가한다.
> 용어: 정수 결과를 실수로 복원하는 것은 전부 **dequant(복원)** 로 부른다.

## 0. 확정 결정 요약

**4차 (memory)**

| # | 항목 | 결정 |
|---|---|---|
| M1 | Global | 32-bit 주소, **단위 = 32-bit 칸** → 16 GiB. DMA 전용, dtype 개념 없음 |
| M2 | SRAM | 8 MiB, **nibble 단위** (유효 24-bit, ver.08 32-bit field 재사용). compute는 SRAM만 접근 |
| M3 | rows/cols/stride | **element 단위** — 16-bit field·값 ver.08과 동일 |
| M4 | dtype | **0x88/0x89의 spare 2-bit [26:25]**: `00=FP16 01=FP32 10=INT8 11=INT4` (vector 피연산자도 동일 descriptor 사용) |
| M5 | DMA | GLOAD 0xA0 / GSTORE 0xA8, dtype 무관·동기, 4-word (SRAM 주소 24-bit는 opcode word에 수납) |

**5차 (compute/양자화)**

| # | 항목 | 결정 |
|---|---|---|
| C1 | dtype 조합 | 5개: FP16×FP16 / FP16×INT8(W8A16) / FP16×INT4(W4A16) / INT8×INT8(W8A8) / INT8×INT4(W4A8). 그 외 오류 |
| C2 | W8A16 dequant | **출구(drain)에서 scale** — 입구는 무손실 형변환만 |
| C3 | W8A8 출력 | **항상 FP16으로 dequant** (INT 출력은 다음 VQUANT 담당) |
| C4 | scale 전달 | 신설 설정 명령 **0x8A(a_scale) / 0x8B(w_scale)** |
| C5 | granularity | act **per-row 동적** / weight **per-col + K-group g∈{64,128}**. g<64·act per-channel 미지원 |
| C6 | act 양자화 | **조립식**: 기존 vector 연산 + 신설 `VQUANT`. 대칭(zero-point 없음), RNE, 포화 ±127(INT4는 ±7) |
| C7 | vector INT | 산술은 FP 유지, INT는 **VQUANT/VDEQUANT**로만 출입 |
| C8 | scale dtype | **FP32** |

---

# Part I — Memory (확정, v3와 동일)

## 1. 메모리 모델

```
[host 입력 파일] ──적재──▶ ┌────────────────────────────────┐
                          │ Global Memory                   │  32-bit 주소, 32-bit 칸 단위
                          │ 최대 16 GiB (실크기 = 파일 크기)  │  내용물 해석 없음
                          └───────────┬────────────────────┘
                     GLOAD 0xA0 / GSTORE 0xA8 (dtype 무관, 동기, 원본 이동)
                          ┌───────────▼────────────────────┐
                          │ SRAM  8 MiB                     │  32-bit field, nibble 단위
                          │ FP32/FP16/INT8/INT4 공존         │  (유효 24-bit, 상위는 0 필수)
                          └───────────┬────────────────────┘
                    기존 ver.08 주소설정/load/save/연산 (SRAM만 접근)
                          ┌───────────▼────────────────────┐
                          │ Matrix 64×64 / Vector 256-lane  │  dtype 해석은 여기서만
                          └────────────────────────────────┘
```

### 1.1 단위 규칙

| 값의 종류 | 단위 | field | 비고 |
|---|---|---|---|
| Global 주소·stride·cols(DMA) | **32-bit 칸** | 32-bit / 16-bit | 16 GiB. DMA에서만 등장 |
| SRAM 시작 주소 (0x80/0x8A/0x8B) | **nibble** | 32-bit (2-half) | |
| rows | 행 개수 | 16-bit | 단위 아님 |
| cols·stride (0x88/0x89) | **element 개수** | 16-bit | ver.08과 의미·값 동일 |
| vlen (0x82) | element(lane) 개수 | 기존 그대로 | |
| 환산 | global 1칸 = SRAM **8 nibble** | — | 유일한 환산 관계 |

- 주소 생성: `addr(r,c) = base_nibble + (r×stride + c) × w`,
  w = dtype 폭(nibble) ∈ {FP32:8, FP16:4, INT8:2, INT4:1}
- **정렬 규칙**: global tensor 행은 32-bit 칸 경계 시작·칸 배수 길이 /
  SRAM tensor 시작은 dtype 폭 배수 / DMA의 SRAM 주소는 8-nibble 정렬.
  64-배수 차원에서 모두 자동 충족. 위반은 오류
- 표현 한계: cols·stride 16-bit = dtype 무관 행당 최대 65,535 element

### 1.2 칸 내부 배치 순서 (little-endian 고정)

**낮은 주소 = 낮은 bit.** 칸의 bit `[4k+3:4k]` ↔ SRAM nibble `base+k` (k=0..7).
예: 칸에 든 FP16 (a,b) = bit[15:0], bit[31:16] → SRAM에서 a가 `[S,S+4)`,
b가 `[S+4,S+8)`. INT4 8개는 값0이 bit[3:0]부터. host numpy(little-endian)와 일치.

## 2. dtype 운반 — spare bit (신규 명령 없음)

dtype은 **operand 단위** 속성. 자기 descriptor와 같은 word에 원자적으로 싣는다:
`0x88`/`0x89`(rows/cols)의 spare 2-bit **[26:25]** 가 해당 operand의 dtype이다.
vector 연산의 피연산자도 같은 descriptor(`desc_[operand]`)를 거치므로 별도 통로가
필요 없다 — `0x82`(vlen)는 길이만 나르고 dtype을 싣지 않는다 (단일 기제 원칙).
**FP16=00** 이므로 ver.08 프로그램은 그대로 "dtype FP16인 유효한 v09 프로그램".

## 3. DMA — GLOAD 0xA0 / GSTORE 0xA8 (4 words, 동기)

```
w0: [31:8] SRAM 시작 주소 (nibble, 24-bit 전폭 = 8 MiB, 8-nibble 정렬)  [7:0]=0xA0|0xA8
w1: global 시작 주소 (32-bit 칸 단위)
w2: global 행 stride (칸 단위)          ← wide stride 직접 표현 (V3-027 해소)
w3: [31:16] rows  [15:0] cols (칸 개수)
```

- SRAM 주소는 유효 폭이 정확히 24-bit이므로 opcode word의 예약 구간에 그대로
  수납된다 (다른 주소 설정 명령의 32-bit 2-half 형식과 달리 DMA만 단일 word —
  주소가 명령과 원자적으로 이동해 stale 위험도 없음)
- 확장 여지: SRAM-scatter 등 후속 플래그가 필요해지면 w3의 상위 예약 구간을
  쓰거나(rows/cols는 16-bit로 충분) w4를 덧붙이는 형식으로 확장한다

행 r: global `[w1+r×w2 .. +cols)` 칸 ↔ SRAM `[w0[31:8]+r×(cols×8) ..)` nibble
(§1.2 순서, SRAM 쪽 행 연속). dtype 무관 — packed blob도 그냥 "칸들".
범위 초과/비정렬/상위 8-bit≠0 → **즉시 오류 종료**.

## 4. 기존 ver.08 ISA 재해석 (인코딩 불변)

| 명령 | v09 재해석 |
|---|---|
| `0x80` 주소 (2-half) | **SRAM nibble 주소** — 두 half 모두 emit 필수 (V3-025 관례를 규칙으로) |
| `0x88/0x89` rows/cols | element 개수 — 값 그대로 + spare bit dtype |
| `0x82` vlen | 그대로 (상한 256 lane) |
| `0x90/0x98` load/save | SRAM↔유닛 (인코딩·strided bit 그대로) |
| 연산 전체 | 그대로 — dtype mode는 Part II |
| MAIN/PARTIAL | 그대로 (접근=PARTIAL, MAIN=stride) |

FP16 이관 공식: shape/stride/vlen field 값 ver.08과 동일, 시작 주소만 SRAM
nibble — bit-exact 불변식의 기계적 근거.

## 5. 제어 및 host I/O

| opcode | 명령 | 동작 |
|---|---|---|
| 0x00 | NOP | |
| 0xF0 | SNAPSHOT | global 전체(실크기) 출력 파일 append |
| 0xFF | **HALT** | 종료 + global 전체 기록 — 유일한 결과 회수 경로 |

입력 파일 = global 초기 이미지. SRAM은 0 초기화, host 직접 접근 불가.
per-invocation 실행, SRAM은 프로그램 간 비유지.

---

# Part II — Compute / 양자화 (5차 결정 반영)

기본 산술 계약은 ver.08 유지: **FP16 저장 / FP32 내부 연산 / 저장 시 RNE**.
양자화의 수학적 뼈대: x ≈ s·q (s=scale 실수 1개, q=정수),
y = Σₖ a·w ≈ **s_a·s_w × Σₖ(qa·qw)** — scale이 누적 도중 상수인 한
정수 곱-누적 후 실수 곱 1번으로 복원(dequant)된다.

## 6. dtype 조합표 (C1) — matrix unit

| src0(act) × src1(weight) | 통칭 | 입구(feeder) | 배열 | 누적 | 출구(drain) |
|---|---|---|---|---|---|
| FP16 × FP16 | 기존 | 통과 | FP16×FP16 | FP32 | FP16 저장 (기존 그대로) |
| FP16 × INT8 | W8A16 | INT8→FP16 **무손실 변환** | FP16×FP16 | FP32 | × w_scale → FP16 |
| FP16 × INT4 | W4A16 | INT4→FP16 무손실 변환 | FP16×FP16 | FP32 | × w_scale(group) → FP16 |
| INT8 × INT8 | W8A8 | 통과 | INT8×INT8 | **INT32** | × a_scale × w_scale → FP16 |
| INT8 × INT4 | W4A8 | INT4→INT8 부호확장(무손실) | INT8×INT8 | INT32 | 동일 |

- 그 외 조합(FP32 관여 포함)은 **오류**. FP32 dtype의 compute 실사용처는
  scale 벡터(§8)와 vector FP32 저장(§10)뿐
- 무손실 근거: INT8(−128..127)·INT4(−8..7)는 FP16/INT8에 정확히 표현됨 —
  입구에는 반올림 지점이 없다 (C2의 핵심)
- 연산 명령(0x40~0x43)에 **신규 mode bit 없음** — 동작은 operand descriptor의
  dtype 조합에서 유도. MAC bit([27]) 등 기존 bit 유지

## 7. Matrix 파이프라인 — 입구 / 배열 / 출구 dequant (C2·C3)

```
SRAM ─▶ feeder(무손실 변환만) ─▶ 64×64 배열(곱-누적: FP32 또는 INT32)
                                        │  MAC bit: K-tile 사슬 잇기 (scale 상수 구간)
                                        ▼
                       drain: FP32 dequant-누적기 (64×64)
                       acc_out += FP32(배열 누적값) × w_scale[n] (× a_scale[m])
                                        │  writeout 시 RNE → FP16 저장
                                        ▼
                                      SRAM (FP16)
```

- **scale 적용 규칙**: src1이 INT이면 w_scale[열] 적용, src0이 INT이면
  a_scale[행] 적용, FP16 operand에는 미적용. FP16×FP16은 drain을 통과만
  (기존과 동일 — 불변식 유지)
- **drain 누적 flag** (matrix save 0x98의 spare 2-bit, 가안 [27]=carry-in,
  [26]=hold): group-wise(§9)에서 group마다 scale이 바뀌므로, group g의 배열
  누적값을 dequant해 **FP32 drain 누적기에 더하고**(hold=1) 마지막 group에서
  FP16으로 기록(hold=0). `00` = 기존 동작(즉시 FP16 기록) → ver.08 프로그램
  무영향. per-col만 쓰는 경우 사슬 1개라 flag 불필요
- FP16 중간 반올림이 group마다 생기는 것을 막는 장치가 이 FP32 drain
  누적기다 — vector reduce carry(§10)와 동일한 설계

## 8. scale 전달 — 0x8A / 0x8B 신설 (C4·C8)

- `0x8A` = **a_scale 주소** (activation, per-row), `0x8B` = **w_scale 주소**
  (weight, per-col 또는 group행). 인코딩은 0x80과 동일한 2-half 형식,
  값은 SRAM nibble 주소. 가리키는 대상은 **FP32 벡터** (8 nibble/원소)
- scale 벡터 배치: a_scale = 길이 M(행 수) / w_scale(per-col) = 길이 N /
  w_scale(group-wise) = [G, N] 행렬, **group g의 사슬 전에 compiler가 0x8B를
  g행 주소로 재설정** — 인덱싱 하드웨어 불요
- 0x8A/0x8B는 0x80과 같은 stateful descriptor — 기존 주소 설정과 동일한
  관례("사용 직전 재설정") 적용. FP16 조합에서는 읽지 않음

## 9. Granularity 규칙 (C5)

| 지원 | 형식 | 하드웨어 근거 |
|---|---|---|
| ✅ act | **per-row(=per-token) 동적** symmetric | 행별 scale은 drain에서 행 곱 |
| ✅ weight | **per-col(=per-channel)** symmetric | 열별 scale은 drain에서 열 곱 |
| ✅ weight | **K-group, g ∈ {64, 128}** | 배열이 K를 64씩 잘라 누적하므로 g가 64의 배수면 "group 사슬 → drain 누적" (§7 flag)으로 처리 |
| ❌ | g < 64 | 누적 도중 scale 교체 — 배열 내부 로직 필요, 미지원 |
| ❌ | act per-channel(K방향) | scale이 합산축에 있어 밖으로 못 뺌 — 수학적으로 불가. 필요 시 SmoothQuant류 host 전처리(가중치로 scale 재배분)로 해결 |

INT4 weight는 group-wise 필수(품질), INT8 weight는 per-col 기본.
업계 표준(GPTQ/AWQ g=128)과 정합.

## 10. Vector unit — FP 유지 + VQUANT/VDEQUANT (C6·C7)

- **산술 연산(0x01..0x19)은 FP 전용 유지** (ver.08 그대로). INT operand 투입은 오류
- vector load/save는 descriptor dtype **FP16/FP32** 지원: dst=FP32(01)면
  내부 FP32 결과를 **반올림 없이 그대로 저장** (scale 생산에 사용, 신규 반올림
  지점 아님 — FP16 모드에서는 미사용)
- **`VQUANT` 0x1A** (신설, ver.08 빈 자리): src descriptor의 FP16 벡터(vlen개)를
  0x8A가 가리키는 **FP32 scale 1개**로 `q = clamp(RNE(x/s))` 후 dst dtype에
  맞게 pack 저장 (INT8: ±127, INT4: ±7). 나눗셈은 내부 FP32
- **`VDEQUANT` 0x1B** (신설): src의 packed INT8/INT4를 0x8A의 FP32 scale로
  `x = q×s` → FP16 저장 (양자화 embedding 행 등)
- zero-point 없음(대칭만), 반올림 RNE — C6 확정값
- **per-row 양자화 시퀀스** (조립식, row마다):
  ① |x|의 최대: `sign_inv`+`max`(0x12) → `reduce_max`(0x19, **seeded**)
  ② `s = absmax ÷ 127` (div imm) → **FP32로 저장** (dst dtype=01)
  ③ `VQUANT` (0x8A → s)
- **256-lane reduce carry**: reduce(0x14/0x19)에 spare 2-bit
  (가안 [27]=carry-in, [26]=hold — drain flag와 동일 의미):
  chunk 내부는 순차 FP32, chunk 간은 **FP32 내부 누적기**로 잇고 마지막
  chunk에서만 FP16 기록. `00`=기존 동작. flat 순차와 동일한 연산 순서 →
  bit-exact 불변식 성립 조건

## 11. ver.08 버그 수정 3건 (v09에서 확정, FP16 불변식 무해 확인)

| 수정 | 내용 | 불변식에 무해한 이유 |
|---|---|---|
| reduce-max seed | 첫 원소로 seed (0 seed 폐지, V3-003) | 기존 codegen은 0-seed를 **회피**(max-fold)하지 실행하지 않음 — 출력 불변. §10 absmax의 전제 |
| immediate 부호 | [23:8]을 **signed int16**으로 해석 (V3-030) | 기존 codegen은 음수 imm을 emit하지 않음 (N3에서 전수 확인 gate) |
| GELU 표준 mode | activation mode에 표준 tanh-GELU 추가, vendor 수식은 legacy 인코딩으로 병존 (V3-004) | 신규 인코딩 — 기존 프로그램 무영향 |

추가 정리: save lane 수 = 직전 연산 결과 길이 (V3-006) — 동작을 문서와 일치시킴.

## 12. 수치 안전성 (검토 완료 사항)

- INT32 누적: 최악 |q|² = 128×128 = 2¹⁴, 최대 K = 12,288 → ≈2²⁷·⁶ ≪ 2³¹ ✅
- INT32→FP32 변환: 2²⁴ 초과 정수는 상대 2⁻²⁴ 반올림 — INT8 양자화 잡음
  대비 무시 가능 (spec에 주석으로만)
- drain 곱(FP32) 후 FP16 RNE 1회 — 반올림 지점은 출구 1곳
- 검증 기준: FP16 mode = 3-모델 golden **bit-exact** (협상 불가) /
  양자화 mode = **우리 산술을 재현한 host reference와 bit-exact** +
  모델 수준 token·logits 지표 (HF와 bit 일치는 반올림 위치가 달라 정의상 비대상)

## 13. 워크스루 — W8A8 한 tile 사슬 (검토용)

act A[64,128](INT8, 행 scale sa[64]) × weight W[128,64](INT8, 열 scale sw[64]):

```
; 사전: VQUANT로 A 생산(§10), W·sw는 host 양자화 후 GLOAD로 SRAM 상주
0x8A ← sa 주소(FP32×64)     0x8B ← sw 주소(FP32×64)
src0: 0x80=A, 0x88 rows=64, 0x89 cols=128 + dtype=10(INT8)
src1: 0x80=W, 0x88 rows=128, 0x89 cols=64 + dtype=10(INT8)

matmul (K-tile 0, MAC=0)   ; INT8×INT8 → INT32 누적 시작
matmul (K-tile 1, MAC=1)   ; K=128 = 64×2, 같은 scale 구간이므로 사슬로
save   (flag 00)           ; drain: FP32(INT32) × sw[n] × sa[m] → RNE → FP16
```

group-wise(W4A8, g=64)라면: K-tile마다 `0x8B ← sw[g]행` 재설정 후
`save(hold=1/carry-in=1)`로 FP32 drain 누적, 마지막 tile에서 `hold=0` 기록.

## 14. 보류 목록 (후속 단계)

- drain에서 INT8 직접 재양자화 출력(fused) — 출력 scale 사전 결정 문제,
  최적화 후보
- async DMA + barrier (double-buffering) / loop·repeat / index 기반 sincos
- activation 정적 scale calibration, SmoothQuant류 host 전처리
- 유닛별 SRAM 분리, descriptor 단일-word화 등 인코딩 압축 (측정 후)
