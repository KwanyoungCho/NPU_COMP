# ISA ver.09 명세 초안 (N0 — 리뷰용)

> branch `cmodel-v09` · 2026-08-24
> 근거: ver.08 전수 실측(report_0818.md §7) + 확정 결정(PLAN_V09.md)
> 상태: **초안** — 사용자 승인 후 고정, N1(mysim_v09.cpp)부터 이 문서가 계약

## 0. 기계 모델

| 구성 | 사양 |
|---|---|
| Global Memory | 32-bit 주소, **단위 = 16-bit 원소**, 최대 8 GiB. 실제 크기 = 입력 파일 크기(동적). FP16 tensor + packed INT8/INT4 blob |
| Unified SRAM | **8 MiB = 4,194,304 원소 → 22-bit 원소 주소**. matrix/vector 공용 scratchpad(SW 관리). FP16 activation과 packed INT8 weight 공존 |
| Matrix Unit | 64×64 PE. 누적기 버퍼 64×64 (FP32 또는 INT32, mode별). 연산은 SRAM만 접근 |
| Vector Unit | **256 lanes**, 내부 FP32. vlen ∈ [1,256]. SRAM만 접근 |
| 명령 word | 32-bit little-endian, `[7:0]` = opcode (ver.08 관례 유지) |
| 수치 계약 | 저장 FP16(RNE) / 연산 FP32 / matmul·reduce 누적 FP32(INT8 mode는 INT32). inf/NaN IEEE 전파. **FP16 mode는 ver.08과 bit-exact** (PLAN §2.4) |
| 실행 종료 | `HALT` 명령으로만 종료. **HALT 시 global memory 전체를 출력 파일로 기록** (ver.08의 "결과 유실" 함정 제거). `SNAPSHOT`은 중간 checkpoint 덤프(선택) |
| 오류 | global/SRAM 범위 초과, vlen>256, 미정의 opcode/mode → **즉시 오류 종료** (silent corruption 금지) |

## 1. 상태(descriptor) 모델 — ver.08 대비 대폭 단순화

operand 슬롯 4개: `SRC1(0) · SRC2(1) · DST(2) · SCALE(3)`. 슬롯별 상태는 3개뿐:

```
addr   : 22-bit SRAM 원소 주소      (SADDR, 단일 word)
shape  : rows(11) × cols(11)        (SSHAPE, 단일 word)
stride : 22-bit 행 간격(원소)        (SSTRIDE, 단일 word; 기본값 = cols)
```

- ver.08의 MAIN/PARTIAL 이중 구조, low/high half 분할, 죽은 상태
  (main_addr/main_rows) **전부 폐지** → V3-025류 hazard 원천 제거
- vector 연산은 `addr`(+`vlen`)만 소비 — shape/stride 비적용을 spec에 명문화
- **DMA(GLOAD/GSTORE)는 무상태** — 필요한 모든 것을 명령 word에 내장 (§3)

### 1.1 설정 명령 인코딩

| 명령 | 인코딩 (32-bit) |
|---|---|
| `SADDR`  | `[31:30] op` `[29:8] addr22` `[7:0]=0x80` |
| `SSHAPE` | `[31:30] op` `[29:19] rows11` `[18:8] cols11` `[7:0]=0x88` |
| `SSTRIDE`| `[31:30] op` `[29:8] stride22` `[7:0]=0x89` |
| `VLEN`   | `[16:8] n (1..256)` `[7:0]=0x82` |

## 2. 연산 명령 — "load 단계 제거"가 핵심 구조 변화

ver.08은 `주소설정 → load → 연산 → 주소설정 → save`였다 (제어/이동이 word의 93%).
v09는 **연산이 descriptor를 통해 SRAM을 직접 읽고, vector는 결과를 DST에 직접
쓴다** — load/save 명령 자체가 사라진다 (matrix의 누적기 flush만 예외).

### 2.1 Vector 명령 (`SRC1[,SRC2] → DST`, vlen개, 256 초과는 compiler가 chunk)

| opcode | 명령 | 비고 |
|---|---|---|
| 0x01/02/0A/0B | VADD/VSUB/VMUL/VDIV | `[31:30] mode`(IMM/SCALAR/VECTOR), `[23:8] imm` — **imm은 signed int16** (V3-030 수정). SCALAR mode는 SRC2.addr의 값 사용 (ver.08 0x15 주소겸용 hazard 제거) |
| 0x0E/0F | VSQRT/VEXP | 단항 |
| 0x12 | VMINMAX | `[28]` min/max |
| 0x14 | VREDSUM | FP32 순차 누적, 결과 1원소. `[27]=1`이면 **carry-in**: DST 기존값에 이어 누적 (256 초과 reduce의 chunk 간 in-order carry — bit-exact 불변식의 근거) |
| 0x19 | VREDMAX | **seed = 첫 원소** (V3-003 수정). `[27]` carry-in 동일 |
| 0x15 | VBROADCAST | scalar(SRC2.addr 또는 imm) → vlen개 |
| 0x16/0x17/0x18 | VSIGNINV/VCOPY/VCOSSIN | 0x18 `[27]` cos/sin |
| **0x1A** | **VQUANT** (신규) | SRC1(FP16, vlen) → absmax→scale 산출 → RNE 반올림 → **INT8 packed를 DST에**, **scale(FP16) 1원소를 SCALE.addr에** 기록. per-token 동적 양자화의 생산자 |
| **0x1B** | **VDEQUANT** (신규) | packed INT8(SRC1) × scale(SCALE.addr) → FP16 DST. 검증·보조용 |

### 2.2 Matrix 명령

| opcode | 명령 | 비고 |
|---|---|---|
| 0x42 | MMUL | `SRC1[r,k] × SRC2[k,c] → 누적기[r,c]`. `[27]` MAC(누적 이어가기). **`[29:28]` dtype mode**: `00` FP16(FP32 누적) · `01` **W8A16**(SRC2가 packed INT8, feeder가 SCALE.addr의 per-channel scale로 dequant 후 FP16 연산) · `10` **W8A8**(양쪽 packed INT8, INT32 누적) |
| 0x40 | MADD | elementwise 행렬 덧셈(+activation). `[26:25]` activation: off / SiLU / **GELU-std(신규, 표준 tanh식)** / GELU-legacy(vendor식 `x·sigmoid(2x)`) |
| 0x98 | **MSAVE** | 누적기 → SRAM(DST). `[29:28]` mode: `00` FP32→FP16(RNE) · `01` **requant**: INT32 × (SCALE.addr의 w_scale[col] × SCALE2 영역의 a_scale[row]) → FP16. scale 벡터 주소는 SCALE 슬롯 descriptor로 지정 |

- 누적기 semantics: MMUL이 K-tile 순서로 누적(FP32/INT32), MSAVE가 유일한
  flush 지점 → 반올림 지점이 ver.08과 동일 (bit-exact 불변식)
- transpose: SSTRIDE 기반 strided 접근으로 표현 (ver.08 strided load의 계승,
  "모든 접근 형태가 모든 유닛에 공급 가능" 계약을 spec에 명문화)

### 2.3 제거되는 ver.08 명령 (전수 실측 §7.2-B 근거)

`0x08 logical · 0x09 shift · 0x0C muladd · 0x0D vmove · 0x11 compare ·
0x13 convert · 0x41 msub · 0x43 mmove` — 3-model 전 범위에서 사용 0회.
(정수 계열의 역할은 VQUANT/VDEQUANT가 목적 특화로 대체)

## 3. DMA 명령 (무상태, multi-word)

### `GLOAD` — global → SRAM (5~6 words)

```
w0: [7:0]=0x90  [9:8] dtype(00 FP16 / 01 INT8 / 10 INT4)  [10] dq(scale 첨부)
w1: global 시작 주소 (32-bit 원소 단위)
w2: global 행 stride (32-bit)         ← V3-027 해결: vocab 262144 stride 직접 표현
w3: [31:10] SRAM 목적 주소(22)  [9:0] 예약
w4: [31:16] rows  [15:0] cols   (논리 원소 수 기준)
w5: (dq=1일 때) per-channel scale 벡터의 global 주소 (32-bit)
```

- dtype=INT8/INT4: **packed 그대로 이동** (SRAM에 INT 상주). `dq=1`이면
  적재하며 FP16으로 풀기(디버그/W-only 검증 경로)
- packed 이동 시 cols는 논리 값 수 — 원소 환산(÷2, ÷4)은 HW/모델 내부.
  정렬 규칙: 행 시작은 원소 경계 (64-배수 차원에서 자동 충족)

### `GSTORE` — SRAM → global (5 words, FP16 전용)

w0(0x91) / w1 global 주소 / w2 global stride / w3 SRAM 주소 / w4 rows·cols

**효과**: ver.08의 "region 설정 8 word + load" 패턴이 통째로 대체된다.
2D 블록 하나의 이동이 상태 오염 위험 없이 5-6 word로 완결.

## 4. 제어

| opcode | 명령 | |
|---|---|---|
| 0x00 | NOP | |
| 0xF0 | SNAPSHOT | global 전체를 출력 파일에 append (중간 checkpoint, 선택) |
| 0xFF | **HALT** | 실행 종료 + global 전체를 출력 파일로 기록 (유일한 정상 종료) |

## 5. Word 수 영향 추정 (ver.08 실측 대비)

| 패턴 | ver.08 | v09 | 근거 |
|---|---:|---:|---|
| matmul K-tile 1회 | ~13 words (region×2=8~16 + load×2 + mul) | **~7** (SADDR×2 + SSHAPE 변경분 + MMUL) — 불변 descriptor는 재설정 불필요 | load 명령 제거, 단일-word 설정 |
| vector 2항 연산 1회 | ~8 (addr 2×2 + vlen + load×2 + op + addr 2 + save) | **~5** (SADDR×3 + VLEN + op) | load/save 제거, 단일-word 주소 |
| 2D 블록 이동 | ~9 (region 8 + load) | **5~6** (GLOAD) | 무상태 DMA |

ver.08 실측에서 제어·이동이 93%였으므로, 보수적으로도 **프로그램 word 총량
40~50% 절감**이 기대치. N4에서 `analyze_isa_stats`로 실측 비교(word + DMA
bytes + SRAM 점유)가 gate.

## 6. 미해결/리뷰 요청 사항 (사용자 확인)

1. `SSHAPE` rows/cols 11-bit(≤2047) — DMA는 w4에서 16-bit라 무관, compute
   tile은 64 이하라 여유. 이 배분으로 충분한가?
2. MSAVE의 a_scale[row] 벡터 위치 지정: SCALE 슬롯 하나로 w_scale만 지정하고
   a_scale은 "DST 직전 원소들" 같은 관례로 둘지, **operand 슬롯을 5개로 늘릴지**
   (현재 안: SCALE.addr = w_scale, SCALE.addr+stride = a_scale 관례) — N1에서
   구현하며 확정 제안
3. INT4의 group-wise scale(g=64~128) 인코딩은 N6b 시점에 추가 (현재 spec은
   INT8 per-channel까지)
