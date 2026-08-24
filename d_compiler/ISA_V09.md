# ISA ver.09 명세 — Memory 편 (N0 v3, 확정 반영)

> branch `cmodel-v09` · 2026-08-24 (v3 — row/col/stride·dtype·global 단위 확정)
> 이 문서는 **메모리 계층과 데이터 이동만** 확정한다.
> quant/dequant·matmul·vector 연산 세부는 **보류** (§7).
> 원칙: **ver.08 ISA 구조를 유지**하고 메모리 계층에 필요한 최소만 추가한다.

## 0. 확정 결정 (2026-08-24, 사용자 4차)

| # | 항목 | 결정 |
|---|---|---|
| 1 | Global Memory | **32-bit 주소, 단위 = 32-bit 칸** → **16 GiB**. dtype/원소 개념 없음 — DMA 전용 저장소, 크기도 32-bit 칸 개수로만 정의 |
| 2 | SRAM | 8 MiB, **단위 = 4-bit nibble** (유효 24-bit), 주소 field는 기존 ver.08 32-bit 기제 재사용. compute unit은 **SRAM만** 접근 |
| 3 | rows/cols/stride | **element 단위** (2안) — 16-bit field 유지, 값이 ver.08과 동일(FP16 시), 표현 낭비 없음 |
| 4 | dtype | **기존 명령의 spare bit**로 operand별 설정 (신규 상태 레지스터 없음). FP16=0b00 → ver.08 프로그램이 그대로 유효 |
| 5 | DMA | dtype 무관(blind), **동기(blocking)**. GLOAD `0xA0` / GSTORE `0xA8` (ver.08 미사용 영역, load 0x90/save 0x98의 +8 대칭) |

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
| SRAM 시작 주소 (0x80) | **nibble** | 32-bit (2-half) | 물리 위치는 nibble이 정밀도 기준 |
| rows | 행 개수 | 16-bit | 단위 아님 |
| cols·stride (0x88/0x89) | **element 개수** | 16-bit | 행렬 차원 그 자체 — ver.08과 의미·값 동일 |
| vlen (0x82) | element(lane) 개수 | 기존 그대로 | |
| 환산 | global 1칸 = SRAM **8 nibble** | — | 기계 전체의 유일한 환산 관계 |

- 주소 생성: `addr(r,c) = base_nibble + (r×stride + c) × w`,
  w = dtype 폭(nibble) ∈ {FP32:8, FP16:4, INT8:2, INT4:1}
- **정렬 규칙**
  - global: tensor 행은 32-bit 칸 경계에서 시작, 행 길이는 32-bit의 배수
    (FP16 짝수 개 / INT8 4배수 / INT4 8배수 — 64-배수 차원에서 자동 충족)
  - SRAM: tensor 시작 nibble은 자기 dtype 폭의 배수
  - DMA의 SRAM 쪽 주소: **8-nibble(32-bit 칸) 경계** — 기능상 임의 nibble도
    가능하나 HW가 칸 단위로 옮길 때 nibble-shift 배선이 생기므로 규칙으로 고정
    (우리 배치는 어차피 칸 경계만 사용, 실질 제약 0)
  - 위반은 컴파일러/시뮬레이터 오류
- 표현 한계: cols·stride 16-bit = dtype 무관 행당 최대 **65,535 element**
  (현 최대 행 Gemma double-wide 12,288의 5배 여유)

### 1.2 칸 내부 배치 순서 (little-endian 고정)

global↔SRAM 이동 시 32-bit 칸과 nibble 8개의 대응을 한 가지로 못박는다:
**낮은 주소 = 낮은 bit**.

- 칸의 bit `[4k+3 : 4k]` ↔ SRAM nibble `base + k` (k = 0..7)
- 예: FP16 두 값 (a, b)이 든 칸 — a = bit[15:0], b = bit[31:16] → SRAM에서
  a가 nibble `[S, S+4)`, b가 `[S+4, S+8)`. INT4 8개는 값0이 bit[3:0]부터 순서대로
- host 파일도 동일 (numpy little-endian 배열을 그대로 쓰면 일치)

## 2. dtype 설정 — 신규 명령 없이 spare bit

dtype은 **operand 단위** 속성이다 (한 연산 안에서 src0/src1/dst가 다를 수
있음 — 예: W8A16). 전용 설정 명령을 만들면 stale 상태 hazard(V3-025류)가
하나 늘므로, **자기가 설명하는 descriptor와 같은 word에 원자적으로** 싣는다:

| 위치 | 대상 | field |
|---|---|---|
| `0x88`(rows)/`0x89`(cols)의 spare bit 2개 | 해당 matrix operand | `[26:25]` (기존 [29] strided/[28] target과 비충돌 위치, N1에서 bit 자리 최종 확인) |
| `0x82`(vlen)의 spare bit 2개 | vector operand | 동일 인코딩 |

인코딩: `00=FP16, 01=FP32, 10=INT8, 11=INT4`.
**FP16=00** 이므로 spare bit가 전부 0인 ver.08 프로그램은 "dtype=FP16인
유효한 v09 프로그램"으로 그대로 읽힌다 (재해석 규칙의 완결).

- 연산 명령은 operand들의 dtype 조합에 맞는 mode로 동작. **합법 조합표**
  (FP16×FP16, INT8×INT8→requant 등)와 불법 조합 오류 규정은 compute 편(§7)
- 누적기(FP32/INT32)는 주소 공간에 없으므로 ISA field 불요

## 3. DMA 명령 (신규 — 이번 추가의 전부)

dtype·해석 없음. "global의 2D 블록을 SRAM으로(또는 반대로) 원본 그대로 복사."
**동기(blocking)** — DMA 완료까지 다음 명령 미실행 (async+barrier는 후속 §7).

### `GLOAD` (0xA0) — global → SRAM (5 words)

```
w0: [7:0]=0xA0, 나머지 예약(0)
w1: global 시작 주소        (32-bit, 32-bit 칸 단위)
w2: global 행 stride        (32-bit, 칸 단위)      ← wide stride 직접 표현 (V3-027 해소)
w3: SRAM 시작 주소          (32-bit field, nibble 단위, 유효 24-bit, 8-nibble 정렬)
w4: [31:16] rows  [15:0] cols (cols는 32-bit 칸 개수)
```

- 동작: 행 r에 대해 global `[w1 + r×w2 .. +cols)` 칸들을
  SRAM `[w3 + r×(cols×8) ..)` nibble에 §1.2 순서로 복사 (SRAM 쪽 행 연속)
- packed INT8/INT4 blob도 그냥 "칸들" — DMA는 모름
- SRAM 쪽 scatter(stride)는 불요로 판단(흩뿌릴 데이터는 global에 삶) — 확장 여지로 w0 spare bit 예약

### `GSTORE` (0xA8) — SRAM → global (5 words)

`w0: 0xA8` / 이하 GLOAD와 동일 형식 (방향만 반대)

### 오류 규칙

- global/SRAM 범위 초과, SRAM 주소 상위 8-bit ≠ 0 또는 8-nibble 비정렬
  → **즉시 오류 종료** (silent corruption 금지)

## 4. 기존 ver.08 ISA의 재해석 규칙 (인코딩 불변)

| 명령 | ver.08 의미 | v09 재해석 |
|---|---|---|
| `0x80` 주소 (2-half) | G-buffer 원소 주소 | **SRAM nibble 주소** (24-bit → 두 half 모두 emit 필수 — V3-025 관례를 규칙으로 승격) |
| `0x88`/`0x89` rows/cols | 원소 개수 | **element 개수 — 값 그대로** + spare bit에 dtype |
| `0x82` vlen | lane 개수 | 그대로 + spare bit에 dtype |
| `0x90`/`0x98` load/save | G-buffer↔유닛 | **SRAM↔유닛** (인코딩·strided bit 그대로) |
| 연산 전체 (0x01..0x43) | | 그대로 — dtype mode 세부는 compute 편 |
| MAIN/PARTIAL | 접근=PARTIAL, MAIN=stride | 그대로 (b_program류 혼동 방지 명문화) |

- **FP16 이관 공식**: rows/cols/stride/vlen field 값은 ver.08과 **동일**,
  바뀌는 것은 시작 주소뿐 (SRAM 배치 주소, nibble 단위). bit-exact 불변식의
  기계적 근거이자, 기존 codegen 수정 범위의 정의

## 5. 제어 및 host I/O 계약

| opcode | 명령 | 동작 |
|---|---|---|
| 0x00 | NOP | |
| 0xF0 | SNAPSHOT | global 전체(실크기)를 출력 파일에 append (중간 checkpoint) |
| 0xFF | **HALT** | 종료 + global 전체(실크기) 기록 — **유일한 결과 회수 경로** |

- 입력 파일 = global 초기 이미지 (32-bit 칸의 나열, little-endian)
- SRAM은 0 초기화로 시작, host 직접 접근 불가 (GLOAD로만 채움)
- 실행 단위 = 프로그램 1회 (per-invocation), SRAM 상태는 프로그램 간 비유지

## 6. 워크스루 — 메모리 이동만 (검토용)

FP16 행렬 A[2,64] (행 = 32칸)와 packed INT8 행렬 B[64,128] (행당 128값 = 32칸):

```
host 배치: A @ global 0x1000 (행 stride 32칸)
           B @ global 0x2000 (행 stride 32칸)

GLOAD g=0x1000, g_stride=32, sram=0,     rows=2,  cols=32
      ; A → SRAM nibble [0 .. 2×32×8) = [0..512)
GLOAD g=0x2000, g_stride=32, sram=512,   rows=64, cols=32
      ; B(packed) → SRAM nibble [512 .. 512+64×256)

연산 설정 (ver.08 방식 그대로):
  A: 0x80 주소=nibble 0 (두 half), 0x88 rows=2, 0x89 cols=64+dtype=00(FP16)
  B: 0x80 주소=nibble 512,        0x88 rows=64, 0x89 cols=128+dtype=10(INT8)
  ; cols가 곧 행렬 차원 — ver.08과 같은 값. 주소만 nibble

GSTORE (결과 SRAM 위치) → global 0x8000 ...
HALT   ; global 이미지 → 출력 파일, host가 0x8000에서 결과 읽음
```

## 7. 보류 목록 (memory 확정 후 compute 편에서 설계)

- dtype **합법 조합표**와 연산 mode (FP16/W8A16/W8A8), VQUANT/requant, scale operand
- 256-lane reduce의 FP32 내부 누적 + carry 기제
- ver.08 버그 수정 3건 (reduce-max seed, signed imm, 표준 GELU mode) 반영 지점
- **async DMA + barrier** (double-buffering용 — ISA 겉모습 변화 없이 HW가
  지원 가능하도록 동기 semantics만 지금 명시, 실행 모델 확장은 나중)
- descriptor 개편·단일-word 주소 등 v1 초안의 구조 변경은 측정 후 후보
