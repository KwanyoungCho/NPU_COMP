# ISA ver.09 명세 — Memory 편 (N0 v2, 리뷰용)

> branch `cmodel-v09` · 2026-08-24 (v2 — 사용자 방향 확정 반영)
> 이 문서는 **메모리 계층과 데이터 이동만** 확정한다.
> quant/dequant·matmul·vector 연산 세부는 **보류** (§6) — 메모리 확정 후 별도 설계.
> 원칙: **ver.08 ISA 구조를 유지**하고 메모리 계층에 필요한 최소만 추가한다.

## 0. 확정 결정 (2026-08-24, 사용자)

| # | 항목 | 결정 |
|---|---|---|
| 1 | Global Memory 주소 | **32-bit 주소, 단위 = 16-bit** (8 GiB). "원소/dtype" 개념 없음 — 그냥 16-bit 칸이 나열된 저장소 |
| 2 | SRAM 주소 | **32-bit 주소 field, 단위 = 4-bit nibble**. 8 MiB = 2²⁴ nibble (상위 8-bit는 여유 — 기존 32-bit 주소 기제를 그대로 사용, 부족해지면 그때 축소) |
| 3 | DMA | **dtype 무관(blind)** — global↔SRAM 사이를 16-bit 단위 원본 그대로 이동. 변환·해석 없음 |
| 4 | dtype 해석 | **연산 유닛에서만** — FP32(8nib)/FP16(4)/INT8(2)/INT4(1)가 SRAM에 공존, dtype field가 값 폭을 알려줌 (세부는 보류 §6) |
| 5 | 기존 ISA | 주소설정·load/save·연산 전부 **ver.08 구조 그대로, 대상만 SRAM** (주소 단위만 nibble로 재해석) |

## 1. 메모리 모델

```
[host 입력 파일] ──적재──▶ ┌────────────────────────────────┐
                          │ Global Memory                   │  32-bit 주소, 16-bit 단위
                          │ 최대 8 GiB (실크기 = 파일 크기)   │  내용물 해석 없음(그냥 비트들)
                          └───────────┬────────────────────┘
                         GLOAD / GSTORE (dtype-blind 원본 이동)
                          ┌───────────▼────────────────────┐
                          │ SRAM  8 MiB                     │  32-bit 주소 field, 4-bit 단위
                          │ FP32/FP16/INT8/INT4 공존         │  (유효 24-bit, 상위는 0 필수)
                          └───────────┬────────────────────┘
                    기존 ver.08 방식의 주소설정/load/save/연산 (SRAM만 접근)
                          ┌───────────▼────────────────────┐
                          │ Matrix 64×64 / Vector 256-lane  │  dtype 해석은 여기서만
                          └────────────────────────────────┘
```

### 1.1 단위 규칙 (이 문서의 핵심)

| 값의 종류 | 단위 | 근거 |
|---|---|---|
| Global 주소·stride·개수 | **16-bit 칸** | 32-bit로 8 GiB 확보 (nibble이면 2 GiB뿐이라 불가) |
| SRAM 주소·stride·cols | **nibble** | 최소 dtype(INT4)=1 nibble → 모든 dtype이 정수 개 칸을 차지, 주소 산술이 dtype 독립 |
| rows | 행 개수 | 단위 아님 (양쪽 공통) |
| 환산 | global 16-bit 1칸 = SRAM 4 nibble | DMA가 암묵 적용하는 유일한 관계 |

- 정렬 관례: SRAM에서 tensor의 시작 nibble 주소는 **자기 dtype 폭의 배수**
  (FP16→4, INT8→2, FP32→8). 행이 통째 칸으로 구성되도록 배치 (64-배수
  차원에서 자동 충족). 위반은 컴파일러 오류
- 크기 한계 참고: ver.08 형식의 16-bit cols/stride field를 nibble 단위로
  쓰면 행 폭 최대 65,535 nibble = FP16 16,383개 — 현 workload 최대 행
  (Gemma double-wide 12,288 = 49,152 nibble)까지 수용. 초과 필요 시 확장 항목

## 2. DMA 명령 (신규 — 이번 추가의 전부)

dtype·해석 없음. "global의 2D 블록을 SRAM으로(또는 반대로) 원본 그대로 복사."

### `GLOAD` — global → SRAM (5 words)

```
w0: [7:0]=0x90(신설 opcode 번호는 N1에서 기존과 충돌 없게 확정), 나머지 예약(0)
w1: global 시작 주소        (32-bit, 16-bit 단위)
w2: global 행 stride        (32-bit, 16-bit 단위)   ← wide stride 직접 표현 (V3-027 해소)
w3: SRAM 시작 주소          (32-bit field, nibble 단위, 유효 24-bit)
w4: [31:16] rows  [15:0] cols (cols는 16-bit 단위 개수)
```

- 동작: 행 r에 대해 global `[w1 + r×w2 .. +cols)` 의 16-bit 칸들을
  SRAM `[w3 + r×(cols×4) ..)` nibble에 그대로 복사 (SRAM 쪽은 행 연속 배치)
- packed INT8/INT4 blob도 그냥 "16-bit 칸들"로 이동 — DMA는 모름

### `GSTORE` — SRAM → global (5 words)

`w0: 0x91` / 이하 GLOAD과 동일 형식 (방향만 반대)

### 오류 규칙

- global/SRAM 범위 초과, SRAM 주소 상위 8-bit ≠ 0 → **즉시 오류 종료**
  (silent corruption 금지)

## 3. 기존 ver.08 ISA의 SRAM 재해석 규칙

- `0x80`(주소, low/high 2-half) `0x88/0x89`(rows/cols) load/save 및 모든
  연산: **인코딩·의미 불변, 주소·cols·stride 값만 nibble 단위**
- FP16 텐서 기준 환산: ver.08 원소 주소 × 4 = v09 nibble 주소 —
  **compiler가 곱하기만 하면 기존 codegen이 그대로 동작** (bit-exact 불변식의
  기계적 근거)
- 2-half 주소 설정의 stale-high 위험(V3-025)은 구조 유지에 따라 잔존 —
  "**두 half를 항상 모두 emit**" 관례를 필수 규칙으로 명문화 (현 compiler 관례)
- MAIN/PARTIAL 의미도 ver.08 그대로 (접근은 PARTIAL, MAIN은 stride) —
  b_program류 혼동 방지를 위해 spec에 명시

## 4. 제어 및 host I/O 계약

| opcode | 명령 | 동작 |
|---|---|---|
| 0x00 | NOP | |
| 0xF0 | SNAPSHOT | **global 전체**를 출력 파일에 append (중간 checkpoint) |
| 0xFF | **HALT** | 실행 종료 + global 전체를 출력 파일로 기록 (**유일한 정상 종료·결과 회수 경로**) |

- 입력 파일 = global 초기 이미지 (16-bit little-endian 나열)
- SRAM은 0으로 초기화되어 시작, host가 직접 채울 수 없음 (GLOAD로만)

## 5. 워크스루 — 메모리 이동만 (검토용)

FP16 행렬 A[2,64]와 packed INT8 행렬 B[64,128](행당 128값 = 16-bit 64칸):

```
host 배치: A @ global 0x1000 (행 stride 64칸)
           B @ global 0x2000 (행 stride 64칸 — packed라 절반)

GLOAD 0x1000, stride 64, → sram nibble 0,     rows 2,  cols 64
      ; A → SRAM nibble [0 .. 2×64×4) = [0..512)
GLOAD 0x2000, stride 64, → sram nibble 512,   rows 64, cols 64
      ; B(packed) → SRAM nibble [512 .. 512+64×256)

이후 연산(ver.08 방식): A의 SADDR = nibble 0, B의 SADDR = nibble 512
  — FP16 A의 열 c는 nibble 0 + 4c, INT8 B의 값 c는 nibble 512 + 2c (유닛이 dtype으로 해석)

GSTORE (결과 SRAM 위치) → global 0x8000 ...
HALT   ; global 이미지가 출력 파일로 — host가 0x8000에서 결과 읽음
```

## 6. 보류 목록 (메모리 확정 후 별도 설계)

- 연산 유닛의 dtype field 위치/인코딩, W8A16/W8A8 mode, VQUANT/requant
- 256-lane vector의 reduce carry 기제 (FP32 내부 누적 — 방향은 합의됨)
- ver.08 버그 수정 3건(reduce-max seed, signed imm, 표준 GELU) 반영 지점
- descriptor 개편·단일-word 주소·load/save 제거 등 v1 초안의 구조 변경은
  **측정 후 후속 최적화 후보로 강등**
