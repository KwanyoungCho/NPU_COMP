# PLAN — ver.09 자체 C-model 및 ISA 제안 (branch `cmodel-v09`)

> 시작일: 2026-08-22 · 결정 확정: 2026-08-24 (사용자)
>
> 목표: vendor 재현이 아닌 **우리가 설계하는 차기 C-model/ISA**를 만든다.
> 범위: ① SRAM scratchpad 메모리 계층 (**1차: matrix/vector 공유 단일 SRAM**),
> ② quantization — **weight(INT8/INT4 packed) + activation(W8A8, matrix unit
> sandwich)** 을 ISA에 모두 스펙화, 구현은 W8A16 → W8A8 순,
> ③ vector 유닛 256-lane 고정, ④ 이에 맞는 ISA 확정 (ver.08 리뷰 §7.5 반영).
> 산출물은 설계팀에 전달할 **spec 문서 + 동작 C-model + compiler backend + 평가표**다.

## 확정된 결정사항 (2026-08-24, 사용자 · 3차 최종)

| # | 항목 | 결정 |
|---|---|---|
| 1 | Global 주소 | **32-bit 주소, 단위 = 16-bit** (8 GiB). global에는 원소/dtype 개념 없음 — "16-bit 칸이 나열된 저장소". dtype 해석은 연산 유닛에서만 |
| 2 | SRAM 구성 | **matrix/vector 공유 단일 SRAM(scratchpad), 8 MiB**. 주소 단위 = **4-bit nibble** (FP32/FP16/INT8/INT4 공존을 dtype 독립 주소 산술로). 32-bit 주소 field에 유효 24-bit — **기존 ver.08 주소 기제 그대로 사용**, 여유 bit는 유지(부족 시 그때 축소). 유닛별 분리는 후속 옵션(§2.3) |
| 3 | DMA | GLOAD/GSTORE는 **dtype 무관(blind)** — 16-bit 단위 원본 이동만. 환산 규칙: global 1칸 = SRAM 4 nibble |
| 4 | 기존 ISA | 주소설정·load/save·연산 **ver.08 인코딩 유지, 주소/stride/cols만 nibble 재해석** (FP16 기준 ×4 환산 — compiler 기계적 변환) |
| 5 | 작업 순서 | **memory 관련 이슈 먼저 확정, quant/dequant·matmul 설계는 보류** (3차 결정). W8A8 방향성(2차 결정)은 유지하되 세부는 compute 설계 단계로 이월 |
| 6 | loop/repeat | v09 범위 밖 (별도 단계; index-sincos op도 이와 한 묶음) |

> 3차 결정으로 **강등/폐기된 v1 초안 항목**: 단일-word SRAM 주소(op2+addr22),
> 2-word GLOAD 주소 형식, descriptor 3요소 재정의(MAIN/PARTIAL 폐지) —
> ver.08 구조 유지 원칙에 따라 "측정 후 후속 최적화 후보"로 이동 (ISA_V09.md §6).

nibble 주소 + packed 저장의 함의:

- 모든 dtype이 정수 개 nibble 칸(FP32=8, FP16=4, INT8=2, INT4=1)을 차지 →
  주소 산술이 dtype 독립, packing straddle 미발생
- 정렬 규칙: tensor 시작 nibble 주소는 자기 dtype 폭의 배수 — 64-배수
  차원에서 자동 충족, 위반은 컴파일러 오류 (spec 명문화)
- ver.08 주소 산술의 ×4 배율일 뿐이라 **bit-exact 불변식(§2.4)에 가장 유리**

---

## 0. 불변 원칙

1. **ver.08 세계는 동결** — vendor `a.out`, `mysim_0818.cpp`, `isa_0818.py`,
   `backend_0818.py`는 한 줄도 바꾸지 않는다. 기존 36개 테스트와 세 모델
   golden은 이 branch에서도 상시 회귀 gate다.
2. v09는 **전부 새 아티팩트**: `_poc/mysim_v09.cpp`, `npu_compiler/isa_v09.py`,
   `npu_compiler/backend_v09.py`, `backend="v09"` target.
3. **핵심 수치 불변식**: v09의 FP16(비양자화) 모드는 0818 결과와
   **bit-exact 동일**해야 한다 (§2.4). 성립하면 세 모델 golden이 v09 검증에
   공짜로 재사용된다.
4. 단계마다 상세 commit + 측정. 새 simulator는 perf counter를 처음부터 내장
   (word 수 / DMA bytes / SRAM 점유 — latency 표 수신 시 비용 모델로 직결).

## 1. 메모리·유닛 아키텍처 (모델링 대상)

```
        ┌─────────────────────────────────────────────┐
        │  Global Memory (DRAM/HBM 모델)               │
        │  32-bit 주소, 단위 16-bit (8 GiB)            │
        │  내용물 해석 없음 — 그냥 16-bit 칸 나열       │
        └──────────────────┬──────────────────────────┘
                    DMA: GLOAD / GSTORE  (dtype 무관 — 원본 이동만)
        ┌──────────────────▼──────────────────────────┐
        │  Unified SRAM (공유 scratchpad, 8 MiB)       │  ← SW 관리, 결정적
        │  FP32/FP16/INT8/INT4 공존                    │     4-bit nibble 주소
        └─────────┬───────────────────┬───────────────┘     (32-bit field, 유효 24-bit)
   ┌──────────────▼─────────────┐ ┌───▼──────────────────┐
   │ [Quant] act FP16→INT8      │ │ Vector Unit          │
   │  Matrix Unit 64×64          │ │ 256 lanes, FP16/FP32 │
   │  mode FP16: FP16×FP16→FP32 │ │ (norm·softmax·rope + │
   │  mode INT8: INT8×INT8→INT32│ │  QUANT 보조: absmax→ │
   │ [Requant] acc×scale→FP16   │ │  scale→round→pack)   │
   └────────────────────────────┘ └──────────────────────┘
```

- **compute 유닛은 SRAM만 접근** (global은 DMA 전용) — 계층 명확화
- **quant/requant sandwich는 matrix unit에만** (LLM W8A8 표준형).
  vector 연산(norm/softmax/RoPE)은 FP16 유지 — CNN식 전면 정수화는
  LLM에서 정확도 불가
- SRAM은 dtype 공존: activation FP16, weight는 **packed INT8 그대로 상주**
  (SRAM 유효 용량 2×, DMA는 변환 없이 단순 이동). weight dequant는
  matrix 입구 feeder에서 (W8A16 mode) 또는 불필요 (W8A8 mode — INT8 직결)
- 주소는 nibble 단위, **ver.08의 2-half 주소 설정 기제 유지** (stale-high
  위험은 "두 half 항상 emit" 관례를 필수 규칙으로 명문화해 관리)
- 공유 SRAM: allocator/DMA 엔진 1개. 유닛별 분리는 §2.3 후속 옵션
- vector 길이: vlen ∈ [1, 256] (초과는 backend가 chunk)

## 2. ISA v09 — ver.08 대비 변경 목록

### 2.1 신규 — memory 편 (이번 범위의 본체, ISA_V09.md 확정)
- `GLOAD` (5 words): global→SRAM DMA. w0 opcode / w1 global 주소 32-bit /
  w2 global 행 stride 32-bit (wide stride 직접 표현 — V3-027 해소) /
  w3 SRAM nibble 주소 / w4 rows|cols(16-bit 단위 개수).
  **dtype 무관 원본 이동** — packed blob도 그냥 "16-bit 칸들"
- `GSTORE` (5 words): SRAM→global, 동일 형식 역방향
- 종료/저장: `HALT` 신설(종료 + global 전체 기록 = 유일한 결과 회수 경로),
  `SNAPSHOT`(0xF0)은 중간 checkpoint 전용
- 범위 초과 접근(global/SRAM 모두)은 **오류** (silent corruption 금지)
- 기존 주소설정/load/save/연산: **ver.08 인코딩 그대로, SRAM nibble 재해석**

### 2.2 compute 편 — **보류** (3차 결정: memory 확정 후 별도 설계)

아래는 방향성 합의만 있고 세부(인코딩·mode bit 위치)는 이월:

- matrix dtype mode (FP16 / W8A16 feeder-dequant / W8A8 INT8×INT8→INT32),
  requant 출력단, `QUANT` vector 명령
- 256-lane reduce의 FP32 내부 누적 + carry-in/writeout 기제
- ver.08 버그 수정: reduce-max 첫 원소 seed(V3-003), 표준 tanh-GELU
  mode(V3-004), signed int16 immediate(V3-030), save lane 수 =
  직전 결과 길이(V3-006)
- descriptor 3요소 재정의(MAIN/PARTIAL 폐지)와 단일-word 주소는
  **후속 최적화 후보로 강등** (ver.08 구조 유지 원칙)

### 2.3 명시적으로 미룸 (v09 범위 밖, spec에 후보로만 기재)
- **유닛별 SRAM 분리** (mSRAM/vSRAM + 유닛 간 전송) — 공유로 시작,
  N7 접근 통계로 분리 필요성을 데이터로 판단
- loop/repeat 명령 (제어 word 93% 문제의 근본 해법이나 실행 모델 변경이 큼)
- **index 기반 `sincos(pos:int32, freq:fp16)` op** (§7.3-⑨a 계측): 각도의
  FP16 운반 한계를 op 내부 FP32로 해소 — host 개입이 스텝마다 있는 현 실행
  모델에서는 host cos/sin row 전달로 충분하므로 **loop 도입과 한 묶음**
- activation의 **정적(static) scale calibration** 경로 — 동적 per-token이
  기본, 정적은 후보로만
- 주소 확장(40-bit 등) — FP16 weight 상주 full-model이 필요해지는 시점 항목
- cycle-accurate timing (counter 통계까지만; latency 표 수신 후)

### 2.4 수치 불변식의 설계 근거 (FP16 모드 ≡ 0818 bit-exact)
- 저장 FP16 / 연산 FP32 / 저장 시 RNE — 0818 계약 유지 (실측 §7 검증됨)
- 주소 체계 = ver.08 그대로에 nibble ×4 배율만 적용 → 주소 산술 동형
- 256-lane chunking이 순서를 바꾸지 않도록: **reduce는 chunk 내부 순차 +
  chunk 간 FP32 carry를 in-order 누적** = 기존 flat 순차와 동일 순서
- matmul K 누적: tile 순서 유지, FP32 누적기 유지
- SRAM staging은 FP16 값의 이동일 뿐 반올림 지점을 추가하지 않음
  (dequant-on-load는 FP16 모드에서 비활성 → 경로 자체가 동일)

## 3. Quantization 설계 (weight + activation, 2026-08-24 2차 결정 반영)

**동작 모드 3단** (spec은 전부, 구현은 순차):

| Mode | weight | activation | matrix 연산 | 비고 |
|---|---|---|---|---|
| FP16 | FP16 | FP16 | FP16×FP16→FP32 | 기존 — **0818 bit-exact 불변식 유지** |
| **W8A16** (N6a) | INT8 packed (SRAM 상주) | FP16 | feeder dequant → FP16×FP16→FP32 | LLM weight-only 표준 (GPTQ/AWQ류) |
| **W8A8** (N6b) | INT8 packed | INT8 (per-token 동적) | INT8×INT8→INT32 → requant | quant/requant sandwich가 여기서 발동 |

- weight 형식: **INT8 per-output-channel symmetric**, 원소당 2개 packed
  (2차: INT4 group-wise g=64~128, 원소당 4개). 호스트 quantizer
  (`make_quant_weights.py`)가 packed blob + scale 벡터 생성
- activation 형식: **per-token(행) 동적 symmetric** — 정적 per-tensor는
  LLM activation outlier로 품질 저하가 알려져 있어 기본에서 제외.
  scale 산출은 device의 `QUANT` 명령(absmax 경유) 또는 host(검증용 참조)
- requant 산술: INT32 누적 × (w_scale[col] × a_scale[row]) → FP16 (RNE).
  FP32 곱은 requant 유닛 내부에만 (기존 FP32 누적기와 동일한 위치)
- 정렬 규칙: packed tensor의 행/타일 시작은 원소 경계 — 차원이 64 배수라
  자동 충족 (spec에 명문화)
- 검증 사다리:
  ① unpack/dequant/QUANT/requant 단위 테스트 (host 계산과 bit-exact)
  ② layer별 FP16 대비 오차 계측 (W8A16, W8A8 각각)
  ③ 세 모델 3-token greedy — **수용 기준: W8A16은 token 일치 기대,
  W8A8은 token 일치 또는 불일치 시 logits 지표와 함께 문서화**
  (양자화 모드는 golden bit-일치를 요구하지 않는 유일한 모드)

## 4. Compiler 변경 (전부 v09 신규 파일, 기존 경로 불변)

- `backend_v09.py`: 동일 Relax 파이프라인(`passes.npu_pipeline` 재사용) 소비.
  codegen이 **2단 메모리 계획** 수행:
  - global memplan: 기존 `memplan` 재사용 (packed weight는 논리 shape /
    저장 원소 수 구분 추가)
  - SRAM allocator: 프로그램 내 tile staging slot (정적, double-buffer)
  - matmul: A/B tile GLOAD → 연산/누적 → C GSTORE
  - vector: 256-lane chunk 단위 GLOAD → 연산 → GSTORE
  - 공유 SRAM이므로 producer-consumer 배치 최적화 여지 (1차는 단순 staging,
    최적화는 N7 계측 후)
- `analyze_isa_stats.py` 확장: v09 프로그램의 **word 수 / DMA bytes /
  SRAM 점유 / 유닛별 연산 수** — 전·후 비교표의 축

## 5. 단계 계획 (각 단계 = commit + gate)

| 단계 | 내용 | Gate |
|---|---|---|
| **N0** | ISA v09 spec — **memory 편 v2 작성 완료(2026-08-24, 3차 결정 반영)**. compute 편은 memory 승인 후 별도 N0' | 사용자 리뷰·승인 |
| **N1** | `mysim_v09.cpp` 골격: global(16-bit 단위)/공유 SRAM(nibble 단위) 메모리 객체, decode 루프, HALT/SNAPSHOT, 경계 검사, perf counter | 단위 테스트 |
| **N2** | 데이터 이동: GLOAD/GSTORE (5-word 형식) + ver.08 load/save·주소설정의 nibble 재해석 | `isa_v09.py` round-trip + 이동 단위 테스트 |
| **N0'** | compute 편 spec (dtype mode·QUANT·requant·reduce carry·버그수정 3건) | 사용자 리뷰·승인 |
| **N3** | 연산: vector 256-lane 전 연산 + matrix 64×64 (N0' 반영) | op별 numpy FP16-step reference와 bit-exact |
| **N4** | `backend_v09.py` + SRAM staging codegen | **proxy layer가 0818 결과와 bit-exact** (불변식 1차 증명) |
| **N5** | 세 모델 golden을 v09 FP16 모드로 실행 | **token+logits가 기존 golden과 bit-exact** |
| **N6a** | W8A16: quantizer(pack) + feeder dequant + INT8 weight 상주 실행 | §3 사다리 ①②③ |
| **N6b** | W8A8: `QUANT` 명령 + INT8 MAC(INT32 누적) + requant | §3 사다리 ①②③ |
| **N7** | 통계·문서화: v09 vs 0818 비교표(word/DMA/SRAM) + 공유 SRAM 접근 통계(분리 필요성 판단 자료), spec 최종판 | `report/report_v09.md` |

예상 규모: N1~N3 = C-model 신작(~1,000줄), N4 = compiler 최대 작업(staging
codegen). N5까지가 "동작 동일 증명", N6부터가 신기능.

## 6. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| SRAM staging으로 프로그램 word 증가 (DMA 명령 추가) | GLOAD 5-word가 ver.08의 strided load 대비 크지 않음(주소 4-word + rows/cols 2-word + load 1-word = 7-word 상당). N4에서 `analyze_isa_stats` 전·후 비교로 정량 관리 (DMA는 bytes 축 별도 계상). 단일-word 주소 등 인코딩 압축은 측정 후 후보 |
| 공유 SRAM의 유닛 간 경합 | 1차 모델은 기능 검증이라 무관. N7 접근 통계로 분리 필요성을 **데이터로** 판단 (§2.3 옵션) |
| 256-lane chunking이 수치 변경 | §2.4 순서 보존 규칙 spec 명문화 + N3 bit-exact 검증 |
| packed 접근의 경계 오류 | 정렬 규칙 spec 명문화 (64-배수 차원에서 자동 충족) + GLOAD 경계 검사 + unpack 단위 테스트 |
| INT8에서 greedy token 변화 | 수용 기준 사전 정의(§3), per-channel → group-wise 세분화 여지 |
| W8A8의 LLM activation outlier | per-token 동적 scale을 기본으로 채택. 부족 시 smoothing(가중치-활성 스케일 재배분)류 기법을 host 전처리로 후보화 |
