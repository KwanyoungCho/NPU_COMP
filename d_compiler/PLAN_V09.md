# PLAN — ver.09 자체 C-model 및 ISA 제안 (branch `cmodel-v09`)

> 시작일: 2026-08-22 · 결정 확정: 2026-08-24 (사용자)
>
> 목표: vendor 재현이 아닌 **우리가 설계하는 차기 C-model/ISA**를 만든다.
> 범위: ① SRAM scratchpad 메모리 계층 (**1차: matrix/vector 공유 단일 SRAM**),
> ② quantization (weight-only INT8, packed 저장, dequant-on-load),
> ③ vector 유닛 256-lane 고정, ④ 이에 맞는 ISA 확정 (ver.08 리뷰 §7.5 반영).
> 산출물은 설계팀에 전달할 **spec 문서 + 동작 C-model + compiler backend + 평가표**다.

## 확정된 결정사항 (2026-08-24, 사용자)

| # | 항목 | 결정 |
|---|---|---|
| 1 | Global 주소 | **32-bit, 주소 단위 = 16-bit 원소(FP16 1개)** — ver.08과 동일 체계, 공간 8 GiB |
| 2 | SRAM 구성 | **matrix/vector 공유 단일 SRAM(scratchpad)** 먼저. 유닛별 분리는 후속 옵션(§2.3) |
| 3 | Quantization | **weight-only**, **packed 저장** (16-bit 원소당 INT8×2 / INT4×4), INT8 per-output-channel symmetric 먼저 |
| 4 | loop/repeat | v09 범위 밖 (별도 단계; index-sincos op도 이와 한 묶음) |

주소 단위 16-bit + packed 저장의 함의:

- packing은 이 주소 체계에서 **필수** (원소당 INT8 1개면 절반 낭비 → 용량 절감 소멸)
- **정렬 문제 없음**: 모든 tensor 차원이 64 배수이고 packing 계수(2, 4)가 64를
  나누므로, 행·tile 경계가 항상 원소 경계에 정렬 — straddle 미발생.
  spec에 정렬 규칙("packed tensor의 행 시작은 원소 경계", 우리 shape에서 자동
  충족)을 명문화
- 추가 복잡도는 국소적 3곳: host quantizer의 pack(numpy view), GLOAD의
  unpack 루프(원소당 2~4값 + scale 곱), memplan의 논리 shape/저장 원소 수 구분
- ver.08과 같은 주소 산술이라 **bit-exact 불변식(§2.4)에 가장 유리**

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
        │  32-bit 주소, 단위 16-bit 원소 (8 GiB)        │
        │  FP16 tensor + packed INT8/INT4 weight blob  │
        └──────────────────┬──────────────────────────┘
                    DMA: GLOAD / GSTORE
                    (dequant-on-load: packed INT→FP16)
        ┌──────────────────▼──────────────────────────┐
        │  Unified SRAM (공유 scratchpad)              │  ← SW 관리, 결정적
        │  16-bit 원소 주소, 기본 256KB (config param)  │     matrix/vector 공용
        └─────────┬───────────────────┬───────────────┘
        ┌─────────▼─────────┐ ┌───────▼──────────┐
        │ Matrix Unit       │ │ Vector Unit      │
        │ 64×64 PE, FP32누적 │ │ 256 lanes, FP32  │
        └───────────────────┘ └──────────────────┘
```

- **compute 유닛은 SRAM만 접근** (global은 DMA 전용) — 계층 명확화
- SRAM 내부는 항상 FP16 (dequant 후) — 원소 단위 16-bit 주소,
  **단일 word 주소 설정** (half 상태머신 hazard 제거)
- 공유 SRAM: allocator/DMA 엔진 1개, matmul 결과를 vector 연산이 **복사 없이**
  소비 가능. 유닛별 분리는 §2.3 후속 옵션 (분리 시 ISA는 SRAM id field 추가로 수용)
- vector 길이: vlen ∈ [1, 256] (초과는 backend가 chunk)

## 2. ISA v09 — ver.08 대비 변경 목록

### 2.1 신규 (이번 범위의 본체)
- `GLOAD  sram_dst, g_addr32, shape/stride, dtype` — global→SRAM DMA.
  32-bit global 주소는 **2-word 명령 형식**(명령 word + 주소 word)으로 한 번에
  지정 (half 분할 설정 폐지). **dtype ∈ {FP16, INT8, INT4}**: INT는
  per-channel scale 벡터(global 주소 지정)로 **unpack + dequant하며 적재**
  (원소당 2/4값 전개) → SRAM에는 항상 FP16
- `GSTORE g_addr32, sram_src, shape/stride` — SRAM→global (FP16)
- vector/matrix 연산의 operand 주소 = SRAM 16-bit 주소 (단일 word 설정)

### 2.2 ver.08 버그·불일치 수정 (리뷰 §7.3 반영, v09에서 확정)
- reduce-max: **첫 원소 seed** (V3-003 수정) → softmax column-fold 우회 제거
- 표준 tanh-GELU activation mode 추가 (vendor식은 legacy mode로 병존, V3-004)
- immediate: **signed int16 정의** (V3-030 수정)
- save lane 수 = 직전 연산 결과 길이 (V3-006 수정)
- descriptor: **start + stride + shape** 3요소로 재정의 (MAIN/PARTIAL 폐지,
  죽은 main_addr/main_rows 제거, vector에는 비적용 명문화)
- 종료/저장 분리: `HALT` 신설, `SNAPSHOT`은 저장 전용 (0xF0 정리)
- 범위 초과 접근(global/SRAM 모두)은 **오류** (silent corruption 금지)

### 2.3 명시적으로 미룸 (v09 범위 밖, spec에 후보로만 기재)
- **유닛별 SRAM 분리** (mSRAM/vSRAM + 유닛 간 전송) — 공유로 시작,
  N7 접근 통계로 분리 필요성을 데이터로 판단
- loop/repeat 명령 (제어 word 93% 문제의 근본 해법이나 실행 모델 변경이 큼)
- **index 기반 `sincos(pos:int32, freq:fp16)` op** (§7.3-⑨a 계측): 각도의
  FP16 운반 한계를 op 내부 FP32로 해소 — host 개입이 스텝마다 있는 현 실행
  모델에서는 host cos/sin row 전달로 충분하므로 **loop 도입과 한 묶음**
- activation quantization / INT MAC datapath (weight-only가 우선)
- 주소 확장(40-bit 등) — FP16 weight 상주 full-model이 필요해지는 시점 항목
- cycle-accurate timing (counter 통계까지만; latency 표 수신 후)

### 2.4 수치 불변식의 설계 근거 (FP16 모드 ≡ 0818 bit-exact)
- 저장 FP16 / 연산 FP32 / 저장 시 RNE — 0818 계약 유지 (실측 §7 검증됨)
- 주소 체계(32-bit, 16-bit 원소)가 ver.08과 동일 → 주소 산술 동일
- 256-lane chunking이 순서를 바꾸지 않도록: **reduce는 chunk 내부 순차 +
  chunk 간 FP32 carry를 in-order 누적** = 기존 flat 순차와 동일 순서
- matmul K 누적: tile 순서 유지, FP32 누적기 유지
- SRAM staging은 FP16 값의 이동일 뿐 반올림 지점을 추가하지 않음
  (dequant-on-load는 FP16 모드에서 비활성 → 경로 자체가 동일)

## 3. Quantization 설계 (weight-only, packed)

- 형식: **INT8 per-output-channel symmetric**, 16-bit 원소당 2개 packed
  (scale = FP16 벡터, 채널당 1개) → 2차로 INT4 group-wise(g=64~128), 원소당 4개
- 흐름: checkpoint(BF16) → 호스트 quantizer(`make_quant_weights.py`)가
  packed INT8 blob + scale 벡터 생성(global 배치) → `GLOAD(dtype=INT8,
  scale)` 가 tile 적재 시 unpack+FP16 복원 → 이후 연산 경로는 FP16 모드와 동일
- 정렬 규칙: packed tensor의 행/타일 시작은 원소 경계 — 차원이 64 배수라
  자동 충족 (spec에 명문화)
- 검증 사다리: ① unpack+dequant 단위 테스트(복원값 = 호스트 계산과 bit-exact)
  ② layer별 FP16 대비 오차 계측 ③ 세 모델 3-token greedy — **수용 기준:
  token 일치(기대) 또는 불일치 시 logits 지표와 함께 문서화** (양자화는
  근사이므로 golden bit-일치를 요구하지 않는 유일한 모드)

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
| **N0** | ISA v09 spec 문서(`d_compiler/ISA_V09.md`) — §1~3을 인코딩 수준까지 확정 | 사용자 리뷰·승인 |
| **N1** | `mysim_v09.cpp` 골격: global/공유 SRAM 메모리 객체, decode 루프, HALT/SNAPSHOT, 경계 검사, perf counter | 단위 테스트 |
| **N2** | 데이터 이동: GLOAD/GSTORE (FP16, 2-word 주소), descriptor | `isa_v09.py` round-trip + 이동 단위 테스트 |
| **N3** | 연산: vector 256-lane 전 연산 + matrix 64×64 (§2.2 수정 반영) | op별 numpy FP16-step reference와 bit-exact |
| **N4** | `backend_v09.py` + SRAM staging codegen | **proxy layer가 0818 결과와 bit-exact** (불변식 1차 증명) |
| **N5** | 세 모델 golden을 v09 FP16 모드로 실행 | **token+logits가 기존 golden과 bit-exact** |
| **N6** | quantization: quantizer(pack) + unpack/dequant GLOAD + INT8 weight 실행 | §3 사다리 |
| **N7** | 통계·문서화: v09 vs 0818 비교표(word/DMA/SRAM) + 공유 SRAM 접근 통계(분리 필요성 판단 자료), spec 최종판 | `report/report_v09.md` |

예상 규모: N1~N3 = C-model 신작(~1,000줄), N4 = compiler 최대 작업(staging
codegen). N5까지가 "동작 동일 증명", N6부터가 신기능.

## 6. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| SRAM staging으로 프로그램 word 증가 (DMA 명령 추가) | 단일-word SRAM 주소 + 2-word global 주소 + 결과길이 save로 상쇄. N4에서 `analyze_isa_stats` 전·후 비교로 정량 관리 (DMA는 bytes 축 별도 계상) |
| 공유 SRAM의 유닛 간 경합 | 1차 모델은 기능 검증이라 무관. N7 접근 통계로 분리 필요성을 **데이터로** 판단 (§2.3 옵션) |
| 256-lane chunking이 수치 변경 | §2.4 순서 보존 규칙 spec 명문화 + N3 bit-exact 검증 |
| packed 접근의 경계 오류 | 정렬 규칙 spec 명문화 (64-배수 차원에서 자동 충족) + GLOAD 경계 검사 + unpack 단위 테스트 |
| INT8에서 greedy token 변화 | 수용 기준 사전 정의(§3), per-channel → group-wise 세분화 여지 |
