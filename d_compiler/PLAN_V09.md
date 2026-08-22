# PLAN — ver.09 자체 C-model 및 ISA 제안 (branch `cmodel-v09`)

> 시작일: 2026-08-22
>
> 목표: vendor 재현이 아닌 **우리가 설계하는 차기 C-model/ISA**를 만든다.
> 범위: ① SRAM scratchpad 메모리 계층 (vector/matrix 유닛별 분리),
> ② quantization (weight INT8, dequant-on-load), ③ vector 유닛 256-lane 고정,
> ④ 이에 맞는 ISA 확정 (ver.08 리뷰 §7.5의 수정사항 반영).
> 산출물은 설계팀에 전달할 **spec 문서 + 동작 C-model + compiler backend + 평가표**다.

---

## 0. 불변 원칙

1. **ver.08 세계는 동결** — vendor `a.out`, `mysim_0818.cpp`, `isa_0818.py`,
   `backend_0818.py`는 한 줄도 바꾸지 않는다. 기존 36개 테스트와 세 모델
   golden은 이 branch에서도 상시 회귀 gate다.
2. v09는 **전부 새 아티팩트**: `_poc/mysim_v09.cpp`, `npu_compiler/isa_v09.py`,
   `npu_compiler/backend_v09.py`, `backend="v09"` target.
3. **핵심 수치 불변식**: v09의 FP16(비양자화) 모드는 0818 결과와
   **bit-exact 동일**해야 한다. 이것이 성립하도록 산술 계약을 설계한다
   (§2.4). 성립하면 세 모델 golden이 v09 검증에 공짜로 재사용된다.
4. 단계마다 상세 commit + 측정 (기존 방법론 유지). 새 simulator는
   perf counter를 처음부터 내장한다 (latency 표가 오면 비용 모델로 직결).

## 1. 메모리·유닛 아키텍처 (모델링 대상)

```
                 ┌─────────────────────────────────────┐
                 │  Global Memory (기존 G-buffer 역할)  │  ← DRAM/HBM 모델
                 │  byte-addressed, base-reg + offset  │     크기: config 파라미터
                 └───────┬──────────────────┬──────────┘
                    DMA(GLOAD/GSTORE)  DMA(GLOAD/GSTORE)
                         │ (dequant-on-load 지원)
                 ┌───────▼───────┐  ┌───────▼────────┐
                 │  Matrix SRAM  │  │  Vector SRAM   │   ← scratchpad (SW 관리,
                 │  (mSRAM)      │  │  (vSRAM)       │      cache 아님 — 결정적)
                 └───────┬───────┘  └───────┬────────┘
                 ┌───────▼───────┐  ┌───────▼────────┐
                 │ Matrix Unit   │  │ Vector Unit    │
                 │ 64×64 PE,     │  │ 256 lanes,     │
                 │ FP32 누적기    │  │ FP32 내부       │
                 └───────────────┘  └────────────────┘
```

확정 제안 (N0에서 사용자 승인 후 고정):

| 항목 | 제안값 | 근거 |
|---|---|---|
| Global 주소 | **operand별 wide base register(40-bit byte) + 32-bit offset** | ISA 미팅 대안 3 (우리 제안의 자체 구현) |
| SRAM 주소 | 16-bit 원소 주소, 단일 word 설정 | V3-025/주소 상태머신 hazard 제거 |
| mSRAM 크기 | 파라미터 (기본 192KB = A/B/C 64×64 FP16 tile × double-buffer 여유) | tile staging + 이중버퍼 |
| vSRAM 크기 | 파라미터 (기본 64KB = 256-lane row × 128) | reduce/elementwise 작업창 |
| compute의 메모리 접근 | **유닛은 자기 SRAM만 접근**, global은 DMA 전용 | scratchpad 모델의 정석, 계층 명확화 |
| vector 길이 | vlen ∈ [1, 256] (256 초과는 backend가 chunk) | HW 사양 반영 |

## 2. ISA v09 — ver.08 대비 변경 목록

### 2.1 신규 (이번 범위의 본체)
- `SBASE r, addr40` — global base register 설정 (단일 확장 word 또는 2-word)
- `GLOAD dst_sram, base_r, offset, shape/stride, dtype` — global→SRAM DMA.
  **dtype ∈ {FP16, INT8, INT4}**: INT는 per-channel scale(글로벌의 scale 벡터
  주소 지정)로 **dequant하며 적재** → SRAM에는 항상 FP16
- `GSTORE` — SRAM→global (FP16)
- vector/matrix 연산의 operand 주소 = 소속 SRAM 주소 (16-bit, 단일 word)

### 2.2 ver.08 버그·불일치 수정 (리뷰 §7.3 반영, v09에서 확정)
- reduce-max: **첫 원소 seed** (V3-003 수정) → softmax의 column-fold 우회 제거
- 표준 tanh-GELU를 activation mode로 추가 (vendor식은 legacy mode로 병존, V3-004)
- immediate: **signed int16 정의** (V3-030 수정)
- save lane 수 = 직전 연산 결과 길이 (V3-006 수정)
- MAIN/PARTIAL → **start + stride + shape** 3요소 descriptor로 재정의
  (죽은 main_addr/main_rows 제거, vector에는 비적용 명문화)
- 종료/저장 분리: `HALT` 신설, `SNAPSHOT`은 저장 전용 (0xF0 semantics 정리)
- 범위 초과 접근은 **오류** (silent corruption 금지, V3-023/027)

### 2.3 명시적으로 미룸 (v09 범위 밖, spec에 후보로만 기재)
- loop/repeat 명령 (제어 word 93% 문제의 근본 해법이나 실행 모델 변경이 큼)
- activation quantization / INT MAC datapath (weight-only가 우선)
- cycle-accurate timing (counter 기반 통계까지만; latency 표 수신 후)

### 2.4 수치 불변식의 설계 근거 (FP16 모드 ≡ 0818 bit-exact)
- 저장 FP16 / 연산 FP32 / 저장 시 RNE — 0818 계약 유지 (실측 §7 검증됨)
- 256-lane chunking이 순서를 바꾸지 않도록: **reduce는 chunk 내부 순차 +
  chunk 간 FP32 carry를 in-order 누적** = 기존 flat 순차와 동일 순서
- matmul K 누적: tile 순서 유지, FP32 누적기 SRAM/레지스터 상주
- dequant-on-load는 FP16 모드에서 비활성 → 경로 자체가 동일

## 3. Quantization 설계 (weight-only 1차)

- 형식: **INT8 per-output-channel symmetric** (scale = FP16 벡터, weight당 1개)
  → 2차로 INT4 group-wise(g=64~128) packed
- 흐름: checkpoint(BF16) → 호스트 quantizer(`make_quant_weights.py`)가
  INT8 blob + scale 벡터 생성 → `GLOAD(dtype=INT8, scale_addr)` 가
  tile 적재 시 FP16으로 복원 → 이후 연산 경로는 FP16 모드와 동일
- 검증 사다리: ① dequant 단위 테스트(복원값 = 호스트 계산과 bit-exact)
  ② layer별 FP16 대비 오차 계측 ③ 세 모델 3-token greedy — **수용 기준:
  token 일치(기대) 또는 불일치 시 logits 지표와 함께 문서화** (양자화는
  근사이므로 golden bit-일치를 요구하지 않는 유일한 모드)

## 4. Compiler 변경 (전부 v09 신규 파일, 기존 경로 불변)

- `backend_v09.py`: 동일 Relax 파이프라인(`passes.npu_pipeline` 재사용) 소비.
  codegen이 **2단 메모리 계획**을 수행:
  - global memplan: 기존 `memplan` 재사용 (tensor의 global 배치)
  - SRAM allocator: 프로그램 내 tile staging slot (정적, double-buffer)
  - matmul: A/B tile GLOAD → mSRAM 연산/누적 → C GSTORE
  - vector: 256-lane chunk 단위 GLOAD → vSRAM 연산 → GSTORE
- memplan에 dtype(INT8/4) 표기 추가 (weight tensor 한정)
- `analyze_isa_stats.py` 확장: v09 프로그램의 **DMA bytes / 유닛별 연산 수 /
  SRAM 점유** 통계 (word 수와 함께 전·후 비교표의 축)

## 5. 단계 계획 (각 단계 = commit + gate)

| 단계 | 내용 | Gate |
|---|---|---|
| **N0** | ISA v09 spec 문서(`d_compiler/ISA_V09.md`) 작성 — §1~3의 결정사항을 인코딩 수준까지 확정, 사용자 리뷰 | 사용자 승인 |
| **N1** | `mysim_v09.cpp` 골격: global/mSRAM/vSRAM 메모리 객체, decode 루프, HALT/SNAPSHOT, perf counter | 단위 테스트 |
| **N2** | 데이터 이동: SBASE/GLOAD/GSTORE (FP16), 경계 검사, descriptor | `isa_v09.py` round-trip + 이동 단위 테스트 |
| **N3** | 연산: vector 256-lane 전 연산 + matrix 64×64 (§2.2 수정 반영: seeded reduce-max, 표준 GELU mode, signed imm) | op별 numpy FP16-step reference와 bit-exact |
| **N4** | `backend_v09.py` + SRAM staging codegen | **proxy layer가 0818 결과와 bit-exact** (불변식 1차 증명) |
| **N5** | 세 모델 golden을 v09 FP16 모드로 실행 | **token+logits가 기존 golden과 bit-exact** |
| **N6** | quantization: quantizer + dequant-on-load + INT8 weight 실행 | §3 사다리 (token 일치 또는 계측 문서화) |
| **N7** | 통계·문서화: v09 vs 0818 비교표(word/DMA/SRAM), spec 최종판 → 설계팀 전달 패키지 | 보고서 `report/report_v09.md` |

예상 규모: N1~N3가 C-model 신작(~1,000줄 예상), N4가 compiler 최대 작업
(staging codegen). N5까지가 "동작 동일 증명", N6부터가 신기능.

## 6. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| SRAM staging으로 프로그램 word 증가 (DMA 명령 추가) | descriptor 단일-word화 + 결과 길이 save로 상쇄; N4에서 `analyze_isa_stats` 전·후 비교로 정량 관리. DMA는 word가 아니라 bytes 축으로 별도 계상 |
| 256-lane chunking이 수치 변경 | §2.4 순서 보존 규칙을 spec에 명문화하고 N3에서 flat-순차 reference와 bit-exact 검증 |
| INT8에서 greedy token 변화 | 수용 기준을 사전 정의(§3); per-channel → group-wise로 세분화 여지 |
| 범위 팽창 (loop, activation quant 등) | §2.3에 명시적 out-of-scope 목록 유지 |
