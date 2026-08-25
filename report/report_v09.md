# 차세대 NPU C-model/ISA (ver.09) 설계 및 구현 보고서

> 2026-08-25 · branch `cmodel-v09`
> 스펙: `d_compiler/ISA_V09.md` · 구현: `_poc/mysim_v09.cpp` · 컴파일러: `d_compiler/npu_compiler/backend_v09.py`

## 0. 개요

- **배경**: 기존 vendor C-model(ver.08)은 ① 단일 평면 메모리(FP16 전용, 8 GiB 한계)
  ② 양자화 미지원 ③ 오류를 조용히 넘기는 실행 규약 등 실제 하드웨어 설계에
  그대로 쓰기 어려운 한계가 있음
- **목표**: SRAM scratchpad 메모리 계층 + INT8/INT4 양자화 + 256-lane vector 유닛을
  갖춘 **차세대 ISA와 C-model을 직접 설계·구현·검증**
- **설계 원칙**: 기존 ver.08 명령 인코딩을 최대한 유지하고 해석만 확장
  → 기존 프로그램·컴파일러 자산이 그대로 이어짐
- **검증 원칙(불변식)**: 새 모델의 FP16 모드는 기존 결과와 **bit 단위로 동일**해야 함
  → 기존에 확보한 3개 LLM(Llama 3.2 3B / Gemma 4 E2B / Qwen3-4B)의 생성 결과가
  회귀 검증 기준으로 그대로 재사용됨
- **현재 상태**: 설계 확정 + C-model/컴파일러 구현 완료, **3개 모델 모두 FP16 모드에서
  token·중간값·logits까지 bit 단위 일치 확인**. 양자화 실측(W8A16/W8A8)이 다음 단계

## 1. 메모리 계층

### 1.1 구조 — 2단 메모리 + DMA

```
Global Memory (DRAM 모델, 16 GiB) ←─ DMA ─→ SRAM (8 MiB scratchpad) ─→ 연산 유닛
        "데이터 창고"                            "작업대"           (SRAM만 접근 가능)
```

- **Global Memory — 데이터 창고**
  - 32-bit 주소, 주소 단위 = 32-bit → 총 **16 GiB**
  - 데이터 형식(dtype) 개념이 없음: DMA로 옮기기만 하는 비트 저장소
  - 연산 유닛은 직접 접근 불가 (계층 분리를 강제)
- **SRAM — 공유 scratchpad (matrix/vector 유닛 공용), 8 MiB**
  - 주소 단위 = **4-bit(nibble)** → FP32(8칸)/FP16(4)/INT8(2)/INT4(1)가
    한 공간에 혼재 가능, 모든 형식이 정수 개의 칸을 차지해 주소 계산이 단순
  - 주소 field는 기존 32-bit 형식 재사용 (유효 24-bit, 여유 bit는 예비)
- **DMA 명령 2개 (GLOAD/GSTORE)**
  - 2차원 블록(행 수 × 열 수 + 행 간격)을 **형식 무관하게 원본 그대로** 복사
  - 행 간격(stride)이 32-bit → 기존 16-bit 한계로 불가능하던 넓은 행렬 표현 해소
  - 동기 방식(전송 완료 후 다음 명령), 비동기+barrier는 후속 확장 항목
- **규약**
  - 단위 환산은 기계 전체에서 단 하나: Global 1칸 = SRAM 8 nibble (little-endian 고정)
  - 범위 초과·정렬 위반 접근은 **즉시 오류 종료** (기존의 조용한 오동작 제거)
  - 종료는 HALT 명령이 유일한 정상 경로(결과 회수 겸용), 0xF0은 중간 checkpoint 전용

### 1.2 컴파일러의 SRAM 운용 (staging)

- 연산마다 필요한 데이터만 SRAM에 올리고(GLOAD) → 연산 → 결과를 되내림(GSTORE)
- **superset staging**: 텐서를 포함하는 최소 32-bit 경계 구간을 통째로 이동
  → 홀수 크기/홀수 위치 텐서도 주소 체계 수정 없이 처리 (경계의 이웃 원소는
  원값 그대로 왕복하므로 무해)
- SRAM(8 MiB)보다 큰 행렬(가중치 등)은 64×64 타일/행 패널 단위로 **스트리밍**
- reduce(합/최대)는 256-lane 조각 사이를 FP32로 이어붙여 **기존과 동일한 누적
  순서** 유지 → 결과가 bit 단위로 변하지 않는 근거

## 2. 양자화

### 2.1 원리

- 양자화 = 실수 x를 "scale(실수 1개) × 정수"로 근사 (x ≈ s·q)
- 행렬곱 y = Σ a·w 에 대입하면 y ≈ **s_a·s_w × Σ(qa·qw)**:
  scale이 누적 도중 변하지 않는 한, **정수로 곱-누적을 끝낸 뒤 실수 곱 1번**으로 복원(dequant) 가능
- 이 성질이 지원 범위(granularity)와 하드웨어 구조를 결정함

### 2.2 지원 조합 (matrix 연산)

| activation × weight | 통칭 | 내부 곱 | 누적기 |
|---|---|---|---|
| FP16 × FP16 | 기본 | FP16 | FP32 |
| FP16 × INT8 / INT4 | W8A16 / W4A16 (weight만 양자화) | FP16 | FP32 |
| INT8 × INT8 / INT4 | W8A8 / W4A8 (weight+activation) | 정수 | INT32 |

- 그 외 조합은 오류로 규정. INT32 누적기 여유 검증 완료(최악 2^27.6 ≪ 2^31)

### 2.3 연산 파이프라인 — 반올림 지점은 출구 한 곳

```
SRAM ─▶ 입구(무손실 형변환만) ─▶ 곱-누적 배열(FP32 또는 INT32) ─▶ 출구(dequant) ─▶ SRAM
                                                        누적값 × scale → FP16 반올림 1회
```

- 입구: INT8/INT4 값은 FP16/INT8에 **오차 없이** 표현되므로 변환만 수행 (반올림 없음)
- 출구: 모든 양자화 조합이 **같은 출구 하드웨어**(FP32 dequant-누적기)를 공유,
  FP16 모드는 출구를 통과만 함 → 기존 결과와의 bit 일치가 구조적으로 보장

### 2.4 Scale 단위 (granularity) — "누적 도중 scale 불변" 원칙의 적용

| 지원 | 형식 | 근거 |
|---|---|---|
| ✅ activation | 토큰(행)마다 1개, 실행 중 동적 산출 | 출구에서 행별 곱 |
| ✅ weight | 출력 채널(열)마다 1개 | 출구에서 열별 곱 (INT8 표준) |
| ✅ weight | K방향 그룹 g ∈ {64, 128} | 연산기가 K를 64씩 잘라 누적하므로 그룹 경계가 누적 단위와 일치 — 그룹 결과를 출구의 FP32 누적기로 이어붙임 (INT4 품질에 필수, GPTQ/AWQ 표준과 동일) |
| ❌ | g < 64, activation의 K방향 scale | 누적 도중 scale 교체 필요 → 하드웨어로 불가/비효율. 필요 시 host 전처리(SmoothQuant류)로 해결 |

### 2.5 Activation 동적 양자화 (device 내 수행)

- 절차: 절대값 최대 → scale(=최대/127) 산출 → 반올림·포화(±127) → 압축 저장
- 신규 명령은 압축 저장 1개(VQUANT)뿐, 나머지는 기존 vector 연산 조합
- 대칭 양자화만 지원(zero-point 없음, LLM 표준), 반올림은 RNE
- scale은 FP32로 저장·전달 (FP16 표현 하한 문제 회피)
- norm/softmax/RoPE 등 vector 연산은 FP16 유지 — LLM 정확도상 정수화 부적합

## 3. ISA 변경 및 추가

**원칙: 인코딩은 유지, 해석만 확장.** 예비 bit가 전부 0인 기존 ver.08 프로그램은
그대로 유효한 v09 프로그램이다 (dtype 00 = FP16, flag 00 = 기존 동작).

### 3.1 유지하되 재해석된 명령 (인코딩 불변)

| 명령 | 기존 의미 | v09 재해석 |
|---|---|---|
| 0x80 주소 설정 | 평면 메모리 원소 주소 | **SRAM nibble 주소** (유일하게 값의 의미가 바뀜) |
| 0x88/0x89 행/열 | 원소 개수 | 값 그대로 + 예비 2-bit에 **operand dtype** |
| 0x82 vector 길이 | lane 수 | 그대로, 단 최대 256 |
| 0x90/0x98 load/save | 메모리↔유닛 | SRAM↔유닛 (형식 그대로) |
| 산술 연산 전체 | | 그대로 — 동작 mode는 operand dtype 조합에서 자동 유도 |

### 3.2 신규 명령

| opcode | 이름 | 역할 |
|---|---|---|
| 0xA0 / 0xA8 | GLOAD / GSTORE | Global↔SRAM DMA (5-word: 명령/Global주소/행stride/SRAM주소/행·열) |
| 0x8A / 0x8B | scale 주소 설정 | 출구 dequant가 읽을 FP32 scale 벡터의 위치 (activation용/weight용) |
| 0x1A / 0x1B | VQUANT / VDEQUANT | FP16 ↔ 압축 INT8/INT4 (대칭, RNE, 포화) |
| 0xFF | HALT | 종료 + 결과 기록 (유일한 정상 종료 경로) |

### 3.3 기존 명령의 예비 bit 확장 (기본값 0 = 기존 동작)

| 위치 | 의미 |
|---|---|
| matrix save의 2-bit | **carry-in / hold** — 그룹별 scale 양자화에서 그룹 결과를 FP32 출구 누적기로 잇고 마지막에만 FP16 기록 |
| reduce의 1-bit | **carry-in** — 256-lane 조각 사이 FP32 이어붙이기 |
| vector save의 1-bit | **FP32 저장** — scale 생산 시 반올림 없이 저장 |
| activation code 1 | **표준 tanh-GELU** 추가 (기존 vendor 근사식은 legacy로 병존) |

### 3.4 의미 수정 (기존 오류 정정)

- immediate 상수: 부호 없는 수로 오해석(−3 → 약 43억) → **signed 16-bit로 정정**
- reduce-max: 0에서 시작해 전부 음수인 입력이 틀림 → **첫 원소에서 시작**
- vector save: 결과보다 긴 구간을 0으로 덮어쓰던 동작 → **결과 길이만큼만 기록**
- 범위 초과 접근: 조용히 0 반환/무시 → **즉시 오류 종료**

## 4. 검증 결과

| 단계 | 대상 | 방법 | 결과 |
|---|---|---|---|
| 명령 단위 | 시뮬레이터 연산 전체 | 기존 프로그램을 기계 변환(주소 ×4)해 신·구 모델 실행 | vector 13종·행렬곱 배터리 **byte 단위 동일** |
| 신규 기능 | 양자화·carry·수정 3건 | 산술 순서를 재현한 numpy 기준값 | **bit 단위 동일** |
| 컴파일러 | 홀수 차원 attention 층 | 신·구 backend 결과 비교 | **bit 단위 동일** |
| **모델 전체** | **Llama / Gemma / Qwen3** | 3-token 생성 전 과정 재실행 | **생성 token, 중간 hidden, KV cache, logits 전부 기존 golden과 bit 단위 동일** |

- Llama [358, 2846, 4560] · Gemma [108, 236777, 236789] · Qwen3 [358, 1184, 311]
- 테스트 30개(시뮬레이터 24 + backend 6) 전부 통과, 기존 ver.08 테스트 무영향

## 5. 남은 일정

- W8A16(weight만 INT8) → W8A8(weight+activation) 실측: 층별 오차 계측과
  token 일치 여부 판정 (양자화 모드는 bit 일치가 아닌 품질 기준으로 평가)
- 명령 수·DMA 이동량·SRAM 점유 통계로 기존 대비 비용 정량화 및 스펙 최종판
