# NPU LLM Compiler (TVM 기반) — 설계 및 구현 계획

> 상태: **draft / 리뷰용**
> **대상 모델: Llama 3.2 3B** (최종 목표: 이 모델을 NPU에서 추론)
> 대상 백엔드: 본 레포의 NPU c-model (`_poc/mysim` 재구현 시뮬레이터)
> 작성일: 2026-06-08

---

## 0. 한 줄 요약

**최종 목표는 Llama 3.2 3B 추론을 NPU c-model에서 돌리는 것.** PyTorch/Relax로 표현된 모델을 받아 → NPU가 실제 지원하는 primitive로 분해(legalize) → 64×64 타일링 → NPU ISA 바이너리로 코드 생성 → `mysim`에서 실행·검증하는 컴파일러를 만든다. **3B의 `D=3072`·`F=8192`는 255를 한참 초과, `HD=128`은 ≤255지만 PE(64) 초과 → 모두 64×64 타일링 필수**이며, 28개 레이어·KV 캐시·autoregressive decode 등 모델 레벨 요소도 다룬다. 손으로 짠 `b_program_examples/`(특히 `npu.py`, `llama_layer.py`)는 **검증 기준(golden)·축소 프록시**로만 쓰고, 컴파일러 본체는 그것과 독립적으로 TVM 위에 새로 짠다.

> ⚠️ **타깃 실행기 = `_poc/mysim`(개선판), 원본 `a.out` 아님.** 본 계획은 `_poc/mysim`의 개선 기능(`--gout` write-back, `HALT`, uncapped buffer)을 **전제로** 한다. 원본 `a.out`(G-buffer 8192·program 32768 캡, write-back/halt 없음)으로는 이 파이프라인이 안 돌아간다. "mysim 수정 불가"는 **이 개선판을 더 이상 안 고친다**는 뜻이며, 원본 `a.out`은 **작은 커널 byte-exact 교차검증용**으로만 쓴다.

---

## 1. 배경: 우리가 가진 것 / 타깃의 제약

### 1.1 이미 확보한 자산
- **`_poc/mysim.cpp`**: 원본 NPU 바이너리와 byte-exact한 재구현 시뮬레이터. 입력 = `program_memory.bin`(uint32 명령어) + `G_buffer_data.bin`(FP16 데이터), 출력 = stdout trace + `--gout`로 G-buffer FP16 write-back.
- **ISA 사양**(`_poc/README.md`): 32비트 명령어 인코딩 전체. 벡터연산(add/sub/mul/div/exp/sqrt/min/max/compare/logical/shift/muladd/move/convert), 행렬연산(add/sub/mul/move + activation), 제어(addr/vlen/tile/load/save/halt).
- **`b_program_examples/`**: 손으로 짠 어셈블러(`npu.py`) + Llama 레이어 코드젠(`llama_layer.py`) + 커널별 테스트. **검증된 golden 결과**(전체 레이어 rel 0.12%) 보유. → 컴파일러 결과의 정답지로 사용.
- **보고서**(`report.md`, `cmodel_requirements.md`): ISA 미지원 연산의 우회 방법, 필요 ISA 우선순위 정리.

### 1.2 NPU의 핵심 제약 (컴파일러가 반드시 다뤄야 함)
| 제약 | 내용 | 컴파일러에서의 처리 |
|---|---|---|
| 연산 정밀도 | 내부 float32, **FP16 반올림은 G-buffer 저장 시에만** | 텐서 dtype=float16, 검증은 FP16 톨러런스 |
| 정적 메모리 | 동적 할당 없음, G-buffer 평탄 주소 | 모든 텐서를 컴파일 타임에 G-buffer 오프셋으로 정적 배치 |
| 즉시값 = 정수 | eps(1e-5), 1/√d 등 분수 상수 인코딩 불가 | 상수 텐서로 G-buffer에 적재(상수 폴딩) |
| 루프/분기 없음 | 현재 ISA에 제어 흐름 없음 → 완전 언롤만 가능 | **전부 언롤**(타일·레이어). ISA 루프 추가는 보류(§1.4) |
| reduce-max 없음 | softmax 수치 안정화(max 빼기) 우회 불가 | 초기엔 생략(작은 가중치로 회피), 후기 ISA 확장 후보 |
| 미지원 연산 다수 | reduce-sum, broadcast, transpose, SiLU 등 | **legalize 패스로 기존 명령 조합으로 분해**(§5) |

### 1.3 타깃 모델 — Llama 3.2 3B 설정
| 파라미터 | 값 | 컴파일러 함의 |
|---|---|---|
| hidden_size `D` | **3072** | ≫255 → matmul 타일링 필수 |
| intermediate_size `F` | **8192** | ≫255 → FFN 타일링 필수 |
| num_hidden_layers | **28** | 레이어 반복 → 루프 or 프로그램 재사용 |
| num_attention_heads `H` | **24** | |
| num_key_value_heads `KV` | **8** | GQA, GPK=H/KV=3 |
| head_dim `HD` | **128** | ≫255는 아니나 64 초과 → 타일 2개 |
| vocab_size | **128256** | embedding/lm_head 거대 matmul |
| max_position_embeddings | 131072 | RoPE 테이블(실제론 필요 길이만) |
| rope_theta | **500000** | + **llama3 rope scaling**(factor 32, low 1, high 4, orig 8192) |
| rms_norm_eps | **1e-5** | 정수 즉시값 불가 → **상수 텐서로 적재**(보고서의 eps=0과 달리 실제값 필요) |
| activation | **SiLU / SwiGLU** | §5 legalize |
| tie_word_embeddings | **true** | embedding과 lm_head 가중치 공유 |

**핵심 함의**: 모든 핵심 차원이 64×64 PE를 한참 넘으므로 **타일링은 필수**다. 루프/누산 ISA가 없으므로 타일·누산은 **전부 언롤/명시적 명령 시퀀스**로 펼친다(아래 §1.4).

### 1.4 현재 구현 스코프 (확정 2026-06-08)
> 빠르게 **prefill 동작**에 도달하기 위한 의도적 단순화. 항목별로 나중에 ISA 추가 시 재검토.

| 항목 | 현재 방침 | 나중에 |
|---|---|---|
| **mysim 수정** | **불가(고정)**. 모든 건 현재 ISA로 | 필수 ISA 여부 별도 판단 후 검토 |
| **루프/분기** | ISA 추가 안 함 → **타일·레이어 전부 언롤(반복 emit)** | loop ISA 추가 시 롤링 |
| **matmul-accumulate** | 누산 비트 없음 → **save→load→matrix_add 시퀀스로 명시적 누산** | accumulate ISA 추가 시 단순화 |
| **softmax max-subtraction** | **제외**(작은 가중치로 회피) | reduce-max ISA 추가 시 적용 |
| **실행 단계** | **prefill만 우선** 동작시킴 | decode는 그 후 |
| **KV 캐시** | **저장만** 한다고 가정(K/V 계산·배치까지) | decode 설계 시 재사용 로직 추가 |
| **transpose** | **원소 단위 복사**로 구현 + **그 명령어 오버헤드를 직접 측정·분석**(산출물) | 블록 전치/strided/전치 ISA 검토 |

→ 즉 **B2(루프/누산 ISA)는 현재 비활성**. 큰 차원은 "언롤된 거대 프로그램"으로 두고, 그 비용(명령어 수·시뮬 시간·transpose 오버헤드)을 **분석 대상**으로 삼는다. 완전 언롤한 풀 28레이어 3B는 시뮬 비현실적일 수 있으므로 **단일 레이어 prefill 동작 + 풀 모델 비용 추정**이 현실 목표(§9·§10).

---

## 2. 설계 원칙

0. **`mysim.cpp`(주어진 c-model)가 유일한 실행기이자 기준(source of truth)이며, 현재는 수정 불가(고정)로 간주한다.** 우리가 만드는 것은 **컴파일러**(모델 → NPU 명령어 바이너리)일 뿐, **NPU 동작 실행기를 새로 만들지 않는다.** 모든 실행·결과는 주어진 `mysim`이 낸다. 명령어 인코딩의 정답 규칙도 `mysim.cpp`의 디코드 로직이 정의한다(즉 `mysim.cpp`가 ISA 사양). ⚠️ `isa.py`는 **명령어를 32비트 워드로 인코딩해 `program_memory.bin`을 만드는 인코더**일 뿐 — **실행기가 아니다.** ⚠️ **mysim에 없는 기능(루프/분기, matmul-accumulate, reduce-max 등)은 ISA를 추가하지 않고 기존 명령으로 우회/언롤한다.** ISA 추가 필요 여부는 **나중에 별도로 판단**(§1.4·§10).
1. **앞단/뒷단 분리.** `Relax import → legalize → memory plan`(앞단, 백엔드 무관, 작업량의 대부분) 과 `codegen → ISA`(뒷단)를 명확히 분리한다. 앞단은 어떤 코드젠 전략을 쓰든 **재사용**된다.
2. **버려지는 코드 없이 점진.** 처음부터 TIR 기반으로 짓고(§4), tensorize/ISA확장 같은 어려운 부분은 한꺼번에 하지 않고 단계로 쌓는다(§4 로드맵). 각 단계는 이전 단계를 그대로 재사용한다.
3. **항상 검증 가능.** 모든 단계에서 출력이 (a) float 참조, (b) 기존 `llama_layer.py` golden, (c) 가능하면 원본 `a.out`과 일치하는지 differential test. **검증 실행기는 항상 `mysim`.**
4. **`npu.py`는 본체에 쓰지 않는다.** (사용자 의도: 그건 레이어 동작 확인용 일회성 코드.) 단, 초기 하네스 스모크 테스트엔 잠깐 활용 가능.

---

## 3. 전체 아키텍처

```
  PyTorch / 합성 nn.Module  (단기: 단일 transformer 레이어)
        │  import  (torch.export → relax,  또는 직접 Relax 작성)
        ▼
  Relax IRModule (high-level ops: matmul, rms_norm, softmax, silu, ...)
        │  ── 그래프 패스 ──
        │   • 상수 폴딩 / dead code 제거
        │   • [핵심] NPU-legalize: 미지원 op를 NPU primitive 조합으로 분해 (§5)
        │   • 메모리 플래닝: 모든 텐서 → G-buffer 정적 오프셋
        ▼
  Relax (NPU-primitive only: matmul, add/sub/mul/div, exp, sqrt, ...)
        │  legalize_to_tir: 각 op → TIR PrimFunc
        ▼
  TIR  ── (B1부터) matmul inner-block을 64x64 PE intrinsic으로 tensorize
        │  custom codegen:  TIR  →  NPU ISA (uint32 명령어 스트림)
        ▼
  program_memory.bin  +  G_buffer_data.bin
        │  runtime wrapper:  mysim 실행  →  gout.bin
        ▼
  결과 텐서  ──(differential test)──  float 참조 / golden
```

---

## 4. 로드맵 — B0 → B0.5 → B1 → M (한 TIR 파이프라인 위에 누적)

> 실제 분기는 **코드젠 한 단계뿐**, 앞단은 공유된다. 어려운 것을 하나씩만 추가한다. (B2=루프/누산/reduce-max ISA는 mysim 수정이라 **보류**.)

| 단계 | 내용 | 산출물 | 검증 기준 |
|---|---|---|---|
| **B0** *(logical bring-up)* | Relax→TIR→**flat** ISA 코드젠, **64×64 타일링 없음**(mysim이 ≤255 logical matmul 허용하므로 그대로 한 방에) | **축소 프록시** 단일 레이어 **prefill** 컴파일·실행 | float 참조 + `llama_layer.py` golden (tolerance) |
| **B0.5** *(64×64 legal oracle)* | **작은 차원에서 진짜 64×64 타일링** + K타일 `save→load→m_add` 누산 | 64×64-legal 커널(matmul/한 레이어) 실행 | **tiled-FP16 참조**와 tolerance 비교(§9) |
| **B1** *(스케일·costing)* | 실제 3.2 3B 차원으로 타일링 코드 **생성** | **선별 커널 실측**(작은 것만 실제 실행) + **전체 레이어 명령어 수/메모리/시간 비용 추정** | tiled-FP16 참조(실행분), 비용 모델(추정분) |
| **M** *(모델, prefill)* | embedding/lm_head, RoPE(llama3 scaling), 28레이어 그래프, **KV는 저장만** | **풀 그래프 lowering + 비용 산정**(실제 풀-3B 실행/HF logits는 비대상). 정확도는 **단일 레이어/축소 프록시에서만** HF·float 참조와 대조 | 단일 레이어 정확도 + 풀모델 cost report |
| ~~B2~~ *(보류)* | 루프/분기 롤링, matmul-accumulate, reduce-max (mysim 수정 필요) | — | 현재 비활성 |
| *(이후)* decode | KV 재사용, autoregressive | decode 동작 | — |

⚠️ **B0의 위상**: B0는 mysim이 64 초과 타일(≤255)을 logical하게 받아주기 때문에 도는 **"logical c-model bring-up"**이다 — 실제 64×64 PE HW에서 legal한 코드가 **아니다**(기존 `llama_layer.py`도 동일하게 logical 전제). 64×64-legal 코드는 **B0.5**에서 처음 나온다. 이 구분을 명확히 둔다.

⚠️ **B1 "실제 차원 단일 레이어 실행"은 과대약속이라 하향**: `mysim.cpp`는 **매 instruction·load·save 원소마다 무조건 `cout` 출력**(quiet 옵션 없음). stdout을 `/dev/null`로 보내도 C++ 포맷팅(float→문자열) 비용은 그대로 → 3B 단일 레이어(수십만~수백만 명령, 타일당 수천 원소 출력)는 **실측 비현실적**. 따라서 B1 산출물 = **선별(작은) 커널 실측 + 전체 비용 추정**.

왜 B0를 건너뛰지 않나: B0.5/B1을 바로 하면 tensorize + TIR→ISA 코드젠을 동시에 디버깅하게 되고 정답지(oracle)가 없다. B0가 앞단(legalize/메모리플래닝/하네스) + logical 정확도를 먼저 잡고, **그 코드가 B0.5/B1/M에 그대로 재사용**된다.

### 4.1 명령어 수 / 비용 분석 (B1의 핵심 산출물 — 일찍 만든다)
- 한 matmul `[M,K]@[K,N]`를 64×64 타일로: 타일 수 = ⌈M/64⌉·⌈N/64⌉·⌈K/64⌉, 타일마다 tile/addr/load/compute/save + (K누산 시) save/load/m_add. 거기에 **mysim이 출력하는 원소 수**(load 2×타일면적 + save 타일면적)까지 모델링 → "실제로 실행 가능한 한계"를 빨리 판단.
- prefill(seq=S) 레이어당 q/k/v/o/gate/up/down 합 → **레이어 1개도 수만~수십만 명령**, ×28 = 완전 언롤 시 수백만+. + 3B 가중치 ≈12GB+(mysim `vector<float> G`).
- **transpose(원소복사) 오버헤드 분석**: 전치 1회당 명령어 = O(행×열), vlen=1 복사 언롤. attention K 전치·Xn 전치가 전체 명령어에서 차지하는 비중을 실측 → "전치 ISA 필요성"의 근거.
- → **비용 모델을 코드젠 초기에 작성**(quiet mysim 없이도 어디까지 실제 실행 가능한지 판단).
- ⚠️ **명확화**: 이 "비용 모델"은 **성능(latency/cycle) 예측이 아니다** — mysim은 시간을 모델링하지 않고 HW 타이밍 스펙도 없어 latency는 불가. 대신 **HW 데이터 없이 프로그램만으로 세는 정적 자원 분석**이다: 명령어 수(+종류별), G-buffer 원소/바이트, MATMUL 타일 수, **load+save 원소 수(=mysim 출력량=시뮬 시간 직결)**, copy/transpose 오버헤드 비중. 목적은 **실행가능성·크기·상대 오버헤드** 판단. (나중에 HW 타이밍 스펙이 생기면 `counts × per-op cost`로 진짜 latency 모델을 위에 얹을 수 있고, 이 정적 카운터가 그 토대가 됨.)

---

## 5. [핵심] NPU-legalize 매핑표

> Relax high-level op → NPU가 실제 지원하는 primitive 조합. (보고서 §5의 우회 트릭을 패스로 옮긴 것)

| Relax op | NPU primitive 분해 | 비고 / 비용 |
|---|---|---|
| `matmul` (행렬모드) | `m_mul` (matrix mode) | 네이티브. B1부터 64×64 타일. **K>64는 타일별 부분곱을 save→load→`m_add`로 명시적 누산**(accumulate ISA 없음), 전부 언롤 |
| `add/sub/mul/div` | `v_add/v_sub/v_mul/v_div` | 네이티브 |
| `exp`, `sqrt` | `v_exp`, `v_sqrt` | 네이티브 |
| `reduce_sum`(행 합) | `ones[1×K]`와 **matmul** | reduction마다 matmul 1회 |
| `broadcast`(스칼라→벡터) | `ones[N×1]`와 **matmul**(외적) | matmul 1회 |
| `rsqrt`/`reciprocal` | `sqrt` 후 `ones / x` (div) | |
| `rms_norm` | `x²`(mul self) → reduce_sum → `÷D` → sqrt → `1/rms`(div) → broadcast → `×x` → `×weight` | 다수 명령 |
| `softmax` | `exp` → reduce_sum → broadcast → div. **max-subtraction 생략** | ⚠ FP16 exp 오버플로 위험. 작은 가중치로 회피(보고서 §6.1) |
| `silu` | `x · sigmoid(x)`; `sigmoid(x)=1/(1+exp(-x))`; `-x`는 `0-x` | activation 1개당 5~6 명령 |
| `negate` | `0 - x` (zeros 상수 벡터) | |
| `copy`/`move` | `x + 0` | |
| `transpose` | 가중치는 **호스트 사전 전치**; 런타임 전치는 원소 복사(가급적 회피) | 레이아웃 제약 |
| causal mask | `-30000` 마스크 행렬을 score에 add | |
| residual add | `v_add` | 네이티브 |
| RoPE (`rope_theta=500000` + llama3 scaling) | cos/sin은 NPU에서 못 만듦 → **호스트 사전계산 테이블을 상수로 적재**; rotate_half는 negate+copy+mul+add | scaling(factor 32 등) 반영해 호스트에서 각도 계산 |
| eps (`1e-5`) | 정수 즉시값 불가 → **상수 텐서로 G-buffer에 적재** 후 add | 보고서 프록시는 eps=0이었음; 실제 모델은 1e-5 필요 |
| 1/√d 스케일 (`HD=128`) | 정수 즉시값(√128≈11.3) 부정확 → 상수 텐서 또는 사전 스케일 | 정밀도 위해 상수 적재 권장 |

구현 형태: `relax.transform`의 커스텀 패스로 작성. 일부는 TVM 기본 legalize가 우리가 원치 않는 형태(예: softmax에 max-reduction)를 만들므로 **NPU 전용 legalize로 오버라이드**한다.

---

## 6. 단기 목표 (B0) — 상세 구현 계획

### 6.1 목표 (Definition of Done)
- 입력: **Llama 3.2 3B와 동일한 구조의 축소 프록시**(GQA·RoPE·SwiGLU·RMSNorm 그대로, 차원만 축소: 예 SEQ8, D64, H4, KV2, HD16, F128) 단일 레이어를 표현한 **Relax IRModule**. → 구조 검증이 목적이므로 작은 차원으로 빠르게 반복하고, 실제 차원(3072 등)은 B1에서 타일링과 함께 도입.
- 출력: `program_memory.bin` + `G_buffer_data.bin` 자동 생성 → `mysim` 실행 → 결과가
  - float 참조 대비 rel ≤ ~0.5%,
  - 기존 `llama_layer.py` golden과 동등 수준.
- 부분 목표(먼저 통과시킬 순서): ① 단일 `matmul` → ② elementwise(add/mul/div/exp/sqrt) → ③ rms_norm → ④ attention(softmax 포함) → ⑤ swiglu → ⑥ 전체 레이어.
- ⚠️ B0는 **축소 프록시 + logical(64×64 타일링 없음) bring-up**이다. mysim이 ≤255 타일을 받아줘서 도는 것이며 실제 64×64 PE-legal 코드가 아니다(64×64 legal은 B0.5부터). 목적은 앞단(legalize/메모리플래닝/코드젠/하네스) 검증과 oracle 확보.

### 6.2 디렉토리 레이아웃 (제안)
```
d_compiler/
  PLAN.md                  # (이 문서)
  README.md                # 빌드/실행 빠른 안내
  pyproject.toml           # 의존성(tvm 등)
  npu_compiler/
    __init__.py
    frontend.py            # PyTorch/합성 → Relax IRModule
    legalize.py            # §5 매핑표를 구현한 Relax 패스들
    memplan.py             # G-buffer 정적 오프셋 할당
    isa.py                 # 명령어 인코더(32비트 워드 → program_memory.bin). 실행기 아님; mysim.cpp 디코드 규칙을 따름
    codegen.py             # TIR → ISA (B0: flat / B0.5~: 64x64 타일링)
    intrin.py              # (B0.5~) 64x64 PE tensorize intrinsic
    runtime.py             # program/G-buffer 작성 → 주어진 mysim 실행 → gout 파싱
    cost.py                # 비용 모델: 명령어 수 + mysim 출력 원소 수 추정 (§4.1)
    reference.py           # float 참조 + tiled_fp16_reference (B0.5/B1 oracle, §9-4)
    config.py              # 차원/주소맵/상수
  tests/
    test_matmul.py
    test_elementwise.py
    test_rmsnorm.py
    test_attention.py
    test_swiglu.py
    test_layer.py          # 전체 레이어 vs golden
    conftest.py            # mysim 빌드 픽스처
```

### 6.3 핵심 컴포넌트별 작업
1. **`isa.py` — 명령어 인코더 (가장 먼저, 의존성 없음) — 실행기 아님**
   - **`mysim.cpp`의 디코드 로직이 ISA 사양의 source of truth.** 그 디코드 규칙(opcode/필드 비트 위치)에 맞춰 32비트 명령어 워드를 인코딩하는 함수들만 작성한다(`addr/vlen/tile/load/save/compute/halt`). `_poc/README.md`는 그 요약본.
   - `npu.py`를 베끼지 말고 사양에서 작성하되, **출력 바이너리가 기존 예제(`b_program/inst_*`의 `program_memory.bin`)와 바이트 일치**하는지로 인코더의 정확성을 검증.
   - 단위 테스트: 몇 개 명령어 인코딩이 기존 `program_memory.bin`과 바이트 일치 → 인코더가 mysim이 기대하는 형식을 정확히 따름을 보장.

2. **`runtime.py` — 실행 하네스 (주어진 mysim을 호출만)**
   - `program_memory.bin`/`G_buffer_data.bin` 작성 → **주어진 `mysim`** `--run N --gout` 실행 → `gout.bin`을 numpy로 파싱.
   - mysim은 주어진 `_poc/mysim.cpp`를 그대로 빌드(`g++ -O2 -std=c++17 _poc/mysim.cpp -o ...`). **실행 의미론은 우리가 재구현하지 않는다.**
   - (스모크 테스트 한정으로 기존 `npu.py`의 `run()` 잠깐 재사용 가능.)

3. **`memplan.py` — 정적 메모리 플래닝**
   - 모든 텐서(입력/가중치/중간/상수)에 G-buffer 오프셋 부여(bump allocator, 중간 버퍼 재사용 가능하면 재사용).
   - 상수(가중치, ones, zeros, mask, RoPE cos/sin)를 초기 `G_buffer_data.bin`에 배치.

4. **`legalize.py` — §5 분해 패스**
   - 우선순위: matmul/elementwise(통과) → rms_norm → softmax → silu → reduce/broadcast.
   - 각 패스마다 단위 differential test.

5. **`codegen.py` — TIR → flat ISA**
   - B0: tensorize 없이, 각 primitive를 `tile/vlen/addr/load/compute/save` 명령 시퀀스로 평탄 emit. 끝에 `halt`.

6. **`frontend.py` — Relax 입력**
   - 단기: `llama_layer.py`와 동일 구성의 레이어를 **Relax로 직접 기술**(가장 통제 쉬움). 이후 `torch.export`→Relax import로 확장.

### 6.4 B0 마일스톤 체크리스트
- [x] **환경**: `npu-tvm` + TVM v0.19 소스 빌드 (§8)
- [x] **M1**: `isa.py` 인코더 — golden 21 + **14,336 워드(56개 .bin) 라운드트립 일치** (`tests/test_isa.py`)
- [x] **M0**: `runtime.py` — mysim 빌드·실행·gout 파싱; vector-add·matmul e2e 검증 (`tests/test_runtime.py`)
- [x] **M2**: 단일 matmul **Relax → memplan → codegen → ISA → mysim**, FP16 참조 **byte-exact**(rel=0, dims≤255) (`tests/test_matmul.py`). + `memplan.py`/`codegen.py`(operator-level)/`driver.py` 추가
- [x] **M3**: elementwise(add/sub/mul/div/sqrt/exp) + 체인 — 전부 byte-exact (`tests/test_elementwise.py`)
- [x] **M4**: rms_norm — reduce/broadcast=ones-matmul legalize + 상수배치, rel 8e-4 (`tests/test_rmsnorm.py`); `legalize.py` 등장
- [x] **M5**: single-head causal attention — transpose(원소복사 macro)+softmax(max-sub 제외)+mask, rel 2e-4 (`tests/test_attention.py`)
- [x] **M6**: swiglu — silu=z/(1+exp(-z)) legalize, rel 7e-4 (`tests/test_swiglu.py`)
- [x] **M7**: 전체 레이어(RMSNorm→GQA+RoPE+causal→res→RMSNorm→SwiGLU→res) — float 참조 대비 **rel=0.13%** (golden 0.12% 수준), 3171 instr, G-buffer 65488 FP16 (`tests/test_layer.py`). **→ B0 완성**

**transpose 오버헤드 실측(§4.1 분석)**: attention seq8/hd16에서 k^T 원소복사 = 768명령 = **전체 프로그램의 68%**. → 전치/strided-load ISA 필요성의 정량 근거.

### 6.5 B0.5 — 64×64 hardware-legal 타일링 (진행)
- [x] **B0.5 K-타일링**: K를 ≤64로 쪼개 부분곱을 `save→load→add`로 누산(매 save FP16 반올림). codegen `tile=64` 경로 + memplan `scratch_alloc` + A-타일 gather(strided→연속). (`tests/test_tiling.py`)
  - 타일 출력 == **`tiled_fp16_reference` byte-exact** (K=192/130/256/128)
  - one-shot과 512중 260개 다름 → **B0 oracle로 byte-exact 비교 금지** 실증 (리뷰어 지적 확인), float64 대비 rel 5e-4
  - 모든 m_mul 타일 **≤64×64 (hardware-legal)** 자동 검사
- [x] **B1 (M/N/K 일반 타일링)**: 출력을 64×64 타일로 쪼개고 A/B gather(`copy2d`)·K누적·C scatter. **임의 차원(M>64·N>64·K>64) hardware-legal** (`tests/test_tiling.py`)
  - M>64·N>64·비64배수 포함 8케이스 전부 **tiled_fp16_ref와 byte-exact**, 모든 m_mul 타일 ≤64×64 자동검사
- [ ] `cost.py` 정적 자원/실행가능성 분석(§4.1) — 실제 3.2 3B 차원 코드생성 + 명령어/메모리/출력량 추정
- [x] gather/scatter 최적화: **연속이면 복사 생략**(A는 kt==K, B/C는 nt==N일 때 직접 load/store). 4×128×4 기준 1156→100. 진짜 strided(N>64 등)일 때만 gather/scatter 수행 — 잔여 비용은 strided-load ISA 부재의 본질적 비용.
- [ ] (선택) 추가 최적화: 스크래치 버퍼 재사용, 가중치 호스트 사전 타일링

---

## 7. MLC-LLM 및 외부 레퍼런스 활용 방침

| 레퍼런스 | 활용 방식 | 채택 여부 |
|---|---|---|
| **VTA** (`apache/tvm/vta`) | **가장 유사한 선례**(커스텀 ISA + GEMM 가속기 + 시뮬레이터 + tensorize). B1의 64×64 tensorize intrinsic, 시뮬레이터 연동 패턴을 **구조 참고** | 코드 직접 사용 X, **패턴 차용** |
| **MLC-LLM** | LLM 프런트엔드의 정석(Relax 기반 Llama 정의, KV cache, 파이프라인). **단기엔 미사용**(우리는 단일 레이어를 직접 Relax로 기술). **중기 풀모델 단계에서** 모델 정의/KV cache/양자화 흐름을 **참고 또는 부분 차용** | 단기 X / 중기 참고 |
| **TVM BYOC** (`FuseOpsByPattern`, `relax.ext.*`) | CPU+NPU 혼합 실행이 필요해질 때 서브그래프 오프로딩 골격으로. 단일 레이어 전부 NPU면 불필요 | 필요 시 도입 |
| **TVM legalize / dlight** | §5 분해 패스 작성 시 기존 분해 규칙 참고 | 참고 |

방침 요약:
- **단기(B0)**: MLC-LLM은 안 쓴다. 축소 프록시 레이어를 Relax로 직접 기술하는 게 통제·디버깅이 쉽다.
- **B1**: VTA의 tensorize 패턴을 참고해 64×64 PE intrinsic 작성.
- **모델 단계(M)**: **Llama 3.2 3B 전체**(28레이어·KV캐시·prefill/decode·embedding/lm_head·RoPE scaling)는 MLC-LLM이 이미 Relax로 정의해 둔 자산이 크다 → MLC-LLM의 **Llama 모델 정의, Paged/일반 KV cache, RoPE(llama3 scaling) 구현, Relax 파이프라인**을 본격 참고/차용하고, **우리 NPU 코드젠(legalize+tensorize+ISA emit)을 그 파이프라인의 백엔드 target으로 연결**하는 게 가장 빠르다. (3B 가중치 로딩·tokenizer도 MLC 자산 활용.)
- 공통: **Relay(구버전) 사용 금지, Relax(TVM Unity)** 로 일원화. 단 **실행은 항상 주어진 `mysim`** (MLC의 GPU/CPU 런타임이 아니라 우리 ISA→mysim 경로).

---

## 8. 환경 / 설치

현재 상태: 기존 conda env 들(`base/ssd/ssd-int8/tvm/tvm-study`) 존재. g++ 11.2.
- ⚠️ **기존 conda env `ssd`는 다른 작업에서 사용 중 → 절대 건드리지 않는다.**
- ⚠️ 기존 `tvm-study` env에 깔린 TVM은 **mlc.ai nightly(0.20.dev1070)** 인데, `tir→tirx`, `nd→runtime.tensor`로 **개명된 리팩토링 중간 빌드**다. Relax 파이프라인은 동작(검증함)하나 **표준 `tvm.tir`/`tvm.script.tir`가 없고 TVMScript TIR 작성(`T.block`)이 깨져** 공식 문서·MLC-LLM 소스를 그대로 못 따라간다. → **사용 안 함.**
- mlc.ai 휠 인덱스엔 이 nightly 한 버전뿐 → 표준 API TVM은 **소스 빌드**로만 확보 가능.

**결정: 표준 API TVM을 소스 빌드.** (이유: 문서·튜토리얼·MLC-LLM이 전부 표준 `tvm.tir` 사용 → study/장기 개발엔 표준 API 필수)
- 새 conda env **`npu-tvm`** (Python 3.11, conda-forge: `cmake ninja llvmdev numpy cython`). 기존 env엔 변경 없음.
- TVM 소스: **apache/tvm v0.19.0**(tirx 리팩토링 이전 = 표준 레이아웃 + Relax 지원), 레포 밖 `~/tvm-src`에 클론.
- 빌드: `USE_LLVM` 켜고 cmake+ninja → `pip install -e python`.
- mysim 빌드는 `runtime.py` 픽스처가 자동 수행.
- 설치 확정 커맨드는 빌드 성공 후 `README.md`에 고정 기록.

---

## 9. 검증 전략 (differential testing)

1. **float 참조**: 각 커널을 순수 Python/numpy로 독립 구현 → rel error(FP16 tolerance) 비교.
2. **golden 대조**: 동일 입력으로 기존 `llama_layer.py` 결과와 비교(logical 전체 레이어 oracle, B0용).
3. **원본 a.out 교차검증**(작은 커널 한정): 원본 `a.out`은 G-buffer 8192·program 32768 캡이라, **그 한계 내 작은 커널만** 원본에서도 돌려 `_poc/mysim`과 **같은 표현끼리** byte-exact 확인(원본은 보조 검증용, 메인 타깃 아님).
4. ⚠️ **B0.5/B1은 B0와 byte-exact 비교 금지.** B0(one-shot matmul)는 **최종 1회만** FP16 반올림하지만, B0.5/B1은 **K타일마다 `save`→FP16 반올림→`load`→`m_add`** 누산이라 결과가 **정상적으로 달라진다**(mysim.cpp 저장 시 반올림). → 별도의 **`tiled_fp16_reference`**(동일한 타일링·중간 FP16 반올림 순서를 numpy로 모사)를 만들어 그것과 비교한다. B0와는 **tolerance** 비교만.
5. **FP16 톨러런스**: 저장 시 반올림 특성 반영(절대/상대 혼합 임계). 비교는 **항상 같은 표현(저장된 FP16값 vs FP16 참조)** 끼리.

---

## 10. 리스크 / 오픈 퀘스천 (리뷰 포인트)

- **mysim 수정: 현재 불가(결정됨).** 루프/누산/reduce-max 등은 ISA 추가 없이 언롤·우회로 구현(§1.4). 필수 ISA 여부는 나중에 비용 분석(§4.1) 근거로 별도 판단. → 당분간 모든 한계는 "컴파일러가 우회"로 흡수.
- **시뮬레이션 실현성(최대 리스크).** 루프 ISA 없이 완전 언롤 → 풀 28레이어 3B 프로그램은 명령어 수백만+, 게다가 mysim의 `vector<float> G`에 3B 가중치(≈12GB+ RAM) → **풀 모델 end-to-end 실행은 비현실적일 수 있음.** 대응: **단일 레이어 prefill을 실제로 돌려 검증** + **풀 모델은 명령어 수/메모리 추정**으로 다룬다. (레이어별 `--gout` 체이닝, 가중치 스트리밍, 양자화는 추후 옵션.)
- **softmax 안정성**: max-subtraction 제외(§1.4) → 큰 score에서 FP16 오버플로 위험. 작은 가중치로 회피. 근본 해결은 reduce-max ISA(보류).
- **transpose 오버헤드(분석 대상)**: 원소복사 전치 = 전치당 O(행×열) 명령. 실측해서 전체 명령어 중 비중을 보고(§4.1) → 전치 ISA 필요성 판단 근거.
- **메모리 재사용 정책**: 중간 버퍼 재사용 vs **KV는 저장만**(전용 영역 확보). G-buffer 평탄 주소 안에서 가중치/활성/KV 영역 분리 배치.
- **frontend 범위**: 축소 프록시 직접 기술(B0) → 실제 차원(B1) → MLC 경로로 풀 모델 prefill(M) 전환 시점.
- **decode(이후)**: autoregressive(M=1) + KV 재사용은 별도 단계. 지금은 KV 저장 포맷/위치만 prefill에서 확정해 둠.

---

## 11. 다음 액션 (제안)
1. ✅ 환경: **`npu-tvm`** env + TVM v0.19 소스 빌드 완료(§8, README).
2. M1: `isa.py` **인코더**(실행기 아님) + 기존 예제 `program_memory.bin`과 바이트 교차검증.
3. M0: `runtime.py`로 **주어진 `_poc/mysim`** 빌드·실행·gout 파싱.
4. **비용 모델**(`cost.py`): 명령어 수 + mysim 출력 원소 수 추정 → "실제 실행 가능 한계"를 일찍 확보(§4.1).
5. M2: 단일 matmul **logical** e2e (B0, 축소 프록시) → float/golden tolerance.
6. **`tiled_fp16_reference`** 구현 → **B0.5**: 작은 차원 64×64 타일링+누산 oracle 통과.
7. B0 전체 레이어(prefill, logical) → **B1**: 실제 차원 코드 생성 + 선별 실측 + 비용 추정 → **M**(풀 그래프 lowering + costing).
