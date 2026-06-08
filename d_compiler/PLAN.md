# NPU LLM Compiler (TVM 기반) — 설계 및 구현 계획

> 상태: **draft / 리뷰용**
> **대상 모델: Llama 3.2 3B** (최종 목표: 이 모델을 NPU에서 추론)
> 대상 백엔드: 본 레포의 NPU c-model (`_poc/mysim` 재구현 시뮬레이터)
> 작성일: 2026-06-08

---

## 0. 한 줄 요약

**최종 목표는 Llama 3.2 3B 추론을 NPU c-model에서 돌리는 것.** PyTorch/Relax로 표현된 모델을 받아 → NPU가 실제 지원하는 primitive로 분해(legalize) → 64×64 타일링 → NPU ISA 바이너리로 코드 생성 → `mysim`에서 실행·검증하는 컴파일러를 만든다. **3B의 모든 차원(D=3072, F=8192, HD=128)이 255를 초과하므로 타일링은 필수**이며, 28개 레이어·KV 캐시·autoregressive decode 등 모델 레벨 요소도 다룬다. 손으로 짠 `b_program_examples/`(특히 `npu.py`, `llama_layer.py`)는 **검증 기준(golden)·축소 프록시**로만 쓰고, 컴파일러 본체는 그것과 독립적으로 TVM 위에 새로 짠다.

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
| 루프/분기 없음 | 현재 ISA에 제어 흐름 없음 → 완전 언롤만 가능 | 초기엔 축소 차원 언롤, 풀 차원은 후기(B2)에서 ISA 확장 |
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

**핵심 함의**: 모든 핵심 차원이 64×64 PE를 한참 넘으므로 **타일링(B1)은 선택이 아니라 필수**다. 또한 한 레이어의 단일 projection matmul만 64×64 타일로 펼쳐도 수천~수만 명령이 되어, 28레이어를 완전 언롤하면 프로그램이 비현실적으로 커진다 → **루프 지원(B2)이 사실상 필수 경로**가 된다(mysim은 명령어 캡이 없어 *실행*은 가능하나, 프로그램 크기·시뮬레이션 시간이 비현실적). 자세한 영향은 §4·§10.

---

## 2. 설계 원칙

0. **`mysim.cpp`(주어진 c-model)가 유일한 실행기이자 기준(source of truth)이다.** 우리가 만드는 것은 **컴파일러**(모델 → NPU 명령어 바이너리)일 뿐, **NPU 동작 실행기를 새로 만들지 않는다.** 모든 실행·결과는 주어진 `mysim`이 낸다. 명령어 인코딩의 정답 규칙도 `mysim.cpp`의 디코드 로직이 정의한다(즉 `mysim.cpp`가 ISA 사양). ⚠️ `isa.py`는 **명령어를 32비트 워드로 인코딩해 `program_memory.bin`을 만드는 인코더**일 뿐 — **실행기가 아니다.** (B2의 루프/누산처럼 `mysim`을 *수정*해야 하는 항목은 별도의 명시적 결정 사항으로 다룬다 → §10.)
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

## 4. 로드맵 — B0 → B1 → B2 → M (한 TIR 파이프라인 위에 누적)

> "A(npu.py 비지터) vs B(tensorize)"는 폐기된 프레이밍이다. 실제 분기는 **코드젠 한 단계뿐**이고, 앞단은 공유된다. 아래는 어려운 것을 하나씩만 추가하는 경로. **타깃이 Llama 3.2 3B이므로 B1·B2는 선택이 아니라 최종 목표의 필수 단계**이고, B0는 그 토대를 검증하는 bring-up 단계다.

| 단계 | 추가되는 것 | mysim ISA 변경 | 산출물 | 검증 기준 |
|---|---|---|---|---|
| **B0** *(단기 목표, bring-up)* | Relax→TIR→**flat** ISA 코드젠 (tensorize 없음) | 불필요 | **축소 프록시**(3.2 3B 구조, 작은 차원) 단일 레이어 컴파일·실행 | float 참조 + `llama_layer.py` golden |
| **B1** *(필수)* | matmul을 64×64 PE intrinsic으로 **tensorize** (바깥 타일루프 **언롤**) | 불필요 (언롤로 회피) | **실제 차원** 단일 레이어(예: D=3072) 타일링 실행 | B0 출력을 oracle로, float 참조 |
| **B2** *(사실상 필수)* | 바깥 타일루프를 **롤된 채** emit + matmul-accumulate | **필요**: 루프/분기 + 누산 명령 추가 | 프로그램 크기 유한(언롤 폭증 제거) → 28레이어 현실화 | B1 출력 / float 참조 |
| **M** *(모델 레벨)* | KV 캐시, prefill/decode, 28레이어, embedding/lm_head, RoPE scaling | (KV캐시용 구조) | **Llama 3.2 3B end-to-end** 추론 | HF 참조 logits/토큰 |

핵심 정정: **tensorize(64×64 매핑) 자체는 ISA 루프가 불필요**하다(바깥 루프 언롤). ISA 루프(B2)는 "차원이 커질 때 명령어 폭증을 막기 위해서" 필요하다 — 3.2 3B에선 이게 곧 현실적 필수.

왜 B0를 건너뛰지 않나: B1/B2를 바로 하면 tensorize + TIR→ISA 코드젠 + ISA확장을 **동시에** 디버깅하게 되고 정답지(oracle)가 없다. B0가 그 oracle과 앞단(legalize/메모리플래닝/하네스) 검증을 먼저 제공한다. **B0의 모든 앞단 코드는 B1/B2/M에서 그대로 재사용**된다.

### 4.1 명령어 수 개략 추정 (왜 B2가 필요한가)
- 한 matmul `[M,K]@[K,N]`를 64×64 타일로: 타일 MAC 수 = ⌈M/64⌉·⌈N/64⌉·⌈K/64⌉, 각 타일마다 tile/addr/load/compute/save 수 명령.
- decode(1토큰) q-proj `[1,3072]@[3072,3072]` ≈ 1·48·48 ≈ 2,304 타일 → 수천~만 명령. 레이어당 q/k/v/o/gate/up/down 합치면 **레이어 1개도 수만 명령**, ×28 → 완전 언롤은 수십만~수백만 명령.
- mysim은 명령어 캡이 없어 *실행*은 가능하지만 프로그램 파일·시뮬레이션 시간이 비현실적 → **루프(B2)로 타일·레이어를 롤**해야 함.

---

## 5. [핵심] NPU-legalize 매핑표

> Relax high-level op → NPU가 실제 지원하는 primitive 조합. (보고서 §5의 우회 트릭을 패스로 옮긴 것)

| Relax op | NPU primitive 분해 | 비고 / 비용 |
|---|---|---|
| `matmul` (행렬모드) | `m_mul` (matrix mode) | 네이티브. B1부터 64×64 타일 |
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
- ⚠️ B0는 **축소 프록시 전용 bring-up**이다(실제 3B 차원은 타일링 없이는 못 돌림). 목적은 앞단(legalize/메모리플래닝/코드젠/하네스)을 검증하고 B1의 oracle을 확보하는 것.

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
    codegen.py             # TIR → ISA (B0: flat)
    intrin.py              # (B1) 64x64 PE tensorize intrinsic
    runtime.py             # program/G-buffer 작성 → 주어진 mysim 실행 → gout 파싱
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
- [ ] M0: 환경 구축(TVM 설치) + `runtime.py`로 mysim 빌드·실행·gout 파싱
- [ ] M1: `isa.py` 인코더 + 기존 예제 바이트 교차검증
- [ ] M2: 단일 matmul Relax → ISA → mysim, float 참조 일치
- [ ] M3: elementwise 세트 통과
- [ ] M4: rms_norm (reduce/broadcast legalize 포함) 통과
- [ ] M5: attention(softmax 포함) 통과
- [ ] M6: swiglu(silu legalize) 통과
- [ ] M7: 전체 레이어 vs `llama_layer.py` golden 동등

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

현재 상태: **TVM 미설치**, Python 3.12.7, g++ 11.2.
- ⚠️ **기존 conda env `ssd`는 다른 작업에서 사용 중 → 절대 건드리지 않는다.** (`env_setup.sh`의 lib 경로도 `ssd` 것이므로 본 프로젝트에선 의존하지 않는다.)

계획:
- **TVM 전용 conda env를 새로 만든다: `tvm-study`** (Python 3.11/3.12). `ssd`에는 어떤 변경도 가하지 않는다.
- TVM(Relax 포함) 설치: 프리빌트 휠(mlc-ai nightly 계열) 또는 LLVM 켜고 소스 빌드 중 택1. **버전 고정**(reproducibility)하고 `pyproject.toml`에 명시.
- mysim 빌드는 `runtime.py` 픽스처가 자동 수행.
- (정확한 설치 커맨드는 M0에서 환경 확정 후 README에 기록.)

---

## 9. 검증 전략 (differential testing)

1. **float 참조**: 각 커널을 순수 Python/numpy로 독립 구현(이미 `test_*.py`에 존재) → rel error 비교.
2. **golden 대조**: 동일 입력으로 기존 `llama_layer.py` 결과와 비교(전체 레이어 oracle).
3. **원본 a.out 교차검증**(가능 범위): G-buffer 한계(8192) 내 작은 커널은 원본에서도 실행해 byte-exact 확인.
4. **단계 oracle**: B1/B2 출력은 B0 출력을 oracle로 byte-exact 비교.
5. FP16 톨러런스: 저장 시 반올림 특성 반영(절대/상대 혼합 임계).

---

## 10. 리스크 / 오픈 퀘스천 (리뷰 포인트)

- **[결정 필요] `mysim` 수정 허용 범위.** 원칙상 주어진 c-model이 기준(§2.0)이지만, 3.2 3B를 현실적으로 돌리려면 B2의 **루프/분기 + matmul-accumulate**가 사실상 필수다. 이건 `mysim.cpp` *수정*을 의미한다(보고서가 말한 "기존 c-model에 비파괴적 추가"와 동일 노선). → **mysim을 어디까지 수정해도 되는가**가 핵심 결정 사항. 옵션: (a) mysim as-is만 사용, 풀 모델은 언롤로 강행(비현실적 크기), (b) 보고서 권고대로 루프/누산/reduce-max를 비파괴적으로 추가, (c) 절충(루프만 추가).
- **softmax 안정성**: max-subtraction 부재 → 큰 score에서 FP16 오버플로. 3.2 3B 실제 가중치에선 위험할 수 있음. 단기엔 회피, 근본 해결은 mysim에 reduce-max 추가(위 결정에 종속).
- **시뮬레이션 실현성(중요)**: 3B 가중치를 mysim의 `vector<float> G`에 올리면 **파라미터 32억 × 4B ≈ 12GB+ 호스트 RAM**. 풀 모델 end-to-end를 mysim에서 한 번에 돌리는 건 비현실적일 수 있음 → **레이어 단위 검증 + 가중치 스트리밍/체이닝(`--gout`)** 전략, 또는 양자화 고려. 풀 모델 정확도 검증은 HF 참조와 레이어별 대조로.
- **TVM 버전/설치 방식**: 프리빌트 vs 소스 빌드, Python 3.12 호환성, `tvm-study` env.
- **메모리 재사용 정책**: 중간 버퍼 재사용(라이브러리 vs 활성값 vs KV캐시 영역 분리). G-buffer 평탄 주소 안에서 KV캐시 배치.
- **frontend 범위**: 축소 프록시 직접 기술(B0) → 실제 차원(B1) → MLC 경로로 풀 모델(M) 전환 시점.
- **transpose 비용**: 원소 복사 방식의 명령어 폭증 → 풀 차원에서 블록 전치/레이아웃(호스트 사전 전치) 전략 필요.
- **decode vs prefill**: autoregressive decode(M=1)와 prefill(M=seq)의 타일링·KV캐시 접근이 다름 → 둘 다 코드젠 필요.

---

## 11. 다음 액션 (제안)
1. 본 PLAN.md 리뷰 → 범위·우선순위 확정. **특히 §10의 "mysim 수정 허용 범위" 결정**(B2/풀모델의 전제).
2. M0: `tvm-study` env에 TVM 설치 + `runtime.py`로 **주어진 mysim** 빌드·실행·gout 파싱.
3. M1: `isa.py` **인코더**(실행기 아님) + 기존 예제 `program_memory.bin`과 바이트 교차검증.
4. M2: 단일 matmul end-to-end PoC (축소 프록시).
5. (이후) B0 전체 레이어 → B1 실제 차원 타일링 → §10 결정에 따라 B2 → M(풀 3.2 3B).
