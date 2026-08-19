# NPU Compiler 작업 인계 문서 — 2026-08-19

이 문서는 현재 대화 세션을 종료하고 다른 Codex/IDE 세션에서 작업을 그대로 이어가기 위한
상세 handoff다. 새 세션은 먼저 이 문서 전체와 `report/report_0818.md`를 읽고, 아래의
현재 branch/commit 상태를 확인한 뒤 작업해야 한다.

> 핵심 상태: `compiler-v3`에서 vendor-compatible source C-model을 대상으로 official
> Llama 3.2 3B의 full prefill과 KV-cache autoregressive decode를 완료했다. 다음 큰 목표는
> 동일 compiler core에서 Gemma 4 E2B를 지원하도록 상위 compiler 구조를 model-independent하게
> 정리하고, Gemma 전용 frontend/pass/cache policy를 추가하는 것이다.

---

## 1. 사용자의 최종 의도와 우선순위

### 1.1 0818 C-model 관련 원칙

사용자는 0818 vendor update를 기준으로 compiler를 새로 retarget하기로 했다. 기존 0710
compiler는 새 경로가 완성될 때까지 회귀 기준으로 유지하고, 완전히 대체된 이후 제거하는
방향이다.

가장 중요한 정확성 원칙은 다음과 같다.

1. 우리 source C-model은 vendor와 최대한 똑같아야 한다.
2. vendor에 오류나 이상 동작이 있어도 source C-model이 동일하게 재현하는 쪽을 우선한다.
3. compiler가 올바른 모델 연산을 만들어야 할 때는 C-model을 임의로 고치지 않고,
   compiler legalization이 buggy/native opcode를 피하거나 primitive 조합을 선택해야 한다.
4. full model은 vendor binary의 고정 8192-entry G-buffer로 현실적으로 실행할 수 없으므로,
   arithmetic/quirk를 보존하고 capacity만 확장한 source target을 사용한다.
5. vendor binary는 삭제하거나 수정하지 않고 8192-entry 범위의 parity oracle로 보존한다.

### 1.2 현재 완료된 모델 목표

현재 완료 목표는 다음과 같다.

- official `meta-llama/Llama-3.2-3B`
- prompt full prefill
- 28개 transformer layer
- final RMSNorm 및 full-vocabulary LM head
- prefill에서 28 layer의 K/V cache seed
- autoregressive decode 2 step
- Hugging Face greedy token sequence와 일치

### 1.3 다음 목표

다음 요청은 **Gemma 4 E2B를 같은 compiler에서 지원하는 방향을 준비하는 것**이다.

이미 분석한 결론은 다음과 같다.

- 완전히 별도의 compiler를 만드는 것은 권장하지 않는다.
- 0818 ISA/backend/runtime은 공통으로 유지한다.
- Llama와 Gemma에 각각 model-family frontend/graph/assets adapter를 둔다.
- Relax IR 이하의 legalization, GEMM, memory, attention tiling 같은 pass는 공통화한다.
- PLE, QK-Norm, sliding/global attention, KV sharing 같은 의미론은 Gemma 전용 pass로 둔다.
- 현재 `Llama32SourceCompiler`에 `if gemma`를 계속 추가하는 구조는 피한다.

---

## 2. Git 및 workspace 상태

문서 작성 직전 확인한 상태:

~~~text
branch: compiler-v3
remote tracking: origin/compiler-v3
HEAD: 7c91647 v3 complete source model autoregressive decode
remote: https://github.com/KwanyoungCho/NPU_COMP.git
~~~

최근 핵심 commit:

| Commit | 의미 |
|---|---|
| `7c91647` | source model autoregressive decode, panel LM head, runner/reference 및 최종 보고서 |
| `d2a6232` | expanded source prefill/decode graph, source backend 연결, high-address broadcast 수정 |
| `fc3d244` | vendor parity source C-model의 dynamic G-buffer 및 source runtime |
| `b03dbaa` | vendor program/G-buffer capacity 계약 재감사 및 수정 |
| `b3ae9c2` | fixed vendor streaming 기반 full Llama prefill/token 검증 |
| `0e1b4f8` | full prefill에서 발견한 RMSNorm overflow 수정 |
| `69d8746` | official Llama layer vendor 실행 |
| `90a2a0a` | Relax graph를 vendor kernel로 내리는 stage 3 |

상태 확인 명령:

~~~bash
cd /home/chokwans99/NPU_cmodel
git status --short --branch
git log -8 --oneline --decorate
~~~

이 handoff 파일을 작성하기 전에는 worktree가 clean이었다. 이후 이 문서 자체가 새 변경으로
보일 수 있으므로 새 세션은 무조건 `git status`를 다시 확인해야 한다. 사용자의 기존 지시에
따라 compiler 구현을 진행할 때는 의미 있는 단계마다 commit하고 `origin`에 push한다.

주의:

- 사용자 변경이나 다른 dirty file을 reset/checkout하지 않는다.
- `git reset --hard`, 광범위한 삭제, 기존 branch 강제 덮어쓰기를 하지 않는다.
- 기존 0710 구현은 아직 삭제하지 않는다.

---

## 3. 개발 환경

### 3.1 필수 Python 환경

반드시 `npu-tvm` conda environment를 사용한다.

~~~text
Python: 3.11
TVM: apache/tvm v0.19.0 source build
TVM source: /home/chokwans99/tvm-src
Python executable: /home/chokwans99/anaconda3/envs/npu-tvm/bin/python
~~~

권장 실행 방식:

~~~bash
NPU_PY=/home/chokwans99/anaconda3/envs/npu-tvm/bin/python
"$NPU_PY" -c "import tvm; print(tvm.__version__)"
~~~

base Python에는 TVM이 없다. 예를 들어 base 환경에서
`python d_compiler/run_v3_source_generate.py --help`를 실행하면
`ModuleNotFoundError: No module named 'tvm'`가 발생한다. 이것은 코드 오류가 아니라 잘못된
Python environment 사용이다.

다음 environment는 사용하거나 변경하지 않는다.

- `ssd`: 다른 작업에서 사용 중
- `tvm-study`: refactor 중간 형태의 nightly TVM으로 API가 현재 compiler와 맞지 않음

상세 환경 설명은 `d_compiler/README.md`에 있다.

### 3.2 pytest 주의

`npu-tvm` 환경에는 pytest module이 없어 일반 pytest collection을 사용하지 않았다.
각 `test_*.py` 파일이 자체 entrypoint를 가지므로 직접 실행했다.

~~~bash
NPU_PY=/home/chokwans99/anaconda3/envs/npu-tvm/bin/python
for test_file in d_compiler/tests/test_*.py; do
  "$NPU_PY" "$test_file" || break
done
~~~

현재 repository의 26개 entrypoint가 모두 통과했다. slow official vendor layer 일부는
환경 변수 opt-in이 없으면 의도적으로 skip 메시지를 출력하지만 test 자체는 PASS다.

---

## 4. 모델 asset 및 ignored artifact

### 4.1 Llama 3.2 3B asset

기본 모델 위치:

~~~text
d_compiler/build/llama32_3b_hf
~~~

현재 크기는 약 6.0GB다. 이 경로는 `.gitignore`의 `d_compiler/build/` 규칙으로 무시된다.
다른 위치를 사용하려면 다음 환경 변수를 설정한다.

~~~bash
export NPU_LLAMA32_PATH=/absolute/path/to/llama32_3b_hf
~~~

loader는 `d_compiler/npu_compiler/v3_model.py`의 `Llama32Assets`다. 고정 revision:

~~~text
MODEL_ID = meta-llama/Llama-3.2-3B
MODEL_REVISION = 13afe5124825b4f3751f836b40dafda64c1ed062
~~~

### 4.2 현재 generation artifact

독립 HF reference:

~~~text
d_compiler/build/v3_reference_generate_hello_3.npz
~~~

source generation state/result:

~~~text
d_compiler/build/v3_source_generate_hello/
├── fixed_logits.npy
├── progress.jsonl
├── result.json
└── state.npz
~~~

현재 state는 cache length 9까지 진행된 상태이며 directory 전체는 약 1.6MB다. 모든 build
artifact는 Git에 포함되지 않는다.

중요한 역사적 주의:

- 최초 expanded full prefill에서는 row-major wide LM head가 16-bit stride truncation으로
  token `7272`를 출력했다.
- transformer hidden/cache는 정상이었고 LM head만 잘못되었다.
- panel LM head를 실행해 token/logits를 교정했고 이후 decode는 정상 cache에서 진행했다.
- 현재 checked-in code는 처음부터 `PackedRhsGemm`을 사용하므로 clean rerun에서는 이 문제가
  재발하지 않는다.
- 기존 artifact에는 이 디버깅 이력이 섞여 있으므로 완전한 clean proof가 필요하면 새 output
  directory를 지정해 처음부터 다시 실행한다.

---

## 5. 0818 vendor 동작 계약

권위 oracle:

~~~text
0818_npu_update/a_npu/a.out
~~~

권위 명세는 `0818_npu_update` 내부 instruction format ver.08 PDF다.

확정된 동작:

| 항목 | 실제 계약 |
|---|---|
| Reduce Max | opcode `0x19`; accumulator가 0으로 시작하여 all-negative vector는 잘못된 0 반환 |
| Sub-tile | MAIN/PARTIAL descriptor로 row-major tensor의 임의 sub-tile load/save |
| Broadcast | opcode `0x15`; scalar 주소는 low/high 상태를 모두 구성해야 함 |
| SiLU | `x * sigmoid(x)` |
| vendor GELU | 표준 GELU가 아니라 `x * sigmoid(2*x)` |
| PE arithmetic | 내부 float32 |
| G-buffer save | FP16 반올림 |
| MAC | bit 27 |
| `0xF0` | snapshot 저장; HALT가 아니며 이후 instruction 계속 실행 |
| process exit | implicit snapshot 저장 없음 |
| indirect addressing | 구현되지 않음 |

vendor binary capacity:

- program memory는 input file word 수만큼 동적 할당된다.
- program memory 32768-word 고정 제한은 없다.
- G-buffer는 **8192 FP16 entry** 정적 배열이다.
- 8192보다 큰 input을 vendor binary에 주면 안전하지 않다.
- compiler/runtime은 vendor target에서 8192 초과를 명시적으로 거부해야 한다.

`0xF0` 관련 추가 quirk:

- snapshot 한 개는 8192 FP16 = 16384 bytes다.
- 파일 끝 newline까지 포함하면 16385 bytes다.
- 한 실행에서 여러 `0xF0`을 만나면 snapshot이 같은 파일에 순서대로 append된다.
- backend-generated program은 마지막 유효 word에 `0xF0` 하나를 둔다.

전체 상세는 `d_compiler/RETARGET_0818.md`와 `report/report_0818.md`를 참조한다.

---

## 6. Source C-model의 정확한 역할

파일:

~~~text
_poc/mysim_0818.cpp
~~~

source C-model은 vendor를 임의로 개선한 새로운 연산 모델이 아니다. capacity를 제외한 실행
의미는 vendor와 동일하게 유지한다.

보존된 동작:

- Reduce Max zero-seed 오류
- vendor GELU `x*sigmoid(2x)`
- `0xF0` snapshot 후 실행 계속
- 종료 시 자동 저장하지 않음
- PE float32 arithmetic
- G-buffer load/save FP16 경계
- instruction encoding과 address state

의도적으로 확장한 부분:

- program: 실제 word 수만큼 보관
- G-buffer: input 크기에 맞춰 동적 할당
- 더 높은 destination address write 시 zero-initialized grow
- vendor-compatible input은 최소 8192-entry snapshot 유지
- 대용량 binary input/output을 1M entry chunk로 처리
- `NPU0818_QUIET` 또는 `--quiet`로 trace formatting만 생략

따라서 두 target의 의미는 다음과 같다.

| Backend | 실행 대상 | G-buffer | 목적 |
|---|---|---:|---|
| `0818`, `vendor-0818` | 제공 vendor `a.out` | 8192 고정 | 권위 oracle, 작은 graph/parity |
| `source-0818`, `0818-source` | `_poc/mysim_0818.cpp` build | 동적 | full LLM prefill/decode |

source runtime:

~~~text
d_compiler/npu_compiler/source_runtime_0818.py
~~~

이 runtime은 source가 변경되면 ignored build 영역의 executable을 자동 rebuild한다. 대용량
snapshot 전체를 Python으로 다시 복사하지 않고 필요한 output slice만 읽는 경로를 사용한다.

---

## 7. 현재 compiler 구성

### 7.1 공통 0818 backend

~~~text
d_compiler/npu_compiler/driver.py
d_compiler/npu_compiler/backend_0818.py
d_compiler/npu_compiler/isa_0818.py
d_compiler/npu_compiler/memplan.py
~~~

`driver.compile_module(..., backend="source-0818")`은 vendor와 동일한 row-major ver.08
codegen을 사용하되 vendor의 8192-entry validation만 끈다. assembler에는
`execution_target="source-0818"` metadata가 설정된다.

0818 backend가 처리하는 op:

- `relax.matmul`
- last-axis keepdims `relax.sum`, `relax.max`
- `relax.broadcast_to`
- rank-2 transpose
- last-axis slice/concat
- add/subtract/multiply/divide
- sqrt/exp/negative/cos/sin
- SiLU
- vendor GELU

모든 tensor는 global G-buffer에서 row-major다. 기존 0710의 tile-blocked global layout과
weight gather/scatter는 0818 path에 필요 없다. 단, PE 물리 실행 단위로서 최대 64x64 tiling은
계속 존재한다.

### 7.2 Relax legalization

~~~text
d_compiler/npu_compiler/legalize.py
d_compiler/npu_compiler/import_legalize.py
d_compiler/npu_compiler/frontend.py
~~~

중요 lowering:

- RMSNorm을 multiply/sum/sqrt/divide/broadcast로 분해
- stable softmax를 row max subtraction + exp + row sum으로 분해
- RoPE cos/sin을 position/frequency에서 on-device 계산
- SiLU는 matrix activation field를 사용
- PyTorch export frontend는 concrete input을 사용해 static Relax shape 생성

RMSNorm은 반드시 `x/sqrt(D)`를 먼저 FP16로 적용한 뒤 square/reduce한다. 먼저 `x*x`를 하면
residual outlier가 FP16 65504를 넘어 `inf`가 되는 문제가 실제 full model에서 발생했다.

### 7.3 Llama-specific graph

~~~text
d_compiler/npu_compiler/model.py
~~~

핵심 builder:

- `build_v3_prefill_layer_module(cfg, S, return_cache=True)`
- `build_v3_decode_kv_module(cfg)`
- `build_v3_decode_layer_module(cfg, context)`
- `build_v3_final_norm_module(cfg, S)`

prefill output은 선택적으로 다음을 concat하여 반환한다.

~~~text
[hidden, K0, V0, K1, V1, ..., K7, V7]
~~~

decode는 현재 context 길이를 compile-time static 값으로 specialization한다. current token의
K/V는 별도 source program으로 계산하고 host가 cache에 append한 뒤 attention/FFN program을
실행한다.

### 7.4 Official checkpoint loader

~~~text
d_compiler/npu_compiler/v3_model.py
~~~

특징:

- safetensors shard를 persistent handle로 열어 사용
- BF16 slice를 필요한 시점에 FP16 NumPy로 변환
- embedding은 필요한 token row만 읽음
- linear weight는 checkpoint `[N,K]`를 compiler operand `[K,N]`으로 transpose
- LM head는 tied embedding 사용
- panel LM용 `[K,64]` 연속 layout materialization 지원

### 7.5 Full source orchestration

~~~text
d_compiler/npu_compiler/v3_source_llama.py
~~~

주요 class는 `Llama32SourceCompiler`다.

동작:

1. prompt embedding row를 host gather
2. layer당 하나의 expanded source prefill program
3. output hidden과 roped K/V를 분리해 28 layer cache seed
4. 마지막 hidden을 final RMSNorm + panel LM head
5. host argmax
6. generated token embedding host gather
7. 각 layer에서 K/V source program
8. host cache append
9. 각 layer에서 exact-context attention/FFN source program
10. final norm + LM + argmax 반복

CPU에 남아 있는 작업:

- tokenizer
- embedding row lookup
- 동적 K/V cache append
- argmax
- 실행 orchestration 및 파일 I/O

Transformer layer와 LM-head 산술은 ver.08 source C-model program으로 수행한다.

---

## 8. Full Llama 결과

### 8.1 Prompt와 token

~~~text
prompt: Hello, NPU compiler!
input ids: [128000, 9906, 11, 452, 6459, 19979, 0]
~~~

독립 Hugging Face CPU reference:

~~~text
generated ids: [358, 2846, 4560]
decoded tokens: [" I", "'m", " trying"]
decoded text: " I'm trying"
~~~

source compiler 결과도 정확히 동일하다.

~~~text
prefill: token 358, cache length 7
decode position 7: token 2846, cache length 8
decode position 8: token 4560, cache length 9
~~~

### 8.2 수치 비교

Official layer 0 source vs HF:

~~~text
wall: 8.20 sec
max abs: 0.1953
mean abs: 0.000591
RMSE: 0.003991
cosine: 0.9999837
finite: true
~~~

Corrected prefill panel LM vs independent HF generation logits:

~~~text
max abs: 0.06287
mean abs: 0.007639
RMSE: 0.009760
cosine: 0.9999929
argmax: 358 match
~~~

Final decode logits vs HF:

~~~text
max abs: 0.07422
mean abs: 0.01002
RMSE: 0.01266
cosine: 0.9999881
argmax: 4560 match
~~~

### 8.3 실행 규모

~~~text
source prefill: 30 invocations
  = 28 layer + final norm + LM

2 decode steps: 116 invocations
  = each step 28 * (KV projection + decode layer) + final norm + LM

total: 146 invocations
source execution accumulated: about 774.5 sec (12.9 min)
~~~

현재 `result.json`의 stats가 116 invocation만 표시하는 이유는 saved state에서 resume한
decode process의 compiler instance 통계이기 때문이다. 전체 prefill+decode 합계는 보고서의
146 invocation이 맞다.

정적 program/G-buffer 규모:

| Program | Program words | FP16 G-buffer entries |
|---|---:|---:|
| prefill layer, S=7 | 497,563 | 101,084,262 |
| decode K/V | 30,451 | 6,316,741 |
| decode context 8 | 453,349 | 94,444,391 |
| decode context 9 | 453,757 | 94,446,443 |
| final norm | 98 | 18,436 |
| panel LM head | 1,845,685 | 394,133,760 |

비교를 위해 fixed vendor streaming full prefill은 517,557 invocation, 약 27.8분이었다.
source target은 layer/단계 단위 큰 program을 사용하여 invocation 수를 크게 줄였다.

---

## 9. Wide LM head와 16-bit descriptor 문제

동적 G-buffer와 가변 program이 있더라도 ISA의 matrix row/column descriptor field는 16-bit다.

Llama LM head logical RHS:

~~~text
[K,V] = [3072,128256]
~~~

`main_cols=128256`은 16-bit에 들어가지 않는다. 최초 구현은 값이 truncate되어 transformer
hidden이 정상이어도 token `7272`를 출력했다.

해결 파일:

~~~text
d_compiler/npu_compiler/source_gemm_0818.py
~~~

해결 방식:

1. tied embedding `[V,K]`를 vocab 64개씩 나눔
2. 연속 `[K,64]` RHS panel로 pack
3. 각 panel의 local stride는 64이므로 기존 ISA로 정확히 표현 가능
4. 모든 panel과 K tile을 한 source program에 emit
5. ISA field와 C-model을 넓히지 않음

회귀:

- logical RHS stride 65536 경계 초과 GEMM byte-exact
- K=70 multi-K tile byte-exact
- official vocabulary token/HF logits 일치

이 panel lowering은 Gemma 4의 vocabulary 262144에도 반드시 재사용해야 한다.

---

## 10. 해결된 주요 issue

전체 issue는 `report/report_0818.md`의 issue tracker를 권위 기록으로 사용한다. 특히 다음은
다시 발생하기 쉬우므로 새 작업에서 반드시 기억해야 한다.

### V3-020 — RMSNorm FP16 overflow

잘못된 순서:

~~~text
x*x -> reduce -> /D
~~~

올바르게 사용 중인 순서:

~~~text
scaled = x/sqrt(D)
mean = reduce(scaled*scaled)
result = x/sqrt(mean+eps)*weight
~~~

### V3-022 — memory reuse free-list 중복

동일 physical offset이 free-list에 중복 삽입되어 live tensor 충돌이 발생했다. free-list를
offset set으로 변경하고 bump 조건을 수정했다. 이 영역을 refactor할 때 기존 layout/decode
byte-exact regression을 반드시 실행한다.

### V3-023 — vendor G-buffer를 generator만 늘리면 안 됨

vendor executable 내부 배열은 8192 고정이다. input binary만 크게 만들면 BSS 뒤 메모리를
덮어쓴다. vendor path의 capacity guard를 절대 제거하지 않는다.

### V3-025 — high-address scalar broadcast

8K 안에서는 address high half가 0이라 bug가 감춰졌다. expanded plan에서 scalar broadcast가
주소 65535 밖을 읽자 RMSNorm이 거의 0이 되었다.

`Asm.v_broadcast_addr()`는 low/high를 항상 둘 다 emit한다. high가 0이어도 이전 high state를
clear하기 위해 반드시 emit해야 한다.

회귀 주소:

~~~text
source address 70000 -> destination 66000
~~~

### V3-027/V3-028 — wide LM stride

16-bit descriptor field를 source-only로 넓히지 않는다. panel GEMM을 사용한다.

### V3-029 — official decode 연결

prefill graph가 hidden뿐 아니라 roped K/V를 반환하고, decode는 current token K/V projection과
attention/FFN을 분리한다. host는 append만 담당한다.

---

## 11. 남아 있는 일반 backend 과제

### 11.1 V3-004 — vendor GELU semantic

현재 OPEN/HIGH다.

~~~text
vendor GELU = x * sigmoid(2*x)
standard tanh-GELU != vendor GELU
~~~

Llama 3.2는 SiLU를 사용하므로 현재 목표를 막지 않았다. Gemma 4 E2B는
`gelu_pytorch_tanh`를 사용하므로 다음 목표에서는 correctness blocker가 된다.

정책:

- source C-model의 native GELU 의미는 vendor와 동일하게 유지한다.
- Gemma correctness mode에서는 compiler가 표준 tanh-GELU를 primitive sequence로 lowering한다.
- vendor native GELU는 별도의 fast/approximate profile에서만 선택한다.
- FP16 overflow와 0 근처 안정성을 독립 micro-test로 검증한다.

### 11.2 Exact-context decode scalability

현재 `build_v3_decode_layer_module(cfg, context)`는 context마다 다른 static program을 만든다.
짧은 정확성 검증에는 적합하지만 128K context에는 부적합하다.

필요한 후속:

- local attention: 고정 512 window/ring cache 프로그램 재사용
- global attention: K/V chunk와 online softmax state 사용
- indirect addressing 없이 host가 현재 chunk를 고정 G-buffer ABI에 binding
- context 길이마다 전체 graph를 다시 compile하지 않음

### 11.3 Vendor 8192 capacity

vendor executable이 실제로 확장되지 않는 한 full-model execution target으로 사용할 수 없다.
vendor는 작은 kernel/parity oracle로 유지하고 source target이 full-model compute를 담당한다.

---

## 12. 재현 명령

### 12.1 Independent Hugging Face reference

~~~bash
cd /home/chokwans99/NPU_cmodel
NPU_PY=/home/chokwans99/anaconda3/envs/npu-tvm/bin/python

"$NPU_PY" d_compiler/make_v3_generation_reference.py \
  --prompt "Hello, NPU compiler!" \
  --tokens 3 \
  --threads 32 \
  --output d_compiler/build/v3_reference_generate_hello_3.npz
~~~

Reference 생성은 CPU에서 약 73초가 걸렸다.

### 12.2 Existing state resume

~~~bash
cd /home/chokwans99/NPU_cmodel
NPU_PY=/home/chokwans99/anaconda3/envs/npu-tvm/bin/python

"$NPU_PY" d_compiler/run_v3_source_generate.py \
  --prompt "Hello, NPU compiler!" \
  --tokens 3 \
  --reference d_compiler/build/v3_reference_generate_hello_3.npz
~~~

`state.npz`가 존재하면 prompt ID가 같은지 검사한 후 자동 resume한다.

### 12.3 완전 clean rerun

기존 artifact를 삭제하지 말고 새 output directory를 사용한다.

~~~bash
cd /home/chokwans99/NPU_cmodel
NPU_PY=/home/chokwans99/anaconda3/envs/npu-tvm/bin/python

"$NPU_PY" d_compiler/run_v3_source_generate.py \
  --prompt "Hello, NPU compiler!" \
  --tokens 3 \
  --output d_compiler/build/v3_source_generate_hello_clean \
  --reference d_compiler/build/v3_reference_generate_hello_3.npz
~~~

예상 시간은 약 13분 이상이다. 실행 중 `progress.jsonl`과 `state.npz`가 단계별로 갱신된다.

### 12.4 핵심 0818 회귀

~~~bash
cd /home/chokwans99/NPU_cmodel
NPU_PY=/home/chokwans99/anaconda3/envs/npu-tvm/bin/python

PYTHONPATH=d_compiler "$NPU_PY" d_compiler/tests/test_isa_0818.py
PYTHONPATH=d_compiler "$NPU_PY" d_compiler/tests/test_cmodel_0818.py
PYTHONPATH=d_compiler "$NPU_PY" d_compiler/tests/test_backend_0818.py
PYTHONPATH=d_compiler "$NPU_PY" d_compiler/tests/test_source_runtime_0818.py
PYTHONPATH=d_compiler "$NPU_PY" d_compiler/tests/test_source_gemm_0818.py
PYTHONPATH=d_compiler "$NPU_PY" d_compiler/tests/test_v3_source_decode.py
~~~

---

## 13. Gemma 4 E2B 공식 구조 분석

분석 대상은 공식 `google/gemma-4-E2B`/`google/gemma-4-E2B-it`다. Text generation 첫 목표는
instruction-tuned E2B-it가 적합하지만 base와 IT는 같은 text architecture를 공유하므로
compiler 구조는 공통으로 사용할 수 있다.

공식 자료:

- Model/config: <https://huggingface.co/google/gemma-4-E2B/blob/main/config.json>
- Transformers docs: <https://huggingface.co/docs/transformers/model_doc/gemma4>
- Official implementation: <https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py>
- Technical report: <https://arxiv.org/html/2607.02770v2>
- Google overview: <https://ai.google.dev/gemma/docs/core>

### 13.1 E2B text config 핵심값

~~~text
hidden_size: 1536
intermediate_size: 6144
num_hidden_layers: 35
num_attention_heads: 8
num_key_value_heads: 1
head_dim (sliding/local): 256
global_head_dim: 512
vocab_size: 262144
hidden_activation: gelu_pytorch_tanh
rms_norm_eps: 1e-6
sliding_window: 512
num_kv_shared_layers: 20
hidden_size_per_layer_input: 256
vocab_size_per_layer_input: 262144
final_logit_softcapping: 30.0
use_double_wide_mlp: true
tie_word_embeddings: true
max_position_embeddings: 131072
~~~

Layer type pattern은 sliding attention 4개 뒤 full attention 1개가 반복된다. 35 layer이므로
full layer는 7개, sliding layer는 28개다.

RoPE:

~~~text
sliding attention: rope_theta=10000, default RoPE
global/full attention: rope_theta=1000000, proportional/p-RoPE,
                       partial_rotary_factor=0.25
~~~

### 13.2 Llama와 다른 핵심 semantics

| 항목 | Llama 3.2 3B | Gemma 4 E2B |
|---|---|---|
| hidden | 3072 | 1536 |
| layers | 28 | 35 |
| Q/KV heads | 24/8 | 8/1 |
| head dim | 128 고정 | local 256, global 512 |
| attention | full causal | 4 local + 1 global 반복 |
| scale | `1/sqrt(HD)` | QK-Norm 후 scale 1 |
| norm | attention/FFN pre-norm | pre/post RMSNorm + QK-Norm |
| FFN activation | SiLU/SwiGLU | tanh-GELU gated MLP |
| PLE | 없음 | token/context PLE, layer당 256 |
| KV | layer별 독립 | 마지막 20 layer 공유 |
| vocab | 128256 | 262144 |
| logit | raw | tanh softcap 30 |
| modality | text | text/image/audio |

현재 Llama graph의 `H*HD == D` 가정은 Gemma에서 성립하지 않는다.

~~~text
local Q width  = 8 * 256 = 2048 != 1536
global Q width = 8 * 512 = 4096 != 1536
~~~

따라서 기존 graph builder에서 dimension 숫자만 바꾸는 방식은 사용할 수 없다.

### 13.3 Norm과 residual ordering

Gemma decoder layer는 대략 다음 순서다.

~~~text
residual = hidden
x = input_rmsnorm(hidden)
x = self_attention(x)
x = post_attention_rmsnorm(x)
hidden = residual + x

residual = hidden
x = pre_feedforward_rmsnorm(hidden)
x = gated_gelu_mlp(x)
x = post_feedforward_rmsnorm(x)
hidden = residual + x

hidden = PLE residual injection(hidden, per_layer_input)
hidden *= layer_scalar
~~~

Attention 내부에는 Q RMSNorm, K RMSNorm, scale 없는 V RMSNorm도 존재한다. 현재 Llama의 두
RMSNorm 구조와 다르므로 builder와 weight loader 모두 별도여야 한다.

### 13.4 PLE

PLE token-identity embedding logical shape:

~~~text
[vocab_size, num_layers * ple_dim]
= [262144, 35 * 256]
= [262144, 8960]
~~~

전체 table을 G-buffer에 올리지 않는다.

권장 실행:

1. input token row만 safetensors에서 slice
2. `[S,35,256]`으로 view
3. layer별 `[S,256]`만 전달
4. context-aware projection도 `[1536,8960]` 전체 output을 한 번에 만들지 않고
   layer별 `[1536,256]` weight slice로 실행
5. per-layer projection RMSNorm 후 token component와 `1/sqrt(2)`로 결합
6. decoder layer 마지막 gate/projection residual에 사용

Embedding lookup은 host control로 유지할 수 있지만 PLE projection/norm/GELU/residual 산술은
source C-model에서 수행한다.

### 13.5 KV sharing

마지막 20 layer에는 K/V projection weight와 K/V cache가 독립적으로 존재하지 않는다. 이전
non-sharing layer가 만든 state를 attention type별로 재사용한다.

runtime cache를 단순 `cache[layer]` 배열로 유지하면 안 된다. 최소한 다음 metadata가 필요하다.

~~~text
CacheSlot
  owner_layer
  attention_type: sliding | full
  head_dim
  window
  shared_layer_ids
  K storage
  V storage
~~~

공유는 cache copy가 아니라 같은 cache slot alias여야 한다.

### 13.6 Long context

현재 exact-context decode는 128K에서 확장되지 않는다.

권장 lowering:

- sliding layer
  - 512-token 고정 window
  - host가 최신 window slice를 고정 ABI 위치로 전달
  - warmup 이후 동일 source program 반복
- global layer
  - cache를 고정 크기 chunk로 순회
  - running max/sum/weighted-value를 유지하는 online softmax
  - 동일 chunk program 반복
- indirect addressing은 요구하지 않음
- host는 address selection/chunk binding만 담당

이 `LowerAttentionToStreaming`은 Gemma 전용이 아니라 이후 Llama long-context에도 재사용할
공통 optimization pass로 구현하는 것이 좋다.

### 13.7 Final softcap

Gemma logits:

~~~text
logits = 30 * tanh(logits / 30)
~~~

greedy argmax만 필요하면 monotonic transform이므로 생략해도 token은 변하지 않는다.

- greedy/token correctness: `ElideMonotonicLogitTransform`으로 제거 가능
- sampling/top-k 확률/logprob: 제거 불가
- exact logits reference: 제거 불가

### 13.8 Multimodal scope

E2B full model은 text 외에 vision/audio encoder도 포함한다.

권장 순서:

1. text-only prefill/decode
2. vision encoder
3. audio encoder
4. MTP speculative drafter

Vision은 ViT, 2D absolute position, axial 2D RoPE, non-causal attention, pooling이 필요하다.
Audio는 two-stage Conv2D subsampling, 12 Conformer layer, depthwise causal Conv1D, GLU,
relative attention 등이 있어 현재 ISA primitive coverage에서 가장 큰 추가 작업이다. Text와
동시에 시작하면 정확성 문제를 격리하기 어렵다.

---

## 14. 권장 compiler 구조

완전히 새 backend가 아니라 다음 계층 분리가 적합하다.

~~~text
Llama/Gemma checkpoint or torch module
                |
                v
model-family frontend + weight adapter
                |
                v
canonical Relax transformer IR
                |
                v
common canonicalization/optimization passes
                |
                +---- model-specific semantic passes
                |
                v
common npu0818 legalization/memory/codegen
                |
                v
vendor-0818 or source-0818 runtime
~~~

권장 directory 예시:

~~~text
npu_compiler/
├── frontends/
│   ├── llama32.py
│   └── gemma4.py
├── models/
│   ├── llama32/
│   │   ├── config.py
│   │   ├── assets.py
│   │   └── graph.py
│   └── gemma4/
│       ├── config.py
│       ├── assets.py
│       ├── graph.py
│       └── ple.py
├── passes/
│   ├── canonicalize.py
│   ├── attention.py
│   ├── norm.py
│   ├── kv_cache.py
│   ├── memory.py
│   └── gemma4.py
├── targets/
│   └── npu0818/
└── runtime/
    ├── generation.py
    └── cache.py
~~~

실제 refactor는 한 번에 모든 파일을 이동하지 말고 interface를 먼저 추출한 뒤 Llama regression을
통과시키면서 점진적으로 진행한다.

### 14.1 공통 pass 후보

- `CanonicalizeLinear`
- `CanonicalizeRMSNorm`
- `LegalizeStableSoftmax`
- `FuseQKVProjection`
- `FuseOutputProjection`
- `EliminateKVRepeat`
- `RoPECSE`
- `FoldBroadcastConstant`
- `SpecializeStaticShape`
- `ReuseActivationStorage`
- `PackWideRhs`
- `LowerAttentionToTiles`
- `LowerAttentionToStreaming`
- `PlanKVCache`
- `ElideMonotonicLogitTransform`

### 14.2 Gemma-specific pass 후보

- `LowerGemmaPLE`
- `LowerGemmaQKNorm`
- `LowerGemmaPostNorm`
- `LowerSlidingGlobalAttention`
- `LowerPartialProportionalRoPE`
- `PlanSharedKVLayers`
- `LowerDoubleWideMLP`
- `ApplyLayerScalar`
- `LowerGemmaGeluTanh`
- `ApplyFinalLogitSoftcap`

Gemma-specific pass 결과가 일반 matmul/reduction/broadcast/elementwise/attention IR이 된 이후에는
Llama와 동일한 0818 backend를 사용한다.

---

## 15. Gemma 구현 권장 단계와 gate

### Stage G4.0 — branch와 기준선

- `compiler-v3` 최신 commit에서 Gemma 전용 작업 branch 생성
- branch 이름 예시: `gemma4-e2b`
- 기존 26개 회귀 결과 기록
- Llama generation golden artifact/metrics 보존

### Stage G4.1 — model-independent interface 추출

- `Llama32Assets`와 graph/runtime interface 분리
- generic `ModelSpec`, `LayerSpec`, `AttentionSpec`, `CachePlan` 정의
- 실제 동작은 바꾸지 않음
- Llama 26개 test 및 3-token generation 결과 유지

### Stage G4.2 — Gemma asset/reference

- monolithic safetensors와 sharded index 모두 지원하는 generic store
- official config validation
- Gemma tokenizer/chat template
- 독립 HF FP16/BF16 reference artifact 생성
- checkpoint keyset inventory

### Stage G4.3 — primitive correctness

- standard `gelu_pytorch_tanh`
- Gemma RMSNorm
- Q/K/V Norm
- local/global RoPE
- partial rotary factor 0.25
- post-norm ordering
- layer scalar
- PLE lookup/projection

각 primitive를 HF intermediate와 비교한다.

### Stage G4.4 — 단일 layer

- sliding attention owner layer
- global attention owner layer
- shared-KV sliding layer
- shared-KV global layer
- normal FFN와 double-wide FFN

단일 layer output에서 finite, max/mean/RMSE/cosine을 기록한다.

### Stage G4.5 — text-only full prefill

- 35 layer
- final norm
- tied panel LM head 262144 vocab
- HF next-token argmax 일치
- layer checkpoint를 저장하여 drift 시작 layer를 식별 가능하게 함

### Stage G4.6 — short decode

- prefill cache seed
- KV owner/shared alias
- local/global cache semantics
- 3~5 greedy token HF 일치

### Stage G4.7 — scalable decode

- local 512 fixed-window kernel
- global chunked online softmax
- context별 recompile 제거
- short exact-context path와 byte/numeric comparison

### Stage G4.8 이후

- vision
- audio
- quantized checkpoint
- MTP drafter/speculative decoding

첫 번째 사용자-visible 완료 기준:

~~~text
Gemma 4 E2B-it text-only
prompt -> 35-layer prefill -> shared-KV decode -> 3~5 greedy tokens
Hugging Face token sequence와 전부 일치
Llama 3.2 3B regression 유지
~~~

---

## 16. 테스트 전략

새 모델은 full generation만 비교하지 말고 다음 checkpoint ladder를 유지해야 한다.

1. token IDs 및 chat template
2. main embedding scale
3. PLE token component
4. PLE context projection
5. each RMSNorm
6. Q/K/V projection
7. Q/K Norm
8. RoPE local/global
9. attention scores
10. stable softmax probabilities
11. attention output/O projection
12. post-attention norm/residual
13. MLP gate/up/GELU/down
14. post-FFN norm/residual
15. PLE injection
16. layer scalar
17. final norm
18. panel LM logits
19. greedy token
20. K/V cache after each decode position

최소 test matrix:

- random small proxy shapes
- real E2B single layer with official weight
- all-negative reduce max
- high address broadcast >65535
- wide logical RHS stride >65535
- local attention boundary: context 511/512/513
- global attention chunk boundary
- KV sharing owner/shared pair
- PLE repeated token IDs
- GELU negative/zero/positive/extreme inputs
- FP16 finite/overflow test

---

## 17. 새 세션 시작 체크리스트

새 세션은 다음 순서로 시작한다.

1. 이 문서 전체 읽기
2. `report/report_0818.md` 전체 또는 적어도 issue tracker와 Stage V3.7~V3.10 읽기
3. `d_compiler/RETARGET_0818.md` 읽기
4. `git status --short --branch` 확인
5. HEAD가 `7c91647` 이후인지 확인
6. 사용자 변경/dirty file 확인 후 보존
7. `/home/chokwans99/anaconda3/envs/npu-tvm/bin/python` 사용
8. 핵심 source/backend 회귀 실행
9. Gemma 구현을 시작하기 전 branch 생성 여부를 사용자 지시와 맞춰 결정
10. Llama 전용 코드를 바로 삭제하거나 대규모 이동하지 말고 interface 추출부터 진행

새 세션에 전달할 짧은 시작 문구 예시:

~~~text
먼저 report/SESSION_HANDOFF_0819.md 전체와 report/report_0818.md의
Stage V3.7~V3.10을 읽어라. 현재 compiler-v3의 Llama 3.2 3B source prefill/decode
결과를 회귀 기준으로 보존하면서, Gemma 4 E2B text-only 지원을 위해 model-independent
frontend/pass/runtime interface를 설계하고 Stage G4.0부터 진행하라. vendor/source C-model의
capacity 외 arithmetic/quirk parity는 절대 변경하지 마라.
~~~

---

## 18. 권위 문서 및 파일 index

| 파일 | 역할 |
|---|---|
| `report/report_0818.md` | 전체 issue tracker, 단계별 수치, 최종 PASS 판정 |
| `d_compiler/RETARGET_0818.md` | 0818 ISA/vendor contract 및 layout 결정 |
| `_poc/mysim_0818.cpp` | vendor-compatible source C-model |
| `d_compiler/npu_compiler/isa_0818.py` | ver.08 bit-exact assembler |
| `d_compiler/npu_compiler/runtime_0818.py` | fixed vendor executable runtime |
| `d_compiler/npu_compiler/source_runtime_0818.py` | dynamic source runtime |
| `d_compiler/npu_compiler/backend_0818.py` | row-major Relax-to-0818 backend |
| `d_compiler/npu_compiler/legalize.py` | RMSNorm/RoPE/softmax/SwiGLU primitive builder |
| `d_compiler/npu_compiler/model.py` | 현재 Llama-specific graph builder |
| `d_compiler/npu_compiler/v3_model.py` | official Llama asset loader |
| `d_compiler/npu_compiler/v3_source_llama.py` | full source prefill/decode orchestration |
| `d_compiler/npu_compiler/source_gemm_0818.py` | 16-bit-safe RHS panel GEMM |
| `d_compiler/run_v3_source_generate.py` | resumable source generation runner |
| `d_compiler/make_v3_generation_reference.py` | independent HF reference generator |
| `d_compiler/tests/test_source_runtime_0818.py` | dynamic source/high-address regression |
| `d_compiler/tests/test_source_gemm_0818.py` | panel/stride regression |
| `d_compiler/tests/test_v3_source_decode.py` | source prefill/decode proxy regression |

---

## 19. 최종 상태 판정

현재 상태:

| 목표 | 상태 |
|---|---|
| 0818 vendor/source arithmetic 및 quirk parity | PASS |
| source dynamic G-buffer/program | PASS |
| Llama 3.2 3B official 28-layer prefill | PASS |
| full vocabulary LM head | PASS |
| prefill K/V cache seed | PASS |
| autoregressive decode context 7→8→9 | PASS |
| HF greedy IDs `[358,2846,4560]` | PASS |
| decoded text `" I'm trying"` | PASS |
| repository 26개 regression entrypoint | PASS |
| Gemma 4 E2B implementation | NOT STARTED — architecture analysis complete |

다음 작업의 가장 중요한 판단은 **기존 backend를 버리는 것이 아니라, Llama에 고정된 상위
graph/assets/runtime를 model-family adapter와 공통 pass 구조로 분리하는 것**이다. 이 과정에서
Llama 결과가 깨지지 않는 것이 각 refactor 단계의 필수 gate다.
