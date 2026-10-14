# 12. 재현 방법, 지원 범위와 남은 검토

[목차](../README.md) · 이전: [실험](11_experiments.md) · 별도 후반부: [양자화](13_quantization.md)

기본 FP16 설명은 여기까지다. 실험을 재현할 때는 작은 예제부터 시작하고,
소스·환경·설정을 기록한다. 기존 측정값이 현재도 같을 것이라고 가정하여 재사용하지 않는다.

## 검증 환경

TVM 0.19.0, commit `c4dc0c29ff81ddae688da24625a603da3b4a2c0e`, LLVM 18.1.8,
Python 3.11.15가 기준이다. 나머지 핵심 버전은 [toolchain.lock.json](../../toolchain.lock.json)에 있다.
이 파일은 conda 전체 solver lock이 아니다. 이미 구성된 `npu-tvm` 환경을 사용한다.

```bash
conda activate npu-tvm
bash d_compiler/npu_codegen/build.sh
python d_compiler/docs/examples/reproduce.py --out /tmp/npu-doc-examples
python d_compiler/docs/tools/check_docs.py
```

Native build 명령은 library가 없거나 소스와 맞지 않을 때 필요하다. 작은 예제 runner는
전체 Llama나 checkpoint 다운로드를 실행하지 않는다. `--case pipeline` 또는
`--case memory`로 한 실험만 선택할 수 있다. 보존된 문서 예제를 덮어쓰지 않도록
재현 결과는 별도 디렉터리에 둔다.

실제 checkpoint를 준비한 뒤 사용하는 generation 명령은 다음과 같다.

```bash
python d_compiler/run_nn_generate.py --model llama \
  --prompt 'Hello, NPU compiler!' --tokens 3 --capacity 9
```

이 명령은 전체 모델을 컴파일·실행할 수 있으므로 작은 문서 검사와 다르다.
현재 generation CLI는 Llama를 대상으로 한다. 다른 model-family frontend의 존재를
같은 CLI의 전체 모델 검증 완료로 확대하지 않는다.

## API와 설정

```python
from npu_compiler import npu_target
from npu_compiler.npu_compilation import compilation

with compilation(progress=lambda event: print(event, flush=True)):
    executable = npu_target.build(module, func_name="prefill", fusion="auto")

# values는 최종 signature에 맞춘 이름 → 배열이다.
output, counters = executable.run(values)
executable.export_bundle("/tmp/npu-bundle", values)
```

`build_image`나 `run`이 raw checkpoint의 모든 변환을 자동 추측하는 것은 아니다.
Host transform과 `_bind_inputs` 역할은 [3장](03_graph_host.md)과 [10장](10_runtime_llama.md)을 따른다.

09-29에 추가한 MetaSchedule 기록 재사용은 바깥쪽 compilation context에서 켠다.
첫 호출에서 기존 탐색 결과를 저장하고, 다음 context/build에서는 같은 조건의 기록을
재적용한다. 기본 컴파일 경로와 탐색 정책을 일괄 교체하는 옵션은 아니다.

```python
with compilation(schedule_database="d_compiler/build/schedule_db"):
    executable = npu_target.build(module, func_name="prefill", fusion="auto")

print(executable.plan.compilation["meta_schedule"])
```

저장된 결과를 기준으로 더 탐색하려면 `schedule_database_mode="tune"`을 지정한다.
기본 `reuse`에서는 기록이 있으면 탐색하지 않고 재검증하여 적용한다.
탐색 전략·seed·예산이 달라도 실행 조건이 같으면 기록을 사용할 수 있으며,
shape·dtype·가용 SRAM·scope·scratch·수치 정책·소스 버전 등은 계속 구분한다.

```python
with compilation(schedule_database="d_compiler/build/schedule_db",
                 schedule_database_mode="tune"):
    asm, plan = npu_link.compile_program(semantic_module, profile=tuning_profile)

# 다음 build: 같은 실행 조건에서는 작은 예산으로도 위 결과를 재사용한다.
with compilation(schedule_database="d_compiler/build/schedule_db"):
    asm, plan = npu_link.compile_program(semantic_module, profile=fast_profile)
```

중첩된 linker/fusion 호출은 바깥 context의 DB를 공유한다. 중첩 context에서 다른 DB를
지정할 수 없다. 새 버전은 별도 하위 디렉터리를 사용하며 이전 버전 기록을 적용하지 않는다.
빠른 검사만 하려면 `python d_compiler/tests/test_meta_schedule.py`를 실행한다.
설계·검증 범위는 [6장](06_matmul_schedule.md)의 MetaSchedule 절을 따른다.

기존 탐색과 실제 MetaSchedule 탐색의 Llama 비교는 다음 실행기를 사용한다.
로컬 Llama 3.2 3B checkpoint와 native backend/C-model이 준비돼 있어야 한다.
각 커널의 두 방식 컴파일과 출력 검증 합계가 90초를 넘으면 해당 worker와 자식 프로세스를
종료한다. 기본 세 커널과 SRAM 압박을 주는 512-token 커널을 분리해서 실행할 수 있다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python d_compiler/docs/examples/compare_llama_search.py \
  --out d_compiler/build/llama_meta_schedule --attempts 16 --trials 4 --seed 0 --timeout 90

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python d_compiler/docs/examples/compare_llama_search.py \
  --cases prefill_down_512 --reuse-check --out d_compiler/build/llama_meta_schedule_pressure \
  --attempts 16 --trials 4 --seed 0 --timeout 90
```

실행 중에는 worker별 log와 `progress.json`, 끝나면 `comparison.json`을 확인한다.
Semantic TIR, 두 방식의 binary, 후보별 거절 이유·비용·시간·출력 오차도 남긴다.
이 명령은 전체 모델 생성 검증을 하지 않는다. `--reuse-check`는 각 방식의 튜닝 결과를
별도 DB에 저장하고, 전략·seed·예산을 바꾼 새 context에서 같은 명령이 재생되는지 확인한다.
이 비교에는 새 output 디렉터리를 사용해야 한다. 개선 전 고정 예산 결과는
[E8](11_experiments.md#e8-metaschedule로-탐색하면-llama-커널이-더-좋아지는가),
개선 후 결과는 [E9](11_experiments.md#e9-긴-입력의-탐색과-튜닝-결과-재사용-개선)에 있다.

이미 legalization한 module을 linker에서 비교하려면 profile로 선택한다.
이 경로에서도 기존 탐색이 기본값이며 DB 저장 여부는 별도 옵션이다.

```python
from dataclasses import replace
from npu_compiler import npu_link
from npu_compiler.device import PROFILES
from npu_compiler.npu_compilation import compilation

profile = replace(PROFILES["v09"], schedule_search="meta_schedule",
                  schedule_seed=0, schedule_max_attempts=16, schedule_space_trials=4)
with compilation():
    asm, plan = npu_link.compile_program(semantic_module, profile=profile)
```

| 옵션 | 기본 동작 | 오해하기 쉬운 점 |
|---|---|---|
| fusion auto | 합법 기준과 여러 graph 후보 비교 | 전역 최적 보장 아님 |
| fusion off | attention/generic fusion을 끔 | 메모리 최적화까지 전부 끄는 것은 아님 |
| fusion on | pattern/generic fusion 요청 | 지원되지 않는 복합 kernel은 실패할 수 있음 |
| joint_memory | 중간 FP16 tensor 전체 보관과 schedule 함께 선택 | 부분 tile 보관이나 runtime eviction 아님 |
| reuse_sram | 커널 내부 수명에 따른 공간 재사용 | 커널 사이 보관과 다른 옵션 |
| graph_search_seconds=120 | 첫 합법 모듈 뒤 추가 탐색 | 전체 compile timeout 아님 |
| schedule_max_attempts=64 | 성공/실패/공간 부족을 포함해 시도 제한 | 64개의 성공 후보를 보장하지 않음 |
| schedule_search="space" | 예상 DMA 순서의 탐색 | 실험용 "meta_schedule"은 ReplayFunc 추첨, 학습형 탐색 아님 |
| schedule_seed=0 | MetaSchedule의 재현 가능한 추첨 seed | seed 하나의 결과로 전체 탐색 전략의 우열을 보장하지 않음 |
| schedule_probe_word_budget=100000 | 기준 words×성공 후보 목표 수와 비교해 큰 쪽 사용 | 설정값 자체가 항상 엄격한 상한은 아님; 후보 완료 후 검사 |
| cache_buckets | 미리 만든 capacity별 decode 선택 | 매 step 임의 shape의 동적 실행 아님 |

## 전체 Llama의 fusion 탐색 시간을 해제해서 비교하기

09-29 실행은 큰 cache fusion의 Python 명령 생성에서 메모리를 과도하게 사용하여
중단했다. [E10](11_experiments.md#e10-256-token-전체-prefill의-fusion-탐색)의 수정 후
독립 검사는 통과했지만 전체 256-token 모델은 다시 검증하지 않았다. 아래 명령을
이미 완료된 실험으로 해석하지 않는다. `--compile-memory-gib`는 컴파일 과정의 Linux
주소 공간 상한이며 기본 24 GiB다. 장치 SRAM이나 graph 탐색 시간과 별개다.

다음 실행은 Llama 3.2 3B 전체 28개 layer의 **256-token prefill 하나만** 컴파일한다.
기본 120초 제한을 해제하고 현재 구현이 생성하는 graph 후보를 끝까지 처리한다.
분리 커널에도 SRAM 전달과 joint-memory 탐색을 적용한다. 두 조건 중 하나만 메모리
최적화를 끄는 비교가 아니다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python d_compiler/docs/examples/compile_llama_full_search.py \
  --out d_compiler/build/llama_prefill256_full_search --tokens 256 --trials 16 --attempts 128
```

실행기는 `graph_search_seconds=None`, `schedule_probe_word_budget=1000000000`을
사용한다. Matmul의 성공 후보 목표 16개와 최대 시도 128회는 계속 적용되므로 schedule 전수
탐색은 아니다. 기준 schedule 탐색은 별도다. `request.json`에 실제 profile과 소스
hash, 입력 token을 저장하고 `events.jsonl`에 진행 경계를 기록한다.

`selection.json`은 각 graph의 비용·영역 선택·탈락 사유이며 `schedule_searches.json`은
최종 승자뿐 아니라 거절한 graph와 영역 검사에서 생성한 recipe의 탐색 기록이다.
`compiled.pkl`과 명령 binary는 재컴파일 없는 출력 검증에 쓴다. 이 pickle은 직접
생성한 로컬 파일만 읽는다. `--validate-only`는 그 결과를 사용하며 compiler 소스가
달라졌으면 실행을 거절한다. `--compile-only`는 출력 검증을 생략한다.

출력 검증은 같은 256개 token을 HF와 C-model에 넣고 마지막 위치의 전체 logits,
최대 logit의 token, layer별 K/V cache 값과 DMA counters를 확인한다. Decode나 연속 token 생성의 검증으로
해석하지 않는다. SRAM/DMA/words 비교는 정적 비용이며 실장치 latency 측정은 아니다.

## 현재 지원 경계

| 항목 | 지원 또는 제한 |
|---|---|
| Shape | 정적 tensor shape. FP16 matmul의 rank≥2, batch broadcast, unit/ragged 축. Rank1은 명시적 reshape 필요 |
| Control flow | 지원된 정적 호출을 직선 stream으로 전개. 임의 dynamic control flow/packed call은 거절 |
| I/O | Tensor와 nested tuple, dtype별 byte 크기, typed output tree |
| View | 지원되는 연속 reshape/split/offset. 임의 strided slice나 모든 INT8 byte offset이 zero-copy는 아님 |
| FP16 | Matmul/sum FP32 누산과 정의된 FP16 저장. 범용 FP32 vector tensor 실행은 별도 |
| Softmax | 마지막 축, 정적 길이 1..65504, 유한 logits를 전제로 한 계약 |
| RMSNorm | 마지막 축, FP16 입력/weight, 정적 양의 width, 지원 epsilon |
| Attention | 지원되는 정적 FP16 layout/bias/scale/causal/window. Full-fusion 조건은 더 좁음 |
| RoPE | default/linear/llama3의 지원 필드. 다른 설정은 확인 없이 허용하지 않음 |
| Cache | FP16 정적 capacity, additive mask, versioned append metadata |
| Runtime | C-model adapter와 image/file patch. RTL/실장치 runtime 검증 아님 |
| Snapshot | 관찰해야 하는 중간값 때문에 SRAM-only 전달을 제한하거나 끔 |

ABI verifier는 정의·호출·배치의 일관성을 검사하지만 임의 TIR의 모든 index 접근이
안전함을 증명하는 일반 정리 증명기는 아니다. 지원 subset 밖의 계산을 조용히 통과시키지 않는다.

## 오류가 나면 어느 단계부터 보는가?

| 관찰한 증상 | 먼저 확인할 것 |
|---|---|
| LLVM부터 HF와 크게 다름 | 모델 수식, weight 순서, mask/RoPE, 정밀도 정책 |
| 작은 K는 맞고 큰 K가 틀림 | FP32 accumulator 연결과 부분합 순서 |
| 첫 타일만 맞고 뒤 타일이 틀림 | MAIN stride, PARTIAL 주소, dtype/주소 단위 |
| SRAM 보관을 켤 때만 틀림 | 값 버전, 마지막 소비, alias, 덮어쓰기 |
| 주소 압축 후 출력/cache가 틀림 | DMA와 output/constant/append 주소의 일관성 |
| Prefill은 맞고 다음 decode가 틀림 | 현재 token, position, mask, cache append 슬롯 |
| Words가 너무 많음 | 짧은 vector, 주소 설정 반복, 정적 loop 전개 |
| Compile이 오래 걸림 | trial/entry/region cache, 거절 이유, phase 시간, 종료 한도 |

Native/Python 일치는 두 구현이 같은 계약을 따른다는 근거다. 둘이 공유한 모델 수식의
오류까지 배제하지는 못하므로 HF/NumPy 또는 LLVM 의미 기준을 함께 확인한다.

```bash
bash d_compiler/run_gate.sh --tests architecture_contracts schedule_memory native_codegen \
  --require-no-skips --output /tmp/npu-gate
```

전체 gate는 작은 선택 검사보다 시간이 길다. 실제 subprocess exit, timeout, skip을
`validation.json`으로 확인한다. `[PASS]` 문자열만 찾아 성공으로 판정하지 않는다.

## 남은 이슈: 구현 착수 목록이 아닌 검토 목록

사용자는 효과가 작거나 custom 복잡도가 큰 변경을 원하지 않는다. 아래 항목은
기존 구현을 유지한 상태에서 필요성과 규모를 먼저 평가한다. 이번 문서 작업에서
compiler 최적화를 추가 구현하지 않았다.

| 항목 | 현재 한계 | 재개할 때 먼저 확인할 것 |
|---|---|---|
| Vector 256 / 긴 VLEN | 문서의 과거 상한 설명과 후속 결정 충돌, C-model은 물리 버퍼 모형 아님 | 실제 장치의 LOAD/연산/SAVE·유효 길이·버퍼 계약. 256 제한이면 공통 lowering과 reduction 설계 |
| Vector schedule/비용 | 주로 용량에 맞춘 행 축소, cycle·lane 활용률 미반영 | 대표 kernel의 짧은 명령과 SRAM 임시값 비용, 간단한 공통 변환의 효과 |
| 작은 fusion 영역 | 큰 TVM 영역 채택/해제 중심 | 일부만 합치면 실제 최종 프로그램이 좋아지는지 제한된 비교 |
| ISA activation | 자동 matmul activation OFF | 마지막 K, bias 순서, FP16 반올림과 품질 차이 |
| Attention | Query 분할, key online softmax 없음 | 장치 누적 상태와 긴 context에서의 예상 이득. 큰 구현이면 설계부터 |
| Decode 길이 | Capacity 전체 계산과 append 전송 | Bucket 효과, 실제 빈 슬롯 낭비, KV 최종 위치 직접 저장의 주소 계약 |
| 메모리 | Whole tensor 보관, 고정 호출 순서, 일부 조각화 | 보관 포기 사례의 실제 손실. 실행 순서/퇴거/부분 tile은 큰 변경으로 분리 |
| 큰 generic fusion | 반복 계산이 증명되는 `compute_at` 배치를 거절 | 분기별 생산자 배치, 큰 명령 stream의 저장·후처리 비용을 별도 개선. 256-token 전체 출력은 재검증 필요 |
| DRAM 압축 | 완전 미사용 pool 제거 | 사용 중 pool 내부 낭비가 의미 있는지 확인 후 수명 재분석 검토 |
| 병렬 matmul 결합 | QKV/gate-up 표준 pass 효과 미확인 | CombineParallelMatmul과 split/layout 비용까지 포함해 비교 |
| DLight/MetaSchedule | DLight 미사용. MetaSchedule trace DB와 ReplayFunc는 선택형 지원 | 여러 seed·실제 비용 개선과 탐색 비용을 비교. 학습형 CostModel은 별도 |
| 정밀도 선택 | 고정 계약, 자동 오차 예산 탐색 없음 | 허용 오차·모델 품질 기준을 먼저 정함 |
| 성능 모델 | DMA→words→SRAM 비교 | RTL/장치 timing을 확보한 뒤 상관 확인 |
| 양자화 generation | 공개 generate에 선택 미연결 | 후반부 범위부터 검증, KV 양자화까지 완료로 확대하지 않음 |

## 코드를 찾는 지도

| 책임 | 파일 |
|---|---|
| 공개 build / 결과물 | [npu_target](../../npu_compiler/npu_target.py) |
| 모델 / checkpoint | [nn_models/](../../npu_compiler/nn_models/), [assets/](../../npu_compiler/assets/) |
| Graph / 후보 | [tvm_pipeline](../../npu_compiler/tvm_pipeline.py), [npu_build](../../npu_compiler/npu_build.py), [npu_fusion](../../npu_compiler/npu_fusion.py) |
| 수치 / schedule | [npu_legalize](../../npu_compiler/npu_legalize.py), [npu_schedule](../../npu_compiler/npu_schedule.py), [npu_intrin](../../npu_compiler/npu_intrin.py) |
| 메모리 / ABI | [npu_memplan](../../npu_compiler/npu_memplan.py), [npu_memory_search](../../npu_compiler/npu_memory_search.py), [npu_sram](../../npu_compiler/npu_sram.py), [npu_abi](../../npu_compiler/npu_abi.py) |
| ISA / 실행 | [npu_link](../../npu_compiler/npu_link.py), [npu_codegen/](../../npu_codegen/), [npu_generate](../../npu_compiler/npu_generate.py), [v09_runtime](../../npu_compiler/v09_runtime.py) |

처음부터 파일 순서로 읽기보다는, 관심 있는 현상이 어느 단계의 결정인지 먼저 정하고
해당 예제의 IR/metadata에서 실제 선택을 확인한다.
