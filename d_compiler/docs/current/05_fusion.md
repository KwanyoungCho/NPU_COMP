# 5. Fusion 후보를 만들고 선택하는 과정

[목차](../README.md) · 이전: [TIR](04_semantic_tir.md) · 다음: [Schedule](06_matmul_schedule.md)

Fusion은 연결된 계산을 하나의 실행 커널로 묶는 것이다. 커널 수가 줄어도 반드시
빠르거나 DMA가 적어지는 것은 아니다. 현재는 **실제로 생성한 프로그램의 비용**으로
비교하며, SRAM 전달이 가능한 분리 커널도 기준에 포함한다.

## 예제의 세 구성

```text
A  [matmul] → [add] → [SiLU]
B  [matmul] → [add + SiLU]
C  [matmul + add + SiLU]
```

C는 생산자와 소비자를 함께 배치하기 쉽다. 하지만 많은 임시값 때문에 SRAM이
부족해지거나, 행렬 타일마다 64원소짜리 SiLU를 반복하여 명령이 길어질 수 있다.
A/B에서도 중간 tensor를 SRAM에 계속 두면 DRAM 저장·재읽기를 제거할 수 있다.
따라서 fusion의 이득을 “중간 tensor 크기만큼 DMA가 무조건 줄어든다”로 계산하지 않는다.

09-29에는 큰 cache fusion에서 `compute_at`이 같은 생산자 계산을 반복하는 문제를
확인했다. 현재 generic 행 분할은 TVM `arith.Analyzer`로 이동 전후의 반복 횟수 범위를
검사한다. 이동 후의 최소 횟수가 이동 전의 최대 횟수보다 크면, 그 스케줄은 명령을
만들기 전에 거절하고 해당 fusion 영역의 분리된 기준 커널을 유지한다. TVM이 만든
fusion 후보 전체를 끄거나 탐색 시간을 줄이는 조치가 아니다. 반복 계산과 전송 절감의
교환까지 최적화하는 스케줄은 현재 지원하지 않는다. [재현과 검증](11_experiments.md#e10-256-token-전체-prefill의-fusion-탐색).

## 표준 TVM은 무엇을 보고 묶는가?

`AnnotateTIROpPattern`은 TIR의 연산 성격을 분석해 `op_pattern`을 붙인다.
TVM 0.19의 pattern 종류는 다음과 같다. 연속된 모든 정수에 종류가 있는 것은 아니다.

| 값 | 이름 | 대표 의미 |
|---:|---|---|
| 0 | kElemWise | 같은 위치끼리 원소별 계산 |
| 1 | kBroadcast | 크기가 작은 입력을 여러 위치에서 사용 |
| 2 | kInjective | transpose 등 원소 대응이 명확한 변환 |
| 3 | kCommReduce | 여러 값을 합/최댓값 등으로 축소 |
| 4 | kOutEWiseFusable | matmul처럼 출력 뒤 원소별 계산을 붙일 수 있는 복합 연산 |
| 7 | kTuple | tuple 구성과 관련된 분류 |
| 8 | kOpaque | 일반 fusion의 경계로 취급 |

FuseOps는 단순히 이 숫자가 같다는 이유로 묶지 않는다. 데이터 의존 그래프와
post-dominator 관계, 경로상의 pattern, 이미 합쳐진 group의 제약을 이용한다.
Post-dominator는 “이 값에서 아래로 이어지는 경로들이 다시 만나는 지점”을 찾는 데
사용하는 그래프 개념이다. 서로 무관한 두 matmul을 이름이 같다는 이유로 합치지 않는다.
장치에서 빠른지, SRAM에 들어가는지는 이 표준 분류만으로 알 수 없다.

NPU 쪽은 지원하지 않는 matrix 복합 영역과 W8A8 등을 opaque 경계로 둘 수 있다.
따라서 영역 생성은 TVM을 사용하지만, 장치의 지원 경계와 채택 판단까지 전부
TVM 기본 규칙에 맡긴 구조는 아니다.

### 실제로 어떤 연결을 검사하는가?

설치된 TVM의 graph partitioner는 여러 차례 그래프를 훑는다. 먼저 복잡한 연산의
출력 뒤에 붙일 원소별 계산을 처리하고, injective 연결과 tuple 관련 연결을 후속
차례에서 처리한다. 대표 조건은 다음과 같다.

| 출발 group | 검사하는 경로/도착점 | 예를 읽는 방법 |
|---|---|---|
| OutEWiseFusable | 도착 관계가 elemwise이고 그 사이가 elemwise/broadcast인지 검사 | matmul 출력에 add/activation을 붙일 후보 |
| Elemwise/Broadcast | 중간 경로가 injective 이하이고 도착점이 허용 pattern인지 검사 | add→SiLU, 원소별 전처리→reduction 등 |
| Injective/Tuple | 후속 차례에서 injective 이하 경로를 검사 | 지원되는 reshape/transpose와 후속 계산 연결 |
| CommReduce | 이 분기에서 임의로 다른 복합 연산에 붙이지 않음 | reduction 둘을 무조건 하나로 합치지 않음 |
| Opaque | 여기서 일반 fusion을 진행하지 않음 | NPU가 보호한 경계 |

이미 합친 group의 성격, 함수 인자 수, 합칠 node 수 한도도 확인한다. 가지가 나뉘면
**중간의 모든 관련 경로**가 조건을 만족해야 한다. 예를 들어 P를 SiLU뿐 아니라
함수 출력으로도 반환한다면, `P→SiLU`만 보고 P의 독립적인 관찰을 없애면 안 된다.
위 표는 주요 분기를 설명한 것이며 arbitrary graph의 합법성을 수작업으로 판정하는
대체 규칙은 아니다. 실제 annotation·group·외부 사용 관계를 TVM이 함께 검사한다.

작은 예제에서 TVM이 C를 후보로 만들 수 있어도 B까지 반드시 별도로 생성하는 것은
아니다. 현재 backend는 큰 영역을 받아 합침/펼침을 비교하는 비중이 크다. “A/B/C 모두
비교 가능”이라는 개념 설명과 “현재 모든 부분 영역 조합을 탐색함”을 구분해야 한다.

## FuseOps와 FuseTIR의 차이

```text
LegalizeOps 뒤: call_tir A → call_tir B → call_tir C
FuseOps 뒤:    이 호출들을 담은 작은 Relax 함수
FuseTIR 뒤:    작은 함수의 TIR 계산들을 하나의 PrimFunc로 결합
```

FuseOps는 graph의 그룹을 정하고 FuseTIR는 실제 TIR 계산을 합친다. 이 과정만으로
중간 buffer가 SRAM에 배치되지는 않는다. 뒤의 schedule과 메모리 계획이 필요하다.

Attention은 Relax 연산 pattern이 보일 때 `FuseOpsByPattern`으로 먼저 그룹을
정할 수 있다. 그 후 LegalizeOps/FuseTIR를 적용한다. 일반 fusion은 legalize 뒤
pattern annotation을 보고 동작한다. 두 경로는 서로 배타적인 두 compiler가 아니다.
같은 후보에서 attention composite와 주변 generic fusion이 함께 존재할 수 있다.

## 부분 검사와 전체 비교는 왜 둘 다 필요한가?

| 단계 | 범위 | 알고 있는 것 | 모르는 것 |
|---|---|---|---|
| `npu_fusion.select` | TVM이 만든 fusion 영역 하나 | 영역 안의 unfused/fused 계산과 실제 codegen 비용 | 주변 전체 그래프의 모든 수명·다른 선택과의 조합 |
| `npu_build.lower_and_select` | 요청한 전체 entry들 | 전체 호출의 SRAM 보관·주소·최종 명령과 공유 ABI | 검사하지 않은 graph 조합의 최적성 |

부분 검사도 해당 영역을 schedule·메모리·명령 생성까지 내려 비용을 얻는다. 분리된
기준에도 SRAM 전달을 적용한다. 합친 결과가 불가능하거나 더 비싸면 그 영역을 다시
펼친다. 이렇게 남은 영역들로 전체 모듈을 만들고 전체 entry를 비교한다.

일부 작업이 겹치지만 질문이 다르다. “이 영역을 합칠 가치가 있는가?”와
“주변을 포함한 프로그램을 이렇게 선택해도 좋아지는가?”를 나눠 확인한다.
동일 영역·entry의 중복 계산은 build 안의 cache로 줄인다. 다만 부분 검사에서
버린 영역을 모든 다른 영역 조합과 함께 다시 살펴보는 완전 탐색은 하지 않는다.

## 전체 후보와 선택 기준

Attention은 off/score/full, GQA·producer 변환은 세 구성, generic fusion은
off/vector/matmul-epilogue 구성을 조합한다. 원시 조합은 최대 27개지만 동일한 IR은
제거하고, 실패·비경쟁 후보·시간 한도에 따라 실제 검사는 더 적다.

각 entry의 비용은 `(DMA bytes, ISA words, SRAM peak)`의 사전식 순서다.
앞 값이 작으면 뒤 값보다 우선한다. Hardware latency 모델은 아니다.
Prefill/decode를 함께 선택할 때 모든 entry가 나빠지지 않고 적어도 하나가 좋아지는
모듈만 채택한다. Prefill은 A, decode는 ABI가 다른 B에서 임의로 가져오지 않는다.

한 entry가 이미 더 나쁘면 남은 entry를 컴파일해도 해당 후보가 이길 수 없으므로
조기 종료한다. 이 경우 기록된 비용은 부분 결과이며 전체 Pareto 후보로 취급하지 않는다.

### 실제 256-token prefill에서는 몇 개가 나오는가?

28-layer Llama 3.2 3B의 `prefill_cache`에 graph 변환만 적용해 확인한 결과다.
GQA·producer 옵션을 바꿔도 같은 IR이 되는 조합을 제거하면 attention 구성 3개가 남는다.
여기에 generic 구성 3개를 적용해 **전체 graph 구성은 9개**다.

| Attention 구성 | Generic off | Generic vectors | Generic matmul |
|---|---|---:|---:|
| off | 일반 fusion 없음 | 영역 10종 | 영역 11종 |
| score | 일반 fusion 없음 | 영역 9종 | 영역 10종 |
| full | 일반 fusion 없음 | 영역 9종 | 영역 10종 |

표의 영역 수는 TVM이 묶은 **서로 다른 함수 종류**다. 28개 layer의 호출을 모두 더한
숫자나 최종 채택 수가 아니다. 예를 들어 gate→SiLU→multiply, down→residual add,
RoPE의 reshape·multiply·add 등이 후보가 된다. Attention 자체의 pattern 영역도
generic 영역 수와 구분한다. Off 열은 attention까지 반드시 끈다는 뜻이 아니다.

이후 영역별 비용 검사에서 일부가 풀릴 수 있고, 전체 module도 같은 결과가 되면
중복으로 제거될 수 있다. 따라서 **9개 구성 생성**, **영역 합치기 허용**, **최종 fusion
모듈 선택**은 서로 다른 단계다. 현재 실험의 최종 선택은 측정 중이다.
[구성 목록과 함수 이름](../evidence/current/llama_prefill256_full_search_progress.json),
[실험 E10](11_experiments.md#e10-256-token-전체-prefill의-fusion-탐색)을 함께 본다.

## 컴파일 시간이 왜 제한되는가?

기본 graph 예산 120초는 **최초 합법 모듈을 완성한 뒤의 추가 탐색 시간**이다.
최초 prefill/decode 생성, 실행, parameter load까지 120초 안에 끝난다는 뜻이 아니다.
검사는 작업 경계에서 이뤄져 긴 TVM 작업 하나만큼 초과할 수 있다.

Schedule 후보는 실패·공간 부족도 시도로 세며 기본 64회 한도를 적용한다.
동일 계산은 TVM structural hash와 equality, profile·입출력 배치 등으로 확인해
재사용한다. Shape만 같으면 같은 결과로 취급하는 cache가 아니다. 반복 layer에서도
실제 weight 주소가 들어간 명령은 호출별로 확인한다.

과거 7-token 입력의 전체 28-layer 검증 기록은 시간 한도 내에서 unfused 기준 모듈이 선택됐다.
이는 모든 fusion 후보가 더 나쁘다는 증거가 아니다. [실험 장](11_experiments.md)에서
검사 완료 범위와 결과를 분리한다.

### 시간을 충분히 주는 실험에서는 무엇을 바꾸는가?

`graph_search_seconds=None`이면 시간 때문에 다음 graph 후보를 포기하지 않는다.
중복 IR 제거, 지원하지 않는 후보의 거절, 이미 비용이 더 큰 후보의 탈락은 그대로다.
이들은 시간을 아끼기 위해 임의로 후보 수를 잘라내는 것과 구분해야 한다.
현재 구현이 만드는 최대 27개 조합을 끝까지 처리해도 **임의의 모든 부분 fusion 조합을
탐색한 것은 아니다**. TVM이 만든 영역을 부분 검사에서 거절하면 그 영역의 다른
조합을 전체 그래프에서 다시 탐색하지 않는 제한도 남는다.

또한 graph 시간과 schedule 후보 수는 서로 다른 한도다. Graph 시간만 풀어도
matmul당 기본 성공 후보 8개·최대 시도 64회는 그대로이므로, 확장 실험에서는 이 값도
명시해야 한다. 256-token prefill 재현 실행기는 성공 후보 목표를 16개, 시도를
128회로 늘리고 words 예산도 충분히 크게 둔다. 실패를 포함한 시도 한도에 도달하면
16개를 모두 확보하지 못할 수 있으므로 각 schedule의 종료 사유도 기록한다.

반복되는 28개 layer를 각각 처음부터 탐색하지 않는다. 같은 계산·같은 메모리 조건의
recipe와 trial은 재사용하고, layer별 주소와 전체 tensor 수명을 반영한 최종 명령은
전체 모델에 대해 만든다. 한 layer만 컴파일한 결과를 전체 모델 결과로 환산하지 않는다.

Prefill만 요청한 실험의 선택 기준은 prefill 한 entry의 비용이다. Prefill/decode를
동시에 요청하는 실행의 “두 entry 모두 나빠지면 안 된다”는 조건과 혼동하지 않는다.
실행 방법은 [12장](12_operations_limits.md#전체-llama의-fusion-탐색-시간을-해제해서-비교하기)에 있다.

### 첫 graph 후보도 오래 걸릴 수 있는 이유

```text
Graph 구성 하나
  ├─ DRAM 전달을 기준으로 실행 가능한 프로그램 생성
  └─ 중간값을 SRAM에 보관하는 선택 비교
       ├─ 보관 공간 때문에 현재 커널의 가용 SRAM이 달라짐
       ├─ 해당 메모리 조건에 맞는 schedule 후보 검사
       │    → TIR 변환 → 실제 SRAM 배치 → 명령 생성과 비용
       └─ 전체 호출의 주소·명령을 연결하고 불필요한 전송/명령 정리
  → 이 graph의 최종 비용 확보
다음 graph 구성과 비교
```

따라서 “첫 graph를 아직 만드는 중”과 “아무 탐색도 하지 않는 중”은 다르다.
Fusion을 끈 기준에서도 여러 schedule과 메모리 조건을 검사한다. 큰 출력의 concat을
나누거나 긴 최종 명령 stream을 분석하는 시간은 matmul 후보 수와도 별개다.
시간을 보고할 때 graph 개수, 새로 측정한 schedule 수, 재사용 횟수와 단계별 시간을
같이 읽어야 한다. `phase`의 전체 시간에는 내부 단계가 포함될 수 있으므로 단순히
모두 더하지 않는다. 합산할 때는 `self_seconds`를 사용한다.

예제: [선택된 IR](../examples/pipeline/04_selected_semantic_tir.py),
[전체 Llama 선택 기록](../evidence/current/llama_selection.json).
구현: [영역 선택](../../npu_compiler/npu_fusion.py), [전체 선택](../../npu_compiler/npu_build.py).
