# 3. 그래프 정리와 host parameter transform

[목차](../README.md) · 이전: [모델과 Relax](02_model_relax.md) · 다음: [TIR와 수치](04_semantic_tir.md)

이 단계의 목적은 실행마다 필요한 계산과 모델을 준비할 때 한 번 할 수 있는 계산을
구분하고, 후속 lowering이 이해하기 쉬운 graph를 만드는 것이다.

## Graph prefix의 실제 순서

| 순서 | Pass | 입력에서 어떤 문제를 해결하는가? |
|---|---|---|
| 1 | `CanonicalizeAttention` | 고수준 attention을 matmul·scale·mask·softmax·matmul로 드러냄 |
| 2 | `CanonicalizeBindings` | 불필요한 alias와 binding 표현을 정리하여 같은 값의 추적을 단순화 |
| 3 | `EliminateCommonSubexpr` | 같은 입력의 같은 순수 계산이 반복되면 결과 공유 |
| 4 | `RewriteDataflowReshape` | 지원되는 reshape 표현을 데이터 복사가 필요 없는 형태로 정리 |
| 5 | `LiftTransformParams` | parameter에만 의존하는 계산을 별도 host 함수로 분리 |

1은 NPU용 변환이고 나머지는 TVM 표준 pass다. 표준 pass를 호출해도 임의의
layout 변환이 모두 zero-copy가 되거나 모든 연산이 자동으로 장치에 맞는 것은 아니다.
정확한 배열 접근과 지원 조건은 이후 단계가 확인한다.

## CanonicalizeAttention은 왜 필요한가?

고수준 attention 하나를 그대로 LegalizeOps에 넘기면 표준 attention legalizer가
내부 TE 계산을 직접 만든다. 우리가 등록한 `relax.matmul`과 `relax.nn.softmax`의
NPU용 legalizer를 내부에서 다시 호출한다고 가정할 수 없다.

따라서 먼저 Relax에서 계산을 드러낸다.

```text
attention(Q,K,V,bias,scale,mask 조건)
             ↓
QKᵀ → scale 곱 → mask/bias 더하기 → softmax → PV
```

이제 각 연산에 맞는 수치 계약을 적용하고, scale+mask+softmax만 합칠지 전체를 합칠지
비교할 수 있다. “분해했으므로 영원히 분리 실행한다”는 뜻이 아니다.

입력 BSNH layout을 지원하는 matmul layout으로 바꾸고 scale, bias, causal 정렬,
지원되는 window 의미를 유지해야 한다. 단순히 연산 이름을 바꾸는 pass가 아니다.
Authoring한 Llama는 이미 이러한 세부 연산을 쓰므로, 해당 부분에서 이 pass가
새로 할 일이 없을 수 있다.

설치된 TVM 0.19의 `DecomposeOpsForInference`가 이 attention 예제를 그대로 둔
[IR 비교 기록](../examples/attention_canonicalization/comparison.json)을 남겼다.
표준 pass 이름이 비슷하다는 이유로 같은 변환이라고 판단하지 않는다. 표준 attention
legalization과 NPU canonicalization 뒤 legalization의 차이는 같은 폴더의 dump에서
볼 수 있다. 이 예제는 IR/compile 검사이며 수치·성능 benchmark가 아니다.

## W transpose는 언제 실행되는가?

작은 예제의 변환 전은 다음과 같다.

```text
device(x, W, bias):
    B = transpose(W)
    return SiLU(x @ B + bias)
```

Lifting 뒤에는 역할이 나뉜다.

```text
host_transform(W, bias):
    B = transpose(W)
    return 변환된 parameter 묶음

device(x, transformed_parameters):
    return SiLU(x @ B + bias)
```

Compile은 host 함수를 만드는 시점이다. 실제 weight 배열을 transpose하는 것은
runtime 준비 단계에서 host 함수를 실행할 때다. Parameter가 Constant이면
constant folding으로 미리 계산되는 경우도 있으며 둘을 구분해야 한다.

이제 token마다 큰 weight를 transpose할 필요가 없다. Llama LM head의 weight만 해도
3072×128256 FP16, 약 751.5 MiB다. 이 변환이 device 호출마다 들어가면 매우 큰 이동이 된다.

## Parameter 순서를 왜 확인해야 하는가?

Host transform은 원래 순서와 다른 순서로 배열을 반환하거나 일부 결과를 묶을 수 있다.
Runtime은 “원래 W가 첫 번째였으니 변환 뒤에도 첫 번째”라고 추측하지 않는다.
변환된 함수 signature와 tuple 구조에 맞춰 이름·shape·dtype을 결합한다.

[graph prefix IR](../examples/pipeline/02_graph_prefix.py)과
[host transform IR](../examples/pipeline/05_host_transform.py)을 비교하면서
device 함수에서 transpose가 사라졌는지, host 함수의 반환 tuple은 어떤지 확인한다.
실제 입력 순서는 예제의 `stage_summary.json`에도 기록된다.

## 이 단계 다음에 아직 남는 것

Graph prefix는 tile 크기, SRAM 주소, DMA 순서를 정하지 않는다. 예를 들어 W.T를
host에서 준비했더라도 W.T 전체가 SRAM에 들어가는 것은 아니다. 이후 schedule이
DRAM의 W.T 중 얼마를 한 번에 가져올지 정한다.

선택적으로 GQA의 `repeat`를 broadcast matmul로 바꾸거나 concat producer를 묶는
후보도 생성한다. 이것들은 항상 적용되는 graph prefix가 아니라 비용 비교 대상이다.
사용하지 않는 함수·값은 표준 DeadCodeElimination으로 정리한다.

구현 연결: [graph pass 구성](../../npu_compiler/tvm_pipeline.py),
[attention 변환](../../npu_compiler/npu_attention.py),
[parameter 실행과 cache](../../npu_compiler/npu_generate.py).
