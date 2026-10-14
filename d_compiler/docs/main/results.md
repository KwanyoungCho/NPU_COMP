# Main backend의 보존된 검증 결과

[Main 개요](README.md) · [실행 흐름](pipeline.md)

아래는 main commit의 기존 보고서에서 보존한 결과다. 이번 문서 정리에서 vendor나
전체 모델을 다시 실행하지 않았다. 원본 report와 line 위치, 발췌 text의 근거는
[source_excerpts.json](../evidence/main/source_excerpts.json)에 있다.

## Llama 3.2 3B

28 layers, hidden3072, MLP8192, Q24/KV8, head128, vocabulary128256을 사용했다.
Prefill과 두 번의 autoregressive decode에서 HF greedy token `[358,2846,4560]`이
일치했다. 최종 decode logits cosine은 약 0.9999881이다.

| 단계 | 처리한 위치/유효 cache | 예측 token |
|---|---|---:|
| Prefill | prompt 7 tokens | 358 |
| Decode | position7, cache8 | 2846 |
| Decode | position8, cache9 | 4560 |

당시 context8 decode 프로그램은 453349 words, context9는 453757 words였다.
이는 **layer 프로그램**의 통계다. 최신 v09 전체 28-layer entry의 수백만 words와
같은 컴파일 범위가 아니므로 그대로 배율 비교하지 않는다.

## 다른 model family의 당시 결과

| 모델 | 범위 | HF와 같은 생성 token | 당시 최종 logits cosine |
|---|---|---|---:|
| Gemma 4 E2B text | 35 layers, sliding/global·KV 공유·PLE 포함 | [108,236777,236789] | 약 0.9999975 |
| Qwen3-4B text | 36 layers, QK norm, Q 폭≠hidden | [358,1184,311] | 약 0.9999915 |

여기서 model family 지원 기록은 main backend에 대한 것이다. 최신 v09 generation
CLI에서 같은 범위가 재검증되었다는 뜻은 아니다.

## 실행기 호환성의 의미

작은 데이터는 원본 vendor binary와 source C-model을 비교한다. 전체 모델은
vendor의 고정8192-entry G-buffer를 넘으므로 용량을 확장한 source target을 사용한다.
원본 program memory가 무조건32768 words로 제한된다는 초기 판단은 후속 검사로
정정됐다. Global capacity와 program 길이를 혼동하지 않는다.

Vendor opcode의 quirks를 재현한 parity 검사와, 모델 출력을 HF에 맞추기 위한
compiler의 우회는 별도의 검증이다. 짧은 prompt의 token 일치를 장문 품질이나
실장치 성능 검증으로 확대하지 않는다.

## 남겨둘 제한

Context별 재컴파일, host cache concatenate, layer별 image 재구성, wide LM head의
별도 packing, 많은 주소 설정·정적 전개가 당시 구조의 특성이다. Main backend를
다시 개발하는 계획은 이 문서에 만들지 않는다. 기능을 비교하거나 결과를 재현할 때
그 commit의 실행 조건과 vendor 계약을 사용한다.
