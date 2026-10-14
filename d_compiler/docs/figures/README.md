# 그림과 발표자료에 사용할 근거

[문서 목차](../README.md). 원본은 편집 가능한 SVG다. 개념 그림은 직접 그렸고,
측정 그림은 보존한 JSON에서 matplotlib으로 생성한다. 실제 측정값을 개념도에
임의로 넣거나, 명령 수를 실행 시간으로 바꾸지 않는다.

| ID / 원본 | 설명할 메시지 | 대응하는 장과 주의점 |
|---|---|---|
| [01 pipeline](01_pipeline.svg) | 모델에서 실행까지 각 단계의 책임 | [1장](../current/01_overview.md). 후보 비교는 이전 단계로 돌아갈 수 있다. |
| [02 Llama layer](02_llama_layer.svg) | Attention/MLP와 두 residual의 연결 | [2장](../current/02_model_relax.md). 실제 3B 폭, SRAM 배치 그림은 아니다. |
| [03 Relax/TIR](03_relax_tir.svg) | Legalize 이후에도 Relax 호출 그래프가 남는다 | [4장](../current/04_semantic_tir.md). FuseOps와 FuseTIR의 역할을 나누어 설명한다. |
| [04 누산](04_matmul_accumulator.svg) | K 전체 동안 FP32 누산, 마지막에 FP16 저장 | [6장](../current/06_matmul_schedule.md). 입력 panel과 출력 tile 크기를 구분한다. |
| [05 SRAM 공유](05_sram_retention.svg) | 분리된 커널도 H를 DRAM에 쓰지 않고 전달한다 | [8장](../current/08_memory.md). 44 KiB 예제의 실제 byte 주소다. |
| [06 Attention](06_attention_blocks.svg) | Query를 나누면 줄어드는 중간값과 남는 제약 | [7장](../current/07_operators.md). FlashAttention 구현을 뜻하지 않는다. |
| [07 KV append](07_kv_decode.svg) | Capacity C와 다음으로 쓸 위치 L은 다르다 | [10장](../current/10_runtime_llama.md). 실제 C=9/L=7 예를 도식화했다. |
| [08 양자화](08_quantization.svg) | 저장 dtype과 연산 dtype을 구분한다 | [13장](../current/13_quantization.md). 검증이 제한된 후반부 기능이다. |
| [09 메모리 실험](09_memory_measurements.svg) | SRAM 여유에 따라 보관 여부와 DMA가 달라진다 | [E2](../current/11_experiments.md). 이번 작은 예제 5개 구성의 결과다. |
| [10 Compile 관측](10_compile_observations.svg) | 같은 binary를 얻는 데 걸린 compile 작업 감소 | [E5](../current/11_experiments.md). 과거 단일 관측이며 반복 benchmark가 아니다. |

색은 host/일반 상태 회색, DRAM 또는 외부 데이터 주황, SRAM 파랑, 연산장치 보라,
IR 초록으로 사용했다. 메모리 배치 그림에서는 buffer를 구별하는 색도 사용하므로
W의 색이 SRAM 안의 W를 DRAM에 있다고 표시하는 것은 아니다. 그림 안의 위치·명령을
함께 읽는다. SVG 안에는 검색·수정 가능한 텍스트가 있으며, font가 없는 PC에서는
설치된 sans-serif로 표시된다.

발표에서는 먼저 01로 전체 흐름을 제시하고, 02의 Llama 계산을 작은 예제에 대응시킨다.
03→04→05로 계산 의미·실행 순서·저장 위치를 구분한 후, 06/07로 attention과 반복 생성을
설명한다. 끝으로 09/10에서 실제 확인한 효과와 측정 한계를 보이고 08은 부록으로 둔다.
각 슬라이드는 이 표의 한 메시지를 중심으로 만들고, 자세한 API/IR은 본문 링크로 연결한다.
이 흐름을 한국어 발표에 맞춰 도형과 설명으로 구성한 [발표자료](../presentation/README.md)도 있다.
PPTX의 주요 도형·텍스트는 편집 가능하며 PDF와 발표자 노트가 함께 제공된다.

재생성은 저장소 루트의 `npu-tvm` 환경에서 실행한다.

```bash
python d_compiler/docs/tools/render_figures.py
```

Figure 09의 데이터는 [memory/measurements.json](../examples/memory/measurements.json),
Figure 10은 [compile_comparison.json](../evidence/current/compile_comparison.json)이다.
개념 그림은 코드의 동작을 설명하며, timing simulator의 출력이 아니다.

09-29 탐색 갱신 그림은 발표자료의 편집 가능한 도형으로 추가했다. 기본 설명판 44~46장과
상세판 41·77·78장은 후보 공간/검사/trace 재사용의 연결, E9 비교, E10의 9개 graph 구성을
설명한다. 진행 중인 E10에는 비용 막대그래프를 그리지 않는다. 최종 측정과 출력 검증이
끝난 뒤 결과를 반영하며, 과거 Figure 10의 시간 관측은 원래 조건 그대로 보존한다.
