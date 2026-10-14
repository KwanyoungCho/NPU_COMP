# NPU compiler 설명서

이 문서는 **현재 메모리 계층을 사용하는 v09 backend**를 처음부터 이해하기 위한 설명서다.
모델의 계산이 어떻게 TVM IR, SRAM 배치, NPU 명령, 실제 token 생성으로 이어지는지 설명한다.
기본 설명과 전체 모델 기록은 2026-09-28 작업 트리와 TVM 0.19.0을 기준으로 한다.
09-29에 추가한 선택형 MetaSchedule 기록 재사용과 ReplayFunc 탐색은
[6장](current/06_matmul_schedule.md)의 별도 절에 명시한다.
[작은 예제 검증](evidence/current/meta_schedule_reuse.json)과
[실제 Llama 커널 탐색 비교](current/11_experiments.md#e8-metaschedule로-탐색하면-llama-커널이-더-좋아지는가)를 구분한다.
긴 입력의 후보 중복·SRAM 초과·예산과 튜닝 결과 재사용을 보완한 결과는
[실험 E9](current/11_experiments.md#e9-긴-입력의-탐색과-튜닝-결과-재사용-개선)에 있다.
**28-layer·256-token prefill 재컴파일은 09-29 메모리 과다 사용으로 중단했다.**
큰 cache fusion의 반복 계산을 거절하는 검사와 안전한 행 분할을 수정했다.
중단 원인·수정·검증 범위는 [실험 E10](current/11_experiments.md#e10-256-token-전체-prefill의-fusion-탐색)에 있다.
완료 전까지 전체 모델의 새 검증 결과로 주장하지 않는다. `main`의 최종 ver.08 backend는
[별도 문서](main/README.md)에 둔다. 두 backend의 수치와 실행 방식을 서로 섞지 않는다.

날짜별 진행 보고서나 계획을 먼저 읽을 필요가 없다. 아래 순서대로 읽고,
실제 결과를 확인하고 싶을 때만 각 장이 가리키는 작은 예제와 JSON을 열면 된다.
양자화는 FP16 본문을 이해한 뒤 읽는 [후반부](current/13_quantization.md)다.
구현이 있다는 것과 전체 모델에서 검증됐다는 것을 구분한다.

## 읽는 순서와 각 장의 질문

| 순서 | 문서 | 해결할 질문 | 함께 볼 결과 |
|---|---|---|---|
| 1 | [전체 구조](current/01_overview.md) | Compiler와 runtime은 무엇을 만들고 실행하는가? 두 backend는 무엇이 다른가? | 전체 pipeline 그림, Llama shape 표 |
| 2 | [모델에서 Relax까지](current/02_model_relax.md) | Token, embedding, weight, 함수 입력은 어떻게 표현되는가? | 작은 Linear+SiLU의 authored Relax |
| 3 | [그래프 정리와 host 처리](current/03_graph_host.md) | Attention 분해, parameter lifting, 표준 graph pass는 무엇을 바꾸는가? | graph prefix와 host transform IR |
| 4 | [LegalizeOps와 수치 의미](current/04_semantic_tir.md) | Relax가 사라지는가? TIR block/reduction/cast는 무엇인가? | unfused semantic TIR, K=128 예제 |
| 5 | [Fusion과 후보 선택](current/05_fusion.md) | TVM이 무엇을 묶고, 부분/전체 비교는 무엇이 다른가? | selected IR, 후보 비용과 종료 사유 |
| 6 | [Matmul scheduling](current/06_matmul_schedule.md) | 어떤 데이터를 얼마나 올리고, 어떤 순서로 계산하는가? | compute_at 전후, 실제 scheduled TIR |
| 7 | [Vector·정규화·attention](current/07_operators.md) | SiLU, softmax, RMSNorm, RoPE, attention은 어떤 명령과 버퍼를 쓰는가? | 단계별 수식, query 분할 그림 |
| 8 | [DRAM/SRAM 메모리 계획](current/08_memory.md) | 수명·USMP·보관·직접 저장·주소 재배치는 어떻게 연결되는가? | 두 matmul의 SRAM 주소와 77-word 프로그램 |
| 9 | [TIR에서 ISA와 bundle까지](current/09_isa.md) | LOAD와 GLOAD는 어떻게 다르고, 주소의 단위는 무엇인가? | 실제 명령 해독과 program.json |
| 10 | [Runtime과 Llama 생성](current/10_runtime_llama.md) | Prefill 뒤 decode를 어떻게 반복하고 cache를 어디에 쓰는가? | 실제 7-token prompt의 3-token 생성 기록 |
| 11 | [실험과 검증 결과](current/11_experiments.md) | 무엇을 실행 검증했고, 어떤 최적화가 얼마나 달라졌는가? | 출처·shape·설정·검증 수준이 붙은 측정표 |
| 12 | [사용법·지원 범위·남은 이슈](current/12_operations_limits.md) | 어떻게 재현하고, 오류는 어디서 찾으며, 아직 무엇이 남았는가? | 실행 명령, 기능 경계, 후속 검토 목록 |
| 13 | [양자화: 별도 검증 범위](current/13_quantization.md) | W8A16/W8A8은 실제로 무엇을 줄이고 무엇을 계산하는가? | scale 예제, 전용 schedule, 현재 검증 한계 |

각 장은 **입력 → 해결할 문제 → 구체적인 동작 → 결과 → 다음 단계** 순서로 읽는다.
간단한 의사 코드는 동작 설명용이며, `examples/`의 `.py` IR dump와 구분한다.
TIR에 보이는 논리 buffer와 물리 accumulator도 구분한다.

## 참고 자료는 세 종류만 사용한다

- [TVM API 설명](reference/tvm_api.md): 본문에서 사용하는 표준 pass와 Schedule API의 입력·효과·주의점.
- [실행 계약](reference/execution_contract.md): dtype, 주소 단위, ISA, native/Python, 지원 경계.
- [예제 안내](examples/README.md): 소수의 실제 IR·명령·실행 결과와 재현 방법.

[실험 근거](evidence/README.md)는 수치의 원본이다. 과거 기록은 source hash와 함께
보존하고, 이번 문서 작업에서 재실행한 결과와 구분한다. 과거 보존 기록과 이번 E10 전체 모델 재컴파일을 구분한다. [그림 안내](figures/README.md)는 이후 PPT에서 재사용할
그림, 설명할 메시지, 연결된 실험을 모은다. [발표자료](presentation/README.md)에는
그림 중심의 PPTX·PDF와 발표자 노트가 있다.

## 설명의 기준과 단위

- 기본 모델: Llama 3.2 3B, 28 layers, hidden 3072, MLP 8192, Q heads 24,
  KV heads 8, head dimension 128, vocabulary 128256.
- 기본 실행: FP16 저장, 정해진 FP32 누산, 정적 shape, SRAM 8 MiB, matrix tile 64.
- `KiB/MiB`는 1024/1048576 bytes. `word`는 명령 저장 공간의 32-bit 단위다.
- DMA bytes, ISA words, SRAM peak, DRAM image, compile seconds는 서로 다른 지표다.
  ISA words나 simulator wall time으로 하드웨어 latency를 주장하지 않는다.
- 설명 대상 compiler의 코드 지도는 [실행·지원 문서](current/12_operations_limits.md)에 있다.
- 기존 문서의 보존·제거 내역은 [정리 기록](evidence/cleanup.json)으로 추적한다.

## 두 backend의 경계

현재 작업 트리의 `npu_target.build`와 `npu_generate.generate`가 본문의 대상이다.
`main`은 commit `1143abbc6bedddb1eed8d2bcd8ecae6b346d9fd1`로 고정해 설명한다.
현재 작업 트리에 구형 backend를 복원하거나, `main` branch를 수정하지 않는다.
Vendor 원본 실행기와 ISA PDF는 저장소의 원래 위치에 보존한다.
