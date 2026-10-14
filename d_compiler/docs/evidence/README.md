# 실험 근거와 자료 정리 기록

[목차](../README.md) · [해석과 측정표](../current/11_experiments.md)

`current/`는 현재 backend의 보존된 기록이고 `main/`은 고정 main commit의 결과다.
새 문서 예제 실행은 [examples/](../examples/README.md)에 둔다.
Source hash가 다른 측정을 같은 시점의 benchmark로 묶지 않는다.

| 파일 | 역할 |
|---|---|
| [current/llama_prefill256_full_search_progress.json](current/llama_prefill256_full_search_progress.json) | E10 진행 snapshot: 실제 28-layer·256-token prefill 조건, 중복 제거 후 graph 9개, 최종 비용·검증은 대기 |
| [current/completion.json](current/completion.json) | 최종 검증 범위와 full cold-build 시간이 미측정임을 명시 |
| [current/llama_validation.json](current/llama_validation.json) | 실제 prompt/tokens/logits 비교, DMA·runtime patch/I/O |
| [current/llama_selection.json](current/llama_selection.json) | graph 후보와 시간 한도, 실제 선택 |
| [current/refinement.json](current/refinement.json) | 출력 직접 저장·vector·DRAM 정리 전후, 실행 여부 포함 |
| [current/mlp64.json](current/mlp64.json) | 실제 폭 MLP의 schedule·보관 진단 |
| [current/compile_comparison.json](current/compile_comparison.json) | 단일 compile 시간 비교와 binary hash |
| [current/meta_schedule_reuse.json](current/meta_schedule_reuse.json) | 09-29 선택형 trace DB 재사용: 작은 FP16 예제 3개, 새 후보 생성 0회·동일 명령·동일 출력; 전체 Llama 재검증 아님 |
| [current/llama_meta_schedule_search.json](current/llama_meta_schedule_search.json) | 09-29 Llama 커널 4개의 기존 탐색/ReplayFunc 비교: 실제 weight, 합성 activation, 동일 binary·C-model 검증, 초기화 시간·후보 예산 포함 |
| [current/llama_schedule_refinement.json](current/llama_schedule_refinement.json) | 512-token down의 중복·용량·예산 보완 후 비교, 합법 후보 4개씩 평가, 변경된 탐색 설정에서 DB 재사용, C-model·34개 회귀 검사 |
| [current/memory_search_before_refinement.json](current/memory_search_before_refinement.json) | 개선 전 메모리 탐색/fusion 비교의 과거 기준 |
| [current/regression_validation.json](current/regression_validation.json) | 17개 gate의 exit/status/시간 |
| [current/gate_logs/](current/gate_logs/) | 해당 gate의 원본 출력 |
| [current/quantization.log](current/quantization.log) | 양자화 작은 예제 검증, 본문 FP16과 별도 |
| [current/quantization_history.json](current/quantization_history.json) | 과거 전체 모델 quantized prefill 결과 발췌 |
| [main/source_excerpts.json](main/source_excerpts.json) | main의 당시 보고서 line·내용·원본 hash |
| [retained_sources.json](retained_sources.json) | 이동한 artifact의 원래 위치와 byte hash |
| [cleanup.json](cleanup.json) | 제거한 날짜별 문서·예제·중복 산출물의 목록과 크기 |
| [checks.json](checks.json) | 새 문서/예제 연결과 측정값 교차 검사 결과 |

보존된 JSON 안의 absolute log path나 `reused_artifacts` path는 당시 provenance다.
그 경로가 현재도 실행 가능한 cache라는 뜻은 아니다. 중복 대형 pickle과 full-model
binary는 문서 정리에서 제거하며, 필요한 통계와 생산/재사용 source identity를 남긴다.
현재 보관된 파일의 대응은 retained_sources.json과 이 index를 사용한다.

보존한 측정 파일은 수치를 바꾸지 않고 복사했다. Attention 예제 실행기는 이동한
폴더에 맞춰 저장소 root를 찾는 부모 경로만 변경했고, 원본/이동 후 hash를 둘 다
기록했다. Schedule API 예제는 두 CPU 검사를 다시 수행했으며 dump 내용이 기존과
같음을 확인했다. 예외적으로 설명용 발췌와 새 검사 기록은
그 성격을 명시한 새로운 파일이다. 새 실행 결과를 과거 결과 파일에 덮어쓰지 않는다.
