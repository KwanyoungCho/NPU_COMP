# 대표 예제와 실제 결과

[목차](../README.md). 모든 날짜별 예제를 모으지 않고, 서로 다른 질문을 설명하는
사례만 남겼다. `.py` dump는 TVM Script IR이며 일반 Python 프로그램처럼 그대로
실행하는 파일이 아니다. 실행용 파일은 `reproduce.py`, `export_llama_ir.py`,
`compare_llama_search.py`, `compile_llama_full_search.py`다.

| 예제 | 범위 | 무엇을 볼 것인가? | 검증 수준 |
|---|---|---|---|
| [pipeline/](pipeline/) | Linear+SiLU, [64,128]→[64,64] | authored Relax→host→semantic→scheduled TIR→ISA | 이번에 C-model/LLVM 실행 |
| [memory/](memory/) | 두 matmul, SRAM32/44/64 KiB | 보관 on/off에 따른 DMA·주소·출력 | 이번에 5개 구성 C-model 실행 |
| [two_matmuls_recorded/](two_matmuls_recorded/) | 44 KiB의 기존 최종 77 words | H 직접 저장의 실제 scheduled TIR와 binary | 기존 전후 bit-exact, 이번 재현과 hash 비교 |
| [schedule_api/](schedule_api/) | 4×4 교육 계산 | compute_at/reverse_compute_at/cache/scope/compaction | CPU 두 변환 실행, SRAM 예는 TIR 구성만 |
| [attention_canonicalization/](attention_canonicalization/) | [B,S,H,D]=[1,4,2,8] | 표준 attention lowering과 분해 경로 비교 | 보존된 IR/compile 비교, 수치 benchmark 아님 |
| [llama_ir/](llama_ir/) | 실제3B 폭, 1 layer, S7/C9 | QKV·norm·MLP·cache 호출을 같은 IR에서 연결 | 이번 graph export/legalization, checkpoint/실행 없음 |
| [compare_llama_search.py](compare_llama_search.py) | 실제 Llama decode/64-token prefill, 선택형 512-token down | 기존 후보 검사 순서와 TVM ReplayFunc 비교 | 실제 layer-0 weight, 합성 activation, C-model 검증; 전체 생성 아님 |

| [compile_llama_full_search.py](compile_llama_full_search.py) | 실제 Llama 28-layer, 256-token prefill | 시간 제한 없는 graph 탐색, 확장 schedule 예산, 전체 출력 검증 | E10 진행 중; HF reference 준비, 최종 NPU 결과 대기 |

## Pipeline 파일의 읽는 순서

1. `01_authored_relax.py`: x/weight/bias와 원래 연산.
2. `02_graph_prefix.py`: transpose가 host로 이동하고 parameter 순서가 바뀌는 위치.
3. `03_unfused_semantic_tir.py`: Relax call_tir와 FP32/FP16 matmul 정의.
4. `04_selected_semantic_tir.py`: 실제 선택된 fusion.
5. `05_host_transform.py`: 실행 전에 수행할 함수.
6. `06_initial_dram_plan.py`: SRAM 선택·최종 압축 이전의 기본 storage 계획.
7. `07_scheduled_*.py`: 최종 fused 커널의 실제 compile recipe에서 가져온 schedule. 별도 재탐색 결과가 아니다.
   Joint-memory 경로의 호출별 TIR는 `two_matmuls_recorded/scheduled_*.py`에 있다.
8. `bundle/program.json`, `09_packets.json`: 최종 주소와 명령.
9. `stage_summary.json`, `outputs.npz`: 비교 조건과 수치 결과.

Initial DRAM plan과 최종 program.json의 주소가 다를 수 있다. 미사용 pool 제거가
뒤에서 수행되기 때문이다. Weight 값은 inputs.npz, 실행 image는 bundle에 있다.

## 작은 예제 재현

저장소 루트, `npu-tvm` 환경에서 실행한다.

```bash
python d_compiler/docs/examples/reproduce.py --out /tmp/npu-doc-examples
python d_compiler/docs/examples/export_llama_ir.py --out /tmp/npu-doc-llama-ir
python d_compiler/docs/examples/schedule_api/reproduce.py
```

첫 명령은 작은 compile 7개 구성과 C-model 실행을 한다. 두 번째는 큰 weight 배열을
읽거나 전체 Llama를 codegen하지 않고 IR만 만든다. 마지막은 같은 폴더의 schedule
교육 dump를 갱신한다. Attention 재현기도 작은 compile을 포함하며 오래된 기록과
비교할 때 source identity가 달라질 수 있다.

새 실행의 [reproduction.json](reproduction.json)은 실제 수행 범위를 기록한다.
기존 full-model benchmark와 섞지 않는다. 예제 출력이 달라졌을 때 문서 숫자만
먼저 바꾸지 말고 source/profile/numeric policy와 선택 후보를 확인한다.

MetaSchedule 비교의 [실행 명령](../current/12_operations_limits.md)과
[조건·수치·해석](../current/11_experiments.md#e8-metaschedule로-탐색하면-llama-커널이-더-좋아지는가)은 본문에 있다.
대형 binary를 문서에 중복 저장하지 않고 실행기가 `build/`에 TIR·binary·진행 상황을
출력한다. 최초 비교 기록은 [JSON](../evidence/current/llama_meta_schedule_search.json)에 남긴다.
이후 후보·예산·DB 호환성 개선 결과는 [별도 기록](../evidence/current/llama_schedule_refinement.json)이다.
`--reuse-check`로 튜닝 뒤 탐색 설정을 바꿔 재생하는 경로도 검사할 수 있다.
