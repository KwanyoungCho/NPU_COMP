# TVM 기반 NPU compiler

**설명은 [docs/README.md](docs/README.md)부터 읽는다.** 날짜별 보고서 대신 현재 코드의
동작을 설명하는 하나의 문서 체계로 정리했다. 현재 메모리 계층 v09와 기존 `main`의
ver.08 backend를 구분하며, 본문은 FP16 기준이다.

| 필요한 내용 | 문서 |
|---|---|
| 처음부터 이해하기 | [전체 목차와 읽는 순서](docs/README.md) |
| 모델 → Relax → TIR → schedule → 메모리 → ISA | [Overview](docs/current/01_overview.md)부터 단계별 1~9장 |
| 실제 Llama prefill / decode / KV cache | [10장](docs/current/10_runtime_llama.md) |
| 검증된 기능·출력·compile 비용·메모리 수치 | [실험](docs/current/11_experiments.md) |
| 실행 방법, 옵션, 지원 경계, 남은 검토 | [사용법](docs/current/12_operations_limits.md) |
| TVM API, 수치·ISA 계약 | [API 사전](docs/reference/tvm_api.md), [실행 계약](docs/reference/execution_contract.md) |
| 실제 IR와 재현 가능한 작은 예제 | [예제 안내](docs/examples/README.md) |
| 검증 범위를 구분한 양자화 설명 | [별도 후반부](docs/current/13_quantization.md) |
| 기존 main 최종 backend | [main 문서](docs/main/README.md) |
| 발표자료에 사용할 그림 | [SVG와 근거](docs/figures/README.md) |

## 작은 예제부터 실행하기

저장소 루트에서 검증된 `npu-tvm` 환경을 사용한다. 버전 기준은
[toolchain.lock.json](toolchain.lock.json)에 있다.

```bash
conda activate npu-tvm
bash d_compiler/npu_codegen/build.sh
python d_compiler/docs/examples/reproduce.py --out /tmp/npu-doc-examples
python d_compiler/docs/tools/check_docs.py
```

작은 예제는 전체 Llama를 컴파일하지 않는다. 큰 모델의 실행은 위 10~12장의 조건과
측정 범위를 확인한 뒤 `run_nn_generate.py`/`run_nn_npu.py`로 진행한다.
일반 compile 진입점은 `npu_compiler.npu_target.build`, 생성 반복은
`npu_compiler.npu_generate.generate`다. 실행 결과는 현재 v09 C-model 기준이며
실장치 latency와 동일한 지표가 아니다.

회귀 검사의 기본 출력은 `d_compiler/build/validation`이다. 명시적인 `--output`으로
실험별 경로를 지정할 수 있다.

```bash
bash d_compiler/run_gate.sh --tests architecture_contracts schedule_memory native_codegen \
  --require-no-skips --output /tmp/npu-gate
```

날짜별 계획·handoff·중복 dump의 제거 내역과 보존한 원본은
[자료 정리 기록](docs/evidence/README.md)에 있다. Compiler/runtime 구현, vendor 원본,
checkpoint와 일반 build 산출물은 문서 정리 대상에 포함하지 않았다.
