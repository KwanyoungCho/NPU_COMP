# Native v09 codegen

스케줄과 SRAM 주소가 정해진 TIR를 v09 명령으로 바꾸는 C++ 구현이다.
같은 역할의 Python walker가 있으며, 지원되지 않는 native coverage의 fallback과
잘못된 schedule·내부 오류를 구분한다. 자세한 동작은
[ISA와 bundle](../docs/current/09_isa.md),
[실행 계약](../docs/reference/execution_contract.md)을 따른다.

저장소 루트의 `npu-tvm` 환경에서 빌드한다.

```bash
bash d_compiler/npu_codegen/build.sh
```

`TVM_HOME`으로 TVM 경로를 지정할 수 있다. 생성되는 `libnpu_codegen.so`는
실험 결과가 아닌 build 산출물이다. Library는 공통 계약 및 source identity가
현재 Python compiler와 맞아야 한다.

`npu_link.compile_program(..., native_mode="use")`는 지원되는 kernel에 native를 사용한다.
`native_mode="compare"`는 Python과 native 명령을 대조하는 검증 경로다.
변경 시 [test_native_codegen.py](../tests/test_native_codegen.py)로 encoders·kernel·전체
stream의 일치를 확인한다. 이 일치는 HF 모델 수식의 정확성이나 hardware latency를
독립적으로 입증하지는 않는다.

실제 측정과 검증 범위는 [실험 문서](../docs/current/11_experiments.md)에 모았다.
과거의 다른 codegen 세대에서 얻은 속도 배율을 현재 baseline에 적용하지 않는다.
