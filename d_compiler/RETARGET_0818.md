# NPU ver.08(0818) 검증 및 compiler 재타겟 결과

대상 오라클은 `0818_npu_update/a_npu/a.out`, 명세는
`20260803_01_NPU_design_instruction_format_ver_08...pdf`이다. 0710 경로는
회귀 비교를 위해 유지하고, 새 구현은 별도 ISA·runtime·backend로 분리했다.

## 1. 문서와 실행 파일에서 확정한 변경점

| 항목 | ver.08 실제 동작 |
|---|---|
| Reduce Max | opcode `0x19`. 구현되어 있으나 초기 accumulator가 `0`이라서 전부 음수인 벡터는 잘못된 `0`을 반환한다. |
| 주소 설정 `0x80` | `[31:30]` operand, `[29]` low/high, `[28]` MAIN/PARTIAL. 실제 load/save는 PARTIAL address를 사용한다. |
| 행 설정 `0x88` | `[31:30]` operand, `[29]` MAIN/PARTIAL, `[23:8]` rows. |
| 열 설정 `0x89` | `[31:30]` operand, `[29]` MAIN/PARTIAL, `[23:8]` cols. |
| 임의 sub-tile | `partial_addr + row * main_cols + col`로 접근한다. 따라서 global row-major 행렬에서 임의 크기/위치의 sub-tile을 직접 load/save할 수 있다. |
| Broadcast `0x15` | immediate/scalar 모두 동작한다. scalar 주소는 `[29]` low/high 상태로 구성한다. |
| Matrix activation | `[29:28]`: `00` off, `10` SiLU, `11` GELU. 실행 파일에서는 예약값 `01`도 GELU와 동일하게 동작한다. |
| GELU | 실행 파일의 식은 표준 erf/tanh GELU가 아니라 `x * sigmoid(2*x)`이다. |
| Matmul MAC | `[27]`이 누산 비트다. 0710의 `[28]`에서 이동했다. |
| G-buffer binary save | opcode `0xF0`이 고정 파일 `saved_G_buffer_data.bin`을 기록한다. HALT가 아니며 다음 명령을 계속 실행한다. |
| Indirect addressing | 문서/전달 내용대로 구현되지 않았다. |

0710의 strided matrix load/save(`0x90`/`0x98` bit 29)는 유지된다. 내부 계산은
float32이고 G-buffer save에서 FP16으로 반올림한다.

## 2. 문서 외에 반드시 반영한 vendor quirk

- G-buffer는 여전히 **8192 FP16** 정적 배열이다. program은 file size만큼 동적으로
  `malloc`되어 32768-word 고정 제한이 없다. 8192개를 넘긴 G-buffer 입력은 정적 배열
  뒤를 덮어쓰므로 실행 파일에서 안전하게 사용할 수 없다.
- `0xF0` 출력은 8192 FP16(16384 bytes)과 마지막 newline으로 총 16385 bytes다.
  한 실행에서 `0xF0`을 여러 번 실행하면 snapshot들이 같은 파일에 순서대로 추가되고,
  전체 파일 끝에 newline 하나만 붙는다.
- `0xFF`는 ver.08 HALT가 아니다. backend는 프로그램 끝에 `0xF0` 하나를 놓고,
  runtime은 파일에 기록된 instruction word 수까지만 실행하는 vendor 동작을 사용한다.
- vector save가 쓰는 lane 수는 연산 결과의 논리 크기가 아니라 그 시점의 `vlen`이다.
  따라서 Reduce Sum/Max scalar 결과를 저장하려면 save 직전에 `vlen=1`이 필요하다.
- 제공된 64개 예제 중 대부분은 0710 인코딩을 그대로 보존한다. 새 MAIN/PARTIAL
  행·열 설정을 실제로 사용하는 대표 예제는 `inst_1003_matrix_add_matrix`다.
  따라서 새 기능 검증은 제공 예제만으로 충분하지 않아 별도 경계/음수/MAC/GELU
  프로그램을 추가했다.

## 3. 구현 구조

| 파일 | 역할 |
|---|---|
| `npu_compiler/isa_0818.py` | ver.08 전용 bit-exact encoder와 assembler |
| `npu_compiler/runtime_0818.py` | 제공 vendor `a.out` 실행, 고정 입출력 파일 및 용량 검사 |
| `_poc/mysim_0818.cpp` | 실행 파일의 동작과 quirk를 그대로 구현한 분석 가능한 소스 C-model |
| `npu_compiler/backend_0818.py` | Relax에서 ver.08 명령으로 내리는 row-major backend |

사용법:

```python
asm, mp = driver.compile_module(mod, backend="0818")
result = driver.run_compiled(asm, mp, inputs)
```

`driver.run_compiled`은 assembler의 format version을 보고 자동으로 제공 vendor
C-model을 실행한다. 소스 C-model은 compiler의 기본 실행 대상이 아니라 parity
oracle의 반대편이며, backend의 최종 대상은 요청대로 vendor `a.out`이다.

## 4. Layout 결정

ver.08 backend의 G-buffer tensor는 모두 **row-major**다. 기존 0710 backend에서
필요했던 tile-blocked global layout, weight pre-pack, matmul gather/scatter는 사용하지
않는다. Matmul은 다음과 같이 원본 row-major tensor를 직접 읽고 쓴다.

1. MAIN descriptor에 전체 행렬 address/rows/cols를 기록한다.
2. PARTIAL descriptor에 현재 PE tile의 시작 address/rows/cols를 기록한다.
3. 최대 64x64 PE tile을 load하고, K tile은 MAC bit 27로 누산한다.
4. 결과 PARTIAL region을 row-major 목적 행렬에 직접 save한다.

즉, **tile-blocked global layout은 제거**되지만 **64x64 PE execution tiling은 유지**된다.
후자는 layout 정책이 아니라 물리 PE buffer의 실행 단위다. Transpose도 strided load로
PE에서 전치된 sub-tile을 만든 뒤 목적 row-major PARTIAL region에 직접 저장하므로
별도 transpose scratch/tile layout이 없다.

## 5. 정확성 정책

- 소스 C-model은 오류까지 vendor와 같아야 하므로 native Reduce Max의 zero-seed
  동작과 vendor GELU 근사식을 그대로 재현한다.
- Relax `max` lowering은 정확해야 하므로 buggy opcode `0x19`를 사용하지 않는다.
  첫 번째 실제 열을 accumulator로 잡고 column vector-max fold를 수행한다.
- native GELU 명령을 선택하면 vendor 식 `x*sigmoid(2x)`가 결과 계약이다.
- compiler는 vendor의 8192-entry G-buffer 한계를 compile/run 시 명시적 오류로 만든다.
  program은 실제 file-backed 실행 계약대로 별도 고정 상한을 두지 않는다.

## 6. 검증 결과

- ver.08 encoder: 제공된 64개 binary, 총 13766 words decode/re-encode 일치.
- 소스 C-model: 64개 제공 프로그램의 G-buffer snapshot이 vendor와 전부 일치.
- 추가 parity: 임의 sub-tile, K-tile MAC, 음수 Reduce Max quirk, SiLU/GELU,
  broadcast, `0xF0`, 33,001-word program 및 종료 시 비자동 save를 vendor와 비교.
- backend E2E: multi-tile row-major matmul, all-negative row max, scalar broadcast,
  64x64 full-capacity transpose를 vendor에서 검증.
- 기존 0710 ISA/runtime/matmul/elementwise 회귀 테스트 통과.

실행 명령:

```bash
PYTHONPATH=d_compiler python d_compiler/tests/test_isa_0818.py
PYTHONPATH=d_compiler python d_compiler/tests/test_cmodel_0818.py
conda run -n npu-tvm python d_compiler/tests/test_backend_0818.py
```

## 7. 0710 제거 조건

현재 0710 파일은 삭제하지 않는다. 다음 조건을 충족한 후 별도 정리 단계에서 제거한다.

1. 실제 compiler graph 전체가 `backend="0818"`에서 compile되고 operator coverage가 확인됨.
2. 실제 layer/model working set이 vendor의 8192-entry G-buffer 한도 안에서 실행되거나,
   vendor가 동적/확장 G-buffer로 다시 제공됨.
3. 전체 graph의 unrolled program 크기와 실행 시간이 현실적인 범위이거나 loop/branch
   지원이 추가됨. vendor C-model에는 32768-word 고정 상한이 없다.
4. 실제 모델 weight와 입력으로 layer/decode parity 및 품질 검증이 통과함.

특히 현재 제공 실행 파일의 8192 FP16은 LLM layer의 weight/activation 전체를 담을 수
없다. ISA 기능 업데이트와 별개인 물리 C-model 한계이므로, 이 문제가 해결되기 전에
기존 실행 경로를 제거하면 전체 모델 회귀 기준도 함께 사라진다.

## 8. 확장 source target 및 decode 후속 결과

vendor 오류/quirk parity를 유지한 `_poc/mysim_0818.cpp`의 G-buffer만 동적 flat storage로
확장하고 `backend="source-0818"`을 추가했다. official Llama 3.2 3B에서 28-layer prefill,
layer별 K/V cache seed, exact-context autoregressive decode 두 step을 수행했다.

- generated ids: `[358, 2846, 4560]`
- decoded text: `" I'm trying"`
- Hugging Face greedy reference와 전부 일치
- logical stride 128256인 LM head는 16-bit descriptor field를 변경하지 않고 `[K,64]`
  RHS panel lowering으로 실행
- vendor binary는 8192 parity oracle로 유지하고, expanded full model만 source target 사용

실행기는 `run_v3_source_generate.py`, 독립 reference 생성기는
`make_v3_generation_reference.py`다. 세부 issue, 수치 및 실행 규모는
`report/report_0818.md`의 Stage V3.7~V3.10에 기록했다.
