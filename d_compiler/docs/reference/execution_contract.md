# 현재 v09 compiler의 실행 계약

[목차](../README.md). 이 파일은 현재 코드가 사용하는 계약을 정리한다.
실제 RTL이 모든 항목을 구현·검증했다는 선언이 아니다. 원본 ver.08 vendor 자료는
저장소의 `0818_npu_update`에 남긴다.

## 메모리와 인코딩

| 항목 | 현재 의미 |
|---|---|
| Global | DMA만 접근, 주소 단위 32-bit cell, 32-bit field |
| SRAM | 기본 8 MiB, nibble 주소, 유효 24-bit |
| MAIN/PARTIAL | MAIN의 stride와 PARTIAL의 실제 접근 영역을 구분 |
| Rows/cols/stride | element 수, 16-bit field |
| Dtype | descriptor의 FP16/FP32/INT8/INT4 표현. 각 명령별 허용 조합은 별도 |
| GLOAD/GSTORE | 동기 DMA, 4-word, dtype 없는 cell 이동 |
| Matrix | 기본 tile64, FP32 MAC chain, 최종 FP16 저장 |
| Vector | 기본 산술 FP16 입력, 내부 연산 결과. 정해진 ABI에 FP32 save 등 사용 |
| VLEN | 현재 논리 길이 1..65535. Softmax 수치 한도는 65504 |
| HALT | 0xFF. Stream 정상 종료 |

DMA packet: w0 상위24-bit는 SRAM nibble 주소, 하위8-bit는 opcode.
w1은 global cell 주소, w2는 global row stride cells, w3는 rows/cols 각각16-bit다.
SRAM의 DMA 행은 연속 배치다. 4-byte cell 경계의 부분 데이터는 이웃 값을 손상하지
않도록 별도 처리해야 한다. Int8의 홀수 offset을 무조건 zero-copy할 수는 없다.

## Vector 256에 대해 확정된 것과 미확인인 것

`device.py`는 lanes=256을 정의한다. 2026-08-26 S3 문서 결정과 현재 compiler/C-model은
긴 논리 vector를 한 명령으로 처리하며 내부에서 256단위로 반복한다는 전제다.
이전 ISA 표에는 상한256 설명이 남아 있었고 새 문서에서는 이 충돌을 숨기지 않는다.
현재 C-model은 VLEN 전체를 C++ 배열로 보관한다. 물리 buffer 수용량, 내부 주소 증가,
LOAD/연산/SAVE 사이의 상태와 tail lane 처리의 RTL 검증은 별도로 필요하다.

따라서 “compiler가 256으로 나누지 않으므로 즉시 오류”도, “C-model이 되므로
하드웨어까지 충분히 검증됨”도 단정하지 않는다. 실제 ISA 상한이 256으로 확정되면
공통 vector lowering에서 분할하고 긴 reduction의 누산·반올림 계약도 함께 검토한다.

## Descriptor와 주요 field를 읽는 방법

Descriptor는 다음 LOAD/연산/SAVE가 사용할 주소·shape·dtype 설정이다. 명령마다
배열 전체 정보를 반복하여 넣는 대신, 설정 상태를 바꾸고 뒤의 명령이 참조한다.
주소가 이전 값과 같아 보여도 중간 명령이 설정을 바꾸었는지 확인해야 한다.

| Opcode | 주요 field | 의미 |
|---|---|---|
| 0x80 | operand[31:30], high-half[29], MAIN/PARTIAL[28], value[23:8] | SRAM 주소를 16-bit씩 설정. 상·하위 half를 모두 설정하여 이전 상태가 남지 않게 함 |
| 0x88 / 0x89 | operand[31:30], MAIN/PARTIAL[29], dtype[26:25], rows/cols[23:8] | 원소 단위 행·열 수. MAIN cols는 전체 row stride를 전달 |
| 0x82 | VLEN[23:8] | Vector 논리 길이 |
| 0x90 / 0x98 | matrix[31], strided[29], ncols[23:16], start[15:8] | SRAM↔연산장치 LOAD/SAVE. LOAD의 operand[30]은 두 입력 중 선택 |
| 0x42 | mode[31:30], activation[29:28], MAC[27] | M_MUL. MAC=0 새 출력 누산, MAC=1 기존 출력 누산에 더함 |
| 0x8A / 0x8B | high-half[29], value[23:8] | Activation/weight scale의 SRAM nibble 주소 |

Operand는 SRC1=0, SRC2=1, DST=2이며 이름과 실제 수학적 좌변/우변을 함께 본다.
Mode는 IMM=0, SCALAR=1, VECTOR=2다. Dtype은 FP16=0, FP32=1, INT8=2, INT4=3이다.
Activation은 OFF=0, 표준 tanh-GELU=1, SiLU=2, vendor legacy GELU=3이다.
이 값이 인코딩에 존재하는 것과 graph compiler가 그 조합을 자동 선택하는 것은 다르다.
특히 현재 FP16 TIR matmul은 activation OFF다.

주소 예: byte 1548은 nibble 3096이다. `[64,128]` FP16 H에서 오른쪽 64열 타일의
시작은 H의 byte 주소+128이며 row stride는 여전히 128 elements=256 bytes다.
PARTIAL 주소와 크기만 바꾸고 MAIN stride는 유지해야 올바른 위치에 SAVE한다.

## 산술 opcode와 데이터 배치

| 분류 | 명령 |
|---|---|
| Vector 기본 | ADD 0x01, SUB 0x02, MUL 0x0A, DIV 0x0B, MULADD 0x0C, MOVE 0x0D |
| Vector 수학 | SQRT 0x0E, EXP 0x0F, SIGN_INV 0x16, COSSIN 0x18 |
| Reduce·확장 | REDUCE_SUM 0x14, BROADCAST 0x15, REDUCE_MAX 0x19 |
| 기타 공유 인코딩 | LOGICAL 0x08, SHIFT 0x09, COMPARE 0x11, MINMAX 0x12, CONVERT 0x13, COPY 0x17 |
| Matrix | ADD 0x40, SUB 0x41, MUL 0x42, MOVE 0x43 |
| Quantization | VQUANT 0x1A, VDEQUANT 0x1B |
| 제어 | NOP 0x00, SNAPSHOT 0xF0, HALT 0xFF |

모든 산술 opcode가 모든 Relax op/dtype으로 노출되는 것은 아니다. Immediate 산술의
16-bit 값은 v09에서 signed 의미를 가진다. Reduce-max는 첫 원소로 시작하며 과거
ver.08의 0 seed를 그대로 사용하지 않는다. 표준 GELU와 legacy GELU도 구분한다.

Global image와 program words는 little-endian이다. 32-bit cell에는 FP16 두 원소,
INT8 네 원소, INT4 여덟 원소를 넣을 수 있다. INT4는 낮은 nibble부터 순서대로
배치하며 signed 값을 해석해야 한다. 이 표현 지원이 현재 INT4 모델 pipeline의
검증 완료를 의미하지는 않는다.

Scale descriptor는 FP32 배열을 가리킨다. Mixed matrix ISA는 지원 조합에 대해
행별 activation scale과 열별 weight scale을 적용한 K tile 결과를 FP32 MAC에 더한다.
K group의 scale이 달라지면 group 경계에서 scale 주소를 바꾼다. 현재 공개 compiler의
W8A16은 이 mixed matrix 경로 대신 FP16 복원 weight를 사용하므로
[양자화 장](../current/13_quantization.md)의 실제 경로를 따른다.
별도 carry/drain flag로 두 번째 accumulator를 유지하는 기능은 현재 계약이 아니다.

## 수치 의미

FP16 matmul은 K=64 FP32 부분합들을 FP32 accumulator에 더하고 완료 후 FP16 저장한다.
중간 vector 작업으로 FP16 partial store/reload가 필요한 schedule은 거절한다.
일반 vector 표현식은 FP16 SRAM 중간 저장을 포함한다. Softmax와 scaled RMSNorm의
정의는 semantic TIR에 명시되어 있다. 단순히 더 높은 precision으로 계산하면 현재
reference와 bit-exact할 것이라고 가정하지 않는다.

W8A8은 INT32 dot64 → FP32 scales → FP32 tile 누산 → FP16 저장이다.
일부 ISA가 지원하는 dtype 조합과 graph compiler에서 실제 선택하는 경로는 다르다.
Activation bit가 ISA에 있어도 현재 자동 TIR matmul은 OFF다.

## ABI와 검증 경계

입력·parameter·상수·외부 출력·cache update는 최종 byte 주소 계약을 가진다.
SRAM-only 중간값은 최종 외부 tensor인 것처럼 내보내지 않는다. Nested output tree는
leaf별 shape/dtype/offset을 보존한다. Cache append 주소는 최종 DRAM 압축 뒤 plan에서
얻는다. Alias와 in-place 갱신의 의미를 확신할 수 없으면 기존 DRAM 계약을 유지한다.

Native library는 공통 JSON 계약 hash와 C++ source identity를 확인한다. Unsupported
coverage는 Python fallback이 될 수 있지만 invalid schedule과 내부 오류는 구분한다.
Backend 구현의 source hash, native binary hash, C-model source hash, TVM commit을
함께 기록한다. 문서 수정은 compiler source digest 자체에 포함되지 않는다.

## 보장하지 않는 것

실장치 latency/cycle, 모든 동적 shape, 모든 TVM op, 모든 NaN/Inf edge semantics,
모든 HF 모델, 자동 mixed precision, quantized KV, key online-softmax, 임의 inter-kernel
tile streaming은 현재 지원·검증 완료 범위가 아니다. 상세 상태는
[지원 범위와 보류 이슈](../current/12_operations_limits.md)를 따른다.

원본 구현: [공통 계약 JSON](../../npu_compiler/v09_contract.json),
[ISA 인코더](../../npu_compiler/isa_compute.py), [DMA 인코더](../../npu_compiler/isa_v09.py),
[C-model](../../../_poc/mysim_v09.cpp).
