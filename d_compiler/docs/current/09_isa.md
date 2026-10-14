# 9. Scheduled TIR에서 ISA와 실행 bundle까지

[목차](../README.md) · 이전: [메모리](08_memory.md) · 다음: [Runtime](10_runtime_llama.md)

이 단계는 계산과 배치가 정해진 TIR를 실제 명령으로 펼친다. 현재 NPU 프로그램은
정적으로 전개된 명령 스트림이다. TIR에 loop가 있다고 최종 ISA에도 loop 명령이
남는 것은 아니다. 이것이 큰 모델의 ISA words와 compile 시간에 영향을 준다.

## 입력은 주소가 연결된 TIR다

BufferLoad/Store의 index를 실제 SRAM/DRAM 주소로 바꾸고, tensorized matmul은
입력 descriptor와 M_MUL로, 원소별 수식은 vector 단계로, 복사 stage는 DMA 또는
SRAM 내부 이동으로 내린다. Python walker와 C++ native walker는 같은 계약을 따른다.
Native 미지원 기능의 fallback과 잘못된 schedule 거절을 구분한다. 내부 오류나 계약
hash 불일치를 조용히 성공한 fallback으로 숨기지 않는다.

## 네 가지 이동 명령

| 명령 | 출발 → 도착 | 데이터가 DRAM을 지나는가? |
|---|---|---|
| GLOAD | DRAM → SRAM | 예 |
| LOAD | SRAM → 연산장치 입력 | 아니오 |
| SAVE | 연산장치 결과 → SRAM | 아니오 |
| GSTORE | SRAM → DRAM | 예 |

H가 SRAM에 보관되면 H의 GSTORE/GLOAD가 없어질 수 있지만, H를 matrix unit에
넣는 LOAD까지 없어지는 것은 아니다. 서로 다른 메모리 경계를 구분한다.

## 주소를 읽기 전에 단위를 확인한다

| 위치 | 단위 | Byte 변환 |
|---|---|---|
| 기본 DRAM plan.address | `unit_bytes`, 기본 2 bytes | 값 × unit_bytes |
| DMA global 주소/stride/cols | 32-bit cell | 값 × 4 bytes |
| SRAM descriptor 주소 | nibble | 값 ÷ 2 bytes |
| shape/row stride/VLEN | element 수 | dtype 크기로 변환 |

한 숫자를 서로 다른 단위로 읽는 오류는 2배/4배 어긋난 주소를 만든다. 기본 FP16
행렬의 stride 128은 128 bytes가 아니라 256 bytes다. MAIN descriptor는 전체 stride,
PARTIAL descriptor는 이번 접근 영역의 시작과 크기를 표현한다.

## 실제 77-word 예제를 해독한다

[바이너리](../examples/two_matmuls_recorded/program_memory.bin)는 308 bytes다.
32-bit words 77개지만 명령은 62개다. DMA는 명령 하나가 4 words이고 그 외 해당
명령은 1 word이기 때문이다. 이 프로그램에는 GLOAD 4개, GSTORE 1개가 있다.

```text
상수 DRAM → SRAM
W1 DRAM → SRAM
X DRAM → SRAM
첫 matmul, H의 왼쪽/오른쪽 출력 타일 계산
W2 DRAM → SRAM (W1 공간 재사용)
두 번째 matmul, K 타일 두 개를 누적
Y SRAM → DRAM
HALT
```

단위 변환 예로 X의 DMA packet은 다음과 같다.

```text
word offset 8:
000c18a0 00000000 00000800 00010800
```

Opcode A0는 GLOAD다. 첫 word의 주소 필드는 SRAM nibble 0xC18=3096,
즉 SRAM byte 1548이다. Global 주소는 0 cell, 한 행 0x800=2048 cells,
총 8192 bytes를 가져온다. DRAM의 X가 앞 장의 SRAM 주소 1548에 도착한다.

| Word offset | 명령의 의미 |
|---:|---|
| 30, 44 | 첫 matmul의 서로 다른 H 출력 타일, 각각 MAC=0 |
| 60 | 두 번째 matmul의 첫 K 타일, MAC=0 |
| 67 | 같은 출력의 다음 K 타일, MAC=1 |
| 71 | K 누적을 마친 Y를 SRAM에 SAVE |
| 72 | Y를 DRAM에 GSTORE하는 4-word packet |
| 76 | HALT |

`80000042`와 `88000042`의 차이는 accumulate bit다. 둘은 모두 activation OFF다.
첫 matmul의 두 출력 타일을 서로 누적하면 틀리므로 둘 다 새 accumulator를 시작한다.

## Linker의 마무리

커널별 명령을 실제 호출 순서로 연결하고 필요한 scalar pool과 descriptor 상태를
관리한다. 사용 가능한 SRAM 사본을 추적하는 forwarding과 효과 분석 기반 peephole로
중복 이동/설정을 줄인다. 사용하지 않는 DRAM pool을 제거한 뒤 주소를 일관되게
재배치하고 HALT로 끝나는 프로그램을 만든다.

이때 효과 분석은 “이 명령이 어떤 주소를 읽고 쓰는가”를 아는 것이다. 단지 이전과
인코딩이 같다고 명령을 삭제하지 않는다. 중간에 해당 SRAM이 덮어써졌으면 사본은
더 이상 유효하지 않다.

## Bundle에서 확인할 파일

| 파일 | 담긴 내용 | 담기지 않은 것 |
|---|---|---|
| program_memory.bin | little-endian uint32 명령 | weight 값 전체 |
| program.json | 주소·통계·선택·출력 ABI·provenance | 실장치 측정 cycle |
| lowered.py | 선택된 semantic IRModule | 최종 주소를 포함한 모든 runtime 상태 |
| scheduled_*.py | joint-memory에서 실제 선택된 호출별 TIR | 원래 모델 설명 전체 |
| global_memory.bin | values를 지정한 경우 초기 DRAM image | 지속적인 SRAM 상태 |

Binary만으로 inference 결과를 얻을 수 없다. 같은 plan에 맞는 입력과 weight image가
필요하다. 최종 shape·dtype·output offset을 사용하여 결과를 읽어야 한다.
작은 [pipeline bundle](../examples/pipeline/bundle/program.json)은 입력 image까지 보존한다.
전체 모델은 매우 큰 image와 중복 pickle을 문서에 보관하지 않고 측정·hash를 남겼다.

구현: [linker](../../npu_compiler/npu_link.py),
[Python codegen](../../npu_compiler/tir_codegen_v09.py),
[native codegen](../../npu_codegen/v09_walker.cc), [명령 통계](../../npu_compiler/v09_stats.py).

현재 `export_bundle`은 joint-memory 경로의 scheduled TIR를 내보낸다. 일반 recipe 경로의
단일 fused 커널은 자동 dump 대상이 아니므로, 작은 예제 실행기는 실제 compile context에
저장된 recipe를 꺼내 [07_scheduled_fused_matmul_add_silu.py](../examples/pipeline/07_scheduled_fused_matmul_add_silu.py)로 남겼다.
선택된 함수와 진단이 일치하는지 검사하며 schedule을 다시 탐색하지 않는다.
