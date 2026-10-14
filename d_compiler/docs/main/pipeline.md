# Main backend의 실제 실행 흐름

[Main 개요](README.md) · [결과](results.md)

이 설명은 오래된 README의 “다음 단계”가 아니라 고정한 main commit의 실행 함수들을
읽어 작성했다. 특히 class docstring에 남은 분리 decode 설명보다 실제
`decode_token`이 호출하는 fused layer graph를 기준으로 한다.

## 1. 모델과 shape를 준비한다

Llama32Assets가 checkpoint의 embedding·linear·norm 배열을 읽는다.
Llama32SourceCompiler는 sequence 길이를 받아 prefill layer graph, final norm,
LM-head GEMM을 준비한다. Model family의 계산과 공통 실행 bookkeeping은 나뉘어 있다.
Gemma/Qwen은 각자 graph/assets adapter를 사용한다.

Prefill layer의 입력에는 hidden, Wn1/Wn2, Wq/Wk/Wv/Wo/Wg/Wu/Wd가 있다.
한 프로그램을 여러 layer에서 재사용하며 입력 weight만 해당 layer의 것으로 바꾼다.
프로그램을 재사용한다는 사실이 weight가 연산장치에 계속 남는다는 뜻은 아니다.

## 2. Relax를 row-major G-buffer 연산으로 내린다

`driver.compile_module(..., backend="source-0818", reuse=True)`가
backend_0818의 planner/emitter를 사용한다. NPU별 emitter가 Relax의 연산을 보고
matmul, copy, norm 등의 명령을 생성한다. 최신 v09처럼 모든 연산을 semantic TIR로
낮춘 뒤 일반 schedule 탐색을 수행하는 경로로 설명하면 틀리다.

Memory plan은 parameter·상수·중간값의 flat offset과 shape를 정한다. 수명 재사용을
켜면 더 이상 쓰지 않는 중간값의 주소를 재사용한다. Matrix는 row-major 상태로
MAIN stride와 PARTIAL 시작/크기를 설정한다.

```text
각 출력 row/column tile:
    각 K tile:
        A sub-tile LOAD
        B sub-tile LOAD
        M_MUL (첫 K는 MAC=0, 다음은 MAC=1)
    출력 sub-tile SAVE
```

끝 타일은 실제 남은 row/column/K 크기를 descriptor에 넣는다. 이전 tile-blocked
global packing을 매번 요구하지 않는다. 매우 넓은 LM head는 별도 panel 예외가 있다.

## 3. Vendor quirk를 compiler에서 우회한다

Vendor reduce-max가 0에서 시작하면 음수만 있는 행의 최댓값이 틀릴 수 있다.
Backend는 실제 첫 원소로 시작하는 vector max 조합 등으로 올바른 모델 계산을 만든다.
Activation/GELU, 즉시값 등 vendor 차이도 기록된 우회가 있다. C-model을 이상적인
수학으로 임의 변경하여 compiler가 맞는 것처럼 검증하지 않는다.

이 호환성 정책과 최신 v09에서 명시적으로 고친 ISA/C-model 의미는 같은 것이 아니다.
양쪽 검증 결과를 비교할 때 각 수치 계약을 먼저 확인한다.

## 4. 큰 LM head는 왜 특별한가?

Llama의 logical RHS는 [3072,128256]이다. 16-bit MAIN stride field에 128256을
직접 넣을 수 없다. `PackedRhsGemm`은 출력 열을 최대64개씩 묶어 K×64 panel로
저장하고 각 panel의 작은 stride로 계산한다. Wide shape 전체를 field에 조용히
masking하는 것은 올바른 처리법이 아니다.

Host가 weight panel을 준비하고, one-row GEMM 프로그램이 panel들을 차례로 사용한다.
이 특수 경로는 당시 full-vocabulary LM head를 실행하기 위한 구조다.

## 5. Prefill에서 cache seed까지

Host가 token embedding을 준비한다. 28개 layer를 host loop로 돌면서 같은
prefill 프로그램에 각 layer weight와 hidden을 넣는다. 각 결과에는 다음 hidden과
그 layer의 K/V가 함께 있다. Host는 이를 분리해 KV head별 key/value 배열로 보관한다.
마지막 hidden 한 행을 norm과 LM head로 처리하고 argmax로 첫 token을 고른다.

## 6. Decode는 context별로 프로그램을 만든다

`_decode_program(context)`는 처음 본 context 길이이면 fused decode layer graph를
컴파일하고 dictionary에 보관한다. 같은 context에서는 재사용하지만 길이가 바뀌면
다른 shape의 프로그램이 필요하다.

최종 `decode_token`은 Q/K/V projection과 현재 token의 attention 입력 concat을
포함한 layer 프로그램을 실행한다. 현재 K/V를 반환받아, 모든 layer 실행 뒤 host가
기존 cache 배열에 concatenate한다. Attention 안의 현재 K/V 사용과 다음 step에
보존하는 cache 갱신을 구분한다.

## 7. Source runtime의 I/O

매 실행에서 입력·weight·상수를 G-buffer에 배치하고 program_memory.bin과
G_buffer_data.bin을 쓴다. Source C-model을 실행한 뒤 0xF0 snapshot으로 나온
saved_G_buffer_data.bin의 출력 범위를 읽는다. Host invocation 횟수와 파일 I/O도
실행 시간에 포함될 수 있다. 실제 NPU cycle로 간주하지 않는다.

원본 파일은 main commit의 `driver.py`, `backend_0818.py`, `v3_source_llama.py`,
`generation.py`, `source_gemm_0818.py`, `source_runtime_0818.py`다.
