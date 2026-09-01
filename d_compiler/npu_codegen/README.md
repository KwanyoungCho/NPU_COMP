# native codegen — 스케줄된 TIR → v09, C++로

```
./build.sh            # -> libnpu_codegen.so (커밋하지 않는다)
TVM_HOME=/other/tvm ./build.sh
```

## 왜 있나

v09에는 분기·루프 명령이 없어서 프로그램이 완전히 언롤된 하나의 직선 스트림이다.
즉 방출량이 **언롤된 반복 횟수에 비례**한다. 이걸 파이썬으로 하면 반복마다 TIR
노드를 만지는데, TIR 노드는 자식 속성을 읽는 것(`e.a`)도 dict 키로 해시하는 것도
전부 FFI 호출이라 그 비용이 반복 횟수만큼 곱해진다. TVM 자신의 codegen
(LLVM/CUDA/C)이 전부 `src/target/` 밑의 C++ 방문자인 이유가 이것이다.

실측: 실차원 Llama 1층(495,386 word) 링크가 파이썬 40.6s → native **4.6s (8.9배)**.
전체 28층은 8,147s → **127.0s (64.1배)**. 자세한 내역은 `report/report_0901.md`
§7.1·§11.1과 `OPTIMIZATION_BACKLOG.md` A5·A6.

## 무엇이 정의인가

**`npu_compiler/tir_codegen_v09.py`의 파이썬 Walker가 lowering의 정의다.**
여기 있는 C++은 더 빠른 두 번째 구현일 뿐이고, 모르는 구조를 만나면 예외를 던진다.
그러면 링커가 **그 커널만 파이썬으로 방출**한다 — 커버리지가 모자라도 속도만
손해이고 결과는 달라지지 않는다.

## 구성

| 파일 | 내용 |
|---|---|
| `v09_isa.h` | word 인코더 (`isa_0818.py` + `isa_v09.py` 포팅). 범위 검사도 포팅 대상이다 — 벤더 인코딩은 조용히 마스킹하고, 16-bit vlen 초과가 한 번은 `[7,128256]` 복사를 45,696개만 옮기게 했다 |
| `v09_asm.h` | 어셈블러와 SRAM/DMA emitter (`Asm`/`V09Asm`/`SramEmitter` 포팅). 서술자 상태는 일부러 추적하지 않는다 — 중복 설정은 peephole이 지운다 |
| `v09_walker.cc` | walker 본체와 TVM 등록 (`npu.codegen_kernel`) |
| `codegen_v09.cc` | 인코더 자기검증 진입점 (`npu.encode_selftest`) |

TVM 소스 트리를 건드리지 않는 **out-of-tree** 빌드다. `libtvm.so`에 링크해
TVM 전역 레지스트리에 등록하므로 TVM 자체는 다시 빌드하지 않는다.

## 켜고 끄기

라이브러리가 있으면 기본으로 켜진다.

```
NPU_NATIVE=0                                  # 끄기
compile_program(..., native_mode="python")    # 파이썬만
compile_program(..., native_mode="use")       # native + 폴백 (기본)
compile_program(..., native_mode="compare")   # 파이썬으로 방출하되 word 대조
```

## 검증

`tests/test_native_codegen.py` 가 세 겹으로 막는다.

1. 모든 인코더·서술자·DMA 형태를 같은 스크립트로 돌려 word 대조
2. 네 그래프(llama prefill · qwen3 prefill · prefill_cache · decode)에서
   **커널별 word 대조** (`native_mode="compare"`)
3. 프로그램 전체가 파이썬 단독 결과와 **word 하나까지 동일**한지
4. native가 거부한 커널이 예외가 아니라 폴백으로 처리되는지

Gemma는 빌더가 실제 체크포인트 spec을 요구해서 여기 대신 `test_nn_gemma`가
native 켜진 채로 전체 링크·실행하며 덮는다.

**여기를 고치면 파이썬 walker와의 대조가 통과하는지부터 확인할 것.** 한 word가
달라도 증상은 "숫자가 이상하다"뿐이다.
