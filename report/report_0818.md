# 0818 Vendor C-model 기반 Compiler V3 진행 보고서

> 시작일: 2026-08-18
>
> 기준 브랜치: `compiler-v2` → 개발 브랜치 `compiler-v3`
>
> 1차 목표: **vendor 0818 C-model만으로 Llama 3.2 3B full-model prefill을 실행하고,
> logits에서 정상적인 next token을 출력한다.**

이 문서는 compiler-v3 개발 중 확인된 사실, 재현 방법, 해결 여부와 검증 결과를
누적 기록한다. issue는 `V3-###` 식별자를 유지하며, 해결 커밋과 회귀 테스트를 함께
기록한다.

---

## 1. 완료 기준

1. compiler/runtime의 최종 실행 대상은
   `0818_npu_update/a_npu/a.out` 하나다. `_poc/mysim_0818.cpp`는 분석 및 vendor
   parity 검증에만 사용한다.
2. Meta Llama 3.2 3B의 실제 config와 weight를 사용하여 prompt prefill 전체 layer가
   끝까지 실행된다.
3. 동일 FP16 입력에 대한 reference 실행과 주요 layer/checkpoint를 비교한다.
4. 최종 logits의 argmax token이 reference와 일치하고, tokenizer로 decode 가능한
   token을 출력한다.
5. 모든 우회·제약·정확도 차이를 이 문서에 기록하고 재현 테스트를 남긴다.
6. 단계별 완료 시 report를 포함하여 commit하고 `origin/compiler-v3`에 push한다.

---

## 2. 0818 기준선

### 2.1 구현

- ver.08 전용 ISA encoder/assembler
- 고정 파일 인터페이스를 사용하는 vendor runtime
- vendor 실행 의미를 재현한 분석용 C++ source C-model
- MAIN/PARTIAL descriptor로 row-major tensor의 sub-tile을 직접 접근하는 backend
- `driver.compile_module(..., backend="0818")` 및 vendor 자동 실행

### 2.2 검증

- 제공 64개 프로그램, 13,766 instruction words decode/re-encode 일치
- 64개 프로그램의 vendor/source G-buffer snapshot 및 실행 trace 본문 일치
- 임의 sub-tile, MAC, broadcast, Reduce Max quirk, SiLU/GELU, 반복 `0xF0` parity
- multi-tile matmul, row sum/max, full-capacity 64x64 transpose vendor E2E
- 기존 0710 ISA/runtime/matmul/elementwise 회귀 유지

### 2.3 Layout 결정

- G-buffer tensor: row-major
- 제거: 0710용 tile-blocked global layout, weight pre-pack, matmul gather/scatter
- 유지: 물리 PE 실행 단위인 최대 64x64 tiling과 K-tile MAC

---

## 3. Issue tracker

| ID | 상태 | 심각도 | 내용 | 현재 대응 |
|---|---|---:|---|---|
| V3-001 | MITIGATED | BLOCKER | vendor G-buffer가 8192 FP16으로 고정되어 3B weight는 물론 일반적인 한 layer working set도 한 번에 담을 수 없음 | host-resident tensor와 8192 이하 `[A,B,C]` window로 분할. vendor 한계 자체는 유지 |
| V3-002 | RESOLVED | HIGH | program memory가 32768 words 고정이라고 판단했으나 실제 실행기는 전체 file을 `malloc`하여 실행 | 32K compiler/runtime guard 제거. 33,001-word program의 마지막 `0xF0` vendor/source parity로 검증 |
| V3-003 | MITIGATED | HIGH | native Reduce Max `0x19`가 accumulator를 0으로 시작하여 all-negative row를 0으로 반환 | compiler는 첫 실제 열에서 시작하는 vector-max fold 사용. source C-model은 vendor bug 그대로 재현 |
| V3-004 | OPEN | HIGH | vendor GELU가 표준 GELU가 아니라 `x*sigmoid(2x)` | native GELU 사용 시 모델 정확도 영향 측정. 필요하면 표준식을 primitive 조합으로 lowering |
| V3-005 | MITIGATED | MEDIUM | `0xF0`은 HALT가 아니며 snapshot 후 계속 실행 | generated program의 마지막 유효 word로 `0xF0` 하나만 배치 |
| V3-006 | MITIGATED | MEDIUM | vector save lane 수가 연산 결과 크기가 아니라 현재 `vlen`으로 결정 | scalar reduction 직후 `vlen=1` 명시 |
| V3-007 | OPEN | MEDIUM | 제공 64개 예제 대부분이 0710 encoding을 유지하여 새 기능 coverage가 부족 | compiler-owned targeted vendor parity 프로그램 유지·확장 |
| V3-008 | RESOLVED | BLOCKER | full checkpoint와 tokenizer가 로컬에 없어 실제 token 검증 불가 | official revision `13afe512...`의 2개 shard와 tokenizer를 ignored build 영역에 확보. 총 weight 6,425,529,048 bytes |
| V3-009 | RESOLVED | HIGH | official weight는 BF16이지만 vendor G-buffer 입력은 FP16 | safetensors tile에서 FP16 변환. fresh full logits cosine 0.999988, argmax token HF와 일치 |
| V3-010 | MITIGATED | HIGH | tied embedding/lm_head가 128256x3072이며 전체 복사 시 약 788MB | embedding row와 LM-head `[K,V]` tile만 safetensors에서 slice하여 로드 |
| V3-011 | RESOLVED | BLOCKER | A/B/C 64x64 세 tile은 12288 FP16이라 한 invocation에 들어가지 않음 | tile별 `m*k + k*n + m*n <= 8192`를 만족하도록 K를 동적으로 축소. 64x64 출력은 K=32 |
| V3-012 | MITIGATED | HIGH | vendor invocation 사이에는 PE MAC 상태가 사라짐 | running C를 snapshot에서 host로 carry하고 다음 invocation 시작 시 PE output으로 load한 후 MAC bit 27 실행 |
| V3-013 | MITIGATED | HIGH | vendor stdout trace가 큰 실행에서 pipe 메모리·I/O를 폭증시킴 | parity 때만 capture하고 production streaming은 `/dev/null`. vendor 내부 formatting 비용은 잔존 |
| V3-014 | RESOLVED | HIGH | SiLU는 독립 unary opcode가 아니라 matrix ALU activation field라 기본 VECTOR mode 사용 시 미초기화 SRC2를 참조 | `matrix add immediate 0 + ACT_SILU`로 identity 연산 후 activation 수행 |
| V3-015 | RESOLVED | BLOCKER | 기존 Relax backend는 graph 전체 tensor를 한 G-buffer에 배치하므로 real layer가 8192 capacity를 근본적으로 초과 | Relax binding을 host-resident value와 bounded vendor kernel 호출로 compile하는 `RelaxVendorPlan` 구현 |
| V3-016 | MITIGATED | MEDIUM | indirect addressing이 없어 slice/concat을 vendor 안에서 동적 주소로 연결할 수 없음 | slice/concat은 산술 없는 host layout operation, transpose/broadcast와 모든 model arithmetic은 vendor opcode로 실행 |
| V3-017 | RESOLVED | HIGH | per-head weight API는 official fused q/k/v/o tensor와 직접 대응하지 않아 full checkpoint mapping 오류 위험 | fused-projection Relax graph 추가: Q/K/V 1회 projection, head slice/GQA, context concat 후 official O projection |
| V3-018 | MITIGATED | HIGH | vendor executable을 weight window마다 새 process로 실행해 layer 0 serial wall 177초 | 독립 output-column group만 4개 workdir에서 병렬 실행. serial과 bit-exact, layer 0 wall 67.8초 |
| V3-019 | RESOLVED | HIGH | streaming FP16 경계가 HF eager FP16보다 많아 layer별 수치 drift 누적 가능 | 독립 HF reference로 전 layer 추적. fresh final normalized cosine 0.999803, logits cosine 0.999988, argmax 일치 |
| V3-020 | RESOLVED | BLOCKER | RMSNorm이 `x*x` 후 `/D`하여 residual outlier 330.75의 square가 FP16 `inf`; BOS가 layer 2부터 330.75에 고정되고 attention 오염 | `x/sqrt(D)`를 먼저 FP16 적용한 뒤 square/reduce. 330.75 입력 full-D RMSNorm finite, float reference mean abs 7.75e-6 |
| V3-021 | MITIGATED | LOW | TVM conda 환경에 `pytest` 모듈이 없어 collection 명령 사용 불가 | repository의 23개 `test_*.py` 자체 entrypoint를 직접 실행. 환경 package 설치 없이 전체 범위 검증 |
| V3-022 | RESOLVED | MEDIUM | RMS binding 추가 후 legacy A1 reuse에서 동일 physical offset이 free-list에 중복 삽입되어 live tensor 충돌 | free-list를 offset set으로 변경하고 bump 조건 수정. prefill/decode byte-exact, v2 comprehensive 회귀 통과 |
| V3-023 | MITIGATED | BLOCKER | G-buffer generator만 8192보다 크게 만들면 vendor `a.out`의 16384-byte 정적 배열을 넘겨 BSS를 덮어씀 | vendor runtime은 계속 8192 초과를 거부하고, full-model은 parity-tested 동적 source C-model target으로 분리 |
| V3-024 | RESOLVED | HIGH | source C-model의 G-buffer를 확장하면서 vendor arithmetic/quirk parity가 달라질 위험 | 기본 8192 parity는 64개 program+targeted quirk로 유지하고, compiler-owned source runtime만 동적 flat buffer와 quiet mode 사용 |
| V3-025 | RESOLVED | BLOCKER | 8K 안에서는 드러나지 않던 scalar broadcast 주소 상위 16-bit 누락으로 확장 source RMSNorm이 거의 0 출력 | opcode `0x15` low/high를 매번 모두 emit해 stale high state도 제거. address 70000 broadcast 및 official layer 0 검증 |
| V3-026 | RESOLVED | HIGH | official V3 prefill output에 K/V가 없어 decode cache를 seed할 수 없음 | fused prefill이 `[hidden,K0,V0,...]`을 반환하는 선택 경로와 exact-context fused decode/KV graph 구현 |

---

## 4. 진행 로그

### 2026-08-18 — Baseline: ver.08 retarget

- PDF와 vendor executable을 독립적으로 분석해 ver.08 field와 실행 의미를 확정했다.
- source C-model과 vendor parity를 확보했다.
- row-major direct sub-tile backend를 추가했다.
- full-model 착수 전 blocker는 V3-001/002다. ISA 기능 추가와 달리 vendor executable의
  고정 메모리 용량은 그대로이므로, compiler-v3에서는 한 번의 거대한 program이 아닌
  **vendor invocation 간 분할 실행** 가능성을 가장 먼저 검증한다.

다음 단계:

1. 현재 기준선을 `compiler-v2`에 commit/push
2. `compiler-v3` 생성/push
3. 기존 Llama 3.2 3B import·weight·prefill 경로 inventory
4. V3-001/002를 실제 최소 graph로 재현하고 분할 실행 계약 결정

### 2026-08-18 — Stage V3.1: branch 및 official model asset

- `compiler-v2` 기준선 `507322d`를 push하고 같은 지점에서 `compiler-v3`를 생성·push했다.
- Hugging Face gated access를 확인하고 `meta-llama/Llama-3.2-3B` revision
  `13afe5124825b4f3751f836b40dafda64c1ed062`를 받았다.
- checkpoint는 2개 safetensors, 6,425,529,048 bytes이며 config는 D=3072,
  F=8192, H=24, KV=8, HD=128, 28 layers, vocab=128256과 일치한다.
- tokenizer smoke test:
  - text: `Hello, NPU compiler!`
  - ids: `[128000, 9906, 11, 452, 6459, 19979, 0]`
  - decode: `<|begin_of_text|>Hello, NPU compiler!`
- `v3_model.py`는 전체 tensor를 복제하지 않고 safetensors의 필요한 row/column tile만
  BF16→FP16으로 변환한다. tied LM head도 embedding row slice를 전치해 공급한다.

남은 핵심: 모델 weight를 host에 보관하는 것은 가능하지만 vendor 한 invocation의
G-buffer는 8192 FP16뿐이다. 다음 단계는 `[A tile, B tile, running C tile]`이 동시에
8192 안에 들어오도록 GEMM을 분할하고, invocation 사이에는 snapshot의 C만 host가
다음 invocation으로 전달하는 streaming runtime이다.

### 2026-08-18 — Stage V3.2: fixed-buffer streaming vendor runtime

- `VendorSession`이 동일 임시 작업 디렉터리를 재사용하며, 매 산술 tile마다 제공
  `0818_npu_update/a_npu/a.out`을 실행한다. 분석용 source C-model은 사용하지 않는다.
- streaming GEMM working set은 `[A(m,k), B(k,n), running C(m,n)]`이고 항상
  `m*k + k*n + m*n <= 8192`, 각 PE dimension은 64 이하다.
- 첫 K tile은 일반 matmul, 이후 tile은 이전 FP16 C를 PE output에 올린 뒤 MAC bit 27로
  누산한다. invocation 경계마다 C가 FP16 반올림되는 계약을 reference에도 적용했다.
- vendor-only primitive: add/sub/mul/div/max, sqrt/exp/neg/cos/sin, SiLU,
  row Reduce Sum, all-negative-safe Reduce Max fold.
- 검증:
  - irregular GEMM `[65,70]@[70,67]` streaming reference와 bit-exact
  - elementwise/SiLU/Reduce Sum/negative Reduce Max bit-exact
  - official layer-0 q-projection `[1,3072]@[3072,128]`을 safetensors tile loader로
    실행하고 같은 FP16-boundary reference와 bit-exact
  - layer-0 Q projection 128 output 열: 96 vendor invocation, vendor 실행 누적 0.731초
    (현재 머신 1회 측정, 프로그램 3,254 words)

현재 구조는 V3-001을 기능적으로 우회한다. 남은 성능 하한은 full-model 3B weight를
8192-entry window로 읽기 위해 필요한 수십만 회의 vendor process 실행과 vendor 내부
trace formatting이다. 다음 단계에서 Relax graph를 host-resident tensor execution plan으로
compile하여 layer 연산 전체를 이 primitive들로 연결하고 실제 invocation/time을 측정한다.

### 2026-08-18 — Stage V3.3: Relax small-graph end-to-end

- `RelaxVendorPlan`은 정규화된 Relax `SeqExpr`를 사전에 검증하고 17종 binding을
  static execution plan으로 변환한다. 기존 `memplan.top` 전역 배치는 사용하지 않는다.
- 지원 경로: matmul, add/sub/mul/div, sqrt/exp/negative/cos/sin/SiLU,
  last-axis sum/max, broadcast, transpose, strided-slice, concat.
- native broadcast(0x15)는 scalar/row/column source를 8192-entry 단위로 분할한다.
- transpose는 최대 64x64 arbitrary sub-tile의 strided matrix load/save로 실행한다.
- slice/concat만 host가 row-major layout을 재조합한다. 값 계산은 없으며, 나머지 Relax
  binding은 모두 제공 vendor binary를 호출한다.
- reduced Llama prefill(`S=2,D=64,H=4,KV=2,HD=16,F=128`) 결과:
  - Relax binding 135개, matmul 23개, stable softmax max 4개
  - host layout binding 19개(slice 12 + concat 7)
  - vendor invocation 131회, vendor 실행 누적 0.703초, cache된 program 31개
  - 동일 streaming-FP16-boundary NumPy reference와 max error **0.0**

이제 컴파일 경로가 small graph 수준에서 연결되었다. 다음 단계는 official tensor naming과
fused Q/K/V/O projection을 이 execution model에 연결하여 layer 하나를 실제 3B shape와
weight로 실행하는 것이다.

### 2026-08-18 — Stage V3.4: official layer 및 parallel schedule

- official checkpoint 자체를 lazy `LinearTileSource[K,N]`로 Relax parameter에 연결했다.
  safetensors mmap handle은 유지하며 필요한 tile만 BF16→FP16 변환한다.
- V3 prefill graph는 official architecture대로 fused Q/K/V projection, GQA head slice,
  RoPE/causal stable softmax, context concat, fused O projection, SwiGLU를 구성한다.
- independent reference:
  - prompt `Hello, NPU compiler!`
  - ids `[128000, 9906, 11, 452, 6459, 19979, 0]`
  - Hugging Face eager FP16 full forward 0.566초
  - reference next token `358`, decode `" I"`
  - layer hidden 29개와 last logits를 ignored build artifact로 보관
- official layer 0 vendor 결과:
  - serial: 16,693 invocation, wall 177.4초
  - HF `hidden_states[1]` 대비 max abs 0.1953, mean abs 0.0006135,
    RMSE 0.004120, cosine 0.9999815
- GEMM scheduler는 동일 A를 공유하는 여러 output-column tile을 한 G-buffer window에
  packing한다. 공식 Q projection 3072열은 1,200 invocation이다.
- 누산 의존성이 없는 column group을 4개 vendor workdir에서 병렬화했다.
  - Q projection wall 17.9초 → 5.0초
  - layer 0 wall 177.4초 → 67.8초
  - invocation 16,693 및 최종 output은 serial과 bit-exact
- `run_v3_prefill.py`는 layer별 hidden checkpoint/resume, HF metric JSONL,
  최종 logits/token artifact를 제공한다.

다음은 동일 7-token prompt로 28개 layer를 checkpoint하면서 끝까지 실행하고 final norm,
tied LM head, CPU argmax token을 reference와 비교하는 최종 장시간 단계다.

### 2026-08-18 — Stage V3.5: full prefill 1차 실행 및 RMSNorm blocker

- 28 layer + final norm + tied LM head를 vendor-only로 처음 끝까지 실행했다.
  - 500,864 vendor invocation
  - 4-worker vendor 실행시간 합 6,773.0초, wall 2,187.0초(36.5분)
  - vendor token `452` (`" N"`) vs HF token `358` (`" I"`): **불일치**
- layer 2 output까지는 HF cosine 0.9999993/mean abs 0.00133이지만 layer 3부터 drift가
  급증했다. checkpoint를 조사한 결과 BOS residual이 layer 2에서 330.75가 된 후 끝까지
  정확히 같은 값으로 고정됐다.
- 원인은 기존 RMSNorm 순서다. `x*x`가 FP16 max 65504를 먼저 넘어 `inf`가 되고,
  나중의 `1/D` scale은 overflow를 복구하지 못했다.
- 수정 lowering은 `scaled=x/sqrt(D)`, `mean=sum(scaled*scaled)`이다. full D=3072,
  outlier=330.75 회귀에서 finite이며 float reference 대비 max abs 0.000244,
  mean abs 0.00000775다.
- overflow 전의 `hidden_after_layer_02` checkpoint는 HF와 거의 일치하므로 해당 지점부터
  수정 graph로 resume한다. 실패 결과와 모든 checkpoint도 ignored build 영역에 보존했다.

### 2026-08-18 — Stage V3.6: fresh full-model prefill 성공

- RMSNorm 수정 후 layer-2 checkpoint resume run에서 token `358` 일치를 먼저 확인했다.
  이어서 현재 compiler를 embedding부터 시작하는 별도 fresh checkpoint에서 재실행했다.
- fresh 실행 범위:
  `embedding → 28 transformer layers → final RMSNorm → tied LM head[128256] → CPU argmax`
- 실행 대상 산술은 전부 제공 `0818_npu_update/a_npu/a.out`이다. 분석용
  `_poc/mysim_0818.cpp`는 호출하지 않았다. host는 tokenizer/embedding gather,
  slice/concat layout, checkpoint, 최종 argmax만 담당한다.
- fresh full 결과:
  - prompt: `Hello, NPU compiler!`
  - input ids: `[128000, 9906, 11, 452, 6459, 19979, 0]`
  - vendor/HF next token: **358**, decode **`" I"`** — argmax 일치
  - vendor invocation: **517,557**
  - 8-worker vendor 실행시간 합: 8,213.7초
  - wall: **1,665.3초 (27.8분)**
  - final normalized vs HF: max abs 0.3594, mean abs 0.01579,
    RMSE 0.03067, cosine **0.9998027**
  - last logits vs HF: max abs 0.07422, mean abs 0.009960,
    RMSE 0.01273, cosine **0.9999877**
- 마지막 HF가 직접 노출하는 raw layer checkpoint(layer 27)도 mean abs 0.01069,
  cosine 0.9999903이다.
- 재현 명령:

```bash
/home/chokwans99/anaconda3/envs/npu-tvm/bin/python \
  d_compiler/run_v3_prefill.py --workers 8 \
  --checkpoint d_compiler/build/v3_prefill_hello_fresh
```

- ignored 실행 산출물:
  - HF reference: `d_compiler/build/v3_reference_hello.npz`
  - fresh vendor result: `d_compiler/build/v3_prefill_hello_fresh/final.npz`
  - 요약: `d_compiler/build/v3_prefill_hello_fresh/result.json`
  - layer checkpoints/metrics: 같은 디렉터리의 `hidden_after_layer_*.npy`,
    `progress.jsonl`, `state.json`
- 회귀:
  - repository의 23개 test script를 직접 실행
  - 0818 ISA/C-model parity/backend, V3 model/runtime/Relax executor 통과
  - real 3B q/k/v/o/FFN weight test 통과
  - legacy decode/generation, layout/tiling/TIR, v2 comprehensive 통과
  - slow official layer test는 일반 sweep에서는 opt-in skip이며, 그보다 큰 fresh full run으로 검증
- 회귀 중 발견한 legacy A1 allocator 중복 free-offset 문제(V3-022)도 함께 수정했다.
  `test_layout.py`의 prefill/decode reuse는 byte-exact이고 G-buffer peak는
  1,144,512 → 635,968(-44.4%)이다.

### 2026-08-18 — 실행기 메모리 계약 재감사

- ELF symbol과 disassembly를 다시 확인한 결과 두 용량의 성격이 서로 달랐다.
  - `G_buffer_data_array`: 16384 bytes = 8192 FP16 정적 배열
  - `G_buffer_data_array_fp16`: 16384 bytes = 8192 FP16 정적 배열
  - `Global_buffer_file_write()` 반복 상한: `0x1fff`, 즉 출력도 항상 8192 FP16
  - program: file size만큼 `malloc`한 `Program_mem_malloc`에서 fetch하고,
    main loop도 `file_size / 4` words 전체를 실행
- 따라서 기존의 **32768-word program limit 판단은 오판**이었다. 실제 vendor에서
  33,001-word program의 word 33,000에 둔 `0xF0`이 실행되어 16,385-byte snapshot이
  생성되는 것을 확인했고, 분석용 source C-model도 동적 program vector로 수정했다.
- compiler/runtime의 `PROGRAM_CAPACITY=32768` 검사와 backend의 program overflow 오류를
  제거했다. 이 변경은 program을 크게 한 번에 내보내는 것을 허용하지만, 현재 full-model
  분할 실행의 주원인인 8192-entry G-buffer 제약을 해소하지는 않는다.
- G-buffer 쪽은 generator의 출력 개수만 늘려서는 안 된다. vendor의 input read는 file
  size 전체를 위 정적 배열 주소로 읽기 때문에 8192 초과분은 truncate가 아니라
  **out-of-bounds write**가 된다. 확장하려면 vendor 실행기의 배열·read/write loop를 함께
  동적화하거나 더 큰 용량으로 rebuild한 새 `a.out`이 필요하다.
- 종료/save 동작도 재확인했다.
  - program에 `0xF0`이 없으면 정상 process 종료 후에도 snapshot이 생성되지 않는다.
  - `0xF0` 두 개는 모두 실행되며 2개 snapshot + newline인 32,769-byte 파일을 만든다.
  - 즉 제공 PDF의 `Finish (end of program)` 표기와 달리 현재 `a.out`에서 `0xF0`은
    **save이며 halt가 아니다**. compiler는 마지막 word에 정확히 한 번 둔다.

이 재감사는 위 Stage V3.1/V3.2에 기록된 “8192/32768 고정” 설명 중 program 부분을
정정한다. 517,557회 invocation은 32K program limit 때문이 아니라, 8192-entry G-buffer에
weight/activation/누산 tile을 반복 반입해야 했기 때문이다.

### 2026-08-18 — Stage V3.7: vendor-compatible source C-model 용량 확장

- 이후 full-model/decode 실행 대상은 분석 가능한 `_poc/mysim_0818.cpp`로 전환한다.
  capacity 외의 실행 의미는 vendor와 동일하게 유지한다.
  - Reduce Max `0x19`의 zero-seed 오류 유지
  - vendor GELU `x*sigmoid(2x)` 유지
  - `0xF0` snapshot 후 실행 계속, 무조건 종료 save 없음
  - load/save 시 FP16 반올림 및 PE float32 계산 유지
- G-buffer는 입력 FP16 entry 수에 맞춰 동적 할당하고, instruction이 더 높은 목적 주소에
  쓰면 해당 주소까지 zero-initialized 확장한다. vendor 호환 입력의 최소 snapshot 크기는
  기존과 같은 8192 entry다.
- 대용량 snapshot은 1M-entry chunk별로 동일한 little-endian FP16 byte를 기록한다.
  `NPU0818_QUIET`/`--quiet`는 trace formatting만 억제하며 계산과 파일 결과에는 관여하지 않는다.
- `source_runtime_0818.py`가 source 변경 시 ignored build 영역에 C++ binary를 자동 rebuild하고,
  동적 크기의 G-buffer input/output을 실행한다.
- 검증:
  - 64개 제공 program 및 targeted vendor/source snapshot/trace parity 전부 통과
  - 33,001-word program 및 non-implicit-save parity 통과
  - G-buffer address 9000 load → address 10000 save 결과 `3.25 + 2 = 5.25` 통과

이 단계는 V3-023의 vendor binary 자체를 변경하지 않는다. vendor oracle은 계속 8192
경계까지의 parity 기준으로 보존하고, 확장 실행은 동일 source semantics 위에서 수행한다.

### 2026-08-18 — Stage V3.8: expanded row-major backend와 decode graph

- `backend="source-0818"`은 vendor와 같은 ver.08 row-major codegen을 사용하되 8192-entry
  compile guard 없이 동적 source runtime으로 실행한다. 입력 G-buffer와 snapshot은 FP16로
  유지하여 대형 graph의 host memory 복제를 줄였다.
- 8K vendor 경계 안에서는 scalar broadcast 주소가 항상 16-bit여서 backend가 opcode
  `0x15` low half만 emit해도 우연히 동작했다. expanded plan에서는 RMSNorm scalar가
  address 65535 밖에 배치되어 잘못된 값을 읽었다(V3-025).
  - `v_broadcast_addr()`가 32-bit address low/high를 모두 emit
  - high가 0이어도 항상 emit하여 이전 broadcast의 high state를 제거
  - address 70000의 `6.5`를 address 66000에 3-lane broadcast하는 회귀 통과
- official fused graph 확장:
  - prefill 선택 output: `[hidden, K0, V0, ..., K7, V7]`
  - decode K/V projection: fused official `k_proj`/`v_proj`, 현재 position RoPE
  - decode layer: exact populated context 길이의 Kt/V cache, fused Q/O, GQA,
    stable softmax, SwiGLU
  - host는 산술 없이 현재 token K/V를 cache 끝에 append
- 작은 GQA graph에서 단일-program source와 fixed-window streaming oracle 비교를 통과했다.
  K-tile 사이 snapshot 반올림이 사라져 decode residual은 최대 FP16 1 ULP 차이가 허용된다.
- official 7-token/3B 정적 크기:
  - prefill layer: 497,206 words, 101,084,262 FP16 entries
  - decode K/V: 30,449 words, 6,316,741 FP16 entries
  - LM head: 1,845,685 words, 394,133,760 FP16 entries
  - 모두 uint32 flat G-buffer address 범위 이내
- official layer 0 source 실행:
  - wall 8.20초
  - HF hidden 대비 max abs 0.1953, mean abs 0.000591, RMSE 0.003991,
    cosine 0.9999837
  - 기존 vendor streaming 결과와 동등한 정확도이며 finite

## 5. 1차 목표 판정

| 완료 조건 | 결과 |
|---|---|
| vendor 0818 executable만 compute target으로 사용 | PASS |
| official Llama 3.2 3B config/weight, 28 layers full prefill | PASS |
| layer/reference 수치 검증 | PASS |
| full vocabulary logits 생성 | PASS |
| HF와 argmax token 일치 및 tokenizer decode | PASS — `358`, `" I"` |
| issue 기록, 재현 script, 단계별 commit/push | PASS |

따라서 compiler-v3의 **Llama 3.2 3B full-model prefill 및 정상 next-token 출력** 1차 목표는
완료했다. V3-004 GELU semantic과 V3-007 vendor example coverage는 Llama의 SiLU
prefill 성공 경로를 막지 않는 일반 backend 후속 과제다. V3-023 G-buffer 확대는 vendor
실행기 rebuild가 필요한 별도 blocker다.
