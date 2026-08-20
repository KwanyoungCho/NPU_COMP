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
   *(2026-08-18 Stage V3.7에서 갱신: vendor binary의 8192-entry G-buffer로는
   full model이 물리적으로 불가하여, 이후 full-model 실행은 capacity만 확장한
   parity source C-model을 사용하고 vendor binary는 8192 범위의 oracle로
   보존하는 것으로 사용자와 합의함.)*
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
| V3-004 | RESOLVED | HIGH | vendor GELU가 표준 GELU가 아니라 `x*sigmoid(2x)` | (2026-08-19, G4-001) `legalize.gelu_tanh`가 표준 tanh-GELU를 mul/add/neg/exp/div 조합으로 lowering. vendor/source/FP16-emulation bit-exact, Gemma 4 E2B full model에서 실사용 검증. native GELU opcode 의미는 vendor 그대로 유지 |
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
| V3-021 | MITIGATED | LOW | TVM conda 환경에 `pytest` 모듈이 없어 collection 명령 사용 불가 | repository의 `test_*.py` 자체 entrypoint를 직접 실행(작성 당시 26개, 2026-08-19 현재 35개). 환경 package 설치 없이 전체 범위 검증 |
| V3-022 | RESOLVED | MEDIUM | RMS binding 추가 후 legacy A1 reuse에서 동일 physical offset이 free-list에 중복 삽입되어 live tensor 충돌 | free-list를 offset set으로 변경하고 bump 조건 수정. prefill/decode byte-exact, v2 comprehensive 회귀 통과 |
| V3-023 | MITIGATED | BLOCKER | G-buffer generator만 8192보다 크게 만들면 vendor `a.out`의 16384-byte 정적 배열을 넘겨 BSS를 덮어씀 | vendor runtime은 계속 8192 초과를 거부하고, full-model은 parity-tested 동적 source C-model target으로 분리 |
| V3-024 | RESOLVED | HIGH | source C-model의 G-buffer를 확장하면서 vendor arithmetic/quirk parity가 달라질 위험 | 기본 8192 parity는 64개 program+targeted quirk로 유지하고, compiler-owned source runtime만 동적 flat buffer와 quiet mode 사용 |
| V3-025 | RESOLVED | BLOCKER | 8K 안에서는 드러나지 않던 scalar broadcast 주소 상위 16-bit 누락으로 확장 source RMSNorm이 거의 0 출력 | opcode `0x15` low/high를 매번 모두 emit해 stale high state도 제거. address 70000 broadcast 및 official layer 0 검증 |
| V3-026 | RESOLVED | HIGH | official V3 prefill output에 K/V가 없어 decode cache를 seed할 수 없음 | fused prefill이 `[hidden,K0,V0,...]`을 반환하는 선택 경로와 exact-context fused decode/KV graph 구현 |
| V3-027 | RESOLVED | BLOCKER | LM head 논리 stride 128256이 matrix descriptor의 16-bit `main_cols`를 넘어 source logits 오염 | ISA/C-model field는 바꾸지 않고 RHS를 연속 `[K,64]` column panel로 pack하는 단일-program GEMM lowering 구현 |
| V3-028 | RESOLVED | HIGH | 첫 expanded full prefill에서 transformer hidden은 정상이지만 잘못 lowering된 LM head가 token 7272 출력 | panel LM 재실행 token 358, HF logits cosine 0.999993. 잘못된 state token/logits만 교체 후 cache는 그대로 decode에 사용 |
| V3-029 | RESOLVED | BLOCKER | official 3B KV-cache autoregressive decode가 V3에 연결되지 않음 | prompt cache seed 후 position 7/8 exact-context decode, cache append, final norm/LM head 수행. HF와 `[358,2846,4560]` 일치 |

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
  - prefill layer: 497,563 words, 101,084,262 FP16 entries
  - decode K/V: 30,451 words, 6,316,741 FP16 entries
  - LM head: 1,845,685 words, 394,133,760 FP16 entries
  - 모두 uint32 flat G-buffer address 범위 이내
- official layer 0 source 실행:
  - wall 8.20초
  - HF hidden 대비 max abs 0.1953, mean abs 0.000591, RMSE 0.003991,
    cosine 0.9999837
  - 기존 vendor streaming 결과와 동등한 정확도이며 finite

### 2026-08-18 — Stage V3.9: 16-bit stride 보존형 wide LM head

- expanded G-buffer/program이 있어도 instruction의 matrix row/column field는 16-bit다.
  transformer의 최대 stride 8192는 합법이지만 LM head의 논리 RHS `[3072,128256]`은
  `main_cols=128256`을 직접 encode할 수 없다.
- 최초 full source transformer hidden은 기존 fresh vendor hidden과 max abs 0.046875,
  mean abs 0.003179, cosine 0.999994로 정상이었다. 반면 row-major wide LM head는 stride
  truncation 때문에 token 7272를 출력했다. 따라서 transformer 재실행이 아니라 LM lowering
  문제로 분리했다.
- vendor/source ISA field를 넓히는 비호환 변경은 하지 않았다. tied embedding `[V,K]`를
  64개 vocabulary column 단위의 연속 `[K,64]` panel로 변환하고, 각 panel은 합법적인
  local stride 64로 기술한다. 모든 2004 panel과 K tile은 하나의 program에서 실행한다.
- `PackedRhsGemm` 검증:
  - logical RHS stride 65536 경계 초과 GEMM byte-exact
  - K=70 multi-K-tile panel GEMM byte-exact
  - official LM: 1,845,685 words, 394,133,760 FP16 entries
- corrected official prefill LM 결과:
  - token 358 (`" I"`)로 HF 일치
  - 기존 fresh vendor logits 대비 max abs 0.046875, mean abs 0.005183,
    cosine 0.9999969
  - 독립 HF generation logits 대비 max abs 0.06287, mean abs 0.007639,
    RMSE 0.009760, cosine 0.9999929
  - panel LM wall 31.21초

### 2026-08-18 — Stage V3.10: official 3B autoregressive decode 완료

- 독립 Hugging Face CPU greedy reference를 같은 local revision에서 생성했다.
  - prompt ids: `[128000, 9906, 11, 452, 6459, 19979, 0]`
  - generated ids: `[358, 2846, 4560]`
  - decoded tokens: `[" I", "'m", " trying"]`
  - decoded text: `" I'm trying"`
- source prefill은 각 layer output과 함께 roped K/V를 반환하여 28 layer cache를 seed한다.
  decode token마다 다음을 수행한다.
  1. token embedding host gather
  2. fused `k_proj`/`v_proj` source program과 position RoPE
  3. host가 K column/V row를 해당 layer cache 끝에 append
  4. populated exact context의 fused Q/GQA attention/O/SwiGLU source program
  5. 28 layers 후 final RMSNorm + panel LM head + host argmax
- 실제 결과:
  - prefill: token 358, cache length 7
  - decode position 7: token 2846, cache length 8
  - decode position 8: token 4560, cache length 9
  - 최종 sequence와 text가 HF와 전부 일치
  - final decode logits vs HF: max abs 0.07422, mean abs 0.01002,
    RMSE 0.01266, cosine 0.9999881, argmax 4560 일치
- 실행 규모:
  - source prefill: 30 invocation(28 layers + norm + LM)
  - 2 decode steps: 116 invocation(각 step 28×KV/decode + norm + LM)
  - 합계 146 invocation, source 실행 누적 약 774.5초(12.9분)
  - fixed vendor prefill의 517,557 invocation과 달리 layer/단계 단위 큰 program 사용
  - decode context 8: 453,349 words / 94,444,391 FP16 entries
  - decode context 9: 453,757 words / 94,446,443 FP16 entries
- ignored 재현 산출물:
  - HF reference: `d_compiler/build/v3_reference_generate_hello_3.npz`
  - source state/result: `d_compiler/build/v3_source_generate_hello/`
- 재현 명령:

```bash
/home/chokwans99/anaconda3/envs/npu-tvm/bin/python \
  d_compiler/make_v3_generation_reference.py --tokens 3

/home/chokwans99/anaconda3/envs/npu-tvm/bin/python \
  d_compiler/run_v3_source_generate.py --tokens 3 \
  --reference d_compiler/build/v3_reference_generate_hello_3.npz
```

- 전체 회귀:
  - repository의 26개 `test_*.py` entrypoint 통과
  - source dynamic G-buffer/full-address broadcast/panel GEMM/fused decode 신규 회귀 포함
  - legacy prefill→decode KV-cache generation, 0710/0818 ISA, layout/TIR/V2/V3 통과
  - slow official vendor layer는 opt-in skip이며, 이번 source 28-layer+2 decode 실실행으로
    더 큰 official 범위를 검증

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
완료했다. 이어서 parity source C-model의 동적 G-buffer와 wide-stride panel lowering을
사용해 official KV-cache decode 2 step 및 3-token greedy sequence까지 완료했다.
V3-004 GELU semantic과 V3-007 vendor example coverage는 Llama의 SiLU 경로를 막지 않는
일반 backend 후속 과제이며, vendor binary 자체의 8192 제한은 유지되지만 source target으로
full prefill/decode blocker를 해소했다.

## 6. source full-model + decode 목표 판정

| 완료 조건 | 결과 |
|---|---|
| vendor arithmetic/quirk parity 유지 | PASS — 64 programs + targeted parity |
| 동적 G-buffer 및 가변 program | PASS |
| official 28-layer source prefill | PASS |
| prefill K/V cache seed | PASS — 28 layers × 8 KV heads |
| autoregressive cache append/decode | PASS — context 7→8→9 |
| full-vocabulary LM head | PASS — 16-bit field 보존 panel lowering |
| HF greedy token sequence | PASS — `[358, 2846, 4560]` |
| 전체 regression | PASS — 26 scripts |

따라서 이번 목표인 **우리 source C-model 기반 official Llama 3.2 3B prefill 및 decode**는
완료했다. CPU에 남은 동작은 tokenizer/embedding gather, 동적 KV-cache append, argmax이며,
모델 산술은 ver.08 source C-model program으로 수행한다.

---

## 7. 최종 ISA 리뷰 — HW 설계 이관 (2026-08-19)

이 절은 NPU 설계팀과의 최종 ISA 미팅 및 실제 하드웨어 설계 착수를 위해,
지금까지의 **전체 검증 스코프에서 확정된 명령어별 사용 현황·오류·개선 요구**를
집대성한 것이다. 근거는 전부 실측이다: ver.08 encoder/decoder round-trip,
vendor `a.out` 64개 프로그램 + targeted parity, 그리고 세 개의 official model
full 실행이다.

### 7.1 검증 스코프 (이 리뷰의 근거)

| Model | 구조 특성 | 결과 |
|---|---|---|
| Llama 3.2 3B (28L) | GQA 24/8, SwiGLU, RoPE(llama3 scaling) | HF greedy `[358,2846,4560]` 일치, logits cosine 0.9999881 |
| Gemma 4 E2B (35L) | sliding/global 혼합, KV 공유 20층, QK/V-Norm, PLE, double-wide MLP, tanh-GELU, partial RoPE | HF greedy `[108,236777,236789]` 일치, cosine 0.9999975 |
| Qwen3-4B (36L) | GQA 32/8, QK-Norm, q폭(4096)≠hidden(2560), SwiGLU | HF greedy `[358,1184,311]` 일치, cosine 0.9999915 |

모든 모델 산술(matmul, 4종 norm, softmax, RoPE, SiLU/GELU, PLE, 활성화)이
ver.08 프로그램으로 수행되었고, host는 gather/layout/argmax만 담당했다.
따라서 아래 명령어 평가는 "LLM inference 전 범위를 실제로 돌려본" 결과다.

실행체 구분을 명확히 하면:

- **명령어 semantics·오류·계약(§7.2, §7.3)은 전부 제공 vendor `a.out`에서
  직접 확정**한 것이다. source C-model은 그 parity 재현본(64개 프로그램 +
  targeted 프로그램 snapshot 일치)이며, capacity 외의 실행 의미는 동일하다.
- 위 표의 full-model 3종 실행은 G-buffer 8192 한계 때문에 capacity 확장
  source C-model에서 수행했다(§1 주석의 합의). 단 **Llama full prefill은
  Stage V3.6에서 vendor `a.out`만으로도 완주**했고(517,557 invocations,
  token 일치), tanh-GELU lowering과 Gemma/Qwen layer는 vendor에서
  kernel/layer 규모로 완전 일치를 확인했다.
- §7.2의 word 통계는 ver.08 인코딩 자체의 통계로, 실행체와 무관하다.

### 7.2 명령어 전수 조사

프로그램 word 사용 통계는 세 model family의 대표 layer/decode 프로그램 6개
(합계 20,114 words)에서 opcode를 decode해 집계했다. full-scale 프로그램
(Llama prefill layer 497,563 words 등)도 같은 구성비를 가진다.

**A. 사용 중 — LLM 실행의 필수 명령** (word 점유율 포함)

| Opcode | 명령 | 용도 | 표본 내 word 수 (비중) |
|---|---|---|---:|
| `0x80` | 주소 설정 (lo/hi × MAIN/PARTIAL) | 모든 접근의 주소 상태 | 9,652 (48.0%) — 주소 1회 설정 = word 2개(lo+hi) |
| `0x90` | load (strided 포함) | tile/vector 반입, transpose | 2,382 (11.8%) |
| `0x82` | vlen 설정 | vector 길이 상태 | 2,063 (10.3%) |
| `0x98` | save (strided 포함) | 결과 기록 | 2,020 (10.0%) |
| `0x17` | v_copy | 값 이동(누산기 반입 등) | 912 (4.5%) |
| `0x88`/`0x89` | rows/cols 설정 | matrix descriptor | 1,738 (8.6%) |
| `0x15` | broadcast (scalar/row/col) | norm 분모, scale 살포 | 460 (2.3%) |
| `0x0A` | v_mul | elementwise 곱 | 249 |
| `0x14` | v_reduce_sum | norm/softmax 행 합 | 160 |
| `0x42` | m_mul + MAC(bit27) | **matmul/K-tile 누산 — 연산의 심장** | 93 |
| `0x01` | v_add | residual/eps 등 | 93 |
| `0x12` (max) | v_max | softmax 행 최대 fold (0x19 대체) | 74 |
| `0x0B` | v_div | softmax/norm/sigmoid | 60 |
| `0x0E` | v_sqrt | RMSNorm | 36 |
| `0x16` | v_sign_inv | RoPE rotate-half, GELU | 33 |
| `0x40` | m_add (+ACT_SILU) | SiLU 활성화 경로 | 27 |
| `0x0F` | v_exp | softmax, GELU/sigmoid lowering | 24 |
| `0x02` | v_sub | stable softmax max 차감 | 20 |
| `0x18` | v_cos/sin | on-device RoPE | 12 |
| `0xF0` | snapshot | 프로그램 끝 결과 저장 | 6 |

핵심 구성비: **주소/모양/길이 설정 66.9% + load/save 21.9% + 데이터 이동(v_copy)
4.5% = 93.3%가 제어/이동이고, 실제 산술은 6.7%**다. §7.5의 제안 1~3이 여기서
나온다.

**B. 구현되어 있으나 사용하지 않는 명령**

| Opcode | 명령 | 미사용 이유 |
|---|---|---|
| `0x19` | native reduce max | **버그(7.3-①)로 의도적 회피** — v_max fold로 대체 |
| `0x40..43`의 ACT_GELU(11) | native GELU | **수식이 표준과 다름(7.3-②)** — 정확성 경로에서 회피, parity test로만 검증 |
| `0x0D`/`0x43` | v_move/m_move | v_copy/m_add+0으로 충분 |
| `0x41` | m_sub | 필요 시 v_sub 사용 |
| `0x0C` | v_muladd | 누산은 m_mul MAC로; elementwise FMA 수요는 현재 없음 |
| `0x11` | v_compare | LLM FP 경로에 불필요 |
| `0x08`/`0x09` | logical/shift | 정수 연산 불필요 |
| `0x13` | int/float 변환 | 불필요 |
| `0x00` | nop | 사용 안 함 |
| activation 예약값 `01` | — | 실행기에서 GELU와 동일 동작(정의 필요, 7.3-⑦) |
| `0xFF` | (v07 HALT) | ver.08에서 HALT 아님 — 미사용 |

미사용 명령을 HW에서 제거할지는 설계팀 판단이나, **0x19와 native GELU는
"제거"가 아니라 "수정"을 권고**한다(7.5).

**C. 존재하지 않아 SW로 우회 중인 기능** — 7.5의 개선 제안으로 연결

- indirect addressing (KV cache append를 host가 수행)
- loop/repeat (모든 프로그램 완전 unroll: layer당 18만~51만 words, LM head 180만~190만 words)
- 32-bit 폭의 rows/cols descriptor (LM head stride 128256/151936/262144 표현 불가 → panel 재배열)
- invocation 간 상태 유지 (weight/cache 상주 불가 → 매 실행 재공급)

### 7.3 오류·불일치 상세 (재현·우회·권고)

**① `0x19` Reduce Max — accumulator 0 초기화 버그 [V3-003, HW 수정 필수]**
- 현상: 전부 음수인 vector의 max를 0으로 반환.
- 우회: 첫 실제 열을 accumulator로 삼는 column-strided `v_max` fold. softmax
  행마다 (rows/64 × cols)회의 strided load+max가 추가됨.
- 권고: accumulator를 **첫 원소 또는 -inf로 초기화**. 수정되면 softmax의
  max 단계가 행당 1개 op로 줄어든다.

**② native GELU = `x·sigmoid(2x)` — 표준식 아님 [V3-004→G4-001, HW 결정 필요]**
- 현상: 표준 `gelu_pytorch_tanh`(=`0.5x(1+tanh(√(2/π)(x+0.044715x³)))`)와 다른
  근사식. Gemma 등 GELU 모델의 정확성 요건을 만족하지 못함.
- 우회: 8-op primitive 조합으로 lowering (mul/add/neg/exp/add/div/mul).
  vendor에서 bit-exact 검증됨. 비용: 활성화 1회가 8배 word.
- 권고: **표준 tanh-GELU를 native로 제공**(또는 activation field에 mode 추가:
  현재식/표준식). SiLU는 현재 그대로 정확하고 유용함.

**③ `0xF0`/종료 semantics — 문서와 실행 불일치 [V3-002/005, 문서·HW 정합 필요]**
- PDF는 `Finish (end of program)`로 기재하나 실제 실행기는 **snapshot 저장 후
  계속 실행**하고, 프로그램에 0xF0이 없으면 **정상 종료에도 저장하지 않음**.
  복수 0xF0은 같은 파일에 순서대로 append.
- program memory도 PDF의 32768-word 고정과 달리 file 크기만큼 동적 할당.
- 권고: HW 스펙 확정 시 halt/save/flush를 명시적으로 분리 정의하고 문서를
  실행기와 일치시킬 것.

**④ 주소 상태 머신의 hazard [V3-025 실사고]**
- 32-bit 주소를 16-bit half 두 번으로 설정하는 상태식 설계라, high half를
  생략하면 **이전 명령의 stale high가 그대로 사용**됨. 8K 범위에서는 항상 0이라
  숨어 있다가 확장 주소(>65535)에서 오작동(RMSNorm 출력 0).
- 추가: `0x15`(broadcast)는 **주소 half를 설정하는 동시에 실행**되므로, 32-bit
  주소를 만들려면 저-half 설정 시 무의미한 broadcast가 한 번 더 실행됨.
- 권고: (a) 주소를 한 word로 설정하는 형식(또는 24-bit immediate), (b) 설정과
  실행의 분리, (c) 최소한 상태 레지스터의 명세화(초기값/유지 규칙).

**⑤ vector save의 lane 수가 결과 크기가 아니라 현재 `vlen` [V3-006]**
- reduce 후 scalar 저장 시 `vlen=1` 재설정을 잊으면 쓰레기 lane까지 기록.
- 권고: save가 직전 연산의 결과 길이를 따르게 하거나, 최소한 스펙에 명시.

**⑥ matrix descriptor rows/cols 16-bit [V3-027 실사고]**
- LM head 논리 RHS `[K, vocab]`의 stride(128256/151936/262144)가 표현 불가.
  최초 구현이 조용히 truncate되어 오답 token(7272)을 냈다 — **범위 초과가
  오류가 아니라 silent wrap**인 점이 특히 위험.
- 우회: `[K,64]` panel 재배열 GEMM(프로그램 180만~190만 words, host 재배열
  788~805MB).
- 권고: rows/cols를 **24/32-bit로 확장**하거나, 범위 초과 시 오류를 내도록.
  descriptor 폭 확장은 LLM vocab 규모에서 필수적이다.

**⑦ activation field 예약값 `01`이 GELU로 동작**
- 예약값의 동작을 정의하거나 거부해야 함. (실행기 관찰로 발견)

**⑧ FP16 저장 경계로 인한 overflow class [V3-020 실사고, G4-005]**
- PE 내부는 float32이나 모든 중간 tensor가 FP16으로 저장되므로,
  `x²`(RMSNorm) 같은 중간값이 65504를 넘으면 즉시 inf. Llama BOS residual
  330.75에서 실제 발생(token 오답). "제곱 전 1/√D 스케일링" lowering으로 해결.
- exp의 overflow → inf → `1/inf=0` 전파는 IEEE와 동일함을 실측 확인했고,
  tanh-GELU lowering의 포화가 이에 **의존**한다.
- 권고: (a) IEEE inf/NaN 전파 semantics를 스펙에 보증으로 명시, (b) 가능하면
  선택적 FP32 저장 모드(또는 norm 계열 fused op)를 검토. 필수는 아님 —
  현재 lowering 순서로 세 모델 모두 정확성을 달성했다.

### 7.4 vendor C-model 실행기 이슈 (ISA 외적, HW/차기 C-model 반영 권고)

| # | 이슈 | 상세 및 권고 |
|---|---|---|
| 1 | **G-buffer 8192 고정 + 경계 검사 없음** [V3-001/023] | 8192 entry 초과 입력 파일이 truncate가 아니라 **정적 배열 밖 BSS를 덮어씀**. 3B 모델 한 layer working set(≈1억 entry)도 수용 불가 → full model은 확장 source C-model로만 실행 가능했음. HW: 실용 용량 산정 필수(layer 단위 실행 기준 최소 수억 FP16) + 모든 입출력 경로에 경계 검사 |
| 2 | invocation 간 상태 소멸 | 프로세스 단위 실행이라 weight/KV cache 상주 불가. 매 invocation마다 layer weight(수십 MB)를 입력 파일로 재공급 — 시뮬레이션 wall time의 지배 요인. HW: weight/cache 상주 메모리 계층. C-model: state save/restore 옵션 |
| 3 | trace 출력 항상 on [V3-013] | 대형 실행에서 stdout formatting이 pipe/I/O 폭증 유발. source 재현본에 `NPU0818_QUIET` 추가로 해결 — vendor 배포본에도 quiet 스위치 권고 |
| 4 | 고정 파일명 I/O (`G_buffer_data.bin` 등) | 병렬 실행 시 workdir 분리 필요 [V3-018]. 인자화 권고 |
| 5 | indirect addressing 부재 [V3-016] | KV cache append(위치가 token마다 변동)를 host가 수행. HW: base+offset register 간접 주소 또는 append 전용 기제 — decode의 host 개입을 완전 제거 가능 |
| 6 | loop/branch 부재 | 완전 unroll로 layer당 18만~51만 words, decode는 **context 길이마다 재컴파일**(예: ctx8/ctx9 별도 프로그램 45만 words씩). HW: repeat count + 주소 auto-increment만 있어도 프로그램 크기 수백 배 압축 및 context-독립 decode 가능 |
| 7 | 주소 설정 오버헤드 | §7.2 실측: 전체 word의 48%가 `0x80`(주소 설정, half 2회씩), 제어/이동 합계 93%. HW: descriptor 자동 증가, tile loop, 단일-word 주소 설정으로 program 대역폭 대폭 절감 |
| 8 | PE 64×64 + float32 내부 + MAC(bit27) | **문제 없음 — 유지 권고.** 세 모델 전부 이 계약으로 정확성 달성. K-tile 간 PE 누산 유지가 FP16 경계 반올림을 줄여 single-program 방식의 정확도 이점을 만들었음 |
| 9 | MAIN/PARTIAL sub-tile 직접 접근 | **매우 유용 — 유지 권고.** row-major 원본에서 임의 sub-tile을 직접 load/save하게 되어 0710의 tile-blocked 재배열/gather/scatter가 전부 제거됨 |
| 10 | on-device cos/sin (`0x18`) | **유용 — 유지.** RoPE 테이블 상주 없이 position 입력만으로 decode 가능해짐 |

### 7.5 HW 개선 제안 우선순위

**[필수 — 이대로면 full-model 실사용 불가]**
1. G-buffer 실용 용량 확보(+경계 검사). 기준: 대상 모델 한 layer의
   weight+activation working set (3~4B급에서 FP16 수억 entry).
2. rows/cols descriptor 16-bit → 24/32-bit (LLM vocab stride).
3. weight/cache 상주(invocation 간 상태 유지)를 전제한 실행 모델.

**[강력 권장 — SW 우회 비용이 큼]**
4. Reduce Max(0x19) accumulator 초기화 수정.
5. 표준 tanh-GELU 제공(또는 activation mode bit).
6. loop/repeat + 주소 auto-increment: 프로그램 크기(현재 layer당 수십만 word,
   제어 word 93%)와 context별 재컴파일 제거.
7. indirect addressing(base+offset): KV append/chunk binding의 host 개입 제거.

**[정리·명세화 — 저비용]**
8. 0xF0/종료/HALT semantics 문서-실행 일치, 0xFF 처리 정의.
9. 주소 half 상태 머신 명세화(가능하면 단일-word 설정) + 0x15의 설정/실행 분리.
10. vector save lane 규칙 명세화, activation 예약값 01 정의.
11. IEEE inf/NaN 전파 보증 명시(현재 lowering이 의존).
12. 미사용 opcode(0x08/09/0C/0D/11/13/41/43) 존치 여부 결정 — LLM 용도로는
    제거해도 무방하나, 0x0C(muladd)는 elementwise FMA로 활용 여지 있음.

### 7.6 통합 이슈 색인

- `V3-001 ~ V3-029`: 본 문서 §3 (V3-004는 2026-08-19 RESOLVED로 갱신).
- `G4-001 ~ G4-008`: `report/report_gemma4.md` §1 — 표준 GELU lowering(G4-001),
  double-wide MLP semantics(G4-002), weight 없는 V-Norm의 keyset 부재(G4-004),
  FP16 range 감시(G4-005, 유일한 OPEN), proportional RoPE 확정(G4-006),
  scale 사슬 검증(G4-007), KV 공유 decode 순서(G4-008).
- Qwen3: 신규 ISA 이슈 없음. 발견 사항은 HF `output_hidden_states` 마지막
  원소가 final-norm 적용값이라는 reference 해석 주의뿐
  (`report/report_qwen3.md` §2).
- 전체 회귀: 35개 test entrypoint, 세 모델 golden 전부 PASS 상태에서 본
  리뷰를 작성함.
