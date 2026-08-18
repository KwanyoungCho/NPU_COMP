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
| V3-002 | MITIGATED | BLOCKER | program memory 32768 words 고정, loop/branch/indirect addressing 없음 | Relax execution plan이 bounded program을 여러 vendor invocation으로 순차·병렬 dispatch |
| V3-003 | MITIGATED | HIGH | native Reduce Max `0x19`가 accumulator를 0으로 시작하여 all-negative row를 0으로 반환 | compiler는 첫 실제 열에서 시작하는 vector-max fold 사용. source C-model은 vendor bug 그대로 재현 |
| V3-004 | OPEN | HIGH | vendor GELU가 표준 GELU가 아니라 `x*sigmoid(2x)` | native GELU 사용 시 모델 정확도 영향 측정. 필요하면 표준식을 primitive 조합으로 lowering |
| V3-005 | MITIGATED | MEDIUM | `0xF0`은 HALT가 아니며 snapshot 후 계속 실행 | generated program의 마지막 유효 word로 `0xF0` 하나만 배치 |
| V3-006 | MITIGATED | MEDIUM | vector save lane 수가 연산 결과 크기가 아니라 현재 `vlen`으로 결정 | scalar reduction 직후 `vlen=1` 명시 |
| V3-007 | OPEN | MEDIUM | 제공 64개 예제 대부분이 0710 encoding을 유지하여 새 기능 coverage가 부족 | compiler-owned targeted vendor parity 프로그램 유지·확장 |
| V3-008 | RESOLVED | BLOCKER | full checkpoint와 tokenizer가 로컬에 없어 실제 token 검증 불가 | official revision `13afe512...`의 2개 shard와 tokenizer를 ignored build 영역에 확보. 총 weight 6,425,529,048 bytes |
| V3-009 | MITIGATED | HIGH | official weight는 BF16이지만 vendor G-buffer 입력은 FP16 | safetensors tile에서 FP16 변환. official layer-0 HF FP16 대비 mean abs 0.000613/cosine 0.999982 확인, full token 영향은 최종 측정 |
| V3-010 | MITIGATED | HIGH | tied embedding/lm_head가 128256x3072이며 전체 복사 시 약 788MB | embedding row와 LM-head `[K,V]` tile만 safetensors에서 slice하여 로드 |
| V3-011 | RESOLVED | BLOCKER | A/B/C 64x64 세 tile은 12288 FP16이라 한 invocation에 들어가지 않음 | tile별 `m*k + k*n + m*n <= 8192`를 만족하도록 K를 동적으로 축소. 64x64 출력은 K=32 |
| V3-012 | MITIGATED | HIGH | vendor invocation 사이에는 PE MAC 상태가 사라짐 | running C를 snapshot에서 host로 carry하고 다음 invocation 시작 시 PE output으로 load한 후 MAC bit 27 실행 |
| V3-013 | MITIGATED | HIGH | vendor stdout trace가 큰 실행에서 pipe 메모리·I/O를 폭증시킴 | parity 때만 capture하고 production streaming은 `/dev/null`. vendor 내부 formatting 비용은 잔존 |
| V3-014 | RESOLVED | HIGH | SiLU는 독립 unary opcode가 아니라 matrix ALU activation field라 기본 VECTOR mode 사용 시 미초기화 SRC2를 참조 | `matrix add immediate 0 + ACT_SILU`로 identity 연산 후 activation 수행 |
| V3-015 | RESOLVED | BLOCKER | 기존 Relax backend는 graph 전체 tensor를 한 G-buffer에 배치하므로 real layer가 8192 capacity를 근본적으로 초과 | Relax binding을 host-resident value와 bounded vendor kernel 호출로 compile하는 `RelaxVendorPlan` 구현 |
| V3-016 | MITIGATED | MEDIUM | indirect addressing이 없어 slice/concat을 vendor 안에서 동적 주소로 연결할 수 없음 | slice/concat은 산술 없는 host layout operation, transpose/broadcast와 모든 model arithmetic은 vendor opcode로 실행 |
| V3-017 | RESOLVED | HIGH | per-head weight API는 official fused q/k/v/o tensor와 직접 대응하지 않아 full checkpoint mapping 오류 위험 | fused-projection Relax graph 추가: Q/K/V 1회 projection, head slice/GQA, context concat 후 official O projection |
| V3-018 | MITIGATED | HIGH | vendor executable을 weight window마다 새 process로 실행해 layer 0 serial wall 177초 | 독립 output-column group만 4개 workdir에서 병렬 실행. serial과 bit-exact, layer 0 wall 67.8초 |
| V3-019 | OPEN | HIGH | streaming FP16 경계가 HF eager FP16보다 많아 layer별 수치 drift 누적 가능 | 독립 HF hidden/logits reference 저장, 매 layer max/mean/RMSE/cosine 및 최종 argmax 추적 |

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
