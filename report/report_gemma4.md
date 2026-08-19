# Gemma 4 E2B 지원 진행 보고서 (compiler gemma4-e2b branch)

> 시작일: 2026-08-19
>
> 기준 브랜치: `compiler-v3` (`2fb81b0`) → 개발 브랜치 `gemma4-e2b`
>
> 목표: **동일 0818 compiler core에서 Gemma 4 E2B text-only prefill/decode를 지원한다.**
> Llama 3.2 3B source prefill/decode 결과(`[358, 2846, 4560]`, `" I'm trying"`)는
> 모든 refactor 단계의 필수 회귀 gate다.

배경 분석과 단계 계획은 `report/SESSION_HANDOFF_0819.md` §13~§16을 권위 문서로 사용한다.
issue는 `G4-###` 식별자로 기록한다.

---

## 1. Issue tracker

| ID | 상태 | 심각도 | 내용 | 현재 대응 |
|---|---|---:|---|---|
| G4-001 | OPEN | HIGH | Gemma `gelu_pytorch_tanh`는 vendor native GELU `x*sigmoid(2x)`와 다름 (V3-004 승계) | correctness mode에서 표준 tanh-GELU를 primitive sequence로 lowering 예정. C-model은 변경하지 않음 |
| G4-002 | OPEN | MEDIUM | `use_double_wide_mlp: true`의 정확한 semantics 미확정 | Stage G4.2에서 official 구현/checkpoint keyset으로 확정 후 spec/graph에 반영 |

---

## 2. 진행 로그

### 2026-08-19 — Stage G4.0: branch 및 기준선

- `compiler-v3` tip `2fb81b0`(handoff 문서 commit, 코드는 `7c91647`과 동일)에서
  `gemma4-e2b` branch를 생성했다.
- 기준선 회귀: 핵심 0818 회귀 6종(ISA/C-model parity/backend/source runtime/
  panel GEMM/V3 source decode) 전부 PASS를 확인한 후 refactor를 시작했다.
- Llama golden 기준은 기존 artifact를 그대로 보존한다.
  - HF reference: `d_compiler/build/v3_reference_generate_hello_3.npz`
  - source 결과: `d_compiler/build/v3_source_generate_hello/` (cache length 9)

### 2026-08-19 — Stage G4.1: model-independent interface 추출

새 파일:

- `d_compiler/npu_compiler/model_spec.py`
  - `AttentionSpec`: kind(full/sliding), Q/KV heads, head_dim, window,
    rope_theta, partial_rotary_factor, llama3 rope scaling, qk_norm.
  - `LayerSpec`: index, attention, ffn_hidden, activation, `kv_owner`, ple_dim.
  - `ModelSpec`: hidden/vocab/eps/layers, tie_word_embeddings,
    scale_embeddings, final_logit_softcapping.
  - `CacheSlot`/`CachePlan`/`build_cache_plan`: KV sharing을 cache copy가 아닌
    owner slot alias로 표현. cross-kind sharing과 forward owner 참조는 오류.
  - `llama32_spec(config)`: official HF config dict에서 Llama 3.2 3B spec 생성.
  - `gemma4_e2b_spec()`: handoff §13.1의 official E2B text config로 Gemma spec 생성.
    4 sliding + 1 full 패턴, sliding HD=256/window=512/theta=10000,
    full HD=512/theta=1e6/partial rotary 0.25, 마지막 20 layer는 같은 kind의
    최신 owner(sliding→layer 13, full→layer 14)를 alias.
- `d_compiler/npu_compiler/generation.py`
  - `SourceGenerationSession`: family-독립 실행 계층. program 실행/시간/invocation
    집계(`_run`), program 크기 stats, greedy prefill+decode `generate()` loop,
    `SourcePrefillResult`/`SourceGenerationResult`.

변경 파일:

- `d_compiler/npu_compiler/v3_source_llama.py`
  - `Llama32SourceCompiler`가 `SourceGenerationSession`을 상속하고
    `llama32_spec`/`build_cache_plan`으로 구동된다.
  - layer loop는 `spec.num_layers`, cache slicing 기하는 `CachePlan.slot_for(layer)`의
    `num_kv_heads`/`head_dim`을 사용한다. LM head vocab도 spec에서 읽는다.
  - 공개 API(생성자 signature, `prefill`, `decode_token`, `generate`, `stats`)와
    `state.npz`/`result.json` 형식, 수치 경로는 그대로다. graph builder와
    weight binding은 변경하지 않았다.

semantic 사항 (spec에 넣지 않기로 한 것):

- norm ordering(pre vs sandwich), PLE 산술, activation lowering 방식은 spec이 아니라
  model-family graph builder 소관으로 남긴다. spec은 orchestration/cache가 필요로
  하는 구조 정보만 담는다.
- QK-Norm 모델의 score scale 규칙: `qk_norm=True`이면 scale 1, 아니면
  `1/sqrt(head_dim)` (spec docstring에 계약으로 명시).

검증:

- 신규 `d_compiler/tests/test_model_spec.py`
  - Llama spec: 28 full layer, 전 layer owner, plan 28 slot.
  - Gemma spec: 35 layer(7 full/28 sliding), per-kind head_dim/RoPE, 15 owner slot,
    shared 20 layer의 kind별 alias, PLE 256, vocab 262144, softcap 30.
  - invalid spec: window 규칙, GQA 배수, cross-kind/forward sharing 거부.
- repository 전체 test entrypoint sweep: 27개 전부 PASS (아래 §3).
- full-model gate: 공개 API를 통과하는 clean 3-token 재실행이 HF greedy와 일치 (아래 §3).

## 3. Stage G4.1 gate 결과

repository test sweep: 27개 entrypoint 전부 PASS (기존 26 + `test_model_spec.py`).

clean full-model gate — 공개 API 경로의 처음부터 재실행:

~~~bash
/home/chokwans99/anaconda3/envs/npu-tvm/bin/python \
  d_compiler/run_v3_source_generate.py \
  --prompt "Hello, NPU compiler!" --tokens 3 \
  --output d_compiler/build/v3_source_generate_hello_g41 \
  --reference d_compiler/build/v3_reference_generate_hello_3.npz
~~~

결과 (refactor 이전 golden과 동일):

~~~text
generated ids: [358, 2846, 4560]
decoded text: " I'm trying"
HF greedy reference: match
final decode logits vs HF: max abs 0.07422, mean abs 0.01002,
                           RMSE 0.01266, cosine 0.9999881
invocations: 146 (prefill 30 + decode 116)
source execution accumulated: 453.9 sec
~~~

정적 program 규모도 refactor 이전과 동일하다: prefill layer 497,563 words,
decode context 8/9 = 453,349/453,757 words, panel LM head 1,845,685 words.

| Stage G4.1 완료 조건 | 결과 |
|---|---|
| generic `ModelSpec`/`LayerSpec`/`AttentionSpec`/`CachePlan` 정의 | PASS |
| `Llama32Assets`/graph/runtime interface 분리 (동작 불변) | PASS |
| Llama 27개 test entrypoint | PASS |
| 3-token official generation 결과 유지 | PASS — `[358, 2846, 4560]` |

다음 단계는 Stage G4.2 — Gemma 4 E2B asset store(monolithic/sharded safetensors),
official config validation, tokenizer/chat template, 독립 HF reference,
checkpoint keyset inventory다. G4-002(`use_double_wide_mlp` semantics)를 이 단계에서
확정한다.

### 2026-08-19 — 단일-program decode layer 융합

decode layer 하나가 기존에는 두 program(K/V projection → host cache append →
attention/FFN)으로 실행되었다. 이를 layer당 하나의 program으로 융합했다.
Gemma decode(G4.6~G4.7)도 이 구조를 기본으로 사용한다.

- `model.build_v3_decode_fused_layer_module(cfg, context)` (신규 Relax graph builder):
  - cache 입력 `Kt{kv}[HD,context-1]`/`Vc{kv}[context-1,HD]`는 이전 position까지만 보관
  - program 내부에서 현재 token의 roped K/V를 계산하고 on-device concat으로
    `[HD,context]`/`[context,HD]` full cache를 구성 (context가 compile-time 상수라
    append 위치도 static address)
  - backend concat은 rank-2 last-axis 전용이므로 V append는 transpose 경유
    (copy 연산만 추가, 산술 없음)
  - 출력 `[y, K0,V0,...]` — host는 **decode step당 1회** 모든 layer의 cache를 연장
    (layer L의 새 K/V는 다음 position에서만 다시 읽히므로 layer 사이 개입 불필요)
  - 기존 pipeline 그대로: BlockBuilder + legalize primitive → memplan → ver.08 codegen.
    별도 실행 경로/ISA 변경 없음
- `Llama32SourceCompiler.decode_token`: layer당 fused program 1회 실행, projection을
  모아 step 끝에 일괄 append. 공개 API와 `state.npz` 형식 불변.
  기존 split builder(`build_v3_decode_kv_module`/`build_v3_decode_layer_module`)는
  회귀 비교용으로 유지.

검증:

- 신규 `test_single_program_decode_matches_split` (proxy REDUCED):
  fused output이 split 경로(kv program + host append + decode program)와
  **byte-exact**, 신규 K/V도 kv program 출력과 byte-exact, streaming oracle 비교 통과
- clean full-model 재실행 (`v3_source_generate_hello_fused/`):
  - token `[358, 2846, 4560]`, `" I'm trying"` — HF greedy 일치
  - 최종 decode logits vs HF가 split 경로 golden과 **마지막 자리까지 동일**
    (max abs 0.07421875, cosine 0.9999881354510036) — full 3B 규모에서 동등성 확인
  - invocation **146 → 90** (token당 58 → 30), source 실행 누적 452.1초
  - fused decode program: context 8/9 = 517,781/518,189 words
    (기존 split 합계 453,349+30,451과 유사, cache copy 명령 포함)
- 전체 test sweep 27개 entrypoint PASS
