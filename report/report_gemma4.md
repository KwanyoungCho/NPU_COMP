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
| G4-001 | RESOLVED | HIGH | Gemma `gelu_pytorch_tanh`는 vendor native GELU `x*sigmoid(2x)`와 다름 (V3-004 승계) | `legalize.gelu_tanh` primitive lowering 구현. C-model 무변경. vendor/source/FP16-emulation bit-exact, float64 대비 max 1.86e-3, 극단값 포화 정확 (`test_gelu_tanh.py`) |
| G4-002 | RESOLVED | MEDIUM | `use_double_wide_mlp: true`의 정확한 semantics 미확정 | checkpoint header + official 구현으로 확정: KV-shared 마지막 20개 layer(15~34)의 gate/up/down intermediate가 6144→12288로 2배, gating 수식 자체는 동일한 `down(act(gate(x)) * up(x))`. spec의 per-layer `ffn_hidden`에 반영 |
| G4-003 | RESOLVED | BLOCKER | Gemma 4 E2B checkpoint를 받을 디스크 공간 부족 — 단일 10.25GB safetensors, text-only(`model.language_model.*`)만 골라도 9.29GB인데 여유 8.8GB | 사용자 지시로 `/data2/chokwans99/npu_models/gemma4_e2b_hf/`에 전체 다운로드. repo 기본 경로 `d_compiler/build/gemma4_e2b_hf/model.safetensors`는 symlink |
| G4-004 | RESOLVED | HIGH | owner layer의 V에 **weight 없는 RMSNorm**(`with_scale=False`)이 적용됨 — weight parameter가 없어 checkpoint keyset에는 흔적이 없으므로 tensor 목록만으로 graph를 만들면 누락 위험 | official 구현에서 확인. attention graph에 weight-less RMSNorm(d=head_dim, eps) 포함. `rms_norm` builder의 weight 곱 생략 변형 사용 |
| G4-005 | OPEN | HIGH | 35 layer FP16 range: HF는 norm을 float32 내부 연산으로 하지만 우리는 모든 중간값이 FP16 — Gemma 계열의 큰 activation에서 V3-020류 overflow 가능성 | 안전 순서 norm(선-스케일링) 유지, layer별 checkpoint/finite 검사로 발생 지점 즉시 격리. HF FP16 reference가 finite 완주한 것은 긍정 신호 |
| G4-006 | RESOLVED | MEDIUM | full-attention layer의 `proportional` RoPE + `partial_rotary_factor 0.25`의 정확한 주파수/적용 식 미확정 | HF `_compute_proportional_rope_parameters` 확인: full head_dim 지수의 주파수 중 rotary 초과분을 **0으로** 두고 full-dim rotate-half를 그대로 적용 (freq 0 → cos=1/sin=0 → passthrough). slice/concat 불필요, 기존 `legalize.rope` 재사용 — `gemma4_graph.gemma_freqs_row` |
| G4-007 | OPEN | MEDIUM | scale 사슬(embedding ×√1536, projection ×1/√1536, PLE 결합 ×1/√2, PLE token ×√256)의 적용 순서·dtype이 FP16 경계에서 수치를 좌우 | 단계별 FP16 emulation과 HF 중간값 대조로 각 지점 검증 (V3-020 교훈) |
| G4-008 | OPEN | MEDIUM | decode에서 shared layer(15~34)는 **현재 token을 포함한** owner(13/14)의 K/V 상태를 같은 step 안에서 읽어야 함 | 실행 순서상 owner가 먼저 실행되므로 host가 owner 출력의 새 K/V를 같은 step에서 shared layer 입력으로 전달. cache append 자체는 step 말미 일괄 유지 |

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

### 2026-08-19 — Stage G4.2 착수: official config/keyset inventory (checkpoint 없이)

사용자 방침: vendor로 kernel parity를 먼저 검증하고, full model은 source C-model에서
표준 tanh-GELU를 compiler lowering으로 처리하여 prefill/decode 모두 HF와 일치시키는
것이 최종 목표. 입력은 HF/PyTorch model에서 시작할 수 있어야 한다. C-model의
GELU semantics는 변경하지 않는다(vendor parity 유지).

환경 확인:

- `npu-tvm`의 transformers 5.12.1이 `Gemma4ForCausalLM`/`Gemma4TextConfig` 등을
  완전히 지원. torch 2.12.0+cpu.
- checkpoint는 base/-it 모두 **단일 10.25GB safetensors**. 디스크 여유 8.8GB로 부족
  (G4-003). HF cache의 3.8GB는 다른 연구(layerskip 등) 캐시라 삭제 불가.

checkpoint를 받지 않고 확정한 사실 (safetensors header를 HTTP Range로 fetch,
2011개 tensor inventory):

- 구성: `model.language_model.*` 9.29GB / `model.audio_tower.*` 0.61GB /
  `model.vision_tower.*` 0.33GB. text 내부는 PLE table
  `embed_tokens_per_layer [262144, 8960]` 4.70GB, layers 3.76GB,
  `embed_tokens [262144, 1536]` 0.81GB.
- **layer 분류가 spec과 일치**: full attention(q `[4096,1536]`, HD 512)은
  layer 4,9,14,19,24,29,34. sliding(q `[2048,1536]`, HD 256)은 나머지 28개.
- **double-wide MLP 확정(G4-002)**: layer 0~14 gate/up `[6144,1536]`,
  layer 15~34 `[12288,1536]` — KV-shared 경계와 정확히 일치.
- layer keyset: q/k/v/o + q_norm/k_norm + 4개 layer norm + `layer_scalar[1]` +
  PLE (`per_layer_input_gate [256,1536]`, `per_layer_projection [1536,256]`,
  `post_per_layer_input_norm`). MoE/expert 계열 없음.

`config.json`/`tokenizer.json` 등 소형 파일은 `d_compiler/build/gemma4_e2b_hf/`에
확보. `text_config`는 handoff §13.1 분석과 전부 일치 (`layer_types`,
`rope_parameters`의 full: theta 1e6/partial 0.25, sliding: theta 1e4,
softcap 30, eps 1e-6, `enable_moe_block: false`).

설치된 official `modeling_gemma4.py`에서 decoder layer 수식 확정:

~~~text
residual → input_norm → attn → post_attention_norm → +residual
residual → pre_ff_norm → down(act(gate(x)) * up(x)) → post_ff_norm → +residual
PLE: residual → act(per_layer_input_gate(h)) * per_layer_input
     → per_layer_projection → post_per_layer_input_norm → +residual
h *= layer_scalar
~~~

spec 반영:

- `gemma4_e2b_spec()`이 official `text_config` dict 형식을 직접 소비하도록 변경
  (`layer_types`/`rope_parameters` 사용). 내장 기본값 `GEMMA4_E2B_TEXT`는
  다운로드된 config.json과의 동일성 test로 검증.
- per-layer `ffn_hidden`: 0~14는 6144, 15~34는 12288.

### 2026-08-19 — Stage G4.2/G4.3: checkpoint 확보 및 표준 tanh-GELU lowering

- 사용자 지시로 `/data2`(별도 디스크, 235GB 여유)에 전체 checkpoint를 받았다
  (G4-003 해소). repo 기본 경로에는 symlink만 두어 홈 디스크를 쓰지 않는다.
  경로 재지정은 `NPU_GEMMA4_PATH`.
- `legalize.gelu_tanh(bb, x, rows, cols)` 추가 (G4-001):
  - `0.5x(1+tanh(c(x+0.044715x³))) == x·sigmoid(2c(x+0.044715x³))` 항등식을 사용해
    mul/add/negative/exp/divide primitive만으로 lowering. native GELU opcode는
    사용하지 않고 C-model도 변경하지 않는다.
  - 다항식을 `x·(2c + 2c·0.044715·x²)`로 인수분해해 최대 중간값을 x³이 아닌
    x²로 낮췄다.
  - 포화 계약: 큰 양수는 `exp(-t)=0 → x` 그대로, 큰 음수는 exp가 FP16 inf로
    overflow해 `1/inf=0 → 0`. IEEE inf 전파를 vendor에서 실측 확인.
- `tests/test_gelu_tanh.py` (28번째 entrypoint):
  - dense `[-8,8]`/near-zero/극단값 3개 case에서
    **vendor a.out == source C-model == FP16 step-emulation bit-exact**
  - float64 표준식 대비 max abs 1.86e-3 (FP16 인자 반올림 한계 수준)
  - ±300/±3000 포화 정확 (`gelu(+big)=+big`, `gelu(-big)=0`)
  - native vendor GELU와 결과가 구분됨을 확인 (lowering이 실제로 다른 함수)
- `make_gemma4_generation_reference.py`: official checkpoint의
  `Gemma4ForConditionalGeneration` FP16 eager CPU로 독립 greedy reference 생성
  (Llama 방법론과 동일).
- 독립 HF greedy reference 확보 (CPU 29.4초,
  `d_compiler/build/gemma4_reference_generate_hello_3.npz`):

~~~text
prompt: Hello, NPU compiler!
input ids: [2, 9259, 236764, 646, 11152, 47133, 236888]
generated ids: [108, 236777, 236789]
decoded text: "\n\nI'"
FP16 logits finite: true
~~~

이것이 Gemma 4 E2B text-only prefill/decode의 golden target이다. 다음 단계는
Gemma assets loader(safetensors slice, `model.language_model.*` prefix),
PyTorch/HF module 기반 frontend 경로, Gemma layer graph builder(G4.3~G4.4)다.

### 2026-08-19 — official attention/norm 세부 확정 및 구현 방침

official `modeling_gemma4.py` 추가 분석으로 확정:

- attention score scale은 정확히 **1.0** (`self.scaling = 1.0`) — QK-Norm이
  1/sqrt(HD)를 대체.
- `Gemma4RMSNorm`은 `x * (mean(x^2)+eps)^-0.5 * w` — Gemma 2/3식 `(1+w)`가
  **아니므로** Llama와 같은 `legalize.rms_norm` builder를 그대로 재사용 가능.
- owner layer attention: Q/K는 head별 RMSNorm 후 RoPE, **V는 weight 없는
  RMSNorm**(G4-004). shared layer는 q_proj/q_norm/o_proj만 갖고 K/V는
  layer type별 마지막 owner(sliding→13, full→14)의 full-length 상태를 재사용 —
  `CachePlan` owner alias 설계와 코드가 일치함을 확인.

사용자 확정 방침:

1. 목표는 **Gemma prefill/decode greedy token이 HF golden과 일치**하는 것.
2. graph/lowering은 **TVM Relax IR + 기존 legalize builder + relax.transform
   pass 경로**를 유지한다. backend/ISA 밖의 custom 실행 경로를 만들지 않는다.
3. **GELU는 source C-model 경로에서 표준식으로 정상 동작하는 것을 우선**하고,
   전체 완료 후 vendor 실행파일에서도 kernel 단위 재확인으로 마무리한다
   (lowering 자체는 이미 vendor/source bit-exact 검증됨).
4. **PLE는 사전계산 테이블 방식으로 확정**: vocab 전체 per_layer_input을
   NPU FP16 step-emulation과 동일한 수식으로 오프라인 계산해 `/data2`에
   저장(4.7GB)하고, 실행 시에는 token row lookup만 한다. 테이블 생성 수식은
   표본 token에 대해 실제 source C-model program과 bit-exact 대조로 증명한다.
   (multimodal은 token id가 없어 이 방식이 불가하지만 text-only 범위에서는 유효)

### 2026-08-19 — Stage G4.2 완료 및 G4.3/G4.4 착수

- `npu_compiler/gemma4_model.py` — `Gemma4Assets`: 단일 safetensors에서
  `model.language_model.*` slice 로딩, spec 기반 layer별 shape, 540개 필수 key
  검증(shared layer의 미사용 k/v tensor는 official과 동일하게 무시), embedding
  FP16 scale, PLE row slice, panel LM packing, PLE 테이블 memmap. tokenizer
  golden id 일치 확인. revision `d29ff6b4` 고정.
- PLE 사전계산 pipeline:
  - `make_gemma4_ple_table.py`가 **C-model의 정확한 누산 순서**(64-K tile 내부
    순차합→tile간 MAC, row reduce 순차합)를 elementwise 연산 loop로 재현 —
    elementwise는 SIMD/thread에 무관하게 순서가 보존되므로 결정적.
  - `tests/test_gemma4_ple.py`: 표본 11 token × 8960 값 전체가 실제 source
    C-model PLE program(`build_gemma4_ple_module`)과 **bit-exact** 일치.
  - 전체 262144 token 테이블(4.7GB)은 detached 프로세스로 생성 중 (resume 지원).
- `npu_compiler/gemma4_graph.py` — `build_gemma4_prefill_layer_module`:
  4가지 layer 형태(sliding/full × owner/shared)를 spec에서 읽어 생성.
  QK-Norm, weight-less V-norm(`legalize.rms_norm`의 `w=None` 지원 추가),
  scale-1 attention, banded sliding mask, proportional RoPE(zero-freq
  passthrough), tanh-GELU gated MLP(double-wide 폭 포함), PLE 주입,
  layer_scalar까지 전부 NPU program 산술. shared layer는 owner의 Kt/V를
  입력으로 받음.
- `tests/test_gemma4_layer.py` (proxy, checkpoint 불필요): 4개 layer 형태 모두
  - source C-model vs 고정-buffer streaming oracle ≤ 1e-2 (K-tile 경계 FP16
    반올림 증폭에 대한 문서화된 허용치)
  - float64 official 수식 reference 대비 max rel ≤ 1.1%
  - owner의 K/V cache 출력도 reference와 일치

### 2026-08-19 — Stage G4.4: official 실제 layer HF 대조 (PASS)

`make_gemma4_layer_reference.py`로 HF FP16 eager의 layer별 중간값(embedding,
`per_layer_inputs`, hidden 36개, final norm, logits)을 확보하고,
`tests/test_gemma4_real_layer.py`에서 official weight로 우리 program을 실행해
같은 입력 hidden에서 다음 hidden과 비교했다:

~~~text
PLE per_layer_inputs:        cosine 0.99999983, mean abs 1.5e-4
layer  0 (sliding owner):    cosine 0.99999973, max abs 0.031
layer  4 (full owner):       cosine 0.99999967, max abs 0.031   ← proportional RoPE 검증
layer 13 (sliding owner):    cosine 0.99999927
layer 15 (sliding shared):   cosine 0.99999930                  ← double-wide + KV alias 검증
~~~

### 2026-08-19 — Stage G4.5/G4.6: official 3-token prefill+decode 완료 (PASS)

`Gemma4SourceCompiler` + `run_gemma4_source_generate.py`의 첫 실행에서:

~~~text
prompt: Hello, NPU compiler!
input ids: [2, 9259, 236764, 646, 11152, 47133, 236888]
generated ids: [108, 236777, 236789]
decoded text: "\n\nI'"
HF greedy golden과 완전 일치 (match: true)

최종 decode logits vs HF (softcap은 비교 시에만 적용):
  max abs 0.0452, mean abs 0.0091, RMSE 0.0113, cosine 0.9999975

invocations: 111 = prefill (35 layer + norm + LM)
                 + 2 decode steps × (35 layer + norm + LM)
source 실행 누적: 530.7초 (8.8분)
ple_source: computed (테이블 생성 완료 전 fallback; 값은 bit-exact 동등)
~~~

프로그램 규모 (shape class별 재사용):

| Program | words |
|---|---:|
| prefill sliding/full owner (F=6144) | 182,753 / 217,533 |
| prefill sliding/full shared (F=12288) | 311,689 / 342,665 |
| decode 4 class × context 8/9 | 184,701 ~ 336,453 |
| panel LM head (vocab 262144) | 1,904,641 |

decode 실행 구조: owner layer(13/14 이전)는 fused program이 현재 token K/V를
in-program append하고 host가 slot cache를 즉시 연장 → 같은 step의 shared
layer(15~34)가 연장된 cache를 읽는다 (G4-008 설계 그대로).

### 2026-08-19 — vendor 마무리 확인 (사용자 지시 이행)

- tanh-GELU lowering은 이미 vendor a.out에서 bit-exact 검증됨 (`test_gelu_tanh`).
- 추가로 `test_gemma4_layer.py`에 vendor closure를 넣었다: 가장 연산이 많은
  Gemma layer 형태(sliding owner — QK/V norm, RoPE, tanh-GELU MLP, PLE,
  layer scalar)를 **무수정 vendor a.out**의 fixed-buffer streaming plan으로
  실행 → source 단일 program과 **max abs 0.0 완전 일치**.
