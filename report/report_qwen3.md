# Qwen3-4B 지원 진행 보고서 (gemma4-e2b branch)

> 시작일: 2026-08-19
>
> 목표: **동일 0818 compiler core에서 official Qwen3-4B text prefill/decode를
> 지원하고 HF greedy token과 일치시킨다.** Llama 3.2 3B / Gemma 4 E2B 결과는
> 회귀 기준으로 유지한다.

Qwen3는 Llama형 구조(전 layer full causal attention, layer별 독립 GQA KV,
SiLU gated MLP, pre-norm 2개)에 Gemma형 **per-head QK-Norm**이 더해진 중간형이다.
G4.1에서 추출한 model-independent spec/generation 기반이 그대로 적용된다.

## 1. Official 구성 (Qwen/Qwen3-4B, config.json 확인)

~~~text
hidden_size: 2560
intermediate_size: 9728
num_hidden_layers: 36
num_attention_heads: 32 / num_key_value_heads: 8 (GQA 4:1)
head_dim: 128  →  q width 4096 != hidden 2560 (Gemma형 비대칭)
vocab_size: 151936 (= 64 × 2374, panel LM 재사용 가능)
hidden_act: silu, rms_norm_eps: 1e-6
rope_theta: 1e6, rope_scaling 없음, sliding window 없음
tie_word_embeddings: true
attention scale: 기본 1/sqrt(head_dim) (QK-Norm이 있어도 Gemma와 달리 유지)
V-norm 없음 (Gemma와 다름), Q/K norm만 존재
~~~

checkpoint: 3-shard safetensors 8.06GB, `/data2/chokwans99/npu_models/qwen3_4b_hf/`
에 다운로드, `d_compiler/build/qwen3_4b_hf/`에 symlink. 경로 재지정
`NPU_QWEN3_PATH`. keyset 398개 검증 통과 (q_norm/k_norm 포함, lm_head는 tied라 없음).

## 2. 구현

| 파일 | 내용 |
|---|---|
| `model_spec.py` | `AttentionSpec.score_scale` field 추가 (None=1/sqrt(HD) 기본, Gemma는 명시적 1.0), `qwen3_spec(config)` |
| `npu_compiler/qwen3_model.py` | sharded index 기반 slice loader, spec 기반 layer shape, tied/untied LM 대응 |
| `npu_compiler/qwen3_graph.py` | prefill layer + fused decode layer + final norm. Llama 흐름 + QK-Norm(rope 전) + q폭 4096. RoPE 주파수는 Gemma의 `gemma_freqs_row` 재사용(prf=1.0 → 표준식) |
| `npu_compiler/qwen3_source.py` | `Qwen3SourceCompiler` — 36 layer가 prefill program 1개 재사용, decode는 context별 fused program, cache는 Llama와 동일한 `keys[layer][kv]` layout |
| `make_qwen3_generation_reference.py` | HF FP16 eager greedy golden + layer별 hidden 중간값 |
| `run_qwen3_source_generate.py` | resumable runner |

주의점 (HF 중간값 ladder에서 확인):

- HF `output_hidden_states`의 **마지막 원소는 final RMSNorm이 적용된 값**이다.
  마지막 layer 비교는 우리 출력에 final norm을 적용해 대조한다.

## 3. 검증

- `tests/test_model_spec.py`: qwen3 spec (36 full layer, GQA 32/8, score_scale
  기본, 전 layer owner).
- `tests/test_qwen3_layer.py` (proxy): prefill hidden/K/V 및 fused decode를
  float64 official 수식 reference와 대조 (max rel ≤ 0.2%), streaming oracle
  일치, **무수정 vendor a.out closure max abs 0.0**.
- `tests/test_qwen3_assets.py`: keyset/shape/embedding/tokenizer.
- `tests/test_qwen3_real_layer.py` (official weight): layer 0/18/35를 HF hidden
  중간값과 대조:

~~~text
layer  0: cosine 0.9999980
layer 18: cosine 0.99999999994
layer 35 (+final norm): cosine 0.9999997
~~~

## 4. Golden target 및 최종 결과

독립 HF FP16 greedy reference:

~~~text
prompt: Hello, NPU compiler!
input ids: [9707, 11, 451, 6325, 19415, 0]   (Qwen은 BOS 미부착)
generated ids: [358, 1184, 311]
decoded text: " I need to"
~~~

`Qwen3SourceCompiler` + `run_qwen3_source_generate.py`의 첫 실행 결과:

~~~text
generated ids: [358, 1184, 311]
decoded text: " I need to"
HF greedy golden과 완전 일치 (match: true)
최종 decode logits vs HF: max abs 0.0273, mean abs 0.0089,
                          RMSE 0.0099, cosine 0.9999915
invocations: 114 = prefill (36 layer + norm + LM)
                 + 2 decode steps × (36 layer + norm + LM)
source 실행 누적: 552.8초 (9.2분)
~~~

프로그램 규모: prefill layer 511,076 words (36 layer 재사용), decode context
7/8 = 526,328/526,872 words, panel LM head 1,825,607 words.

## 5. 최종 판정 — Qwen3-4B text

| 완료 조건 | 결과 |
|---|---|
| HF checkpoint에서 출발 (config/tokenizer/weight 직접 소비) | PASS |
| TVM Relax + 공통 legalize/backend 경로 유지 | PASS |
| official 36-layer prefill (QK-Norm, GQA 32/8, q폭 4096) | PASS |
| fused decode (in-program K/V append, step당 host 연장 1회) | PASS |
| HF greedy token 일치 | PASS — `[358, 1184, 311]`, `" I need to"` |
| 최종 logits vs HF | cosine 0.9999915 |
| vendor a.out closure (proxy layer) | PASS — max abs 0.0 |
| Llama/Gemma 회귀 유지 | PASS |
| 전체 regression sweep | PASS — 36 entrypoints |

이로써 세 model family(Llama 3.2 3B, Gemma 4 E2B, Qwen3-4B)가 동일한
0818 compiler core(공통 spec/generation/legalize/backend) 위에서 HF greedy와
일치하는 prefill/decode를 수행한다.
