'use strict';
const { C, F } = require('./lib');

module.exports = (D) => {
  // ---------------------------------------------------------------- 10 relax.frontend.nn
  D.add({ section: '2 · 모델 → Relax', title: 'Llama는 relax.frontend.nn으로 정의하고 정적 명세로 export한다', subtitle: 'Compiler가 보는 것은 Python의 for-loop가 아니라 export된 계산 그래프입니다. Embedding lookup은 host에 남기고 input_embeds를 device 입력으로 받습니다.', source: 'npu_compiler/nn_models/llama.py, docs/current/02_model_relax.md',
    notes: 'nn_models/llama.py는 Attention(q/k/v/o Linear, bias 없음), MLP(down(silu(gate)·up)), DecoderLayer(pre-norm, residual 2개)를 relax.frontend.nn Module로 씁니다. RoPE는 host가 준 cos/sin과 rotate_half(split → negative → concat)로 표현하고, GQA는 op.repeat로 K/V를 3배 반복합니다. get_default_spec(seq, capacity)가 정적 shape를 고정하고 export_tvm이 packed_params tuple과 num_input attribute를 붙입니다. build_generate는 세 entry를 export한 뒤 decode 함수에 npu.cache_append 계약을 선언합니다.' },
  (s, D) => {
    D.code(s, 0.5, 1.5, 6.3, 3.55, [
      [{ text: 'class ', color: C.codeKey }, { text: 'Attention(nn.Module):        # llama.py' }],
      '  q/k/v/o = nn.Linear(..., bias=False)   # k,v out = 8*128',
      [{ text: '  def ', color: C.codeKey }, { text: 'forward(self, x, cos, sin, mask):' }],
      '    q,k,v = reshape → [S,24,128], [S,8,128]',
      '    q,k   = x*cos + rotate_half(x)*sin   # split/neg/concat',
      '    k,v   = op.repeat(k, 3, axis=1)      # GQA: 24 Q ← 8 KV',
      '    scores = matmul(q_h, kᵀ_h) * (1/√128) + mask',
      '    probs  = softmax(scores)             # FP16 (FP32 sum 내부)',
      '    return o_proj(matmul(probs, v_h))',
      [{ text: 'class ', color: C.codeKey }, { text: 'MLP:  down(silu(gate(x)) * up(x))' }],
      [{ text: 'class ', color: C.codeKey }, { text: 'DecoderLayer:  x + attn(norm1(x)); r + mlp(norm2(r))' }],
      '',
      'spec = get_default_spec(seq=7, decode_context=9)',
      'mod, params = LlamaModel(cfg).export_tvm(spec)   # packed params',
      'mod["decode"] = npu_cache.declare(mod["decode"],',
      '     [("k_cache", 1, 3), ("v_cache", 2, 2)])  # npu.cache_append',
    ], { size: 9 });
    D.table(s, 7.0, 1.5, 5.83, [
      ['Entry', 'Device 입력 (FP16)', '반환'],
      ['prefill', 'input_embeds [S,3072], cos/sin [S,1,128], mask [1,S,S], packed_params', 'logits [S,128256]'],
      ['prefill_cache', '위와 같음 (num_input = 4)', '(logits [1,V]*, K [L,8,128,S], V [L,8,S,128])'],
      ['decode', 'input_embeds [1,3072], cos/sin [1,1,128], k_cache [L,8,128,C], v_cache [L,8,C,128], mask [1,1,C+1], packed_params (num_input = 6)', '(logits [1,V], K_new [L,8,128,1], V_new [L,8,1,128])'],
    ], { colW: [1.15, 2.98, 1.7], size: 8.5, rowH: 0.5 });
    D.bullets(s, 7.0, 3.85, 5.83, 1.3, [
      ['packed_params: ', '9L+2개 tensor를 담은 tuple 인자 하나 (ABI 표현). 물리적으로 합쳐진 배열이 아님. weight는 HF [N,K] 그대로'],
      ['num_input: ', '앞 N개 인자만 실행 입력, 나머지는 parameter라는 경계 attribute → LiftTransformParams가 사용'],
      ['*last_logits: ', 'generation 경로는 마지막 hidden 행만 norm/LM head에 넘겨 [1,V] logits (TE last_hidden)'],
    ], { size: 9.5, psa: 4 });
    D.takeaway(s, 'Cache 효과는 함수 attribute(npu.cache_append)로 선언되어 순수 tensor 계산 밖에 있습니다. layer 수나 cache 이름을 codegen에 하드코딩하지 않습니다.', { y: 6.5 });
    D.text(s, 0.5, 5.15, 6.3, 1.2, 'K cache는 [L, kv, hd, C]로 transposed 저장, V는 [L, kv, C, hd]입니다. decode는 cache와 현재 token의 K/V를 concat하여 C+1 길이로 attention하고, 현재 token은 임시로 slot C에 놓입니다.', { size: 10, color: C.muted });
  });

  // ---------------------------------------------------------------- 11 Llama layer dataflow
  D.add({ section: '2 · 모델 → Relax', title: 'Llama 한 layer의 계산 그래프 (prefill, S=7): tensor의 모양으로 보기', subtitle: '각 glyph는 tensor 하나이고 화살표 위 글자가 연산입니다. 겹친 glyph는 head 축(24 또는 8), 회색 삼각형은 causal mask. 두 residual과 GQA repeat, RoPE의 위치를 봅니다. 이 그래프가 28번 반복됩니다.', source: 'docs/examples/llama_ir/01_authored_relax.py, docs/current/02_model_relax.md',
    notes: 'Attention 앞 RMSNorm 결과가 Q/K/V projection에 들어갑니다. Q는 [7,24,128], K/V는 [7,8,128]입니다. RoPE는 Q와 K에만 적용됩니다. K/V를 repeat(3, axis=1)로 24 head로 늘린 뒤 head를 앞으로 보내 [24,7,128] batch matmul을 합니다. scores [24,7,7]에 1/√128 scale과 causal mask를 더하고 softmax, PV, O projection 뒤 첫 residual을 더합니다. MLP는 gate/up 두 갈래([7,8192] 각각, weight 48 MiB씩)를 SiLU·곱으로 합쳐 down projection 후 두 번째 residual을 더합니다. glyph의 칸 수는 축약이며 실제 tile 수(예: [7,3072]는 폭 48 tiles)의 비율이 아닙니다.' },
  (s, D) => {
    const stack = (x, y, w, h, rows, cols, kind, n, o) => { for (let i = n - 1; i >= 1; i--) D.matrix(s, x + i * 0.06, y - i * 0.06, w, h, rows, cols, { kind, cells: false, lineWidth: 0.75 }); return D.matrix(s, x, y, w, h, rows, cols, Object.assign({ kind }, o || {})); };
    const lbl = (x, y, w, t, o) => D.text(s, x, y, w, 0.3, t, Object.assign({ size: 7.5, color: C.muted, align: 'center' }, o || {}));
    // ---- attention row
    const y1 = 2.4;
    const X = D.matrix(s, 0.5, y1, 1.2, 0.5, 1, 8, { kind: 'host', title: 'X [7,3072]', titleW: 1.3 });
    D.arrow(s, X.r + 0.02, X.cy, 2.13, X.cy, { width: 1.2 }); lbl(X.r - 0.05, X.cy - 0.32, 0.55, 'RMSNorm');
    const N1 = D.matrix(s, 2.15, y1, 1.2, 0.5, 1, 8, { kind: 'ir', title: 'norm₁ [7,3072]', titleW: 1.4 });
    const Q = stack(4.1, 1.95, 0.9, 0.42, 1, 4, 'unit', 3, { title: 'Q [7,24,128]', titleW: 1.3 });
    const K = stack(4.1, 2.67, 0.9, 0.42, 1, 4, 'unit', 2, { title: 'K [7,8,128]', titleW: 1.2 });
    const V = stack(4.1, 3.39, 0.9, 0.42, 1, 4, 'unit', 2, { title: 'V [7,8,128]', titleW: 1.2 });
    D.arrow(s, N1.r + 0.02, N1.cy, Q.x, Q.cy, { width: 1 }); D.arrow(s, N1.r + 0.02, N1.cy, K.x, K.cy, { width: 1 }); D.arrow(s, N1.r + 0.02, N1.cy, V.x, V.cy, { width: 1 });
    lbl(3.35, 1.85, 0.75, 'q_proj', { color: C.unit, bold: true }); lbl(3.35, 2.45, 0.75, 'k_proj', { color: C.unit, bold: true }); lbl(3.35, 3.3, 0.75, 'v_proj', { color: C.unit, bold: true });
    D.chip(s, 5.15, 2.03, 'RoPE', { kind: 'ir', w: 0.5, size: 7 }); D.chip(s, 5.15, 2.75, 'RoPE', { kind: 'ir', w: 0.5, size: 7 });
    D.chip(s, 5.75, 2.75, 'repeat ×3', { kind: 'ir', w: 0.75, size: 7 }); D.chip(s, 5.75, 3.47, 'repeat ×3', { kind: 'ir', w: 0.75, size: 7 });
    const masked = []; for (let r = 0; r < 7; r++) for (let c = r + 1; c < 7; c++) masked.push({ r0: r, r1: r + 1, c0: c, c1: c + 1, kind: 'faint', lineWidth: 0.3 });
    const Sc = stack(6.85, 1.95, 0.8, 0.8, 7, 7, 'unit', 3, { title: 'scores [24,7,7]', titleW: 1.4, regions: masked });
    D.arrow(s, 5.7, 2.2, Sc.x, Sc.y + 0.2, { width: 1 }); D.arrow(s, 6.55, 2.9, Sc.x, Sc.cy + 0.15, { width: 1 });
    lbl(5.65, 1.6, 1.3, 'Q · Kᵀ (batch matmul)', { size: 7 });
    const P = stack(8.35, 1.95, 0.8, 0.8, 7, 7, 'unit', 3, { title: 'P [24,7,7]', titleW: 1.2, regions: masked });
    D.arrow(s, Sc.r + 0.15, Sc.cy, P.x, P.cy, { width: 1 }); lbl(7.6, 2.8, 0.85, '×1/√128\n+mask\nsoftmax', { size: 6.5 });
    const AO = stack(9.85, 2.15, 0.9, 0.42, 1, 4, 'unit', 3, { title: 'P·V [24,7,128]', titleW: 1.3 });
    D.arrow(s, P.r + 0.15, P.cy, AO.x, AO.cy, { width: 1 }); D.arrow(s, 6.55, 3.6, AO.x, AO.cy + 0.1, { width: 1, color: C.muted });
    lbl(7.3, 3.7, 2.5, 'V (repeat ×3) → P·V', { size: 6.5 });
    const AT = D.matrix(s, 11.3, y1, 1.2, 0.5, 1, 8, { kind: 'ir', title: 'attn [7,3072]', titleW: 1.3 });
    D.arrow(s, AO.r + 0.15, AO.cy, AT.x, AT.cy, { width: 1 }); lbl(10.7, 1.77, 0.8, 'reshape\no_proj', { size: 6.5 });
    const R1 = D.box(s, 12.6, y1, 0.23, 0.5, '+', { kind: 'warn', size: 12 });
    D.arrow(s, AT.r, AT.cy, R1.x, R1.cy, { width: 1 });
    D.line(s, X.cx, X.y, X.cx, 1.73, { arrow: false, color: C.warn, dash: 'dash' }); D.line(s, X.cx, 1.73, R1.cx, 1.73, { arrow: false, color: C.warn, dash: 'dash' }); D.line(s, R1.cx, 1.73, R1.cx, R1.y, { color: C.warn, dash: 'dash' });
    D.text(s, 1.4, 1.52, 2.4, 0.2, 'residual: X 유지 → R = X + attn', { size: 7.5, color: C.warn, bold: true });
    // ---- MLP row
    const y2 = 4.85;
    const R = D.matrix(s, 0.5, y2, 1.2, 0.5, 1, 8, { kind: 'warn', title: 'R [7,3072]', titleW: 1.3 });
    D.line(s, R1.cx, R1.b, R1.cx, 4.3, { arrow: false, color: C.muted }); D.line(s, R1.cx, 4.3, R.cx, 4.3, { arrow: false, color: C.muted }); D.arrow(s, R.cx, 4.3, R.cx, R.y - 0.26, { color: C.muted });
    D.arrow(s, R.r + 0.02, R.cy, 2.13, R.cy, { width: 1.2 }); lbl(R.r - 0.05, R.cy - 0.32, 0.55, 'RMSNorm');
    const N2 = D.matrix(s, 2.15, y2, 1.2, 0.5, 1, 8, { kind: 'ir', title: 'norm₂ [7,3072]', titleW: 1.4 });
    const G = D.matrix(s, 4.1, 4.4, 2.6, 0.42, 1, 16, { kind: 'unit', title: 'G = gate(norm₂) [7,8192]   (W 48 MiB)', titleW: 3.0 });
    const U = D.matrix(s, 4.1, 5.3, 2.6, 0.42, 1, 16, { kind: 'unit', title: 'U = up(norm₂) [7,8192]   (W 48 MiB)', titleW: 3.0 });
    D.arrow(s, N2.r + 0.02, N2.cy, G.x, G.cy, { width: 1 }); D.arrow(s, N2.r + 0.02, N2.cy, U.x, U.cy, { width: 1 });
    lbl(3.35, 4.33, 0.75, 'gate_proj', { color: C.unit, bold: true }); lbl(3.35, 5.4, 0.75, 'up_proj', { color: C.unit, bold: true });
    const Hm = D.matrix(s, 7.5, y2, 2.6, 0.42, 1, 16, { kind: 'ir', title: 'H = SiLU(G) ⊙ U [7,8192]', titleW: 2.8 });
    D.arrow(s, G.r + 0.02, G.cy, Hm.x, Hm.cy - 0.05, { width: 1 }); D.arrow(s, U.r + 0.02, U.cy, Hm.x, Hm.cy + 0.05, { width: 1 });
    lbl(6.65, 4.75, 0.9, 'SiLU · ⊙', { size: 7, color: C.ir, bold: true });
    const DN = D.matrix(s, 11.3, y2, 1.2, 0.5, 1, 8, { kind: 'unit', title: 'down(H) [7,3072]', titleW: 1.4 });
    D.arrow(s, Hm.r + 0.02, Hm.cy, DN.x, DN.cy, { width: 1 }); lbl(10.15, 4.55, 1.1, 'down_proj (W 48 MiB)', { size: 6.5 });
    const R2 = D.box(s, 12.6, y2, 0.23, 0.5, '+', { kind: 'warn', size: 12 });
    D.arrow(s, DN.r, DN.cy, R2.x, R2.cy, { width: 1 });
    D.line(s, R.cx, R.b, R.cx, 5.97, { arrow: false, color: C.warn, dash: 'dash' }); D.line(s, R.cx, 5.97, R2.cx, 5.97, { arrow: false, color: C.warn, dash: 'dash' }); D.line(s, R2.cx, 5.97, R2.cx, R2.b, { color: C.warn, dash: 'dash' });
    D.text(s, 1.4, 6.01, 4, 0.2, 'residual: R 유지 → Y = R + down(H) → 다음 layer의 X', { size: 7.5, color: C.warn, bold: true });
    D.callout(s, 0.5, 6.3, 12.33, 0.5, [
      { text: '읽는 법: ', bold: true }, { text: '보라 = matrix unit(projection·QKᵀ·PV), 파랑 = vector/정규화, 주황 = residual. glyph의 칸 수는 축약(실제 [7,3072]는 폭 48 tiles). Decode는 S=1이며 K/V 대신 cache와 현재 token을 concat한 [8,128,C+1]/[8,C+1,128]을 씁니다.', size: 9.5 },
    ], { kind: 'faint' });
  });

  // ---------------------------------------------------------------- 12 authored relax excerpt
  D.add({ section: '2 · 모델 → Relax', title: '실제 authored Relax (decode, 1 layer, C=9) 발췌', subtitle: '실제 3B 폭, layer 수만 28→1로 줄인 export 결과. 변수 이름은 자동 생성이지만 계산은 앞 슬라이드의 그래프와 1:1 대응합니다.', source: 'docs/examples/llama_ir/01_authored_relax.py (IR export만, checkpoint/실행 없음)',
    notes: 'decode 함수의 signature에 6개 실행 입력과 packed_params가 있고 npu.cache_append attribute가 붙어 있습니다. permute_dims(weight)가 layer마다 나타나는데 이것은 3장의 LiftTransformParams가 host로 옮깁니다. cache concat 뒤 repeat로 24 head가 되고, scores [24,1,10]에 scale 0.0884(=1/√128)와 mask [1,1,10]을 더합니다. 반환 tuple은 logits와 현재 token의 K/V이며 runtime의 append DMA가 cache slot에 옮깁니다.' },
  (s, D) => {
    D.code(s, 0.5, 1.5, 8.3, 5.35, [
      [{ text: '@R.function', color: C.codeKey }],
      [{ text: 'def ', color: C.codeKey }, { text: 'decode(input_embeds: R.Tensor((1, 3072), "float16"), cos: (1,1,128), sin: (1,1,128),' }],
      '           k_cache: (1, 8, 128, 9), v_cache: (1, 8, 9, 128), mask: (1, 1, 10),',
      '           packed_params: R.Tuple(11 tensors)) -> (logits (1,128256), K_new (1,8,128,1), V_new (1,8,1,128)):',
      [{ text: '    R.func_attr({"npu.cache_append": \'{"entries": [{"axis": 3, "output": 1, "parameter": "k_cache"}, ...]}\', "num_input": 6})', color: C.codeStr }],
      '    rms_norm6 = R.nn.rms_norm(input_embeds, layers_0_input_layernorm_weight2, axes=[-1], epsilon=1e-05)',
      [{ text: '    permute_dims28 = R.permute_dims(layers_0_self_attn_q_proj_weight2)      # [3072,3072] Wᵀ → host로 이동 예정', color: C.muted }],
      '    matmul20 = R.matmul(rms_norm6, permute_dims28)            # [1,3072]  Q',
      '    reshape14 = R.reshape(matmul20, [1, 24, 128])',
      '    mul12 = R.multiply(reshape14, cos);  split4 = R.split(reshape14, [64], axis=-1)',
      '    concat6 = R.concat((R.negative(split4[1]), split4[0]), axis=-1)   # rotate_half',
      '    add10 = R.add(mul12, R.multiply(concat6, sin))            # RoPE(Q)  (K도 동일)',
      '    concat8 = R.concat((reshape12 /*k_cache [8,128,9]*/, permute_dims31 /*[8,128,1]*/), axis=2)   # [8,128,10]',
      '    repeat4 = R.repeat(concat8, repeats=3, axis=0)            # [24,128,10]  GQA',
      '    matmul23 = R.matmul(permute_dims33 /*[24,1,128]*/, repeat4)                 # scores [24,1,10]',
      '    add12 = R.add(R.multiply(matmul23, R.const(0.08837890625, "float16")), mask)  # scale + mask',
      '    softmax2 = R.nn.softmax(add12, axis=-1);  matmul24 = R.matmul(softmax2, repeat5)   # PV [24,1,128]',
      '    matmul25 = R.matmul(reshape17, permute_dims35);  add13 = R.add(input_embeds, matmul25)   # O proj + residual',
      '    rms_norm7 → matmul26 (gate) → silu2 → mul17 = silu2 * matmul27 (up) → matmul28 (down) → add14 = add13 + matmul28',
      '    rms_norm8 = R.nn.rms_norm(add14, norm_weight2);  matmul29 = R.matmul(rms_norm8, permute_dims39)   # LM head [1,128256]',
      '    gv2 = (matmul29, concat10 /*K_new*/, concat11 /*V_new*/)',
      [{ text: '    return ', color: C.codeKey }, { text: 'gv2' }],
    ], { size: 8 });
    D.bullets(s, 9.0, 1.5, 3.83, 5.3, [
      ['num_input = 6: ', 'input_embeds, cos, sin, k_cache, v_cache, mask가 실행 입력. packed_params는 parameter.'],
      ['permute_dims(weight): ', 'parameter에만 의존 → 3장에서 host 함수로 분리. token마다 반복하지 않음.'],
      ['reshape/split/concat: ', '일부는 view(zero-copy)로 처리되고 일부는 복사. 실제 여부는 lowering이 결정 (npu_views).'],
      ['scale 상수: ', '1/√128 = 0.0884를 FP16 상수로 곱함. Gemma는 query에 scale을 접어 곱셈이 없음.'],
      ['mask [1,1,10]: ', 'capacity 9 + 현재 token 1. 유효 길이 밖 slot은 -65504 (FP16 floor).'],
      ['출력 tuple: ', '3개 leaf. plan.outputs가 leaf별 shape/dtype/offset을 기록하고 runtime이 그 주소에서 읽음.'],
    ], { size: 9.5, psa: 6 });
  });

  // ---------------------------------------------------------------- 13 small example
  D.add({ section: '2 · 모델 → Relax', title: '작은 예제 ①: SiLU(X @ Wᵀ + bias) — 끝까지 따라갈 4개 연산', subtitle: 'X[64,128], checkpoint weight[64,128], bias[64]. 64×64 tile로 보면 X는 1×2, Wᵀ는 2×1, 출력은 1 tile입니다. bias는 broadcast와 fusion을 설명하기 위해 넣었고 실제 Llama projection에는 없습니다.', source: 'docs/examples/pipeline/01_authored_relax.py, docs/current/02_model_relax.md',
    notes: '이 예제의 authored Relax는 함수 하나에 permute_dims, matmul, add, silu 네 연산이 있습니다. num_input=1이므로 x만 실행 입력이고 weight/bias는 parameter입니다. tile 관점에서 X의 두 tile(K=0..63, 64..127)과 Wᵀ의 두 tile이 같은 출력 tile P에 누적되고, bias는 P의 모든 행에 같은 64개 값을 더하며, SiLU는 원소별입니다. 같은 예제의 host transform, semantic TIR, scheduled TIR, ISA packet, 실제 실행 출력이 docs/examples/pipeline에 모두 남아 있어 이후 슬라이드가 이 파일들을 인용합니다.' },
  (s, D) => {
    D.code(s, 0.5, 1.5, 7.2, 2.15, [
      [{ text: '@R.function', color: C.codeKey }],
      [{ text: 'def ', color: C.codeKey }, { text: 'prefill(x: R.Tensor((64, 128), "float16"), weight: R.Tensor((64, 128), "float16"),' }],
      '            bias: R.Tensor((64,), "float16")) -> R.Tensor((64, 64), "float16"):',
      [{ text: '    R.func_attr({"num_input": 1})      # x만 실행 입력', color: C.codeStr }],
      '    with R.dataflow():',
      '        lv:  (128, 64) = R.permute_dims(weight, axes=[1, 0])   # B = Wᵀ',
      '        lv1: (64, 64)  = R.matmul(x, lv, out_dtype="void")     # P = X @ B',
      '        lv2: (64, 64)  = R.add(lv1, bias)                      # A = P + bias (행마다 broadcast)',
      '        gv:  (64, 64)  = R.nn.silu(lv2)                        # Y = SiLU(A)',
      '        R.output(gv)',
    ], { size: 8.5 });
    // tile-level figure
    const y = 4.3;
    const X = D.matrix(s, 0.5, y, 1.6, 0.8, 1, 2, { kind: 'host', tile: 1, title: 'X [64,128] = 1×2 tiles', regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'sram', label: 'X₀ (k 0..63)' }, { r0: 0, r1: 1, c0: 1, c1: 2, kind: 'ir', label: 'X₁ (k 64..127)' }], below: '실행 입력 (token마다)' });
    D.text(s, 2.15, y + 0.2, 0.3, 0.4, '@', { size: 16, bold: true, align: 'center', color: C.navy });
    const Wt = D.matrix(s, 2.5, y - 0.4, 0.8, 1.6, 2, 1, { kind: 'dram', tile: 1, title: 'Wᵀ [128,64]', regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'sram', label: 'B₀' }, { r0: 1, r1: 2, c0: 0, c1: 1, kind: 'ir', label: 'B₁' }], below: 'host transpose\n(parameter)' });
    D.text(s, 3.35, y + 0.2, 0.3, 0.4, '=', { size: 16, bold: true, align: 'center', color: C.navy });
    const P = D.matrix(s, 3.7, y, 0.8, 0.8, 1, 1, { kind: 'unit', title: 'P [64,64]', regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'unit', label: 'X₀B₀\n+X₁B₁', size: 7.5 }], below: 'K=128: 2 M_MUL\n(MAC 0, 1) → FP16' });
    D.arrow(s, P.r + 0.05, P.cy, P.r + 0.45, P.cy, { width: 1.5 });
    D.text(s, P.r + 0.05, P.cy - 0.32, 0.5, 0.2, '+ bias', { size: 7.5, color: C.muted, align: 'center' });
    // bias lanes broadcast
    const bl = D.lanes(s, 4.75, y - 0.55, 1.2, 0.22, Array.from({ length: 8 }, (_, i) => ({ label: i < 7 ? `b${i}` : '…b63', kind: 'dram', size: 6.5 })), {});
    D.text(s, 4.75, y - 0.8, 1.6, 0.22, 'bias [64]', { size: 8.5, bold: true, color: C.navy });
    [0.15, 0.6, 1.05].forEach(dx => D.arrow(s, 4.75 + dx, y - 0.33, 4.75 + dx, y + 0.02, { width: 0.75, color: C.dram }));
    const A = D.matrix(s, 4.75, y, 0.8, 0.8, 1, 1, { kind: 'ir', regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'ir', label: 'A = P + b', size: 7.5 }], below: '모든 행에 같은\n64개 값 (broadcast)' });
    D.arrow(s, A.r + 0.05, A.cy, A.r + 0.55, A.cy, { width: 1.5 });
    D.text(s, A.r + 0.02, A.cy - 0.32, 0.6, 0.2, 'SiLU', { size: 7.5, color: C.muted, align: 'center' });
    const Y = D.matrix(s, 5.9, y, 0.8, 0.8, 1, 1, { kind: 'good', regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'good', label: 'Y = A·σ(A)', size: 7.5 }], below: '원소별 (vector unit)\n출력 [64,64]' });
    D.text(s, 0.5, 5.95, 7.2, 0.45, 'tile = 64×64 (matrix unit 단위). 같은 출력 tile P에 X₀B₀와 X₁B₁을 누적하고(6장), bias/SiLU는 vector 명령으로 이어집니다(7장). 6장부터 이 그림의 tile이 SRAM 주소와 명령으로 바뀝니다.', { size: 8.5, color: C.muted });
    D.table(s, 8.0, 1.5, 4.83, [
      ['단계', '이 예제에서 볼 파일'],
      ['authored Relax', '01_authored_relax.py'],
      ['graph prefix (host 분리)', '02_graph_prefix.py, 05_host_transform.py'],
      ['semantic TIR (unfused)', '03_unfused_semantic_tir.py'],
      ['선택된 fusion', '04_selected_semantic_tir.py'],
      ['기본 DRAM plan', '06_initial_dram_plan.py'],
      ['scheduled TIR (실제 recipe)', '07_scheduled_fused_matmul_add_silu.py'],
      ['LLVM reference (K=64 rfactor)', '08_llvm_reference.py'],
      ['ISA packets / bundle', '09_packets.json, bundle/program.json'],
      ['입출력 · 검증', 'inputs.npz, outputs.npz, stage_summary.json'],
    ], { colW: [2.0, 2.83], size: 9, rowH: 0.34 });
    D.callout(s, 8.0, 5.2, 4.83, 1.1, [
      { text: '검증 수준: ', bold: true }, { text: '이번 문서 작업에서 fusion off/auto를 실제 C-model로 실행. 두 구성의 FP16 출력이 bit-exact이고 같은 부분합 grouping의 LLVM 출력과 최대 절대차 0.', size: 10 },
    ], { kind: 'good' });
    D.text(s, 0.5, 6.55, 7.2, 0.35, '작은 예제 ②(두 matmul X[64,64]·W1[64,128]·W2[128,64])는 8장 메모리 계획에서, 실제 Llama는 각 장의 "실제 폭" 기록에서 사용합니다.', { size: 9, color: C.muted, italic: true });
  });

  // ---------------------------------------------------------------- 14 relax glossary
  D.add({ section: '2 · 모델 → Relax', title: 'Relax를 읽는 데 필요한 최소 용어', subtitle: 'Dataflow block은 순수 데이터 의존성을 표현할 뿐 하드웨어 동시 실행을 약속하지 않습니다. 실행 순서와 storage 재사용은 후속 단계가 정합니다.', source: 'docs/current/02_model_relax.md, reference/tvm_api.md',
    notes: 'IRModule 하나에 prefill, host transform 함수, 그리고 나중의 TIR PrimFunc들이 함께 있습니다. TensorStructInfo 덕분에 실제 배열 없이도 shape/dtype 추론이 가능합니다. GlobalVar는 모듈 안의 다른 함수를 참조하며 call_tir가 PrimFunc를 호출할 때 사용됩니다. BlockBuilder는 작은 예제나 legalizer가 graph를 만들 때 쓰는 도구입니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 7.6, [
      ['표현', '역할', '작은 예제에서'],
      ['IRModule', '이름 있는 함수들의 집합', 'prefill, prefill_transform_params, 나중의 matmul/add/silu PrimFunc'],
      ['relax.Function', 'tensor 계산의 호출 관계', 'transpose → matmul → add → SiLU'],
      ['TensorStructInfo', 'shape와 dtype 정보 (값 없이 추론)', 'x는 [64,128] FP16, lv1은 [64,64]'],
      ['Var / binding', '값에 이름 부여', 'lv1 = matmul 결과'],
      ['DataflowBlock', '순수 tensor 계산 영역', '분석·변환 범위 (with R.dataflow())'],
      ['num_input', '실행 입력 / parameter 경계 attribute', '1 → x만 입력, weight·bias는 parameter'],
      ['call_tir / GlobalVar', 'TIR 함수 호출 + 결과 sinfo', 'R.call_tir(cls.matmul, (x, lv), out_sinfo=…)'],
      ['BlockBuilder / call_te', 'type·shape를 가진 graph 구성 · TE→PrimFunc', 'legalizer가 matmul TE 정의를 PrimFunc로 변환'],
      ['structural_hash / equal', 'IR 구조 동일성', '후보 dedupe, recipe/entry cache 키'],
    ], { colW: [1.7, 2.6, 3.3], size: 9.5, rowH: 0.4 });
    D.box(s, 8.4, 1.55, 4.43, 0.5, '두 frontend 경로', { kind: 'navy', size: 12 });
    D.bullets(s, 8.4, 2.2, 4.43, 2.3, [
      ['authored (본문): ', 'nn_models/llama.py, gemma.py, qwen3.py. 정적 S/C로 export. generation CLI는 Llama 대상.'],
      ['HF importer (prefill만): ', 'torch.export → HFImporter. weight·position을 상수로 fold, RMSNorm 패턴 복원, FP32 구간을 FP16으로 demote(정밀도 보존 아님), unit batch 축 제거.'],
      ['공통: ', '임의 동적 control flow나 모든 HF 모델을 자동 지원하는 compiler가 아님. 지원 subset을 검사하고 밖이면 거절.'],
    ], { size: 9.5, psa: 5 });
    D.callout(s, 8.4, 4.7, 4.43, 1.6, [
      { text: '28층 export', bold: true, breakLine: true },
      { text: '호출 그래프에 모든 층의 계산이 나타납니다. 같은 shape의 계산은 schedule 판단(recipe)을 공유하지만, layer마다 다른 weight 주소와 residual 관계는 최종 프로그램에 그대로 반영됩니다.', size: 10 },
    ], { kind: 'faint' });
  });

  // ---------------------------------------------------------------- 15 graph prefix
  D.add({ section: '3 · Graph 정리와 host', title: 'Graph prefix: 후속 lowering이 이해하기 쉬운 graph로 만드는 5개 pass', subtitle: '한 번만 적용됩니다(prepared=True). 1번만 NPU 전용이고 나머지는 TVM 표준 pass입니다.', source: 'npu_compiler/tvm_pipeline.py graph_prefix(), docs/current/03_graph_host.md',
    notes: 'graph_prefix는 CanonicalizeAttention, CanonicalizeBindings, EliminateCommonSubexpr, RewriteDataflowReshape, LiftTransformParams 순서의 Sequential입니다. 표준 pass를 호출해도 임의의 layout 변환이 모두 zero-copy가 되거나 모든 연산이 장치에 맞게 되는 것은 아니며, 정확한 지원 조건은 이후 단계가 확인합니다. GQA repeat → broadcast matmul(FoldRepeatedGemv), concat producer fusion은 항상 적용되는 prefix가 아니라 비용 비교 대상인 선택적 후보입니다.' },
  (s, D) => {
    const steps = [
      { title: 'CanonicalizeAttention', detail: 'NPU 전용. 고수준 attention op →\nQKᵀ · scale · mask · softmax · PV', kind: 'sram' },
      { title: 'CanonicalizeBindings', detail: 'alias·binding 정리\n같은 값의 추적 단순화', kind: 'ir' },
      { title: 'EliminateCommonSubexpr', detail: '같은 입력의 같은 순수 계산\n결과 공유', kind: 'ir' },
      { title: 'RewriteDataflowReshape', detail: '지원되는 reshape를\n복사 없는 view 표현으로', kind: 'ir' },
      { title: 'LiftTransformParams', detail: 'parameter만 의존하는 계산을\n별도 host 함수로 분리', kind: 'ir' },
    ];
    D.steps(s, 0.5, 1.75, 12.33, 1.05, steps, { size: 10.5, dsize: 8.5 });
    D.code(s, 0.5, 3.1, 6.0, 1.7, [
      [{ text: 'def ', color: C.codeKey }, { text: 'graph_prefix(lift_params=True):        # tvm_pipeline.py' }],
      '    stages = [CanonicalizeAttention(), relax.transform.CanonicalizeBindings(),',
      '              relax.transform.EliminateCommonSubexpr(),',
      '              relax.transform.RewriteDataflowReshape()]',
      '    if lift_params: stages.append(relax.transform.LiftTransformParams())',
      '    return tvm.transform.Sequential(stages, name="npu_graph_prefix")',
    ], { size: 9 });
    D.table(s, 6.75, 3.1, 6.08, [
      ['선택적 후보 (prefix 아님)', '무엇을 바꾸는가', '언제 채택되는가'],
      ['FoldRepeatedGemv', 'matmul(A[24,1,K], repeat(B[8,K,N],3)) → reshape+expand_dims broadcast matmul (m=1, decode 전용)', '비용 비교에서 이길 때'],
      ['matrix_producer 패턴', 'matmul(x, concat(…)) (+repeat/expand_dims)을 하나의 composite로', 'decode KV concat 등, 비용 비교'],
      ['attention / masked_softmax 패턴', 'FuseOpsByPattern으로 score chain 또는 전체 attention 그룹', '5장 fusion 후보'],
    ], { colW: [1.7, 3.0, 1.38], size: 8.5, rowH: 0.45 });
    D.bullets(s, 0.5, 5.0, 6.0, 1.4, [
      '이 단계는 tile 크기, SRAM 주소, DMA 순서를 정하지 않음. Wᵀ를 host에서 준비해도 Wᵀ 전체가 SRAM에 들어가는 것은 아님',
      '사용하지 않는 함수·값은 표준 DeadCodeElimination이 정리',
      'Authoring한 Llama는 이미 세부 연산으로 쓰여 CanonicalizeAttention이 새로 할 일이 없을 수 있음',
    ], { size: 10, psa: 4 });
    D.takeaway(s, 'Prefix의 목적: 실행마다 필요한 계산과 모델 준비 시 한 번만 할 계산을 구분하고, 같은 값을 추적하기 쉬운 형태로 만드는 것.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 16 LiftTransformParams
  D.add({ section: '3 · Graph 정리와 host', title: 'LiftTransformParams: weight transpose는 언제, 어디서 실행되는가', subtitle: '작은 예제의 실제 IR. device 함수에서 permute_dims가 사라지고 반환 tuple의 순서가 (bias, lv)로 바뀝니다. 변환된 Wᵀ는 DRAM image의 정해진 주소에 한 번 놓입니다.', source: 'docs/examples/pipeline/02_graph_prefix.py, 05_host_transform.py, bundle/program.json (memory.addresses), docs/current/03_graph_host.md',
    notes: 'Compile은 host 함수를 만드는 시점이고, 실제 weight 배열을 transpose하는 것은 runtime 준비 단계에서 host 함수를 LLVM VM으로 실행할 때입니다. 그림 아래의 DRAM image는 bundle/program.json의 실제 주소입니다: x 0, bias 16,384, lv(Wᵀ) 16,512, 출력 gv 32,896, literal pool 41,088 (총 41,100 B). Runtime은 원래 순서를 추측하지 않고 변환된 signature(x, bias, lv)에 맞춰 이름·shape·dtype을 결합합니다. Llama LM head weight만 해도 751.5 MiB라 이 변환이 호출마다 들어가면 매우 큰 이동입니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.3, '변환 전 (01_authored_relax)', { size: 11, bold: true, color: C.navy });
    D.code(s, 0.5, 1.8, 6.0, 1.35, [
      'def prefill(x, weight, bias):          # num_input = 1',
      '    lv  = R.permute_dims(weight, [1,0]) # 매 호출마다 transpose?',
      '    lv1 = R.matmul(x, lv)',
      '    lv2 = R.add(lv1, bias)',
      '    gv  = R.nn.silu(lv2)',
    ], { size: 9 });
    D.text(s, 6.83, 1.5, 6, 0.3, '변환 후 (02_graph_prefix) — device 함수', { size: 11, bold: true, color: C.navy });
    D.code(s, 6.83, 1.8, 6.0, 1.35, [
      'def prefill(x, bias, lv: (128,64)):    # num_input = 1',
      '    lv1 = R.matmul(x, lv)               # transpose 없음',
      '    lv2 = R.add(lv1, bias)',
      '    gv  = R.nn.silu(lv2)',
      '    # parameter 순서: (bias, lv) ← 원래 (weight, bias)와 다름',
    ], { size: 9 });
    D.text(s, 6.83, 3.3, 6, 0.3, '변환 후 — host 함수 (runtime 준비 단계에서 1회 실행)', { size: 11, bold: true, color: C.navy });
    D.code(s, 6.83, 3.6, 6.0, 1.5, [
      'def prefill_transform_params(model_params: R.Tuple((64,128), (64,))):',
      '    R.func_attr({"num_input": 0})',
      '    weight = model_params[0]',
      '    lv = R.permute_dims(weight, axes=[1, 0])     # → call_tir(transpose)',
      '    bias = model_params[1]',
      '    return (bias, lv)                            # 변환된 parameter 묶음',
    ], { size: 9 });
    // tile figure: W -> W^T -> DRAM image
    const y = 3.75;
    const W = D.matrix(s, 0.5, y, 1.6, 0.8, 1, 2, { kind: 'dram', tile: 1, title: 'checkpoint weight [64,128]', regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'dram', label: 'w₀' }, { r0: 0, r1: 1, c0: 1, c1: 2, kind: 'dram', label: 'w₁' }], below: '[N,K] 저장' });
    D.arrow(s, W.r + 0.1, W.cy, W.r + 1.1, W.cy, { width: 2, color: C.host });
    D.text(s, W.r + 0.05, W.cy - 0.45, 1.2, 0.4, 'transpose\nhost VM, 1회', { size: 8, color: C.host, bold: true, align: 'center' });
    const Wt = D.matrix(s, 3.3, y - 0.4, 0.8, 1.6, 2, 1, { kind: 'dram', tile: 1, title: 'lv = Wᵀ [128,64]', regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'dram', label: 'w₀ᵀ' }, { r0: 1, r1: 2, c0: 0, c1: 1, kind: 'dram', label: 'w₁ᵀ' }], below: '[K,N] 계산용' });
    D.arrow(s, Wt.r + 0.1, Wt.cy, Wt.r + 0.9, Wt.cy, { width: 2, color: C.dram });
    D.text(s, Wt.r + 0.05, Wt.cy - 0.45, 1.0, 0.4, 'build_image\n주소 16,512', { size: 8, color: C.dram, bold: true, align: 'center' });
    D.box(s, 5.3, y - 0.1, 1.15, 1.0, 'DRAM\nimage', { kind: 'dram', size: 10.5 });
    // DRAM image strip with real addresses
    D.text(s, 0.5, 5.35, 12.33, 0.24, 'DRAM image (bundle/program.json 실제 주소, bytes): 변환된 signature 순서 (x, bias, lv)로 배치 · 총 41,100 B', { size: 9.5, bold: true, color: C.navy });
    D.membar(s, 0.5, 5.62, 12.33, 0.5, 0, 41100, [
      { from: 0, to: 16384, label: 'x [64,128]  16,384 B', sub: '실행 입력 (token마다 patch)', kind: 'host' },
      { from: 16384, to: 16512, label: '', sub: '', kind: 'dram' },
      { from: 16512, to: 32896, label: 'lv = Wᵀ [128,64]  16,384 B', sub: 'host transform 결과 (1회)', kind: 'dram' },
      { from: 32896, to: 41088, label: 'gv 출력 [64,64]  8,192 B', sub: 'plan.outputs.offset_bytes', kind: 'unit' },
      { from: 41088, to: 41100, label: '', sub: '', kind: 'faint' },
    ], { ticks: [0, 16384, 32896, 41088], size: 8.5 });
    D.text(s, 3.9, 6.32, 3.4, 0.2, '↑ bias [64] 128 B @ 16,384 – 16,512', { size: 7.5, color: C.dram, bold: true });
    D.text(s, 10.3, 6.32, 2.5, 0.2, 'literal pool 12 B @ 41,088 ↑', { size: 7.5, color: C.muted, bold: true, align: 'right' });
  });

  // ---------------------------------------------------------------- 17 CanonicalizeAttention
  D.add({ section: '3 · Graph 정리와 host', title: 'CanonicalizeAttention: attention을 세부 연산으로 드러내는 이유', subtitle: '고수준 relax.nn.attention을 그대로 LegalizeOps에 넘기면 표준 attention legalizer가 자체 TE 계산을 만들어, 우리가 등록한 matmul·softmax의 NPU 정의를 거치지 않습니다.', source: 'npu_compiler/npu_attention.py, docs/examples/attention_canonicalization/comparison.json',
    notes: 'CanonicalizeAttention은 relax.nn.attention / attention_bias만 대상으로 하며 정적 FP16 rank-4 BSNH 입력을 요구합니다. Q를 [b·h,s,d], K를 미리 transpose한 [b·h,d,t], V를 [b·h,t,dv]로 바꾸고 scale을 FP16 상수로 곱하거나 √d로 나눕니다. causal(TopLeft/BottomRight)과 window는 tril/triu로 max 전과 exp 후에 적용해 masked 원소를 정확히 0으로 만듭니다(그림의 회색 삼각형). 결과는 BSNH로 되돌립니다. 설치된 TVM 0.19의 DecomposeOpsForInference는 이 attention 예제를 그대로 두었다는 IR 비교 기록이 있어, 표준 pass 이름이 비슷하다고 같은 변환이라 판단하지 않습니다.' },
  (s, D) => {
    const top = D.box(s, 0.5, 1.55, 5.2, 0.62, 'relax.nn.attention(Q, K, V, bias?, scale?, causal?, window?)', { kind: 'host', size: 10, detail: '입력 BSNH [b, s, h, d] FP16 정적 shape', dsize: 8 });
    D.arrow(s, top.cx, top.b, top.cx, 2.65, { width: 1.5 });
    D.text(s, 0.5, 2.4, 5.2, 0.22, '드러난 계산 (s = t = 4, d = 2인 작은 예; 회색 = causal로 정확히 0)', { size: 8, color: C.muted });
    const y = 2.95;
    const Q = D.matrix(s, 0.5, y, 0.7, 1.1, 4, 2, { kind: 'sram', title: 'Q [b·h, s, d]', titleW: 1.2, cellLabel: () => '' });
    D.text(s, 1.25, y + 0.35, 0.25, 0.4, '×', { size: 14, bold: true, align: 'center', color: C.navy });
    const K = D.matrix(s, 1.55, y + 0.25, 1.1, 0.6, 2, 4, { kind: 'sram', title: 'Kᵀ [b·h, d, t]', titleW: 1.4, below: '미리 transpose', belowW: 1.4 });
    D.text(s, 2.7, y + 0.35, 0.25, 0.4, '=', { size: 14, bold: true, align: 'center', color: C.navy });
    const masked = [];
    for (let r = 0; r < 4; r++) for (let c = r + 1; c < 4; c++) masked.push({ r0: r, r1: r + 1, c0: c, c1: c + 1, kind: 'faint', label: '0', size: 6.5, color: C.muted, lineWidth: 0.4 });
    const S = D.matrix(s, 3.0, y, 1.1, 1.1, 4, 4, { kind: 'unit', title: 'scores [s, t]', titleW: 1.4, regions: masked, below: '×scale, +bias, tril/triu', belowW: 1.6 });
    D.arrow(s, S.r + 0.05, S.cy, S.r + 0.5, S.cy, { width: 1.5 });
    D.text(s, S.r, S.cy - 0.42, 0.6, 0.4, 'softmax\n(행)', { size: 7, color: C.muted, align: 'center' });
    const V = D.matrix(s, 4.65, y, 0.55, 1.1, 4, 2, { kind: 'sram', title: 'V [t, dv]', titleW: 1.0 });
    D.text(s, 5.22, y + 0.35, 0.2, 0.4, '→', { size: 12, bold: true, align: 'center', color: C.navy });
    D.matrix(s, 5.35, y, 0.35, 1.1, 4, 1, { kind: 'good', title: 'out', titleW: 0.6, below: '→ BSNH', belowW: 0.8 });
    D.bullets(s, 0.5, 4.75, 5.2, 1.65, [
      '단순히 연산 이름을 바꾸는 pass가 아님: layout·scale·bias·causal 정렬·window 의미를 유지해야 함',
      '분해 후 각 연산에 NPU 수치 계약 적용 → scale+mask+softmax만 묶을지, 전체를 묶을지 5장에서 비용으로 비교',
      '"분해했으므로 영원히 분리 실행"이 아님. FuseOpsByPattern이 다시 합칠 수 있음. window without causal은 LinkError',
      'Authoring한 Llama는 이미 matmul/softmax를 쓰므로 이 pass가 할 일이 없을 수 있음',
    ], { size: 9, psa: 3 });
    D.text(s, 6.0, 1.55, 6.8, 0.3, 'IR 비교 기록 (examples/attention_canonicalization, [B,S,H,D]=[1,4,2,8])', { size: 11, bold: true, color: C.navy });
    D.table(s, 6.0, 1.9, 6.83, [
      ['경로', '결과 IR의 특징'],
      ['01 입력', 'R.nn.attention(q, k, v) 한 호출'],
      ['02 TVM DecomposeOpsForInference', 'attention을 그대로 둠 (이 예제에서 분해하지 않음)'],
      ['03 NPU CanonicalizeAttention', 'permute_dims → matmul → multiply(scale) → softmax(max/exp/sum/divide) → matmul → permute_dims'],
      ['04 표준 attention legalize', 'attention legalizer의 자체 TE: NPU matmul/softmax 정의 미사용'],
      ['05 canonical → legalize', 'NPU legalize map의 matmul(FP32 누산), softmax(FP32 sum) PrimFunc 사용'],
    ], { colW: [2.3, 4.53], size: 9, rowH: 0.42 });
    D.callout(s, 6.0, 4.75, 6.83, 1.0, [
      { text: '검증 수준: ', bold: true }, { text: 'IR/compile 검사이며 수치·성능 benchmark가 아닙니다. 표준 attention legalization과 NPU canonicalization 뒤 legalization의 차이는 같은 폴더의 dump에서 확인합니다.', size: 10 },
    ], { kind: 'warn' });
    D.takeaway(s, 'Relax에서 먼저 계산을 드러내야 NPU의 수치 계약(FP32 누산·FP16 경계)과 fusion 후보 비교를 적용할 수 있습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 18 host transform runtime
  D.add({ section: '3 · Graph 정리와 host', title: 'Runtime의 host transform 실행과 parameter cache', subtitle: 'IR로 표현된 변환을 그대로 실행해야 linked program의 image가 기대하는 값과 정확히 같습니다. NumPy로 재구현하면 같은 규칙이 두 곳에 생깁니다.', source: 'npu_compiler/npu_generate.py (_run_transform, _transform, _bind_inputs), npu_link.build_image',
    notes: '_run_transform은 entry_transform_params만 도달 가능한 module을 relax.build(…,"llvm")로 빌드해 VirtualMachine으로 실행합니다. _transform은 결과를 디스크 cache에 저장하는데, 키는 host IR JSON과 모든 weight의 shape·dtype·바이트 내용을 넣은 SHA-256입니다. flock, atomic rename, 배열별 sha256 검증, memmap 로드를 사용합니다. _bind_inputs는 func.params의 앞부분(runtime 입력)을 이름으로 검사하고 나머지를 변환 결과와 zip합니다. build_image는 plan.address×unit_bytes 위치에 각 leaf를 쓰고 const_data, 상수 pool, FP32 1.0을 채웁니다.' },
  (s, D) => {
    const steps = [
      { title: '① reachable host IR', detail: 'reachable_module(lowered,\n[entry+"_transform_params"])', kind: 'ir' },
      { title: '② relax.build → LLVM VM', detail: 'vm[entry_transform_params]([weights])\n→ numpy 배열 목록', kind: 'host' },
      { title: '③ params-cache', detail: 'sha256(host IR json + 각 weight의\nshape/dtype/bytes) → 디렉터리', kind: 'dram' },
      { title: '④ _bind_inputs', detail: 'runtime 입력 이름 검사 +\n나머지 param을 변환 결과와 zip', kind: 'host' },
      { title: '⑤ build_image', detail: 'plan.address × unit_bytes에 leaf 기록\nconst_data · literal pool · FP32 1.0', kind: 'dram' },
    ];
    D.steps(s, 0.5, 1.75, 12.33, 1.0, steps, { size: 10, dsize: 8, numbers: false });
    D.code(s, 0.5, 3.05, 6.4, 2.2, [
      'def _run_transform(lowered, entry, weights):',
      '    host = _host_transform_module(lowered, entry)',
      '    vm = relax.VirtualMachine(relax.build(host, "llvm"), tvm.cpu())',
      '    return [t.numpy() for t in vm[entry + "_transform_params"]([weights])]',
      '',
      'def _bind_inputs(func, inputs, transformed):',
      '    runtime_count = len(func.params) - len(transformed)',
      '    names = [p.name_hint for p in func.params[:runtime_count]]',
      '    if set(inputs) != set(names): raise ValueError(...)',
      '    return {**inputs, **dict(zip(names_of_rest, transformed))}',
    ], { size: 8.5 });
    D.table(s, 7.15, 3.05, 5.68, [
      ['cache 안전장치', '이유'],
      ['키에 weight 내용 포함', '파일명만 같은 다른 checkpoint 재사용 방지'],
      ['키에 host IR 포함', '변환 본문·양자화 모드가 바뀌면 miss'],
      ['flock + COMPLETE marker', '동시 실행 시 반쯤 쓴 cache 사용 방지'],
      ['tmp → os.replace', '원자적 commit'],
      ['배열별 sha256 재검증', '손상된 .npy 감지 ("cache is corrupt")'],
      ['np.load(mmap_mode="r")', 'build_image가 복사만 하므로 상주 불필요'],
    ], { colW: [2.3, 3.38], size: 9, rowH: 0.3 });
    D.callout(s, 0.5, 5.5, 12.33, 0.85, [
      { text: '준비 비용과 실행 비용의 분리: ', bold: true }, { text: '변환·image 구성은 모델을 준비할 때의 host 작업이고, token마다 실행되는 것은 NPU 계산과 작은 입력 patch(decode step당 6,676 bytes)뿐입니다. 이 둘을 한 표에 더해 "NPU bandwidth"로 보고하지 않습니다.', size: 10.5 },
    ], { kind: 'faint' });
  });
};
