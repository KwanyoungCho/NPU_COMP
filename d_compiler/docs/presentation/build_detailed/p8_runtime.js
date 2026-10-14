'use strict';
const { C, F } = require('./lib');

module.exports = (D) => {
  // ---------------------------------------------------------------- 62 generate() flow
  D.add({ section: '9 · ISA · linker · runtime', title: 'Runtime 전체 흐름: generate()는 프로그램 2개를 한 번 링크하고 입력만 바꿔 반복한다', subtitle: '배포된 모델 = prompt 길이 S의 prefill_cache 1개 + 고정 capacity C의 decode 1개. 장치에 동적 shape가 없으므로 decode는 항상 C+1 슬롯을 attention하고 mask가 현재 길이를 전달합니다.', source: 'npu_compiler/npu_generate.py generate(), run_tuple_program, DecodeSession; d_compiler/run_nn_generate.py',
    notes: 'generate(family, config, assets, token_ids, steps, capacity)는 capacity(기본 S+steps−1)마다 family.build_generate로 세 entry를 export하고 _lower(lower_and_select, entries=("prefill_cache","decode"))로 링크합니다. weight는 한 번만 읽고 bucket 사이 parameter ABI가 같아야 합니다. prefill은 run_tuple_program(host transform → build_image → v09_runtime.run → read_outputs)이고 첫 token은 logits[-1]의 FP32 argmax입니다. host가 K [L,kv,hd,C], V [L,kv,C,hd] 배열을 만들어 prefill K/V S행을 복사하고, decode loop는 DecodeSession으로 매 step embedding·cos/sin·mask만 patch합니다. 매 step 새로 링크하지 않습니다.' },
  (s, D) => {
    const y = 1.6;
    const st = [['build_generate', 'prefill / prefill_cache / decode\nexport + npu.cache_append 선언', 'ir'], ['lower_and_select', 'entries = (prefill_cache, decode)\n후보 비교 → 프로그램 2개', 'sram'], ['load_params (1회)', 'checkpoint → FP16 배열\nhost transform + params-cache', 'dram'], ['prefill 실행', 'run_tuple_program\n→ logits, K_rows, V_rows', 'unit'], ['decode loop', 'DecodeSession.run(step 입력)\n→ logits, K_new, V_new', 'unit']];
    const w = (12.33 - 4 * 0.2) / 5;
    st.forEach((b, i) => {
      const x = 0.5 + i * (w + 0.2);
      D.box(s, x, y, w, 0.95, b[0], { kind: b[2], size: 10.5, detail: b[1], dsize: 8 });
      if (i < 4) D.arrow(s, x + w + 0.02, y + 0.47, x + w + 0.18, y + 0.47, { width: 1.2 });
    });
    D.code(s, 0.5, 2.85, 7.6, 3.5, [
      'def generate(family, config, assets, token_ids, steps, capacity=None, ...):',
      '    capacity = seq + steps - 1 if capacity is None else capacity',
      '    for cap in used:                                   # bucket별 (기본 1개)',
      '        mod, params, cfg = family.build_generate(config, seq, cap, last_logits=True)',
      '        weights = load once; ABI signature must match across buckets',
      '        modules[cap] = _lower(mod, fusion, compiled=programs[cap],',
      '                              entries=("prefill_cache", "decode"))',
      '    inputs = family.runtime_inputs(assets, cfg, tokens)      # embeds, mask',
      '    cos, sin = family.rope_inputs(cfg, np.arange(seq))',
      '    (logits, k_rows, v_rows) = execute(cap, "prefill_cache", {...})',
      '    generated = [argmax(logits[-1].astype(float32))]        # host greedy',
      '    k_cache[:, :, :, :seq] = k_rows;  v_cache[:, :, :seq, :] = v_rows',
      '    for step in range(1, steps):',
      '        position = seq + step - 1',
      '        step_inputs = {input_embeds: embedding([last]), cos, sin: rope_inputs([position]),',
      '                       mask: length_mask(cap, position)}    # + cache (첫 step만)',
      '        (logits, k_new, v_new), words, decode = execute(cap, "decode", step_inputs, decode)',
      '        generated.append(argmax(logits[-1]))',
    ], { size: 8 });
    D.table(s, 8.35, 2.85, 4.48, [
      ['runner', '무엇을 실행하나'],
      ['run_stateful_tuple_program (기본)', 'DecodeSession: persistent image + DMA append tail'],
      ['run_tuple_program', '호출마다 image 전체 재구성 (cache를 host가 전달)'],
      ['llvm_runner', '같은 프로그램을 tile_order_reference로 CPU 실행 (step별 cross-check)'],
    ], { colW: [1.9, 2.58], size: 8.5, rowH: 0.5 });
    D.callout(s, 8.35, 4.9, 4.48, 1.45, [
      { text: '프로그램 수 ≠ layer 수 ≠ 후보 수: ', bold: true }, { text: '기본은 S에 대한 prefill 1개와 C에 대한 decode 1개. 28층은 하나의 entry stream 안에 모두 전개되어 있습니다.', size: 10 },
    ], { kind: 'faint' });
    D.takeaway(s, 'Greedy argmax는 host에서 합니다. 마지막으로 예측한 token은 아직 NPU 입력으로 처리되지 않았으므로 생성 token 수와 decode 실행 횟수가 한 번 차이 납니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 63 prefill vs decode
  D.add({ section: '9 · ISA · linker · runtime', title: 'Prefill 프로그램과 decode 프로그램: host와 NPU가 나누어 맡는 일', subtitle: 'Prefill은 prompt S개 token을 한 번에, decode는 새 token 하나와 과거 cache를 입력으로 다음 logits를 만듭니다.', source: 'docs/current/10_runtime_llama.md, npu_compiler/nn_models/llama.py (runtime_inputs, causal_mask), npu_generate.length_mask',
    notes: 'host: prompt token → embedding 행(assets.embedding은 전체 table을 올리지 않고 행만 읽음), cos/sin, causal mask [1,S,S] (−1e4, 과거 per-head mask는 S=2048에서 201 MB 전송·192 MiB image였음). NPU: 28 layers → 마지막 hidden의 norm/LM head → logits [1,V] + K/V. host: argmax → 첫 생성 token. 반복: 그 token의 embedding, 현재 position의 cos/sin, valid-length mask [1,1,C+1] (−65504) → NPU decode + cache append → argmax. Persistent cache shape는 K [28,8,128,C], V [28,8,C,128]입니다.' },
  (s, D) => {
    const y = 1.6;
    const p1 = D.box(s, 0.5, y, 2.7, 1.1, 'host: prompt 준비', { kind: 'host', size: 11, detail: 'token → embedding [S,3072]\ncos/sin [S,1,128], causal mask [1,S,S]', dsize: 8 });
    const p2 = D.box(s, 3.6, y, 2.9, 1.1, 'NPU: prefill_cache', { kind: 'unit', size: 11, detail: '28 layers → last hidden → norm → LM head\n→ logits [1,V], K [28,8,128,S], V [28,8,S,128]', dsize: 8 });
    const p3 = D.box(s, 6.9, y, 2.4, 1.1, 'host: 첫 token', { kind: 'host', size: 11, detail: 'argmax(logits) → token₀\ncache 배열 [.., C] 에 S행 복사', dsize: 8 });
    D.arrow(s, p1.r, p1.cy, p2.x, p2.cy); D.arrow(s, p2.r, p2.cy, p3.x, p3.cy);
    const y2 = y + 1.5;
    const d1 = D.box(s, 0.5, y2, 2.7, 1.1, 'host: step 입력', { kind: 'host', size: 11, detail: 'embedding [1,3072] of token\ncos/sin(position), length_mask [1,1,C+1]', dsize: 8 });
    const d2 = D.box(s, 3.6, y2, 2.9, 1.1, 'NPU: decode + append', { kind: 'unit', size: 11, detail: 'cache C + 현재 1 → attention C+1\n→ logits, K_new, V_new; DMA tail → slot L', dsize: 8 });
    const d3 = D.box(s, 6.9, y2, 2.4, 1.1, 'host: 다음 token', { kind: 'host', size: 11, detail: 'argmax → token₁\n다음 step 입력으로', dsize: 8 });
    D.arrow(s, d1.r, d1.cy, d2.x, d2.cy); D.arrow(s, d2.r, d2.cy, d3.x, d3.cy);
    D.line(s, p3.cx, p3.b, p3.cx, y2 - 0.2, { arrow: false }); D.line(s, p3.cx, y2 - 0.2, d1.cx, y2 - 0.2, { arrow: false }); D.arrow(s, d1.cx, y2 - 0.2, d1.cx, d1.y);
    D.line(s, d3.cx, d3.b, d3.cx, d3.b + 0.2, { arrow: false, color: C.warn }); D.line(s, d3.cx, d3.b + 0.2, d1.cx, d3.b + 0.2, { arrow: false, color: C.warn }); D.arrow(s, d1.cx, d3.b + 0.2, d1.cx, d1.b, { color: C.warn });
    D.text(s, 3.0, d3.b + 0.22, 3.5, 0.25, '반복 (같은 decode 프로그램, 입력 patch만)', { size: 8.5, color: C.warn, bold: true });
    D.table(s, 9.6, y, 3.23, [
      ['mask', '값'],
      ['prefill causal [1,S,S]', '대각선 위 −1e4'],
      ['decode length [1,1,C+1]', '유효 슬롯 0, 나머지 −65504, 마지막(현재) 0'],
      ['이유', 'FP16 floor; softmax가 exp를 0으로'],
    ], { colW: [1.5, 1.73], size: 8.5, rowH: 0.42 });
    D.table(s, 0.5, 4.6, 12.33, [
      ['', 'Prefill (S=7)', 'Decode (C=9)'],
      ['device 입력', 'input_embeds [7,3072], cos/sin [7,1,128], mask [1,7,7], packed_params', 'input_embeds [1,3072], cos/sin [1,1,128], k_cache [28,8,128,9], v_cache [28,8,9,128], mask [1,1,10], packed_params'],
      ['출력', 'logits [1,128256] (last_logits), K [28,8,128,7], V [28,8,7,128]', 'logits [1,128256], K_new [28,8,128,1], V_new [28,8,1,128]'],
      ['프로그램 통계 (기록)', '8,582,352 words, DMA 6.46 GB, SRAM peak 8.37 MB', '7,310,801 words (+ append tail 13,896 words), DMA 6.44 GB, SRAM peak 7.34 MB'],
    ], { colW: [1.9, 5.0, 5.43], size: 8.8, rowH: 0.4 });
    D.takeaway(s, 'Prefill의 logits는 마지막 hidden 행만 (last_logits). 전체 위치 logits를 반환하는 prefill API와 구분합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 64 fixed shape vs valid length
  D.add({ section: '9 · ISA · linker · runtime', title: '고정 shape와 유효 길이는 다르다: capacity C=9, 유효 길이 L=7', subtitle: 'Decode는 기존 C개 슬롯과 현재 K/V 하나를 논리적으로 연결해 C+1 길이로 attention합니다. 현재 token의 임시 위치는 C, persistent cache에 기록할 위치는 첫 빈 슬롯 L입니다.', source: 'docs/current/10_runtime_llama.md, npu_generate.length_mask, npu_cache.AppendPlan.position',
    notes: '예를 들어 C=9, L=7이면 attention 입력은 [과거 0..6 | 빈 7,8 | 현재 token 9]이고 mask는 [0…0 | −65504 | 0]입니다. 의미상 제외할 슬롯도 shape와 명령에는 남아 있습니다. mask는 기여를 막을 뿐 계산의 동적 생략이 아닙니다. AppendPlan.position은 mask에서 유효 prefix 길이를 복원하며 하나의 유효 prefix + 현재 token 형태가 아니면 거절하고, cache가 가득 차면(L ≥ C) 오류입니다. DecodeSession은 position이 순차적이어야 함을 검사합니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.55, 8, 0.3, '이번 attention이 보는 논리적 입력 (C+1 = 10 슬롯)', { size: 11, bold: true, color: C.navy });
    for (let i = 0; i < 10; i++) {
      const kind = i < 7 ? 'dram' : (i < 9 ? 'faint' : 'unit');
      const x = 0.5 + i * 1.23;
      D.rect(s, x, 1.95, 1.13, 0.9, { kind, radius: 0.05 });
      D.text(s, x, 2.0, 1.13, 0.35, String(i), { size: 14, bold: true, align: 'center' });
      D.text(s, x, 2.38, 1.13, 0.4, i < 7 ? '과거 (유효)' : (i < 9 ? '빈 슬롯' : '현재 token'), { size: 8.5, align: 'center', color: C.muted });
    }
    D.text(s, 0.5, 2.95, 12.33, 0.3, 'mask [1,1,10]:   0   0   0   0   0   0   0   −65504   −65504   0', { size: 11, font: F.code, bold: true, color: C.navy });
    D.text(s, 0.5, 3.3, 12.33, 0.3, 'cache 갱신: 현재 K/V를 persistent 슬롯 L = 7에 기록 → 다음 step의 유효 길이 8', { size: 10.5, color: C.muted });
    D.code(s, 0.5, 3.8, 6.1, 1.55, [
      'def length_mask(capacity, length, dtype=np.float16):',
      '    mask = np.full((1, 1, capacity + 1), -65504.0, dtype=np.float32)',
      '    mask[0, 0, :length] = 0.0        # 유효 prefix',
      '    mask[0, 0, capacity] = 0.0       # 현재 token (임시 슬롯 C)',
      '    return mask.astype(dtype)',
      '# AppendPlan.position(mask): 유효 prefix 길이 L 복원, L >= C 이면 "cache is full"',
    ], { size: 8.5 });
    D.table(s, 6.85, 3.8, 5.98, [
      ['구분', '값 (C=9, L=7)'],
      ['persistent cache shape', 'K [28,8,128,9], V [28,8,9,128] (고정)'],
      ['attention 길이', 'C + 1 = 10 (항상)'],
      ['현재 token의 임시 위치', 'C = 9 (concat 결과의 마지막)'],
      ['기록 위치', 'L = 7 (첫 빈 슬롯), 다음 step L = 8'],
      ['빈 슬롯의 계산', 'shape·명령에 남아 있음. mask가 기여만 막음'],
    ], { colW: [2.0, 3.98], size: 9, rowH: 0.3 });
    D.callout(s, 0.5, 5.55, 12.33, 0.8, [
      { text: 'Prefill → decode 초기화: ', bold: true }, { text: 'host가 prefill K/V를 capacity 크기 배열의 앞부분에 복사합니다. 이후 step에서는 새 K/V가 별도 DRAM 출력으로 나오고 program 끝의 DMA tail이 cache의 올바른 슬롯으로 옮깁니다. "매 step NPU가 마지막 슬롯에 저장하면 CPU가 다시 옮긴다"는 설명과 다릅니다.', size: 10 },
    ], { kind: 'faint' });
    D.takeaway(s, '큰 음수는 FP16 −65504입니다. 의미상 제외할 슬롯도 계산은 수행되며, capacity 전체 계산이 현재 decode 길이 비용의 한계입니다 (bucket으로 완화).', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 65 KV append
  D.add({ section: '9 · ISA · linker · runtime', title: 'KV append는 누가 실행하는가: npu.cache_append 계약 → AppendPlan → DMA tail', subtitle: 'K cache는 [128, C]로 transposed 저장이라 새 K(128×1)는 열 하나 = 행마다 2 B를 stride C·2 B로 쓰고, V cache는 [C, 128]이라 새 V는 연속 256 B 행 하나입니다. host는 position으로 주소를 정하고 전송은 NPU 명령이 합니다.', source: 'npu_compiler/npu_cache.py (declare, AppendPlan, _template, words), docs/current/10_runtime_llama.md',
    notes: 'declare는 decode 함수에 {"version":1, "mask":"mask", "entries":[{"parameter":"k_cache","output":1,"axis":3}, {"parameter":"v_cache","output":2,"axis":2}]}를 붙입니다. 그림은 layer l, head h 하나의 cache입니다(28 × 8 쌍이 있음). K는 hd × C로 저장되어 slot L에 쓰는 것이 열 하나를 stride C·2 B로 쓰는 2D DMA이고, V는 C × hd라 행 하나를 연속으로 쓰는 1D DMA입니다. 행 bytes가 4의 배수가 아니면(홀수 C 등) cell RMW 경로를 씁니다. AppendPlan은 출력 tuple이 FP16이고 각 출력 shape가 cache shape에서 time 축만 1인지, source/destination이 alias하지 않는지, capacity가 일치하는지, mask가 FP16 [1,1,C+1]인지 검사합니다. parity 0/1 두 template를 미리 만들고 words(position)이 fixup으로 주소를 재배치합니다. 기록: 새 K/V 논리 크기 114,688 B, append DMA 458,752 B, decode base 7,310,801 words, 실행 stream 7,324,697 words.' },
  (s, D) => {
    const y = 1.85, L = 7;
    // K cache [128, C]
    const Kc = D.matrix(s, 0.5, y, 2.7, 1.6, 8, 9, { kind: 'faint', title: 'k_cache[l, h] = [128, C=9]  (열 = slot, transposed 저장)', titleW: 3.4, regions: [
      { r0: 0, r1: 8, c0: 0, c1: 7, kind: 'dram', label: '유효 slot 0 … 6', size: 7 },
      { r0: 0, r1: 8, c0: 7, c1: 8, kind: 'warn', label: 'L=7', size: 7 },
      { r0: 0, r1: 8, c0: 8, c1: 9, kind: 'faint', label: '8', size: 7, color: C.muted },
    ], below: 'append = 128행 × 2 B, stride C×2 = 18 B → 2D DMA (row_bytes % 4 검사; 홀수면 cell RMW)', belowW: 3.4 });
    const Kn = D.matrix(s, 3.75, y, 0.3, 1.6, 8, 1, { kind: 'unit', title: 'K_new\n[128,1]', titleW: 0.7, regions: [{ r0: 0, r1: 8, c0: 0, c1: 1, kind: 'unit' }] });
    D.arrow(s, Kn.x, Kn.cy, Kc.x + Kc.cw * 7.5, Kc.cy, { width: 2, color: C.warn });
    D.text(s, 4.1, y + 0.55, 0.9, 0.5, 'DMA tail\n(별도 DRAM\n출력에서)', { size: 6.5, color: C.warn });
    // V cache [C, 128]
    const Vc = D.matrix(s, 5.0, y - 0.2, 2.4, 2.0, 9, 8, { kind: 'faint', title: 'v_cache[l, h] = [C=9, 128]  (행 = slot)', titleW: 2.6, regions: [
      { r0: 0, r1: 7, c0: 0, c1: 8, kind: 'dram', label: '유효 slot 0 … 6', size: 7 },
      { r0: 7, r1: 8, c0: 0, c1: 8, kind: 'warn', label: 'L=7', size: 7 },
      { r0: 8, r1: 9, c0: 0, c1: 8, kind: 'faint', label: '8', size: 7, color: C.muted },
    ], below: 'append = 연속 256 B 행 1개 → 1D DMA', belowW: 2.6 });
    const Vn = D.matrix(s, 7.85, y + 1.45, 2.4, 0.22, 1, 8, { kind: 'unit', title: 'V_new [1,128] (별도 DRAM 출력 → tail)', titleW: 3.0, regions: [{ r0: 0, r1: 1, c0: 0, c1: 8, kind: 'unit' }] });
    D.arrow(s, Vn.x, Vn.cy, Vc.r, Vc.y + Vc.ch * 7.5, { width: 2, color: C.warn });
    D.bullets(s, 7.85, y - 0.3, 4.98, 1.2, [
      ['① decode 계산: ', 'K_new/V_new를 별도 DRAM 출력에 SAVE→GSTORE'],
      ['② tail (HALT 앞): ', '출력 → SRAM GLOAD → cache[…, L] GSTORE. 주소는 position fixup'],
      ['③ 다음 step: ', '유효 길이 8, mask의 0이 하나 늘어남'],
    ], { size: 8.5, psa: 3 });
    D.code(s, 0.5, 3.95, 6.4, 1.2, [
      'mod["decode"] = npu_cache.declare(mod["decode"],',
      '    [("k_cache", 1, 3), ("v_cache", 2, 2)])   # (param, output idx, time axis)',
      'templates = [_template(parity) for parity in (0, 1)]   # 짝/홀 position',
      'def words(position):',
      '    words, fixups = templates[position % 2]',
      '    for index, coeff in fixups: words[index] += (position // 2) * coeff   # g_addr만 재배치',
    ], { size: 8 });
    D.table(s, 7.15, 3.95, 5.68, [
      ['AppendPlan 검사', '조건'],
      ['출력 tuple', 'FP16, output[i].shape == cache.shape with axis → 1'],
      ['alias', 'source(출력)와 destination(cache)이 겹치지 않음'],
      ['capacity / mask', '모든 slice 동일; mask FP16 [1,1,C+1], position은 mask에서 복원'],
    ], { colW: [1.4, 4.28], size: 8.3, rowH: 0.3 });
    D.table(s, 0.5, 5.5, 12.33, [
      ['기록 (C=9, 28 layers)', 'bytes / words', '뜻'],
      ['host 입력 patch (step당)', '6,676 B', 'embedding 6,144 + cos 256 + sin 256 + mask 20'],
      ['새 K/V 논리 크기', '114,688 B', '2 × 28 × 8 × 128 × 2 B'],
      ['append DMA 전송', '458,752 B', 'layout·정렬·이동 명령 때문에 논리 크기보다 큼'],
      ['decode base / 실행 stream', '7,310,801 / 7,324,697 words', 'tail 13,896 words; 두 번째 decode의 program suffix patch 55,568 B'],
    ], { colW: [2.6, 2.4, 7.33], size: 8.3, rowH: 0.19 });
    D.takeaway(s, 'cache 이름을 추측하거나 layer 수를 codegen에 하드코딩하지 않습니다. metadata의 parameter·output index·time axis로만 append를 만듭니다.', { y: 6.58 });
  });

  // ---------------------------------------------------------------- 66 DecodeSession / transport
  D.add({ section: '9 · ISA · linker · runtime', title: 'DecodeSession과 PersistentTransport: 무엇이 재사용되고 무엇이 바뀌는가', subtitle: '이전 실행의 최종 global image 파일을 다음 입력으로 이어 쓰고, 바뀐 host 입력과 program suffix만 patch합니다. 아래 두 strip은 step마다 실제로 바뀌는 바이트입니다.', source: 'npu_compiler/npu_generate.DecodeSession, v09_runtime.PersistentTransport, docs/current/10_runtime_llama.md',
    notes: 'DecodeSession은 transformed weights, compile_program(residency=True), AppendPlan, build_image, PersistentTransport를 만듭니다. run은 mask에서 position을 얻어 순차성을 검사하고, cache 입력이 초기화 뒤 다시 오면 거절하며, runtime 입력의 shape/dtype을 검사해 plan.address×2 위치의 patch 목록을 만듭니다(6,676 B). tail을 HALT 앞에 넣고 transport.run(words, patches)를 부릅니다. PersistentTransport는 program_memory.bin에서 바뀐 첫 word부터만 다시 쓰고(55,568 B), global_memory.bin에 patch를 적용한 뒤 subprocess를 실행하고 saved_global_memory.bin을 rename으로 다음 입력으로 만듭니다. C-model은 여전히 호출마다 전체 image(≈6.43 GB)를 읽고 씁니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 6.3, [
      ['대상', 'step 사이의 동작'],
      ['기본 decode 계산 명령', '재사용 (prefix 동일)'],
      ['KV append 명령의 주소', 'position에 맞춰 변경 (program suffix patch)'],
      ['embedding, cos/sin, mask', 'host가 새 값으로 patch (6,676 B)'],
      ['DRAM weight', '초기 image 이후 host가 다시 쓰지 않음 (rewrite 0 B)'],
      ['SRAM의 weight 조각', '실행 중 NPU가 DRAM에서 다시 가져옴'],
      ['Persistent DRAM cache', 'saved image → 다음 입력 image (rename)'],
      ['C-model 내부 SRAM', '호출 사이 지속성을 전제하지 않음'],
    ], { colW: [2.2, 4.1], size: 9, rowH: 0.34 });
    // file strips
    D.text(s, 7.05, 1.5, 5.78, 0.24, 'global_memory.bin (≈6.43 GB): step마다 바뀌는 곳', { size: 9.5, bold: true, color: C.navy });
    D.membar(s, 7.05, 1.77, 5.78, 0.5, 0, 100, [
      { from: 0, to: 68, label: 'weight (변환된 parameter)', sub: 'host rewrite 0 B', kind: 'dram' },
      { from: 68, to: 76, label: '입력', sub: 'patch 6,676 B', kind: 'warn' },
      { from: 76, to: 92, label: 'KV cache', sub: 'device tail이 갱신', kind: 'sram' },
      { from: 92, to: 100, label: '출력', sub: 'logits', kind: 'unit' },
    ], { size: 7.5 });
    D.text(s, 7.05, 2.5, 5.78, 0.24, 'program_memory.bin (7,324,697 words): 바뀐 첫 word부터만 다시 씀', { size: 9.5, bold: true, color: C.navy });
    D.membar(s, 7.05, 2.77, 5.78, 0.5, 0, 100, [
      { from: 0, to: 88, label: 'decode base 7,310,801 words', sub: '변경 없음 (재link 없음)', kind: 'ir' },
      { from: 88, to: 99, label: 'tail', sub: '13,896 w, fixup', kind: 'warn' },
      { from: 99, to: 100, label: '', sub: '', kind: 'red' },
    ], { size: 7.5 });
    D.text(s, 7.05, 3.32, 5.78, 0.4, 'suffix patch 55,568 B (두 번째 decode). C-model subprocess는 호출마다 두 파일을 읽고 saved_global_memory.bin(전체 image)을 씀 → rename으로 다음 입력.', { size: 7.5, color: C.muted });
    D.code(s, 7.05, 3.8, 5.78, 1.0, [
      'changed = flatnonzero(program[:common] != self.program[:common])',
      'file.seek(first*4); program[first:].tofile(file)      # program suffix',
      'for offset, data in patches: file.seek(offset); data.tofile(file)',
      'subprocess.run([mysim_v09]); saved.replace(global_path)',
    ], { size: 7.5 });
    D.table(s, 7.05, 4.95, 5.78, [
      ['측정 층', '지표', '섞지 않을 것'],
      ['compiler 정적 통계', 'DMA bytes, words, SRAM peak', '실장치 latency'],
      ['host 전송', 'input patch, program patch bytes', 'NPU bandwidth'],
      ['simulator 파일 I/O', 'image read/write (전체 image)', 'zero-copy runtime'],
    ], { colW: [1.7, 2.3, 1.78], size: 8.3, rowH: 0.3 });
    D.callout(s, 0.5, 4.5, 6.3, 1.85, [
      { text: 'Persistent transport의 실제 뜻: ', bold: true }, { text: 'host weight rewrite를 없애는 transport 재사용입니다. 지속적인 simulator process나 실제 장치의 zero-copy runtime이 아니며, C-model subprocess는 호출마다 전체 image를 읽고 씁니다. NPU_V09_TMPDIR로 큰 image의 임시 위치를 지정합니다.', size: 9.5, breakLine: true },
      { text: ' ', size: 4, breakLine: true },
      { text: '검사: ', bold: true }, { text: 'position 순차성, cache 입력 재전달 거부, 입력 shape/dtype, HALT로 끝나는 프로그램, patch 범위가 image 안인지, 최종 image가 정확히 1개인지.', size: 9.5 },
    ], { kind: 'faint' });
    D.takeaway(s, 'Compiler의 DMA 통계, host patch bytes, simulator 파일 I/O를 한 표에서 더해 "NPU bandwidth"로 보고하지 않습니다. 측정 층이 다릅니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 67 actual 3-token record
  D.add({ section: '9 · ISA · linker · runtime', title: '실제 검증된 3-token 생성: "Hello, NPU compiler!" → " I\'m trying"', subtitle: 'FP16, 28 layers, prompt 7 tokens, capacity 9, greedy 3 tokens. 보존된 기록이며 이번 문서 작업에서 전체 모델을 다시 실행하지 않았습니다.', source: 'docs/evidence/current/llama_validation.json, completion.json (compiler source sha256 44694fc6…)',
    notes: 'tokenizer 결과는 [128000, 9906, 11, 452, 6459, 19979, 0]의 7 tokens입니다. prefill이 358(" I")을 예측하고, decode position 7에서 2846("\'m"), position 8에서 4560(" trying")을 예측합니다. 마지막 4560은 아직 입력으로 처리하지 않았으므로 이 시점 cache에는 그 token의 K/V가 없습니다. HF와 argmax가 모두 일치하고 logits cosine은 0.99999 이상, 최대 절대차는 0.024~0.039입니다. 판정 기준은 유한 logits, token 일치, cosine ≥ 0.999이며 최대 절대차는 기록만 했습니다. 실행 948.99 s는 완료된 첫 entry를 재사용한 검증이므로 cold compile 시간이 아닙니다.' },
  (s, D) => {
    const toks = ['128000', '9906', '11', '452', '6459', '19979', '0'];
    const labels = ['<bos>', 'Hello', ',', ' N', 'PU', ' compiler', '!'];
    D.text(s, 0.5, 1.5, 6, 0.3, 'prompt 7 tokens', { size: 11, bold: true, color: C.navy });
    toks.forEach((t, i) => {
      const x = 0.5 + i * 0.95;
      D.rect(s, x, 1.85, 0.85, 0.7, { kind: 'dram', radius: 0.05 });
      D.text(s, x, 1.88, 0.85, 0.3, labels[i], { size: 8.5, bold: true, align: 'center' });
      D.text(s, x, 2.18, 0.85, 0.3, t, { size: 8, align: 'center', color: C.muted, font: F.code });
    });
    const gen = [['358', ' I', 'prefill'], ['2846', "'m", 'decode @7'], ['4560', ' trying', 'decode @8']];
    gen.forEach((g, i) => {
      const x = 7.4 + i * 1.75;
      D.rect(s, x, 1.85, 1.6, 0.7, { kind: 'unit', radius: 0.05 });
      D.text(s, x, 1.88, 1.6, 0.3, `${g[1]}  (${g[0]})`, { size: 9.5, bold: true, align: 'center' });
      D.text(s, x, 2.18, 1.6, 0.3, g[2], { size: 8, align: 'center', color: C.muted });
      if (i < 2) D.arrow(s, x + 1.62, 2.2, x + 1.73, 2.2, { width: 1 });
    });
    D.arrow(s, 7.2, 2.2, 7.38, 2.2, { width: 1.5 });
    D.table(s, 0.5, 2.85, 12.33, [
      ['단계', 'NPU에 넣는 것', '계산 후 cache 유효 길이', '예측 token', 'HF argmax 일치', 'logits cosine', '최대 절대차'],
      ['prefill_cache', 'prompt 7 tokens', '7', '358 (" I")', '예', '0.9999977148', '0.03515625'],
      ['decode 1', 'token 358, position 7', '8', '2846 ("\'m")', '예', '0.9999958473', '0.02343750'],
      ['decode 2', 'token 2846, position 8', '9', '4560 (" trying")', '예', '0.9999929546', '0.03906250'],
    ], { colW: [1.3, 2.2, 1.9, 1.7, 1.4, 1.9, 1.93], size: 9, rowH: 0.34, aligns: ['left', 'left', 'center', 'left', 'center', 'right', 'right'] });
    D.code(s, 0.5, 4.5, 6.4, 1.0, [
      'python d_compiler/run_nn_generate.py --model llama \\',
      '    --prompt "Hello, NPU compiler!" --tokens 3 --capacity 9',
      '# --llvm: 같은 프로그램을 CPU에서, --cache-buckets, --fusion auto|off|on, --params-cache DIR',
    ], { size: 8.5 });
    D.table(s, 7.15, 4.5, 5.68, [
      ['판정 기준 / 기록', '값'],
      ['판정', '유한 logits, token 일치, cosine ≥ 0.999 (절대차는 기록만)'],
      ['실행 시간', '948.99 s (첫 entry 재사용 검증; cold compile 아님)'],
      ['graph 탐색', '120 s 예산 소진 → unfused 기준 module'],
      ['compiler source', 'sha256 44694fc6…, TVM 0.19.0 c4dc0c29, LLVM 18.1.8'],
    ], { colW: [1.6, 4.08], size: 8.5, rowH: 0.3 });
    D.callout(s, 0.5, 5.65, 6.4, 0.7, [
      { text: '범위: ', bold: true }, { text: '한 prompt의 짧은 greedy 생성입니다. 긴 문맥 품질, 다양한 prompt, 양자화 generation까지 검증됐다고 확대하지 않습니다.', size: 9.5 },
    ], { kind: 'warn' });
    D.takeaway(s, '최종 문자열 " I\'m trying". 마지막 4560은 아직 NPU 입력으로 처리되지 않았으므로 이 시점 cache에는 그 token의 K/V가 없습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 68 buckets, layers, C-model
  D.add({ section: '9 · ISA · linker · runtime', title: 'Bucket, layer 반복, 그리고 C-model 실행 규약', subtitle: '여러 capacity를 미리 컴파일하고 유효 길이가 커질 때 큰 bucket으로 바꿀 수 있습니다. 전환 시 유효 prefix를 한 번 복사합니다. 28층은 최종 stream에 모두 전개되며 schedule 판단만 재사용합니다.', source: 'npu_generate.generate (cache_buckets, transitions), v09_runtime.run, _poc/mysim_v09.cpp 파일 규약',
    notes: 'cache_buckets는 증가하는 capacity 목록이고 마지막이 capacity와 같아야 합니다. bucket(position)은 position보다 큰 첫 bucket이며 실제 필요한 bucket만 컴파일합니다(used). 그림: C=16 프로그램으로 position 15까지 진행하다가 유효 길이가 16이 되면 DecodeSession을 닫고 유효 cache prefix 16 slot을 C=32 배열로 복사해 새 session을 만듭니다. transitions에 valid_cache_bytes/input_cache_bytes를 기록합니다. 작은 context의 낭비(빈 slot 계산)를 줄이는 대신 프로그램 수, 초기 image, 전환 복사가 늘어납니다. C-model 파일 규약: program_memory.bin과 global_memory.bin을 읽고 SNAPSHOT(0xF0)마다 그리고 HALT(0xFF)에서 saved_global_memory.bin에 image를 덧붙이며 perf_counters.txt를 씁니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6.3, 0.28, 'cache_buckets = [16, 32, 64] (capacity 64): slot 사용과 전환', { size: 10.5, bold: true, color: C.navy });
    const slots = (n, valid, cur) => Array.from({ length: n }, (_, i) => ({ label: '', kind: i < valid ? 'dram' : (i === cur ? 'warn' : 'faint') }));
    D.lanes(s, 2.0, 1.85, 1.5, 0.26, slots(16, 12, 12), { title: 'C=16 (pos < 16)', titleW: 1.45, right: '유효 12, 현재 위치 12: 빈 slot 3개만 낭비', rightW: 3.0, rightSize: 7 });
    D.arrow(s, 2.75, 2.15, 2.75, 2.45, { width: 1.2, color: C.warn });
    D.text(s, 2.9, 2.15, 3.6, 0.3, '유효 길이 16 도달 → 전환: prefix 16 slot 복사 (1회), 새 DecodeSession', { size: 7, color: C.warn, bold: true });
    D.lanes(s, 2.0, 2.5, 3.0, 0.26, slots(32, 16, 16), { title: 'C=32 (16 ≤ pos < 32)', titleW: 1.45, right: '복사된 prefix 16 + 빈 slot', rightW: 2.2, rightSize: 7 });
    D.arrow(s, 3.5, 2.8, 3.5, 3.1, { width: 1.2, color: C.warn });
    D.lanes(s, 2.0, 3.15, 6.0, 0.26, slots(64, 32, 32), { title: 'C=64 (32 ≤ pos < 64)', titleW: 1.45, right: '', rightW: 0.1 });
    D.text(s, 2.0, 3.48, 6.3, 0.45, '단일 C=64 프로그램이었다면 position 12에서도 64 slot 전체를 attention 계산(마스크만 0). bucket은 이 낭비를 줄이는 대신 컴파일 프로그램 3개, 초기 image 3개, 전환 복사 2회가 생깁니다. 매 step 새 compiler 실행과는 다릅니다.', { size: 8, color: C.muted });
    D.table(s, 0.5, 4.15, 6.3, [
      ['layer 반복', '내용'],
      ['재사용하는 것', '같은 구조 PrimFunc의 schedule recipe, 영역/entry 비용 cache'],
      ['호출마다 반영하는 것', 'layer별 weight 주소, 입력·출력 값, residual 관계 → 최종 stream에 28층 모두 전개'],
      ['main backend와의 차이', 'main은 host가 같은 layer 프로그램을 weight만 바꿔 반복 실행 (부록 A1)'],
    ], { colW: [1.7, 4.6], size: 8.5, rowH: 0.42 });
    D.text(s, 7.05, 1.5, 6, 0.3, 'C-model (mysim_v09.cpp) 파일 규약', { size: 10.5, bold: true, color: C.navy });
    D.table(s, 7.05, 1.85, 5.78, [
      ['파일', '방향', '내용'],
      ['program_memory.bin', '입력', 'little-endian uint32 명령 words'],
      ['global_memory.bin', '입력', '초기 DRAM image (32-bit cells)'],
      ['saved_global_memory.bin', '출력', 'SNAPSHOT(0xF0)마다 + HALT(0xFF)에서 전체 image 추가'],
      ['perf_counters.txt', '출력', 'words_executed, gload, gstore, dma_cells_loaded/stored, loads, saves, vector_ops, matrix_ops, vquant, vdequant'],
    ], { colW: [1.8, 0.6, 3.38], size: 8.5, rowH: 0.42 });
    D.code(s, 7.05, 4.15, 5.78, 1.35, [
      'def run(program_words, global_cells):        # v09_runtime.py',
      '    build_v09_model()   # g++ -std=c++17 -O2, 소스가 바뀌면 재빌드',
      '    with TemporaryDirectory(dir=NPU_V09_TMPDIR):',
      '        write program_memory.bin, global_memory.bin; subprocess.run([mysim_v09])',
      '        images = split(saved_global_memory.bin, image_bytes)   # snapshot list',
      '    return images, counters   # 크기 불일치 → 디스크 부족 진단 메시지',
    ], { size: 8 });
    D.callout(s, 7.05, 5.6, 5.78, 0.75, [
      { text: '기능 검증용 실행기: ', bold: true }, { text: 'cycle을 세지 않고 counters만 냅니다. words_executed와 DMA cells는 정적 해독과 대조하지만 latency로 바꾸지 않습니다.', size: 9.5 },
    ], { kind: 'warn' });
    D.takeaway(s, 'C-model의 성공은 기능 검증입니다. RTL 구조(256 lane 버퍼, VLEN tail, accumulator 상태)의 검증을 대신하지 않습니다.', { y: 6.5 });
  });

};
