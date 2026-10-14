'use strict';
const { C, F } = require('./lib');

module.exports = (D) => {
  // ---------------------------------------------------------------- 41 elementwise & SiLU
  D.add({ section: '7 · Vector · 정규화 · attention', title: '원소별 연산의 codegen: SiLU 하나가 ISA 하나가 아니다', subtitle: 'VLEN=128인 한 행이 SRAM scratch slot을 거치며 SIGN_INV → EXP → BROADCAST(1.0) → ADD → DIV로 sigmoid가 되고, 마지막 곱은 별도 block입니다. 각 단계마다 descriptor·LOAD·SAVE가 붙습니다.', source: 'npu_compiler/tir_codegen_v09.py (_emit_pointwise, _materialize, _sigmoid), docs/examples/pipeline/09_packets.json words 440-468',
    notes: 'Walker는 innermost 축을 vector 행으로 보고 표현식 tree를 6개 scratch slot에 직렬화합니다(slot 폭 = 커널의 가장 넓은 행). 연속 load는 주소만, stride-0 load나 literal은 BROADCAST, binary op는 VLEN·ADDR·LOAD·ADDR·LOAD·OP·ADDR·SAVE 순서입니다. 그림의 각 행은 SRAM에 실제로 저장되는 FP16 128-원소 행이며 오른쪽 word가 그 단계의 실제 opcode입니다. E1 fused 프로그램의 opcode 통계가 이를 보여줍니다: SIGN_INV 32, EXP 32, ADD 96, DIV 32, MUL 32, BROADCAST 64, 주소 설정 0x80이 606 words.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.28, '실제 packet (fused 커널, 한 행 VLEN=128, peephole 뒤)', { size: 10.5, bold: true, color: C.navy });
    D.code(s, 0.5, 1.8, 6.5, 1.75, [
      '00008082  VLEN 128',
      '104c1880  SRC1.P = x행    00000090 LOAD   00000016 SIGN_INV   90001880 DST.P=slot0   00000098 SAVE',
      '10001880  SRC1 = slot0    00000090 LOAD   0000000f EXP        90021880 DST=slot1     00000098 SAVE',
      '40000415  BROADCAST lo (1.0 @ pool 4)   60000015 BROADCAST hi                90041880 DST=slot2   SAVE',
      '10021880 50041880  LOAD×2 (slot1, slot2)   80000001 ADD   90061880 DST=slot3   00000098 SAVE',
      '10041880 50061880  LOAD×2 (slot2, slot3)   8000000b DIV   90cc1880 DST=sig     00000098 SAVE',
      '... (별도 T_multiply block) LOAD x, LOAD sig   8000000a MUL   SAVE y',
    ], { size: 7.5 });
    // scratch-slot lane figure
    const lx = 2.45, lw = 3.4, lh = 0.27, tW = 1.85;
    const mk = (pre, kind) => Array.from({ length: 8 }, (_, i) => ({ label: i === 7 ? '…' : `${pre}${i}`, kind, size: 6.5 }));
    D.text(s, 0.5, 3.6, 6.5, 0.22, 'SRAM scratch slot을 지나는 한 행 (128 원소, FP16 저장) — 오른쪽은 실제 opcode word', { size: 9, bold: true, color: C.navy });
    const rows = [
      ['x 행 (T_add 결과)', mk('x', 'host'), 'SRC1.PARTIAL = 19480'],
      ['slot0 = −x', mk('−x', 'unit'), 'SIGN_INV  00000016'],
      ['slot1 = exp(slot0)', mk('e', 'unit'), 'EXP  0000000f'],
      ['slot2 = 1.0 (broadcast)', Array.from({ length: 8 }, () => ({ label: '1', kind: 'ir', size: 6.5 })), 'BROADCAST  40000415 / 60000015'],
      ['slot3 = slot1 + slot2', mk('1+e', 'unit'), 'ADD  80000001'],
      ['sig = 1 / slot3', mk('σ', 'sram'), 'DIV  8000000b → compute buffer'],
      ['y = x × sig', mk('y', 'good'), 'MUL  8000000a (T_multiply block)'],
    ];
    rows.forEach((r, i) => {
      const y = 3.88 + i * 0.36;
      D.lanes(s, lx, y, lw, lh, r[1], { title: r[0], titleW: tW, right: r[2], rightW: 1.1, rightMono: true, rightSize: 6.5 });
      if (i > 0 && i < 6) D.arrow(s, lx + lw / 2, y - 0.09, lx + lw / 2, y, { width: 0.75, color: C.muted });
    });
    D.table(s, 7.2, 1.5, 5.63, [
      ['E1 fused 프로그램 opcode', 'count', '역할'],
      ['0x80 ADDR', '606', 'descriptor 주소 (peephole 뒤에도 대부분)'],
      ['0x90 LOAD / 0x98 SAVE', '388 / 258', 'SRAM ↔ vector/matrix unit'],
      ['0x01 ADD', '96', 'bias add 64행 + sigmoid의 +1 32행'],
      ['0x16 SIGN_INV / 0x0f EXP', '32 / 32', 'sigmoid 전개 (행 32 = 64×64를 128폭으로)'],
      ['0x0b DIV / 0x0a MUL', '32 / 32', '1/(1+e⁻ˣ), x·sig'],
      ['0x15 BROADCAST', '64', '상수 1.0 (2 words 씩)'],
      ['0x42 M_MUL / 0xa0 GLOAD / 0xa8 GSTORE', '2 / 4 / 1', 'matmul 2 K-tile, 입력·상수 DMA, 출력'],
      ['합계', '1,577 words', '1,562 instructions'],
    ], { colW: [2.3, 1.0, 2.33], size: 8.8, rowH: 0.34 });
    D.text(s, 7.2, 4.75, 5.63, 1.5, '중간값은 FP16 scratch slot(6개 × 최대 행 폭)에 저장·재사용됩니다. 한 단계 = VLEN·ADDR·LOAD(·LOAD)·OP·ADDR·SAVE이므로 원소별 연산 하나가 명령 여러 개가 되고, fusion해도 vector 명령이 matmul activation bit로 바뀌지 않습니다. 행이 32개(64×64를 128폭으로 결합)라 각 opcode가 32회씩 나타납니다.', { size: 9, color: C.muted });
    D.takeaway(s, '명령 수(words)는 짧은 vector 행과 descriptor 반복에 좌우됩니다. 이것이 정적 전개 stream의 비용이며 cycle과는 다른 지표입니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 42 VLEN & generic vector scheduling
  D.add({ section: '7 · Vector · 정규화 · attention', title: '물리 256 lanes와 논리 VLEN, 그리고 generic vector scheduling', subtitle: 'v09 규약은 물리 폭 256과 논리 길이 ≤65535를 구분합니다. Generic vector 커널은 tensor 전체를 SRAM에 놓아 보고 안 들어가면 행 묶음을 반으로 줄이며, 연속 원소별 축은 결합해 긴 행으로 만듭니다.', source: 'docs/current/07_operators.md, reference/execution_contract.md, npu_schedule.py (generic stripe halving, _coalesce_pointwise)',
    notes: 'VLEN=1000이면 장치 내부에서 256+256+256+232로 처리한다는 전제이며 RTL 검증은 별도입니다. 오른쪽 위 그림: 64×64 tensor와 임시값 전체가 SRAM에 안 들어가면 32행, 16행으로 stripe를 반씩 줄이고 compute_at과 buffer compaction으로 각 묶음의 작업 공간을 줄입니다. 오른쪽 아래: 연속된 원소별 축을 표준 fuse/split으로 합쳐 64행×VLEN 64를 32행×VLEN 128로 만들면 명령 수가 절반이 됩니다(작은 예제의 npu.pointwise_vector_width=128). broadcast, 비연속 stride, 겹치는 입출력, 임시 공간 한도를 검사하고 행별 reduction은 축을 합치지 않습니다. 현재 비용 모델에는 lane 활용률이나 연산별 cycle이 없습니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.3, 'VLEN 1000 = 256 + 256 + 256 + 232 (장치 내부 반복 전제)', { size: 10.5, bold: true, color: C.navy });
    D.grid(s, 0.5, 1.85, 6.0, 0.5, 1, 4, { highlight: (r, c) => c === 3 ? 'warn' : 'unit', label: (r, c) => c === 3 ? '232 (tail)' : '256 lanes', size: 9 });
    D.bullets(s, 0.5, 2.55, 6.0, 1.7, [
      'Compiler가 배열을 1024로 padding해야 한다는 뜻은 아님. C-model은 VLEN 전체를 C++ 배열로 보관',
      '실제 장치가 256만 지원한다면 공통 vector lowering에서 분할하고 긴 reduction의 누산·반올림 계약도 재검토 (미확인 항목)',
      '길이 1024 명령 하나 ≠ 길이 256 명령 하나의 비용. 현재 (DMA, words, SRAM) 모델에는 lane 활용률이 없음. REDUCE 길이 ≤ 65535, softmax 한도 65504',
    ], { size: 9, psa: 4 });
    D.table(s, 0.5, 4.45, 6.0, [
      ['축 결합 검사', '결과'],
      ['broadcast, 비연속 stride, 겹치는 입출력, 임시 공간 한도', '검사 통과 시에만 결합'],
      ['행별 reduction (softmax sum, RMSNorm)', '축을 합치지 않음 (의미가 달라짐)'],
      ['SiLU처럼 입력을 두 번 읽는 경우', '같은 묶음의 cache 공유 (DRAM 재적재 없음)'],
    ], { colW: [3.6, 2.4], size: 9, rowH: 0.36 });
    // stripe halving figure
    D.text(s, 6.9, 1.5, 6, 0.28, 'Generic vector 커널: SRAM에 안 들어가면 행 묶음(stripe)을 반으로', { size: 10.5, bold: true, color: C.navy });
    D.matrix(s, 6.9, 1.85, 1.3, 1.3, 8, 8, { kind: 'faint', regions: [{ r0: 0, r1: 8, c0: 0, c1: 8, kind: 'red', label: '64행 전체\n+ 임시값 ✗', size: 7.5 }], below: '초과 → 반분', belowW: 1.3 });
    D.arrow(s, 8.25, 2.5, 8.6, 2.5, { width: 1.5 });
    D.matrix(s, 8.65, 1.85, 1.3, 1.3, 8, 8, { kind: 'faint', regions: [{ r0: 0, r1: 4, c0: 0, c1: 8, kind: 'warn', label: '32행 stripe ✗', size: 7.5 }], below: '아직 초과 → 반분', belowW: 1.3 });
    D.arrow(s, 10.0, 2.5, 10.35, 2.5, { width: 1.5 });
    D.matrix(s, 10.4, 1.85, 1.3, 1.3, 8, 8, { kind: 'faint', regions: [{ r0: 0, r1: 2, c0: 0, c1: 8, kind: 'sram', label: '16행 ✓', size: 7.5 }], below: 'compute_at + compaction으로\n묶음별 작업 공간 축소', belowW: 1.6 });
    D.text(s, 11.8, 1.85, 1.05, 1.3, '4 stripe ×\n(LOAD·OP·SAVE)\n반복', { size: 7, color: C.muted });
    // coalescing figure
    D.text(s, 6.9, 3.65, 6, 0.28, '축 결합: 64행 × VLEN 64 → 32행 × VLEN 128 (명령 수 절반)', { size: 10.5, bold: true, color: C.navy });
    D.matrix(s, 6.9, 4.0, 1.1, 1.1, 4, 4, { kind: 'sram', regions: [{ r0: 0, r1: 4, c0: 0, c1: 4, kind: 'sram', label: '64 × 64\n64 명령/단계', size: 7 }], below: '행 = VLEN 64', belowW: 1.2 });
    D.arrow(s, 8.1, 4.55, 8.5, 4.55, { width: 1.5 });
    D.text(s, 8.05, 4.15, 0.5, 0.3, 'fuse', { size: 7, color: C.muted, align: 'center' });
    D.matrix(s, 8.55, 4.28, 2.4, 0.55, 2, 8, { kind: 'sram', regions: [{ r0: 0, r1: 2, c0: 0, c1: 8, kind: 'sram', label: '32 × 128 — 32 명령/단계', size: 7 }], below: '행 = VLEN 128 (연속 축이므로 같은 원소 순서)', belowW: 3.0 });
    D.code(s, 6.9, 5.25, 5.93, 1.05, [
      '# 작은 예제 scheduled TIR: npu.pointwise_vector_width = 128',
      'for ax0_ax1_fused_0, ax0_ax1_fused_1 in T.grid(32, 128):',
      '    compute_global_sram[(f0*128+f1)//64, (f0*128+f1)%64] =',
      '        T.sigmoid(T_add_intermediate_global_sram[...])',
    ], { size: 7.8 });
    D.takeaway(s, 'Vector schedule은 "공간에 들어가는 큰 행 묶음"을 찾는 성격이 강합니다. 짧은 명령과 SRAM 임시값 비용은 남은 검토 항목입니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 43 RoPE & GQA
  D.add({ section: '7 · Vector · 정규화 · attention', title: 'RoPE와 GQA: host가 cos/sin을 만들고 NPU는 곱셈·부호 반전·재배치만 한다', subtitle: 'Head dimension 128을 앞/뒤 절반 (a | b)로 나누고 rotate_half = [−b | a]를 만든 뒤 x·cos + rotate_half(x)·sin을 계산합니다. Q/K에만 적용하고 V에는 적용하지 않습니다.', source: 'npu_compiler/nn_models/llama.py (_rotate_half, rope_freqs, llama3_scale_freqs, validate_rope_scaling), npu_graph_opt.FoldRepeatedGemv',
    notes: '그림의 각 행은 한 token·한 head의 128 원소 벡터입니다(8칸으로 축약). IR에서는 split(64) → negative → concat → multiply(cos) / multiply(sin) → add로 표현되며 모두 vector 명령입니다. rope_freqs는 base^(−2i/hd)를 만들고 linear는 factor로 나누며 llama3는 wavelength에 따라 세 구간을 적용합니다. 결과를 [freqs, freqs]로 이어 rotate_half와 맞추고, rope_inputs가 float64로 각도를 계산해 FP16 cos/sin [S,1,hd]로 내립니다. validate_rope_scaling은 default/linear/llama3만 받습니다. GQA는 op.repeat(k, 3, axis=1)이며 persistent cache는 KV head 8개로 저장됩니다. decode에서는 FoldRepeatedGemv가 repeat를 broadcast matmul로 바꾸는 후보를 만들 수 있습니다(m=1 전용).' },
  (s, D) => {
    const lx = 2.4, lw = 3.9, lh = 0.27, tW = 1.85;
    D.text(s, 0.5, 1.5, 6.1, 0.24, 'RoPE: 한 token · 한 head의 128 원소 (8칸으로 축약)', { size: 10.5, bold: true, color: C.navy });
    const A = (p, k, n) => Array.from({ length: n }, (_, i) => ({ label: `${p}${i}`, kind: k, size: 6.5 }));
    const rows = [
      ['x = [a | b]', A('a', 'sram', 4).concat(A('b', 'ir', 4)), ''],
      ['rotate_half = [−b | a]', A('−b', 'ir', 4).concat(A('a', 'sram', 4)), ''],
      ['cos[pos] (host, FP16)', A('c', 'host', 8), ''],
      ['sin[pos] (host, FP16)', A('s', 'host', 8), ''],
      ['x·cos + rot·sin', Array.from({ length: 8 }, (_, i) => ({ label: i < 4 ? 'a·c−b·s' : 'b·c+a·s', kind: 'unit', size: 5.5 })), ''],
    ];
    rows.forEach((r, i) => {
      const y = 1.85 + i * 0.37;
      D.lanes(s, lx, y, lw, lh, r[1], { title: r[0], titleW: tW });
    });
    D.text(s, 0.5, 3.75, 6.1, 0.4, 'IR: split(64) → negative → concat → multiply(cos), multiply(sin) → add.  new_a = a·cos − b·sin, new_b = b·cos + a·sin (짝 (aᵢ,bᵢ)의 2D 회전). V에는 적용하지 않음.', { size: 8, color: C.muted });
    D.table(s, 0.5, 4.3, 6.1, [
      ['host 준비 (nn_models/llama.py)', '내용'],
      ['rope_freqs(cfg)', 'θ^(−2i/128), θ=500000; linear: ÷factor'],
      ['llama3_scale_freqs', 'wavelength > 8192/1 → f/32; < 8192/4 → f; 사이는 보간'],
      ['rope_inputs(cfg, positions)', 'float64 각도 → FP16 cos/sin [S,1,128] (heads로 broadcast)'],
      ['validate_rope_scaling', 'default / linear / llama3만 허용, 알 수 없는 키·factor<1 거절'],
    ], { colW: [2.2, 3.9], size: 8.8, rowH: 0.36 });
    D.text(s, 6.85, 1.5, 6, 0.3, 'GQA: 24 Q heads ← 8 KV heads', { size: 10.5, bold: true, color: C.navy });
    const y = 1.85;
    for (let h = 0; h < 8; h++) {
      D.rect(s, 6.85 + h * 0.74, y, 0.66, 0.42, { kind: 'dram', radius: 0.03 });
      D.text(s, 6.85 + h * 0.74, y, 0.66, 0.42, `KV${h}`, { size: 8.5, bold: true, align: 'center', valign: 'middle' });
    }
    for (let q = 0; q < 24; q++) {
      const x = 6.85 + q * 0.247;
      D.rect(s, x, y + 0.85, 0.22, 0.36, { kind: 'unit', radius: 0.02 });
      D.text(s, x, y + 0.85, 0.22, 0.36, String(q), { size: 6.5, align: 'center', valign: 'middle' });
    }
    for (let h = 0; h < 8; h++) D.arrow(s, 6.85 + h * 0.74 + 0.33, y + 0.42, 6.85 + h * 3 * 0.247 + 0.37, y + 0.85, { width: 0.75, color: C.muted });
    D.text(s, 6.85, y + 1.3, 5.98, 0.5, 'prefill: op.repeat(k, 3, axis=1) → [S,24,128]. persistent cache는 KV head 8개로 저장 (Q head 수만큼 복제하지 않음)', { size: 9, color: C.muted });
    D.table(s, 6.85, 3.75, 5.98, [
      ['GQA 관련 후보 / 계약', '내용'],
      ['FoldRepeatedGemv (decode, m=1)', 'matmul(A[24,1,K], repeat(B[8,K,N],3)) → matmul(A.reshape(8,3,1,K), B.expand_dims(1)) broadcast. 후보이며 항상 선택되지는 않음'],
      ['prefill의 repeat (axis 1)', 'compound attention schedule에 맡김'],
      ['reshape / transpose / concat', 'reshape는 view 가능, transpose·concat·repeat는 모두 view는 아님 → lowering이 결정'],
    ], { colW: [2.0, 3.98], size: 8.5, rowH: 0.5 });
    D.takeaway(s, 'RoPE 설정은 검사 후 지원하거나 거절합니다. 알 수 없는 설정을 조용히 근사하지 않습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 44 attention & query-blocked fusion
  D.add({ section: '7 · Vector · 정규화 · attention', title: 'Attention을 연결해서 읽기, 그리고 full-fusion의 query 분할', subtitle: 'Prefill head 하나: Q[S,128]×Kᵀ[128,S] → scores[S,S] → /√128 + mask → 행별 softmax → P×V[S,128]. 전체를 묶으면 query 행 block만 계산하고 그 block의 score 행 전체를 SRAM에서 소비합니다.', source: 'npu_compiler/npu_intrin.py schedule_attention_blocked, npu_schedule._fit_attention, docs/current/07_operators.md',
    notes: '그림의 격자 한 칸은 64×64 tile입니다(S=512로 축약). 진한 행은 현재 query block이고 회색 삼각형은 causal mask로 기여가 0인 영역입니다. schedule_attention_blocked는 query 행을 block으로 split하고 reorder(row_o, head, row_i, …)로 row block을 head loop 바깥에 둡니다. 다른 모든 block은 head loop에 compute_at되고 scores [1,block,seq], max, exp, sum, 정규화 결과, PV accumulator는 set_scope로 SRAM에 둡니다. Q/K/V cache_read는 head에, mask(head 축 1 broadcast)는 row_o에 둡니다. _fit_attention은 가장 큰 block부터 시작해 실패하면 반으로 줄이고 floor min(rows,64)에서도 실패하면 LinkError입니다. 한 block이 score 행 전체를 갖고 있어 softmax는 정확하며 online rescaling(FlashAttention)이 없습니다.' },
  (s, D) => {
    const y = 2.0, blk = [2, 4];
    const Q = D.matrix(s, 0.5, y, 0.6, 2.0, 8, 1, { kind: 'faint', title: 'Q [S,128]', titleW: 1.0, regions: [{ r0: blk[0], r1: blk[1], c0: 0, c1: 1, kind: 'unit', label: 'block', size: 7 }], below: 'S=512 → 8 tiles\n(그림)', belowW: 1.0 });
    const K = D.matrix(s, 1.45, y - 0.25, 2.4, 0.4, 1, 8, { kind: 'sram', title: 'Kᵀ [128,S] (head 전체, SRAM)', titleW: 2.6, regions: [{ r0: 0, r1: 1, c0: 0, c1: 8, kind: 'sram', label: 'cache_read @ head', size: 7 }] });
    const masked = [];
    for (let r = 0; r < 8; r++) for (let c = r + 1; c < 8; c++) masked.push({ r0: r, r1: r + 1, c0: c, c1: c + 1, kind: 'faint', lineWidth: 0.3 });
    const Sc = D.matrix(s, 1.45, y + 0.4, 2.4, 1.6, 8, 8, { kind: 'faint', regions: [{ r0: blk[0], r1: blk[1], c0: 0, c1: 8, kind: 'unit' }].concat(masked), below: 'scores [S,S]: block 행만 계산 (/√128, +mask). 회색 = causal 0', belowW: 2.6 });
    D.text(s, 1.5, y + 0.4 + 1.6 * blk[0] / 8, 1.2, 0.4, 'scores block\n[block, S]', { size: 7, bold: true, color: C.unit, valign: 'middle' });
    D.arrow(s, Q.r + 0.03, Q.y + 2.0 * 3 / 8, Sc.x, Sc.y + 1.6 * 3 / 8, { width: 1.5, color: C.unit });
    D.arrow(s, Sc.r + 0.05, Sc.y + 1.6 * 3 / 8, Sc.r + 0.5, Sc.y + 1.6 * 3 / 8, { width: 1.5, color: C.unit });
    D.text(s, Sc.r + 0.02, Sc.y + 1.6 * 3 / 8 - 0.4, 0.6, 0.38, 'softmax\n(행 전체)', { size: 6.5, color: C.muted, align: 'center' });
    const P = D.matrix(s, 4.45, y + 0.4, 2.4, 1.6, 8, 8, { kind: 'faint', regions: [{ r0: blk[0], r1: blk[1], c0: 0, c1: 8, kind: 'unit' }].concat(masked), below: 'P block: max·exp·FP32 sum·divide 모두 SRAM', belowW: 2.6 });
    D.text(s, 4.5, y + 0.4 + 1.6 * blk[0] / 8, 1.2, 0.4, 'P block', { size: 7, bold: true, color: C.unit, valign: 'middle' });
    const V = D.matrix(s, 7.25, y, 0.6, 2.0, 8, 1, { kind: 'sram', title: 'V [S,128]', titleW: 1.0, regions: [{ r0: 0, r1: 8, c0: 0, c1: 1, kind: 'sram', label: 'head\n전체', size: 6.5 }] });
    const O = D.matrix(s, 8.3, y, 0.6, 2.0, 8, 1, { kind: 'faint', title: 'output [S,128]', titleW: 1.3, regions: [{ r0: blk[0], r1: blk[1], c0: 0, c1: 1, kind: 'good', label: 'out', size: 7 }], below: 'cache_write\n→ DRAM', belowW: 1.0 });
    D.arrow(s, P.r + 0.03, P.y + 1.6 * 3 / 8, V.x, V.y + 2.0 * 3 / 8, { width: 1.5, color: C.unit });
    D.arrow(s, V.r + 0.03, V.y + 2.0 * 3 / 8, O.x, O.y + 2.0 * 3 / 8, { width: 1.5, color: C.good });
    D.text(s, 7.0, y + 2.05, 1.2, 0.3, '× (P V)', { size: 7, color: C.muted, align: 'center' });
    D.text(s, 0.5, y + 2.45, 8.4, 0.4, '다음 row block (아래로 이동) → 같은 head의 K/V를 다시 사용. mask [1,S,S]은 head 축이 1이라 row_o loop에서 한 번만 GLOAD (head들이 공유).', { size: 8, color: C.muted });
    D.table(s, 9.2, y - 0.4, 3.63, [
      ['S=3072 크기', '값'],
      ['24-head score 전체', '432 MiB'],
      ['head 1개 64행 score', '384 KiB'],
      ['K, V (head 1개)', '768 KiB 씩'],
      ['Q block 64행', '16 KiB'],
    ], { colW: [2.0, 1.63], size: 9, rowH: 0.32 });
    D.table(s, 0.5, 4.72, 7.1, [
      ['schedule_attention_blocked 단계', '내용'],
      ['split(row, [None, block]) → reorder(row_o, head, row_i, …)', 'row block이 head loop 바깥: block마다 모든 head 처리'],
      ['모든 중간 block compute_at(head), set_scope(SRAM)', 'scores·max·exp·sum·정규화·PV accumulator가 SRAM에만 존재'],
      ['cache_read Q/K/V → compute_at(head); mask → row_o', 'K/V는 block마다 다시 읽음 (block이 클수록 재읽기 감소)'],
      ['두 matmul: pad_einsum? → split 64 → decompose_reduction → tensorize ×2', '일반 matmul과 같은 64×64 MAC chain'],
      ['_fit_attention: block = rows부터, 실패 시 max(floor, block//2)', 'floor = min(rows, 64). floor에서도 실패하면 LinkError'],
    ], { colW: [3.6, 3.5], size: 8.3, rowH: 0.28 });
    D.callout(s, 7.85, 4.72, 4.98, 1.65, [
      { text: '없는 것: ', bold: true }, { text: 'key까지 나누어 진행 중 max·sum·output을 보정하는 online softmax(FlashAttention)는 없습니다. 한 block이 query 행의 key score 전체를 갖고 있어 softmax는 정확합니다.', size: 9.5, breakLine: true },
      { text: '조건: ', bold: true }, { text: 'full 패턴은 정적 rank-3 FP16, m·d·t·dv 64배수. decode(m=1)는 해당 없음. mask는 계산을 동적으로 생략하지 않습니다.', size: 9.5 },
    ], { kind: 'warn' });
    D.takeaway(s, '384 KiB score만 있으면 커널이 들어간다는 뜻이 아닙니다. K/V, mask, 확률, 출력, 임시값이 함께 SRAM에 있어야 하므로 block 크기를 줄여가며 맞춥니다.', { y: 6.5 });
  });

};
