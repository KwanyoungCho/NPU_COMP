'use strict';
const { C, F } = require('./lib');

module.exports = (D) => {
  // ---------------------------------------------------------------- 55 walker overview
  D.add({ section: '9 · ISA · linker · runtime', title: 'Codegen: 주소가 연결된 scheduled TIR를 정적으로 전개된 명령 stream으로', subtitle: 'Walker는 TIR을 직접 해석합니다. 모든 For는 완전히 unroll되고 loop/branch 명령은 없습니다. 아래는 두 matmul 예제의 두 번째 커널(K loop 2회)이 29 words의 직선 stream이 되는 실제 결과입니다.', source: 'npu_compiler/tir_codegen_v09.py (Walker.visit, _match_nest, _emit_pointwise, _emit_reduction, _emit_movement, emit_gemm), docs/examples/two_matmuls_recorded/program.json',
    notes: 'Linker가 Walker(asm, bases, SramEmitter)를 만들고 parameter를 signature 순서로 2-byte 단위 base에 바인딩하며 SRAM buffer를 nibble로 선언합니다. visit는 For를 unroll하고 IfThenElse를 compile 시점에 결정하며 LetStmt는 거절합니다. _match_nest는 loop nest 전체를 잡아 출력 행마다 vector 명령 하나를 냅니다. 그림의 strip: GLOAD W2(4 words) → K tile 0의 descriptor·LOAD·M_MUL(MAC=0) → K tile 1(MAC=1) → DST descriptor·SAVE → GSTORE Y(4 words) → HALT. 네 부류의 emission이 있습니다: tensorized matmul, elementwise(6 scratch slot 직렬화), reduction(REDUCE_SUM/MAX + BROADCAST), 복사 stage(scope가 다르면 DMA, 같으면 LOAD/COPY/SAVE 또는 BROADCAST/strided LOAD). Python walker와 C++ native walker는 같은 계약을 따르며 커널 단위로 섞일 수 있습니다. 정적 전개이므로 Llama prefill은 8.58M words가 됩니다.' },
  (s, D) => {
    D.code(s, 0.5, 1.55, 3.85, 1.3, [
      '# scheduled_0001.py (핵심만)',
      'v_sram = cache_read(W2) → GLOAD',
      'npu_fill_zero(C)        # MAC=0 상태',
      'for k_0 in range(2):    # unroll',
      '  npu_gemm_acc(C, lv[:,k_0*64:], v_sram[k_0*64:,:])',
      'compute[i,j] = C[i,j]   # SAVE → GSTORE',
    ], { size: 7.8 });
    D.arrow(s, 4.4, 2.2, 4.85, 2.2, { width: 2 });
    D.text(s, 4.3, 1.7, 0.75, 0.45, 'Walker\nunroll', { size: 7.5, color: C.muted, align: 'center', bold: true });
    const W = (label, kind, n, extra) => Object.assign({ n: n || 1, label, kind, size: 6 }, extra || {});
    D.text(s, 4.9, 1.5, 7.9, 0.22, '실제 word 48–76 (29 words = 22 instructions): 4-word DMA packet 2개 + 1-word 명령 20개 + HALT', { size: 8.5, bold: true, color: C.navy });
    D.wordStrip(s, 4.9, 1.8, 7.93, 0.42, [
      W('GLOAD W2\n(4 words)', 'dram', 4, { tick: 48 }),
      W('A', 'ir', 1, { above: 'K tile 0' }), W('C', 'host'), W('A', 'ir'), W('L', 'sram', 1, { bold: true }), W('A', 'ir'), W('C', 'host'), W('A', 'ir'), W('L', 'sram', 1, { bold: true }), W('M0', 'unit', 1, { bold: true, tick: 60 }),
      W('A', 'ir', 1, { above: 'K tile 1' }), W('A', 'ir'), W('L', 'sram', 1, { bold: true }), W('A', 'ir'), W('A', 'ir'), W('L', 'sram', 1, { bold: true }), W('M1', 'unit', 1, { bold: true, tick: 67 }),
      W('A', 'ir', 1, { above: 'DST' }), W('C', 'host'), W('A', 'ir'), W('S', 'good', 1, { bold: true }),
      W('GSTORE Y\n(4 words)', 'dram', 4, { tick: 72 }), W('HALT', 'red', 1, { tick: 76 }),
    ], { size: 6 });
    D.text(s, 4.9, 2.42, 7.93, 0.22, 'A = ADDR(0x80)  C = ROWS/COLS(0x88/0x89)  L = LOAD(0x90)  M0/M1 = M_MUL MAC 0/1(0x42)  S = SAVE(0x98)  — loop·branch 명령 없음', { size: 7, color: C.muted, font: F.code });
    D.table(s, 0.5, 2.78, 12.33, [
      ['TIR 조각', '인식', '생성하는 명령'],
      ['call_extern npu_fill_zero / npu_gemm_acc (tensorize 결과)', '_bind_match로 64×64 view 해석', 'fill: 명령 없음(zeroed 기록). gemm: region(SRC1) LOAD, region(SRC2) LOAD, M_MUL(mac). flush: region(DST) SAVE'],
      ['원소별 BufferStore (innermost 축 = 행)', '_emit_pointwise → _materialize (6 scratch slot)', 'VLEN, ADDR SRC1.P, LOAD, ADDR SRC2.P, LOAD, OP(VECTOR), ADDR DST.P, SAVE. 상수/stride-0 → BROADCAST'],
      ['reduction (LowerInitBlock 형태: init + 누산 store)', '_emit_reduction (Add/npu_sum_fp32_update → RSUM, Max → RMAX)', '출력 위치마다 VLEN, SRC1.P(행 또는 x·x 같은 materialized 피연산자), LOAD, REDUCE, DST.P, SAVE. 길이 ≤ 65535'],
      ['복사 block (scope 다름: global ↔ global.sram)', '_emit_dma: [global byte, sram nibble, nbytes] 구간 병합', '2D DMA(정규 행·cell 정렬) / 1D dma_in·out / bounce(scratch gather + cell RMW)'],
      ['복사 block (같은 scope)', '_emit_movement', '연속: LOAD → COPY(0x17) → SAVE; stride-0: BROADCAST; strided: strided LOAD(matrix, ncols=1)'],
      ['select / if_then_else 구간 (concat, padding)', '_pieces / _split_row', '행을 구간 경계에서 나눠 각각 emission'],
    ], { colW: [3.3, 3.0, 6.03], size: 8.2, rowH: 0.43 });
    D.text(s, 0.5, 6.02, 12.33, 0.4, '주소 단위: _flat은 SRAM 주소를 nibble(fp16 4, fp32 8, int8 2, int4 1)로, global 주소를 byte로 돌려줍니다. Python walker(tir_codegen_v09.py)와 native C++ walker(v09_walker.cc)는 같은 contract JSON hash를 공유하며 커널 단위로 섞일 수 있습니다.', { size: 8, color: C.muted });
    D.takeaway(s, 'TIR에 loop가 있어도 최종 ISA에는 loop 명령이 없습니다. words는 "명령 저장 공간"이며 실행 cycle이 아닙니다 (Llama prefill 8.58M words).', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 56 movement ops, units, descriptors
  D.add({ section: '9 · ISA · linker · runtime', title: '네 가지 이동 명령, 세 가지 주소 단위, descriptor 상태 모델', subtitle: 'Descriptor는 다음 LOAD/연산/SAVE가 쓸 주소·shape·dtype "상태"입니다. 명령마다 배열 정보를 반복하는 대신 register를 바꾸고 뒤 명령이 참조합니다. 아래는 77-word 예제 두 번째 matmul 직전의 실제 register 값입니다.', source: 'docs/current/09_isa.md, reference/execution_contract.md, npu_compiler/isa_compute.py, isa_v09.py, two_matmuls_recorded/program.json',
    notes: 'H가 SRAM에 보관되면 GSTORE/GLOAD는 사라질 수 있지만 H를 matrix unit에 넣는 LOAD까지 없어지지는 않습니다. 한 숫자를 다른 단위로 읽으면 2배/4배 어긋난 주소가 됩니다. 0x80은 SRC1/SRC2/DST × MAIN/PARTIAL 주소의 16-bit 절반을, 0x88/0x89는 rows/cols와 dtype을, 0x82는 VLEN을 씁니다. MAIN은 전체 stride, PARTIAL은 이번 접근 영역의 시작과 크기입니다. 표의 값: SRC1 MAIN 0xCC18 = nibble 52,248 = byte 26,124(H), cols 128(행 stride 256 B); SRC2 MAIN 0x4C18 = 19,480(W2), cols 64; DST 0x0C18 = 3,096 = byte 1,548(Y). 두 번째 K tile에서 SRC1은 +0x100(64열), SRC2는 +0x4000(64행)만 바뀌고 COLS는 peephole이 생략합니다. region()은 operand 하나의 64×64 view를 위해 8 words를 냅니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 5.6, [
      ['명령', '출발 → 도착', 'DRAM을 지나는가'],
      ['GLOAD 0xA0', 'DRAM → SRAM', '예 (4-word packet)'],
      ['LOAD 0x90', 'SRAM → 연산장치 입력', '아니오'],
      ['SAVE 0x98', '연산장치 결과 → SRAM', '아니오'],
      ['GSTORE 0xA8', 'SRAM → DRAM', '예 (4-word packet)'],
    ], { colW: [1.4, 2.4, 1.8], size: 9.5, rowH: 0.32 });
    D.table(s, 0.5, 3.3, 5.6, [
      ['위치', '단위', 'byte 변환'],
      ['plan.address (DRAM)', 'unit_bytes = 2', '× 2'],
      ['DMA global 주소 / stride / cols', '32-bit cell', '× 4'],
      ['SRAM descriptor 주소', 'nibble', '÷ 2 (fp16 = 4 nibbles)'],
      ['shape / row stride / VLEN', 'element', '× dtype 크기'],
    ], { colW: [2.4, 1.6, 1.6], size: 9.5, rowH: 0.32 });
    D.text(s, 6.35, 1.5, 6.5, 0.3, 'Descriptor 상태 모델 (설정 → 참조)', { size: 11, bold: true, color: C.navy });
    D.table(s, 6.35, 1.8, 6.48, [
      ['opcode', 'field', '쓰는 상태'],
      ['0x80 ADDR', 'operand[31:30], high[29], PARTIAL[28], value[23:8]', '{SRC1,SRC2,DST} × {MAIN,PARTIAL} 주소의 16-bit 절반 (nibble)'],
      ['0x88 / 0x89 ROWS / COLS', 'operand[31:30], PARTIAL[29], dtype[26:25], count[23:8]', 'rows/cols (element) + 그 operand의 dtype. MAIN cols = 전체 row stride'],
      ['0x82 VLEN', 'n[23:8]', 'vector 논리 길이 1..65535'],
      ['0x90 LOAD', 'matrix[31], operand[30], strided[29], ncols[23:16], start[15:8]', 'SRAM→unit. matrix는 PARTIAL rows/cols + MAIN stride, vector는 VLEN'],
      ['0x98 SAVE', 'matrix[31], strided[29], ncols, start', '결과 → DST'],
      ['0x42 M_MUL', 'mode[31:30], activation[29:28], MAC[27], imm[23:8]', 'SRC1×SRC2 → accumulator (MAC=0 새로, 1 누적)'],
      ['0x8A / 0x8B', 'high[29], value[23:8]', 'activation / weight scale의 SRAM 주소 (양자화)'],
    ], { colW: [1.5, 2.9, 2.08], size: 8, rowH: 0.4 });
    // descriptor register state example
    D.text(s, 0.5, 5.02, 5.6, 0.24, 'Register 상태 예 (word 52–71, H[64,128] @ W2[128,64] → Y)', { size: 9.5, bold: true, color: C.navy });
    D.table(s, 0.5, 5.28, 5.6, [
      ['register', 'SRC1 (A = H)', 'SRC2 (B = W2)', 'DST (Y)'],
      ['MAIN addr', '0xCC18 → nib 52,248 (byte 26,124)', '0x4C18 → 19,480 (9,740)', '0x0C18 → 3,096 (1,548)'],
      ['MAIN cols = stride', '128 el (256 B)', '64 el', '64 el'],
      ['PARTIAL addr (K tile 1)', '+0x100 = +64열', '+0x4000 = +64행', '—'],
      ['rows × cols / dtype', '64 × 64 FP16', '64 × 64 FP16', '64 × 64 FP16'],
    ], { colW: [1.3, 1.65, 1.4, 1.25], size: 7.5, rowH: 0.22 });
    D.text(s, 6.35, 5.1, 6.48, 1.25, 'region(operand, view) = 8 words: ADDR MAIN lo/hi, ROWS MAIN, COLS MAIN(=stride), ADDR PARTIAL lo/hi, ROWS/COLS PARTIAL. 주소가 이전과 같아 보여도 중간 명령이 설정을 바꾸었는지 확인해야 합니다.\nEnum: operand SRC1=0, SRC2=1, DST=2 · mode IMM=0, SCALAR=1, VECTOR=2 · dtype FP16=0, FP32=1, INT8=2, INT4=3 · activation OFF=0, tanh-GELU=1, SiLU=2, legacy GELU=3. 모든 field는 range-check됩니다 ([7,128256] tensor가 16-bit field를 조용히 넘친 사례 이후).', { size: 8, color: C.muted });
    D.takeaway(s, '기본 FP16 행렬의 stride 128은 128 bytes가 아니라 256 bytes입니다. MAIN stride는 유지하고 PARTIAL 주소·크기만 바꿔야 올바른 위치에 SAVE합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 57 DMA packet decode
  D.add({ section: '9 · ISA · linker · runtime', title: '실제 명령 해독 ①: X를 SRAM으로 가져오는 4-word GLOAD packet', subtitle: '77-word 예제의 word offset 8: 000c18a0 00000000 00000800 00010800. DRAM의 X[64,64] FP16 8,192 B가 SRAM byte 1548에 도착합니다.', source: 'docs/examples/two_matmuls_recorded/program_memory.bin, program.json, npu_compiler/isa_v09.py',
    notes: 'w0의 상위 24 bit는 SRAM nibble 주소(8-nibble 정렬), 하위 8 bit는 opcode입니다. 0xC18 = 3096 nibbles = byte 1548, 앞 슬라이드의 X 주소와 일치합니다. w1은 global cell 주소 0(= byte 0, X가 image 맨 앞), w2는 row stride 0x800 = 2048 cells, w3는 rows 1(상위 16 bit) / cols 0x800 = 2048 cells = 8192 bytes입니다. 1D DMA는 rows=1, cols=cells, stride=cols로 인코딩되고 0xFFFF cells/rows를 넘으면 분할됩니다. DMA는 dtype 없는 cell 이동이며 4-byte cell 경계의 부분 데이터는 이웃 값을 손상하지 않도록 별도 처리(bounce)합니다.' },
  (s, D) => {
    const words = [['w0', '000c18a0', [{ name: 'SRAM nibble 주소', hi: 31, lo: 8, kind: 'sram', value: '0x000C18 = 3096 nib = byte 1548' }, { name: 'opcode', hi: 7, lo: 0, kind: 'unit', value: '0xA0 GLOAD' }]],
      ['w1', '00000000', [{ name: 'global cell 주소', hi: 31, lo: 0, kind: 'dram', value: '0 → byte 0 (image 맨 앞 = X)' }]],
      ['w2', '00000800', [{ name: 'global row stride (cells)', hi: 31, lo: 0, kind: 'dram', value: '0x800 = 2048 cells = 8192 B' }]],
      ['w3', '00010800', [{ name: 'rows', hi: 31, lo: 16, kind: 'ir', value: '1' }, { name: 'cols (cells)', hi: 15, lo: 0, kind: 'ir', value: '0x800 = 2048 cells = 8192 B' }]]];
    words.forEach((w, i) => {
      const y = 1.95 + i * 0.95;
      D.text(s, 0.5, y + 0.08, 0.5, 0.3, w[0], { size: 12, bold: true, color: C.navy });
      D.text(s, 1.0, y + 0.08, 1.3, 0.3, w[1], { size: 12, font: F.code, bold: true });
      D.bitfield(s, 2.5, y, 10.3, 0.55, w[2], { size: 8.5 });
    });
    D.table(s, 0.5, 5.85, 12.33, [
      ['해독 결과', 'X[64,64] FP16 = 64 × 64 × 2 = 8,192 B = 2,048 cells를 DRAM byte 0에서 SRAM byte 1548로 한 행(1D)으로 이동. rows 1 이므로 stride는 의미 없음'],
      ['같은 프로그램의 GSTORE (offset 72)', '000c18a8 00002800 …: opcode 0xA8, SRAM nibble 3096 → global cell 0x2800 = byte 40,960 (Y 출력, plan.outputs.offset_bytes = 40960)'],
    ], { colW: [2.6, 9.73], size: 9, rowH: 0.3, header: false });
    D.takeaway(s, '77 words = 62 instructions인 이유: DMA는 명령 하나가 4 words이고 나머지는 1 word입니다 (GLOAD 4개, GSTORE 1개 → 15 words 추가).', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 58 M_MUL decode + timeline
  D.add({ section: '9 · ISA · linker · runtime', title: '실제 명령 해독 ②: M_MUL의 MAC bit와 77-word 프로그램의 흐름', subtitle: '80000042와 88000042의 차이는 bit 27(MAC)뿐입니다. 둘 다 mode VECTOR, activation OFF. 아래 strip은 77 words 전체를 opcode 종류별로 칠한 것입니다.', source: 'docs/examples/two_matmuls_recorded/program.json, program_memory.bin, docs/current/09_isa.md, npu_compiler/isa_compute.py',
    notes: 'Word 0-3은 상수 pool GLOAD, 4-47은 커널 0(첫 matmul: W1 GLOAD, X GLOAD, 두 출력 tile 각각 MAC=0 → H 직접 SAVE), 48-75는 커널 1(W2 GLOAD가 W1 자리에, K tile 두 개 MAC=0/1, Y SAVE, Y GSTORE), 76은 HALT입니다. strip에서 DMA packet(주황)이 4 words씩 차지하고 나머지는 descriptor(파랑)·LOAD/SAVE(청록)·M_MUL(보라) 1 word씩임을 볼 수 있습니다. Peephole 효과: 두 번째 tile(word 40)에서는 SRC1 descriptor가 통째로 사라지고 SRC2 low half만 바뀌며, 커널 1에는 0x88 word가 없습니다(rows=64가 커널 0에서 이어짐). opcode 통계는 0x42×4, 0x80×26, 0x88×6, 0x89×9, 0x90×8, 0x98×3, 0xA0×4, 0xA8×1, 0xFF×1입니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.3, 'M_MUL word (opcode 0x42)', { size: 11, bold: true, color: C.navy });
    D.bitfield(s, 0.5, 2.05, 7.3, 0.55, [
      { name: 'mode', hi: 31, lo: 30, kind: 'unit', value: '2 = VECTOR' }, { name: 'act', hi: 29, lo: 28, kind: 'ir', value: 'OFF' }, { name: 'MAC', hi: 27, lo: 27, kind: 'red', value: '0/1' },
      { name: 'reserved', hi: 26, lo: 24, kind: 'faint' }, { name: 'imm[23:8]', hi: 23, lo: 8, kind: 'faint', value: '0' }, { name: 'opcode', hi: 7, lo: 0, kind: 'unit', value: '0x42' },
    ], { size: 8 });
    D.table(s, 0.5, 2.8, 7.3, [
      ['word', '값', '뜻'],
      ['30, 44', '80000042', '첫 matmul의 두 H 출력 tile, 각각 MAC=0 (새 accumulator)'],
      ['60', '80000042', '두 번째 matmul, 첫 K tile, MAC=0'],
      ['67', '88000042', '같은 출력의 다음 K tile, MAC=1 (bit 27)'],
      ['404c1880 / 504c1880', 'ADDR', 'SRC2 MAIN lo = 0x4C18 (W1 nibble 19480 = byte 9740) / PARTIAL'],
      ['40008089', 'COLS', 'SRC2 MAIN cols = 128 (= row stride, elements)'],
      ['c0000090 / 80000098', 'LOAD / SAVE', 'matrix LOAD SRC2 / matrix SAVE → DST'],
    ], { colW: [1.6, 1.3, 4.4], size: 8.8, rowH: 0.3 });
    D.text(s, 8.05, 1.5, 4.8, 0.3, '77-word 프로그램 timeline', { size: 11, bold: true, color: C.navy });
    const rows = [['0–3', '상수 pool GLOAD (3 cells → SRAM nibble 0)', 'host'], ['4–7', 'W1 GLOAD → SRAM 9740', 'dram'], ['8–11', 'X GLOAD → SRAM 1548', 'dram'], ['12–29', 'descriptor 설정 (SRC1/SRC2 region)', 'ir'], ['30', 'M_MUL MAC=0 → H tile 0', 'unit'], ['31–43', 'SAVE tile 0, SRC2 재설정 (+64열)', 'ir'], ['44', 'M_MUL MAC=0 → H tile 1', 'unit'], ['45–47', 'SAVE tile 1 (H 직접)', 'ir'], ['48–51', 'W2 GLOAD → SRAM 9740 (W1 재사용)', 'dram'], ['52–59', 'descriptor (0x88 없음: rows 64 유지)', 'ir'], ['60', 'M_MUL MAC=0 (K tile 0)', 'unit'], ['61–66', 'K tile 1 operand 설정', 'ir'], ['67', 'M_MUL MAC=1 (K tile 1)', 'unit'], ['68–71', 'Y SAVE (FP16)', 'ir'], ['72–75', 'Y GSTORE → DRAM byte 40960', 'dram'], ['76', 'HALT 0xFF', 'host']];
    rows.forEach((r, i) => {
      const y = 1.85 + i * 0.27;
      D.rect(s, 8.05, y, 4.78, 0.25, { kind: r[2], radius: 0.02, lineWidth: 0.5 });
      D.text(s, 8.12, y, 0.75, 0.25, r[0], { size: 8, bold: true, valign: 'middle', font: F.code });
      D.text(s, 8.9, y, 3.9, 0.25, r[1], { size: 8, valign: 'middle' });
    });
    // 77-word strip
    const seq = 'GGGAARRAARRLAARRAARRLMAARRAARRSLAALMAASGARALARALMAALAALMARASTH';
    const kindOf = { G: 'dram', T: 'dram', A: 'ir', R: 'host', L: 'sram', S: 'good', M: 'unit', H: 'red' };
    const items = []; let w = 0;
    for (const ch of seq) {
      const n = (ch === 'G' || ch === 'T') ? 4 : 1;
      const it = { n, kind: kindOf[ch], label: ch === 'M' ? 'M' : (ch === 'G' ? 'GLOAD' : (ch === 'T' ? 'GSTORE' : (ch === 'H' ? 'HALT' : ''))), size: 5.5 };
      if (w === 0 || w === 4 || w === 48 || w === 72 || ch === 'M' || ch === 'H') it.tick = w;
      items.push(it); w += n;
    }
    D.text(s, 0.5, 5.15, 7.3, 0.22, '77 words 전체 (색 = opcode 종류; 주황 4칸 = DMA packet 1개)', { size: 9, bold: true, color: C.navy });
    D.wordStrip(s, 0.5, 5.4, 7.3, 0.34, items, { size: 5.5 });
    D.text(s, 0.5, 5.95, 7.3, 0.42, '주황 DMA (0xA0/0xA8) · 파랑 ADDR (0x80) · 회색 ROWS/COLS (0x88/0x89) · 청록 LOAD (0x90) · 초록 SAVE (0x98) · 보라 M_MUL (0x42) · 빨강 HALT.  62 instructions = 77 words = 308 bytes. DMA load 40,972 B + store 8,192 B (C-model counters와 일치).', { size: 7.5, color: C.muted });
    D.takeaway(s, 'Peephole의 효과: 두 번째 tile에서 SRC1 descriptor 생략, 커널 1에서 0x88 생략. 값이 알려져 있고 바뀌지 않았을 때만 삭제합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 59 linker
  D.add({ section: '9 · ISA · linker · runtime', title: 'Linker: 커널 stream을 호출 순서로 연결하고, forwarding·peephole·주소 압축으로 마무리한다', subtitle: 'npu_link._compile_program. 커널은 독립적으로 emit되고(각자 필요한 descriptor를 다시 냄) 경계의 중복 설정은 뒤의 peephole이 제거합니다. 아래 strip은 77-word 예제의 실제 구성입니다.', source: 'npu_compiler/npu_link.py, peephole.py, npu_residency.py, v09_effects.py, two_matmuls_recorded/program.json kernel_records',
    notes: '순서: 계약 검사(tile 64, cell 32, numeric policy) → npu_memplan.assign_addresses → scalar pool(호출되는 PrimFunc의 FP16 literal + {1.0, 2.0}, 짝수 개로 padding, DRAM constant_base와 FP32 1.0, 첫 커널 전에 SRAM nibble 0으로 GLOAD, scratch_base 계산) → binding 순서대로 커널(joint search가 미리 emit한 words / W8A8 reference emitter / GlobalVar별 cache된 recipe를 _place_sram 후 native 또는 Python walker) → HALT → residency forwarding → peephole → joint vs all-DRAM baseline 비교 → compact_unused_storage → cache append. kernel_records의 span [4,48), [48,76)이 strip의 경계입니다. Peephole은 descriptor register 상태만 정확히 추적하며 LOAD/SAVE/연산/DMA/BROADCAST는 건드리지 않습니다. Forwarding의 효과 분석은 각 명령이 어떤 주소를 쓰는지(WriteEffects) 알아야 하며, 알 수 없는 opcode는 cache 전체를 비웁니다.' },
  (s, D) => {
    const st = [['계약 검사', 'tile 64 · cell 32\nnumeric policy', 'host'], ['DRAM plan', 'assign_addresses\ncodegen_contract hash', 'dram'], ['scalar pool', 'FP16 literal (+1.0, 2.0)\n→ SRAM nibble 0 GLOAD', 'host'], ['커널 emission', 'binding 순서\n3 경로 (joint / W8A8 / recipe)', 'unit'], ['HALT', '0xFF', 'host'], ['residency', 'GSTORE 사본 → GLOAD 제거/복사', 'sram'], ['peephole', 'descriptor 중복 제거', 'sram'], ['baseline 비교', 'joint vs all-DRAM cost', 'warn'], ['DRAM 압축', 'compact_unused_storage', 'dram'], ['KV append', 'AppendPlan 재구성', 'dram']];
    const w = (12.33 - 9 * 0.1) / 10;
    st.forEach((b, i) => {
      const x = 0.5 + i * (w + 0.1);
      D.box(s, x, 1.55, w, 0.9, b[0], { kind: b[2], size: 8.5, detail: b[1], dsize: 7 });
      if (i < 9) D.arrow(s, x + w + 0.005, 2.0, x + w + 0.095, 2.0, { width: 0.75 });
    });
    // stream concatenation strip
    D.text(s, 0.5, 2.62, 12.33, 0.22, '연결된 stream (77 words): kernel_records.span = [4, 48), [48, 76)', { size: 9, bold: true, color: C.navy });
    D.wordStrip(s, 0.5, 2.87, 12.33, 0.42, [
      { n: 4, kind: 'host', label: 'const\nGLOAD', tick: 0, size: 6.5 },
      { n: 44, kind: 'unit', label: 'kernel 0 = matmul (44 words): GLOAD W1, GLOAD X · tile 0 MAC=0 SAVE · tile 1 MAC=0 SAVE (H 직접)', tick: 4, size: 7 },
      { n: 28, kind: 'sram', label: 'kernel 1 = matmul1 (28 words): GLOAD W2 · MAC 0/1 · SAVE Y · GSTORE Y', tick: 48, size: 7 },
      { n: 1, kind: 'red', label: 'H', tick: 76, size: 6 },
    ], { size: 7 });
    D.text(s, 0.5, 3.44, 12.33, 0.22, '각 커널은 독립적으로 emit(필요한 descriptor를 다시 냄) → 경계의 중복 설정은 peephole이 제거(커널 1에 0x88 없음). residency는 GSTORE된 SRAM 사본을 커널 진입에서 재사용.', { size: 7.5, color: C.muted });
    D.text(s, 0.5, 3.7, 6, 0.28, 'Peephole (peephole.py): 정확한 register 상태 추적', { size: 10.5, bold: true, color: C.navy });
    D.bullets(s, 0.5, 4.0, 6.1, 1.3, [
      '추적 상태: (addr, operand, partial, high), (vlen), (0x88/0x89, operand, partial), (dtype, operand), (scale, op, high)',
      '값이 알려져 있고 같을 때만 write 삭제. shape word는 count 또는 dtype이 바뀌면 유지',
      'DMA packet은 4-word 단위로 건너뜀. LOAD·SAVE·연산·DMA·BROADCAST(0x15)는 건드리지 않음. branch가 없어 정확함',
    ], { size: 8.8, psa: 3 });
    D.text(s, 6.85, 3.7, 6, 0.28, 'Residency forwarding (npu_residency.forward): 효과 분석 기반', { size: 10.5, bold: true, color: C.navy });
    D.bullets(s, 6.85, 4.0, 5.98, 1.3, [
      'GSTORE마다 (global byte 구간 ↔ SRAM nibble 구간, version, producer, canonical FP16) 사본 등록. GSTORE 자체는 유지 (write-through)',
      '겹치는 GSTORE는 global 쪽, GLOAD 목적지와 정밀 compute write는 SRAM 쪽에서 무효화. WriteEffects.step: vector SAVE = 길이 × dtype nibble, matrix SAVE = ((rows−1)·stride + cols)·4; 미지 opcode → cache 전체 무효',
      '커널 진입 시 완전 포함된 연속 GLOAD만: 같은 주소면 삭제, 아니면 SRAM→SRAM V_COPY + descriptor 복원',
    ], { size: 8.5, psa: 3 });
    D.table(s, 0.5, 5.4, 12.33, [
      ['커널 emission 경로', '언제', '내용'],
      ['joint search가 미리 emit한 words', 'plan_program이 채택된 호출 (이 예제의 두 커널)', '보관 주소로 이미 검증된 stream 재사용 (재link 없음)'],
      ['W8A8 reference emitter', 'w8a8 PrimFunc', '전용 경로 (부록 A2)'],
      ['scheduled recipe', '나머지 호출', 'GlobalVar별 recipe cache (28 layers: 약 32개 커널이 1,206회 호출) → _place_sram → native/Python walker'],
    ], { colW: [2.6, 3.0, 6.73], size: 8.3, rowH: 0.26 });
    D.takeaway(s, '효과 분석은 "이 명령이 어떤 주소를 읽고 쓰는가"를 아는 것입니다. 인코딩이 같다는 이유로 명령을 삭제하지 않습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 60 native/python consistency & stats
  D.add({ section: '9 · ISA · linker · runtime', title: 'Python codegen과 native C++ codegen의 일치, 그리고 통계의 정의', subtitle: '하나의 contract JSON이 양쪽 op table을 만들고 hash로 묶입니다. native의 unsupported는 Python fallback, invalid_schedule은 fallback 없는 LinkError, 내부 오류는 propagate합니다.', source: 'npu_compiler/v09_contract.json, native.py, npu_codegen/build_identity.py, v09_stats.py, tests/test_native_codegen.py',
    notes: 'v09_contract.py는 JSON 바이트의 sha256을 CONTRACT_HASH로, npu_codegen/generate_contract.py는 같은 hash와 kMaxVlen, op X-macro를 v09_contract.h로 씁니다. build_identity.py는 .cc/.h를 정렬 순서로 hash해 NPU_BUILD_SOURCE_HASH로 export합니다. native.load()는 .so가 없으면 Python을 쓰고 hash 불일치면 RuntimeError("rebuild")를 냅니다. 모드는 use(기본, NPU_NATIVE=0로 비활성), compare(둘 다 실행해 첫 다른 word 보고), python입니다. v09_stats.analyze는 DMA를 4-word packet으로 읽어 words와 instructions를 구분하고, dma bytes를 Σrows·cols·4로, sram_peak를 커널 top의 최댓값/2로, image를 top·unit_bytes로 계산합니다. check_counters가 C-model counters와 대조합니다.' },
  (s, D) => {
    const a = D.box(s, 0.5, 1.6, 2.6, 0.9, 'v09_contract.json', { kind: 'host', size: 11, detail: 'dtype nibble, opcode, max_vlen,\nnumeric policy, op 목록', dsize: 8 });
    const b = D.box(s, 3.7, 1.6, 2.6, 0.9, 'Python: v09_contract.py', { kind: 'ir', size: 10.5, detail: 'CONTRACT_HASH = sha256(bytes)\ntir_codegen_v09 op table', dsize: 8 });
    const c = D.box(s, 3.7, 2.75, 2.6, 0.9, 'C++: v09_contract.h', { kind: 'unit', size: 10.5, detail: 'generate_contract.py → 같은 hash,\nkMaxVlen, X-macro op 목록', dsize: 8 });
    D.arrow(s, a.r, a.cy, b.x, b.cy); D.arrow(s, a.r, a.cy + 0.2, c.x, c.cy);
    const d = D.box(s, 6.9, 2.15, 2.6, 0.9, 'native.load()', { kind: 'sram', size: 11, detail: '.so 없음 → Python\nhash 불일치 → RuntimeError("rebuild")', dsize: 8 });
    D.arrow(s, b.r, b.cy, d.x, d.cy - 0.15); D.arrow(s, c.r, c.cy, d.x, d.cy + 0.15);
    D.table(s, 9.8, 1.6, 3.03, [
      ['status', '처리'],
      ['ok', 'native words 사용'],
      ['unsupported', 'Python fallback (native_fallbacks 기록)'],
      ['invalid_schedule', 'LinkError, fallback 없음'],
      ['그 외 오류', 'propagate'],
    ], { colW: [1.1, 1.93], size: 8.5, rowH: 0.34 });
    D.text(s, 0.5, 3.85, 6, 0.3, '모드와 검증', { size: 10.5, bold: true, color: C.navy });
    D.bullets(s, 0.5, 4.15, 6.1, 1.6, [
      ['use (기본): ', 'native 사용. NPU_NATIVE=0으로 끔, NPU_CODEGEN_SO로 경로 지정'],
      ['compare: ', '둘 다 실행해 첫 번째 다른 word를 보고 (_compare_native)'],
      ['build_identity: ', 'npu_codegen/*.cc,*.h 정렬 hash → npu.codegen_source_hash로 export, provenance에 기록'],
      ['gate: ', 'EncodeSelfTest 인코더 자체 검사, family별 compare, 전체 프로그램 end-to-end 비교'],
    ], { size: 9, psa: 3 });
    D.text(s, 6.9, 3.85, 6, 0.3, 'v09_stats.analyze / cost', { size: 10.5, bold: true, color: C.navy });
    D.table(s, 6.9, 4.15, 5.93, [
      ['지표', '정의'],
      ['words / instructions', '32-bit word 수 / packet 수 (DMA = 4 words = 1 instruction)'],
      ['opcodes', '각 packet 첫 word의 하위 byte histogram'],
      ['dma_load/store_bytes', 'Σ rows × cols × 4 (이동 bytes, strided span 아님)'],
      ['sram_peak_bytes', 'max(커널 top) ÷ 2 (pool + scratch + buffers)'],
      ['global_image_bytes', 'plan.top × unit_bytes'],
      ['cost', '(dma_bytes, words, sram_peak_bytes) 사전식'],
    ], { colW: [1.7, 4.23], size: 8.5, rowH: 0.3 });
    D.takeaway(s, 'Native/Python 일치는 두 구현이 같은 계약을 따른다는 근거입니다. 둘이 공유한 모델 수식의 오류는 HF/NumPy 또는 LLVM 기준으로 따로 확인합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 61 bundle
  D.add({ section: '9 · ISA · linker · runtime', title: '결과물: NpuExecutable과 export_bundle', subtitle: 'Binary만으로는 inference 결과를 얻을 수 없습니다. 같은 plan에 맞는 입력과 weight image, 그리고 출력 tree의 shape/dtype/offset이 필요합니다.', source: 'npu_compiler/npu_target.py NpuExecutable, docs/current/09_isa.md, docs/examples/pipeline/bundle/',
    notes: 'NpuExecutable에는 words, plan, 낮춰진 module, 입력 signature(param_names)가 있습니다. export_bundle은 program_memory.bin(little-endian uint32), program.json(주소·통계·선택·출력 ABI·provenance), lowered.py(선택된 semantic IRModule), scheduled_*.py(joint-memory 경로의 호출별 TIR), values를 주면 global_memory.bin(초기 DRAM image)을 씁니다. 일반 recipe 경로의 단일 fused 커널은 자동 dump 대상이 아니므로 작은 예제 runner가 compile context의 recipe를 꺼내 07_scheduled_fused_matmul_add_silu.py로 남겼습니다. 전체 모델은 매우 큰 image와 pickle을 문서에 보관하지 않고 측정·hash를 남겼습니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 7.4, [
      ['파일', '담긴 내용', '담기지 않은 것'],
      ['program_memory.bin', 'little-endian uint32 명령 stream (HALT로 끝)', 'weight 값 전체'],
      ['program.json', 'statistics, memory(addresses/sizes), schedules, sram_layouts, joint_memory, kernel_records, dram_compaction, fusion, residency, outputs, cache_append, provenance, compilation', '실장치 측정 cycle (hardware_measurement = null)'],
      ['lowered.py', '선택된 semantic IRModule (TVM Script)', '최종 주소를 포함한 runtime 상태'],
      ['scheduled_NNNN.py', 'joint-memory에서 실제 선택된 호출별 TIR', '원래 모델 설명 전체'],
      ['global_memory.bin', 'values를 지정한 경우 초기 DRAM image', '지속적인 SRAM 상태'],
    ], { colW: [1.7, 3.9, 1.8], size: 8.8, rowH: 0.5 });
    D.code(s, 8.15, 1.55, 4.68, 2.6, [
      'class NpuExecutable:',
      '    words, plan, func, kernel_count, target, module',
      '    param_names   # 최종 signature 순서',
      '    image_bytes = plan.top * plan.unit_bytes',
      '    build_image(values) → npu_link.build_image',
      '    run(values) → (output, counters)   # C-model',
      '    save(path)  # little-endian <u4',
      '    export_bundle(dir, values=None)',
      '',
      '# provenance: python/numpy/torch/transformers,',
      '#   tvm 0.19.0 commit, llvm 18.1.8, compiler_head,',
      '#   compiler_source_sha256, native .so sha256,',
      '#   cmodel_source_sha256, contract_sha256',
    ], { size: 8 });
    D.text(s, 8.15, 4.3, 4.68, 0.3, '작은 예제 bundle/program.json 발췌', { size: 10.5, bold: true, color: C.navy });
    D.code(s, 8.15, 4.6, 4.68, 1.75, [
      '"statistics": {"words": 1577, "instructions": 1562,',
      '   "dma_packets": 5, "dma_bytes": 41100,',
      '   "sram_peak_bytes": 42508, "global_image_bytes": 41100}',
      '"outputs": {"name": "gv", "shape": [64, 64],',
      '   "dtype": "float16", "offset_bytes": 32896}',
      '"kernel_records": [{"name": "fused_matmul_add_silu",',
      '   "backend": "native", "sram_peak_nibbles": 85016,',
      '   "arguments": [{"address_bytes": 0, "size_bytes": 16384}, …]}]',
    ], { size: 7.8 });
    D.callout(s, 0.5, 4.75, 7.4, 1.6, [
      { text: 'Runtime이 필요로 하는 것: ', bold: true }, { text: 'program words + plan.address(입력·parameter 위치) + plan.outputs(leaf별 shape/dtype/offset) + cache_append(있으면). C-model은 program_memory.bin과 global_memory.bin을 읽고 saved_global_memory.bin(HALT 시 1개, SNAPSHOT마다 추가)과 perf_counters.txt를 씁니다.', size: 10, breakLine: true },
      { text: ' ', size: 4, breakLine: true },
      { text: 'C-model counters: ', bold: true }, { text: 'words_executed, gload/gstore, dma_cells_loaded/stored, loads/saves, vector_ops, matrix_ops, vquant/vdequant → v09_stats.check_counters가 정적 해독과 대조', size: 10 },
    ], { kind: 'faint' });
    D.takeaway(s, '작은 pipeline bundle은 입력 image까지 보존합니다. 전체 모델은 수 GB image 대신 측정값과 hash를 evidence로 남겼습니다.', { y: 6.5 });
  });
};
