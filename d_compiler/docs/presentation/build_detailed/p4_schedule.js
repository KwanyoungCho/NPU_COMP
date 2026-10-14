'use strict';
const { C, F } = require('./lib');

module.exports = (D) => {
  // ---------------------------------------------------------------- 30 64x64 and K chain
  D.add({ section: '6 · Matmul scheduling', title: '64×64 장치로 K=128을 계산하기: 출력 tile 하나의 K 누산은 끊지 않는다', subtitle: '작은 예제 ② 두 번째 matmul H[64,128] @ W2[128,64]의 실제 명령 20 words. X₀·B₀로 accumulator를 시작하고(MAC=0), X₁·B₁을 같은 accumulator에 더한 뒤(MAC=1), 한 번만 FP16으로 SAVE합니다.', source: 'docs/examples/two_matmuls_recorded/program.json (words 52–71), scheduled_0001.py, docs/current/06_matmul_schedule.md',
    notes: '위 그림은 tile 단위 데이터 흐름이고 아래 strip은 77-word 프로그램의 word 52~71을 그대로 옮긴 것입니다. 각 operand는 descriptor(ADDR MAIN/PARTIAL, COLS)를 설정한 뒤 LOAD로 matrix unit에 들어갑니다. 두 번째 K tile에서는 SRC1 MAIN 주소가 +256 nibbles(= 64열), SRC2가 +16384 nibbles(= 64행) 이동하고 COLS는 peephole이 생략했습니다. accumulator가 1개이고 partial spill이 금지이므로 출력 tile 하나의 K chain을 끝내고 다음 tile로 갑니다(K 최내측). 이것은 모든 하드웨어의 일반 법칙이 아니라 이 장치의 계약입니다.' },
  (s, D) => {
    const y = 1.85;
    const X = D.matrix(s, 0.5, y + 0.4, 1.6, 0.8, 1, 2, { kind: 'sram', tile: 1, title: 'A = H [64,128] (SRAM 보관값)', titleW: 2.2, regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'sram', label: 'X₀' }, { r0: 0, r1: 1, c0: 1, c1: 2, kind: 'ir', label: 'X₁' }], below: 'MAIN stride 128 el, PARTIAL 64×64', belowW: 2.2 });
    const B = D.matrix(s, 2.6, y, 0.8, 1.6, 2, 1, { kind: 'dram', tile: 1, title: 'B = W2 [128,64]', titleW: 1.4, regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'sram', label: 'B₀' }, { r0: 1, r1: 2, c0: 0, c1: 1, kind: 'ir', label: 'B₁' }], below: 'GLOAD → SRAM 9740', belowW: 1.4 });
    // matrix unit
    D.rect(s, 4.3, y - 0.1, 3.1, 1.85, { kind: 'unit', radius: 0.08 });
    D.text(s, 4.4, y - 0.05, 2.9, 0.22, 'Matrix unit', { size: 9.5, bold: true, color: C.unit });
    D.text(s, 4.4, y + 0.2, 1.75, 0.95, '① LOAD X₀ → SRC1\n   LOAD B₀ → SRC2\n   M_MUL MAC=0\n② LOAD X₁, B₁\n   M_MUL MAC=1', { size: 8, color: C.ink, font: F.code });
    const acc = D.matrix(s, 6.3, y + 0.25, 0.95, 0.95, 4, 4, { kind: 'unit', fill: 'F3EFFC', title: 'acc FP32', titleW: 1.0, regions: [{ r0: 0, r1: 4, c0: 0, c1: 4, kind: 'unit', fill: 'DCD3F5', label: 'X₀B₀\n+X₁B₁', size: 7.5, lineWidth: 0 }] });
    D.text(s, 6.2, y + 1.25, 1.15, 0.4, '64×64 × 1개\n(SRAM 아님)', { size: 6.5, color: C.muted, align: 'center' });
    D.arrow(s, X.x + 0.4, X.y, 4.3, y + 0.35, { width: 1.5, color: C.sram }); D.arrow(s, X.x + 1.2, X.y, 4.3, y + 0.95, { width: 1.5, color: C.ir, dash: 'dash' });
    D.arrow(s, B.r, B.y + 0.4, 4.3, y + 0.5, { width: 1.5, color: C.sram }); D.arrow(s, B.r, B.y + 1.2, 4.3, y + 1.1, { width: 1.5, color: C.ir, dash: 'dash' });
    const P = D.matrix(s, 8.3, y + 0.4, 0.8, 0.8, 1, 1, { kind: 'good', title: 'Y [64,64] (SRAM 1548)', titleW: 1.9, regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'good', label: 'f16(acc)', size: 8 }], below: '③ SAVE 1회 → 이후 GSTORE', belowW: 2.2 });
    D.arrow(s, 7.4, y + 0.8, P.x, P.cy, { width: 2, color: C.good });
    D.text(s, 10.35, y - 0.1, 2.48, 1.9, 'K 반복이 출력 tile 반복 안에 있음(K 최내측). M/N이 크면 출력 tile마다 ①②③를 반복하고 panel 내부 순서는 inner_order가 정합니다. npu_fill_zero는 명령을 내지 않고 "새 누산 시작"만 기록해 첫 M_MUL이 MAC=0이 됩니다. activation bit는 자동 경로에서 OFF입니다.', { size: 8.5, color: C.muted });
    // actual word strip (words 52..71)
    D.text(s, 0.5, 4.1, 12, 0.24, '실제 명령 (program_memory.bin word 52–71, 20 words = 이 matmul 전체)', { size: 9.5, bold: true, color: C.navy });
    const W = (label, kind, extra) => Object.assign({ n: 1, label, kind, size: 6.5 }, extra || {});
    D.wordStrip(s, 0.5, 4.55, 12.33, 0.42, [
      W('ADDR\nSRC1 M', 'ir', { above: '① X₀ → SRC1', tick: 52 }), W('COLS\n128', 'host'), W('ADDR\nSRC1 P', 'ir'), W('LOAD\nSRC1', 'sram', { bold: true }),
      W('ADDR\nSRC2 M', 'ir', { above: 'B₀ → SRC2' }), W('COLS\n64', 'host'), W('ADDR\nSRC2 P', 'ir'), W('LOAD\nSRC2', 'sram', { bold: true }),
      W('M_MUL\nMAC=0', 'unit', { bold: true, tick: 60 }),
      W('ADDR\nSRC1 M', 'ir', { above: '② X₁ (+256 nib)' }), W('ADDR\nSRC1 P', 'ir'), W('LOAD\nSRC1', 'sram', { bold: true }),
      W('ADDR\nSRC2 M', 'ir', { above: 'B₁ (+16384 nib)' }), W('ADDR\nSRC2 P', 'ir'), W('LOAD\nSRC2', 'sram', { bold: true }),
      W('M_MUL\nMAC=1', 'unit', { bold: true, tick: 67 }),
      W('ADDR\nDST M', 'ir', { above: '③ Y 저장' }), W('COLS\n64', 'host'), W('ADDR\nDST P', 'ir'), W('SAVE', 'good', { bold: true, tick: 71 }),
    ], { size: 6.5 });
    D.text(s, 0.5, 5.2, 12.33, 0.4, '두 번째 K tile에서 COLS(0x89)가 없는 이유: rows/cols가 바뀌지 않아 peephole이 생략. SRC1 MAIN 0xCC18 → 0xCD18 (+64열), SRC2 MAIN 0x4C18 → 0x8C18 (+64행 × 64열). DST 0x0C18 = byte 1548 (X 자리 재사용).', { size: 8, color: C.muted });
    D.table(s, 0.5, 5.7, 12.33, [
      ['질문', '답'],
      ['왜 K가 최내측인가?', 'accumulator가 1개이고 partial spill 금지 → 출력 tile 하나의 K chain을 끝내고 다음 tile. M, N이 크면 tile마다 ①②③ 반복'],
    ], { colW: [2.0, 10.33], size: 8.5, rowH: 0.26 });
    D.takeaway(s, 'Schedule의 자유도는 tile 안이 아니라 tile 바깥에 있습니다: 어떤 panel을 SRAM에 얼마나 오래 두고, 어떤 순서로 출력 tile을 돌 것인가.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 31 panel vs tile
  D.add({ section: '6 · Matmul scheduling', title: 'SRAM에 올리는 panel은 matrix tile보다 크다: 실제 폭 gate projection', subtitle: 'X[64,3072] @ W[3072,8192]. W는 48 MiB = 48×128 tiles라 SRAM에 들어가지 않으므로 X(1×48 tiles, 384 KiB)를 상주시키고 W의 열 panel j(48×1 tiles, 384 KiB)를 128번 교체합니다.', source: 'docs/current/06_matmul_schedule.md, evidence/current/refinement.json (M=64 MLP 정적 compile 기록)',
    notes: '그림의 각 칸은 64×64 tile입니다. X panel 하나를 한 번 올리면 모든 출력 tile이 재사용하고, W의 열 panel j가 올라오면 그 안의 K 방향 48개 tile로 M_MUL 48개를 연속 실행해 출력 tile (0, j)를 완성합니다. 출력 열 tile은 128개이므로 총 6,144 M_MUL이며 W 전체 48 MiB가 한꺼번에 SRAM에 올라가는 일은 없습니다. 같은 X/W 크기라도 W를 몇 번 다시 읽는지에 따라 DMA는 달라지고, SRAM을 가득 채우는 것이 목표가 아니라 앞 커널 출력 보관 공간과 임시값 공간이 함께 필요합니다. down projection은 반대로 W panel(8192×256, 4 MiB)을 유지하고 H는 앞 커널의 SRAM 보관값을 씁니다.' },
  (s, D) => {
    // DRAM W as tile grid 6x16 (represents 48x128)
    const Wd = D.matrix(s, 0.5, 1.85, 4.3, 1.6, 6, 16, { kind: 'dram', title: 'DRAM: W [3072,8192] = 48 × 128 tiles (48 MiB) — 그림은 6×16으로 축약', titleW: 4.6, regions: [{ r0: 0, r1: 6, c0: 3, c1: 4, kind: 'sram', label: 'j', size: 7 }], below: '열 panel j = 48 tiles = 384 KiB. j = 0 … 127 순서로 GLOAD', belowW: 4.4 });
    // SRAM box
    D.rect(s, 5.35, 1.6, 4.2, 2.35, { kind: 'sram', radius: 0.08, fill: 'EEF8F6' });
    D.text(s, 5.45, 1.62, 4, 0.22, 'SRAM (8 MiB 중 일부)', { size: 9.5, bold: true, color: C.sram });
    const Xp = D.matrix(s, 5.5, 2.1, 3.9, 0.28, 1, 12, { kind: 'sram', title: 'X [64,3072] = 1×48 tiles 상주 (384 KiB), a_at = m_panel', titleW: 4.0, titleSize: 7.5, regions: [{ r0: 0, r1: 1, c0: 0, c1: 12, kind: 'sram', label: 'k = 0 … 47', size: 7 }] });
    const Wp = D.matrix(s, 5.5, 2.75, 0.28, 1.1, 8, 1, { kind: 'sram', regions: [{ r0: 0, r1: 8, c0: 0, c1: 1, kind: 'dram', label: '', lineWidth: 1 }] });
    D.text(s, 5.85, 2.75, 1.7, 0.6, 'W panel j\n48×1 tiles 384 KiB\n(b_at = n_panel)', { size: 7.5, color: C.dram });
    D.arrow(s, Wd.x + Wd.cw * 3.5, Wd.y, 5.55, 2.75, { width: 1.5, color: C.dram, dash: 'dash' });
    D.text(s, 3.2, 2.2, 1.6, 0.22, 'GLOAD panel j', { size: 7.5, color: C.dram, bold: true, align: 'center' });
    const Ot = D.matrix(s, 8.3, 2.95, 0.6, 0.6, 1, 1, { kind: 'unit', title: '출력 tile (0, j)', titleW: 1.4, titleSize: 7.5, regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'unit', label: '48\nM_MUL', size: 7 }] });
    D.arrow(s, Wp.r + 0.05, Wp.cy, Ot.x, Ot.cy, { width: 1.5, color: C.unit });
    D.arrow(s, Xp.cx, Xp.b, Ot.cx, Ot.y, { width: 1.5, color: C.unit });
    D.text(s, 6.6, 3.5, 1.7, 0.4, 'X tile k × W tile k\nk = 0..47 (MAC 0,1,1,…)', { size: 7, color: C.muted });
    // output row
    const Y = D.matrix(s, 0.5, 4.35, 9.05, 0.32, 1, 32, { kind: 'faint', title: '출력 [64,8192] = 1 × 128 tiles (그림 32칸): tile j 완성(FP16 SAVE) → 다음 panel j+1', titleW: 6, regions: [{ r0: 0, r1: 1, c0: 7, c1: 8, kind: 'unit', label: 'j', size: 7 }, { r0: 0, r1: 1, c0: 0, c1: 7, kind: 'good', label: '완료', size: 7 }] });
    D.arrow(s, Ot.cx, Ot.b, Y.x + Y.cw * 7.5, Y.y, { width: 1.5, color: C.unit });
    D.bullets(s, 9.95, 1.6, 2.88, 3.1, [
      ['출력 tile 하나: ', 'M_MUL 48개 (K tile 48)'],
      ['출력 열 tile: ', '8192/64 = 128 → 총 6,144 M_MUL'],
      ['W 전송: ', '128 panel × 384 KiB = 48 MiB, 각 1회'],
      ['X 전송: ', '384 KiB 1회'],
      ['SRAM: ', 'X 384 KiB + W panel 384 KiB + 출력 tile + 임시 (나머지는 보관값·작업 공간)'],
    ], { size: 8.5, psa: 4 });
    D.table(s, 0.5, 5.0, 12.33, [
      ['Projection', '계산', '보존된 선택 (M=64)', '입력 배치', 'M_MUL'],
      ['gate / up', '[64,3072] × [3072,8192]', '("A", 64): input-stationary', 'X 384 KiB 유지, W [3072,64] 384 KiB × 128회 교체', '6,144 씩'],
      ['down', '[64,8192] × [8192,3072]', '("B", 256): weight-stationary', 'W [8192,256] 4 MiB 유지, 출력 64열 tile 4개 계산. H는 앞 커널 SRAM 보관값', '12 panel × 4 tile × 128 K = 6,144'],
    ], { colW: [1.0, 2.0, 2.1, 5.0, 2.23], size: 8.5, rowH: 0.4 });
    D.takeaway(s, 'K chunk 숫자만 보고 DMA 크기를 판단하면 안 됩니다. cache stage가 어느 loop에 있는지(a_at/b_at)가 실제 전송 영역과 재읽기 횟수를 정합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 32 search space
  D.add({ section: '6 · Matmul scheduling', title: '탐색 공간: m/n panel, K chunk, loop 순서, 그리고 두 operand의 cache 위치', subtitle: 'npu_schedule_space.candidates: 64×64 명령과 끊기지 않는 K chain은 제약이지 mapping 템플릿이 아닙니다. 출력 tile을 도는 순서와 각 operand의 cache 위치가 재읽기 횟수를 정합니다.', source: 'npu_compiler/npu_schedule_space.py (factors, loop_spec, candidates), evidence/current/refinement.json (09-28 기록)',
    notes: 'factors(extent, 64)는 padding한 축 길이의 64배수 약수입니다. 예를 들어 N=256이면 64/128/256, N=8192면 8개, K=3072면 10개입니다. product(m_panel, n_panel, k_chunk, outer_order, inner_order)에서 loop_spec은 항상 [outer panel 2개, inner tile 2개, k_group, k_tile] 구조를 만들어 K가 최내측입니다. 왼쪽 아래 격자는 출력 [128,256] = 2×4 tiles를 m_panel=n_panel=64로 도는 순서입니다: mn은 X 행 panel을 고정한 채 W 열 panel을 바꾸고(X 재사용), nm은 그 반대입니다. a_at/b_at는 cache stage를 넣을 loop이며, unit loop와 같은 축의 연속 loop는 실제 순회와 복사 주기가 같으면 signature로 합칩니다(다음 슬라이드). 추정 DMA는 FP16 operand bytes × 반복 횟수이며 검사 순서만 정하고 선택 기준은 실제 emission입니다. 오른쪽 아래 표의 "개선 전" 후보 수는 09-28 compile 기록이고 "현재" 수는 candidates()/prioritize()로 계산한 값입니다. 새 compile 기록이 아닙니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.5, 5.4, [
      ['설정', '의미'],
      ['m_panel / n_panel', '바깥에서 묶는 출력 행/열 수. 그 안은 64 단위 tile'],
      ['k_chunk', 'K를 묶는 크기. 그 안에서 64개씩 M_MUL'],
      ['outer_order', 'M panel과 N panel 중 무엇을 먼저 반복'],
      ['inner_order', 'panel 내부 M/N tile 순서'],
      ['a_at / b_at', 'X/W 복사 stage를 넣을 loop. 그 안에서 반복 사용하는 영역이 SRAM에 올라감'],
      ['resident', '이미 SRAM에 있는 operand: cache stage 없음, 위치 선택 1개로 축소'],
    ], { colW: [1.4, 4.0], size: 9, rowH: 0.38 });
    // loop nest diagram
    D.text(s, 6.15, 1.5, 6.5, 0.3, 'loop_spec: 항상 이 구조 (K 최내측)', { size: 11, bold: true, color: C.navy });
    const loops = [['m_panel / n_panel', 'outer_order', 'ir'], ['n_panel / m_panel', '', 'ir'], ['m_tile (64)', 'inner_order', 'sram'], ['n_tile (64)', '', 'sram'], ['k_group (k_chunk)', '', 'unit'], ['k_tile (64) → M_MUL', 'MAC chain', 'unit']];
    loops.forEach((l, i) => {
      const x = 6.15 + i * 0.28, y = 1.85 + i * 0.36;
      D.rect(s, x, y, 6.65 - i * 0.28, 0.32, { kind: l[2], radius: 0.03 });
      D.text(s, x + 0.1, y + 0.05, 4.5, 0.25, `for ${l[0]}:`, { size: 9.5, font: F.code, bold: true });
      if (l[1]) D.text(s, x + 4.5, y + 0.05, 2.0, 0.25, l[1], { size: 8.5, color: C.muted, align: 'right' });
    });
    D.text(s, 6.15, 4.1, 6.65, 0.5, 'a_at = "m_panel" 이면 X 복사 stage가 m_panel loop 안, 즉 그 panel의 X 행 전체(K 전체 포함)를 한 번 올림. b_at = "k_group"이면 W는 K chunk마다 다시 올림.', { size: 9.5, color: C.muted });
    // traversal order grids
    D.text(s, 0.5, 4.35, 5.4, 0.24, '출력 [128,256] = 2×4 tiles (panel 64)를 도는 순서 — 무엇이 SRAM에 남아 재사용되는가', { size: 9, bold: true, color: C.navy });
    D.grid(s, 0.5, 4.65, 2.2, 1.1, 2, 4, { kind: 'faint', label: (r, c) => String(r * 4 + c + 1), size: 9 });
    D.text(s, 0.5, 5.78, 2.2, 0.45, 'outer_order = mn: X 행 panel 고정, W 열 panel 교체 → X 재사용, W는 M panel 수만큼 재읽기', { size: 7, color: C.muted });
    D.grid(s, 3.1, 4.65, 2.2, 1.1, 2, 4, { kind: 'faint', label: (r, c) => String(c * 2 + r + 1), size: 9 });
    D.text(s, 3.1, 5.78, 2.2, 0.45, 'outer_order = nm: W 열 panel 고정, X 행 panel 교체 → W 재사용, X는 N panel 수만큼 재읽기', { size: 7, color: C.muted });
    D.text(s, 5.45, 4.65, 0.45, 1.1, '→', { size: 12, color: C.muted, align: 'center', valign: 'middle' });
    D.table(s, 6.15, 4.7, 6.68, [
      ['실제 폭 예', '개선 전 후보 (기록)', '현재 정규화 공간 (계산)', '기록의 선택'],
      ['gate/up [64,3072]×[3072,8192]', '1,433', '203 → 증명 제외 41 → 162', '합법 2개 뒤 word 예산 → incumbent ("A",64)'],
      ['down [64,8192]×[8192,3072], H resident', '299', '17 → 제외 5 → 12', '("B",256) 선택, 43,410 words'],
      ['작은 예제 [64,128]×[128,64]', '4', '4 (모두 추정 32,768 B)', 'a_at = b_at = m_panel 선택'],
    ], { colW: [2.45, 1.05, 1.55, 1.63], size: 8, rowH: 0.36 });
    D.text(s, 0.5, 6.25, 5.4, 0.25, '추정 DMA = Σ 2·K·(M|N)·repeats (operand가 쓰지 않는 축의 바깥 loop 반복 곱) → 검사 순서만 결정', { size: 7.5, color: C.muted, font: F.code });
    D.text(s, 6.15, 6.18, 6.68, 0.3, '"개선 전"은 09-28 compile 기록(refinement.json), "현재"는 candidates()/prioritize() 계산값(새 compile 기록 아님).', { size: 7.5, color: C.muted });
    D.takeaway(s, 'Factor는 padding한 축 길이의 64배수 약수에서 만듭니다 (N=256 → 64/128/256). 같은 순회의 다른 표기는 합치고, 증명된 용량 초과만 미리 빼며, 나머지는 예산으로 제한합니다(다음 두 슬라이드).', { y: 6.55 });
  });

  // ---- 32b candidate normalization & capacity filter (09-29)
  D.add({ section: '6 · Matmul scheduling', title: '후보 정규화와 용량 필터: 같은 tile 순회·복사 주기는 한 후보로, 증명 가능한 SRAM 초과만 미리 제외', subtitle: 'npu_schedule_space.candidates / prioritize (09-29 개선). 512-token down [512,8192]×[8192,3072]: 22,236 → 5,034 후보, 그중 1,596은 두 입력 cache의 합만으로 8 MiB를 넘어 제외, 3,438을 검사 목록에 남김 (E9 기록).', source: 'npu_compiler/npu_schedule_space.py (_iteration_signature, candidates, prioritize), npu_schedule.py (_fit_matmul의 ordinary 판정), docs/current/06_matmul_schedule.md, 11_experiments.md E8/E9',
    notes: '① loop_spec은 같은 축을 panel과 tile 두 loop로 나누므로 n_panel=64/128/256은 모두 N tile을 1,2,3,4 순서로 도는 같은 순회입니다. _iteration_signature는 크기 1인 loop를 빼고 같은 축의 이웃 loop를 곱해 (m2, n4, k2) 하나로 만듭니다. cache 위치는 그 loop까지의 prefix signature로 구분하므로, b_at=n_panel이라도 n_panel=64는 tile마다, 128은 2 tile마다, 256은 한 번 복사하는 서로 다른 후보로 남습니다. toy [128,256,128]에서 loop 위치 조합 1,728개가 후보 73개가 됩니다. ② prioritize는 두 operand cache가 matrix 명령 시점에 동시에 살아 있다는 사실만 사용합니다. 두 operand가 서로 다른 함수 parameter이고 FP16, scope global/global.sram, shape가 정확히 (M,K)/(K,N), M·N·K가 모두 64배수일 때만(ordinary) 두 cache bytes 합이 (SRAM − scratch prefix)를 넘는 후보를 제외합니다. padding이나 computed producer, W8A8처럼 증명이 성립하지 않으면 footprint 추정으로 순서만 뒤로 보내고 제거하지 않습니다. 출력·scratch·epilogue 추정은 제외 근거로 쓰지 않습니다. 표의 512-down 수치는 E9 기록이고 나머지 "계산" 수치는 현재 코드의 candidates()/prioritize()를 scratch 96 KiB 가정으로 실행한 값입니다.' },
  (s, D) => {
    // ① normalization
    D.text(s, 0.5, 1.5, 6.1, 0.26, '① 같은 순회의 다른 표기는 signature 하나로 — toy [128,256]×K128 (2×4 tiles), N 축의 세 표기', { size: 10, bold: true, color: C.navy });
    const spell = [['n_panel=64', ['m_panel 2', 'n_panel 4', 'm_tile 1', 'n_tile 1', 'k_group 2', 'k_tile 1']], ['n_panel=128', ['m_panel 2', 'n_panel 2', 'm_tile 1', 'n_tile 2', 'k_group 2', 'k_tile 1']], ['n_panel=256', ['m_panel 2', 'n_panel 1', 'm_tile 1', 'n_tile 4', 'k_group 2', 'k_tile 1']]];
    spell.forEach((sp, i) => {
      const cells = sp[1].map(t => ({ label: t, kind: t.endsWith(' 1') ? 'faint' : (t.startsWith('n_') ? 'sram' : (t.startsWith('k') ? 'unit' : 'ir')), size: 7, color: t.endsWith(' 1') ? C.muted : C.ink }));
      D.lanes(s, 1.55, 1.85 + i * 0.36, 4.0, 0.29, cells, { title: sp[0], titleW: 1.0, titleSize: 8 });
    });
    D.text(s, 5.62, 1.85, 1.0, 1.0, '→ N tile 순서\n1 2 3 4 동일\n(unit loop는\n순회를 안 바꿈)', { size: 7, color: C.muted, valign: 'middle' });
    D.matrix(s, 1.55, 3.28, 1.6, 0.3, 1, 4, { kind: 'sram', cellLabel: (r, c) => String(c + 1), cellSize: 8, title: '실제 순회 (m 2회 × n 4 tiles × k 2회)', titleW: 3.2, titleSize: 7.5 });
    D.arrow(s, 3.2, 3.43, 3.5, 3.43, { width: 1.2 });
    D.chip(s, 3.55, 3.3, 'signature (m2, n4, k2)', { kind: 'navy', color: C.white, w: 1.9, size: 8, h: 0.26 });
    D.text(s, 0.5, 3.68, 6.1, 0.55, 'cache 위치는 그 loop까지의 prefix signature로 구분: b_at = n_panel이면 n_panel=64는 tile마다, 128은 2 tile마다, 256은 한 번 복사 → 복사 주기가 다르므로 서로 다른 후보로 유지. 순회와 복사 주기가 모두 같은 표기끼리만 합칩니다.', { size: 8.5, color: C.ink });
    D.text(s, 0.5, 4.28, 6.1, 0.24, 'toy 전체: loop 위치 조합 1,728 → 후보 73 (현재 코드 계산). resident operand는 cache stage가 없어 위치 1개로 축소.', { size: 8.5, bold: true, color: C.navy });
    // ② capacity proof
    D.text(s, 6.85, 1.5, 5.98, 0.26, '② 용량 하한 증명: 두 입력 cache는 matrix 명령 시점에 동시에 살아 있다', { size: 10, bold: true, color: C.navy });
    const MB = 1048576, hi = 9.6 * MB;
    D.membar(s, 6.85, 1.88, 5.3, 0.36, 0, hi, [
      { from: 0, to: 98304, label: '', kind: 'host' },
      { from: 98304, to: 98304 + 8 * MB, label: 'X cache 8 MiB (H [512,8192] 전체, a_at=m_panel)', kind: 'red' },
      { from: 98304 + 8 * MB, to: 98304 + 9 * MB, label: 'W 1 MiB', kind: 'dram' },
    ], { size: 7.5 });
    D.text(s, 12.2, 1.88, 0.65, 0.36, '제외\n(증명)', { size: 7.5, bold: true, color: C.red, valign: 'middle' });
    D.membar(s, 6.85, 2.42, 5.3, 0.36, 0, hi, [
      { from: 0, to: 98304, label: '', kind: 'host' },
      { from: 98304, to: 98304 + 4 * MB, label: 'X cache 4 MiB (256행, m_panel=256)', kind: 'sram' },
      { from: 98304 + 4 * MB, to: 98304 + 5 * MB, label: 'W 1 MiB', kind: 'dram' },
    ], { size: 7.5 });
    D.text(s, 12.2, 2.42, 0.65, 0.36, '검사\n목록', { size: 7.5, bold: true, color: C.good, valign: 'middle' });
    const lim = 6.85 + 5.3 * (98304 + 8 * MB) / hi;
    D.line(s, lim, 1.8, lim, 2.86, { arrow: false, color: C.red, width: 1.5, dash: 'dash' });
    D.text(s, lim - 1.1, 2.82, 2.2, 0.2, '가용 = SRAM 8 MiB − scratch prefix (96 KiB)', { size: 7, color: C.red, align: 'center' });
    D.text(s, 6.85, 2.82, 0.6, 0.2, '0', { size: 7, color: C.muted });
    D.bullets(s, 6.85, 3.12, 5.98, 1.5, [
      ['증명 조건 (ordinary): ', '두 operand가 서로 다른 함수 parameter, FP16, scope global/global.sram, shape 정확히 (M,K)/(K,N), M·N·K 모두 64배수 → 두 cache bytes 합 > 가용이면 제외'],
      ['해당 안 되면 (padding, computed producer, W8A8): ', 'footprint 추정으로 순서만 뒤로(deprioritize), 제거하지 않음'],
      ['출력·scratch·epilogue 추정은 제외 근거가 아님: ', '남은 후보는 실제 TIR compaction·SRAM 배치·명령 생성으로 검사'],
    ], { size: 8.5, psa: 3 });
    // counts table
    D.table(s, 0.5, 4.72, 9.0, [
      ['계산 (M,K)×(K,N)', '개선 전 후보', '정규화 후', '용량 증명 제외', '검사 목록', '출처'],
      ['512-token down (512,8192)×(8192,3072)', '22,236', '5,034', '1,596', '3,438', 'E9 기록'],
      ['gate M=64 (64,3072)×(3072,8192)', '1,433', '203', '41', '162', '현재 코드 계산'],
      ['down M=64, H resident (64,8192)×(8192,3072)', '299', '17', '5', '12', '현재 코드 계산'],
      ['작은 예제 (64,128)×(128,64)', '4', '4', '0', '4', '기록 = 계산'],
    ], { colW: [3.3, 1.1, 1.0, 1.2, 1.0, 1.4], size: 8.3, rowH: 0.3, aligns: ['left', 'right', 'right', 'right', 'right', 'left'] });
    D.callout(s, 9.7, 4.72, 3.13, 1.5, [
      { text: 'E8에서 본 문제: ', bold: true }, { text: '추정 DMA만으로 정렬하면 SRAM에 안 들어가는 후보가 앞을 차지 → 512-down 확장 탐색 16회가 모두 용량 초과. 개선 후 같은 커널: 4회 검사 = 합법 4개 (E9).', size: 8.5 },
    ], { kind: 'warn' });
    D.text(s, 0.5, 6.25, 9.0, 0.22, '"개선 전"은 정규화 전 candidates() 기록(E8), "계산"은 현재 코드를 scratch 96 KiB 가정으로 실행한 값이며 새 compile 기록이 아닙니다.', { size: 7.5, color: C.muted });
    D.takeaway(s, '정규화는 표기 중복을 없애고, 용량 필터는 증명된 경우만 제외합니다. 남은 후보는 여전히 실제 TIR compaction·SRAM 배치·명령 생성으로 검사합니다.', { y: 6.55 });
  });

  // ---------------------------------------------------------------- 33 trial flow & budgets
  D.add({ section: '6 · Matmul scheduling', title: '후보 검사 흐름: DB 재생(선택) → incumbent → 정규화된 후보를 실제 schedule·SRAM 배치·명령 생성으로 검사', subtitle: '_fit_matmul. 실패·용량 초과도 시도로 세고, words 한도는 기준 후보 크기에 맞춰 늘어나며, 합법 후보 2개를 보기 전에는 words로 끝내지 않습니다. 검사 순서만 space(추정 DMA 순)와 meta_schedule(ReplayFunc 추첨)이 다릅니다.', source: 'npu_compiler/npu_schedule.py (_fit_matmul, _measure_trial, _probe_accumulator_chain), npu_schedule_space.prioritize, device.NpuProfile',
    notes: '순서: schedule_database가 있으면 workload(계산 구조 + 실행 조건)로 trace를 조회해 새 Schedule에 재적용하고 공통 lowering·SRAM 배치·용량·codegen을 다시 검사합니다. reuse 모드는 여기서 끝나고(database_hit, 새 후보 0회), tune 모드는 그 결과를 incumbent로 둡니다. DB가 없거나 실패하면 legacy 규칙(A/B/B_stream/None, legal 목표 2, 시도 64)으로 incumbent를 잡습니다. 확장 탐색은 candidates()로 정규화된 pool을 만들고 prioritize로 증명된 용량 초과를 뺀 뒤 space는 추정 DMA 순으로, meta_schedule은 ReplayFunc가 sample_categorical로 뽑은 순서로 검사합니다. 각 후보는 _measure_trial(trial cache)에서 tir.Schedule 변환과 TIR 후처리(LowerInitBlock → PlanAndUpdateBufferAllocationLocation → ConvertBlocksToOpaque → CompactBufferAllocation), lower_stores, forward_output_tiles, pointwise coalescing을 거친 뒤 plan_sram으로 USMP 배치를 얻습니다. SRAM top이 용량을 넘으면 capacity_rejected, MAC chain이 끊기면 codegen_rejected입니다. words 한도는 max(schedule_probe_word_budget 100,000, 기준 words × legal 목표 8)이며 기준은 incumbent words, 없으면 첫 합법 후보 words입니다. 종료 사유와 generated/eligible/examined/legal 수, 유효 words 한도가 진단에 남습니다. 승자가 DB 기록보다 좋으면(또는 기록이 없으면) trace를 commit합니다.' },
  (s, D) => {
    const y = 1.55;
    const st = [
      ['DB 조회 (선택)', 'trace 재생·재검증\nreuse: 즉시 채택 · tune: incumbent', 'host'],
      ['incumbent', 'legacy A / B / B_stream / None\n시도 64 · legal 2 (별도 예산)', 'ir'],
      ['후보 pool', 'candidates → 정규화\nprioritize: 증명된 초과 제외', 'ir'],
      ['_measure_trial', 'tir.Schedule + 후처리\nCompactBufferAllocation', 'sram'],
      ['plan_sram (USMP)', 'lifetime 충돌 → greedy 배치\ntop ≤ 8 MiB?', 'sram'],
      ['probe codegen', 'walker emission + peephole\n→ (dma, words, sram)', 'unit'],
    ];
    const w = (12.33 - 5 * 0.15) / 6;
    st.forEach((b, i) => {
      const x = 0.5 + i * (w + 0.15);
      D.box(s, x, y, w, 0.9, b[0], { kind: b[2], size: 9.5, detail: b[1], dsize: 7.5 });
      if (i < 5) D.arrow(s, x + w + 0.01, y + 0.45, x + w + 0.14, y + 0.45, { width: 1.2 });
    });
    // loop back: next candidate
    const xs = 0.5 + 2 * (w + 0.15) + w / 2, xe = 0.5 + 5 * (w + 0.15) + w / 2;
    D.line(s, xe, y + 0.9, xe, y + 1.05, { arrow: false, color: C.warn, dash: 'dash', width: 1.2 });
    D.line(s, xe, y + 1.05, xs, y + 1.05, { arrow: false, color: C.warn, dash: 'dash', width: 1.2 });
    D.line(s, xs, y + 1.05, xs, y + 0.92, { color: C.warn, dash: 'dash', width: 1.2 });
    D.text(s, xs + 0.1, y + 1.06, 6.0, 0.2, '다음 후보: space는 정렬 순서대로, meta_schedule은 sample_categorical 추첨 (중복 추첨은 시도만 소비)', { size: 7.5, color: C.warn, bold: true });
    // stop reasons
    D.text(s, 0.5, 2.85, 12.33, 0.22, '종료 사유 (diagnostics.stop_reason) — 먼저 오는 조건에서 멈춤', { size: 9.5, bold: true, color: C.navy });
    D.lanes(s, 0.5, 3.08, 12.33, 0.5, [
      { label: 'database_hit\nreuse 모드 DB 적중 → 새 후보 0회', kind: 'host', size: 7.5 },
      { label: 'legal_budget\nlegal ≥ 8 (schedule_space_trials)', kind: 'good', size: 7.5 },
      { label: 'word_budget\nlegal ≥ 2 ∧ Σwords ≥ max(100k, 기준 words×8)', kind: 'warn', size: 7.5 },
      { label: 'attempt_budget\n검사 수 ≥ 64 (실패·용량 초과 포함)', kind: 'red', size: 7.5 },
      { label: 'exhausted\n후보 목록 끝 (attempts=None이면 여기까지)', kind: 'faint', size: 7.5 },
    ], { size: 7.5 });
    D.table(s, 0.5, 3.75, 6.1, [
      ['후보 상태', '원인', '예'],
      ['schedule_rejected', 'Schedule API 실패, lower_stores 규칙 위반, ReplayFunc 중복 추첨', 'FP32 buffer의 비정형 store · "duplicate sample"'],
      ['capacity_rejected', 'USMP top > SRAM 용량 (보관 예약 제외한 cap)', '큰 panel 조합'],
      ['codegen_rejected', 'walker가 MAC chain 중단 감지 (LinkError)', '"FP32 matrix chain interrupted"'],
      ['legal → selected', '명령 생성 성공 → (dma, words, sram) 사전식 최소', 'incumbent보다 작을 때만 교체'],
    ], { colW: [1.45, 2.75, 1.9], size: 8.3, rowH: 0.4 });
    D.table(s, 6.85, 3.75, 5.98, [
      ['예산 (NpuProfile 기본값)', '값', '뜻'],
      ['schedule_search', '"space"', '"meta_schedule"=ReplayFunc 추첨 · "legacy"=기준만'],
      ['schedule_max_attempts', '64', '실패·용량 초과 포함. incumbent/확장 각각. None=무제한'],
      ['schedule_space_trials', '8', '확장 탐색의 legal 목표 (E10 실행기 16)'],
      ['schedule_candidates', '2', 'legacy incumbent 탐색의 legal 목표'],
      ['schedule_probe_word_budget', '100,000', 'max(설정, 기준 words×legal 목표), 후보 완료 후 검사'],
      ['schedule_seed', '0', 'ReplayFunc 추첨 seed (rand_state)'],
    ], { colW: [1.95, 0.75, 3.28], size: 8, rowH: 0.3 });
    D.text(s, 0.5, 5.98, 12.33, 0.42, [{ text: '"예산 내 합법 후보 없음"(attempt budget exhausted … within budget)과 "원리상 불가능"(no legal matmul schedule fits)은 다른 메시지입니다. ', options: { bold: true, color: C.warn, fontSize: 8.5, fontFace: F.body } }, { text: 'trial cache는 mapping 키로 결과(실패 포함)를 재사용하고 용량은 키 밖이라 다른 보관 예약에서도 재사용됩니다.', options: { color: C.muted, fontSize: 8.5, fontFace: F.body } }], { size: 8.5 });
    D.takeaway(s, '추정 DMA와 추첨은 검사 순서만 정합니다. 합법성과 비용은 실제 scheduled TIR, buffer compaction, SRAM 배치, 명령 생성으로 확인합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 34 schedule API table
  D.add({ section: '6 · Matmul scheduling', title: '표준 Schedule API가 실제로 바꾸는 것과 그 자체로 하지 않는 일', subtitle: 'tir.Schedule은 TIR를 받아 block/loop 변환을 적용합니다. 변환 뒤 loop 구조가 달라지므로 이전 handle을 그대로 쓰지 않습니다.', source: 'docs/current/06_matmul_schedule.md, reference/tvm_api.md, npu_compiler/npu_intrin.py schedule_matmul_sram',
    notes: 'schedule_matmul_sram이 적용하는 순서는 계약 검사 → 다른 producer(pad, dequant)의 cache_read/set_scope → 필요 시 pad_einsum → split(i,[None,m_panel])→split(·,[None,64]), j·k 동일 → reorder → cache_write + reverse_compute_at(출력 write-back) → fused epilogue의 reverse_compute_at/set_scope → compute_inline → operand cache_read + compute_at(a_at/b_at) → decompose_reduction(k_group) → tensorize 두 번(gemm, fill)입니다. compute_at이 주소 배치를 최적화하는 것은 아니며, buffer가 어디서 얼마나 쓰이는지 드러낸 뒤 compaction·USMP·보관 계획이 그 정보를 사용합니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.5, 7.3, [
      ['API', '실제 변화', '그 자체로 하지 않는 일'],
      ['split', 'loop 하나를 바깥/안쪽 loop로 분할 (predicate 포함)', 'SRAM 복사 생성'],
      ['reorder', '의존성이 허용하는 loop 순서 변경', 'DMA 비용 자동 최소화'],
      ['cache_read', '입력을 새 cache buffer에 복사하는 stage 생성', '복사의 적절한 반복 위치 결정'],
      ['compute_at', '생산 block을 소비 loop의 특정 위치 안으로 이동', 'ISA 명령으로 남기기'],
      ['reverse_compute_at', '소비 block을 생산 loop 위치로 이동', '무조건 in-place 실행'],
      ['cache_write', '출력 계산이 cache를 쓰고 원래 출력으로 복사', '최종 출력 ABI 제거'],
      ['set_scope', '기존 buffer의 storage scope 변경', '별개 복사 stage 추가'],
      ['pad_einsum', 'einsum 축을 tile 배수로 확장 (ragged shape)', '결과 의미 변경 (zero padding 증명 유지)'],
      ['decompose_reduction', 'init과 update 분리 → 어느 loop에서 누산 시작인지', '수치 순서 변경'],
      ['tensorize', '일치하는 조각을 등록한 intrinsic으로 치환', '임의 수식의 자동 ISA 구현'],
    ], { colW: [1.6, 3.3, 2.4], size: 9, rowH: 0.36 });
    D.text(s, 8.05, 1.5, 4.8, 0.3, 'schedule_matmul_sram의 실제 호출 순서', { size: 11, bold: true, color: C.navy });
    D.bullets(s, 8.05, 1.85, 4.78, 4.5, [
      '계약 검사: A/B FP16, C FP32, 소비자는 FP16 cast 하나, epilogue는 shape/FP16 유지',
      'pad 등 다른 producer: cache_read + set_scope(SRAM); ragged면 pad_einsum',
      'split(i,[None,m_panel]) → split(·,[None,64]); j, k 동일 → reorder(loop_spec 순서, ii, ji, ki)',
      'cache_write(C, SRAM) → set_scope → reverse_compute_at(enclosing tile loop); unpad·FP16 store도 reverse_compute_at',
      'fused epilogue(add/SiLU): reverse_compute_at + set_scope/cache_write; 입력은 cache_read + compute_at',
      'compute_inline(unpad, write_back)',
      'operand 0/1: cache_read → compute_at(a_at / b_at). SRAM resident면 생략',
      'decompose_reduction(block, k_group) → init block',
      'tensorize(k_tile loop, "npu_gemm_acc_sram"); tensorize(init, "npu_fill_zero_sram")',
      '후처리: LowerInitBlock → PlanAndUpdateBufferAllocationLocation → ConvertBlocksToOpaque → CompactBufferAllocation → lower_stores → forward_output_tiles → pointwise coalescing',
    ], { size: 9, psa: 3 });
    D.takeaway(s, 'compute_at 등은 IR을 바꾸는 명령이라 결과 TIR에 그 이름이 남지 않습니다. 대신 loop 안으로 이동한 block과 좁아진 buffer를 볼 수 있습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 35 compute_at example
  D.add({ section: '6 · Matmul scheduling', title: 'compute_at을 작은 예제로 보기: 필요한 생산을 사용 위치로 옮긴다', subtitle: 'B = A + 1, C = B × 2 (4×4 FP32 CPU 교육 예제, NPU tile이 2라는 뜻은 아님). C의 행 loop를 2행씩 나눈 뒤 B를 그 안으로 옮깁니다.', source: 'docs/examples/schedule_api/01_before.py, 02_compute_at.py, 02_compute_at.trace.txt, 03_reverse_compute_at.py',
    notes: '변환 전에는 B 전체를 만든 뒤 C 전체를 만들므로 B 4×4가 오래 살아 있습니다. split으로 i를 (2,2)로 나누고 B를 i_0 안에 compute_at하면 각 행 묶음에서 필요한 B 2행만 만들고 바로 C가 사용합니다. reverse_compute_at은 반대로 B를 만드는 loop 기준으로 소비자 C를 끌어옵니다. Matmul 뒤 add/SiLU나 SRAM 출력 복사 stage를 생산 tile 근처에 두려면 이 구분이 필요합니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 4, 0.3, '변환 전 (01_before)', { size: 11, bold: true, color: C.navy });
    D.code(s, 0.5, 1.8, 3.9, 2.9, [
      'B = T.alloc_buffer((4, 4))',
      'for i, j in T.grid(4, 4):',
      '  with T.block("B"):',
      '    B[i, j] = A[i, j] + 1.0',
      'for i, j in T.grid(4, 4):',
      '  with T.block("C"):',
      '    C[i, j] = B[i, j] * 2.0',
      '',
      '# B 전체(4×4) 생성 후',
      '# C 전체 생성 → B 수명 김',
    ], { size: 8.5 });
    D.text(s, 4.7, 1.5, 4, 0.3, 'compute_at 뒤 (02)', { size: 11, bold: true, color: C.navy });
    D.code(s, 4.7, 1.8, 3.9, 2.9, [
      'for i_0 in range(2):        # 2행 묶음',
      '  for ax0, ax1 in T.grid(2, 4):',
      '    with T.block("B"):',
      '      v_i = i_0*2 + ax0',
      '      B[v_i, ax1] = A[v_i, ax1] + 1.0',
      '  for i_1, j in T.grid(2, 4):',
      '    with T.block("C"):',
      '      v_i = i_0*2 + i_1',
      '      C[v_i, j] = B[v_i, j] * 2.0',
      '# 묶음마다 필요한 B 2행만 생성',
    ], { size: 8.5 });
    D.text(s, 8.9, 1.5, 4, 0.3, 'reverse_compute_at 뒤 (03)', { size: 11, bold: true, color: C.navy });
    D.code(s, 8.9, 1.8, 3.93, 2.9, [
      'for i_0 in range(2):',
      '  for i_1, j in T.grid(2, 4):',
      '    with T.block("B"):       # 생산자 기준',
      '      B[i_0*2+i_1, j] = A[..] + 1.0',
      '  for ax0, ax1 in T.grid(2, 4):',
      '    with T.block("C"):       # 소비자를 끌어옴',
      '      C[i_0*2+ax0, ax1] = B[..] * 2.0',
      '',
      '# compute_at: 생산자를 소비 위치로',
      '# reverse_compute_at: 소비자를 생산 위치로',
    ], { size: 8.5 });
    D.text(s, 0.5, 4.9, 6, 0.3, '적용한 API trace (02_compute_at.trace.txt)', { size: 11, bold: true, color: C.navy });
    D.code(s, 0.5, 5.2, 8.1, 1.15, [
      'b0 = sch.get_block(name="C", func_name="main")',
      'l1, l2 = sch.get_loops(block=b0)',
      'l3, l4 = sch.split(loop=l1, factors=[None, 2], preserve_unit_iters=True, disable_predication=False)',
      'b5 = sch.get_block(name="B", func_name="main")',
      'sch.compute_at(block=b5, loop=l3, preserve_unit_loops=False, index=-1)',
    ], { size: 8.5 });
    D.callout(s, 8.9, 4.9, 3.93, 1.45, [
      { text: 'NPU에서의 쓰임: ', bold: true }, { text: 'matmul 출력 tile 근처에 add/SiLU(epilogue)를 reverse_compute_at으로 배치하고, 입력 cache stage는 compute_at으로 a_at/b_at loop에 넣습니다.', size: 10 },
    ], { kind: 'sram' });
  });

  // ---------------------------------------------------------------- 36 cache_read + compaction
  D.add({ section: '6 · Matmul scheduling', title: 'cache_read → compute_at → CompactBufferAllocation: 복사 buffer가 실제 접근 범위로 줄어든다', subtitle: 'cache_read만 추가하면 입력 전체를 복사하는 buffer가 생깁니다. 복사 stage를 행 묶음 안에 넣으면 필요한 영역이 좁아지고, compaction이 allocation을 그 범위로 축소합니다.', source: 'docs/examples/schedule_api/04_cache_read.py, 05_cache_read_compute_at.py, 08_cache_read_compacted.py',
    notes: '아래 격자는 SRAM cache buffer의 실제 크기와 한 반복에서 채워지는 영역입니다. ①에서는 4×4 전체를 한 번에 복사하고, ②에서는 i_0마다 2행만 복사하지만 buffer 선언은 여전히 4×4이며, ③ CompactBufferAllocation이 접근 범위(2×4)로 선언을 줄입니다. 이 메커니즘이 실제 matmul schedule에서 panel 크기를 결정합니다. a_at을 m_panel loop에 두면 X cache buffer가 [m_panel, K]로, k_group loop에 두면 [m_panel, k_chunk]로 compaction됩니다. USMP는 compaction된 크기와 수명을 보고 SRAM offset을 배치합니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 4, 0.3, '① cache_read(C, 0, "global.sram")', { size: 11, bold: true, color: C.navy });
    D.code(s, 0.5, 1.8, 3.9, 2.35, [
      'A_global_sram = T.alloc_buffer(',
      '    (4, 4), scope="global.sram")',
      'for ax0, ax1 in T.grid(4, 4):',
      '  with T.block("A_global.sram"):',
      '    A_global_sram[v0,v1] = A[v0,v1]',
      'for i, j in T.grid(4, 4):',
      '  with T.block("C"):',
      '    C[i,j] = A_global_sram[i,j] + 1.0',
    ], { size: 8.5 });
    D.text(s, 4.7, 1.5, 4, 0.3, '② compute_at(copy stage, i_0)', { size: 11, bold: true, color: C.navy });
    D.code(s, 4.7, 1.8, 3.9, 2.35, [
      'A_global_sram = T.alloc_buffer(',
      '    (4, 4), scope="global.sram")   # 아직 4×4',
      'for i_0 in range(2):',
      '  for ax0, ax1 in T.grid(2, 4):',
      '    with T.block("A_global.sram"):',
      '      v0 = i_0*2 + ax0',
      '      A_global_sram[v0,v1] = A[v0,v1]',
      '  for i_1, j in T.grid(2, 4):',
      '    with T.block("C"):',
      '      C[i_0*2+i_1, j] = A_global_sram[..] + 1.0',
    ], { size: 8.5 });
    D.text(s, 8.9, 1.5, 4, 0.3, '③ CompactBufferAllocation', { size: 11, bold: true, color: C.navy });
    D.code(s, 8.9, 1.8, 3.93, 2.35, [
      'for i_0 in range(2):',
      '  with T.block(""):',
      '    T.reads(A[i_0*2:i_0*2+2, 0:4])',
      '    A_global_sram = T.alloc_buffer(',
      '        (2, 4), scope="global.sram")  # 2×4',
      '    for ax0, ax1 in T.grid(2, 4):',
      '      A_global_sram[i_0*2+ax0 - i_0*2, ax1]',
      '          = A[i_0*2+ax0, ax1]',
      '    for i_1, j in T.grid(2, 4):',
      '      C[i_0*2+i_1, j] = A_global_sram[..] + 1.0',
    ], { size: 8.5 });
    D.arrow(s, 4.42, 3.0, 4.68, 3.0, { width: 2 }); D.arrow(s, 8.62, 3.0, 8.88, 3.0, { width: 2 });
    // SRAM buffer grids
    const gy = 4.55;
    D.text(s, 0.5, gy - 0.28, 12, 0.24, 'SRAM cache buffer의 실제 모양 (진한 칸 = 한 반복(i_0 = 0)에서 채워지는 영역)', { size: 9.5, bold: true, color: C.navy });
    D.matrix(s, 1.3, gy, 1.6, 1.3, 4, 4, { kind: 'sram', regions: [{ r0: 0, r1: 4, c0: 0, c1: 4, kind: 'sram', label: '4×4 전체 복사\n(16 원소)', size: 8 }], below: 'buffer (4,4): 입력 전체', belowW: 2.0 });
    D.text(s, 3.05, gy + 0.4, 1.4, 0.5, '→ compute_at', { size: 8, color: C.muted, align: 'center' });
    D.matrix(s, 4.7, gy, 1.6, 1.3, 4, 4, { kind: 'faint', regions: [{ r0: 0, r1: 2, c0: 0, c1: 4, kind: 'sram', label: 'i_0 = 0: 2행만 복사', size: 8 }, { r0: 2, r1: 4, c0: 0, c1: 4, kind: 'faint', label: '(다음 반복에 사용)', size: 7, color: C.muted, lineWidth: 0.4 }], below: 'buffer 선언은 아직 (4,4)', belowW: 2.2 });
    D.text(s, 6.45, gy + 0.4, 1.4, 0.5, '→ compaction', { size: 8, color: C.muted, align: 'center' });
    D.matrix(s, 8.9, gy + 0.32, 1.6, 0.65, 2, 4, { kind: 'sram', regions: [{ r0: 0, r1: 2, c0: 0, c1: 4, kind: 'sram', label: '2×4 (8 원소)', size: 8 }], below: 'buffer (2,4): 접근 범위로 축소 → USMP가 이 크기로 offset 배치', belowW: 3.6 });
    D.text(s, 10.7, gy - 0.05, 2.15, 1.4, 'Matmul에서: a_at = m_panel → X cache [m_panel, K] (K 전체), a_at = k_group → [m_panel, k_chunk]. 같은 원리로 panel 크기가 정해집니다.', { size: 8, color: C.muted });
    D.callout(s, 0.5, 6.0, 12.33, 0.4, 'set_scope(기존 buffer scope 변경)와 cache_write 예제도 같은 폴더(docs/examples/schedule_api)에 있습니다.', { kind: 'faint', size: 9 });
  });

  // ---------------------------------------------------------------- 37 tensorize
  D.add({ section: '6 · Matmul scheduling', title: 'Tensorize: 64×64 계산 조각을 등록한 TensorIntrin으로 치환한다', subtitle: '수식·dtype·scope가 일치하면 npu_gemm_acc_sram / npu_fill_zero_sram으로 바뀝니다. 이는 최종 opcode가 아니라 codegen이 descriptor·LOAD·M_MUL로 펼칠 call_extern입니다.', source: 'npu_compiler/npu_intrin.py (_sram_gemm_desc/_impl, register_all), docs/examples/pipeline/07_scheduled_fused_matmul_add_silu.py',
    notes: 'description은 A,B (64,64) FP16, C (64,64) FP32, 모두 scope global.sram, strides [s,1] symbolic, offset_factor 1로 등록됩니다. symbolic stride 덕분에 64×64 view가 더 넓은 panel 안에 있을 수 있습니다. implementation은 call_extern("npu_gemm_acc", C.ptr, sc, A.ptr, sa, B.ptr, sb)입니다. fill intrinsic은 C=0을 call_extern("npu_fill_zero")로 바꾸며, codegen에서 이것은 명령을 내지 않고 새 누산 시작 상태로만 기록되어 첫 M_MUL이 MAC=0이 됩니다. register_all은 import 시 실행되고 구조 동일성 검사로 중복 등록을 막습니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.3, 'TensorIntrin 등록 (description ↔ implementation)', { size: 11, bold: true, color: C.navy });
    D.code(s, 0.5, 1.8, 6.1, 2.7, [
      '# description: 계산의 수식 (schedule이 이 모양을 찾음)',
      'A = T.match_buffer(a, (64,64), "float16", scope="global.sram", strides=[sa,1], offset_factor=1)',
      'B = ... (64,64) "float16" ; C = ... (64,64) "float32"',
      'for i, j, k in T.grid(64, 64, 64):',
      '    C[vi, vj] = C[vi, vj] + Cast(f32, A[vi, vk]) * Cast(f32, B[vk, vj])',
      '',
      '# implementation: 장치 호출 (codegen이 해석)',
      'T.call_extern("int32", "npu_gemm_acc", C.access_ptr("rw"), sc,',
      '              A.access_ptr("r"), sa, B.access_ptr("r"), sb)',
      '# fill: C[vi,vj] = 0  →  T.call_extern("npu_fill_zero", C.access_ptr("w"), sc)',
      'TensorIntrin.register("npu_gemm_acc_sram", desc, impl)   # register_all()',
    ], { size: 8.2 });
    D.text(s, 6.85, 1.5, 6, 0.3, '실제 scheduled TIR 발췌 (작은 예제 fused 커널)', { size: 11, bold: true, color: C.navy });
    D.code(s, 6.85, 1.8, 5.98, 2.7, [
      'x_global_sram  = T.alloc_buffer((64,128), "float16", scope="global.sram")',
      'lv_global_sram = T.alloc_buffer((128,64), "float16", scope="global.sram")',
      '... x/lv 복사 block (K loop 바깥: a_at = b_at = m_panel) ...',
      'with T.block("matmul_init_o"):',
      '    C = T.match_buffer(matmul_global_sram[0:64, 0:64], (64,64), "float16", strides=("C_s0",1), ...)',
      '    T.call_extern("int32", "npu_fill_zero", T.tvm_access_ptr(..., C.data, ...), C.strides[0])',
      'for k_0, k_1_0 in T.grid(2, 1):',
      '    with T.block("matmul_update_o"):',
      '        A = T.match_buffer(x_global_sram[0:64, k_0*64 : k_0*64+64], (64,64), ...)',
      '        B = T.match_buffer(lv_global_sram[k_0*64 : k_0*64+64, 0:64], (64,64), ...)',
      '        T.call_extern("int32", "npu_gemm_acc", C.ptr, C.strides[0], A.ptr, A.strides[0], B.ptr, B.strides[0])',
    ], { size: 7.8 });
    D.table(s, 0.5, 4.75, 12.33, [
      ['이름', '어디에 나타나는가', '뜻'],
      ['npu_gemm_acc_sram / npu_fill_zero_sram', 'TensorIntrin 이름 (tensorize 인자)', '등록된 desc/impl 쌍'],
      ['npu_gemm_acc / npu_fill_zero', 'scheduled TIR의 call_extern', 'codegen이 descriptor 설정 + LOAD SRC1/SRC2 + M_MUL(MAC)로, fill은 명령 없이 "새 누산" 상태'],
      ['C의 dtype이 FP16?', 'match_buffer가 FP16 SRAM view', 'lower_stores가 FP32 논리 buffer를 FP16 물리 slot으로 retype (실제 FP32 상태는 accumulator)'],
      ['strides=("C_s0",1)', 'symbolic row stride', '64×64 view가 더 넓은 panel/출력 안에 위치 가능 (직접 저장의 근거, 8장)'],
    ], { colW: [2.6, 3.0, 6.73], size: 9, rowH: 0.36 });
    D.takeaway(s, 'Tensorize는 등록한 수식과 정확히 일치하는 조각만 치환합니다. 틀린 수식을 이름만으로 치환하지 않습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 38 actual choices
  D.add({ section: '6 · Matmul scheduling', title: '실제 선택 기록: 작은 예제의 4개 후보와 실제 폭 MLP의 incumbent', subtitle: '작은 예제는 X/W 16 KiB씩을 K loop 바깥에서 한 번 올리고 2개의 M_MUL이 같은 SRAM 배열의 다른 부분을 읽습니다. 실제 폭 MLP는 정적 compile 기록입니다(executed=false). 후보 수는 개선 전 탐색 공간의 기록이며 현재 정규화 공간에서는 203 / 17개입니다.', source: 'docs/examples/pipeline/bundle/program.json schedules, evidence/current/refinement.json, mlp64.json',
    notes: '작은 예제에서 space 후보 4개는 모두 추정 32,768 B이고, legacy incumbent ("B_stream",64)는 1,574 words, 선택된 a_at=b_at=m_panel 후보는 1,572 words로 DMA가 같고 words에서 이겼습니다. 선택 후보의 SRAM peak(85,016 nibbles)가 incumbent(68,632)보다 크지만 사전식 비교에서 SRAM은 세 번째 기준입니다. 실제 폭 MLP(M=64)는 5커널, 140,594 words, DMA 152,174,604 B, SRAM peak 7,446,540 B, compile 21.96 s로 기록되어 있으며 전체 모델 출력 검증이 아닙니다. 표의 후보 수 1,433/299는 09-28 기록의 정규화 전 공간이며, 현재 코드의 candidates()는 같은 shape에서 203개(용량 증명 제외 후 162)와 H resident 17개(제외 후 12)를 만듭니다. 기록의 선택 자체는 그대로입니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.3, '작은 예제 fused_matmul_add_silu [64,64,128]', { size: 11, bold: true, color: C.navy });
    D.table(s, 0.5, 1.82, 6.2, [
      ['후보', 'mapping', '추정 B', 'words', 'DMA B', 'SRAM nib', 'status'],
      ['incumbent', '("B_stream", 64)', '32,768', '1,574', '41,088', '68,632', 'legal'],
      ['space #1', 'm64 n64 k64, a_at=b_at=k_group', '32,768', '1,575', '41,088', '68,632', 'legal'],
      ['선택', 'm64 n64 k64, mn/mn, a_at=b_at=m_panel', '32,768', '1,572', '41,088', '85,016', 'legal ✓'],
    ], { colW: [0.8, 2.3, 0.65, 0.6, 0.7, 0.7, 0.45], size: 8, rowH: 0.34, aligns: ['left', 'left', 'right', 'right', 'right', 'right', 'left'] });
    D.bullets(s, 0.5, 3.3, 6.2, 1.5, [
      'DMA가 같으면 words로, 그 다음에야 SRAM peak로 비교 → SRAM을 더 써도 words가 적은 후보가 이김',
      'a_at=b_at=m_panel: X [64,128]·W [128,64] 16 KiB씩 K loop 바깥에서 1회 복사, k_0=0,1의 두 M_MUL이 같은 SRAM 배열의 다른 열/행을 읽음',
      '최종 커널: 4 DMA packets, 2 M_MUL, 1,572 words (link 뒤 전체 1,577 words)',
    ], { size: 9.5, psa: 4 });
    D.text(s, 6.95, 1.5, 6, 0.3, '실제 폭 Llama MLP (M=64, hidden 3072, MLP 8192) 정적 compile', { size: 11, bold: true, color: C.navy });
    D.table(s, 6.95, 1.82, 5.88, [
      ['커널', '선택 mapping', '후보 수 (개선 전)', 'words', '비고'],
      ['gate', '("A", 64) incumbent', '1,433', '≈47,000', 'X 유지, W panel 3072×64. legal 2개 뒤 word 예산 도달'],
      ['up', '("A", 64)', '1,433', '≈47,000', 'gate와 같은 구조 (recipe 공유)'],
      ['down', '("B", 256)', '299', '43,410', 'W 8192×256 4 MiB 유지, H는 SRAM resident'],
    ], { colW: [0.55, 1.35, 1.0, 0.75, 2.23], size: 8.5, rowH: 0.44 });
    D.table(s, 6.95, 3.75, 5.88, [
      ['전체 (5 kernels: gate → SiLU → up → mul → down)', '값'],
      ['ISA words', '140,594'],
      ['DMA bytes', '152,174,604 (≈145 MiB; weight 3×48 MiB + X/H 이동)'],
      ['SRAM peak bytes', '7,446,540 (8 MiB 중)'],
      ['DRAM image bytes', '151,781,388'],
      ['compile seconds (단일 관측)', '21.96, attempts 64, executed=false'],
    ], { colW: [3.3, 2.58], size: 8.5, rowH: 0.3 });
    D.callout(s, 0.5, 5.0, 12.33, 1.3, [
      { text: 'Gate 하나의 M_MUL 수: ', bold: true }, { text: '출력 열 tile 128 × K tile 48 = 6,144. 세 projection 합 18,432 = mlp64.json의 opcode 0x42 count(18,432)와 일치합니다. 0x80(주소 descriptor) 82,093 words와 0x90(LOAD) 37,504 words가 프로그램의 대부분을 차지하며, 이것이 "정적 전개 명령 stream"의 비용 구조입니다.', size: 10 },
    ], { kind: 'faint' });
    D.takeaway(s, '실제 폭에서는 후보가 수백~수천 개라 예산이 먼저 끝나고 incumbent가 남는 경우가 있습니다. 예산 안의 결과이지 전 공간의 최적이 아닙니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 39 long prefill arithmetic
  D.add({ section: '6 · Matmul scheduling', title: '긴 prefill(S=3072)에서 X와 W가 모두 큰 경우: 재읽기 횟수가 비용을 정한다', subtitle: '설명용 계산이며 새 compile 측정이 아닙니다. X 18 MiB, gate W 48 MiB. X는 64행(384 KiB), W는 256열(1.5 MiB)씩 올린다고 가정합니다.', source: 'docs/current/06_matmul_schedule.md (설명용 산술)',
    notes: 'X panel을 바깥에 유지하면 48개 X panel마다 W 전체를 열 panel로 순회하므로 X 18 MiB + W 48 MiB × 48회입니다. W panel을 바깥에 두면 32개 W panel마다 X 전체를 행 panel로 순회하므로 W 48 MiB + X 18 MiB × 32회입니다. 원래 tensor 크기는 같지만 재읽기 횟수가 다릅니다. 실제 탐색은 더 큰 panel과 다른 cache 위치도 검사하고 앞 커널의 SRAM 입력을 쓸 수 있으면 그 부분의 DRAM 읽기도 제외합니다. 그래프 전체의 tile streaming은 지원하지 않으므로 이 산술을 장문 전체 모델의 최적 실행 결과로 해석하지 않습니다.' },
  (s, D) => {
    const y = 1.6;
    D.box(s, 0.5, y, 5.9, 0.5, 'X panel을 바깥에 유지 (a_at 바깥, b_at 안쪽)', { kind: 'sram', size: 11.5 });
    D.grid(s, 0.5, y + 0.65, 1.2, 2.4, 6, 1, { highlight: (r) => r === 1 ? 'sram' : 'faint', label: (r) => r === 1 ? 'X 64행' : (r === 5 ? '…48' : ''), size: 7.5 });
    D.grid(s, 1.9, y + 0.65, 4.5, 0.6, 1, 8, { kind: 'dram', label: (r, c) => c === 0 ? 'W 256열' : c === 7 ? '…32' : '', size: 7.5 });
    D.text(s, 1.9, y + 1.35, 4.5, 0.9, 'X panel 하나(384 KiB)가 SRAM에 있는 동안 W 열 panel 32개(1.5 MiB씩)를 차례로 올림 → X panel마다 W 48 MiB 재전송', { size: 9.5, color: C.muted });
    D.text(s, 1.9, y + 2.3, 4.5, 0.6, '전송량 ≈ X 18 MiB + W 48 MiB × 48 = 2,322 MiB', { size: 11, bold: true, color: C.navy });
    D.box(s, 6.93, y, 5.9, 0.5, 'W panel을 바깥에 유지 (b_at 바깥, a_at 안쪽)', { kind: 'dram', size: 11.5 });
    D.grid(s, 6.93, y + 0.65, 4.5, 0.6, 1, 8, { highlight: (r, c) => c === 1 ? 'dram' : 'faint', label: (r, c) => c === 1 ? 'W 256열' : c === 7 ? '…32' : '', size: 7.5 });
    D.grid(s, 11.63, y + 0.65, 1.2, 2.4, 6, 1, { kind: 'sram', label: (r) => r === 0 ? 'X 64행' : r === 5 ? '…48' : '', size: 7.5 });
    D.text(s, 6.93, y + 1.35, 4.5, 0.9, 'W panel 하나(1.5 MiB)가 SRAM에 있는 동안 X 행 panel 48개(384 KiB씩)를 차례로 올림 → W panel마다 X 18 MiB 재전송', { size: 9.5, color: C.muted });
    D.text(s, 6.93, y + 2.3, 4.5, 0.6, '전송량 ≈ W 48 MiB + X 18 MiB × 32 = 624 MiB', { size: 11, bold: true, color: C.navy });
    D.table(s, 0.5, 4.85, 12.33, [
      ['가정', '바깥에 유지', '재읽기 때문에 생기는 전송', '합계 (설명용)'],
      ['X 64행 panel, W 256열 panel', 'X panel', 'X 18 MiB + W 48 MiB × 48회', '≈ 2.3 GiB'],
      ['X 64행 panel, W 256열 panel', 'W panel', 'W 48 MiB + X 18 MiB × 32회', '≈ 0.6 GiB'],
      ['더 큰 panel (예: W 1024열 6 MiB)', 'W panel', 'W 48 MiB + X 18 MiB × 8회', '≈ 0.19 GiB (SRAM 8 MiB 중 나머지가 출력·임시·보관에 필요)'],
    ], { colW: [3.0, 1.5, 3.5, 4.33], size: 9, rowH: 0.34 });
    D.takeaway(s, 'SRAM을 채우는 양보다 재사용 범위와 재읽기 횟수가 중요합니다. 실제 탐색은 여기에 앞 커널의 SRAM 입력(보관값)까지 포함해 비교합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 40 accumulator chain rules
  D.add({ section: '6 · Matmul scheduling', title: '누산 체인 검사: codegen이 MAC chain을 보장하고, lower_stores가 FP32 buffer 사용을 제한한다', subtitle: '"SRAM에 공간이 충분하다"만으로 schedule이 실행 가능한 것은 아닙니다. 출력 tile의 K 누적을 끊는 후보는 세 단계 중 하나에서 거절됩니다.', source: 'npu_compiler/tir_codegen_v09.py (emit_gemm, flush), npu_accumulator.py (lower_stores, forward_output_tiles), npu_sram.py (pinned)',
    notes: 'emit_gemm은 C가 zeroed 집합에 있으면 이전 tile을 flush하고 pending=(C, stride)로 두어 MAC=0을 냅니다. 같은 C가 이어지면 MAC=1이고, 다른 C가 오면 "FP32 matrix chain interrupted"를 raise합니다. flush(=FP16 SAVE)는 vector op, reduction, SRAM→SRAM 이동, write-back DMA 전에, 그리고 kernel 끝에 실행되며 입력 GLOAD staging은 flush하지 않습니다. lower_stores는 FP32 buffer의 store를 copy/zero/acc+cast(fp16)만 허용하고 load는 FP16 cast 아래에서만 허용해 FP16 물리 slot으로 retype합니다. USMP는 matrix 목적지 buffer를 pin해 SAVE가 지연되는 동안 다른 buffer와 겹치지 않게 합니다.' },
  (s, D) => {
    D.code(s, 0.5, 1.5, 6.3, 3.0, [
      '# tir_codegen_v09.Walker (Python; native v09_walker.cc 동일 계약)',
      'def call("npu_fill_zero", C):  self.zeroed.add(C)    # 명령 없음',
      'def emit_gemm(C, A, B):',
      '    if C in zeroed:      flush(); pending = C; first = True   # MAC=0',
      '    elif pending == C:   first = False                         # MAC=1',
      '    else: raise LinkError("FP32 matrix chain interrupted: ',
      '                           resuming from an FP16 store would change the reduction")',
      '    region(SRC1, A); LOAD(matrix, SRC1); region(SRC2, B); LOAD(matrix, SRC2)',
      '    M_MUL(mode=VECTOR, activation=OFF, mac=not first)',
      'def flush():   # 완료된 tile을 FP16으로 SAVE',
      '    region(DST, pending); SAVE(matrix=1)',
      '# flush 시점: vector op / reduction / SRAM→SRAM 이동 / write-back DMA 전, kernel 끝',
      '# (입력 GLOAD staging은 flush하지 않음)',
    ], { size: 8.2 });
    D.table(s, 7.05, 1.5, 5.78, [
      ['검사 층', '규칙', '위반 시'],
      ['npu_accumulator.lower_stores', 'FP32 buffer store는 copy / zero / acc + Cast(f32, fp16)만. load는 FP16 cast 아래에서만', 'schedule_rejected ("unsupported FP32 accumulator use", "escapes without FP16 store")'],
      ['Walker.emit_gemm', '한 C tile의 K chain 사이에 flush를 일으키는 명령 금지', 'codegen_rejected (LinkError, native: invalid_schedule → fallback 없음)'],
      ['npu_sram.plan_sram', 'gemm/fill 목적지 buffer는 pinned: 모든 buffer와 충돌로 취급', 'SAVE 지연 중 덮어쓰기 방지'],
      ['forward_output_tiles', '출력이 SRAM에 보관되면 fill+gemm view를 출력에 직접 → identity copy 제거', '조건 불충족 시 임시 tile 유지'],
    ], { colW: [1.6, 2.4, 1.78], size: 8.5, rowH: 0.62 });
    D.table(s, 0.5, 4.7, 6.3, [
      ['chain을 끊는 예', '왜'],
      ['K step 사이의 pad fill / dequantize (vector op)', 'vector 명령 전 flush → 누산 상태 손실'],
      ['K step 사이의 SRAM→SRAM copy, bounce DMA', '이동 전 flush'],
      ['두 출력 tile을 번갈아 누산', 'accumulator 1개: 다른 C가 오면 LinkError'],
    ], { colW: [3.3, 3.0], size: 9, rowH: 0.34 });
    D.callout(s, 7.05, 4.7, 5.78, 1.6, [
      { text: '기록: ', bold: true }, { text: '전체 Llama compile에서 native_rejections["FP32 matrix chain interrupted"] = 5. 이런 후보는 조용한 fallback이 아니라 invalid_schedule로 거절되어 다음 후보로 넘어갑니다. 내부 오류·계약 hash 불일치는 별도로 propagate됩니다.', size: 10 },
    ], { kind: 'warn' });
  });

  // ---- 40b MetaSchedule (optional): ReplayFunc ordering and JSONDatabase trace reuse
  D.add({ section: '6 · Matmul scheduling', title: 'MetaSchedule 연결(선택형): ReplayFunc 추첨 순서와 JSONDatabase trace 재생', subtitle: '기본은 space(추정 DMA 순). schedule_search="meta_schedule"는 같은 후보 pool에서 sample_categorical로 번호를 뽑고, compilation(schedule_database=…)는 실행 조건이 같은 커널의 trace를 저장·재검증합니다. 학습형 cost model·latency 측정은 없습니다.', source: 'npu_compiler/npu_meta_schedule.py (replay_candidates, SelectedScheduleDatabase), npu_schedule.py _fit_matmul, evidence/current/meta_schedule_reuse.json, docs/current/06_matmul_schedule.md',
    notes: 'ReplayFunc 경로: TuneContext(mod=canonical PrimFunc, target=npu, space_generator=ScheduleFn(make_schedule, sch_rules=[], postprocs=[], mutator_probs={}), search_strategy=ReplayFunc(), rand_state=schedule_seed, num_threads=1)를 만들고 pre_tuning(max_trials=schedule_max_attempts, num_trials_per_iter=1)합니다. generate_measure_candidates마다 make_schedule이 sample_categorical(range(len(pool)), 균등)로 번호를 뽑아 그 mapping으로 schedule_matmul_sram을 실행합니다. 같은 번호가 다시 나오면 schedule을 다시 만들지 않고 "duplicate sample"로 시도만 소비합니다. postprocs를 비워 TVM 내부의 무제한 재시도를 막고, 거절은 공통 NPU 검사 경로에서 합니다. notify_runner_results에는 RunnerResult(None, None)을 주어 latency를 넣지 않고, 승자는 기존 (dma, words, sram) 평가기가 고릅니다. DB 경로: SelectedScheduleDatabase는 compiler source sha·TVM 버전/commit·contract·native library sha의 fingerprint로 디렉터리를 나누고 JSONDatabase(module_equality="structural")를 엽니다. workload는 global_symbol·op_pattern을 뺀 PrimFunc에 npu.selected_schedule_contract(profile에서 탐색 설정 joint_memory/beam/graph_search_seconds/candidates/trials/attempts/words/search/seed를 뺀 것 + scratch_base + slot_count + native_probe)를 붙인 module이며, 조회는 structural hash와 equal입니다. 기록은 TuningRecord(trace, workload, run_secs=None, target)이고 같은 workload의 최신 개선 기록을 사용합니다. 재사용 시 trace.apply_to_schedule → 공통 lowering → plan_sram → 용량 → probe codegen을 다시 거치며 실패하면 database_rejection을 남기고 원래 탐색으로 돌아갑니다. 작은 예제 3개에서 세 경로의 명령이 같고 C-model 출력이 bit 단위로 같았습니다. Attention 전체·일반 vector·W8A8은 DB 대상이 아닙니다.' },
  (s, D) => {
    // left: same pool, two visiting orders
    D.text(s, 0.5, 1.5, 6.1, 0.24, '같은 pool (정규화·용량 필터 후, 추정 DMA 순 index) — 검사 순서만 다르고 평가는 공통', { size: 9.5, bold: true, color: C.navy });
    const px = 1.6, pw = 5.0, N = 12, cw = pw / N;
    const cells = []; for (let i = 0; i < N; i++) cells.push({ label: String(i), kind: 'faint', size: 8, mono: true });
    D.lanes(s, px, 2.3, pw, 0.34, cells, { title: '후보 pool', titleW: 1.0, titleSize: 8.5 });
    // space: sequential
    D.text(s, 0.5, 1.78, 1.05, 0.4, 'space', { size: 8.5, bold: true, color: C.sram, align: 'right', valign: 'middle' });
    [0, 1, 2, 3].forEach((i, k) => {
      const cx = px + (i + 0.5) * cw;
      D.arrow(s, cx, 1.98, cx, 2.28, { width: 1.5, color: C.sram });
      D.text(s, cx - 0.2, 1.76, 0.4, 0.2, ['①', '②', '③', '④'][k], { size: 8, color: C.sram, bold: true, align: 'center' });
    });
    D.text(s, px + 4.5 * cw, 1.78, 3.0, 0.4, '… 정렬 순서대로 (legal 8 / words / 64회까지)', { size: 7.5, color: C.sram, valign: 'middle' });
    // ReplayFunc: sampled
    D.text(s, 0.5, 2.72, 1.05, 0.4, 'ReplayFunc\nseed 0', { size: 8, bold: true, color: C.unit, align: 'right', valign: 'middle' });
    const picks = [[7, '①'], [2, '②'], [7, '③ 중복'], [10, '④'], [0, '⑤']];
    picks.forEach(([i, t]) => {
      const dup = t === '③ 중복';
      const cx = px + (i + 0.5) * cw + (dup ? 0.12 : 0);
      D.arrow(s, cx, dup ? 3.22 : 3.05, cx, 2.66, { width: 1.5, color: dup ? C.red : C.unit });
      D.text(s, cx - 0.35, dup ? 3.22 : 3.05, 0.7, 0.2, t, { size: 7.5, color: dup ? C.red : C.unit, bold: true, align: 'center' });
    });
    D.text(s, px, 3.44, pw, 0.36, 'sample_categorical(range(N), 균등) → 번호 → 그 mapping으로 schedule_matmul_sram. 중복 번호는 schedule을 다시 만들지 않고 시도만 소비. 시도 한도는 유한해야 함.', { size: 7.5, color: C.muted });
    // common evaluator
    const ev = [['schedule_matmul_sram', 'ir'], ['_lower_scheduled', 'ir'], ['plan_sram · 용량', 'sram'], ['probe codegen', 'unit'], ['(dma, words, sram)', 'warn']];
    const ew = (6.1 - 4 * 0.12) / 5;
    ev.forEach((b, i) => { const x = 0.5 + i * (ew + 0.12); D.box(s, x, 3.86, ew, 0.4, b[0], { kind: b[1], size: 8 }); if (i < 4) D.arrow(s, x + ew + 0.01, 4.06, x + ew + 0.11, 4.06, { width: 1 }); });
    D.text(s, 0.5, 4.3, 6.1, 0.6, 'TuneContext(mod, target=npu, ScheduleFn(make_schedule, sch_rules=[], postprocs=[]), ReplayFunc(), rand_state=seed) · pre_tuning(max_trials=schedule_max_attempts) · 후보마다 notify_runner_results(RunnerResult(None, None)): latency를 넣지 않고 NPU 평가기가 승자를 고름', { size: 7.5, color: C.muted });
    // right: DB record anatomy
    D.text(s, 6.85, 1.5, 5.98, 0.24, 'JSONDatabase 기록의 실제 구성 (run_secs=None, 실행 조건이 같은 커널만 적중)', { size: 9.5, bold: true, color: C.navy });
    D.code(s, 6.85, 1.78, 5.98, 2.02, [
      '# <schedule_db>/<fingerprint: source sha · TVM · contract · native>/',
      'Workload.mod = PrimFunc − {global_symbol, op_pattern}',
      '  .attr["npu.selected_schedule_contract"] = {',
      '     profile − {search, seed, trials, attempts, words,',
      '                joint_memory, beam, graph_search_seconds},',
      '     scratch_base, slot_count, native_probe }',
      'TuningRecord(trace, workload, run_secs=None, target=npu)',
      'trace = split · reorder · cache_read(global.sram) ·',
      '        compute_at · tensorize(npu_gemm_acc_sram) …',
    ], { size: 7.8 });
    const nb = [['lookup', 'structural hash + equal\n(shape·dtype·scope·SRAM·버전)', 'host'], ['trace 재적용', 'apply_to_schedule\n(새 tir.Schedule)', 'ir'], ['공통 재검증', 'lowering → SRAM 배치\n→ 용량 → probe codegen', 'sram'], ['reuse / tune', '채택(탐색 0회) /\nincumbent → 개선 시만 commit', 'good']];
    const nw = (5.98 - 3 * 0.12) / 4;
    nb.forEach((b, i) => { const x = 6.85 + i * (nw + 0.12); D.box(s, x, 3.9, nw, 0.85, b[0], { kind: b[2], size: 8.5, detail: b[1], dsize: 7 }); if (i < 3) D.arrow(s, x + nw + 0.01, 4.32, x + nw + 0.11, 4.32, { width: 1 }); });
    // bottom: small-example evidence
    D.table(s, 0.5, 4.95, 7.4, [
      ['작은 예제 (FP16, 각 1회 측정)', '기존 build', '첫 저장 build', 'DB 재사용 build', '새 후보 TIR 생성'],
      ['64×64 matmul 3회, 커널 사이 SRAM 보관', '0.419 s', '0.499 s', '0.307 s', '8 → 0'],
      ['M=64 K=128 N=64, SiLU+multiply fusion', '0.343 s', '0.383 s', '0.264 s', '3 → 0'],
      ['M=7 K=128 N=64, padding', '0.369 s', '0.422 s', '0.164 s', '7 → 0'],
    ], { colW: [3.0, 0.95, 1.05, 1.15, 1.25], size: 8.2, rowH: 0.3, aligns: ['left', 'right', 'right', 'right', 'right'] });
    D.callout(s, 8.1, 4.95, 4.73, 1.2, [
      { text: '세 경로의 명령이 같고 C-model 출력은 bit 단위로 같았습니다. ', bold: true }, { text: 'Llama 4개 커널에서는 space와 ReplayFunc의 최종 binary가 같았고(E8), 512-token down은 개선 후 두 방식 모두 4회 검사·합법 4개였습니다(E9). DB는 최외곽 compilation context에서만 설정하며 E10은 DB 없이 실행합니다.', size: 8.3 },
    ], { kind: 'faint' });
    D.text(s, 0.5, 6.18, 7.4, 0.22, '시간에는 context/DB 준비가 포함되며 일반적인 배율이 아닙니다. 재사용 시 "탐색"이 0회이지 trace 재생·lowering·검증이 없어지는 것은 아닙니다.', { size: 7.5, color: C.muted });
    D.takeaway(s, 'ReplayFunc와 DB 모두 NPU 공통 평가기(실제 SRAM 배치·명령 생성)를 거칩니다. TVM의 latency 순위로 고르지 않고, DMA를 run_secs로 넣지도 않습니다.', { y: 6.5 });
  });

};
