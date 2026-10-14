'use strict';
const { C, F, evidence } = require('./lib');

// 13-slide condensed deck: one mechanism figure per slide, IR only as 2-4 line summaries.
module.exports = (D) => {
  // ================================================================ 1 cover
  D.add({ dark: true, bare: true, notes: '13장 요약판입니다. 각 장은 그림 하나로 그 단계의 실제 동작을 보여주고, IR은 2~4줄 요약만 둡니다. 예제는 SiLU(X[64,128]·Wᵀ+bias), 두 matmul 44 KiB, Llama 3.2 3B(S=7, C=9)이며 과거 측정과 현재 탐색 개선을 구분하고 E10은 메모리 증가로 중단됐으며 전체 재검증은 아직 수행하지 않았습니다.' },
  (s, D) => {
    D.text(s, 0.8, 0.7, 11, 0.3, 'NPU COMPILER  ·  V09 BACKEND  ·  TVM 0.19 RELAX / TIR  ·  요약판', { size: 11, color: C.sram, bold: true, charSpacing: 2 });
    D.text(s, 0.8, 1.35, 11.5, 1.6, 'TVM 기반 LLM NPU Compiler', { size: 44, color: C.white, bold: true });
    D.text(s, 0.8, 2.45, 11.5, 0.9, '모델의 계산이 NPU 명령과 다음 token이 되기까지 — 전체 pipeline의 동작을 그림으로', { size: 20, color: C.darkText });
    const chips = [['계산의 의미', 'Relax · TIR · FP16 저장/FP32 누산', 'ir'], ['배치', 'fusion · 64×64 tile과 panel · SRAM/DRAM 주소', 'sram'], ['실행', 'ISA stream · linker · prefill/decode runtime', 'unit']];
    chips.forEach((c, i) => {
      const x = 0.8 + i * 3.9;
      D.rect(s, x, 3.9, 3.6, 1.15, { fill: '18293F', line: '2B3E58', radius: 0.08 });
      D.text(s, x + 0.25, 4.02, 3.1, 0.4, c[0], { size: 18, bold: true, color: C.white });
      D.text(s, x + 0.25, 4.45, 3.1, 0.5, c[1], { size: 12, color: C.darkMuted });
    });
    D.text(s, 0.8, 5.55, 11.5, 0.9, '예제: SiLU(X[64,128]·Wᵀ+bias) · 두 matmul X·W1·W2 (SRAM 44 KiB) · Llama 3.2 3B (28 layers, prompt 7 tokens, capacity 9)\n설정: FP16 저장 / FP32 누산 · SRAM 8 MiB · tile 64×64 · 현재 탐색 개선 + 256-token E10 진행 상태 포함', { size: 12, color: C.darkMuted });
    D.text(s, 0.8, 6.75, 11.5, 0.3, '2026-09-29 탐색 개선 반영 · 근거: d_compiler/docs (13장, examples, evidence)', { size: 10, color: C.darkMuted });
  });

  // ================================================================ 2 whole pipeline in one picture
  D.add({ section: '전체 pipeline', title: '한 장으로 보는 전체 pipeline: 같은 계산이 단계마다 다른 모습이 된다', subtitle: '작은 예제 SiLU(X[64,128]·Wᵀ+bias)의 실제 산출물로 본 5단계. 후보 graph마다 ③④⑤까지 내려가 (DMA bytes, words, SRAM peak)를 얻고 그 비용으로 선택합니다.', source: 'docs/examples/pipeline (01~09), bundle/program.json; npu_target.build → npu_build.lower_and_select → npu_link.compile_program',
    notes: '① 모델을 relax.frontend.nn으로 export하면 연산 호출과 shape만 있는 Relax 그래프가 됩니다. ② graph prefix가 weight만 쓰는 transpose를 host 함수로 옮기고, LegalizeOps가 각 연산의 원소 계산(FP32 누산, FP16 cast)을 TIR PrimFunc로 정의합니다. ③ FuseOps 영역을 실제 명령 비용으로 검사하고, matmul은 panel·순서·cache 위치를 탐색해 64×64 intrinsic으로 tensorize합니다. ④ DRAM은 TVM StaticPlanBlockMemory의 pool을 flat 주소로, SRAM은 USMP가 충돌하지 않는 buffer의 offset을 공유하고 커널 사이 보관값을 위쪽에 예약합니다. ⑤ walker가 TIR을 완전히 unroll해 descriptor·LOAD/SAVE·M_MUL·DMA packet의 직선 stream을 만들고, runtime은 DRAM image와 함께 C-model에 넘겨 출력 offset에서 결과를 읽습니다. 후보 비교는 ⑤의 통계로 ③으로 되돌아갑니다.' },
  (s, D) => {
    const cw = 2.3, gap = 0.2075, cx = i => 0.5 + i * (cw + gap);
    const heads = [['① 모델 → Relax', 'ir'], ['② Graph 정리 · LegalizeOps', 'ir'], ['③ Fusion · Schedule', 'sram'], ['④ 메모리 계획', 'dram'], ['⑤ ISA · 실행', 'unit']];
    heads.forEach((h, i) => { D.box(s, cx(i), 1.55, cw, 0.42, h[0], { kind: h[1], size: 10.5 }); if (i < 4) D.arrow(s, cx(i) + cw + 0.01, 1.76, cx(i) + cw + gap - 0.01, 1.76, { width: 1.5 }); });
    // ---- ① tiles (vertical)
    let x = cx(0);
    D.matrix(s, x + 0.55, 2.3, 1.2, 0.5, 1, 2, { kind: 'host', tile: 1, title: 'X [64,128] = 1×2 tiles', titleW: 1.8, regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'sram', label: 'X₀', size: 7 }, { r0: 0, r1: 1, c0: 1, c1: 2, kind: 'ir', label: 'X₁', size: 7 }] });
    D.arrow(s, x + 1.15, 2.82, x + 1.15, 3.12, { width: 1 }); D.text(s, x + 1.22, 2.82, 1.1, 0.3, '@ Wᵀ [128,64]', { size: 7, color: C.muted });
    D.matrix(s, x + 0.88, 3.15, 0.55, 0.55, 1, 1, { kind: 'unit', regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'unit', label: 'P', size: 8 }] });
    D.text(s, x + 1.48, 3.2, 0.85, 0.5, 'P [64,64]\nK = 128', { size: 7, color: C.muted });
    D.arrow(s, x + 1.15, 3.72, x + 1.15, 4.0, { width: 1 }); D.text(s, x + 1.22, 3.72, 1.1, 0.3, '+ bias, SiLU', { size: 7, color: C.muted });
    D.matrix(s, x + 0.88, 4.03, 0.55, 0.55, 1, 1, { kind: 'good', regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'good', label: 'Y', size: 8 }] });
    D.text(s, x + 1.48, 4.08, 0.85, 0.5, 'Y [64,64]\n출력', { size: 7, color: C.muted });
    D.code(s, x, 4.75, cw, 0.78, ['lv  = permute_dims(W)   # Wᵀ', 'lv1 = matmul(x, lv)', 'lv2 = add(lv1, bias)', 'gv  = silu(lv2)  # num_input=1'], { size: 6.5 });
    // ---- ② host split + call_tir chain
    x = cx(1);
    D.matrix(s, x + 0.05, 2.6, 0.8, 0.4, 1, 2, { kind: 'dram', tile: 1, title: 'W [64,128]', titleW: 1.0, titleSize: 7 });
    D.arrow(s, x + 0.9, 2.8, x + 1.35, 2.8, { width: 1.2, color: C.host }); D.text(s, x + 0.82, 2.97, 0.7, 0.3, 'host 1회', { size: 6.5, color: C.host, bold: true, align: 'center' });
    D.matrix(s, x + 1.4, 2.4, 0.4, 0.8, 2, 1, { kind: 'dram', tile: 1, title: 'Wᵀ → DRAM', titleW: 1.0, titleSize: 7 });
    D.text(s, x, 3.25, cw, 0.3, 'LiftTransformParams: weight만 쓰는 계산을 분리 (LM head 751 MiB도 1회)', { size: 6.5, color: C.muted });
    const nodes = [['matmul', 'unit', '4'], ['add', 'ir', '0'], ['silu', 'ir', '0']];
    nodes.forEach((n, i) => { const nx = x + 0.05 + i * 0.77; D.box(s, nx, 3.68, 0.68, 0.42, n[0], { kind: n[1], size: 8, detail: 'pattern ' + n[2], dsize: 6 }); if (i < 2) D.arrow(s, nx + 0.69, 3.89, nx + 0.76, 3.89, { width: 1 }); });
    D.text(s, x, 4.14, cw, 0.3, 'call_tir ×3 + PrimFunc (Relax 호출 순서는 남음)', { size: 6.5, color: C.muted });
    D.code(s, x, 4.5, cw, 1.03, ['# matmul PrimFunc 요약', 'C32[i,j] = 0', 'C32[i,j] += f32(x[i,k])·f32(Wᵀ[k,j])', 'Y[i,j] = f16(C32[i,j])   # 한 번만', 'softmax: FP32 sum, RMSNorm: absmax'], { size: 6.5 });
    // ---- ③ fusion region + schedule (SRAM panels)
    x = cx(2);
    D.rect(s, x, 2.2, cw, 0.6, { fill: C.sramFill, line: C.sram, radius: 0.08, lineWidth: 1.25 });
    ['matmul', 'add', 'silu'].forEach((n, i) => { const nx = x + 0.1 + i * 0.75; D.box(s, nx, 2.3, 0.62, 0.36, n, { kind: i ? 'ir' : 'unit', size: 7.5 }); if (i < 2) D.arrow(s, nx + 0.63, 2.48, nx + 0.73, 2.48, { width: 0.75 }); });
    D.text(s, x, 2.85, cw, 0.3, 'FuseOps 영역: unfused 2,812 vs fused 1,577 words → 채택', { size: 6.5, color: C.muted });
    D.rect(s, x, 3.2, cw, 1.3, { kind: 'sram', radius: 0.06, fill: 'EEF8F6' });
    D.text(s, x + 0.08, 3.22, 2.1, 0.2, 'SRAM (schedule 결과)', { size: 7, bold: true, color: C.sram });
    D.matrix(s, x + 0.1, 3.5, 0.9, 0.35, 1, 2, { kind: 'sram', tile: 1, regions: [{ r0: 0, r1: 1, c0: 0, c1: 2, kind: 'sram', label: 'X panel 16 KiB', size: 6 }] });
    D.matrix(s, x + 0.1, 3.95, 0.35, 0.5, 2, 1, { kind: 'dram', tile: 1 });
    D.text(s, x + 0.5, 3.95, 0.7, 0.5, 'Wᵀ panel\n16 KiB', { size: 6, color: C.muted });
    D.matrix(s, x + 1.5, 3.8, 0.45, 0.45, 1, 1, { kind: 'unit', regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'unit', label: 'P', size: 7 }] });
    D.text(s, x + 1.2, 4.28, 1.1, 0.25, 'MAC=0 → MAC=1', { size: 6, color: C.unit, bold: true });
    D.arrow(s, x + 1.02, 3.68, x + 1.5, 3.95, { width: 0.75, color: C.unit }); D.arrow(s, x + 0.47, 4.2, x + 1.5, 4.1, { width: 0.75, color: C.unit });
    D.text(s, x, 4.6, cw, 0.95, 'matmul: m/n panel, K chunk, loop 순서, X/W cache 위치를 탐색 → 64×64 intrinsic으로 tensorize. 후보마다 실제 명령을 내어 비용 측정 (중복 정규화 · 증명된 용량 초과 제외 · 시도 64 · legal 8 · 적응형 words 예산)', { size: 6.5, color: C.muted });
    // ---- ④ memory bars
    x = cx(3);
    D.text(s, x, 2.15, cw, 0.2, 'DRAM image 41,100 B (실제 주소)', { size: 7, bold: true, color: C.dram });
    D.membar(s, x, 2.37, cw, 0.36, 0, 41100, [{ from: 0, to: 16384, label: 'x', kind: 'host' }, { from: 16384, to: 16512, label: '', kind: 'dram' }, { from: 16512, to: 32896, label: 'Wᵀ', kind: 'dram' }, { from: 32896, to: 41088, label: 'gv', kind: 'unit' }, { from: 41088, to: 41100, label: '', kind: 'faint' }], { size: 7 });
    D.text(s, x, 2.76, cw, 0.2, '0        16,384      32,896   41,088', { size: 5.5, color: C.muted, font: F.code });
    D.text(s, x, 3.05, cw, 0.2, 'SRAM: USMP offset 공유 (nibble)', { size: 7, bold: true, color: C.sram });
    D.membar(s, x, 3.27, cw, 0.36, 0, 81920, [{ from: 0, to: 16384, label: 'acc', kind: 'unit' }, { from: 16384, to: 49152, label: 'A / Wᵀ panel', kind: 'sram' }, { from: 49152, to: 81920, label: 'σ · bias / X panel', kind: 'sram' }], { size: 6 });
    D.text(s, x, 3.66, cw, 0.3, 'peak 150,808 → 85,016 nib (bump vs greedy)', { size: 6.5, color: C.muted });
    D.text(s, x, 4.02, cw, 0.2, '커널 사이 보관 (두 matmul, 44 KiB)', { size: 7, bold: true, color: C.warn });
    D.membar(s, x, 4.24, cw, 0.36, 0, 45056, [{ from: 0, to: 1548, label: '', kind: 'host' }, { from: 1548, to: 9740, label: 'X→Y', kind: 'sram' }, { from: 9740, to: 26124, label: 'W1→W2', kind: 'dram' }, { from: 26124, to: 42508, label: 'H 보관', kind: 'warn' }], { size: 6.5 });
    D.text(s, x, 4.63, cw, 0.9, 'H를 DRAM에 쓰지 않고 다음 matmul이 직접 읽음 → DMA 81,932 → 49,164 B, 미사용 DRAM pool 제거 (image 65,548 → 49,164 B)', { size: 6.5, color: C.muted });
    // ---- ⑤ ISA + runtime
    x = cx(4);
    D.text(s, x, 2.15, cw, 0.2, '명령 stream 1,577 words (정적 전개)', { size: 7, bold: true, color: C.unit });
    D.wordStrip(s, x, 2.37, cw, 0.36, [{ n: 12, kind: 'dram', label: 'GLOAD', size: 6 }, { n: 20, kind: 'ir', label: 'descriptor', size: 6 }, { n: 2, kind: 'sram' }, { n: 1, kind: 'unit', label: 'M', size: 6 }, { n: 8, kind: 'ir' }, { n: 1, kind: 'unit', label: 'M', size: 6 }, { n: 60, kind: 'sram', label: 'vector: bias·SiLU (LOAD/op/SAVE ×…)', size: 6 }, { n: 4, kind: 'dram', label: 'GST', size: 6 }, { n: 1, kind: 'red' }], { size: 6 });
    D.text(s, x, 2.76, cw, 0.3, 'loop·branch 없음 · DMA packet 4 words', { size: 6.5, color: C.muted });
    const f1 = D.box(s, x, 3.12, cw, 0.42, 'program_memory.bin + global_memory.bin', { kind: 'host', size: 7 });
    D.arrow(s, x + cw / 2, f1.b, x + cw / 2, f1.b + 0.22, { width: 1 });
    const f2 = D.box(s, x, 3.78, cw, 0.42, 'C-model 실행 (mysim_v09)', { kind: 'unit', size: 7.5 });
    D.arrow(s, x + cw / 2, f2.b, x + cw / 2, f2.b + 0.22, { width: 1 });
    D.box(s, x, 4.44, cw, 0.42, 'saved image → 출력 offset 32,896 [64,64]', { kind: 'good', size: 7 });
    D.text(s, x, 4.9, cw, 0.65, 'DMA 41,100 B · SRAM peak 42,508 B · LLVM reference(K=64 grouping)와 최대 절대차 0 · fused/unfused bit-exact', { size: 6.5, color: C.muted });
    // feedback arrow
    D.line(s, cx(4) + cw / 2, 5.6, cx(4) + cw / 2, 5.85, { arrow: false, color: C.warn, dash: 'dash', width: 1.5 });
    D.line(s, cx(4) + cw / 2, 5.85, cx(2) + cw / 2, 5.85, { arrow: false, color: C.warn, dash: 'dash', width: 1.5 });
    D.line(s, cx(2) + cw / 2, 5.85, cx(2) + cw / 2, 5.6, { color: C.warn, dash: 'dash', width: 1.5 });
    D.text(s, cx(2) + cw / 2 + 0.2, 5.9, 5.0, 0.3, '비용 (DMA bytes, words, SRAM peak)으로 다음 후보 선택 — 기본 추가 탐색 120 s / None 해제 가능 · 요청 entry가 비악화일 때 채택', { size: 8, color: C.warn, bold: true });
    D.takeaway(s, '"모델을 compile한다" = graph 선택 + schedule + 주소 배치 + 명령 생성. 실행은 그 뒤 별도 단계이며, 입력만 바꿔 같은 명령을 재사용합니다.', { y: 6.5 });
  });

  // ================================================================ 3 device & data movement
  D.add({ section: '① 장치', title: '장치: DRAM ↔ SRAM ↔ 연산장치, 데이터는 panel과 64×64 tile로 움직인다', subtitle: 'weight는 DRAM에서 panel 단위로 SRAM에 올라오고(GLOAD), 64×64 tile이 연산장치로 들어가(LOAD) FP32 accumulator 한 개에서 K를 누산한 뒤 FP16으로 SRAM에 내려오며(SAVE), 결과가 DRAM으로 나갑니다(GSTORE).', source: 'docs/current/01_overview.md, reference/execution_contract.md, npu_compiler/device.py',
    notes: 'DRAM(global)은 DMA만 접근하며 32-bit cell 단위 주소를 씁니다. SRAM은 8 MiB, nibble 주소입니다. 연산장치는 64×64 matrix unit(FP32 accumulator 1개)과 256-lane vector unit입니다. GLOAD/GSTORE는 DRAM↔SRAM, LOAD/SAVE는 SRAM↔연산장치 이동이라 서로 다른 경계입니다. SRAM 구성(literal·scratch, 입력 panel, 출력 tile·임시, 커널 사이 보관값)은 메모리 계획의 실제 배치 순서입니다. 한 숫자를 다른 단위로 읽으면 2배/4배 어긋난 주소가 됩니다. C-model 통과는 기능 검증이며 RTL 검증을 대신하지 않습니다.' },
  (s, D) => {
    const x0 = 0.5, fw = 7.4;
    D.text(s, x0, 1.55, fw, 0.24, 'DRAM / global image  (주소 단위: 32-bit cell = 4 B, DMA만 접근)', { size: 9.5, bold: true, color: C.dram });
    D.membar(s, x0, 1.8, fw, 0.5, 0, 100, [
      { from: 0, to: 52, label: 'weight (Wᵀ, 층마다 192 MiB)', sub: 'panel 단위로 읽음', kind: 'dram' },
      { from: 52, to: 66, label: '입력', sub: 'embeds·cos/sin·mask', kind: 'host' },
      { from: 66, to: 86, label: 'persistent KV cache', sub: '[28,8,128,C]', kind: 'warn' },
      { from: 86, to: 100, label: '출력', sub: 'logits·K/V_new', kind: 'unit' },
    ], { size: 8.5 });
    D.text(s, x0, 3.05, fw, 0.24, 'SRAM 8 MiB  (주소 단위: nibble, fp16 = 4 nibbles)', { size: 9.5, bold: true, color: C.sram });
    D.membar(s, x0, 3.3, fw, 0.5, 0, 100, [
      { from: 0, to: 7, label: 'pool', sub: 'literal·scratch', kind: 'host' },
      { from: 7, to: 30, label: 'X panel', sub: '입력 activation', kind: 'sram' },
      { from: 30, to: 53, label: 'W panel j', sub: '이번에 쓰는 weight 조각', kind: 'dram' },
      { from: 53, to: 72, label: '출력 tile · 임시', sub: 'USMP가 offset 배치', kind: 'ir' },
      { from: 72, to: 100, label: '커널 사이 보관값', sub: 'top-down 예약', kind: 'warn' },
    ], { size: 8.5 });
    D.matrix(s, x0 + 0.15, 4.65, 1.15, 1.15, 8, 8, { kind: 'unit', tile: 8, title: 'Matrix unit 64×64', below: 'FP32 accumulator ×1 · M_MUL MAC bit로 K 누산', belowW: 3.2 });
    D.lanes(s, x0 + 2.4, 4.95, 3.6, 0.38, Array.from({ length: 16 }, (_, i) => ({ label: i === 15 ? '255' : (i === 0 ? '0' : (i === 8 ? '…' : '')), kind: 'unit' })), {});
    D.text(s, x0 + 2.4, 4.7, 3.6, 0.22, 'Vector unit 256 lanes (물리)', { size: 8.5, bold: true, color: C.navy });
    D.text(s, x0 + 2.4, 5.36, 4.8, 0.42, '논리 VLEN 1..65535 → 내부에서 256씩 반복한다는 전제 (RTL 미확인). FP16 입력, reduction 내부 FP32.', { size: 7.5, color: C.muted });
    D.arrow(s, x0 + fw * 0.26, 2.3, x0 + fw * 0.41, 3.3, { color: C.dram, width: 2 });
    D.text(s, x0 + fw * 0.42, 2.5, 2.6, 0.4, 'GLOAD: DRAM → SRAM\n4-word DMA packet, cell 단위', { size: 8, color: C.dram, bold: true });
    D.arrow(s, x0 + fw * 0.62, 3.3, x0 + fw * 0.93, 2.3, { color: C.dram, width: 2 });
    D.text(s, x0 + fw * 0.78, 2.8, 1.8, 0.3, 'GSTORE: SRAM → DRAM', { size: 8, color: C.dram, bold: true });
    D.arrow(s, x0 + fw * 0.18, 3.8, x0 + 0.75, 4.65, { color: C.sram, width: 2 });
    D.text(s, x0 + 1.4, 4.0, 2.4, 0.4, 'LOAD: SRAM 64×64 tile → unit\n(descriptor로 주소·stride 지정)', { size: 8, color: C.sram, bold: true });
    D.arrow(s, x0 + 1.3, 4.95, x0 + fw * 0.62, 3.8, { color: C.sram, width: 2 });
    D.text(s, x0 + 4.4, 4.05, 2.4, 0.4, 'SAVE: FP16 결과 → SRAM 출력 tile', { size: 8, color: C.sram, bold: true });
    D.table(s, 8.2, 1.55, 4.63, [
      ['이동 명령', '경계', 'DRAM을 지나는가'],
      ['GLOAD 0xA0', 'DRAM → SRAM', '예 (4 words)'],
      ['LOAD 0x90', 'SRAM → unit', '아니오'],
      ['SAVE 0x98', 'unit → SRAM', '아니오'],
      ['GSTORE 0xA8', 'SRAM → DRAM', '예 (4 words)'],
    ], { colW: [1.3, 1.6, 1.73], size: 9, rowH: 0.3 });
    D.table(s, 8.2, 3.25, 4.63, [
      ['주소 단위', '값'],
      ['plan.address (DRAM)', 'unit 2 B'],
      ['DMA global 주소·stride', '32-bit cell (×4 B)'],
      ['SRAM descriptor 주소', 'nibble (÷2 B), 8-nibble 정렬'],
      ['shape / VLEN', 'element'],
    ], { colW: [2.1, 2.53], size: 9, rowH: 0.3 });
    D.callout(s, 8.2, 4.95, 4.63, 1.35, [
      { text: '수치 계약: ', bold: true }, { text: 'FP16 저장, K=64 단위 FP32 부분합을 accumulator 1개에 누산, partial spill 금지. 자동 matmul 경로의 activation bit는 OFF. 성능 지표는 DMA bytes → ISA words → SRAM peak (정적, cycle 아님).', size: 9.5 },
    ], { kind: 'faint' });
    D.takeaway(s, '연산장치 안의 FP32 accumulator는 SRAM buffer가 아닙니다. 어떤 projection weight도 8 MiB에 통째로 들어가지 않으므로 panel streaming이 compiler의 핵심 결정입니다.', { y: 6.5 });
  });

  // ================================================================ 4 model -> Relax -> host split
  D.add({ section: '② 계산의 의미', title: '모델을 Relax로 쓰고, weight만 쓰는 계산은 host로 옮긴다', subtitle: 'Llama 한 layer를 tensor 모양으로 본 그래프(왼쪽). 오른쪽: LiftTransformParams가 W transpose를 host 함수로 분리해 Wᵀ가 DRAM image의 정해진 주소에 한 번 놓이고, device 함수는 token마다 입력만 바꿉니다.', source: 'nn_models/llama.py, docs/examples/llama_ir/01_authored_relax.py, pipeline/02_graph_prefix.py, bundle/program.json',
    notes: 'relax.frontend.nn으로 정의한 Llama는 export_tvm으로 정적 shape의 Relax IRModule이 됩니다(prefill S=7, decode C=9). Attention은 Q/K/V projection, RoPE(Q,K), GQA repeat(8→24 head), QKᵀ·scale·mask·softmax·PV, O projection과 residual이고 MLP는 gate/up([7,8192], weight 48 MiB씩)·SiLU·곱·down과 residual입니다. graph prefix(CanonicalizeAttention, CanonicalizeBindings, EliminateCommonSubexpr, RewriteDataflowReshape, LiftTransformParams)가 parameter만 쓰는 permute_dims를 host 함수로 옮기고, runtime 준비 단계에서 LLVM VM이 이를 1회 실행해 DRAM image에 넣습니다. 반환 순서가 (bias, lv)로 바뀌므로 runtime은 signature로 binding합니다.' },
  (s, D) => {
    const stack = (x, y, w, h, rows, cols, kind, n, o) => { for (let i = n - 1; i >= 1; i--) D.matrix(s, x + i * 0.05, y - i * 0.05, w, h, rows, cols, { kind, cells: false, lineWidth: 0.75 }); return D.matrix(s, x, y, w, h, rows, cols, Object.assign({ kind }, o || {})); };
    const lbl = (x, y, w, t, o) => D.text(s, x, y, w, 0.3, t, Object.assign({ size: 7, color: C.muted, align: 'center' }, o || {}));
    D.text(s, 0.5, 1.5, 7.6, 0.24, 'Llama 한 layer (prefill S=7): glyph = tensor, 겹침 = head 축, 회색 삼각형 = causal mask', { size: 9, bold: true, color: C.navy });
    const y1 = 2.35;
    const X = D.matrix(s, 0.5, y1, 0.9, 0.42, 1, 6, { kind: 'host', title: 'X [7,3072]', titleW: 1.0, titleSize: 7.5 });
    D.arrow(s, X.r + 0.02, X.cy, 1.75, X.cy, { width: 1 }); lbl(X.r - 0.15, X.cy - 0.32, 0.7, 'RMSNorm');
    const N1 = D.matrix(s, 1.78, y1, 0.9, 0.42, 1, 6, { kind: 'ir', title: 'norm₁', titleW: 0.9, titleSize: 7.5 });
    const Q = stack(3.1, 1.95, 0.65, 0.34, 1, 3, 'unit', 3, { title: 'Q [7,24,128]', titleW: 1.1, titleSize: 7 });
    const K = stack(3.1, 2.5, 0.65, 0.34, 1, 3, 'unit', 2, { title: 'K [7,8,128]', titleW: 1.0, titleSize: 7 });
    const V = stack(3.1, 3.05, 0.65, 0.34, 1, 3, 'unit', 2, { title: 'V [7,8,128]', titleW: 1.0, titleSize: 7 });
    [Q, K, V].forEach(b => D.arrow(s, N1.r + 0.02, N1.cy, b.x, b.cy, { width: 0.75 }));
    lbl(2.55, 1.95, 0.6, 'q/k/v\nproj', { size: 6.5, color: C.unit, bold: true });
    D.chip(s, 3.9, 2.0, 'RoPE', { kind: 'ir', w: 0.45, size: 6.5 }); D.chip(s, 3.9, 2.55, 'RoPE·rep×3', { kind: 'ir', w: 0.8, size: 6.5 }); D.chip(s, 3.9, 3.1, 'rep×3', { kind: 'ir', w: 0.55, size: 6.5 });
    const masked = []; for (let r = 0; r < 7; r++) for (let c = r + 1; c < 7; c++) masked.push({ r0: r, r1: r + 1, c0: c, c1: c + 1, kind: 'faint', lineWidth: 0.3 });
    const Sc = stack(4.9, 1.95, 0.62, 0.62, 7, 7, 'unit', 3, { title: 'QKᵀ [24,7,7]', titleW: 1.1, titleSize: 7, regions: masked });
    D.arrow(s, 4.4, 2.15, Sc.x, Sc.y + 0.15, { width: 0.75 }); D.arrow(s, 4.75, 2.7, Sc.x, Sc.cy + 0.1, { width: 0.75 });
    const P = stack(5.95, 1.95, 0.62, 0.62, 7, 7, 'unit', 3, { title: 'softmax → P', titleW: 1.1, titleSize: 7, regions: masked });
    D.arrow(s, Sc.r + 0.12, Sc.cy, P.x, P.cy, { width: 0.75 }); lbl(5.45, 2.65, 0.6, '/√128\n+mask', { size: 6 });
    const AO = stack(6.95, 2.1, 0.65, 0.34, 1, 3, 'unit', 3, { title: 'P·V', titleW: 0.7, titleSize: 7 });
    D.arrow(s, P.r + 0.12, P.cy, AO.x, AO.cy, { width: 0.75 }); D.arrow(s, 4.5, 3.22, AO.x, AO.cy + 0.08, { width: 0.75, color: C.muted });
    const AT = D.box(s, 7.15, 2.85, 0.9, 0.36, 'o_proj', { kind: 'unit', size: 7.5 });
    D.arrow(s, AO.cx, AO.b + 0.02, AT.cx, AT.y, { width: 0.75 });
    const R1 = D.box(s, 7.45, 3.45, 0.3, 0.36, '+', { kind: 'warn', size: 10 });
    D.arrow(s, AT.cx, AT.b, R1.cx, R1.y, { width: 0.75 });
    D.line(s, X.cx, X.b, X.cx, 3.63, { arrow: false, color: C.warn, dash: 'dash' }); D.arrow(s, X.cx, 3.63, R1.x, 3.63, { color: C.warn, dash: 'dash' });
    D.text(s, 0.55, 3.66, 2.5, 0.2, 'residual: X 유지 → R = X + attn', { size: 6.5, color: C.warn, bold: true });
    const y2 = 4.45;
    const R = D.matrix(s, 0.5, y2, 0.9, 0.42, 1, 6, { kind: 'warn', title: 'R [7,3072]', titleW: 1.0, titleSize: 7.5 });
    D.line(s, R1.cx, R1.b, R1.cx, 4.05, { arrow: false, color: C.muted }); D.line(s, R1.cx, 4.05, R.cx, 4.05, { arrow: false, color: C.muted }); D.arrow(s, R.cx, 4.05, R.cx, R.y - 0.24, { color: C.muted });
    D.arrow(s, R.r + 0.02, R.cy, 1.75, R.cy, { width: 1 }); lbl(R.r - 0.15, R.cy - 0.32, 0.7, 'RMSNorm');
    const N2 = D.matrix(s, 1.78, y2, 0.9, 0.42, 1, 6, { kind: 'ir', title: 'norm₂', titleW: 0.9, titleSize: 7.5 });
    const G = D.matrix(s, 3.1, 4.1, 1.9, 0.34, 1, 12, { kind: 'unit', title: 'G = gate [7,8192] (W 48 MiB)', titleW: 2.2, titleSize: 7 });
    const U = D.matrix(s, 3.1, 4.85, 1.9, 0.34, 1, 12, { kind: 'unit', title: 'U = up [7,8192] (W 48 MiB)', titleW: 2.2, titleSize: 7 });
    D.arrow(s, N2.r + 0.02, N2.cy, G.x, G.cy, { width: 0.75 }); D.arrow(s, N2.r + 0.02, N2.cy, U.x, U.cy, { width: 0.75 });
    const Hm = D.matrix(s, 5.4, y2, 1.9, 0.34, 1, 12, { kind: 'ir', title: 'H = SiLU(G) ⊙ U', titleW: 1.8, titleSize: 7 });
    D.arrow(s, G.r + 0.02, G.cy, Hm.x, Hm.cy - 0.05, { width: 0.75 }); D.arrow(s, U.r + 0.02, U.cy, Hm.x, Hm.cy + 0.05, { width: 0.75 });
    const DN = D.box(s, 7.15, y2 - 0.03, 0.9, 0.4, 'down', { kind: 'unit', size: 7.5, detail: 'W 48 MiB', dsize: 6 });
    D.arrow(s, Hm.r + 0.02, Hm.cy, DN.x, DN.cy, { width: 0.75 });
    const R2 = D.box(s, 7.45, 5.15, 0.3, 0.36, '+', { kind: 'warn', size: 10 });
    D.arrow(s, DN.cx, DN.b, R2.cx, R2.y, { width: 0.75 });
    D.line(s, R.cx, R.b, R.cx, 5.72, { arrow: false, color: C.warn, dash: 'dash' }); D.line(s, R.cx, 5.72, R2.cx, 5.72, { arrow: false, color: C.warn, dash: 'dash' }); D.line(s, R2.cx, 5.72, R2.cx, R2.b, { color: C.warn, dash: 'dash' });
    D.text(s, 0.55, 5.75, 4, 0.2, 'residual: R 유지 → Y = R + down(H) → 다음 layer (×28)', { size: 6.5, color: C.warn, bold: true });
    D.text(s, 0.5, 6.0, 7.6, 0.4, 'export 결과: prefill(input_embeds [7,3072], cos/sin, mask [1,7,7], packed_params) → logits; decode(…, k_cache [28,8,128,9], v_cache, mask [1,1,10]) → (logits, K_new, V_new). num_input이 실행 입력과 parameter의 경계.', { size: 7.5, color: C.muted });
    // right: LiftTransformParams
    const rx = 8.5;
    D.text(s, rx, 1.5, 4.33, 0.24, 'LiftTransformParams (작은 예제의 실제 IR 요약)', { size: 9, bold: true, color: C.navy });
    D.code(s, rx, 1.78, 4.33, 1.15, ['# device: prefill(x, bias, lv)   # transpose 없음', 'lv1 = matmul(x, lv); lv2 = add(lv1, bias); gv = silu(lv2)', '# host (1회): prefill_transform_params(W, bias)', 'lv = permute_dims(W)  →  return (bias, lv)'], { size: 7 });
    D.matrix(s, rx + 0.1, 3.4, 1.2, 0.55, 1, 2, { kind: 'dram', tile: 1, title: 'W [64,128] (checkpoint)', titleW: 1.8, titleSize: 7 });
    D.arrow(s, rx + 1.4, 3.67, rx + 2.05, 3.67, { width: 1.5, color: C.host }); D.text(s, rx + 1.35, 3.85, 0.8, 0.3, 'host VM 1회', { size: 6.5, color: C.host, bold: true, align: 'center' });
    D.matrix(s, rx + 2.1, 3.1, 0.55, 1.15, 2, 1, { kind: 'dram', tile: 1, title: 'Wᵀ [128,64]', titleW: 1.0, titleSize: 7 });
    D.arrow(s, rx + 2.72, 3.67, rx + 3.2, 3.67, { width: 1.5, color: C.dram });
    D.box(s, rx + 3.25, 3.42, 1.05, 0.5, 'DRAM image', { kind: 'dram', size: 8 });
    D.text(s, rx, 4.5, 4.33, 0.22, 'DRAM image 41,100 B (bundle/program.json 실제 주소)', { size: 8, bold: true, color: C.navy });
    D.membar(s, rx, 4.75, 4.33, 0.42, 0, 41100, [{ from: 0, to: 16384, label: 'x (token마다)', kind: 'host' }, { from: 16384, to: 16512, label: '', kind: 'dram' }, { from: 16512, to: 32896, label: 'Wᵀ (1회)', kind: 'dram' }, { from: 32896, to: 41088, label: 'gv', kind: 'unit' }, { from: 41088, to: 41100, label: '', kind: 'faint' }], { ticks: [0, 16384, 32896, 41088], size: 7 });
    D.text(s, rx, 5.4, 4.33, 0.2, '↑ bias 128 B @ 16,384 · literal pool 12 B @ 41,088', { size: 6.5, color: C.muted });
    D.bullets(s, rx, 5.65, 4.33, 0.8, ['LM head Wᵀ 751.5 MiB도 준비 단계 1회 → token마다 이동 0', '반환 순서가 바뀌므로 runtime은 signature(x, bias, lv)로 binding'], { size: 8, psa: 2 });
    D.takeaway(s, 'Compile은 IR을 분리하는 시점, load는 host 함수를 실제 weight로 실행하는 시점, run은 입력만 바꿔 같은 프로그램을 재사용하는 시점입니다.', { y: 6.5 });
  });

  // ================================================================ 5 LegalizeOps & numeric contract
  D.add({ section: '② 계산의 의미', title: '수치 계약: FP16 저장, FP32 누산 — K 전체 동안 accumulator를 끊지 않는다', subtitle: 'LegalizeOps가 각 Relax 연산의 원소 계산을 TIR로 정의합니다(Relax 호출 순서는 남음). matmul은 128개 곱을 64개씩 두 FP32 부분합으로 같은 accumulator에 더하고 마지막에 한 번만 FP16으로 반올림합니다.', source: 'npu_compiler/npu_legalize.py, npu_numeric.py, docs/examples/pipeline/03_unfused_semantic_tir.py, 08_llvm_reference.py',
    notes: 'NPU legalize map은 matmul(FP16×FP16→FP32 sum→FP16), softmax(max·exp·FP32 sum·FP16 sum·divide, 마지막 축 1..65504), rms_norm(absmax로 먼저 scale해 x² overflow 방지), sum/mean, split(view)만 바꾸고 나머지는 TVM 기본 legalization입니다. 장치는 K=64 단위 부분합을 만들므로 CPU reference도 split+rfactor로 같은 grouping을 쓰며, 중간에 FP16으로 저장했다 다시 더하는 schedule은 "FP32 matrix chain interrupted"로 거절됩니다. E1에서 같은 grouping의 LLVM 출력과 최대 절대차 0. 정밀도 정책은 고정이며 오차 예산 자동 탐색은 없습니다.' },
  (s, D) => {
    const cells = Array.from({ length: 16 }, (_, i) => ({ label: i === 0 ? 'k=0' : (i === 7 ? '63' : (i === 8 ? '64' : (i === 15 ? '127' : ''))), kind: i < 8 ? 'sram' : 'ir', size: 6.5 }));
    D.lanes(s, 2.0, 1.65, 5.6, 0.32, cells, { title: '곱 x[i,k]·b[k,j] (FP16×FP16)', titleW: 1.9 });
    const g0 = D.box(s, 2.0, 2.2, 2.8, 0.5, 'g₀ = Σ_{k<64} (FP32 부분합)', { kind: 'sram', size: 9.5 });
    const g1 = D.box(s, 4.8, 2.2, 2.8, 0.5, 'g₁ = Σ_{64≤k<128} (FP32)', { kind: 'ir', size: 9.5 });
    D.arrow(s, 3.4, 1.97, 3.4, 2.2, { width: 1 }); D.arrow(s, 6.2, 1.97, 6.2, 2.2, { width: 1 });
    const a1 = D.box(s, 2.0, 2.95, 1.75, 0.55, 'acc = g₀', { kind: 'unit', size: 10, detail: 'M_MUL MAC=0', dsize: 7.5 });
    const a2 = D.box(s, 4.0, 2.95, 1.75, 0.55, 'acc += g₁', { kind: 'unit', size: 10, detail: 'M_MUL MAC=1', dsize: 7.5 });
    const a3 = D.box(s, 6.0, 2.95, 1.6, 0.55, 'FP16 반올림 1회', { kind: 'good', size: 10, detail: 'SAVE → SRAM', dsize: 7.5 });
    D.arrow(s, g0.cx, g0.b, a1.cx, a1.y, { width: 1 }); D.arrow(s, g1.cx, g1.b, a2.cx, a2.y, { width: 1 });
    D.arrow(s, a1.r, a1.cy, a2.x, a2.cy, { width: 1.5 }); D.arrow(s, a2.r, a2.cy, a3.x, a3.cy, { width: 1.5 });
    D.text(s, 0.5, 3.05, 1.45, 0.4, '허용 (현재 계약)', { size: 8.5, bold: true, color: C.good });
    const r1 = D.box(s, 2.0, 3.7, 1.75, 0.55, 'g₀ → FP16 저장', { kind: 'red', size: 9.5, detail: '중간 반올림', dsize: 7.5 });
    const r2 = D.box(s, 4.0, 3.7, 1.75, 0.55, 'reload + g₁', { kind: 'red', size: 9.5, detail: '누산 재개', dsize: 7.5 });
    const r3 = D.box(s, 6.0, 3.7, 1.6, 0.55, 'FP16 반올림 2회', { kind: 'red', size: 9.5, detail: '결과가 달라짐', dsize: 7.5 });
    D.arrow(s, r1.r, r1.cy, r2.x, r2.cy, { width: 1.5, color: C.red, dash: 'dash' }); D.arrow(s, r2.r, r2.cy, r3.x, r3.cy, { width: 1.5, color: C.red, dash: 'dash' });
    D.text(s, 0.5, 3.8, 1.45, 0.4, '거절 (chain 중단)', { size: 8.5, bold: true, color: C.red });
    D.text(s, 2.0, 4.35, 5.6, 0.3, 'codegen이 같은 출력 tile의 K chain 사이에 다른 명령이 오면 "FP32 matrix chain interrupted"로 거절 (accumulator 1개, partial spill 금지)', { size: 7.5, color: C.red, italic: true });
    D.text(s, 0.5, 4.8, 7.1, 0.24, 'CPU reference도 같은 grouping (split(k,64) + rfactor) — E1: LLVM과 최대 절대차 0', { size: 9, bold: true, color: C.navy });
    D.code(s, 0.5, 5.08, 7.1, 1.25, ['matmul_rf[2,64,64]: g = Σ_{64개} f32(x)·f32(b)      # 부분합 2개', 'matmul[i,j] = g₀ + g₁                            # FP32 결합', 'compute[i,j] = f16(matmul[i,j])                  # 마지막 1회 반올림', '# FoldConstants도 같은 기준. 단순 flat-K loop와 항상 bit-exact는 아님'], { size: 7.5 });
    D.text(s, 7.85, 1.55, 5, 0.24, 'LegalizeOps 뒤 (Relax는 남고 PrimFunc가 생김)', { size: 9, bold: true, color: C.navy });
    D.code(s, 7.85, 1.82, 4.98, 1.3, ['lv1 = call_tir(matmul, (x, lv))   # pattern 4', 'lv2 = call_tir(add, (lv1, bias))  # pattern 0', 'gv  = call_tir(silu, (lv2,))      # pattern 0', '', '# matmul PrimFunc: SSR 축, T.init 0, FP32 buffer,', '#   마지막 block에서 f16 cast'], { size: 7 });
    D.table(s, 7.85, 3.3, 4.98, [
      ['NPU 정의 (custom legalize map)', '왜 다른가'],
      ['matmul', 'FP16×FP16 → FP32 sum → FP16 cast (K=64 grouping)'],
      ['softmax', 'max → exp → FP32 sum → FP16 sum → divide. 마지막 축 1..65504 (exp ≤ 1)'],
      ['rms_norm', 'a = max(absmax, √ε), u = x/a, z = u/√D, Σz² FP32 → FP16 (x² overflow 방지)'],
      ['sum / mean / split', 'FP32 sum → FP16; split은 view (zero-copy)'],
      ['그 외', 'TVM 기본 legalization (add, silu, permute_dims, concat, …)'],
    ], { colW: [1.5, 3.48], size: 8, rowH: 0.42 });
    D.callout(s, 7.85, 5.95, 4.98, 0.45, '지원 밖(길이 70,000 softmax, FP32 RMSNorm 등)은 ValueError로 거절. 조용히 근사하지 않음.', { kind: 'warn', size: 8.5 });
    D.takeaway(s, 'Fusion·schedule은 저장 위치를 바꿀 뿐 이 수치 계약을 바꾸지 않습니다. 정밀도 정책은 고정이며 자동 오차 예산 탐색은 없습니다.', { y: 6.5 });
  });

  // ================================================================ 6 fusion selection
  D.add({ section: '③ 배치', title: 'Fusion은 커널 수가 아니라 실제 프로그램 비용으로 고른다', subtitle: '작은 예제의 분리(off) vs 결합(auto): 두 경우 모두 중간 tile P·A가 SRAM에 머물러 DMA는 같고(41,100 B), 커널 경계의 descriptor·LOAD/SAVE 반복과 짧은 vector 행이 words와 SRAM 작업 공간을 바꿉니다.', source: 'npu_compiler/npu_fusion.py, npu_build.py, docs/examples/pipeline/stage_summary.json (E1), evidence/current/llama_selection.json',
    notes: 'TVM FuseOps가 op_pattern(matmul 4 → add 0 → silu 0)으로 영역을 만들면 npu_fusion.select가 영역마다 unfused 기준과 fused 후보를 실제 schedule·메모리·codegen까지 내려 (DMA, words, SRAM) 사전식으로 비교하고 진 영역은 inline합니다. 그 위에서 lower_and_select가 attention(off/score/full) × GQA·producer × generic(off/vectors/matmul) 최대 27개 graph 후보를 entry 전체 비용으로 비교합니다. 채택 조건은 모든 entry 비악화 + 하나 이상 개선이며 기본 추가 탐색은 120초이고 None으로 해제합니다. 과거 7-token 전체 Llama 기록에서는 후보 0(모두 off)만 legal·selected, 후보 1 검사 중 예산 소진, 25개 미검사였습니다. 이는 "모든 fusion이 더 나쁘다"의 증거가 아닙니다.' },
  (s, D) => {
    const tileAt = (x, y, label, kind, sub) => { D.matrix(s, x, y, 0.5, 0.5, 1, 1, { kind, regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind, label, size: 8 }] }); if (sub) D.text(s, x - 0.3, y + 0.53, 1.1, 0.2, sub, { size: 6.5, color: C.muted, align: 'center' }); };
    let y = 1.75;
    D.text(s, 0.5, y - 0.05, 1.3, 0.6, 'fusion off\n3 kernels', { size: 10, bold: true, color: C.navy });
    D.box(s, 1.8, y, 0.8, 0.5, 'DRAM', { kind: 'dram', size: 9, detail: 'X, Wᵀ, bias', dsize: 6.5 });
    D.arrow(s, 2.62, y + 0.25, 2.95, y + 0.25, { width: 1.2, color: C.dram }); D.text(s, 2.55, y - 0.25, 0.6, 0.2, 'GLOAD', { size: 6.5, color: C.dram, bold: true, align: 'center' });
    const kA = D.box(s, 3.0, y, 1.3, 0.5, 'matmul', { kind: 'unit', size: 9.5, detail: '43 words', dsize: 7 });
    D.arrow(s, kA.r, y + 0.25, kA.r + 0.3, y + 0.25, { width: 1 }); tileAt(kA.r + 0.32, y, 'P', 'sram', 'SRAM 보관');
    D.arrow(s, kA.r + 0.84, y + 0.25, kA.r + 1.12, y + 0.25, { width: 1 });
    const kB = D.box(s, kA.r + 1.15, y, 1.3, 0.5, 'add', { kind: 'ir', size: 9.5, detail: '391 words (64행 × VLEN 64)', dsize: 6.5 });
    D.arrow(s, kB.r, y + 0.25, kB.r + 0.3, y + 0.25, { width: 1 }); tileAt(kB.r + 0.32, y, 'A', 'sram', 'SRAM 보관');
    D.arrow(s, kB.r + 0.84, y + 0.25, kB.r + 1.12, y + 0.25, { width: 1 });
    const kC = D.box(s, kB.r + 1.15, y, 1.55, 0.5, 'silu', { kind: 'ir', size: 9.5, detail: '2,373 words (5단계 × 64행)', dsize: 6.5 });
    D.arrow(s, kC.r, y + 0.25, kC.r + 0.3, y + 0.25, { width: 1 }); tileAt(kC.r + 0.32, y, 'Y', 'good', '');
    D.arrow(s, kC.r + 0.84, y + 0.25, kC.r + 1.12, y + 0.25, { width: 1.2, color: C.dram }); D.text(s, kC.r + 0.8, y - 0.25, 0.7, 0.2, 'GSTORE', { size: 6.5, color: C.dram, bold: true, align: 'center' });
    D.box(s, kC.r + 1.15, y, 0.8, 0.5, 'DRAM', { kind: 'dram', size: 9, detail: 'Y', dsize: 6.5 });
    D.text(s, 3.0, y + 0.85, 8.5, 0.22, '커널 경계마다: 새 descriptor 설정 + P/A를 SRAM에서 다시 LOAD/SAVE (DRAM 왕복은 없음)', { size: 8, color: C.muted });
    y = 3.35;
    D.text(s, 0.5, y - 0.05, 1.3, 0.6, 'fusion auto\n1 kernel', { size: 10, bold: true, color: C.navy });
    D.box(s, 1.8, y, 0.8, 0.5, 'DRAM', { kind: 'dram', size: 9, detail: 'X, Wᵀ, bias', dsize: 6.5 });
    D.arrow(s, 2.62, y + 0.25, 2.95, y + 0.25, { width: 1.2, color: C.dram });
    D.rect(s, 3.0, y - 0.18, 7.35, 0.9, { kind: 'faint', radius: 0.06 });
    D.text(s, 3.1, y - 0.16, 4, 0.2, 'fused_matmul_add_silu  1,572 words', { size: 8, bold: true, color: C.navy });
    const f1 = D.box(s, 3.1, y + 0.08, 1.0, 0.5, 'matmul', { kind: 'unit', size: 9 });
    D.arrow(s, f1.r, y + 0.33, f1.r + 0.25, y + 0.33, { width: 1 }); tileAt(f1.r + 0.27, y + 0.08, 'P', 'sram');
    D.arrow(s, f1.r + 0.79, y + 0.33, f1.r + 1.02, y + 0.33, { width: 1 });
    const f2 = D.box(s, f1.r + 1.05, y + 0.08, 0.9, 0.5, 'add', { kind: 'ir', size: 9 });
    D.arrow(s, f2.r, y + 0.33, f2.r + 0.25, y + 0.33, { width: 1 }); tileAt(f2.r + 0.27, y + 0.08, 'A', 'sram');
    D.arrow(s, f2.r + 0.79, y + 0.33, f2.r + 1.02, y + 0.33, { width: 1 });
    const f3 = D.box(s, f2.r + 1.05, y + 0.08, 1.75, 0.5, 'silu (VLEN 128, 32행)', { kind: 'ir', size: 8.5 });
    D.arrow(s, f3.r, y + 0.33, f3.r + 0.25, y + 0.33, { width: 1 }); tileAt(f3.r + 0.27, y + 0.08, 'Y', 'good');
    D.arrow(s, 10.37, y + 0.25, 10.7, y + 0.25, { width: 1.2, color: C.dram });
    D.box(s, 10.75, y, 0.8, 0.5, 'DRAM', { kind: 'dram', size: 9, detail: 'Y', dsize: 6.5 });
    D.text(s, 3.0, y + 0.85, 8.5, 0.22, '한 커널 안에서 이어짐: descriptor 반복 감소, 축 결합으로 vector 행 64 → 128 (명령 수 절반), USMP offset 공유', { size: 8, color: C.muted });
    y = 4.7;
    D.text(s, 0.5, y - 0.02, 1.3, 0.5, 'SRAM peak', { size: 10, bold: true, color: C.navy });
    D.membar(s, 1.8, y, 4.0, 0.3, 0, 50700, [{ from: 0, to: 50700, label: 'off 50,700 B', kind: 'ir' }], { size: 8 });
    D.membar(s, 1.8, y + 0.4, 4.0, 0.3, 0, 50700, [{ from: 0, to: 42508, label: 'auto 42,508 B', kind: 'sram' }], { size: 8 });
    D.text(s, 6.0, y - 0.02, 1.0, 0.5, 'words', { size: 10, bold: true, color: C.navy });
    D.membar(s, 6.7, y, 4.85, 0.3, 0, 2812, [{ from: 0, to: 43, label: '', kind: 'unit' }, { from: 43, to: 434, label: '391', kind: 'ir' }, { from: 434, to: 2807, label: 'silu 2,373', kind: 'ir' }, { from: 2807, to: 2812, label: '', kind: 'faint' }], { size: 7.5 });
    D.membar(s, 6.7, y + 0.4, 4.85, 0.3, 0, 2812, [{ from: 0, to: 1577, label: 'fused 1,577', kind: 'sram' }], { size: 7.5 });
    // right: candidate grid
    const gx = 12.05 - 4.2, gy = 5.7, gw = 4.2, gh = 0.6;
    D.text(s, 0.5, 5.55, 6.8, 0.24, '선택 규칙: cost = (DMA bytes, words, SRAM peak) 사전식 · 영역 검사(npu_fusion.select) → 전체 module 비교(lower_and_select)', { size: 8.5, bold: true, color: C.navy });
    D.text(s, 0.5, 5.85, 6.8, 0.5, '요청한 entry가 모두 비악화 + 하나 이상 개선일 때 채택. 기본 시간 120초 / E10은 prefill만, 시간 제한 None.', { size: 8, color: C.muted });
    D.text(s, gx - 0.6, gy - 0.22, gw + 0.6, 0.2, '27 조합 → 같은 IR인 source 제거(회색 18, E10 prefill) → 9 구성 · ✓⏱ 과거 7-token 기록', { size: 7, bold: true, color: C.navy });
    D.matrix(s, gx, gy, gw, gh, 3, 9, { kind: 'ir', tile: 3, cellLabel: (r, c) => (r === 0 && c === 0) ? '✓' : ((r === 0 && c === 1) ? '⏱' : (c >= 3 ? '≡' : '')), cellSize: 7, regions: [{ r0: 0, r1: 3, c0: 3, c1: 9, kind: 'faint', fill: 'F3F5F8', dash: 'dash' }, { r0: 0, r1: 1, c0: 0, c1: 1, kind: 'good' }, { r0: 0, r1: 1, c0: 1, c1: 2, kind: 'warn' }] });
    D.text(s, gx - 0.6, gy + gh + 0.02, gw + 1.4, 0.2, '✓ 후보 0 (모두 off) 선택 · ⏱ 후보 1에서 120 s 소진 (7-token, 189.7 s) · ≡ E10 prefill에서 같은 source', { size: 6.5, color: C.muted });
    D.takeaway(s, 'E1: 전송량은 같아도 명령과 작업 공간이 줄어든 사례 (2,812 → 1,577 words, 50,700 → 42,508 B). 이득의 형태는 예제마다 달라 실제 비용으로 확인합니다.', { y: 6.5 });
  });

  // ================================================================ 7 matmul scheduling
  D.add({ section: '③ 배치', title: 'Matmul: 64×64 tile은 고정, 자유도는 panel(무엇을 SRAM에 얼마나 오래)과 순서', subtitle: '실제 폭 gate projection X[64,3072]@W[3072,8192]: W는 48 MiB라 X(1×48 tiles, 384 KiB)를 상주시키고 W 열 panel j(48×1 tiles)를 128번 교체합니다. 아래는 두 matmul 예제 두 번째 커널의 실제 20 words.', source: 'npu_schedule_space.py, npu_schedule.py, npu_intrin.py, evidence/current/refinement.json, two_matmuls_recorded/program.json (words 52–71)',
    notes: '탐색 축은 m/n panel, K chunk, outer/inner 순서, X/W cache 위치(a_at/b_at)이며 K는 항상 최내측입니다(accumulator 1개). 후보는 추정 DMA 순으로 검사하되 실제 tir.Schedule 변환 → CompactBufferAllocation → USMP 배치 → 실제 명령 생성으로 합법성과 (DMA, words, SRAM)을 확인하며 기본 시도64·legal8 목표입니다. 동일 후보를 정규화하고 증명 가능한 SRAM 초과만 미리 제외합니다. Words 예산은 max(설정값, 기준 words×legal 목표)이며 최소 두 합법 후보 전에는 words 때문에 종료하지 않습니다. 시도 한도는 별도입니다. TVM trace DB 재사용과 ReplayFunc 탐색은 선택형입니다. E10은 시도128·legal16 목표로 확장합니다. gate/up은 ("A",64) input-stationary(X 유지, W panel 384 KiB 교체, 6,144 M_MUL), down은 ("B",256) weight-stationary(W 4 MiB 유지, H는 SRAM 보관값)가 선택됐습니다. 아래 word strip은 X₀·B₀(MAC=0) → X₁·B₁(MAC=1) → SAVE의 실제 명령이며 두 번째 tile에서는 주소 descriptor만 바뀝니다(COLS는 peephole이 생략).' },
  (s, D) => {
    const Wd = D.matrix(s, 0.5, 1.85, 4.0, 1.5, 6, 16, { kind: 'dram', title: 'DRAM: W [3072,8192] = 48 × 128 tiles (48 MiB) — 그림 6×16 축약', titleW: 4.4, titleSize: 8, regions: [{ r0: 0, r1: 6, c0: 3, c1: 4, kind: 'sram', label: 'j', size: 7 }], below: '열 panel j = 48 tiles = 384 KiB, j = 0 … 127 순서로 GLOAD', belowW: 4.0, belowSize: 7 });
    D.rect(s, 5.0, 1.6, 4.1, 2.2, { kind: 'sram', radius: 0.08, fill: 'EEF8F6' });
    D.text(s, 5.1, 1.62, 3.9, 0.22, 'SRAM', { size: 9.5, bold: true, color: C.sram });
    const Xp = D.matrix(s, 5.15, 2.1, 3.8, 0.28, 1, 12, { kind: 'sram', title: 'X [64,3072] = 1×48 tiles 상주 (384 KiB), a_at = m_panel', titleW: 3.9, titleSize: 7.5, regions: [{ r0: 0, r1: 1, c0: 0, c1: 12, kind: 'sram', label: 'k = 0 … 47', size: 7 }] });
    const Wp = D.matrix(s, 5.15, 2.75, 0.28, 0.95, 8, 1, { kind: 'sram', regions: [{ r0: 0, r1: 8, c0: 0, c1: 1, kind: 'dram', lineWidth: 1 }] });
    D.text(s, 5.5, 2.75, 1.7, 0.6, 'W panel j\n48×1 tiles 384 KiB\n(b_at = n_panel)', { size: 7, color: C.dram });
    D.arrow(s, Wd.x + Wd.cw * 3.5, Wd.y, 5.2, 2.75, { width: 1.5, color: C.dram, dash: 'dash' }); D.text(s, 4.52, 2.4, 0.5, 0.4, 'GLOAD\npanel j', { size: 6.5, color: C.dram, bold: true, align: 'center' });
    const Ot = D.matrix(s, 8.0, 2.9, 0.55, 0.55, 1, 1, { kind: 'unit', title: '출력 tile (0, j)', titleW: 1.3, titleSize: 7, regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind: 'unit', label: '48\nM_MUL', size: 6.5 }] });
    D.arrow(s, Wp.r + 0.05, Wp.cy, Ot.x, Ot.cy, { width: 1.5, color: C.unit }); D.arrow(s, Xp.cx, Xp.b, Ot.cx, Ot.y, { width: 1.5, color: C.unit });
    D.text(s, 6.3, 3.35, 1.7, 0.4, 'X tile k × W tile k, k=0..47\n(MAC 0, 1, 1, …)', { size: 6.5, color: C.muted });
    const Y = D.matrix(s, 0.5, 4.05, 8.6, 0.28, 1, 32, { kind: 'faint', title: '출력 [64,8192] = 1 × 128 tiles: tile j 완성(FP16 SAVE) → panel j+1 · 128 × 48 = 6,144 M_MUL, W 48 MiB 각 1회, X 1회', titleW: 8.6, titleSize: 7.5, regions: [{ r0: 0, r1: 1, c0: 7, c1: 8, kind: 'unit', label: 'j', size: 7 }, { r0: 0, r1: 1, c0: 0, c1: 7, kind: 'good', label: '완료', size: 7 }] });
    D.arrow(s, Ot.cx, Ot.b, Y.x + Y.cw * 7.5, Y.y, { width: 1.5, color: C.unit });
    D.bullets(s, 9.4, 1.6, 3.43, 2.9, [
      ['탐색 축: ', 'm/n panel, K chunk, outer/inner 순서, X/W cache 위치(a_at/b_at). K는 항상 최내측 (accumulator 1개)'],
      ['검사: ', '같은 순회·복사 주기의 표기는 후보 하나로 정규화, 증명된 SRAM 초과는 제외 → 추정 DMA 순으로 실제 schedule·compaction·USMP·명령 생성 → (DMA, words, SRAM). 시도 64, legal 8, words max(100k, 기준×8)'],
      ['선택형: ', 'MetaSchedule ReplayFunc 추첨 탐색 · trace DB 재사용 (기본 off, 마지막 슬라이드)'],
      ['선택 기록: ', 'gate/up ("A",64) X 유지 · down ("B",256) W 4 MiB 유지, H는 SRAM 보관값'],
      ['tensorize: ', '64×64 조각 → npu_gemm_acc intrinsic → codegen이 descriptor·LOAD·M_MUL로'],
    ], { size: 8, psa: 3 });
    D.text(s, 0.5, 4.55, 12, 0.22, '실제 명령: H[64,128] @ W2[128,64] (word 52–71, 20 words) — ① X₀·B₀ MAC=0 ② X₁·B₁ MAC=1 ③ SAVE 1회', { size: 9, bold: true, color: C.navy });
    const W = (label, kind, extra) => Object.assign({ n: 1, label, kind, size: 6.5 }, extra || {});
    D.wordStrip(s, 0.5, 4.95, 12.33, 0.42, [
      W('ADDR\nSRC1 M', 'ir', { above: '① X₀ → SRC1', tick: 52 }), W('COLS\n128', 'host'), W('ADDR\nSRC1 P', 'ir'), W('LOAD\nSRC1', 'sram', { bold: true }),
      W('ADDR\nSRC2 M', 'ir', { above: 'B₀ → SRC2' }), W('COLS\n64', 'host'), W('ADDR\nSRC2 P', 'ir'), W('LOAD\nSRC2', 'sram', { bold: true }),
      W('M_MUL\nMAC=0', 'unit', { bold: true, tick: 60 }),
      W('ADDR\nSRC1 M', 'ir', { above: '② X₁ (+64열)' }), W('ADDR\nSRC1 P', 'ir'), W('LOAD\nSRC1', 'sram', { bold: true }),
      W('ADDR\nSRC2 M', 'ir', { above: 'B₁ (+64행)' }), W('ADDR\nSRC2 P', 'ir'), W('LOAD\nSRC2', 'sram', { bold: true }),
      W('M_MUL\nMAC=1', 'unit', { bold: true, tick: 67 }),
      W('ADDR\nDST M', 'ir', { above: '③ Y 저장' }), W('COLS\n64', 'host'), W('ADDR\nDST P', 'ir'), W('SAVE', 'good', { bold: true, tick: 71 }),
    ], { size: 6.5 });
    D.text(s, 0.5, 5.6, 12.33, 0.4, 'descriptor(ADDR/COLS)를 바꾸고 LOAD → M_MUL(MAC bit) → 마지막에 SAVE. 두 번째 tile에서 COLS가 없는 이유: rows/cols가 그대로라 peephole이 생략. loop·branch 명령은 없음(정적 전개).', { size: 8, color: C.muted });
    D.takeaway(s, 'K chunk 숫자만 보고 DMA를 판단하면 안 됩니다. cache stage가 어느 loop에 있는지(a_at/b_at)와 순회 순서가 재읽기 횟수, 곧 DMA를 정합니다.', { y: 6.5 });
  });

  // ================================================================ 8 vector & attention
  D.add({ section: '③ 배치', title: 'Matmul 이외: SiLU 하나가 vector 명령 5개, attention은 query block으로 score를 SRAM에서 소비', subtitle: '원소별 연산은 VLEN 길이의 행이 SRAM scratch slot을 거치며 단계마다 descriptor·LOAD·SAVE가 붙습니다(왼쪽, 실제 opcode). Full-attention fusion은 query 행 block마다 K/V 전체와 score 행 전체를 SRAM에 두고 정확한 softmax를 합니다(오른쪽).', source: 'tir_codegen_v09.py (_materialize, _sigmoid), docs/examples/pipeline/09_packets.json, npu_intrin.schedule_attention_blocked, npu_schedule._fit_attention',
    notes: 'Walker는 표현식을 6개 scratch slot에 직렬화합니다. sigmoid는 SIGN_INV → EXP → BROADCAST(1.0) → ADD → DIV이고 SiLU의 마지막 곱은 별도 block입니다. 중간값은 FP16 SRAM 행이라 SiLU 하나가 ISA 하나가 아니며, generic vector 커널은 전체가 안 들어가면 행 묶음을 반으로 줄이고 연속 축을 결합해 행을 길게 만듭니다(64×64 → 32×128). Attention full fusion은 정적 rank-3 FP16 64배수 조건에서만 가능하고 row block을 head loop 바깥에 두어 block마다 모든 head를 처리하며, 실패하면 block을 반으로 줄입니다. key까지 나누는 online softmax(FlashAttention)는 없습니다. decode(m=1)는 조건 밖이라 score chain fusion 또는 분리 matmul을 씁니다.' },
  (s, D) => {
    const lx = 2.35, lw = 3.1, lh = 0.27, tW = 1.8;
    const mk = (pre, kind) => Array.from({ length: 8 }, (_, i) => ({ label: i === 7 ? '…' : `${pre}${i}`, kind, size: 6.5 }));
    D.text(s, 0.5, 1.55, 6.5, 0.24, 'SiLU 한 행 (VLEN 128, FP16 scratch) — 오른쪽은 실제 opcode word', { size: 9, bold: true, color: C.navy });
    const rows = [
      ['x 행 (T_add 결과)', mk('x', 'host'), 'SRC1.PARTIAL'],
      ['slot0 = −x', mk('−x', 'unit'), 'SIGN_INV 00000016'],
      ['slot1 = exp(slot0)', mk('e', 'unit'), 'EXP 0000000f'],
      ['slot2 = 1.0 (broadcast)', Array.from({ length: 8 }, () => ({ label: '1', kind: 'ir', size: 6.5 })), 'BCAST ×2 words'],
      ['slot3 = slot1 + slot2', mk('1+e', 'unit'), 'ADD 80000001'],
      ['sig = 1 / slot3', mk('σ', 'sram'), 'DIV 8000000b'],
      ['y = x × sig', mk('y', 'good'), 'MUL 8000000a'],
    ];
    rows.forEach((r, i) => {
      const y = 1.9 + i * 0.36;
      D.lanes(s, lx, y, lw, lh, r[1], { title: r[0], titleW: tW, right: r[2], rightW: 1.2, rightMono: true, rightSize: 6.5 });
      if (i > 0 && i < 6) D.arrow(s, lx + lw / 2, y - 0.09, lx + lw / 2, y, { width: 0.75, color: C.muted });
    });
    D.text(s, 0.5, 4.5, 6.5, 0.6, '단계마다 VLEN·ADDR·LOAD(·LOAD)·OP·ADDR·SAVE → E1 fused 1,577 words 중 ADDR(0x80) 606, LOAD/SAVE 388/258. 행이 안 들어가면 stripe 반분, 연속 축 결합으로 64행×64 → 32행×128 (명령 절반).', { size: 8, color: C.muted });
    D.callout(s, 0.5, 5.2, 6.5, 1.15, [{ text: 'Softmax / RMSNorm도 같은 vector 경로: ', bold: true }, { text: 'REDUCE_MAX → SUB → EXP → REDUCE_SUM(FP32 내부) → DIV; RMSNorm은 absmax scale 후 Σz². 물리 256 lanes vs 논리 VLEN ≤ 65535는 RTL 미확인 항목이며 현재 비용 모델에 lane 활용률은 없습니다.', size: 8.5 }], { kind: 'faint' });
    // attention figure (right)
    const ax = 7.25, y = 2.2, blk = [2, 4];
    D.text(s, ax, 1.55, 5.6, 0.24, 'Full-attention fusion: query block (진한 행)만 계산, score 행 전체를 SRAM에서 softmax', { size: 9, bold: true, color: C.navy });
    const Q = D.matrix(s, ax, y + 0.3, 0.45, 1.6, 8, 1, { kind: 'faint', title: 'Q [S,128]', titleW: 0.9, titleSize: 7, regions: [{ r0: blk[0], r1: blk[1], c0: 0, c1: 1, kind: 'unit', label: 'blk', size: 6.5 }] });
    const K = D.matrix(s, ax + 0.7, y, 2.0, 0.3, 1, 8, { kind: 'sram', title: 'Kᵀ [128,S] head 전체', titleW: 2.0, titleSize: 7 });
    const masked = []; for (let r = 0; r < 8; r++) for (let c = r + 1; c < 8; c++) masked.push({ r0: r, r1: r + 1, c0: c, c1: c + 1, kind: 'faint', lineWidth: 0.3 });
    const Sc = D.matrix(s, ax + 0.7, y + 0.4, 2.0, 1.5, 8, 8, { kind: 'faint', regions: [{ r0: blk[0], r1: blk[1], c0: 0, c1: 8, kind: 'unit' }].concat(masked), below: 'scores [S,S] block 행 (/√128 +mask), 회색 = causal 0', belowW: 2.3, belowSize: 6.5 });
    D.arrow(s, Q.r + 0.02, Q.y + 1.6 * 3 / 8, Sc.x, Sc.y + 1.5 * 3 / 8, { width: 1.2, color: C.unit });
    D.arrow(s, Sc.r + 0.03, Sc.y + 1.5 * 3 / 8, Sc.r + 0.4, Sc.y + 1.5 * 3 / 8, { width: 1.2, color: C.unit });
    D.text(s, Sc.r - 0.05, Sc.y - 0.03, 0.55, 0.3, 'softmax', { size: 6, color: C.muted, align: 'center' });
    const P = D.matrix(s, ax + 3.15, y + 0.4, 1.4, 1.5, 8, 8, { kind: 'faint', regions: [{ r0: blk[0], r1: blk[1], c0: 0, c1: 8, kind: 'unit' }].concat(masked), below: 'P block (SRAM)', belowW: 1.4, belowSize: 6.5 });
    const V = D.matrix(s, ax + 4.75, y + 0.3, 0.35, 1.6, 8, 1, { kind: 'sram', title: 'V', titleW: 0.5, titleSize: 7 });
    const O = D.matrix(s, ax + 5.25, y + 0.3, 0.35, 1.6, 8, 1, { kind: 'faint', title: 'out', titleW: 0.5, titleSize: 7, regions: [{ r0: blk[0], r1: blk[1], c0: 0, c1: 1, kind: 'good' }] });
    D.arrow(s, P.r + 0.02, P.y + 1.5 * 3 / 8, V.x, V.y + 1.6 * 3 / 8, { width: 1.2, color: C.unit }); D.arrow(s, V.r + 0.02, V.y + 1.6 * 3 / 8, O.x, O.y + 1.6 * 3 / 8, { width: 1.2, color: C.good });
    D.bullets(s, ax, 4.55, 5.6, 1.8, [
      ['schedule: ', 'row block을 head loop 바깥에 (reorder) → block마다 모든 head. scores·max·exp·sum·PV acc는 SRAM에만'],
      ['block 크기: ', '가장 큰 block부터, SRAM 초과 시 반분 (floor = min(rows,64)). K/V는 block마다 재읽기'],
      ['S=3072: ', '24-head score 432 MiB → head 1개 64행 block 384 KiB. 조건: rank-3 FP16, m·d·t 64배수 (decode m=1은 제외)'],
      ['없는 것: ', 'key 방향 online softmax (FlashAttention). 한 block이 행 전체를 가져 softmax는 정확'],
    ], { size: 8, psa: 3 });
    D.takeaway(s, 'words는 짧은 vector 행과 descriptor 반복에 좌우되고, attention은 384 KiB score만이 아니라 K/V·mask·확률·출력이 함께 SRAM에 있어야 합니다.', { y: 6.5 });
  });

  // ================================================================ 9 memory planning
  D.add({ section: '④ 메모리', title: '메모리 계획: DRAM pool, 커널 안의 offset 공유, 커널 사이 SRAM 보관', subtitle: '① TVM StaticPlanBlockMemory가 수명이 겹치지 않는 tensor를 같은 pool에 두고 backend가 flat 주소를 줍니다. ② USMP가 커널 buffer의 충돌을 분석해 offset을 공유합니다. ③ 보관 탐색이 중간값 H를 SRAM 위쪽에 예약하고 남은 용량으로 커널을 다시 schedule합니다.', source: 'npu_memplan.py, npu_sram.py, npu_memory_search.py, bundle/program.json sram_layouts, two_matmuls_recorded/program.json, examples/memory/measurements.json (E2)',
    notes: '왼쪽 아래는 작은 예제 fused 커널의 실제 USMP 결과입니다: 8개 buffer(총 163,840 nibbles)가 충돌하지 않는 것끼리 offset을 공유해 81,920 nibbles에 들어가며(bump 150,808 → peak 85,016), gemm 목적지는 pinned로 offset 0을 혼자 씁니다. 오른쪽은 두 matmul 예제(SRAM 44 KiB)의 실제 주소입니다: 첫 커널이 H를 26,124에 직접 SAVE하고 두 번째 커널이 그 위치를 직접 LOAD하므로 H의 GSTORE/GLOAD 32,768 B가 사라져 DMA 81,932 → 49,164 B, 미사용 DRAM pool 제거로 image 65,548 → 49,164 B가 됩니다. 32 KiB에서는 첫 matmul이 안 들어가 보관이 적용되지 않습니다. 보관은 tensor 전체만, beam 4, all-DRAM 진행도 항상 비교하며 이득이 실제 비용으로 확인될 때만 채택합니다. KV cache와 출력·parameter는 ABI 계약으로 보호됩니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 12, 0.22, '① DRAM image (flat, unit 2 B): parameter → 상수 → storage pool(수명 안 겹치는 tensor 공유) → 출력 → literal', { size: 8.5, bold: true, color: C.dram });
    D.membar(s, 0.5, 1.74, 12.33, 0.38, 0, 100, [
      { from: 0, to: 30, label: 'parameter leaves (변환된 순서)', kind: 'dram' }, { from: 30, to: 38, label: 'graph 상수', kind: 'host' },
      { from: 38, to: 56, label: 'pool A: G → U 재사용', kind: 'ir' }, { from: 56, to: 70, label: 'pool B: S → H', kind: 'ir' }, { from: 70, to: 86, label: '출력 tree', kind: 'unit' }, { from: 86, to: 100, label: 'literal · FP32 1.0', kind: 'faint' },
    ], { size: 7.5 });
    D.lifetimeChart(s, 3.0, 2.75, 3.9, 2.3, 0, 81920, [
      { name: 'matmul acc 목적지 (pinned)', from: 0, to: 16384, kind: 'unit', pinned: true },
      { name: 'T_add (A)', from: 16384, to: 32768, kind: 'sram' },
      { name: 'Wᵀ panel', from: 16384, to: 49152, kind: 'dram', label: '공유' },
      { name: 'P f16 / Y (수명 안 겹침)', from: 32768, to: 49152, kind: 'sram', label: '공유' },
      { name: 'sigmoid 결과', from: 49152, to: 65536, kind: 'sram' },
      { name: 'bias (256 nib)', from: 49152, to: 49408, kind: 'ir' },
      { name: 'X panel', from: 49152, to: 81920, kind: 'dram', label: '공유' },
    ], { labelW: 2.4, size: 7.5, ticks: [0, 16384, 32768, 49152, 65536, 81920], title: '② USMP (작은 예제 fused 커널): 충돌 없는 buffer가 offset 공유 — nibble' });
    D.text(s, 0.5, 5.3, 6.4, 0.6, 'peak 150,808(bump) → 85,016 nibbles (32,896 B 절약). gemm 목적지는 SAVE가 지연되므로 pinned(모든 buffer와 충돌)로 offset 0을 혼자 씀. greedy_by_size / greedy_by_conflicts vs bump 중 최소 peak 채택.', { size: 7.5, color: C.muted });
    // right: two-matmul retention
    const rx = 7.25, rw = 5.58;
    D.text(s, rx, 2.4, rw, 0.22, '③ 커널 사이 보관 (두 matmul, SRAM 44 KiB, 실제 주소 bytes)', { size: 8.5, bold: true, color: C.warn });
    D.text(s, rx, 2.65, rw, 0.2, '커널 1: X @ W1 → H (H를 SRAM에 직접 SAVE)', { size: 7.5, color: C.muted });
    D.membar(s, rx, 2.87, rw, 0.42, 0, 45056, [{ from: 0, to: 1548, label: 'pool', kind: 'host' }, { from: 1548, to: 9740, label: 'X 8,192', kind: 'sram' }, { from: 9740, to: 26124, label: 'W1 16,384', kind: 'dram' }, { from: 26124, to: 42508, label: 'H 16,384 (보관)', kind: 'warn' }], { ticks: [0, 1548, 9740, 26124, 42508, 45056], size: 7 });
    D.text(s, rx, 3.55, rw, 0.2, '커널 2: H @ W2 → Y (H를 그 자리에서 직접 LOAD)', { size: 7.5, color: C.muted });
    D.membar(s, rx, 3.77, rw, 0.42, 0, 45056, [{ from: 0, to: 1548, label: 'pool', kind: 'host' }, { from: 1548, to: 9740, label: 'Y 작업 (X 자리)', kind: 'sram' }, { from: 9740, to: 26124, label: 'W2 (W1 자리)', kind: 'dram' }, { from: 26124, to: 42508, label: 'H 유지 → GSTORE/GLOAD 없음', kind: 'warn' }], { ticks: [0, 1548, 9740, 26124, 42508, 45056], size: 7 });
    D.arrow(s, rx + rw * (34316 / 45056), 3.29, rx + rw * (34316 / 45056), 3.77, { color: C.warn, width: 2 });
    D.table(s, rx, 4.5, rw, [
      ['E2 (실제 실행)', 'DMA bytes', 'words', 'SRAM peak', 'DRAM image'],
      ['32 KiB (보관 불가)', '81,932', '88', '26,124', '65,548'],
      ['44 KiB 보관 off', '81,932', '87', '34,316', '65,548'],
      ['44 KiB 보관 on', '49,164', '77', '42,508', '49,164'],
    ], { colW: [1.6, 1.05, 0.8, 1.0, 1.13], size: 8, rowH: 0.26, aligns: ['left', 'right', 'right', 'right', 'right'] });
    D.text(s, rx, 5.65, rw, 0.7, 'H의 저장·재읽기 32,768 B 제거, H의 DRAM pool 제거(주소 재배치). 보관은 tensor 전체만, beam 4 + all-DRAM 비교, 이득이 실제 비용일 때만 채택. KV cache·출력·parameter는 ABI 계약으로 보호(이름이 아님).', { size: 7.5, color: C.muted });
    D.takeaway(s, '커널을 합치지 않아도 SRAM만으로 연결할 수 있습니다. 같은 주소가 나중에 다른 값이 될 수 있으므로 보관 탐색은 "값의 버전"을 키로 삼습니다.', { y: 6.5 });
  });

  // ================================================================ 10 ISA & linker
  D.add({ section: '⑤ 실행', title: 'ISA: 정적으로 전개된 명령 stream — descriptor 설정, LOAD/SAVE, M_MUL, DMA packet', subtitle: '두 matmul 예제의 77 words 전체(위)와 실제 word 해독(아래). walker가 TIR loop를 완전히 unroll하므로 loop·branch 명령이 없고, linker가 커널 stream을 이어 붙인 뒤 peephole·forwarding·DRAM 압축으로 마무리합니다.', source: 'two_matmuls_recorded/program_memory.bin, program.json; tir_codegen_v09.py, v09_walker.cc, npu_link.py, peephole.py, isa_v09.py',
    notes: 'word 0-3 상수 pool GLOAD, 4-47 커널 0(W1·X GLOAD, 두 출력 tile 각각 MAC=0 → H 직접 SAVE), 48-75 커널 1(W2 GLOAD가 W1 자리, K tile 두 개 MAC=0/1, Y SAVE, GSTORE), 76 HALT. DMA packet은 4 words(w0 SRAM nibble 주소+opcode, w1 global cell, w2 stride, w3 rows/cols)이고 나머지는 1 word입니다. Descriptor(0x80 주소, 0x88/0x89 rows/cols·dtype, 0x82 VLEN)는 상태 register라 뒤 명령이 참조하며, peephole은 값이 같을 때만 중복 설정을 제거합니다(커널 1에 0x88 없음). M_MUL word의 bit 27이 MAC이며 80000042(MAC=0)와 88000042(MAC=1)만 다릅니다. Python walker와 native C++ walker는 같은 contract hash를 공유하고 words·DMA bytes·SRAM peak 통계와 C-model counters를 대조합니다. 결과물 bundle: program_memory.bin, program.json(주소·통계·ABI·provenance), lowered.py, scheduled_*.py, global_memory.bin.' },
  (s, D) => {
    const seq = 'GGGAARRAARRLAARRAARRLMAARRAARRSLAALMAASGARALARALMAALAALMARASTH';
    const kindOf = { G: 'dram', T: 'dram', A: 'ir', R: 'host', L: 'sram', S: 'good', M: 'unit', H: 'red' };
    const items = []; let w = 0;
    for (const ch of seq) { const n = (ch === 'G' || ch === 'T') ? 4 : 1; const it = { n, kind: kindOf[ch], label: ch === 'M' ? 'M' : (ch === 'G' ? 'GLOAD' : (ch === 'T' ? 'GSTORE' : (ch === 'H' ? 'HALT' : ''))), size: 6 }; if (w === 0 || w === 4 || w === 8 || w === 30 || w === 44 || w === 48 || w === 60 || w === 67 || w === 72 || w === 76) it.tick = w; items.push(it); w += n; }
    D.text(s, 0.5, 1.5, 12.33, 0.22, '77 words 전체 = 62 instructions (DMA 5개 × 4 words + 1-word 명령 57개): 주황 DMA · 파랑 ADDR · 회색 ROWS/COLS · 청록 LOAD · 초록 SAVE · 보라 M_MUL · 빨강 HALT', { size: 8.5, bold: true, color: C.navy });
    D.wordStrip(s, 0.5, 1.78, 12.33, 0.42, items, { size: 6 });
    D.text(s, 0.5, 2.42, 12.33, 0.22, '0–3 상수 pool  ·  4–47 커널 0 (W1·X GLOAD, tile 0 MAC=0 SAVE, tile 1 MAC=0 SAVE → H 직접)  ·  48–75 커널 1 (W2 GLOAD, MAC 0/1, SAVE Y, GSTORE)  ·  76 HALT', { size: 7.5, color: C.muted });
    // DMA packet decode (compact)
    D.text(s, 0.5, 2.8, 6.4, 0.22, 'DMA packet 해독 (word 8–11: X를 SRAM byte 1548로)', { size: 9, bold: true, color: C.navy });
    const words = [['000c18a0', [{ name: 'SRAM nibble 주소', hi: 31, lo: 8, kind: 'sram', value: '0xC18 = 3096 nib = byte 1548' }, { name: 'op', hi: 7, lo: 0, kind: 'unit', value: '0xA0' }]],
      ['00000000', [{ name: 'global cell 주소', hi: 31, lo: 0, kind: 'dram', value: '0 → byte 0 (X)' }]],
      ['00000800', [{ name: 'row stride (cells)', hi: 31, lo: 0, kind: 'dram', value: '2048 cells = 8192 B' }]],
      ['00010800', [{ name: 'rows', hi: 31, lo: 16, kind: 'ir', value: '1' }, { name: 'cols (cells)', hi: 15, lo: 0, kind: 'ir', value: '2048 = 8192 B' }]]];
    words.forEach((wd, i) => { const y = 3.3 + i * 0.62; D.text(s, 0.5, y + 0.06, 0.9, 0.3, wd[0], { size: 8.5, font: F.code, bold: true }); D.bitfield(s, 1.45, y, 5.4, 0.42, wd[1], { size: 7 }); });
    D.text(s, 0.5, 5.8, 6.4, 0.5, '단위: SRAM nibble(fp16 = 4), global 32-bit cell(×4 B), plan unit 2 B. 한 숫자를 다른 단위로 읽으면 2배/4배 어긋남. DMA는 dtype 없는 cell 이동.', { size: 7.5, color: C.muted });
    // M_MUL + descriptor + linker (right)
    D.text(s, 7.25, 2.8, 5.6, 0.22, 'M_MUL word: 80000042 (MAC=0) vs 88000042 (MAC=1)', { size: 9, bold: true, color: C.navy });
    D.bitfield(s, 7.25, 3.3, 5.58, 0.42, [{ name: 'mode', hi: 31, lo: 30, kind: 'unit', value: 'VECTOR' }, { name: 'act', hi: 29, lo: 28, kind: 'ir', value: 'OFF' }, { name: 'MAC', hi: 27, lo: 27, kind: 'red', value: '0/1' }, { name: '', hi: 26, lo: 24, kind: 'faint' }, { name: 'imm', hi: 23, lo: 8, kind: 'faint', value: '0' }, { name: 'op', hi: 7, lo: 0, kind: 'unit', value: '0x42' }], { size: 7 });
    D.table(s, 7.25, 3.95, 5.58, [
      ['Descriptor = 상태 register', '뜻'],
      ['0x80 ADDR', 'SRC1/SRC2/DST × MAIN/PARTIAL 주소 (16-bit 절반씩)'],
      ['0x88/0x89 ROWS/COLS', 'rows/cols + dtype. MAIN cols = row stride (128 el = 256 B)'],
      ['0x90 LOAD / 0x98 SAVE', 'PARTIAL 64×64 view를 MAIN stride로 읽고/쓰기'],
      ['peephole', '값이 같을 때만 중복 설정 삭제 (커널 1에 0x88 없음)'],
    ], { colW: [1.9, 3.68], size: 7.5, rowH: 0.27 });
    D.callout(s, 7.25, 5.5, 5.58, 0.85, [{ text: 'Linker: ', bold: true }, { text: '상수 pool GLOAD → 커널 stream 연결(binding 순서) → HALT → residency forwarding → peephole → joint vs all-DRAM 비교 → 미사용 DRAM pool 제거 → KV append 재구성. Python/native walker는 같은 contract hash를 공유하고 C-model counters와 대조합니다.', size: 8 }], { kind: 'faint' });
    D.takeaway(s, 'words는 "명령 저장 공간"이며 cycle이 아닙니다. bundle(program_memory.bin + program.json + IR + image)만으로 실장치 latency를 주장하지 않습니다.', { y: 6.5 });
  });

  // ================================================================ 11 runtime
  D.add({ section: '⑤ 실행', title: 'Runtime: prefill 1개 + decode 1개 프로그램, cache는 DRAM에 남고 새 K/V는 DMA tail이 슬롯에 쓴다', subtitle: 'Decode는 capacity C개 슬롯 + 현재 token으로 항상 C+1 길이를 attention하고 mask가 유효 길이를 전달합니다. host는 embedding·cos/sin·mask만 patch(6,676 B)하고, program 끝의 append tail이 새 K/V를 슬롯 L에 옮깁니다.', source: 'npu_generate.py (generate, DecodeSession, length_mask), npu_cache.AppendPlan, v09_runtime.PersistentTransport, evidence/current/llama_validation.json',
    notes: 'generate()는 prefill_cache(S=7)와 decode(C=9) 두 프로그램을 한 번 링크하고 weight를 한 번 읽습니다. prefill 뒤 host가 argmax로 첫 token을 고르고 K/V S행을 capacity 배열 앞에 복사합니다. 각 decode step에서 host는 token embedding, position의 cos/sin, length_mask(유효 슬롯 0, 나머지 −65504, 마지막 현재 token 0)를 image의 정해진 주소에 patch합니다. K cache는 [128,C]로 transposed 저장이라 새 K(128×1)는 열 하나(행마다 2 B, stride 18 B)이고 V cache는 [C,128]이라 새 V는 연속 256 B 행입니다. AppendPlan이 position으로 DMA 주소를 fixup한 tail을 HALT 앞에 넣고, PersistentTransport가 program suffix와 입력만 patch한 뒤 C-model을 실행하고 saved image를 다음 입력으로 rename합니다. 기록: "Hello, NPU compiler!" → 358, 2846, 4560 = " I\'m trying", HF argmax 일치, cosine ≥ 0.99999.' },
  (s, D) => {
    // token timeline (top-left)
    D.text(s, 0.5, 1.5, 6.4, 0.22, 'prefill(7 tokens) → decode → decode: 실제 3-token 기록', { size: 9, bold: true, color: C.navy });
    const toks = [['<bos>', '128000'], ['Hello', '9906'], [',', '11'], [' N', '452'], ['PU', '6459'], [' compiler', '19979'], ['!', '0']];
    toks.forEach((t, i) => { const x = 0.5 + i * 0.6; D.rect(s, x, 1.78, 0.55, 0.5, { kind: 'dram', radius: 0.04 }); D.text(s, x, 1.8, 0.55, 0.24, t[0], { size: 6.5, bold: true, align: 'center' }); D.text(s, x, 2.02, 0.55, 0.22, t[1], { size: 5.5, align: 'center', color: C.muted, font: F.code }); });
    [['" I" 358', 'prefill'], ['"\'m" 2846', 'decode @7'], ['" trying" 4560', 'decode @8']].forEach((g, i) => { const x = 4.6 + i * 0.88; D.rect(s, x, 1.78, 0.82, 0.5, { kind: 'unit', radius: 0.04 }); D.text(s, x, 1.8, 0.82, 0.24, g[0], { size: 6.5, bold: true, align: 'center' }); D.text(s, x, 2.02, 0.82, 0.22, g[1], { size: 6, align: 'center', color: C.muted }); if (i < 2) D.arrow(s, x + 0.83, 2.03, x + 0.87, 2.03, { width: 0.75 }); });
    D.arrow(s, 4.42, 2.03, 4.58, 2.03, { width: 1 });
    // slot grid
    D.text(s, 0.5, 2.5, 6.4, 0.22, 'decode @7이 보는 논리적 입력: capacity C=9 + 현재 = 10 슬롯, 유효 길이 L=7', { size: 8.5, bold: true, color: C.navy });
    for (let i = 0; i < 10; i++) { const kind = i < 7 ? 'dram' : (i < 9 ? 'faint' : 'unit'); const x = 0.5 + i * 0.63; D.rect(s, x, 2.75, 0.58, 0.5, { kind, radius: 0.04 }); D.text(s, x, 2.77, 0.58, 0.22, String(i), { size: 9, bold: true, align: 'center' }); D.text(s, x, 2.99, 0.58, 0.24, i < 7 ? '유효' : (i < 9 ? '빈' : '현재'), { size: 6, align: 'center', color: C.muted }); }
    D.text(s, 0.5, 3.28, 6.4, 0.22, 'mask [1,1,10]:  0 0 0 0 0 0 0  −65504 −65504  0   → softmax가 빈 슬롯의 기여를 0으로 (계산 자체는 수행)', { size: 7, font: F.code, color: C.navy });
    // KV append figure (bottom-left)
    D.text(s, 0.5, 3.65, 6.4, 0.22, 'KV append: 계산은 별도 DRAM 출력으로, tail DMA가 cache 슬롯 L=7에 기록', { size: 8.5, bold: true, color: C.navy });
    const Kc = D.matrix(s, 0.5, 4.15, 2.3, 1.3, 8, 9, { kind: 'faint', title: 'k_cache[l,h] [128, C=9] (열 = slot)', titleW: 2.6, titleSize: 7, regions: [{ r0: 0, r1: 8, c0: 0, c1: 7, kind: 'dram', label: '유효 0…6', size: 6.5 }, { r0: 0, r1: 8, c0: 7, c1: 8, kind: 'warn', label: '7', size: 6.5 }], below: '열 1개: 128행 × 2 B, stride 18 B → 2D DMA', belowW: 2.6, belowSize: 6.5 });
    const Kn = D.matrix(s, 3.05, 4.15, 0.22, 1.3, 8, 1, { kind: 'unit', title: 'K_new', titleW: 0.6, titleSize: 7, regions: [{ r0: 0, r1: 8, c0: 0, c1: 1, kind: 'unit' }] });
    D.arrow(s, Kn.x, Kn.cy, Kc.x + Kc.cw * 7.5, Kc.cy, { width: 1.5, color: C.warn });
    const Vc = D.matrix(s, 3.7, 4.0, 2.0, 1.55, 9, 8, { kind: 'faint', title: 'v_cache[l,h] [C=9, 128] (행 = slot)', titleW: 2.3, titleSize: 7, regions: [{ r0: 0, r1: 7, c0: 0, c1: 8, kind: 'dram', label: '유효 0…6', size: 6.5 }, { r0: 7, r1: 8, c0: 0, c1: 8, kind: 'warn', label: '7', size: 6.5 }], below: '행 1개: 연속 256 B → 1D DMA', belowW: 2.3, belowSize: 6.5 });
    const Vn = D.matrix(s, 5.85, 5.15, 1.05, 0.18, 1, 8, { kind: 'unit', title: 'V_new', titleW: 0.7, titleSize: 7, regions: [{ r0: 0, r1: 1, c0: 0, c1: 8, kind: 'unit' }] });
    D.arrow(s, Vn.x, Vn.cy, Vc.r, Vc.y + Vc.ch * 7.5, { width: 1.5, color: C.warn });
    // right: files & what changes
    const rx = 7.4, rw = 5.43;
    D.text(s, rx, 1.5, rw, 0.22, 'step마다 실제로 바뀌는 바이트 (DecodeSession · PersistentTransport)', { size: 9, bold: true, color: C.navy });
    D.text(s, rx, 1.76, rw, 0.2, 'global_memory.bin ≈ 6.43 GB', { size: 7.5, bold: true, color: C.dram });
    D.membar(s, rx, 1.97, rw, 0.4, 0, 100, [{ from: 0, to: 68, label: 'weight (변경 없음, rewrite 0 B)', kind: 'dram' }, { from: 68, to: 76, label: '입력 6,676 B', kind: 'warn' }, { from: 76, to: 92, label: 'KV cache (tail이 갱신)', kind: 'sram' }, { from: 92, to: 100, label: '출력', kind: 'unit' }], { size: 6.5 });
    D.text(s, rx, 2.45, rw, 0.2, 'program_memory.bin 7,324,697 words', { size: 7.5, bold: true, color: C.unit });
    D.membar(s, rx, 2.66, rw, 0.4, 0, 100, [{ from: 0, to: 88, label: 'decode base 7,310,801 words (변경 없음, 재link 없음)', kind: 'ir' }, { from: 88, to: 99, label: 'tail 13,896 (fixup)', kind: 'warn' }, { from: 99, to: 100, label: '', kind: 'red' }], { size: 6.5 });
    D.table(s, rx, 3.25, rw, [
      ['대상', 'step 사이'],
      ['decode 계산 명령 · DRAM weight', '재사용 (host rewrite 0 B)'],
      ['embedding · cos/sin · mask', 'host patch 6,676 B (plan.address 위치)'],
      ['KV append 명령 주소', 'position fixup (suffix patch 55,568 B)'],
      ['persistent DRAM cache', 'saved image → 다음 입력 (rename)'],
      ['C-model', '호출마다 전체 image read/write (기능 검증용, cycle 없음)'],
    ], { colW: [2.2, 3.38], size: 7.5, rowH: 0.27 });
    D.table(s, rx, 5.0, rw, [
      ['실행', 'token', 'HF 일치', 'logits cosine', '최대 절대차'],
      ['prefill', '358', '예', '0.9999977', '0.035'],
      ['decode @7', '2846', '예', '0.9999958', '0.023'],
      ['decode @8', '4560', '예', '0.9999930', '0.039'],
    ], { colW: [1.2, 0.8, 0.9, 1.4, 1.28], size: 7.5, rowH: 0.24, aligns: ['left', 'right', 'center', 'right', 'right'] });
    D.text(s, rx, 6.05, rw, 0.35, 'prefill 8.58M words / DMA 6.46 GB, decode 7.31M words. compiler 통계·host patch·simulator I/O는 서로 다른 측정 층.', { size: 7, color: C.muted });
    D.takeaway(s, 'Greedy argmax는 host에서. 마지막 예측 token은 아직 NPU에 넣지 않았으므로 생성 token 수와 decode 실행 횟수가 한 번 차이 납니다.', { y: 6.5 });
  });

  // ================================================================ 12 verification & limits
  D.add({ section: '검증 · 경계', title: '검증된 것과 아직 없는 것', subtitle: '작은 예제는 실제 C-model 실행에서 fused/unfused bit-exact, LLVM reference와 최대 절대차 0. 전체 Llama 3.2 3B는 HF와 같은 greedy token 3개, logits cosine ≥ 0.99999. 아래 수치는 모두 보존 기록이며 ISA words나 DMA를 실장치 latency로 바꾸어 말하지 않습니다.', source: 'docs/current/11_experiments.md (E1~E7), 12_operations_limits.md, evidence/current/*',
    notes: 'E1(fusion off/auto)과 E2(SRAM 32/44/64 KiB)는 이번 문서 작업에서 현재 소스로 실제 실행한 것이고, E3(직접 저장 85→77 words), E6(전체 Llama), E7(17개 회귀 suite)은 보존된 실행 기록, E4(실제 폭 MLP)와 E5(compile 84→22 s, binary 동일)는 정적 분석·단일 관측입니다. 지원 경계: 정적 shape, 지원된 정적 호출의 직선 전개, softmax 마지막 축 1..65504, RMSNorm FP16, full-attention은 rank-3 64배수, RoPE default/linear/llama3, FP16 정적 capacity cache, C-model runtime. 보장하지 않는 것: 실장치 latency, 동적 shape, 모든 TVM op/HF 모델, 자동 mixed precision, quantized KV, key online softmax, inter-kernel tile streaming. 양자화(W8A16/W8A8)는 kernel bit equality까지만이고 generation에는 미연결입니다.' },
  (s, D) => {
    s.addChart(D.pres.charts.BAR, [{ name: 'words', labels: ['fusion off (3 kernels)', 'fusion auto (1 kernel)'], values: [2812, 1577] }], { x: 0.5, y: 1.55, w: 3.0, h: 2.2, barDir: 'col', chartColors: ['1F8A8A'], showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 8, dataLabelFormatCode: '#,##0', showTitle: true, title: 'E1 ISA words (DMA 동일 41,100 B)', titleFontSize: 9, titleColor: C.navy, catAxisLabelFontSize: 7.5, valAxisLabelFontSize: 7, catAxisLabelColor: C.ink, valAxisLabelColor: C.muted, valGridLine: { color: 'E1E7EF', size: 0.5 }, catGridLine: { style: 'none' }, showLegend: false, valAxisLabelFormatCode: '#,##0' });
    s.addChart(D.pres.charts.BAR, [{ name: 'DMA', labels: ['32K off', '32K on', '44K off', '44K on', '64K on'], values: [81932, 81932, 81932, 49164, 49164] }], { x: 3.65, y: 1.55, w: 3.5, h: 2.2, barDir: 'col', chartColors: ['D98A1E'], showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 7, dataLabelFormatCode: '#,##0', showTitle: true, title: 'E2 SRAM 용량별 DMA bytes (두 matmul)', titleFontSize: 9, titleColor: C.navy, catAxisLabelFontSize: 7, valAxisLabelFontSize: 7, catAxisLabelColor: C.ink, valAxisLabelColor: C.muted, valGridLine: { color: 'E1E7EF', size: 0.5 }, catGridLine: { style: 'none' }, showLegend: false, valAxisLabelFormatCode: '#,##0' });
    D.table(s, 0.5, 3.95, 6.65, [
      ['실험', '결과', '검증 수준'],
      ['E3 직접 SRAM 출력·DRAM 압축', 'words 85 → 77, image 65,548 → 49,164 B, bit-exact', '보존 실행'],
      ['E4 실제 폭 MLP (M=64)', '5 kernels, 140,594 words, DMA 152 MB, SRAM peak 7.45 MB', '정적 compile'],
      ['E5 compile 속도', 'M=7: 84.2 → 22.1 s, trial 347 → 112, binary hash 동일', '단일 관측'],
      ['E6 전체 Llama 3B', '358 / 2846 / 4560 = " I\'m trying", cosine ≥ 0.99999', '보존 실행 (948 s, cold compile 아님)'],
      ['E7 회귀 gate', '17 suites exit 0, skip 없음', '보존 실행'],
    ], { colW: [2.0, 3.3, 1.35], size: 7.5, rowH: 0.33 });
    D.text(s, 7.4, 1.55, 5.43, 0.24, '현재 지원 경계', { size: 10, bold: true, color: C.navy });
    D.bullets(s, 7.4, 1.85, 5.43, 2.3, [
      ['Shape·control flow: ', '정적 shape, 지원된 정적 호출의 직선 전개 (dynamic control flow 거절)'],
      ['연산: ', 'softmax 마지막 축 1..65504, RMSNorm FP16, full-attention은 rank-3 FP16 64배수 (decode 제외), RoPE default/linear/llama3'],
      ['메모리: ', 'tensor 전체 보관만, 고정 호출 순서, 완전 미사용 pool만 제거'],
      ['Runtime: ', 'C-model adapter (image/file patch). RTL·실장치 runtime 검증 아님'],
      ['양자화: ', 'W8A16/W8A8 kernel bit equality까지. generation에는 미연결'],
    ], { size: 8, psa: 3 });
    D.text(s, 7.4, 4.3, 5.43, 0.24, '남은 검토 (착수 목록 아님)', { size: 10, bold: true, color: C.navy });
    D.bullets(s, 7.4, 4.6, 5.43, 1.8, [
      'Vector 256 lanes vs 긴 VLEN의 RTL 계약 · vector 비용 모델(lane 활용률·cycle)',
      'Attention key-online softmax · decode capacity 낭비(bucket, 최종 슬롯 직접 저장)',
      '부분 tile 보관·퇴거·호출 순서 변경 · 사용 중 pool 내부 압축',
      '성능 모델: RTL/장치 timing 확보 뒤 DMA→words→SRAM 순서와의 상관 확인',
    ], { size: 8, psa: 3 });
    D.takeaway(s, '지표(DMA/words/SRAM)로 확인되지 않는 개선은 채택하지 않습니다. 수치 정확성(HF 대비)과 프로그램 비용은 별개의 표이며 같은 source hash를 가리킵니다.', { y: 6.5 });
  });
  // ================================================================ 13 search refinement (09-29) + E10 status
  D.add({ section: '탐색 개선 · 전체 모델 실험', title: '09-29 탐색 개선: 중복 후보 정규화, 증명된 용량 초과 제외, 적응형 words 예산 — MetaSchedule은 선택형', subtitle: '같은 tile 순회·복사 주기의 표기는 한 후보로 합치고(512-token down 22,236 → 5,034), 두 입력 cache 합이 8 MiB를 넘는 1,596개를 증명으로 제외한 뒤 3,438개를 검사 목록에 둡니다. 기본 탐색은 space, ReplayFunc 추첨과 trace DB는 옵션입니다.', source: 'npu_schedule_space.py, npu_schedule.py, npu_meta_schedule.py; evidence/current/llama_schedule_refinement.json (E9), llama_prefill256_full_search_progress.json (E10)',
    notes: '① loop_spec은 같은 축을 panel·tile 두 loop로 나누므로 n_panel=64/128/256은 모두 N tile을 같은 순서로 도는 표기입니다. 크기 1 loop를 빼고 같은 축의 이웃 loop를 곱한 signature로 합치되, cache 복사 위치(prefix)가 다르면 별도 후보로 둡니다. toy [128,256,128]은 loop 위치 조합 1,728 → 후보 73입니다. ② 두 operand cache는 matrix 명령 시점에 동시에 살아 있으므로 그 합이 하한입니다. 정렬 rank-2 FP16 직접 입력 2개(64배수, 함수 parameter)일 때만 (SRAM − scratch prefix)를 넘는 후보를 제외하고, padding·computed producer는 순서만 뒤로 보냅니다. ③ words 한도는 max(100,000, 기준 words × legal 목표 8)이며 legal 2개 전에는 words로 끝내지 않고, 시도 64회(실패 포함)는 별도입니다. ④ space는 추정 DMA 순, meta_schedule은 TVM ReplayFunc가 sample_categorical로 뽑은 순서로 같은 pool을 검사하며 둘 다 실제 SRAM 배치·명령 생성으로 (DMA, words, SRAM)을 얻습니다. JSONDatabase에는 run_secs=None인 선택된 trace만 저장하고 reuse는 재생·재검증, tune은 incumbent로 두고 개선될 때만 기록합니다. 실제 Llama 4개 커널에서 두 방식의 최종 binary는 같았고(E8), 512-token down은 개선 후 두 방식 모두 4회 검사·합법 4개로 이전과 같은 명령이었습니다(E9). ⑤ E10은 28-layer·256-token prefill 하나를 graph 시간 제한 없이 컴파일하다 메모리 증가로 중단한 실험이며, GQA·producer 변환이 같은 IR이라 source 3개 × generic 3 = 9 구성입니다. 비용·fusion 채택·출력 일치는 완료 전까지 주장하지 않습니다.' },
  (s, D) => {
    const progress = evidence('llama_prefill256_full_search_progress.json');
    if (progress.graph_inventory.graph_configurations !== 9 || progress.request.prefill_tokens !== 256) throw new Error('Update short slide 13 for the new E10 configuration');
    const MB = 1048576;
    // ① normalization
    D.text(s, 0.5, 1.55, 3.95, 0.24, '① 표기 중복 → signature 하나 (toy [128,256]×K128)', { size: 9, bold: true, color: C.navy });
    const spell = [['n_panel=64', ['m_p 2', 'n_p 4', 'n_t 1', 'k 2']], ['n_panel=128', ['m_p 2', 'n_p 2', 'n_t 2', 'k 2']], ['n_panel=256', ['m_p 2', 'n_p 1', 'n_t 4', 'k 2']]];
    spell.forEach((sp, i) => D.lanes(s, 1.4, 1.85 + i * 0.33, 2.55, 0.27, sp[1].map(t => ({ label: t, kind: t.endsWith(' 1') ? 'faint' : (t.startsWith('n') ? 'sram' : (t.startsWith('k') ? 'unit' : 'ir')), size: 7, color: t.endsWith(' 1') ? C.muted : C.ink })), { title: sp[0], titleW: 0.85, titleSize: 7.5 }));
    D.text(s, 4.0, 1.85, 0.5, 0.95, '→ 같은\n순회\n1 2 3 4', { size: 6.5, color: C.muted, valign: 'middle' });
    D.chip(s, 1.4, 2.9, 'signature (m2, n4, k2)', { kind: 'navy', color: C.white, w: 1.75, size: 7.5, h: 0.24 });
    D.text(s, 0.5, 3.2, 3.95, 0.5, 'loop 위치 조합 1,728 → 후보 73 (현재 코드 계산). 복사 위치(prefix)가 다르면, 예: b_at=n_panel의 tile마다 vs 2 tile마다, 별도 후보로 유지.', { size: 7.5, color: C.ink });
    // ② capacity proof
    D.text(s, 4.7, 1.55, 3.9, 0.24, '② 두 입력 cache 합으로 증명된 초과만 제외', { size: 9, bold: true, color: C.navy });
    const hi = 9.6 * MB;
    D.membar(s, 4.7, 1.85, 3.35, 0.3, 0, hi, [{ from: 0, to: 98304, label: '', kind: 'host' }, { from: 98304, to: 98304 + 8 * MB, label: 'H cache 8 MiB (512행)', kind: 'red' }, { from: 98304 + 8 * MB, to: 98304 + 9 * MB, label: 'W', kind: 'dram' }], { size: 7 });
    D.text(s, 8.1, 1.85, 0.5, 0.3, '제외', { size: 7.5, bold: true, color: C.red, valign: 'middle' });
    D.membar(s, 4.7, 2.3, 3.35, 0.3, 0, hi, [{ from: 0, to: 98304, label: '', kind: 'host' }, { from: 98304, to: 98304 + 4 * MB, label: 'H 4 MiB (256행)', kind: 'sram' }, { from: 98304 + 4 * MB, to: 98304 + 5 * MB, label: 'W', kind: 'dram' }], { size: 7 });
    D.text(s, 8.1, 2.3, 0.5, 0.3, '검사', { size: 7.5, bold: true, color: C.good, valign: 'middle' });
    const lim = 4.7 + 3.35 * (98304 + 8 * MB) / hi;
    D.line(s, lim, 1.78, lim, 2.68, { arrow: false, color: C.red, width: 1.5, dash: 'dash' });
    D.text(s, lim - 1.4, 2.65, 1.6, 0.18, '가용 = 8 MiB − scratch prefix', { size: 6.5, color: C.red, align: 'center' });
    D.text(s, 4.7, 2.9, 3.9, 0.8, '조건: 정렬 rank-2 FP16 직접 입력 2개, 64배수(ordinary). padding·computed producer는 순서만 뒤로. 512-token down: 5,034 중 1,596 제외 → 3,438 검사 목록 (E9). gate M=64: 203 → 162 (계산).', { size: 7.5, color: C.ink });
    // ③ budgets / stop reasons
    D.text(s, 8.9, 1.55, 3.93, 0.24, '③ 예산과 종료 사유 (stop_reason)', { size: 9, bold: true, color: C.navy });
    D.lanes(s, 8.9, 1.85, 3.93, 0.42, [{ label: 'legal_budget\nlegal ≥ 8', kind: 'good', size: 6.5 }, { label: 'word_budget\nlegal ≥ 2 ∧ Σwords ≥\nmax(100k, 기준×8)', kind: 'warn', size: 6 }, { label: 'attempt_budget\n검사 ≥ 64\n(실패 포함)', kind: 'red', size: 6.5 }, { label: 'exhausted /\ndatabase_hit', kind: 'faint', size: 6.5 }], { size: 6.5 });
    D.text(s, 8.9, 2.35, 3.93, 1.3, 'incumbent(legacy A/B/B_stream/None, legal 2·시도 64)를 먼저 잡고 확장 후보를 검사. 기준 words가 큰 커널일수록 words 한도가 같이 커져 큰 커널에서 너무 일찍 끝나지 않음(E8 512-down: 16회 모두 용량 초과 → 개선 후 4회 검사 = 합법 4개). E10 실행기는 legal 16 · 시도 128 · words 10⁹.', { size: 7.5, color: C.ink });
    // ④ space vs ReplayFunc + DB
    D.text(s, 0.5, 3.85, 8.1, 0.24, '④ 검사 순서: space(추정 DMA 순) vs ReplayFunc(sample_categorical 추첨) · 선택된 trace는 JSONDatabase에 저장·재생 — 평가는 공통', { size: 9, bold: true, color: C.navy });
    const px = 1.4, pw = 3.8, N = 12, cw = pw / N;
    const cells = []; for (let i = 0; i < N; i++) cells.push({ label: String(i), kind: 'faint', size: 7, mono: true });
    D.lanes(s, px, 4.55, pw, 0.3, cells, { title: '후보 pool', titleW: 0.85, titleSize: 7.5 });
    D.text(s, 0.5, 4.15, 0.85, 0.3, 'space', { size: 7.5, bold: true, color: C.sram, align: 'right', valign: 'middle' });
    [0, 1, 2, 3].forEach((i, k) => { const cx = px + (i + 0.5) * cw; D.arrow(s, cx, 4.28, cx, 4.53, { width: 1.2, color: C.sram }); D.text(s, cx - 0.2, 4.1, 0.4, 0.18, ['①', '②', '③', '④'][k], { size: 7, color: C.sram, bold: true, align: 'center' }); });
    D.text(s, px + 4.5 * cw, 4.12, 2.4, 0.3, '… 정렬 순서대로', { size: 7, color: C.sram, valign: 'middle' });
    D.text(s, 0.5, 4.9, 0.85, 0.3, 'ReplayFunc', { size: 7.5, bold: true, color: C.unit, align: 'right', valign: 'middle' });
    [[7, '①'], [2, '②'], [7, '③중복'], [10, '④']].forEach(([i, t]) => { const dup = t.includes('중복'); const cx = px + (i + 0.5) * cw + (dup ? 0.1 : 0); D.arrow(s, cx, dup ? 5.36 : 5.2, cx, 4.87, { width: 1.2, color: dup ? C.red : C.unit }); D.text(s, cx - 0.3, dup ? 5.36 : 5.2, 0.6, 0.18, t, { size: 6.5, color: dup ? C.red : C.unit, bold: true, align: 'center' }); });
    D.text(s, 0.5, 5.58, 5.0, 0.55, '중복 추첨은 schedule을 다시 만들지 않고 시도만 소비. 두 순서 모두 schedule → lowering → SRAM 배치 → 명령 생성 → (DMA, words, SRAM). Llama 4개 커널: 최종 binary 동일(E8) · 512-down 개선 후 4회/4 legal, 이전과 같은 명령(E9).', { size: 7.5, color: C.ink });
    D.code(s, 5.55, 4.15, 3.05, 1.9, ['# JSONDatabase (선택형, 기본 off)', 'Workload: PrimFunc + 실행 조건', '  (shape·dtype·scope·SRAM·scratch', '   ·수치·버전) − 탐색 설정', 'TuningRecord: trace, run_secs=None', 'reuse: 재생·재검증 → 탐색 0회', 'tune: incumbent → 개선 시만 기록', '# 작은 예제: 새 후보 TIR 8→0, bit-exact'], { size: 6.8 });
    // ⑤ E10 status
    D.text(s, 8.9, 3.85, 3.93, 0.24, '⑤ E10 중단: 28-layer · 256-token prefill', { size: 9, bold: true, color: C.navy });
    D.table(s, 8.9, 4.15, 3.93, [
      ['조건', '값'],
      ['graph 시간', 'None (120 s 제한 해제)'],
      ['graph 구성', '27 → source 3 × generic 3 = 9'],
      ['matmul 예산', 'legal 16 · 시도 128 · words 10⁹'],
      ['메모리', 'joint on · beam 4 · DB 미사용'],
      ['상태', '메모리 증가로 중단 · 전체 재검증 미실행'],
    ], { colW: [1.15, 2.78], size: 7.3, rowH: 0.3 });
    D.text(s, 8.9, 6.0, 3.93, 0.4, '비용·fusion 채택·출력 일치는 완료 전까지 주장하지 않음', { size: 7, color: C.muted });
    D.takeaway(s, '개선된 것은 탐색의 낭비와 기록 재사용입니다. 512-token down의 최종 명령은 개선 전과 같았고(E9), 전체 모델 결과(E10)는 완료 전까지 주장하지 않습니다.', { y: 6.5 });
  });

};
