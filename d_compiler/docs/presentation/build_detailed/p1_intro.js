'use strict';
const { C, F } = require('./lib');

module.exports = (D) => {
  // ---------------------------------------------------------------- 1 cover
  D.add({ dark: true, bare: true, notes:
    '현재 메모리 계층(DRAM/SRAM/연산장치)을 사용하는 v09 FP16 backend를 설명합니다. 모델의 계산이 어떻게 TVM Relax → TIR → scheduled TIR → ISA로 내려가고, runtime이 그 프로그램으로 실제 token을 생성하는지 Llama 3.2 3B와 작은 예제 두 개로 끝까지 따라갑니다. 과거 측정과 09-29 탐색 개선 결과를 구분합니다. 별도로 실행 중인 E10은 28-layer·256-token prefill이며 완료 전 비용이나 출력 통과를 주장하지 않습니다.' },
  (s, D) => {
    D.text(s, 0.8, 0.7, 11, 0.3, 'NPU COMPILER  ·  V09 BACKEND  ·  TVM 0.19 RELAX / TIR', { size: 11, color: C.sram, bold: true, charSpacing: 2 });
    D.text(s, 0.8, 1.35, 11.5, 1.6, 'TVM 기반 LLM NPU Compiler', { size: 44, color: C.white, bold: true });
    D.text(s, 0.8, 2.45, 11.5, 0.9, '모델의 계산이 NPU 명령과 다음 token이 되기까지:\n전체 compile pipeline과 runtime을 예제로 따라가기', { size: 20, color: C.darkText });
    const chips = [['계산', 'Relax · semantic TIR · 수치 계약', 'ir'], ['배치', 'Schedule · SRAM/DRAM 계획', 'sram'], ['실행', 'ISA · linker · C-model runtime', 'unit']];
    chips.forEach((c, i) => {
      const x = 0.8 + i * 3.9;
      D.rect(s, x, 3.9, 3.6, 1.15, { fill: '18293F', line: '2B3E58', radius: 0.08 });
      D.text(s, x + 0.25, 4.02, 3.1, 0.4, c[0], { size: 18, bold: true, color: C.white });
      D.text(s, x + 0.25, 4.45, 3.1, 0.5, c[1], { size: 12, color: C.darkMuted });
    });
    D.text(s, 0.8, 5.55, 11.5, 0.9, '예제: Llama 3.2 3B (28 layers, hidden 3072, MLP 8192, Q 24 / KV 8 heads, head 128, vocab 128256)\n작은 예제: SiLU(X[64,128]·Wᵀ+bias) 와 두 matmul X·W1·W2 (SRAM 44 KiB) · 설정: FP16 저장 / FP32 누산 · SRAM 8 MiB · tile 64×64', { size: 12, color: C.darkMuted });
    D.text(s, 0.8, 6.75, 11.5, 0.3, '2026-09-29 탐색 개선 반영 · 근거: d_compiler/docs (13장, examples, evidence)', { size: 10, color: C.darkMuted });
  });

  // ---------------------------------------------------------------- 2 agenda
  D.add({ section: '개요', title: '발표 구성: 계산 → 배치 → 실행, 그리고 검증', subtitle: '각 파트는 "입력 → 해결할 문제 → 실제 동작 → 결과" 순서로, 같은 예제를 단계마다 다시 봅니다.', source: 'docs/README.md',
    notes: '발표는 10개 파트입니다. 앞부분(1~4)은 계산의 의미를 정하는 단계, 중간(5~8)은 배치와 명령 생성, 뒷부분(9~10)은 runtime과 실험·한계입니다. 세 예제를 반복해서 사용하므로 청중은 한 예제가 단계마다 어떻게 변하는지 추적할 수 있습니다.' },
  (s, D) => {
    const parts = [
      ['1', '전체 구조', '장치 계약, compile/load/run, 세 수준의 IR, 전체 pipeline'],
      ['2', '모델 → Relax', 'relax.frontend.nn Llama, 입력 명세, 작은 예제의 authored IR'],
      ['3', 'Graph 정리와 host', 'CanonicalizeAttention, LiftTransformParams, host transform 실행'],
      ['4', 'LegalizeOps와 수치', 'call_tir + PrimFunc, FP16 저장/FP32 누산, softmax·RMSNorm 정의'],
      ['5', 'Fusion 후보 선택', 'op_pattern, FuseOps/FuseTIR, 영역 probe와 전체 module 비교'],
      ['6', 'Matmul scheduling', '64×64 누산 체인, panel 탐색 공간, Schedule API, tensorize'],
      ['7', 'Vector·정규화·attention', 'SiLU 전개, VLEN, softmax, RMSNorm, RoPE, query-blocked attention'],
      ['8', '메모리 계획', 'DRAM pool, 수명, USMP, 커널 사이 SRAM 보관, 직접 저장, 주소 압축'],
      ['9', 'ISA·linker·runtime', '명령 인코딩 해독, linker, bundle, prefill/decode, KV append'],
      ['10', '실험·한계·정리', 'E1~E9 측정·E10 진행, 지원 경계, 남은 검토 목록, 부록(main, 양자화, API)'],
    ];
    parts.forEach((p, i) => {
      const col = i < 5 ? 0 : 1;
      const x = 0.5 + col * 6.3, y = 1.55 + (i % 5) * 0.86;
      D.rect(s, x, y, 6.05, 0.74, { fill: C.white, line: C.line, radius: 0.06 });
      D.text(s, x + 0.12, y + 0.12, 0.5, 0.5, p[0], { size: 20, bold: true, color: C.sram, align: 'center', valign: 'middle' });
      D.text(s, x + 0.7, y + 0.08, 5.2, 0.3, p[1], { size: 13, bold: true, color: C.navy });
      D.text(s, x + 0.7, y + 0.38, 5.25, 0.32, p[2], { size: 10, color: C.muted });
    });
    D.takeaway(s, '세 예제를 끝까지 사용: ① SiLU(X·Wᵀ+bias) [64,128]→[64,64]  ② X·W1·W2 두 matmul (SRAM 44 KiB)  ③ Llama 3.2 3B, prompt 7 tokens, capacity 9', { y: 6.0 });
  });

  // ---------------------------------------------------------------- 3 device contract
  D.add({ section: '1 · 전체 구조', title: '대상 장치: 세 층의 메모리와 고정된 수치 계약', subtitle: '데이터가 실제로 어떻게 움직이는가: DRAM의 weight를 panel 단위로 SRAM에 올리고(GLOAD), SRAM의 64×64 tile을 연산장치에 넣고(LOAD), 결과를 SRAM에 내린 뒤(SAVE) DRAM으로 보냅니다(GSTORE).', source: 'docs/current/01_overview.md, reference/execution_contract.md, npu_compiler/device.py',
    notes: 'DRAM(global)은 DMA만 접근하며 32-bit cell 단위 주소를 씁니다. SRAM은 8 MiB, nibble 주소입니다. 연산장치는 64×64 matrix unit(FP32 accumulator 1개)과 256-lane vector unit입니다. GLOAD/GSTORE는 DRAM↔SRAM, LOAD/SAVE는 SRAM↔연산장치 이동이라 서로 다른 경계입니다. 그림의 SRAM 구성(literal·scratch, 입력 panel, 출력 tile, 커널 사이 보관값)은 8장의 실제 배치 순서입니다. 논리 VLEN 65535와 물리 256 lanes의 관계는 RTL 미확인 항목으로 문서가 명시합니다.' },
  (s, D) => {
    const x0 = 0.5, fw = 7.1;
    // DRAM strip
    D.text(s, x0, 1.5, fw, 0.24, 'DRAM / global image  (주소 단위: 32-bit cell = 4 B, DMA만 접근)', { size: 9.5, bold: true, color: C.dram });
    D.membar(s, x0, 1.76, fw, 0.5, 0, 100, [
      { from: 0, to: 52, label: 'weight (Wᵀ, 층마다 192 MiB)', sub: 'panel 단위로 읽음', kind: 'dram' },
      { from: 52, to: 66, label: '입력', sub: 'embeds·cos/sin·mask', kind: 'host' },
      { from: 66, to: 86, label: 'persistent KV cache', sub: '[28,8,128,C]', kind: 'warn' },
      { from: 86, to: 100, label: '출력', sub: 'logits·K/V_new', kind: 'unit' },
    ], { size: 8.5 });
    // SRAM strip
    D.text(s, x0, 3.0, fw, 0.24, 'SRAM 8 MiB  (주소 단위: nibble, fp16 = 4 nibbles, 8-nibble 정렬)', { size: 9.5, bold: true, color: C.sram });
    D.membar(s, x0, 3.26, fw, 0.5, 0, 100, [
      { from: 0, to: 7, label: 'pool', sub: 'literal·scratch', kind: 'host' },
      { from: 7, to: 30, label: 'X panel', sub: '입력 activation', kind: 'sram' },
      { from: 30, to: 53, label: 'W panel j', sub: '이번에 쓰는 weight 조각', kind: 'dram' },
      { from: 53, to: 72, label: '출력 tile · 임시', sub: 'USMP가 offset 배치', kind: 'ir' },
      { from: 72, to: 100, label: '커널 사이 보관값', sub: 'top-down 예약 (8장)', kind: 'warn' },
    ], { size: 8.5 });
    // units
    D.matrix(s, x0 + 0.15, 4.62, 1.15, 1.15, 8, 8, { kind: 'unit', tile: 8, title: 'Matrix unit 64×64', below: 'FP32 accumulator ×1 · M_MUL MAC bit로 K 누산', belowW: 3.2 });
    D.lanes(s, x0 + 2.4, 4.95, 3.6, 0.38, Array.from({ length: 16 }, (_, i) => ({ label: i === 15 ? '255' : (i === 0 ? '0' : (i === 8 ? '…' : '')), kind: 'unit' })), {});
    D.text(s, x0 + 2.4, 4.7, 3.6, 0.22, 'Vector unit 256 lanes (물리)', { size: 8.5, bold: true, color: C.navy });
    D.text(s, x0 + 2.4, 5.36, 4.6, 0.42, '논리 VLEN 1..65535 → 장치 내부에서 256씩 반복한다는 전제 (RTL 미확인). FP16 입력, reduction 내부 FP32.', { size: 7.5, color: C.muted });
    // arrows between tiers
    const gx = x0 + fw * 0.26;
    D.arrow(s, gx, 2.26, x0 + fw * 0.41, 3.26, { color: C.dram, width: 2 });
    D.text(s, x0 + fw * 0.42, 2.45, 2.6, 0.4, 'GLOAD: DRAM → SRAM\n4-word DMA packet, cell 단위', { size: 8, color: C.dram, bold: true });
    D.arrow(s, x0 + fw * 0.62, 3.26, x0 + fw * 0.93, 2.26, { color: C.dram, width: 2 });
    D.text(s, x0 + fw * 0.78, 2.75, 1.8, 0.3, 'GSTORE: SRAM → DRAM', { size: 8, color: C.dram, bold: true });
    D.arrow(s, x0 + fw * 0.18, 3.76, x0 + 0.75, 4.62, { color: C.sram, width: 2 });
    D.text(s, x0 + 1.4, 3.98, 2.4, 0.4, 'LOAD: SRAM 64×64 tile → unit\n(descriptor로 주소·stride 지정)', { size: 8, color: C.sram, bold: true });
    D.arrow(s, x0 + 1.3, 4.95, x0 + fw * 0.62, 3.76, { color: C.sram, width: 2 });
    D.text(s, x0 + 4.4, 4.02, 2.4, 0.4, 'SAVE: FP16 결과 → SRAM 출력 tile', { size: 8, color: C.sram, bold: true });
    D.text(s, x0, 6.12, fw, 0.34, '연산장치 내부의 FP32 accumulator는 SRAM buffer가 아닙니다. TIR의 FP32 buffer가 그대로 SRAM에 materialize된다고 읽으면 안 됩니다.', { size: 8.5, color: C.muted, italic: true });
    // contract table (right)
    D.table(s, 7.85, 1.5, 4.98, [
      ['항목', '현재 계약'],
      ['tile / lanes', '64×64 matrix, 256 lanes (VLEN 1..65535)'],
      ['SRAM', '8 MiB = 2²⁴ nibbles'],
      ['DRAM', 'plan unit 2 B, DMA 주소 32-bit cell'],
      ['수치', 'FP16 저장, K=64 FP32 부분합 누산'],
      ['누산기', '1개, partial spill 금지, K tile 64'],
      ['Activation', 'ISA에 있으나 자동 matmul 경로는 OFF'],
      ['DMA', 'GLOAD 0xA0 / GSTORE 0xA8, 4 words'],
      ['종료', 'HALT 0xFF, SNAPSHOT 0xF0'],
      ['실행기', '_poc/mysim_v09.cpp (cycle 없음)'],
    ], { colW: [1.25, 3.73], size: 9, rowH: 0.36 });
    D.takeaway(s, '성능 지표는 DMA bytes → ISA words → SRAM peak 순서의 정적 비교입니다. C-model 통과가 RTL 검증을 대신하지 않습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 4 llama shapes
  D.add({ section: '1 · 전체 구조', title: 'Llama 3.2 3B: 발표 내내 사용할 shape와 크기', subtitle: 'Decode는 S=1, prefill은 prompt 길이 S. Weight는 checkpoint의 [N,K] 배치이며 계산에는 [K,N]이 필요합니다.', source: 'docs/examples/llama_ir/summary.json, docs/current/01_overview.md',
    notes: 'Llama 한 layer의 weight는 FP16으로 192 MiB, 28 layers면 5.25 GiB, LM head가 751.5 MiB입니다. 전체 검증 기록의 DRAM image 약 6.43 GB와 일치하는 규모입니다. 8 MiB SRAM에는 어떤 projection weight도 통째로 들어가지 않으므로 panel 단위 streaming이 필수라는 점을 여기서 미리 보여줍니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 5.6, [
      ['설정', '값'],
      ['hidden / intermediate', '3072 / 8192'],
      ['layers', '28'],
      ['Q heads / KV heads / head_dim', '24 / 8 / 128 (GQA: Q 3개가 KV 1개 공유)'],
      ['vocabulary', '128256'],
      ['RoPE', 'llama3, θ=500000, factor 32, low 1, high 4, orig 8192'],
      ['RMSNorm ε', '1e-5'],
      ['예제 shape', 'prefill S=7 tokens, decode capacity C=9'],
    ], { colW: [2.2, 3.4], size: 10, rowH: 0.36 });
    D.table(s, 6.4, 1.55, 6.4, [
      ['Weight (FP16)', 'shape [N,K]', '크기'],
      ['q_proj / o_proj', '3072 × 3072', '18 MiB 씩'],
      ['k_proj / v_proj', '1024 × 3072', '6 MiB 씩'],
      ['gate / up', '8192 × 3072', '48 MiB 씩'],
      ['down', '3072 × 8192', '48 MiB'],
      ['layer 합계', '9 tensors', '192 MiB × 28 = 5.25 GiB'],
      ['lm_head', '128256 × 3072', '751.5 MiB'],
      ['activation X (S=64)', '64 × 3072', '384 KiB'],
    ], { colW: [1.9, 1.9, 2.6], size: 10, rowH: 0.36 });
    // per-layer computation table
    D.text(s, 0.5, 4.75, 12, 0.3, 'Llama 한 layer에서 계산하는 것 (입력 X[S,3072])', { size: 12.5, bold: true, color: C.navy });
    D.table(s, 0.5, 5.08, 12.33, [
      ['RMSNorm', 'Q proj (24 heads)', 'K / V proj (8 heads)', 'RoPE', 'Attention', 'O proj + residual', 'RMSNorm', 'gate / up', 'SiLU(gate) × up', 'down + residual'],
      ['[S,3072]', '[S,3072]', '[S,1024] 각각', 'Q, K 회전', 'QKᵀ → scale → mask → softmax → PV', '[S,3072]', '[S,3072]', '[S,8192] 각각', '[S,8192]', '[S,3072]'],
    ], { size: 8.5, rowH: 0.34, aligns: Array(10).fill('center') });
    D.takeaway(s, '어떤 projection weight도 8 MiB SRAM에 통째로 들어가지 않습니다 → panel 단위 streaming과 재사용 순서가 compiler의 핵심 결정입니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 5 compile/load/run
  D.add({ section: '1 · 전체 구조', title: 'Compile, load(prepare), run은 서로 다른 시점의 작업', subtitle: 'Compile은 shape/dtype만 보고 프로그램을 확정하고, load는 weight를 변환·배치하며, run은 입력만 갱신해 같은 명령을 재실행합니다.', source: 'docs/current/01_overview.md, npu_compiler/npu_target.py, npu_generate.py',
    notes: 'npu_target.build는 relax.build와 달리 NpuExecutable(words + plan + lowered module)을 돌려줍니다. 장치에 runtime allocator가 없어 모든 주소가 compile 시점에 정해지기 때문입니다. Weight가 함수 parameter로 남아 있으면 값만 바꿔 같은 프로그램을 재사용할 수 있지만, constant folding된 weight는 그렇지 않습니다. 정적 shape가 달라지면 다른 프로그램이 필요합니다.' },
  (s, D) => {
    const cols = [
      { t: 'Compile', k: 'ir', items: ['입력: Relax IRModule + 정적 shape/dtype', 'graph 후보 선택 · schedule · SRAM/DRAM 주소 · 명령 생성', 'Llama: prompt 길이 S의 prefill_cache 1개 + capacity C의 decode 1개', '출력: NpuExecutable = words + plan + lowered module + 입력 signature'] },
      { t: 'Load / prepare', k: 'dram', items: ['checkpoint 배열 읽기 (embedding table은 host에 유지)', 'host transform 실행: W transpose, (양자화 시) scale/quantize', '변환된 signature 순서대로 DRAM image에 배치 (npu_link.build_image)', '결과를 params-cache에 저장 (IR+weight 내용 hash 키)'] },
      { t: 'Run', k: 'unit', items: ['host: embedding 행, cos/sin, mask 준비 → image의 정해진 주소에 patch', 'C-model: program_memory.bin + global_memory.bin 실행', 'plan.outputs의 shape/dtype/offset으로 logits, K/V 읽기', 'decode: argmax → 다음 token, KV append DMA tail'] },
    ];
    cols.forEach((c, i) => {
      const x = 0.5 + i * 4.2;
      D.box(s, x, 1.6, 3.95, 0.6, c.t, { kind: c.k, size: 16 });
      D.rect(s, x, 2.25, 3.95, 2.75, { fill: C.white, line: C.line });
      D.bullets(s, x + 0.15, 2.38, 3.65, 2.55, c.items, { size: 10.5, psa: 5 });
      if (i < 2) D.arrow(s, x + 3.98, 1.9, x + 4.17, 1.9, { width: 2 });
    });
    D.code(s, 0.5, 5.2, 7.4, 1.2, [
      'from npu_compiler import npu_target',
      'with compilation(progress=print):',
      '    exe = npu_target.build(module, func_name="prefill", fusion="auto")',
      'output, counters = exe.run(values)        # values: 최종 signature 이름 → 배열',
      'exe.export_bundle("/tmp/bundle", values)  # program_memory.bin, program.json, IR',
    ], { size: 9.5 });
    D.callout(s, 8.1, 5.2, 4.73, 1.2, [
      { text: '재사용 조건', bold: true, breakLine: true },
      { text: 'weight가 parameter로 남아 있고 shape·ABI가 같으면 값만 바꿔 재사용. constant folding한 weight, 다른 S/C는 다른 프로그램.', size: 10.5 },
    ], { kind: 'warn', size: 11 });
    D.takeaway(s, '"모델을 compile한다"에는 graph 선택 · schedule · 주소 배치 · 명령 생성이 모두 들어 있고, 모델 실행은 그 뒤의 별도 단계입니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 6 three levels
  D.add({ section: '1 · 전체 구조', title: '같은 계산을 네 수준에서 읽는다: Relax → TIR → scheduled TIR → ISA', subtitle: '작은 예제 P = X[64,128] @ B[128,64]의 matmul이 각 수준에서 어떻게 보이는지. 아래 코드는 모두 실제 dump에서 발췌했습니다.', source: 'docs/examples/pipeline/01~07, two_matmuls_recorded/program.json',
    notes: 'Relax에는 호출 관계(무엇을 계산하는가)만 있습니다. LegalizeOps 뒤 TIR에는 어떤 원소를 읽어 더하는지, FP32로 누산해 FP16으로 cast하는 수치 의미가 있습니다. Schedule은 같은 계산을 64×64 조각과 SRAM 복사 위치로 재배치하고, codegen은 descriptor 설정·LOAD·M_MUL·SAVE·DMA 명령으로 펼칩니다. LegalizeOps 뒤에도 Relax 함수가 남아 호출 순서를 유지한다는 점이 중요합니다.' },
  (s, D) => {
    const cols = [
      { t: 'Relax', k: 'ir', q: '무엇을 계산하는가', code: ['lv1 = R.matmul(x, lv)', 'lv2 = R.add(lv1, bias)', 'gv  = R.nn.silu(lv2)', '', '# 호출 관계, shape/dtype', '# TensorStructInfo'] },
      { t: 'Semantic TIR', k: 'ir', q: '어떤 원소를 읽어 더하는가', code: ['for i0,i1,k in T.grid(64,64,128):', '  with T.block("matmul"):', '    T.init(): C32[i,j] = 0.0', '    C32[i,j] += f32(x[i,k])', '              * f32(lv[k,j])', 'compute[i,j] = f16(C32[i,j])'] },
      { t: 'Scheduled TIR', k: 'sram', q: '조각 · 위치 · 순서', code: ['x_global_sram = alloc((64,128),', '   "float16", scope="global.sram")', 'for k_0 in range(2):   # K tile', '  npu_gemm_acc(C, A[:,k_0*64:], B)', '# cache_read → compute_at →', '# tensorize (64×64 intrinsic)'] },
      { t: 'ISA words', k: 'unit', q: '실제 주소와 장치 명령', code: ['000c18a0 00000000  # GLOAD X', '00000800 00010800  #  → SRAM 1548', '404c1880 ...       # ADDR SRC2', 'c0000090           # LOAD matrix', '80000042           # M_MUL MAC=0', '88000042           # M_MUL MAC=1'] },
    ];
    cols.forEach((c, i) => {
      const x = 0.5 + i * 3.13;
      D.box(s, x, 1.6, 2.95, 0.62, c.t, { kind: c.k, size: 14, detail: c.q, dsize: 9.5 });
      D.code(s, x, 2.35, 2.95, 1.85, c.code, { size: 8.5 });
      if (i < 3) D.arrow(s, x + 2.97, 1.91, x + 3.11, 1.91, { width: 2 });
    });
    const rows = [
      ['수준', '정하는 것', '아직 정하지 않는 것', '만드는 단계'],
      ['Relax', '연산 호출 순서, tensor shape/dtype, parameter 경계(num_input)', '원소 계산 방법, 저장 위치', 'nn_models export, graph prefix'],
      ['Semantic TIR', '원소 수식, reduction 축, FP32→FP16 cast 위치', 'tile 크기, SRAM 주소, DMA 순서', 'LegalizeOps(custom map), FuseTIR'],
      ['Scheduled TIR', 'loop 분할·순서, cache buffer와 복사 위치, intrinsic 치환', '실제 nibble/cell 주소, 명령 word', 'npu_schedule + npu_intrin + USMP'],
      ['ISA', 'descriptor 상태, LOAD/SAVE/M_MUL/vector op, DMA packet 주소', '실장치 cycle', 'tir_codegen_v09 / v09_walker.cc + linker'],
    ];
    D.table(s, 0.5, 4.4, 12.33, rows, { colW: [1.5, 4.3, 3.1, 3.43], size: 9.5, rowH: 0.36 });
    D.takeaway(s, 'LegalizeOps 뒤에도 Relax 함수는 남아 호출 순서를 유지하고, 각 호출의 원소 계산이 TIR PrimFunc로 정의됩니다. 하나의 IRModule에 두 종류의 함수가 공존합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 7 full pipeline
  D.add({ section: '1 · 전체 구조', title: '전체 pipeline: 한 번 지나가는 직선이 아니라 비용으로 되돌아오는 탐색', subtitle: '실제 호출 구조. 후보 graph마다 schedule·메모리·codegen까지 내려가 (DMA bytes, words, SRAM peak)를 얻고 그 비용으로 선택합니다.', source: 'npu_compiler/npu_target.py, npu_build.py, npu_link.py, tvm_pipeline.py',
    notes: 'npu_target.build → npu_build.lower_and_select가 진입점입니다. graph_prefix는 한 번만 적용되고, 그 뒤 attention(off/score/full) × GQA·producer 변환 × generic fusion(off/vectors/matmul) 조합으로 후보 source를 만듭니다. 각 후보는 npu_fusion.select로 영역별 검증을 거친 뒤 npu_link.compile_program으로 entry 전체를 컴파일하고 v09_stats.cost로 비교합니다. 기본값은 최초 합법 module 이후 120초이며 None으로 해제할 수 있습니다. Runtime은 선택된 NpuExecutable과 host transform을 사용합니다.' },
  (s, D) => {
    const y1 = 1.5, h = 0.78, g = 0.22;
    const w4 = (12.33 - 3 * g) / 4;
    // Row 1: frontend & graph
    const r1 = [
      ['모델 + 정적 shape', 'nn_models/llama.py\nrelax.frontend.nn → export_tvm', 'host'],
      ['Graph prefix (1회)', 'CanonicalizeAttention · CanonicalizeBindings\nEliminateCommonSubexpr · RewriteDataflowReshape · LiftTransformParams', 'ir'],
      ['후보 graph 생성 (≤27)', 'attention {off, score, full}\n× {gqa, producers} × generic {off, vectors, matmul}', 'ir'],
      ['LegalizeOps → TIR', 'custom legalize map · FoldConstants(K=64)\nAnnotateTIROpPattern · FuseOps/FuseTIR · DCE', 'ir'],
    ];
    r1.forEach((b, i) => {
      const x = 0.5 + i * (w4 + g);
      D.box(s, x, y1, w4, h, b[0], { kind: b[2], size: 11.5, detail: b[1], dsize: 8 });
      if (i < 3) D.arrow(s, x + w4 + 0.02, y1 + h / 2, x + w4 + g - 0.02, y1 + h / 2, { width: 1.5 });
    });
    // Row 2: selection & per-entry compile
    const y2 = y1 + h + 0.52;
    const r2 = [
      ['npu_fusion.select', '영역(FuseOps region)마다 unfused vs fused를\n실제 codegen 비용으로 probe · 진 영역은 inline', 'sram'],
      ['npu_link.compile_program', 'entry(prefill_cache, decode)마다 아래 단계 실행\nEntryCache: 동일 entry는 재사용', 'sram'],
      ['비용 비교 · 선택', 'cost = (DMA bytes, words, SRAM peak) 사전식\n모든 entry 비악화 + 하나 이상 개선 시 채택', 'warn'],
      ['NpuExecutable', 'words + plan(주소·ABI·통계) + lowered module\nexport_bundle → bin/JSON/IR', 'unit'],
    ];
    r2.forEach((b, i) => {
      const x = 0.5 + i * (w4 + g);
      D.box(s, x, y2, w4, h, b[0], { kind: b[2], size: 11.5, detail: b[1], dsize: 8 });
      if (i < 3) D.arrow(s, x + w4 + 0.02, y2 + h / 2, x + w4 + g - 0.02, y2 + h / 2, { width: 1.5 });
    });
    // arrow from row1 end to row2 start (elbow)
    D.velbow(s, 0.5 + 3 * (w4 + g) + w4 / 2, y1 + h, 0.5 + w4 / 2, y2, { my: y1 + h + 0.14, width: 1.5 });
    // Row 3: inside compile_program
    const y3 = y2 + h + 0.55;
    D.rect(s, 0.5, y3 - 0.28, 12.33, 1.22, { fill: C.faint, line: C.line, radius: 0.08 });
    D.text(s, 0.65, y3 - 0.25, 6, 0.25, 'compile_program 내부 (entry 하나)', { size: 9.5, bold: true, color: C.sram });
    const w5 = (12.03 - 4 * 0.18) / 5;
    const r3 = [
      ['DRAM 계획', 'npu_memplan: StaticPlanBlockMemory\n→ flat pool 주소', 'dram'],
      ['SRAM 보관 탐색', 'npu_memory_search: beam 4\n값 보관 + 호출별 재schedule', 'sram'],
      ['호출별 schedule', 'npu_schedule / npu_intrin\n후보 공간 · tensorize · USMP', 'sram'],
      ['Codegen', 'tir_codegen_v09 (Python)\nv09_walker.cc (native)', 'unit'],
      ['Link 마무리', 'residency · peephole\nDRAM compaction · HALT', 'unit'],
    ];
    r3.forEach((b, i) => {
      const x = 0.65 + i * (w5 + 0.18);
      D.box(s, x, y3, w5, 0.82, b[0], { kind: b[2], size: 10.5, detail: b[1], dsize: 8 });
      if (i < 4) D.arrow(s, x + w5 + 0.02, y3 + 0.41, x + w5 + 0.16, y3 + 0.41, { width: 1.2 });
    });
    // feedback arrow: cost -> candidate generation
    D.line(s, 0.5 + 2 * (w4 + g) + w4 / 2, y2, 0.5 + 2 * (w4 + g) + w4 / 2, y1 + h + 0.06, { color: C.warn, width: 1.5, dash: 'dash', arrow: 'triangle' });
    D.text(s, 0.5 + 2 * (w4 + g) + w4 / 2 + 0.08, y1 + h + 0.2, 2.6, 0.24, '다음 후보 (기본 120 s / None 가능)', { size: 8.5, color: C.warn, bold: true });
    // Row 4: runtime
    const y4 = y3 + 1.15;
    const r4 = [
      ['Host transform 실행', 'relax.build(host,"llvm") VM → W transpose 등\nparams-cache (IR+weight hash)', 'host'],
      ['DRAM image 구성', 'build_image: plan.address에 입력·parameter·상수', 'dram'],
      ['C-model 실행', 'program_memory.bin + global_memory.bin\n→ saved image + perf counters', 'unit'],
      ['출력 / 다음 step', 'read_outputs(plan.outputs) → logits argmax\nKV append tail, 입력 patch 후 재실행', 'host'],
    ];
    r4.forEach((b, i) => {
      const x = 0.5 + i * (w4 + g);
      D.box(s, x, y4, w4, h, b[0], { kind: b[2], size: 11.5, detail: b[1], dsize: 8 });
      if (i < 3) D.arrow(s, x + w4 + 0.02, y4 + h / 2, x + w4 + g - 0.02, y4 + h / 2, { width: 1.5 });
    });
    D.text(s, 0.5, y4 - 0.27, 3, 0.25, 'Runtime (npu_generate)', { size: 9.5, bold: true, color: C.host });
  });

  // ---------------------------------------------------------------- 8 TVM vs NPU
  D.add({ section: '1 · 전체 구조', title: 'TVM 표준 기능 위에 NPU backend의 결정을 얹는다', subtitle: 'TVM이 제공하는 것(표현·변환·분석)과 backend가 정의하는 것(장치 계약·후보·비용·명령)을 구분합니다. DLight는 미사용이며 MetaSchedule trace 재사용·ReplayFunc 탐색은 선택형입니다.', source: 'docs/current/01_overview.md, reference/tvm_api.md',
    notes: 'TVM은 Relax graph pass, TE/TIR 표현, tir.Schedule loop 변환, CompactBufferAllocation, StaticPlanBlockMemory, USMP 충돌 분석을 제공합니다. NPU backend는 LegalizeOps의 custom map, TensorIntrin 등록, 후보 생성과 비용 검사, 커널 사이 SRAM 보관, ISA 인코딩과 runtime image를 담당합니다. DLight는 규칙 모음일 뿐 SRAM 수명 관리의 다른 이름이 아니므로, 표준 Schedule API 위에 직접 탐색을 두었습니다. MetaSchedule은 09-29부터 선택형 두 기능(JSONDatabase trace 재사용, ReplayFunc 추첨 탐색)에만 사용하며 기본 경로는 여전히 space 탐색입니다.' },
  (s, D) => {
    D.box(s, 0.5, 1.55, 6.05, 0.55, 'TVM 0.19이 담당 (표준 API 그대로 사용)', { kind: 'ir', size: 13 });
    D.table(s, 0.5, 2.15, 6.05, [
      ['영역', '사용하는 API'],
      ['모델 표현', 'relax.frontend.nn, export_tvm, BlockBuilder, call_te / call_tir'],
      ['Graph pass', 'CanonicalizeBindings, EliminateCommonSubexpr, RewriteDataflowReshape, LiftTransformParams, LegalizeOps, FoldConstant, AnnotateTIROpPattern, FuseOps(ByPattern), FuseTIR, DCE'],
      ['Loop 변환', 'tir.Schedule: split, reorder, cache_read/write, compute_at, reverse_compute_at, set_scope, pad_einsum, decompose_reduction, tensorize, rfactor'],
      ['TIR 후처리', 'LowerInitBlock, PlanAndUpdateBufferAllocationLocation, ConvertBlocksToOpaque, CompactBufferAllocation, LowerMatchBuffer/LowerOpaqueBlock(분석용)'],
      ['메모리 분석', 'ToNonDataflow, CallTIRRewrite, StaticPlanBlockMemory, USMP extract_buffer_info + greedy 배치'],
      ['검증', 'structural_hash/equal, relax.build(…,"llvm") + VirtualMachine (host transform, CPU reference)'],
      ['탐색·기록 (선택형)', 'meta_schedule: TuneContext, ScheduleFn, ReplayFunc, sample_categorical, JSONDatabase, TuningRecord, Schedule.trace, relax_integration.extract_tasks'],
    ], { colW: [1.3, 4.75], size: 9, rowH: 0.45 });
    D.box(s, 6.78, 1.55, 6.05, 0.55, 'NPU backend가 정의 (npu_compiler/)', { kind: 'sram', size: 13 });
    D.table(s, 6.78, 2.15, 6.05, [
      ['영역', '구현'],
      ['장치 계약', 'device.NpuProfile, v09_contract.json (dtype/nibble/opcode/수치 정책)'],
      ['수치 정의', 'npu_legalize: matmul(FP32 누산), softmax, RMSNorm(scaled absmax), sum/mean, split view; npu_numeric: K=64 rfactor reference'],
      ['후보 생성·선택', 'tvm_pipeline 패턴, npu_attention, npu_graph_opt, npu_fusion.select, npu_build.lower_and_select, npu_compilation 예산/cache'],
      ['Schedule', 'npu_schedule_space(정규화·용량 필터), npu_schedule(_fit_matmul/_fit_attention), npu_intrin(TensorIntrin, schedule 함수), npu_accumulator, npu_meta_schedule(선택형 ReplayFunc·trace DB)'],
      ['메모리', 'npu_memplan(flat 주소), npu_sram(USMP 연결), npu_memory_search(커널 사이 보관), npu_residency, npu_abi'],
      ['명령·실행', 'tir_codegen_v09 / v09_walker.cc, isa_v09, isa_compute, npu_link, peephole, npu_cache, npu_generate, v09_runtime'],
    ], { colW: [1.3, 4.75], size: 9, rowH: 0.45 });
    D.takeaway(s, '표준 pass를 호출했다고 장치 정책이 저절로 생기지는 않습니다. 반대로 backend 코드가 있다고 TVM 기본 pass인 것처럼 부르지도 않습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 9 code map
  D.add({ section: '1 · 전체 구조', title: '코드 지도: 어느 현상이 어느 단계의 결정인지 먼저 정한다', subtitle: 'd_compiler/npu_compiler (약 12.7k lines, Python + C++). 파일 순서로 읽기보다 관심 있는 결정을 만드는 단계부터 봅니다.', source: 'docs/current/12_operations_limits.md, wc -l npu_compiler/*.py npu_codegen/*.cc',
    notes: '이 표는 이후 슬라이드의 "구현 연결"에 대응합니다. 예를 들어 첫 tile만 맞고 뒤 tile이 틀리면 ISA 주소 단위(isa_v09, tir_codegen) 문제이고, SRAM 보관을 켤 때만 틀리면 npu_memory_search의 값 버전·alias 판단을 먼저 봅니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 12.33, [
      ['책임', '파일 (lines)', '핵심 함수 / 클래스'],
      ['공개 build · 결과물', 'npu_target.py (178)', 'npu_target(), build() → NpuExecutable{words, plan, module}, export_bundle()'],
      ['모델 · checkpoint', 'nn_models/llama.py (435), hf.py (437), assets/llama.py', 'LlamaModel(prefill/prefill_cache/decode), get_default_spec, rope_inputs, causal_mask, Llama32Assets'],
      ['Graph · 후보', 'tvm_pipeline.py (169), npu_build.py (124), npu_fusion.py (161), npu_attention.py, npu_graph_opt.py', 'graph_prefix, graph_pipeline, lower_and_select, select(_probe), CanonicalizeAttention, FoldRepeatedGemv'],
      ['수치 · schedule', 'npu_legalize.py (190), npu_numeric.py, npu_schedule_space.py (78), npu_schedule.py (616), npu_intrin.py (621), npu_accumulator.py (201)', 'legalize_map, tile_order_reference, candidates(), _fit_matmul / _fit_attention / _measure_trial, schedule_matmul_sram, lower_stores'],
      ['메모리 · ABI', 'npu_memplan.py (427), npu_sram.py (210), npu_memory_search.py (379), npu_residency.py (179), npu_abi.py (173)', 'assign_addresses, compact_unused_storage, plan_sram(USMP), discover/plan_program/reserve, forward, output_tree/validate_entry'],
      ['ISA · link · 실행', 'npu_link.py (472), tir_codegen_v09.py (1533), npu_codegen/v09_walker.cc (1696), isa_v09.py, isa_compute.py, peephole.py, npu_cache.py, npu_generate.py (398), v09_runtime.py', 'compile_program, Walker.emit_gemm/flush, DMA packet 인코더, AppendPlan, generate/DecodeSession, PersistentTransport'],
      ['Compile 제어', 'npu_compilation.py (373), v09_stats.py, npu_provenance.py', 'compilation() context: recipe/trial/probe/region/Entry cache, optimization_budget, phase; analyze/cost'],
    ], { colW: [1.7, 4.6, 6.03], size: 9, rowH: 0.5 });
    D.callout(s, 0.5, 6.35, 12.33, 0.55, '검증 도구: tests/ 17개 suite (run_gate.sh) · docs/examples/reproduce.py (작은 예제 실제 실행) · run_nn_generate.py (전체 Llama 생성) · run_nn_npu.py (prefill + HF/LLVM cosine)', { kind: 'faint', size: 10 });
  });
};
