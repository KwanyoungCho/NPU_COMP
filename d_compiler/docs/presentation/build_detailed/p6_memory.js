'use strict';
const { C, F } = require('./lib');

module.exports = (D) => {
  // ---------------------------------------------------------------- 46 three plans
  D.add({ section: '8 · 메모리 계획', title: '세 가지 메모리 계획이 서로 다른 범위를 본다', subtitle: '① DRAM은 Relax tensor 수명으로 pool을 나누고 flat 주소를 주며, ② 커널 안에서는 USMP가 buffer offset을 겹치고, ③ 커널 사이에서는 보관값을 SRAM 위쪽에 예약합니다. 커널을 합치지 않아도 SRAM만으로 연결할 수 있습니다.', source: 'docs/current/08_memory.md, npu_compiler/npu_link.py _compile_program (단계 순서)',
    notes: '위 그림은 두 메모리의 실제 구성입니다. DRAM image에는 parameter leaf(변환된 순서), graph 상수, StaticPlanBlockMemory가 만든 storage pool(수명이 겹치지 않는 tensor가 공유), 출력 tree, literal pool이 있습니다. SRAM에는 literal·scratch, 커널의 USMP pool(buffer가 offset을 공유), 그리고 커널 사이 보관값이 위에서 아래로 예약됩니다. 세 계획은 서로 자동으로 해결되지 않습니다. USMP를 호출했다고 persistent KV cache나 전체 그래프의 보관 정책이 생기지 않습니다. compile_program 안의 실제 순서는 assign_addresses → validate_entry → scalar pool → plan_program(joint SRAM search) → 호출별 schedule/USMP/codegen → residency → peephole → baseline 비교 → compact_unused_storage → cache append 재구성입니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 8, 0.24, 'DRAM image (flat, unit 2 B) — ① StaticPlanBlockMemory의 pool + backend의 flat 주소', { size: 9.5, bold: true, color: C.dram });
    D.membar(s, 0.5, 1.76, 12.33, 0.5, 0, 100, [
      { from: 0, to: 30, label: 'parameter leaves (변환된 순서)', sub: 'p.0.1 … weight, bias', kind: 'dram' },
      { from: 30, to: 38, label: 'graph 상수', sub: 'const_data', kind: 'host' },
      { from: 38, to: 56, label: 'storage pool A', sub: 'G → U 재사용 (수명 안 겹침)', kind: 'ir' },
      { from: 56, to: 70, label: 'storage pool B', sub: 'S → H', kind: 'ir' },
      { from: 70, to: 86, label: '출력 tree', sub: 'logits, K_new, V_new', kind: 'unit' },
      { from: 86, to: 100, label: 'literal pool · FP32 1.0', sub: 'linker가 마지막에', kind: 'faint' },
    ], { size: 8.5 });
    D.text(s, 0.5, 2.5, 8, 0.24, 'SRAM 8 MiB (nibble) — ② USMP: 커널 buffer offset 공유   ③ 커널 사이 보관값: top-down 예약', { size: 9.5, bold: true, color: C.sram });
    D.membar(s, 0.5, 2.76, 12.33, 0.5, 0, 100, [
      { from: 0, to: 7, label: 'literal·scratch', sub: '6 slots', kind: 'host' },
      { from: 7, to: 28, label: '② buffer b₀ (pinned)', sub: 'gemm 목적지', kind: 'sram' },
      { from: 28, to: 46, label: '② b₁ / b₃ (offset 공유)', sub: '충돌 없음', kind: 'sram' },
      { from: 46, to: 62, label: '② b₂ / b₄ / input panel', sub: 'compaction된 크기', kind: 'sram' },
      { from: 62, to: 72, label: '(cap)', sub: 'workspace 천장', kind: 'faint' },
      { from: 72, to: 100, label: '③ 보관값 H (다음 커널이 직접 LOAD)', sub: 'reserve(): 위에서 아래로 first-fit', kind: 'warn' },
    ], { size: 8.5 });
    D.table(s, 0.5, 3.5, 12.33, [
      ['계획', '분석하는 범위', '도구와 책임', '결과'],
      ['① 기본 DRAM storage', 'Relax 함수의 tensor 수명과 재사용 (호출 순서 기준)', 'TVM StaticPlanBlockMemory → storage pool; backend가 pool을 flat image 주소로 (npu_memplan.assign_addresses)', 'plan.address[name] (unit 2 B), pool_base/bytes, outputs'],
      ['② 커널 내부 SRAM', 'Scheduled TIR 하나의 임시 buffer 수명·충돌', 'TVM USMP extract_buffer_info + greedy 배치 vs bump; matrix 목적지 pinned (npu_sram.plan_sram)', 'SramLayout: offset(nibble), peak, 알고리즘'],
      ['③ 커널 사이 보관', '생산 호출부터 마지막 소비 호출까지 (값의 버전)', 'NPU whole-tensor 보관 탐색 (beam 4) + 보관 공간을 뺀 용량으로 호출별 재schedule (npu_memory_search.plan_program)', 'retained_values, 호출별 recipe/words, 보관 arena'],
    ], { colW: [1.8, 3.0, 4.6, 2.93], size: 8.5, rowH: 0.46 });
    D.text(s, 0.5, 5.4, 12, 0.24, 'compile_program 안에서의 실제 순서 (entry 하나)', { size: 9.5, bold: true, color: C.navy });
    const st = [['assign_addresses', '① DRAM plan', 'dram'], ['validate_entry', 'ABI 검사', 'host'], ['scalar pool', 'FP16 literal + 1.0/2.0', 'host'], ['plan_program', '③ joint SRAM search', 'sram'], ['호출별 emission', '② schedule·USMP·codegen', 'sram'], ['residency', 'forwarding', 'unit'], ['peephole', 'descriptor 중복', 'unit'], ['baseline 비교', 'joint vs all-DRAM', 'warn'], ['DRAM 압축', 'compact_unused_storage', 'dram'], ['cache append', 'AppendPlan 재구성', 'dram']];
    const w = (12.33 - 9 * 0.1) / 10;
    st.forEach((b, i) => {
      const x = 0.5 + i * (w + 0.1);
      D.box(s, x, 5.66, w, 0.8, b[0], { kind: b[2], size: 7.5, detail: b[1], dsize: 6.3 });
      if (i < 9) D.arrow(s, x + w + 0.005, 6.06, x + w + 0.095, 6.06, { width: 0.75 });
    });
    D.takeaway(s, '같은 DRAM pool이 나중에 다른 값으로 재사용되므로 "주소가 같다"와 "같은 데이터"는 다릅니다. 보관 탐색은 값의 버전(생산 호출, 인자)을 키로 삼습니다.', { y: 6.55 });
  });

  // ---------------------------------------------------------------- 47 DRAM plan
  D.add({ section: '8 · 메모리 계획', title: '기본 DRAM 배치는 표준 TVM 결과에서 시작한다', subtitle: 'ToNonDataflow → RemovePurityChecking → CallTIRRewrite → StaticPlanBlockMemory. CallTIRRewrite가 allocation과 TIR 호출을 명시하고, StaticPlanBlockMemory가 동시에 살아 있지 않은 tensor의 pool 공유를 정합니다.', source: 'npu_compiler/npu_memplan.py (plan_pipeline, assign_addresses), docs/examples/pipeline/06_initial_dram_plan.py, bundle/program.json memory',
    notes: 'plan_pipeline은 RewriteDataflowReshape → OutputViews → ToNonDataflow → RemovePurityChecking → CallTIRRewrite → StaticPlanBlockMemory입니다. assign_addresses는 parameter(인자 순서, tuple leaf p.0.1)를 먼저, graph 상수(내용 키로 dedupe)를 다음, 그리고 binding을 프로그램 순서로 걸으며 alloc_storage→pool, alloc_tensor→pool_base+offset, builtin.alloc_tensor(함수 결과)→자체 slot, reshape/view→alias로 처리합니다. 아래 strip은 작은 예제 bundle의 실제 결과입니다: plan.address는 unit(2 B) 단위로 x 0, bias 8192, lv 8256, gv 16448이며 bytes로는 0 / 16,384 / 16,512 / 32,896입니다. 모든 allocation은 4-byte DMA cell로 반올림되고 literal pool은 linker가 마지막에 붙입니다(41,088).' },
  (s, D) => {
    D.code(s, 0.5, 1.5, 6.4, 2.45, [
      '# 06_initial_dram_plan.py (작은 예제, plan_pipeline 뒤)',
      '@R.function',
      'def prefill(x, bias, lv) -> R.Tensor((64, 64), "float16"):',
      '    R.func_attr({"num_input": 1, "relax.force_pure": True})',
      '    alloc1 = R.builtin.alloc_tensor(R.shape([64, 64]), R.dtype("float16"),',
      '                                    R.prim_value(0), R.str("global"))',
      '    cls.fused_matmul_add_silu(x, lv, bias, alloc1)   # 출력 buffer를 가진 호출',
      '    gv = alloc1',
      '    return gv',
      '# 중간값이 여러 개면 memory.alloc_storage(pool) + memory.alloc_tensor(offset)',
    ], { size: 8.5 });
    D.table(s, 7.15, 1.5, 5.68, [
      ['assign_addresses 순서', '규칙'],
      ['(a) parameters', '인자 순서. tuple은 leaf p.0.1 로 평탄화, plan.tuples 기록'],
      ['(b) graph constants', '(dtype, shape, bytes) 내용 키로 dedupe → const_data'],
      ['(c) bindings (프로그램 순서)', 'alloc_storage → pool_base; alloc_tensor → pool_base + offset//unit; builtin.alloc_tensor(결과) → 자체 slot; reshape/view/ensure_zero_offset → alias + plan.views'],
      ['(d) literal pools (linker)', 'FP16 scalar pool (1.0, 2.0 포함, 짝수 개) → constant_base; FP32 1.0 → one_fp32_base'],
    ], { colW: [1.9, 3.78], size: 8.8, rowH: 0.5 });
    // real DRAM image strip
    D.text(s, 0.5, 4.2, 6.6, 0.24, 'assign_addresses 결과 (작은 예제 bundle, bytes = plan.address × unit 2 B)', { size: 9.5, bold: true, color: C.navy });
    D.membar(s, 0.5, 4.5, 6.4, 0.5, 0, 41100, [
      { from: 0, to: 16384, label: 'x  (address 0)', sub: '16,384 B', kind: 'host' },
      { from: 16384, to: 16512, label: '', sub: '', kind: 'dram' },
      { from: 16512, to: 32896, label: 'lv = Wᵀ (address 8256)', sub: '16,384 B', kind: 'dram' },
      { from: 32896, to: 41088, label: 'alloc1 = gv (16448)', sub: '8,192 B 출력', kind: 'unit' },
      { from: 41088, to: 41100, label: '', sub: '', kind: 'faint' },
    ], { ticks: [0, 16384, 32896, 41088], size: 8 });
    D.text(s, 2.2, 5.2, 2.5, 0.2, '↑ bias (address 8192) 128 B @ 16,384', { size: 7, color: C.dram, bold: true });
    D.text(s, 5.2, 5.2, 1.7, 0.2, 'literal 12 B @ 41,088 ↑', { size: 7, color: C.muted, bold: true, align: 'right' });
    D.text(s, 0.5, 5.45, 6.4, 0.7, '주소는 4-byte cell로 반올림 (bias 128 B는 cell 정렬됨). image 크기 = plan.top × 2 = 41,100 B. 이 배치는 SRAM 선택·최종 압축 이전의 기본 storage 계획이며, 8장 뒤의 compact_unused_storage가 미사용 pool을 제거하면 주소가 당겨집니다.', { size: 8.5, color: C.muted });
    D.text(s, 7.15, 4.2, 5.68, 0.3, 'StaticPlan 객체의 주요 field', { size: 10.5, bold: true, color: C.navy });
    D.bullets(s, 7.15, 4.5, 5.68, 1.9, [
      ['배치: ', 'address, nbytes, pool_base, pool_bytes, tuples, const_data, graph_constants, top, views'],
      ['ABI·linker: ', 'func, outputs(tree), constant_values, constant_base, one_fp32_base, codegen_contract, numeric_policy'],
      ['codegen 결과: ', 'kernel_records(index, name, backend, sram_peak, arguments[address/size/dtype], span), sram_layouts, schedules, call_schedules'],
      ['메모리 결과: ', 'joint_memory, residency, dram_compaction, cache_append/append_plan, compilation'],
    ], { size: 9, psa: 3 });
    D.takeaway(s, '수명은 생성부터 마지막 사용까지입니다. G의 공간에 U를 쓸 수 있는 이유는 같은 배열로 변환해서가 아니라 G의 옛 값이 더 이상 관찰되지 않기 때문입니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 48 lifetime gantt
  D.add({ section: '8 · 메모리 계획', title: '수명 분석: 실제 폭 MLP의 G/S/U/H는 언제까지 살아 있는가', subtitle: 'gate → SiLU → up → multiply → down (최신 실제 폭 MLP 측정의 순서). M=64에서 G/S/U/H는 각각 [64,8192] FP16 = 1 MiB입니다.', source: 'docs/current/08_memory.md, evidence/current/mlp64.json',
    notes: 'G는 SiLU가 끝나면 더 이상 필요하지 않고, S는 multiply까지, U도 multiply까지, H는 down까지 필요합니다. 표는 값의 관계를 설명하며 실제 peak에는 weight 조각, scratch, 정렬, 주소 사이 빈 공간도 들어갑니다. 살아 있는 tensor 크기의 합과 최고 주소는 같지 않을 수 있습니다. 이 순서는 최신 측정의 순서이며 과거 문서의 up→gate 순서를 현재 결과에 적용하지 않습니다.' },
  (s, D) => {
    const kernels = ['gate', 'SiLU', 'up', 'multiply', 'down'];
    const x0 = 2.3, y0 = 1.9, cw = 2.0, rh = 0.55;
    kernels.forEach((k, i) => D.box(s, x0 + i * cw, y0 - 0.55, cw - 0.1, 0.45, k, { kind: i === 0 || i === 2 || i === 4 ? 'unit' : 'ir', size: 11 }));
    const vals = [['G = gate(X)', 0, 1, 'FP16 [64,8192] 1 MiB'], ['S = SiLU(G)', 1, 3, '1 MiB'], ['U = up(X)', 2, 3, '1 MiB'], ['H = S × U', 3, 4, '1 MiB'], ['X (norm 출력)', 0, 2, '384 KiB (입력)']];
    vals.forEach((v, r) => {
      const y = y0 + r * (rh + 0.12);
      D.text(s, 0.5, y + 0.12, 1.75, 0.3, v[0], { size: 10.5, bold: true, color: C.navy });
      D.rect(s, x0, y, kernels.length * cw - 0.1, rh, { fill: C.faint, line: C.faint, radius: 0.02, lineWidth: 0 });
      D.rect(s, x0 + v[1] * cw, y, (v[2] - v[1] + 1) * cw - 0.1, rh, { kind: r === 4 ? 'host' : 'sram', radius: 0.04 });
      D.text(s, x0 + v[1] * cw, y, (v[2] - v[1] + 1) * cw - 0.1, rh, `생성 → 마지막 사용  (${v[3]})`, { size: 9, align: 'center', valign: 'middle' });
    });
    D.table(s, 0.5, 5.35, 7.6, [
      ['계산 중', '동시에 필요한 주요 값', '완료 후 보관'],
      ['gate', 'X, G 출력 + W panel 작업 공간', 'G'],
      ['SiLU', 'G, S + 임시값', 'S'],
      ['up', 'X, S, U + W panel 작업 공간', 'S, U'],
      ['multiply', 'S, U, H', 'H'],
      ['down', 'H + W panel(4 MiB) + 출력', '외부 출력 계약에 따른 Y'],
    ], { colW: [1.2, 4.0, 2.4], size: 8.5, rowH: 0.24 });
    D.callout(s, 8.4, 5.35, 4.43, 1.0, [
      { text: '실제 peak ≠ 살아 있는 값의 합: ', bold: true }, { text: 'weight panel(gate/up 384 KiB, down 4 MiB), scratch, 정렬, 빈 공간이 더해집니다. M=64 MLP 기록의 SRAM peak는 7,446,540 B (8 MiB 중).', size: 9.5 },
    ], { kind: 'warn' });
    D.takeaway(s, '보관 탐색은 이 수명을 값의 버전 단위로 분석하고, 보관할 값의 공간을 예약한 뒤 남은 용량으로 각 커널을 다시 schedule합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 49 USMP intra-kernel
  D.add({ section: '8 · 메모리 계획', title: 'USMP: 커널 내부 buffer의 충돌 분석과 greedy 배치', subtitle: '작은 예제 fused 커널의 실제 결과. 8개 buffer(총 163,840 nibbles)가 충돌하지 않는 것끼리 offset을 공유해 pool 81,920 nibbles에 들어갑니다 (baseline bump 150,808 → peak 85,016 nibbles, 32,896 B 절약).', source: 'npu_compiler/npu_sram.py plan_sram, docs/examples/pipeline/bundle/program.json sram_layouts',
    notes: '분석용 복사본에 LowerMatchBuffer/LowerOpaqueBlock을 적용하고 실행용 TIR은 바꾸지 않습니다. 각 Allocate에 WorkspacePoolInfo("npu_sram")를 붙여 extract_buffer_info로 충돌 정보를 얻고, offset을 buffer.data Var identity로 다시 연결합니다. gemm/fill 목적지(matmul_global.sram)와 알 수 없는 extern call의 pointer는 pinned로 모든 buffer와 충돌 처리되어 offset 0을 혼자 씁니다. 차트에서 같은 offset을 공유하는 쌍(T_add와 lv panel, compute_intermediate와 T_multiply, sigmoid 결과·bias·x panel)은 수명이 겹치지 않는 buffer들입니다. greedy_by_size와 greedy_by_conflicts를 돌려 더 작은 peak를 택하되 bump보다 좋을 때만 씁니다. 크기는 nibble(fp16=4)로 8-nibble(32-bit cell) 정렬입니다.' },
  (s, D) => {
    D.lifetimeChart(s, 3.0, 1.95, 4.8, 2.7, 0, 81920, [
      { name: 'matmul_global.sram (gemm 목적지, pinned)', from: 0, to: 16384, kind: 'unit', pinned: true, label: '0 – 16384' },
      { name: 'T_add_intermediate (A)', from: 16384, to: 32768, kind: 'sram', label: '16384 –' },
      { name: 'lv panel (Wᵀ 128×64)', from: 16384, to: 49152, kind: 'dram', label: '16384 – 49152 (공유)' },
      { name: 'compute_intermediate (P f16)', from: 32768, to: 49152, kind: 'sram', label: '32768 –' },
      { name: 'T_multiply_intermediate (Y)', from: 32768, to: 49152, kind: 'ir', label: '32768 – (공유)' },
      { name: 'compute (sigmoid)', from: 49152, to: 65536, kind: 'sram', label: '49152 –' },
      { name: 'bias (64)', from: 49152, to: 49408, kind: 'ir', label: '256' },
      { name: 'x panel (64×128)', from: 49152, to: 81920, kind: 'dram', label: '49152 – 81920 (공유)' },
    ], { labelW: 2.45, size: 7.5, ticks: [0, 16384, 32768, 49152, 65536, 81920], title: 'USMP greedy_by_conflicts 결과 — 가로축 = pool 안의 offset (nibble, pool base 3096 기준)' });
    D.text(s, 0.5, 4.95, 7.3, 0.55, '같은 세로 구간을 공유하는 buffer는 conflicts 목록에 서로 없는 쌍(수명이 겹치지 않음). pinned buffer는 모든 buffer와 충돌로 취급되어 혼자 offset 0을 씁니다(SAVE가 지연되는 동안 덮어쓰기 방지).', { size: 8.5, color: C.muted });
    D.text(s, 0.5, 5.55, 7.3, 0.24, 'plan_sram 순서', { size: 9.5, bold: true, color: C.navy });
    D.text(s, 0.5, 5.8, 7.3, 0.6, '크기(nibble, 8 정렬) → 분석 복사본(LowerMatchBuffer·LowerOpaqueBlock) → pinned 표시 → WorkspacePoolInfo + extract_buffer_info(conflicts) → greedy_by_size / greedy_by_conflicts vs bump → validate(정렬·겹침·peak) → place(cursor, capacity)', { size: 8, color: C.ink, font: F.code });
    D.text(s, 8.05, 1.5, 4.8, 0.3, 'SRAM 지도 (주소 0부터 위로)', { size: 10.5, bold: true, color: C.navy });
    const segs = [['FP16 literal pool (nibble 0..)', 'host', 0.32], ['FP32 1.0', 'host', 0.22], ['scratch 6 slots × 최대 행 × 4 nib', 'ir', 0.5], ['USMP pool (pool_base 3096 → 위 차트)', 'sram', 1.05], ['커널 사이 보관값 (top-down 예약)', 'warn', 0.75]];
    let yy = 1.85;
    segs.forEach(g => { D.rect(s, 8.05, yy, 4.78, g[2], { kind: g[1], radius: 0.02 }); D.text(s, 8.15, yy, 4.6, g[2], g[0], { size: 9, valign: 'middle', bold: true }); yy += g[2]; });
    D.table(s, 8.05, 4.85, 4.78, [
      ['작은 예제 값', ''],
      ['pool_base', '3096 nibbles = 24 + 6 × 128 × 4'],
      ['peak / baseline', '85,016 / 150,808 nibbles'],
      ['절약', '32,896 B (usmp_pressure 40,960 B)'],
      ['알고리즘', 'greedy_by_conflicts ("validated lifetime reuse")'],
    ], { colW: [1.4, 3.38], size: 8.5, rowH: 0.28 });
    D.takeaway(s, '충돌하지 않는 buffer가 offset을 공유합니다. 지연된 accumulator 상태나 opaque 호출 때문에 사용이 끝났다고 단정할 수 없는 buffer는 pin해 조기 재사용을 막습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 50 inter-kernel retention search
  D.add({ section: '8 · 메모리 계획', title: '커널 사이 SRAM 보관 탐색: 값의 버전을 찾고, 보관 공간을 예약한 뒤, 호출별로 다시 schedule한다', subtitle: 'npu_memory_search.plan_program. 보관값은 SRAM 위에서 아래로 예약되고, 커널 workspace는 그 아래 천장(cap)까지만 쓰도록 재schedule됩니다. beam width 4, all-DRAM 진행도 항상 비교. tensor 전체 보관만 있습니다.', source: 'npu_compiler/npu_memory_search.py (discover, reserve, parameter_scopes, plan_program)',
    notes: 'discover는 PrimFunc 호출마다 인자의 byte 범위와 읽기/쓰기를 얻고, 출력 tree·parameter leaf·graph 상수 범위를 보호합니다. 후보는 FP16이고, 같은 호출에서 읽지 않고(RMW 아님), compact하며, 보호 범위·같은 호출의 다른 인자와 겹치지 않는 쓰기입니다. 이후 호출을 앞으로 훑어 부분 겹침·dtype 재해석, in-place 갱신, 부분 덮어쓰기는 거절하고 전체 덮어쓰기는 버전 종료로 봅니다. 오른쪽 그림처럼 reserve는 SRAM 위에서 아래로 first-fit이며 workspace는 항상 그 아래입니다. parameter_scopes가 bound parameter를 global.sram으로 바꿔 schedule이 cache stage를 생략하고 출력 tile을 직접 쓰게 합니다. 프로그램 수준에서는 all-DRAM baseline과 (dma, words, peak)를 비교해 엄격히 작을 때만 채택합니다.' },
  (s, D) => {
    D.code(s, 0.5, 1.5, 7.3, 4.85, [
      'calls, values, excluded = discover(module, plan)     # 값 = (생산 호출, 인자) 버전',
      'states = [(live={}, prefix=[], dma=0, words=0, peak=0)]',
      'for i, call in enumerate(calls):',
      '  outs = [v for v in values if v.producer == i]; ext = []',
      '  for (live, prefix, dma, words, peak) in states:',
      '    R = [live]                                # 보관/비보관 조합 (≤ 2·beam)',
      '    for v in outs: R += [r + {v: reserve(r, v.size)} for r in R if fits]',
      '    for r in R:',
      '      bind = SRAM 주소 (읽는 입력: live[v], 쓰는 출력: r[v])',
      '      cap  = min(reservation addrs in r) or SRAM   # workspace 천장',
      '      prim = parameter_scopes(call.prim, bind)       # bound param → global.sram',
      '      recipe = _fit_kernel(prim, profile(sram_bytes=cap//2))   # 재schedule',
      '      stats  = probe(...)                           # emit (또는 cost cache)',
      '      live2 = {v in r : v.last > i}                 # 마지막 소비 지나면 해제',
      '      ext.append((live2, prefix+[choice], dma+stats.dma, words+stats.words, max(peak, top)))',
      '  states = best per live-layout, sorted by (dma, words, peak)[:beam] (+ all-DRAM 상태)',
      'winner = min(states, key=(dma, words, peak))',
      '# 재emit: cost cache 재사용분을 실제 주소로 다시 생성·검증',
      '# 보관 arena를 최대 workspace peak 바로 위로 내림 → 재emit',
      '# 프로그램 수준: all-DRAM baseline과 cost 비교, 엄격히 작을 때만 채택',
    ], { size: 8 });
    D.table(s, 8.05, 1.5, 4.78, [
      ['discover의 보관 후보 조건', ''],
      ['dtype', 'FP16'],
      ['같은 호출에서 읽지 않음', 'RMW 제외'],
      ['compact', 'stride 없음, elem_offset 0'],
      ['보호 범위와 겹치지 않음', '출력 tree, parameter leaf, graph 상수'],
      ['이후 호출 검사', '부분 겹침/dtype 재해석 ✗, in-place ✗, 부분 덮어쓰기 ✗, 전체 덮어쓰기 = 버전 종료'],
    ], { colW: [1.9, 2.88], size: 8.5, rowH: 0.34 });
    // vertical SRAM reservation figure
    const fx = 8.15, fy = 3.95, fw = 0.95, fh = 2.4;
    D.text(s, 8.05, 3.65, 4.78, 0.24, 'SRAM 예약 방향 (두 matmul 예제, 44 KiB)', { size: 9.5, bold: true, color: C.navy });
    const segs = [['literal·scratch\n0 – 1548', 'host', 0.04], ['커널 workspace\n(USMP pool)\nX·W1 → 26,124 B', 'sram', 0.55], ['(cap = 26,124 = 예약 주소)', 'faint', 0.04], ['보관값 H\n26,124 – 42,508\n(16,384 B)', 'warn', 0.31], ['빈 공간 → 45,056', 'faint', 0.06]];
    let cy = fy + fh;
    segs.forEach(g => { const hh = g[2] * fh; cy -= hh; D.rect(s, fx, cy, fw, hh, { kind: g[1], radius: 0.02 }); if (hh > 0.2) D.text(s, fx + 0.03, cy, fw - 0.06, hh, g[0], { size: 6, align: 'center', valign: 'middle', color: C.ink }); });
    // labels to the right
    D.text(s, fx + fw + 0.1, fy, 3.5, 0.5, '↑ 위: 보관값이 top-down first-fit로 예약 (8-nibble 정렬). 이 예제는 arena를 workspace peak 바로 위(26,124)로 내림', { size: 7.5, color: C.warn });
    D.text(s, fx + fw + 0.1, fy + 0.95, 3.5, 0.5, 'cap = min(예약 주소) → profile.sram_bytes = cap/2 로 커널 재schedule', { size: 7.5, color: C.muted });
    D.text(s, fx + fw + 0.1, fy + 1.5, 3.5, 0.6, '↓ 아래: 커널 workspace (scratch + USMP pool). 보관 parameter가 workspace와 겹치면 LinkError', { size: 7.5, color: C.sram });
    D.text(s, fx - 0.1, fy - 0.02, 0.1, 0.2, '', { size: 6 });
    D.text(s, fx + fw + 0.1, fy + 2.1, 3.5, 0.3, 'beam 4 + all-DRAM 상태 유지 · residency=True, snapshot 없음, w8a8 tir일 때만', { size: 7, color: C.muted });
    D.takeaway(s, '보관하지 못하면 DRAM 경로를 쓰므로 큰 activation이 곧 컴파일 불가는 아닙니다. 보관은 이득이 실제 비용으로 확인될 때만 채택됩니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 51 two-matmul addresses
  D.add({ section: '8 · 메모리 계획', title: '작은 예제 ②: 두 matmul이 SRAM 44 KiB에서 H를 공유하는 실제 주소', subtitle: 'X[64,64] @ W1[64,128] → H[64,128]; H @ W2[128,64] → Y[64,64]. 단위 bytes. H는 DRAM에 저장되지 않고 두 번째 matmul이 이 위치를 직접 읽습니다.', source: 'docs/examples/two_matmuls_recorded/program.json, scheduled_0000.py, scheduled_0001.py, docs/current/08_memory.md',
    notes: 'SRAM 낮은 주소에는 literal pool과 scratch(6 slots × 128 × 4 nibbles)가 있어 pool base가 nibble 3096 = byte 1548입니다. 첫 커널은 X(8192 B)를 1548에, W1(16384 B)을 9740에, H(16384 B)를 26124에 둡니다. 두 번째 커널은 Y 작업 공간이 X 자리를, W2가 W1 자리를 재사용하고 H는 유지됩니다. 전체 peak는 42,508 B이며 용량 45,056 B(44 KiB) 안입니다. 탐색 기록: H는 처음 nibble 57344에 예약되었다가 arena를 2,548 B 내려 nibble 52248(byte 26124)로 이동했고, 두 커널 모두 bump 배치(USMP가 더 줄이지 못함)였으며 시도 4회였습니다.' },
  (s, D) => {
    const lo = 0, hi = 45056;
    D.text(s, 0.5, 1.55, 6, 0.3, '커널 1: H 생성 (matmul)', { size: 11, bold: true, color: C.navy });
    D.membar(s, 0.5, 1.9, 12.33, 0.7, lo, hi, [
      { from: 0, to: 1548, label: 'pool', sub: 'literal·scratch', kind: 'host' },
      { from: 1548, to: 9740, label: 'X', sub: '8,192 B', kind: 'sram' },
      { from: 9740, to: 26124, label: 'W1', sub: '16,384 B', kind: 'dram' },
      { from: 26124, to: 42508, label: 'H (보관)', sub: '16,384 B, 직접 SAVE', kind: 'unit' },
    ], { ticks: [0, 1548, 9740, 26124, 42508, 45056], size: 10 });
    D.text(s, 0.5, 3.15, 6, 0.3, '커널 2: H 사용 (matmul1)', { size: 11, bold: true, color: C.navy });
    D.membar(s, 0.5, 3.5, 12.33, 0.7, lo, hi, [
      { from: 0, to: 1548, label: 'pool', sub: '', kind: 'host' },
      { from: 1548, to: 9740, label: 'Y 작업 공간', sub: 'X 자리 재사용', kind: 'sram' },
      { from: 9740, to: 26124, label: 'W2', sub: 'W1 자리 재사용', kind: 'dram' },
      { from: 26124, to: 42508, label: 'H (유지 → 직접 LOAD)', sub: 'GSTORE/GLOAD 없음', kind: 'unit' },
    ], { ticks: [0, 1548, 9740, 26124, 42508, 45056], size: 10 });
    D.arrow(s, 0.5 + 12.33 * (34316 / 45056), 2.6, 0.5 + 12.33 * (34316 / 45056), 3.5, { color: C.unit, width: 2 });
    D.text(s, 0.5 + 12.33 * (34316 / 45056) + 0.1, 2.85, 2.5, 0.25, 'H 같은 allocation 유지', { size: 9, color: C.unit, bold: true });
    D.table(s, 0.5, 4.75, 6.4, [
      ['값', 'SRAM 시작', '크기', '이후 재사용'],
      ['X', '1,548', '8,192', '두 번째 커널의 Y 작업 공간'],
      ['W1', '9,740', '16,384', '두 번째 커널의 W2'],
      ['H', '26,124', '16,384', '두 번째 커널까지 유지'],
      ['peak', '42,508', '', '용량 45,056 (44 KiB)'],
    ], { colW: [0.8, 1.2, 1.2, 3.2], size: 9, rowH: 0.3, aligns: ['left', 'right', 'right', 'left'] });
    D.table(s, 7.15, 4.75, 5.68, [
      ['joint_memory 기록', '값'],
      ['policy', 'bounded call-site schedule and retained-SRAM search'],
      ['beam / candidates / trials', '4 / 1 / 4'],
      ['H 예약', 'nibble 57344 → arena 2,548 B 하향 → 52248 (byte 26,124)'],
      ['USMP', '두 커널 모두 bump (greedy가 더 줄이지 못함)'],
      ['제거된 DRAM pool', 'storage 16,384 B @ byte 40,960 (H의 DRAM 예약)'],
    ], { colW: [1.9, 3.78], size: 8.5, rowH: 0.3 });
    D.takeaway(s, 'H의 GSTORE/GLOAD(32,768 B)가 사라져 DMA 81,932 → 49,164 B. 32 KiB에서는 첫 matmul이 더 이상 들어가지 않아 보관이 적용되지 않습니다 (E2).', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 52 direct output store
  D.add({ section: '8 · 메모리 계획', title: '임시 출력 tile을 없애고 H에 바로 쓴다: view와 row stride', subtitle: 'H는 행마다 128 FP16 = row stride 256 B. 두 번째 열 tile의 시작은 H + 128 B이지 H + 8192 B가 아닙니다. 두 tile은 각 행의 왼쪽/오른쪽 절반이므로 stride를 유지해야 합니다.', source: 'npu_compiler/npu_accumulator.forward_output_tiles, docs/examples/two_matmuls_recorded/scheduled_0000.py, evidence/current/refinement.json (E3)',
    notes: '이전에는 별도 64×64 출력 buffer에 SAVE하고 H로 복사했습니다. 현재는 출력 parameter가 SRAM에 보관될 때 forward_output_tiles가 fill+gemm intrinsic의 view를 출력에 직접 겨냥하고 identity copy를 제거합니다. scheduled_0000.py에서 compute가 scope global.sram의 match_buffer이고 C = match_buffer(compute[0:64, i1*64:…], strides=("C_s0",1))로 나타나며 attr npu.forwarded_output_tiles=1이 붙습니다. FP32 누산이 끝난 최종 저장 위치를 바꾸는 것이지 K 중간 partial sum을 H에 내리는 것이 아닙니다. E3에서 words 85→77, bit-exact, DMA/SRAM peak는 그대로였습니다.' },
  (s, D) => {
    // H layout diagram: 64 rows x 128 cols, two tiles
    D.text(s, 0.5, 1.5, 6, 0.3, 'H [64,128] FP16 in SRAM (row-major, stride 256 B)', { size: 11, bold: true, color: C.navy });
    D.grid(s, 0.5, 1.85, 5.2, 2.2, 4, 2, { highlight: (r, c) => c === 0 ? 'sram' : 'unit', label: (r, c) => r === 0 ? (c === 0 ? 'tile 0: 열 0..63' : 'tile 1: 열 64..127') : (r === 3 ? (c === 0 ? 'H + 63·256' : 'H + 63·256 + 128') : (r === 1 ? (c === 0 ? 'H + 256' : 'H + 256 + 128') : '')), size: 8.5 });
    D.text(s, 0.5, 4.1, 5.2, 0.5, 'tile 1의 시작 = H + 128 B (첫 행의 오른쪽 절반). MAIN descriptor는 전체 stride(128 elements), PARTIAL descriptor는 이번 tile의 시작과 64×64 크기.', { size: 9.5, color: C.muted });
    D.text(s, 6.0, 1.5, 6.8, 0.3, '이전 (임시 tile 경유) → 현재 (직접 SAVE)', { size: 11, bold: true, color: C.navy });
    const a1 = D.box(s, 6.0, 1.9, 2.0, 0.7, '완성된 출력', { kind: 'unit', size: 10.5, detail: 'FP32 acc 완료', dsize: 8 });
    const a2 = D.box(s, 8.4, 1.9, 2.0, 0.7, '임시 64×64 tile', { kind: 'sram', size: 10.5, detail: 'SAVE', dsize: 8 });
    const a3 = D.box(s, 10.8, 1.9, 2.03, 0.7, '보관할 H', { kind: 'sram', size: 10.5, detail: 'SRAM→SRAM copy', dsize: 8 });
    D.arrow(s, a1.r, a1.cy, a2.x, a2.cy); D.arrow(s, a2.r, a2.cy, a3.x, a3.cy);
    const b1 = D.box(s, 6.0, 3.0, 2.0, 0.7, '완성된 출력', { kind: 'unit', size: 10.5, detail: 'FP32 acc 완료', dsize: 8 });
    const b3 = D.box(s, 10.8, 3.0, 2.03, 0.7, 'H의 해당 view', { kind: 'sram', size: 10.5, detail: '직접 SAVE (stride 유지)', dsize: 8 });
    D.arrow(s, b1.r, b1.cy, b3.x, b3.cy, { width: 2, color: C.good });
    D.text(s, 8.2, 3.05, 2.4, 0.3, '직접 SAVE', { size: 9.5, color: C.good, bold: true, align: 'center' });
    D.code(s, 6.0, 3.95, 6.83, 1.4, [
      '# scheduled_0000.py: 출력 compute가 SRAM scope, gemm view가 compute를 직접 겨냥',
      'compute = T.match_buffer(var_compute, (64, 128), "float16", scope="global.sram")',
      'C = T.match_buffer(compute[0:64, i1_1_0*64 : i1_1_0*64+64], (64,64), "float16",',
      '                   strides=("C_s0", 1), scope="global.sram", offset_factor=1)',
      'T.func_attr({"npu.accumulator_store_lowered": 1, "npu.forwarded_output_tiles": 1})',
    ], { size: 8 });
    D.table(s, 0.5, 4.75, 5.2, [
      ['E3 (44 KiB 두 matmul)', '전', '후'],
      ['ISA words', '85', '77'],
      ['DMA bytes', '49,164', '49,164'],
      ['SRAM peak bytes', '42,508', '42,508'],
      ['DRAM image bytes', '65,548', '49,164'],
      ['출력', 'bit-exact', ''],
    ], { colW: [2.2, 1.5, 1.5], size: 9, rowH: 0.27, aligns: ['left', 'right', 'right'] });
    D.callout(s, 6.0, 5.5, 6.83, 0.85, [
      { text: '조건: ', bold: true }, { text: '64×64 FP16 tile을 정확히 하나의 npu_fill_zero와 하나의 npu_gemm_acc view가 쓰고 identity copy 하나만 있을 때. 절약한 8 KiB를 더 큰 W panel에 쓸 수 있어 peak가 그대로인 경우도 있습니다.', size: 9.5 },
    ], { kind: 'faint' });
    D.takeaway(s, 'FP32 누산이 끝난 최종 저장 위치만 바꾸는 것이며, K 중간 partial sum을 H에 내리는 것이 아닙니다 (누산 계약 유지).', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 53 DRAM compaction & residency
  D.add({ section: '8 · 메모리 계획', title: '보관 후 남은 DRAM 예약 제거와 주소 재배치, 그리고 후단 residency forwarding', subtitle: 'SRAM-only 전달이 확정되면 아무 곳에서도 쓰지 않는 pool 전체를 제거하고 나머지 주소를 당깁니다. 두 matmul 예제에서 H의 DRAM pool 16,384 B가 사라지고 출력 gv가 57,344 → 40,960으로 이동합니다.', source: 'npu_compiler/npu_memplan.compact_unused_storage, npu_residency.forward, docs/examples/two_matmuls_recorded/program.json (memory, dram_compaction)',
    notes: 'compact_unused_storage는 parameter leaf, 출력, const_data, literal pool, FP32 1.0, 그리고 모든 GLOAD/GSTORE의 strided span(빈틈 포함)을 반드시 남기고, cell 반올림 범위가 이들과 겹치지 않는 pool만 제거합니다. relocate는 제거된 pool 크기만큼 주소를 빼며 DMA g_addr word를 제자리에서 고치고 plan.address, pool_base, const_data, constant_base, one_fp32_base, top, 출력 offset, views, kernel_records(storage: sram/dram/unused), joint_memory, residency 기록을 갱신합니다. 오른쪽 아래 strip은 두 matmul 예제의 실제 주소입니다. npu_residency.forward는 codegen 뒤 명령 stream 위에서 GSTORE된 SRAM 사본을 추적해 커널 진입 시 GLOAD를 제거하거나 SRAM 복사로 바꾸는 write-through forwarding이며 생산자의 GSTORE는 남깁니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.3, 'compact_unused_storage', { size: 11, bold: true, color: C.navy });
    D.table(s, 0.5, 1.8, 6.4, [
      ['단계', '내용'],
      ['반드시 남김', 'parameter leaf, 출력, const_data, literal pool, FP32 1.0, 모든 DMA의 strided span(빈틈 포함)'],
      ['제거 대상', 'cell 반올림 범위가 위와 겹치지 않는 pool 전체 (내부 부분 제거 없음)'],
      ['relocate', '제거된 pool 크기만큼 아래 주소를 당김'],
      ['갱신 metadata', 'DMA g_addr word, plan.address, pool_base/bytes, const_data, constant_base, one_fp32_base, top, outputs offset, views, kernel_records storage(sram/dram/unused), joint_memory, residency'],
      ['후속', 'saved_bytes > 0 이고 KV 계약이 있으면 AppendPlan 재구성 (주소 기준 변경)'],
    ], { colW: [1.3, 5.1], size: 8.5, rowH: 0.42 });
    D.table(s, 0.5, 4.55, 6.4, [
      ['예', 'before → after', '제거'],
      ['두 matmul (44 KiB)', '65,548 → 49,164 B', 'storage 16,384 B (H의 DRAM 예약)'],
      ['Llama prefill_cache', '6,429,599,864 → 6,429,587,576 B', 'storage61/62 각 6,144 B (작은 중간값 pool)'],
      ['실제 폭 MLP M=64', '154,927,116 → 151,781,388 B', '3 MiB (중간값이 같은 pool을 재사용해 4 MiB 아님)'],
    ], { colW: [1.6, 2.4, 2.4], size: 8.5, rowH: 0.36 });
    D.text(s, 7.15, 1.5, 6, 0.3, 'npu_memory_search vs npu_residency (둘 다 켜져 있음)', { size: 11, bold: true, color: C.navy });
    D.table(s, 7.15, 1.8, 5.68, [
      ['', 'memory search (보관 탐색)', 'residency forwarding'],
      ['실행 시점', 'codegen 전', 'codegen 뒤, peephole 전'],
      ['SRAM 공간', '명시적으로 예약', '예약 없음, 기회주의적'],
      ['커널 재schedule', '함', '안 함'],
      ['생산자 GSTORE', '제거', '유지 (write-through)'],
      ['소비자 GLOAD', '애초에 없음 (SRAM 입력)', '커널 진입 시 완전 포함되면 제거 또는 SRAM→SRAM V_COPY'],
      ['효과 분석', '값 버전/byte 범위', 'WriteEffects: 각 명령의 쓰기 범위; 미지 opcode → cache 전체 무효'],
    ], { colW: [1.3, 2.1, 2.28], size: 8.5, rowH: 0.34 });
    // before/after DRAM strips
    D.text(s, 7.15, 4.7, 5.68, 0.24, '두 matmul 예제의 DRAM image (bytes)', { size: 9.5, bold: true, color: C.navy });
    D.text(s, 7.15, 4.96, 0.6, 0.3, 'before', { size: 7.5, color: C.muted, valign: 'middle' });
    D.membar(s, 7.75, 4.96, 5.08, 0.34, 0, 65548, [
      { from: 0, to: 8192, label: 'x', kind: 'host' }, { from: 8192, to: 24576, label: 'w', kind: 'dram' }, { from: 24576, to: 40960, label: 'v', kind: 'dram' },
      { from: 40960, to: 57344, label: 'storage (H) ✗', kind: 'red' }, { from: 57344, to: 65548, label: 'gv', kind: 'unit' },
    ], { size: 7 });
    D.text(s, 7.15, 5.5, 0.6, 0.3, 'after', { size: 7.5, color: C.muted, valign: 'middle' });
    D.membar(s, 7.75, 5.5, 5.08 * 49164 / 65548, 0.34, 0, 49164, [
      { from: 0, to: 8192, label: 'x', kind: 'host' }, { from: 8192, to: 24576, label: 'w', kind: 'dram' }, { from: 24576, to: 40960, label: 'v', kind: 'dram' },
      { from: 40960, to: 49164, label: 'gv', kind: 'unit' },
    ], { size: 7 });
    D.text(s, 7.75, 5.9, 5.08, 0.4, 'gv 57,344 → 40,960; GSTORE packet의 g_addr word와 plan.outputs.offset_bytes를 함께 갱신. residency.events = [] (보관 탐색이 이미 H 전송을 없앰)', { size: 7.5, color: C.muted });
    D.takeaway(s, 'DMA 49,164 B와 최종 image 49,164 B가 같아진 것은 이 예제의 우연이며 두 지표는 원래 다른 양입니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 54 residual vs KV, limits
  D.add({ section: '8 · 메모리 계획', title: 'Residual과 KV cache는 왜 다르게 취급되는가, 그리고 현재 메모리 계획의 경계', subtitle: 'Residual도 중간값이면 마지막 add까지 보관 후보입니다. KV cache는 다음 프로그램 호출에도 필요하고 ABI 입력/출력으로 관찰되므로 DRAM 상태를 유지합니다.', source: 'docs/current/08_memory.md, npu_memory_search.discover (보호 범위)',
    notes: '64×3072 FP16 residual이 384 KiB라는 사실만으로 반드시 보관됐다고 말할 수 없습니다. 사이의 모든 작업 공간과 비용을 봐야 합니다. KV cache는 kv라는 이름을 찾아 예외 처리하는 것이 아니라 외부 가시성 계약(출력 tree, parameter leaf)으로 보호됩니다. 현재는 빈 구간 재사용과 보관 영역 이동이 있지만 모든 조각화를 해결하는 전역 allocator는 아니며, 남은 낭비가 실제 프로그램 비용에 큰지 확인한 뒤 추가 개선을 검토합니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 7.2, [
      ['값', '보관 후보?', '이유'],
      ['Residual X (attention 전 hidden)', '후보', '중간값. 마지막 add(R = X + attn)까지 SRAM 보관 가능. 384 KiB(M=64)이지만 사이 작업 공간과 비용에 따라 결정'],
      ['MLP 중간 G/S/U/H', '후보', 'FP16, 같은 호출에서 RMW 아님, compact → 실제 폭 MLP에서 보관 적용 기록'],
      ['KV cache (k_cache, v_cache)', '보호', 'ABI 입력 parameter이자 append 대상. 다음 호출에서 관찰 → DRAM 유지'],
      ['최종 출력 (logits, K_new, V_new)', '보호', '출력 tree의 leaf: runtime이 그 주소에서 읽음'],
      ['graph 상수, literal pool', '보호', 'const_data / constant_base'],
      ['alias가 모호한 view, in-place 갱신', '제외', '값 버전을 확신할 수 없음 → DRAM 계약 유지'],
    ], { colW: [2.3, 0.9, 4.0], size: 8.8, rowH: 0.5 });
    D.text(s, 7.95, 1.5, 5, 0.3, '현재 메모리 계획의 경계', { size: 11, bold: true, color: C.navy });
    D.bullets(s, 7.95, 1.85, 4.88, 4.5, [
      ['Whole-tensor 보관만: ', '일부 tile만 남기기, 중간 퇴거, 호출 순서 변경은 없음'],
      ['고정 호출 순서: ', 'Relax binding 순서대로. 재배치로 수명을 줄이는 최적화 없음'],
      ['조각화: ', '빈 구간 재사용과 arena 하향은 있지만 전역 allocator는 아님'],
      ['DRAM 압축: ', '완전히 미사용인 pool만 제거. 사용 중 pool 내부 낭비는 미해결'],
      ['Snapshot: ', '관찰해야 하는 중간값이 있으면 SRAM-only 전달을 제한하거나 끔'],
      ['실패 시: ', '보관하지 못하면 DRAM 경로 → 큰 activation이 곧 컴파일 불가는 아님. 개별 연산의 shape/reduction 한도는 별개'],
    ], { size: 9.5, psa: 5 });
    D.takeaway(s, '"kv"라는 이름이 아니라 외부 가시성 계약으로 보호합니다. 남은 낭비가 실제 프로그램 비용에 큰지 확인한 뒤에 다음 개선을 검토합니다.', { y: 6.5 });
  });
};
