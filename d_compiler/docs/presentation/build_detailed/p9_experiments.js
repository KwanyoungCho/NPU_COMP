'use strict';
const { C, F } = require('./lib');

function barChart(D, s, x, y, w, h, title, labels, series, opts = {}) {
  const data = series.map(sr => ({ name: sr.name, labels, values: sr.values }));
  s.addChart(D.pres.charts.BAR, data, Object.assign({
    x, y, w, h, barDir: 'col', chartColors: series.map((sr, i) => sr.color || ['91A5BA', '1F8A8A', '6C4FBF'][i % 3]),
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 8, dataLabelFormatCode: '#,##0',
    showTitle: true, title, titleFontSize: 10.5, titleColor: C.navy,
    catAxisLabelFontSize: 8.5, valAxisLabelFontSize: 8, catAxisLabelColor: C.ink, valAxisLabelColor: C.muted,
    valGridLine: { color: 'E1E7EF', size: 0.5 }, catGridLine: { style: 'none' }, showLegend: series.length > 1, legendPos: 'b', legendFontSize: 8,
    barGapWidthPct: 60, valAxisLabelFormatCode: '#,##0',
  }, opts));
}

module.exports = (D) => {
  // ---------------------------------------------------------------- 69 compile-time control
  D.add({ section: '10 · 실험 · 한계 · 정리', title: 'Compile 시간은 어떻게 제어되는가: build 안의 cache, 예산, phase 계측', subtitle: 'compilation() context는 build 하나 동안 살아 있으며 nested 호출은 바깥 context를 재사용합니다. 모든 cache는 structural_hash로 찾고 structural_equal로 확인합니다.', source: 'npu_compiler/npu_compilation.py (recipe, trial, probe, region_cost, EntryCache, optimization_budget, phase, describe)',
    notes: 'recipe cache는 PrimFunc 구조(global_symbol/op_pattern 제외)와 profile(joint_memory/beam/graph_search_seconds 제외, sram_bytes 포함)·scratch·slot을 키로 schedule+SRAM layout을 재사용합니다. 28 layers에서 약 32개 커널이 1,206회 호출되어 호출별 scheduling이 link 시간의 64%였던 것이 도입 이유입니다. trial cache는 mapping별 측정(실패도 _RejectedTrial로)을, probe cache는 비용만(승자는 실제 주소로 재emit·검증), region cache는 fusion 영역 비용을, EntryCache(256 MiB words / 32 entries LRU)는 graph 후보 사이에 linked entry 전체를 재사용합니다. 예산은 첫 합법 module 뒤 arm되고 phase 진입마다 검사되며, 초과 시 OptimizationBudgetExceeded가 budget_exhausted로 기록됩니다. progress 이벤트는 phase_begin/end, candidate_begin/end(status), entry_compiled/reused 등입니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 7.6, [
      ['cache', '키', '재사용 대상', 'Llama 기록 (hit / miss)'],
      ['recipe', 'PrimFunc 구조 hash (symbol/op_pattern 제외) + profile(sram_bytes·탐색 설정 포함, joint/beam/graph 시간 제외) + scratch/slot', '커널의 schedule + SRAM layout', '272 / 383'],
      ['trial', '(tile, reuse_sram, scratch, slot, stripe, mapping, tile_pad, capture)', '후보 측정 결과 (실패 포함), 용량은 키 밖', '184 / 248'],
      ['probe (cost)', 'SRAM binding, scratch, 상수, native mode, DRAM 주소 mod 4', '비용만; 승자는 실제 주소로 재emit·검증', '294 / 270'],
      ['region', 'fusion 영역의 entry identity (param 이름 제외)', 'FuseOps 영역 unfused/fused 비용', '0 / 40'],
      ['EntryCache', 'entry identity: private GlobalVar 이름 정규화, npu.repeated_gemv/op_pattern 제거한 module hash', 'graph 후보 사이 linked entry 전체 (LRU 256 MiB / 32)', '0 / 3'],
    ], { colW: [1.1, 3.0, 2.3, 1.2], size: 8.5, rowH: 0.52 });
    D.table(s, 8.35, 1.55, 4.48, [
      ['예산 / 제어', '값'],
      ['graph_search_seconds', '기본 120 / E10 None'],
      ['schedule_max_attempts', '기본 64 / E10 128'],
      ['schedule_space_trials', '기본 8 / E10 16 legal 목표'],
      ['schedule_probe_word_budget', 'max(설정, 기준 words × legal 목표) / E10 10억'],
      ['schedule_search / seed', 'space (기본) · meta_schedule = ReplayFunc 추첨, seed 0'],
      ['schedule_database', '선택형 trace DB (reuse / tune). E10 미사용'],
      ['early abort', 'entry 하나가 incumbent보다 나쁘면 후보 중단'],
      ['joint memory', 'baseline보다 싸야 채택'],
    ], { colW: [2.1, 2.38], size: 8.5, rowH: 0.32 });
    D.text(s, 0.5, 4.85, 6, 0.3, 'progress 이벤트와 phase', { size: 10.5, bold: true, color: C.navy });
    D.bullets(s, 0.5, 5.15, 7.6, 1.2, [
      ['이벤트: ', 'phase_begin/end(seconds), optimization_budget_started, candidate_begin/end(status: duplicate/legal/noncompetitive/unsupported/budget_exhausted), entry_compiled/entry_reused'],
      ['phase: ', 'schedule_and_sram, schedule_tir, schedule_sram, schedule_probe_*, entry_identity/compile/statistics, memory_planning, joint_schedule_memory, native_codegen, residency, peephole, dram_compaction, fusion_regions, program_cost, meta_schedule_replay / generate / extract_tasks (DB·ReplayFunc 사용 시)'],
      ['describe(): ', 'hit/miss, inclusive·self seconds, 호출 수, native fallbacks/rejections, meta_schedule(enabled/mode/hits/misses/writes/rejections) → plan.compilation (bundle에 기록)'],
    ], { size: 9, psa: 3 });
    D.callout(s, 8.35, 4.85, 4.48, 1.5, [
      { text: 'Shape만 같으면 같은 결과로 취급하는 cache가 아닙니다. ', bold: true }, { text: '구조·profile·입출력 배치가 키에 들어가고, 반복 layer에서도 실제 weight 주소가 들어간 명령은 호출별로 확인합니다.', size: 9.5 },
    ], { kind: 'warn' });
  });

  // ---------------------------------------------------------------- 70 evidence levels
  D.add({ section: '10 · 실험 · 한계 · 정리', title: '실험 근거의 세 수준과 서로 다른 지표', subtitle: '모든 숫자를 하나의 성능 점수로 합치지 않습니다. 컴파일 시간, 장치가 요청한 전송량, 명령 크기, 저장 공간, 수치 결과를 따로 측정합니다.', source: 'docs/current/11_experiments.md, docs/evidence/README.md, examples/reproduction.json',
    notes: '새 실행 검증은 이번 문서 작업에서 현재 소스로 작은 입력을 실제 실행한 것(E1, E2)이고, 보존된 실행 검증은 기존 로그·JSON·source hash를 그대로 보존한 것(E3, E6, E7)이며, 정적 분석/기존 단일 측정은 명령·배치 분석이나 한 번의 compile wall time(E4, E5)입니다. 작은 예제와 기존 최종 검증의 compiler source SHA-256은 44694fc6…로 같습니다. 중간 개선 단계 기록은 source hash가 다르며 현재 소스의 새 benchmark로 바꾸지 않았습니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 12.33, [
      ['표시', '의미', '이 발표의 예', 'source hash'],
      ['새 실행 검증', '이번 문서 작업에서 현재 소스로 작은 입력을 실제 실행 (C-model + LLVM)', 'E1 pipeline (fusion off/auto), E2 SRAM 용량 5구성', '44694fc6…'],
      ['보존된 실행 검증', '기존 실행 로그·JSON·source hash를 그대로 보존', 'E3 두 matmul 전후, E6 전체 Llama 3 tokens, E7 회귀 gate 17 suites', '44694fc6… (E3는 42843f7e…)'],
      ['정적 분석 / 단일 측정', '명령·배치 분석 또는 한 번의 compile wall time (executed=false)', 'E4 실제 폭 MLP, E5 compile 시간 비교', '7d13cfd9… 등 (개선 단계별로 다름)'],
    ], { colW: [1.7, 4.4, 3.8, 2.43], size: 9, rowH: 0.5 });
    D.table(s, 0.5, 3.7, 7.4, [
      ['지표', '뜻', '무엇이 아닌가'],
      ['DMA bytes', '정적 명령 해독의 GLOAD/GSTORE 이동량 (C-model counters와 대조)', 'host patch, simulator 파일 I/O'],
      ['ISA words', '32-bit 명령 저장 공간', '실행 cycle, 명령 수(instructions)'],
      ['SRAM peak', '커널 top의 최댓값 (pool + scratch + buffers)', '살아 있는 값 크기의 합'],
      ['DRAM image', 'plan.top × unit_bytes (weight + 입력 + 중간 pool + 출력)', 'DMA 전송량 (우연히 같을 수 있음)'],
      ['compile seconds', '단일 wall-clock 관측 (다른 작업과 동시 실행 가능)', '반복 평균, cold compile 보장'],
    ], { colW: [1.4, 3.6, 2.4], size: 8.8, rowH: 0.4 });
    D.table(s, 8.15, 3.7, 4.68, [
      ['provenance 기록 항목', ''],
      ['compiler_source_sha256', '모든 .py/.cc/.h'],
      ['tvm_version / commit', '0.19.0 / c4dc0c29'],
      ['llvm_version', '18.1.8'],
      ['native_sha256', 'libnpu_codegen.so'],
      ['cmodel_source_sha256', '_poc/mysim_v09.cpp'],
      ['contract_sha256', 'v09_contract.json'],
      ['python / numpy / torch', '3.11.15 / 2.4.6 / 2.12.0+cpu'],
    ], { colW: [2.2, 2.48], size: 8.5, rowH: 0.28 });
    D.takeaway(s, 'ISA words나 simulator wall time으로 하드웨어 latency를 주장하지 않습니다. 실장치 timing이 없을 때는 words/DMA 감소까지만 말합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 71 E1
  D.add({ section: '10 · 실험 · 한계 · 정리', title: 'E1. Fusion은 DMA가 같아도 명령과 작업 공간을 줄일 수 있다', subtitle: 'SiLU(X[64,128] @ W[64,128].T + bias[64]), seed 914, FP16. 현재 public build로 fusion off/auto를 각각 컴파일해 같은 값으로 C-model을 실행했습니다. 두 구성 모두 SRAM 전달 기능을 사용합니다.', source: 'docs/examples/pipeline/stage_summary.json (새 실행 검증)',
    notes: 'off는 3커널(matmul 43 words, add 391 words, silu 2,373 words), auto는 1커널 1,572 words입니다. DMA는 41,100 B로 같고(입력·weight·bias GLOAD 4개 + 출력 GSTORE 1개), words 2,812→1,577, SRAM peak 50,700→42,508 B입니다. Fusion 전후 FP16 비트가 같고 동일 수치 grouping의 LLVM 출력과 최대 절대차는 0입니다. C-model DMA counters(dma_cells_loaded 8227, stored 2048)가 정적 해독과 일치했습니다. 이 사례가 모든 fusion의 이득을 대표하지는 않습니다.' },
  (s, D) => {
    barChart(D, s, 0.5, 1.5, 4.0, 2.6, 'ISA words', ['fusion off (3 kernels)', 'fusion auto (1 kernel)'], [{ name: 'words', values: [2812, 1577], color: '1F8A8A' }], { showLegend: false });
    barChart(D, s, 4.65, 1.5, 4.0, 2.6, 'SRAM peak (bytes)', ['fusion off', 'fusion auto'], [{ name: 'sram', values: [50700, 42508], color: '6C4FBF' }], { showLegend: false });
    barChart(D, s, 8.8, 1.5, 4.0, 2.6, 'DMA bytes (동일)', ['fusion off', 'fusion auto'], [{ name: 'dma', values: [41100, 41100], color: 'D98A1E' }], { showLegend: false });
    D.table(s, 0.5, 4.3, 7.4, [
      ['off의 커널별 words', 'matmul 43', 'add 391', 'silu 2,373', '합 2,812 (+ 상수 GLOAD 4 + HALT 1)'],
      ['auto', 'fused_matmul_add_silu 1,572', '', '', '합 1,577'],
      ['C-model counters', 'gload 4, gstore 1', 'loads 644 → 388', 'saves 449 → 258', 'vector_ops 512 → 288, matrix_ops 2'],
    ], { colW: [1.6, 1.8, 1.2, 1.2, 1.6], size: 8.5, rowH: 0.36, header: false });
    D.table(s, 8.15, 4.3, 4.68, [
      ['검증', '결과'],
      ['fused vs unfused FP16 출력', 'bit-exact'],
      ['LLVM reference (K=64 grouping)', '최대 절대차 0'],
      ['DMA counters vs 정적 해독', '일치'],
      ['compile seconds', 'off 1.09, auto 1.51 (단일 관측)'],
    ], { colW: [2.4, 2.28], size: 8.5, rowH: 0.3 });
    D.callout(s, 0.5, 5.55, 12.33, 0.8, [
      { text: '왜 words가 줄었나: ', bold: true }, { text: 'add·SiLU가 같은 SRAM tile에서 이어져 descriptor 설정(0x80 1,171→606)과 LOAD/SAVE(644/449→388/258)가 줄고, 축 결합으로 vector 행이 64→128 폭이 되어 명령 수가 절반(0x16/0x0f 64→32)이 되었습니다. DMA는 입력·출력이 그대로라 같습니다.', size: 10 },
    ], { kind: 'faint' });
    D.takeaway(s, '전송량은 같아도 명령과 작업 공간이 줄어드는 사례입니다. 이득의 형태는 예제마다 다르므로 실제 비용으로 확인합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 72 E2
  D.add({ section: '10 · 실험 · 한계 · 정리', title: 'E2. SRAM을 늘리면 언제 중간 DRAM 왕복이 사라지는가', subtitle: 'X[64,64] @ W1[64,128] @ W2[128,64], seed 928, fusion off. 보관 탐색 옵션과 SRAM 용량만 바꾸어 5개 구성을 실제 실행했습니다. 모든 구성이 같은 FP16 출력이며 DMA counters가 일치했습니다.', source: 'docs/examples/memory/measurements.json (새 실행 검증)',
    notes: '32 KiB에서는 보관 옵션을 켜도 첫 matmul(X 8 KiB + W1 16 KiB + H 16 KiB = 40 KiB + pool)이 들어가지 않아 선택 결과가 같습니다. 44 KiB에서 보관을 켜면 H 16 KiB의 저장·재읽기 합계 32,768 B가 사라져 DMA 81,932→49,164 B, words 87→77, image 65,548→49,164 B가 되고 SRAM peak는 34,316→42,508 B로 늘어납니다. 64 KiB로 늘려도 이 예제에는 추가 이득이 없습니다. 보관 탐색 off에도 기존 후단 residency/peephole은 유지되므로 모든 메모리 최적화를 켜고 끈 실험이 아닙니다.' },
  (s, D) => {
    barChart(D, s, 0.5, 1.5, 6.1, 3.0, 'DMA bytes', ['32K off', '32K on', '44K off', '44K on', '64K on'], [{ name: 'DMA bytes', values: [81932, 81932, 81932, 49164, 49164], color: 'D98A1E' }], { showLegend: false });
    barChart(D, s, 6.75, 1.5, 6.08, 3.0, 'ISA words', ['32K off', '32K on', '44K off', '44K on', '64K on'], [{ name: 'words', values: [88, 88, 87, 77, 77], color: '1F8A8A' }], { showLegend: false });
    D.table(s, 0.5, 4.65, 12.33, [
      ['SRAM 용량', 'whole-tensor 보관 탐색', 'DMA bytes', 'words', '실제 SRAM peak', 'DRAM image', '해석'],
      ['32 KiB', 'off', '81,932', '88', '26,124', '65,548', 'H를 DRAM에 저장·재읽기'],
      ['32 KiB', 'on', '81,932', '88', '26,124', '65,548', '보관 불가: 첫 matmul이 들어가지 않음 → 동일 선택'],
      ['44 KiB', 'off', '81,932', '87', '34,316', '65,548', '후단 residency/peephole만'],
      ['44 KiB', 'on', '49,164', '77', '42,508', '49,164', 'H 32,768 B 왕복 제거, DRAM pool 제거'],
      ['64 KiB', 'on', '49,164', '77', '42,508', '49,164', '추가 이득 없음'],
    ], { colW: [1.0, 2.1, 1.2, 0.8, 1.4, 1.3, 4.53], size: 8.8, rowH: 0.27, aligns: ['left', 'left', 'right', 'right', 'right', 'right', 'left'] });
    D.takeaway(s, '"SRAM을 최대한 채우기"보다 실제 재사용과 작업 공간의 관계가 중요합니다. 용량이 문턱을 넘는 순간에만 왕복이 사라집니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 73 E3/E4
  D.add({ section: '10 · 실험 · 한계 · 정리', title: 'E3 · E4. 직접 SRAM 출력과 미사용 DRAM 제거: 작은 예제와 실제 폭 MLP', subtitle: 'E3는 44 KiB 두 matmul에서 이미 SRAM 보관을 사용하는 상태로 pointwise 결합·출력 직접 저장·미사용 DRAM 제거를 비활성/활성한 보존 비교. E4는 M=64 실제 폭 MLP의 정적 compile 기록(executed=false).', source: 'docs/evidence/current/refinement.json, mlp64.json, examples/two_matmuls_recorded/program.json',
    notes: 'E3: words 85→77, DMA 49,164 그대로, SRAM peak 42,508 그대로, image 65,548→49,164, bit-exact. E2의 보관 off/on(87→77)과 다른 비교이므로 87→77과 85→77을 섞지 않습니다. E4: 선택 커널 5개(gate→SiLU→up→multiply→down) 그대로, words 143,404→140,594, DMA 152,174,604 그대로, SRAM peak 7,446,540 그대로, image 154,927,116→151,781,388. 서로 다른 중간값이 같은 기본 DRAM pool을 재사용하므로 제거량은 4 MiB가 아니라 3 MiB입니다. 과거 fused epilogue 명령이 매우 길었던 문제를 줄였어도 최종 선택이 fusion으로 바뀌지는 않았습니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.3, 'E3. 44 KiB 두 matmul (보존, 실행 bit-exact)', { size: 11, bold: true, color: C.navy });
    D.table(s, 0.5, 1.85, 6.0, [
      ['항목', '개선 전', '개선 후'],
      ['ISA words', '85', '77'],
      ['DMA bytes', '49,164', '49,164'],
      ['SRAM peak bytes', '42,508', '42,508'],
      ['DRAM image bytes', '65,548', '49,164'],
      ['출력', 'bit-exact', ''],
    ], { colW: [2.4, 1.8, 1.8], size: 9.5, rowH: 0.32, aligns: ['left', 'right', 'right'] });
    D.text(s, 6.85, 1.5, 6, 0.3, 'E4. 실제 폭 MLP M=64 (정적 compile, executed=false)', { size: 11, bold: true, color: C.navy });
    D.table(s, 6.85, 1.85, 5.98, [
      ['항목', '개선 전', '개선 후'],
      ['선택 커널 수', '5', '5'],
      ['ISA words', '143,404', '140,594'],
      ['DMA bytes', '152,174,604', '152,174,604'],
      ['SRAM peak bytes', '7,446,540', '7,446,540'],
      ['DRAM image bytes', '154,927,116', '151,781,388'],
    ], { colW: [2.2, 1.9, 1.88], size: 9.5, rowH: 0.32, aligns: ['left', 'right', 'right'] });
    barChart(D, s, 0.5, 4.05, 6.0, 2.3, 'E3 words / image bytes ÷ 1000', ['words 전', 'words 후', 'image KB 전', 'image KB 후'], [{ name: 'v', values: [85, 77, 65.548, 49.164], color: '6C4FBF' }], { showLegend: false, dataLabelFormatCode: '#,##0.0' });
    D.bullets(s, 6.85, 4.05, 5.98, 2.3, [
      ['무엇이 바뀌었나: ', 'pointwise 결합, 출력 tile 직접 저장(H view + stride), 미사용 DRAM pool 제거'],
      ['E4의 3 MiB: ', 'G/S/U/H가 같은 기본 pool을 재사용하므로 4 MiB가 아니라 3 MiB 제거'],
      ['E4의 선택: ', 'gate → SiLU → up → multiply → down 5커널 유지, 중간값은 SRAM 전달'],
      ['혼동 주의: ', 'E2의 87→77(보관 on/off)과 E3의 85→77(직접 저장·정리)은 다른 비교'],
      ['fusion: ', 'fused epilogue words를 줄였어도 최종 선택은 여전히 분리 커널 (커널 수가 아니라 비용으로 판단)'],
    ], { size: 9.5, psa: 4 });
    D.takeaway(s, 'DMA와 SRAM peak가 그대로여도 명령과 image가 줄어드는 개선이 있습니다. 지표를 따로 보아야 어떤 최적화가 무엇을 바꿨는지 알 수 있습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 74 E5
  D.add({ section: '10 · 실험 · 한계 · 정리', title: 'E5. Compile 속도 개선은 코드 품질을 바꾸지 않았다 (binary hash 동일)', subtitle: '기존에 기록한 단일 wall-clock 관측값. 일부 측정 중 다른 검증도 실행됐습니다. 반복 평균이나 하드웨어 inference latency가 아닙니다.', source: 'docs/evidence/current/compile_comparison.json',
    notes: 'M=7 실제 폭 MLP: 중복 실패 검사·시도 제한 개선으로 84.163 s → 22.073 s, TIR trial 생성 347→112회, 143,883 words와 binary hash 동일. M=64: 무제한 시도 → 64회 제한에서 21.791 s → 21.958 s, 두 경우 모두 47회 trial이므로 제한이 실제로 작동하지 않았고 140,594 words hash 동일. 모든 입력에서 일정 배율로 빨라진다는 결론은 낼 수 없습니다. 주요 개선은 invalid accumulator schedule을 native/Python에서 두 번 검사하지 않기, 동일 entry/region 비용 재사용, 실패도 포함하는 탐색 한도입니다. 전체 검증의 948.993 s는 cold compile 시간이 아닙니다(fresh_full_build_seconds=null).' },
  (s, D) => {
    barChart(D, s, 0.5, 1.5, 6.0, 3.0, 'M = 7 (같은 binary, 143,883 words)', ['instrumented', 'invalid_fast', 'bounded'], [{ name: 'seconds', values: [84.163, 35.570, 22.073], color: '1F8A8A' }], { showLegend: false, dataLabelFormatCode: '0.0" s"' });
    barChart(D, s, 6.75, 1.5, 6.08, 3.0, 'M = 64 (같은 binary, 140,594 words)', ['unbounded (47 trials)', 'bounded 64 (47 trials)'], [{ name: 'seconds', values: [21.791, 21.958], color: '91A5BA' }], { showLegend: false, dataLabelFormatCode: '0.0" s"', valAxisMaxVal: 90 });
    D.table(s, 0.5, 4.65, 12.33, [
      ['label', 'shape (M, K, N)', 'seconds', 'words', 'trial misses', 'word sha256', '비고'],
      ['instrumented_mlp7', '7 × 3072 × 8192', '84.163', '143,883', '347', 'c63b82a2…', '개선 전'],
      ['invalid_fast_mlp7', '7 × 3072 × 8192', '35.570', '143,883', '347', 'c63b82a2…', 'invalid accumulator schedule 중복 검사 제거'],
      ['bounded_mlp7', '7 × 3072 × 8192', '22.073', '143,883', '112', 'c63b82a2…', '실패 포함 시도 제한'],
      ['unbounded / bounded_mlp64', '64 × 3072 × 8192', '21.791 / 21.958', '140,594', '47 / 47', 'ec55566f…', '제한이 작동하지 않은 경우 (시간 동일)'],
    ], { colW: [2.0, 1.7, 1.3, 1.1, 1.1, 1.3, 3.83], size: 8.5, rowH: 0.3, aligns: ['left', 'left', 'right', 'right', 'right', 'left', 'left'] });
    D.takeaway(s, '전체 build의 기본 120초는 최초 합법 module 이후의 추가 탐색 예산이며, 전체 검증의 948.99 s는 완료된 첫 entry를 재사용한 값이라 cold compile 시간이 아닙니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 75 E6
  D.add({ section: '10 · 실험 · 한계 · 정리', title: 'E6. 전체 Llama 3.2 3B 출력 검증과 프로그램 규모', subtitle: 'FP16, 28 layers, prompt 7 tokens, capacity 9, greedy 3 tokens. HF와 argmax가 모두 일치하고 logits cosine ≥ 0.99999입니다.', source: 'docs/evidence/current/llama_validation.json, llama_selection.json, completion.json',
    notes: '기본 entry 통계: prefill_cache 8,582,352 words / DMA 6,457,527,556 B / SRAM peak 8,374,292 B / image 6,429,587,576 B, decode base 7,310,801 words / 6,438,980,676 B / 7,342,100 B / 6,427,349,032 B. Decode 실제 실행에는 append tail이 추가되므로 base 통계와 tail 포함 counters를 혼합하지 않습니다. DMA의 대부분은 layer마다 weight 192 MiB × 28 + LM head 751.5 MiB ≈ 6 GB를 panel로 읽는 것입니다. graph 탐색은 시간 한도에 도달해 unfused 기준을 선택했으며 "모든 후보 중 전역 최적"이라는 의미는 없습니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 6.6, [
      ['실행', 'HF와 같은 argmax', 'logits cosine', '최대 절대차'],
      ['prefill', '358', '0.9999977148', '0.03515625'],
      ['decode position 7', '2846', '0.9999958473', '0.02343750'],
      ['decode position 8', '4560', '0.9999929546', '0.03906250'],
    ], { colW: [1.8, 1.6, 1.7, 1.5], size: 9.5, rowH: 0.34, aligns: ['left', 'right', 'right', 'right'] });
    D.table(s, 0.5, 3.15, 6.6, [
      ['기본 entry', 'words', 'DMA bytes', 'SRAM peak', 'DRAM image'],
      ['prefill_cache', '8,582,352', '6,457,527,556', '8,374,292', '6,429,587,576'],
      ['decode (base)', '7,310,801', '6,438,980,676', '7,342,100', '6,427,349,032'],
    ], { colW: [1.4, 1.1, 1.5, 1.2, 1.4], size: 8.8, rowH: 0.34, aligns: ['left', 'right', 'right', 'right', 'right'] });
    D.text(s, 7.35, 1.5, 5.5, 0.3, 'prefill_cache opcode 구성 (words 기준 상위)', { size: 10.5, bold: true, color: C.navy });
    s.addChart(D.pres.charts.BAR, [{ name: 'count', labels: ['0x80 ADDR', '0x90 LOAD', '0x42 M_MUL', '0x98 SAVE', '0x17 COPY', '0x15 BCAST', '0x82 VLEN', '0x0a MUL', '0x01 ADD', 'DMA (a0+a8)'], values: [4385992, 2007966, 787008, 521044, 322764, 237402, 201292, 25541, 11621, 13581] }], {
      x: 7.35, y: 1.8, w: 5.48, h: 3.1, barDir: 'bar', chartColors: ['6C4FBF'], showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 7.5, dataLabelFormatCode: '#,##0',
      showTitle: false, catAxisLabelFontSize: 8, valAxisLabelFontSize: 7.5, catAxisLabelColor: C.ink, valAxisLabelColor: C.muted, valGridLine: { color: 'E1E7EF', size: 0.5 }, catGridLine: { style: 'none' }, showLegend: false, catAxisOrientation: 'maxMin', valAxisLabelFormatCode: '#,##0',
    });
    D.bullets(s, 0.5, 4.35, 6.6, 1.9, [
      ['DMA 6.46 GB의 구성: ', 'weight 192 MiB × 28 layers + LM head 751.5 MiB ≈ 6.43 GB를 panel로 읽는 것이 대부분. 저장은 17.4 MB'],
      ['words 8.58M: ', '주소 descriptor(0x80) 51%, LOAD 23%, M_MUL 787,008 (28 layers의 모든 matmul + LM head) 등 정적 전개'],
      ['graph 탐색: ', '120 s 예산 소진 → unfused 기준 module. "모든 fusion이 더 나쁘다"의 증거 아님'],
      ['decode 실행: ', 'base 통계 + append tail. 두 통계를 섞지 않음'],
    ], { size: 9.5, psa: 4 });
    D.callout(s, 7.35, 5.05, 5.48, 1.3, [
      { text: '판정: ', bold: true }, { text: '유한 logits, token 일치, cosine ≥ 0.999. 최대 절대차는 기록만 하고 threshold로 쓰지 않았습니다. 한 prompt의 짧은 greedy 생성이므로 긴 문맥·다양한 prompt·양자화 generation은 별도 검증 범위입니다.', size: 9.5 },
    ], { kind: 'warn' });
    D.takeaway(s, '수치 정확성(HF 대비)과 프로그램 비용(words/DMA/SRAM)은 별개의 표입니다. 두 표가 같은 source hash 44694fc6…를 가리킵니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 76 E7
  D.add({ section: '10 · 실험 · 한계 · 정리', title: 'E7. 기능 검증 gate: 17개 suite가 확인하는 축', subtitle: '기존 최종 gate는 17개 검사 묶음이 실패/skip 없이 완료되었습니다. 묶음 안에 여러 assertion이 있으므로 "17개 shape만 테스트했다"는 뜻이 아닙니다. 이번 문서 작업에서는 재실행하지 않았습니다.', source: 'docs/evidence/current/regression_validation.json, gate_logs/, d_compiler/run_gate.sh',
    notes: 'run_gate.sh는 실제 subprocess exit code, timeout, skip을 validation.json으로 기록하며 [PASS] 문자열만 찾아 성공으로 판정하지 않습니다. 가장 긴 suite는 nn_decode(207 s)이고 pipeline_extensions 68.6 s, compiler_contracts 44.6 s, priority_contracts 37.3 s, npu_target 34.8 s입니다. 선택 검사는 --tests로 일부 suite만, --require-no-skips로 skip 금지, --output으로 로그 위치를 지정합니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 7.6, [
      ['검증 축', '대표 검사 묶음', '확인할 의미'],
      ['표준 pipeline과 ABI', 'architecture_contracts, compiler_contracts, npu_target', 'entry, output tree, 미지원 입력 거절'],
      ['Schedule · 메모리', 'schedule_memory, sram_lifetime, npu_memplan', '보관, 주소, 수명 충돌, fallback'],
      ['Fusion · 수치 · 확장', 'pipeline_extensions, optimization_contracts, priority_contracts', '후보 지원 조건과 결과'],
      ['명령 생성', 'native_codegen, npu_link', 'Python/native 계약과 연결'],
      ['회귀 개선', 'compiler_refinement, compile_speed, completion_contracts', '출력 직접 저장·정리·cache·예산'],
      ['Frontend · 생성', 'nn_hf, nn_decode', '모델 의미와 cached decode'],
      ['부록 기능', 'npu_quantize', '작은 양자화 kernel/model'],
    ], { colW: [1.8, 3.4, 2.4], size: 9, rowH: 0.42 });
    s.addChart(D.pres.charts.BAR, [{ name: 'seconds', labels: ['nn_decode', 'pipeline_extensions', 'compiler_contracts', 'priority_contracts', 'npu_target', 'native_codegen', 'optimization_contracts', 'schedule_memory', 'npu_quantize', 'compiler_refinement', 'nn_hf', 'npu_link', 'architecture_contracts', 'completion_contracts', 'sram_lifetime', 'compile_speed', 'npu_memplan'], values: [207.3, 68.6, 44.6, 37.3, 34.8, 25.6, 20.4, 18.8, 18.9, 14.3, 14.0, 9.3, 9.1, 8.9, 6.7, 3.5, 2.4] }], {
      x: 8.3, y: 1.5, w: 4.53, h: 4.8, barDir: 'bar', chartColors: ['1F8A8A'], showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 7, dataLabelFormatCode: '0.0',
      showTitle: true, title: 'suite별 wall seconds (exit 0, 17/17 passed)', titleFontSize: 9.5, titleColor: C.navy,
      catAxisLabelFontSize: 7.5, valAxisLabelFontSize: 7, catAxisLabelColor: C.ink, valAxisLabelColor: C.muted, valGridLine: { color: 'E1E7EF', size: 0.5 }, catGridLine: { style: 'none' }, showLegend: false, catAxisOrientation: 'maxMin', valAxisMaxVal: 250,
    });
    D.code(s, 0.5, 5.15, 7.6, 1.2, [
      'bash d_compiler/run_gate.sh --tests architecture_contracts schedule_memory native_codegen \\',
      '     --require-no-skips --output /tmp/npu-gate',
      '# validation.json: 실제 subprocess exit, timeout, skip 기록. "[PASS]" 문자열 검색으로 판정하지 않음',
      'python d_compiler/docs/examples/reproduce.py --out /tmp/npu-doc-examples   # E1/E2 재현 (작은 compile 7개 + C-model)',
    ], { size: 8.5 });
    D.takeaway(s, 'Native/Python 일치와 gate 통과는 계약 준수의 근거입니다. 모델 수식 자체의 오류는 HF/NumPy·LLVM 기준(E6, run_nn_npu.py)으로 따로 확인합니다.', { y: 6.5 });
  });
  require('./search_update')(D);

};
