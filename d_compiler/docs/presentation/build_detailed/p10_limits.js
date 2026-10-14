'use strict';
const { C, F } = require('./lib');

module.exports = (D) => {
  // ---------------------------------------------------------------- 77 support boundary
  D.add({ section: '10 · 실험 · 한계 · 정리', title: '현재 지원 경계: 무엇을 받아들이고 무엇을 거절하는가', subtitle: 'ABI verifier는 정의·호출·배치의 일관성을 검사하지만 임의 TIR의 모든 index 접근이 안전함을 증명하는 정리 증명기는 아닙니다. 지원 subset 밖의 계산을 조용히 통과시키지 않습니다.', source: 'docs/current/12_operations_limits.md, reference/execution_contract.md',
    notes: '정적 shape만 지원하며 FP16 matmul은 rank≥2, batch broadcast, unit/ragged 축을 처리하고 rank-1은 명시적 reshape가 필요합니다. control flow는 지원된 정적 호출을 직선 stream으로 전개하며 임의 dynamic control flow나 packed call은 거절합니다. softmax는 마지막 축 1..65504, RMSNorm은 마지막 축·FP16·정적 width·지원 epsilon, attention full-fusion은 정적 rank-3 FP16 64배수, RoPE는 default/linear/llama3, cache는 FP16 정적 capacity와 additive mask, runtime은 C-model adapter입니다. 보장하지 않는 것: 실장치 latency, 모든 동적 shape, 모든 TVM op, 모든 NaN/Inf edge semantics, 모든 HF 모델, 자동 mixed precision, quantized KV, key online softmax, 임의 inter-kernel tile streaming.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 12.33, [
      ['항목', '지원 또는 제한'],
      ['Shape', '정적 tensor shape. FP16 matmul rank ≥ 2, batch broadcast, unit/ragged 축 (pad_einsum). rank-1은 명시적 reshape 필요'],
      ['Control flow', '지원된 정적 호출을 직선 stream으로 전개. 임의 dynamic control flow / packed call은 거절'],
      ['I/O', 'tensor와 nested tuple, dtype별 byte 크기, typed output tree (leaf별 shape/dtype/offset)'],
      ['View', '지원되는 연속 reshape / axis-0 split / offset view. 임의 strided slice나 모든 INT8 byte offset이 zero-copy는 아님'],
      ['FP16 수치', 'matmul/sum FP32 누산과 정의된 FP16 저장 경계. 범용 FP32 vector tensor 실행은 별도 범위'],
      ['Softmax / RMSNorm', '마지막 축, 정적 길이 1..65504, 유한 logits 전제 / 마지막 축, FP16 입력·weight, 정적 양의 width, 2⁻⁴⁸ ≤ ε ≤ 65504²'],
      ['Attention', '지원되는 정적 FP16 layout / bias / scale / causal / window. Full-fusion은 rank-3 64배수 조건으로 더 좁음, online softmax 없음'],
      ['RoPE / GQA', 'default / linear / llama3의 지원 필드만. 다른 설정은 확인 없이 허용하지 않음. GQA는 repeat 또는 broadcast matmul 후보'],
      ['Cache', 'FP16 정적 capacity, additive mask [1,1,C+1], versioned append metadata, 선택적 bucket'],
      ['Runtime', 'C-model adapter와 image/file patch. RTL / 실장치 runtime 검증 아님'],
      ['Snapshot', '관찰해야 하는 중간값 때문에 SRAM-only 전달을 제한하거나 끔'],
      ['양자화', 'W8A16 / W8A8 pass·kernel 구현, 작은 kernel bit equality. 전체 generation에는 미연결 (부록 A2)'],
    ], { colW: [1.9, 10.43], size: 9, rowH: 0.36 });
    D.takeaway(s, '보장하지 않는 것: 실장치 latency, 모든 동적 shape·TVM op·HF 모델, NaN/Inf edge semantics, 자동 mixed precision, quantized KV, key online-softmax, 임의 inter-kernel tile streaming.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 78 options & API
  D.add({ section: '10 · 실험 · 한계 · 정리', title: '옵션과 API: 무엇을 켜고 끄며, 오해하기 쉬운 점은 무엇인가', subtitle: 'npu_target.build(module, func_name, fusion, residency, profile) 와 NpuProfile 필드. 각 옵션은 한 단계만 바꾸며 다른 최적화를 함께 끄지 않습니다.', source: 'docs/current/12_operations_limits.md, npu_compiler/npu_target.py, device.py',
    notes: 'fusion auto는 합법 기준과 여러 graph 후보를 비교하지만 전역 최적을 보장하지 않습니다. fusion off는 attention/generic fusion만 끄고 메모리 최적화는 유지합니다. fusion on은 pattern/generic fusion을 요청하며 지원되지 않는 복합 kernel은 실패할 수 있습니다. joint_memory는 중간 FP16 tensor 전체 보관과 schedule을 함께 선택하고, reuse_sram은 커널 내부 수명 재사용(USMP)입니다. graph_search_seconds는 전체 compile timeout이 아니고, schedule_max_attempts는 64개의 성공 후보를 보장하지 않습니다. cache_buckets는 미리 만든 capacity별 decode 선택이지 매 step 임의 shape의 동적 실행이 아닙니다. schedule_search/schedule_seed/schedule_probe_word_budget과 compilation(schedule_database=…, schedule_database_mode=…)는 09-29에 추가·정리된 항목이며 기본 동작(space, DB 없음)을 바꾸지 않습니다. DB는 중첩 context에서 지정할 수 없고 버전 fingerprint별 하위 디렉터리를 씁니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 7.6, [
      ['옵션', '기본 동작', '오해하기 쉬운 점'],
      ['fusion = auto', '합법 기준 + graph 후보 비교 (5장)', '전역 최적 보장 아님'],
      ['fusion = off', 'attention / generic fusion을 끔', '메모리 최적화까지 전부 끄는 것은 아님'],
      ['fusion = on', 'pattern / generic fusion 요청', '지원되지 않는 복합 kernel은 실패 가능'],
      ['joint_memory (True)', '중간 FP16 tensor 전체 보관 + schedule 동시 선택', '부분 tile 보관이나 runtime eviction 아님'],
      ['reuse_sram (True)', '커널 내부 수명에 따른 USMP 공간 재사용', '커널 사이 보관과 다른 옵션'],
      ['residency (True)', '후단 versioned SRAM forwarding + joint search 활성', 'compile_program 기본은 False, public build는 True'],
      ['graph_search_seconds', '기본 120초 / None 해제', 'E10은 None, 전체 timeout 아님'],
      ['schedule_max_attempts = 64', '성공/실패/용량 초과 포함 시도 제한', '64개의 성공 후보 보장 아님. None = 무제한'],
      ['schedule_search = "space"', '추정 DMA 순 탐색 (기본)', '"meta_schedule"은 ReplayFunc 추첨, "legacy"는 기준 규칙만. 학습형 탐색 아님'],
      ['schedule_probe_word_budget = 100,000', 'max(설정, 기준 words × legal 목표)', '설정값이 항상 상한은 아님. 후보 완료 후 검사'],
      ['schedule_database (compilation)', 'trace DB 저장·재사용, mode reuse / tune', '기본 꺼짐. 최외곽 context에서만, 버전별 디렉터리'],
      ['cache_buckets', '미리 만든 capacity별 decode 선택', '매 step 임의 shape의 동적 실행 아님'],
      ['NPU_NATIVE / NPU_CODEGEN_SO', 'native codegen 사용 여부 / 경로', '결과 words는 compare 모드로 동일성 검사'],
    ], { colW: [2.35, 2.7, 2.55], size: 8.2, rowH: 0.33 });
    D.code(s, 8.35, 1.55, 4.48, 2.9, [
      'from npu_compiler import npu_target',
      'from npu_compiler.npu_compilation import compilation',
      '',
      'with compilation(progress=lambda e: print(e)):',
      '    exe = npu_target.build(module,',
      '            func_name="prefill", fusion="auto")',
      '',
      '# values: 최종 signature 이름 → 배열',
      'output, counters = exe.run(values)',
      'exe.export_bundle("/tmp/npu-bundle", values)',
      '',
      '# 선택형 trace DB (기본 꺼짐, mode: reuse | tune)',
      'with compilation(schedule_database="build/schedule_db",',
      '                 schedule_database_mode="reuse"):',
      '    exe = npu_target.build(module, func_name="prefill")',
      '',
      '# generation (checkpoint 준비 후)',
      'python d_compiler/run_nn_generate.py --model llama \\',
      '  --prompt "Hello, NPU compiler!" --tokens 3 --capacity 9',
    ], { size: 8 });
    D.callout(s, 8.35, 4.65, 4.48, 1.7, [
      { text: 'build_image / run은 raw checkpoint의 변환을 자동 추측하지 않습니다. ', bold: true }, { text: 'host transform과 _bind_inputs의 역할(3장, 9장)을 따라 최종 signature에 맞는 values를 넘겨야 합니다.', size: 9.5 },
    ], { kind: 'warn' });
    D.takeaway(s, '옵션 하나는 한 단계만 바꿉니다. "fusion off"가 메모리 최적화를 끄지 않고, "joint_memory"가 runtime eviction을 뜻하지 않습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 79 diagnosis
  D.add({ section: '10 · 실험 · 한계 · 정리', title: '오류가 나면 어느 단계부터 보는가', subtitle: '관찰한 증상을 단계의 결정에 대응시키면 확인할 파일과 예제가 정해집니다. Native/Python 일치는 계약 준수의 근거이지 모델 수식 오류까지 배제하지 않습니다.', source: 'docs/current/12_operations_limits.md',
    notes: 'LLVM부터 HF와 크게 다르면 모델 수식, weight 순서, mask/RoPE, 정밀도 정책을 봅니다. 작은 K는 맞고 큰 K가 틀리면 FP32 accumulator 연결과 부분합 순서, 첫 tile만 맞으면 MAIN stride·PARTIAL 주소·단위, SRAM 보관을 켤 때만 틀리면 값 버전·마지막 소비·alias·덮어쓰기, 주소 압축 후 틀리면 DMA와 output/constant/append 주소 일관성, prefill은 맞고 decode가 틀리면 현재 token·position·mask·append 슬롯, words가 너무 많으면 짧은 vector·주소 설정 반복·정적 전개, compile이 오래 걸리면 cache hit/miss·거절 이유·phase 시간·종료 한도를 확인합니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 12.33, [
      ['관찰한 증상', '먼저 확인할 것', '관련 단계 / 파일'],
      ['LLVM부터 HF와 크게 다름', '모델 수식, weight 순서, mask/RoPE, 정밀도 정책', '2~4장: nn_models, npu_legalize, npu_numeric'],
      ['작은 K는 맞고 큰 K가 틀림', 'FP32 accumulator 연결과 부분합 순서', '6장: npu_accumulator, emit_gemm, tile_order_reference'],
      ['첫 tile만 맞고 뒤 tile이 틀림', 'MAIN stride, PARTIAL 주소, dtype/주소 단위(nibble·cell·unit)', '9장: tir_codegen_v09, isa_compute, isa_v09'],
      ['SRAM 보관을 켤 때만 틀림', '값 버전, 마지막 소비, alias, 덮어쓰기 판단', '8장: npu_memory_search.discover, parameter_scopes'],
      ['주소 압축 후 출력/cache가 틀림', 'DMA와 output/constant/append 주소의 일관성', '8장: compact_unused_storage, npu_cache 재구성'],
      ['Prefill은 맞고 다음 decode가 틀림', '현재 token, position, mask, cache append 슬롯', '9장: length_mask, AppendPlan.position/words, DecodeSession'],
      ['Words가 너무 많음', '짧은 vector 행, 주소 설정 반복, 정적 loop 전개', '7장/9장: pointwise coalescing, peephole'],
      ['Compile이 오래 걸림', 'trial/entry/region cache hit·miss, 거절 이유, phase 시간, 종료 한도', '10장: compilation.describe, budget, native_rejections'],
      ['native와 Python이 다름', 'compare 모드의 첫 다른 word, contract/source hash', '9장: native.py, v09_contract.json, build_identity'],
    ], { colW: [2.8, 4.6, 4.93], size: 9, rowH: 0.42 });
    D.takeaway(s, '증상 → 단계 → 예제의 IR/metadata에서 실제 선택 확인. 처음부터 파일 순서로 읽기보다 관심 있는 현상이 어느 단계의 결정인지 먼저 정합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 80 open issues
  D.add({ section: '10 · 실험 · 한계 · 정리', title: '남은 이슈: 구현 착수 목록이 아니라 검토 목록', subtitle: '효과가 작거나 custom 복잡도가 큰 변경은 원하지 않습니다. 기존 구현을 유지한 상태에서 필요성과 규모를 먼저 평가합니다. 이번 문서 작업에서 최적화를 추가 구현하지 않았습니다.', source: 'docs/current/12_operations_limits.md',
    notes: '각 항목은 현재 한계와 재개 시 먼저 확인할 것을 짝지어 둔 것입니다. 특히 vector 256/긴 VLEN은 RTL 계약 확인이 먼저이고, attention online softmax와 메모리 퇴거/부분 tile은 큰 변경이라 설계부터 분리합니다. 성능 모델은 RTL/장치 timing을 확보한 뒤 DMA→words→SRAM 비교와의 상관을 확인해야 합니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 6.1, [
      ['항목', '현재 한계', '재개할 때 먼저 확인'],
      ['Vector 256 / 긴 VLEN', '과거 상한 설명과 후속 결정 충돌, C-model은 물리 버퍼 모형 아님', '실제 장치의 LOAD/연산/SAVE·유효 길이·버퍼 계약'],
      ['Vector schedule / 비용', '용량에 맞춘 행 축소 위주, cycle·lane 활용률 미반영', '대표 kernel의 짧은 명령과 SRAM 임시값 비용'],
      ['작은 fusion 영역', '큰 TVM 영역 채택/해제 중심', '일부만 합치면 최종 프로그램이 좋아지는지 제한 비교'],
      ['ISA activation', '자동 matmul activation OFF', '마지막 K, bias 순서, FP16 반올림과 품질 차이'],
      ['Attention', 'Query 분할, key online softmax 없음', '장치 누적 상태와 긴 context에서의 예상 이득'],
      ['Decode 길이', 'Capacity 전체 계산과 append 전송', 'bucket 효과, 빈 슬롯 낭비, KV 최종 위치 직접 저장 계약'],
    ], { colW: [1.4, 2.5, 2.2], size: 8.5, rowH: 0.62 });
    D.table(s, 6.85, 1.55, 5.98, [
      ['항목', '현재 한계', '재개할 때 먼저 확인'],
      ['메모리', 'Whole-tensor 보관, 고정 호출 순서, 일부 조각화', '보관 포기 사례의 실제 손실. 실행 순서/퇴거/부분 tile은 큰 변경으로 분리'],
      ['DRAM 압축', '완전 미사용 pool 제거', '사용 중 pool 내부 낭비가 의미 있는지 확인 후 수명 재분석'],
      ['병렬 matmul 결합', 'QKV / gate-up 표준 pass 효과 미확인', 'CombineParallelMatmul과 split/layout 비용 포함 비교'],
      ['DLight / MetaSchedule', 'DLight 미사용; trace DB·ReplayFunc 선택형', '여러 seed의 개선과 탐색 비용; 학습형 모델은 별도'],
      ['정밀도 선택', '고정 계약, 자동 오차 예산 탐색 없음', '허용 오차·모델 품질 기준을 먼저 정함'],
      ['성능 모델', 'DMA → words → SRAM 비교', 'RTL/장치 timing 확보 뒤 상관 확인'],
      ['양자화 generation', '공개 generate에 선택 미연결', '후반부 범위부터 검증, KV 양자화까지 완료로 확대하지 않음'],
    ], { colW: [1.4, 2.3, 2.28], size: 8.5, rowH: 0.55 });
    D.takeaway(s, '새 기능을 늘리기 전에 예상 이득, 공통 구현 가능성, 검증 비용을 함께 확인합니다. 지표(DMA/words/SRAM)로 확인되지 않는 개선은 채택하지 않습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 81 summary
  D.add({ section: '정리', title: '정리: 작은 예제 하나가 지나온 길 — 계산에서 실행까지', subtitle: 'SiLU(X[64,128]·Wᵀ+bias)가 각 단계에서 무엇으로 바뀌었는지. 같은 흐름이 실제 Llama 28 layers에서는 8.58M words의 prefill과 7.31M words의 decode가 됩니다.', source: 'docs/examples/pipeline/*, two_matmuls_recorded/*, evidence/current/*',
    notes: '① authored Relax 4연산 → ② graph prefix로 transpose가 host로 (prefill(x, bias, lv) + prefill_transform_params) → ③ LegalizeOps로 call_tir 3개 + PrimFunc(FP32 누산 정의) → ④ fusion 후보 비교로 fused_matmul_add_silu 채택(words 2,812→1,577) → ⑤ schedule: a_at=b_at=m_panel, X/W 16 KiB 1회 복사, M_MUL 2개(MAC 0/1), SiLU vector 전개 폭 128 → ⑥ USMP greedy_by_conflicts peak 85,016 nibbles, 보관할 중간값 없음 → ⑦ codegen/link: 1,577 words, DMA 41,100 B, 5 packets → ⑧ runtime: image 41,100 B, C-model 실행, 출력 offset 32,896에서 [64,64] FP16 읽기, LLVM reference와 최대 절대차 0.' },
  (s, D) => {
    const st = [
      ['① Relax', 'permute_dims · matmul\n· add · silu, num_input=1', 'ir'],
      ['② Graph prefix', 'transpose → host 함수\nprefill(x, bias, lv)', 'ir'],
      ['③ Semantic TIR', 'call_tir ×3 + PrimFunc\nFP32 누산 → FP16 cast', 'ir'],
      ['④ Fusion 선택', 'fused_matmul_add_silu\nwords 2,812 → 1,577', 'sram'],
      ['⑤ Schedule', 'a_at=b_at=m_panel\nM_MUL ×2 (MAC 0/1), VLEN 128', 'sram'],
      ['⑥ SRAM 배치', 'USMP greedy_by_conflicts\npeak 85,016 nib = 42,508 B', 'sram'],
      ['⑦ ISA · link', '1,577 words · 5 DMA packets\nDMA 41,100 B · HALT', 'unit'],
      ['⑧ Runtime', 'image 41,100 B → C-model\n출력 offset 32,896 · LLVM 차 0', 'dram'],
    ];
    const w = (12.33 - 3 * 0.25) / 4;
    st.forEach((b, i) => {
      const r = Math.floor(i / 4), c = i % 4;
      const x = 0.5 + c * (w + 0.25), y = 1.65 + r * 1.55;
      D.box(s, x, y, w, 1.2, b[0], { kind: b[2], size: 12.5, detail: b[1], dsize: 9 });
      if (c < 3) D.arrow(s, x + w + 0.02, y + 0.6, x + w + 0.23, y + 0.6, { width: 1.5 });
    });
    D.velbow(s, 0.5 + 3 * (w + 0.25) + w / 2, 1.65 + 1.2, 0.5 + w / 2, 1.65 + 1.55, { my: 1.65 + 1.37, width: 1.5 });
    D.table(s, 0.5, 4.85, 12.33, [
      ['세 질문', '답을 정하는 단계', '실제 Llama에서의 결과 (보존 기록)'],
      ['무엇을 계산하는가 (계산의 의미)', 'Relax export → graph prefix → LegalizeOps custom map (FP16 저장/FP32 누산, softmax·RMSNorm 정의)', 'HF와 같은 greedy token 3개, logits cosine ≥ 0.99999'],
      ['어떻게 나누는가 (배치)', 'fusion 후보 비교 → matmul panel 탐색 → tensorize → USMP', '5커널 MLP 140,594 words, gate W panel 3072×64 128회, down W 4 MiB 유지'],
      ['어디에 보관하는가 (수명·주소)', 'StaticPlanBlockMemory → 커널 사이 보관 탐색 → 직접 저장 → DRAM 압축', 'H SRAM 전달로 DMA 81,932 → 49,164 B (44 KiB 예제), image 6.43 GB (prefill)'],
    ], { colW: [2.6, 5.2, 4.53], size: 9, rowH: 0.36 });
    D.takeaway(s, '"모델을 compile한다" = graph 선택 + schedule + 주소 배치 + 명령 생성. 각 결정은 실제 프로그램의 (DMA, words, SRAM peak)로 확인하고, 수치는 HF/LLVM 기준으로 따로 검증합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 82 appendix A1 main vs v09
  D.add({ section: '부록 A1 · 기존 main backend', title: '기존 main(ver.08 / source-0818)과 최신 v09는 실행 경계가 다르다', subtitle: 'main commit 1143abbc…로 고정해 설명합니다. 둘의 words나 실행 시간을 숫자만 비교해 성능 순위를 만들 수 없습니다: 컴파일 단위, 메모리 경계, 실행기, codegen 세대가 다릅니다.', source: 'docs/main/README.md, pipeline.md, results.md',
    notes: 'main은 Relax graph를 수작업 연산별 emitter로 연결하고 flat G-buffer와 host의 실행 경계를 사용하며, 같은 layer 프로그램을 host가 weight를 바꾸어 반복 실행하고, context별 정적 decode 프로그램과 host가 반환된 K/V로 배열을 확장하는 방식입니다. LM head는 K×64 packed panel로, 종료는 0xF0 snapshot입니다. 당시 결과: Llama 3.2 3B에서 같은 token [358, 2846, 4560], 최종 decode cosine 약 0.9999881, context8 decode 프로그램 453,349 words(layer 프로그램 통계). Gemma 4 E2B, Qwen3-4B도 main backend에서 HF 일치 기록이 있으나 v09 generation CLI에서 재검증된 것은 아닙니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 12.33, [
      ['항목', 'main ver.08 / source-0818', '최신 v09 (이 발표)'],
      ['주된 lowering', 'Relax graph를 수작업 연산별 emitter로 연결', 'LegalizeOps / TIR / Schedule을 거쳐 공통 codegen'],
      ['메모리', 'Flat G-buffer와 host의 실행 경계 (vendor 8192-entry, 확장 source target)', 'DRAM ↔ SRAM ↔ 연산장치, 명시적 GLOAD/GSTORE와 SRAM 보관 탐색'],
      ['Layer 반복', '같은 layer 프로그램을 host가 weight를 바꾸어 실행', '전체 entry stream, 컴파일 판단(recipe)만 재사용'],
      ['Decode 길이', 'Context별 정적 프로그램 cache', '고정 capacity + mask, 선택적 bucket'],
      ['KV 상태', 'Host가 반환된 K/V로 배열을 확장', 'Persistent DRAM cache에 device DMA append'],
      ['LM head', '큰 stride를 K×64 packed panel로 처리 (16-bit stride field 한계)', '일반 compiler의 지원되는 배치·명령 경로'],
      ['종료 / 출력', '0xF0 snapshot과 ver.08 실행 규약', 'HALT 0xFF와 v09 image 규약'],
      ['Vendor quirk', 'reduce-max 0 seed 등을 compiler에서 우회', 'v09 ISA/C-model에서 의미를 명시적으로 고침 (첫 원소 seed)'],
    ], { colW: [1.7, 5.3, 5.33], size: 9, rowH: 0.4 });
    D.table(s, 0.5, 5.35, 12.33, [
      ['main의 보존 결과', 'Llama 3.2 3B: token [358, 2846, 4560] HF 일치, 최종 decode cosine ≈ 0.9999881, context8 decode 453,349 words (layer 프로그램 통계, v09 entry의 수백만 words와 컴파일 범위가 다름)'],
      ['다른 family (main)', 'Gemma 4 E2B [108, 236777, 236789] cosine ≈ 0.9999975 · Qwen3-4B [358, 1184, 311] cosine ≈ 0.9999915 — v09 generation CLI 재검증은 아님'],
    ], { colW: [2.0, 10.33], size: 8.8, rowH: 0.3, header: false });
    D.takeaway(s, '현재 작업 트리에 구형 backend를 복원하거나 main branch를 수정하지 않습니다. 원본은 git show 1143abb:d_compiler/npu_compiler/backend_0818.py 등으로 읽습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 83 appendix A2 quantization
  D.add({ section: '부록 A2 · 양자화', title: '양자화 W8A16 / W8A8: 저장 dtype과 연산 dtype을 구분한다', subtitle: 'Pass와 커널은 구현되어 있지만 최근 전체 28-layer 연속 생성 검증은 FP16입니다. 양자화 generation까지 완성·검증됐다고 이해하지 않도록 본문과 분리합니다.', source: 'docs/current/13_quantization.md, npu_compiler/npu_quantize.py, npu_w8a8_schedule.py, evidence/current/quantization.log',
    notes: 'W[K,N]의 열 j마다 sW[j]=max(|W[:,j]|,tiny)/127, Wq=clip(round(W/sW),−127,127) INT8, half-to-even, zero-point 없는 대칭 양자화, weight per-output-channel, W8A8 activation per-row입니다. 양자화 rewrite는 LiftTransformParams 전에 적용되어 scale/quantize가 host로 옮겨집니다. W8A16은 DRAM INT8 W + FP16 scale → SRAM에서 FP16 복원 → 기존 FP16 GEMM chain(mixed matrix mode 미사용)입니다. W8A8은 실행 중 행별 absmax·FP32 scale·VQUANT로 INT8 X를 만들고 64개 K마다 INT32 부분합에 sW·sX를 적용해 FP32 accumulator에 더한 뒤 FP16 저장입니다. 검증: W8A16 matmul, W8A8 [64,64]@[64,64], [7,128]@[128,192] bit equality, 작은 모델 cosine ≈ 0.999998. run_nn_generate/npu_generate.generate에는 양자화 선택이 연결되어 있지 않습니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.3, 'W8A16', { size: 11, bold: true, color: C.navy });
    const a = [['DRAM INT8 W', '+ FP16 scale (host FP32 계산)', 'dram'], ['SRAM FP16 복원', 'INT8 → FP16, × scale', 'sram'], ['FP16 GEMM', '기존 FP32 MAC chain', 'unit'], ['FP16 결과', '', 'sram']];
    a.forEach((b, i) => { const x = 0.5 + i * 1.55; D.box(s, x, 1.85, 1.45, 0.8, b[0], { kind: b[2], size: 9.5, detail: b[1], dsize: 7.5 }); if (i < 3) D.arrow(s, x + 1.46, 2.25, x + 1.54, 2.25, { width: 1 }); });
    D.text(s, 0.5, 2.85, 6, 0.3, 'W8A8', { size: 11, bold: true, color: C.navy });
    const b = [['FP16 X → INT8', '행별 absmax, FP32 sX, VQUANT', 'sram'], ['INT8 dot (K 64개)', 'INT32 부분합', 'unit'], ['scale 적용', '× sW[n] × sX[m] → FP32 acc', 'unit'], ['FP16 결과', '전체 K 완료 후', 'sram']];
    b.forEach((bb, i) => { const x = 0.5 + i * 1.55; D.box(s, x, 3.2, 1.45, 0.8, bb[0], { kind: bb[2], size: 9.5, detail: bb[1], dsize: 7.5 }); if (i < 3) D.arrow(s, x + 1.46, 3.6, x + 1.54, 3.6, { width: 1 }); });
    D.table(s, 0.5, 4.25, 6.4, [
      ['계산 / 데이터', '현재 pass의 범위'],
      ['Q/K/V/O, gate/up/down, LM head weight matmul', '조건 만족 시 양자화 대상'],
      ['Attention QKᵀ, PV', 'runtime operand → 대상 아님'],
      ['RMSNorm, softmax, RoPE, residual', 'FP16 경로'],
      ['Projection 입출력', 'FP16 (W8A8은 내부에서 동적 양자화)'],
      ['KV cache', 'FP16'],
    ], { colW: [3.6, 2.8], size: 8.8, rowH: 0.3 });
    D.table(s, 7.15, 1.55, 5.68, [
      ['항목', '내용'],
      ['scale', 'sW[j] = max(|W[:,j]|, tiny)/127, per-output-channel; W8A8 activation per-row FP32'],
      ['반올림', 'half-to-even; 예: scale 1/127에서 0.5 → 64 → 복원 0.50394'],
      ['pass 위치', 'LiftTransformParams 전에 rewrite → scale/quantize가 host로'],
      ['W8A16 경로', 'FP16 복원 weight + 기존 GEMM chain (ISA mixed matrix mode는 미선택)'],
      ['W8A8 수치', '64개 부분합마다 scale 적용 순서까지 기준. 전체 K INT32 합 뒤 1회 scale과 항상 같지 않음'],
      ['schedule', 'versioned W8A8 TIR: K/N 64배수, 정적 rank-2, K ≤ 65535; FP16 X와 INT8 X 공존 구간'],
      ['검증', 'W8A16 matmul, W8A8 [64,64]@[64,64], [7,128]@[128,192] bit equality; 작은 모델 cosine ≈ 0.999998'],
      ['미연결', 'run_nn_npu.py --quantize 경로는 있으나 generate에는 미연결'],
    ], { colW: [1.2, 4.48], size: 8.5, rowH: 0.5 });
    D.takeaway(s, 'INT4 형식이 ISA에 있어도 INT4 모델 compiler 완성이 아니고, 작은 kernel의 bit equality가 양자화 전 모델 품질 보존을 뜻하지 않습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 84 appendix A3 TVM API dictionary
  D.add({ section: '부록 A3 · TVM API 사전', title: 'TVM API를 동작 기준으로 읽기: 무엇을 받아 무엇으로 바꾸며, 아직 무엇을 정하지 않는가', subtitle: '설치된 TVM 0.19.0 기준. 이름을 외우기보다 각 API가 결정하는 것과 결정하지 않는 것을 구분합니다.', source: 'docs/reference/tvm_api.md',
    notes: 'IR을 만드는 API(export_tvm, BlockBuilder, TensorStructInfo, te.placeholder/compute/reduce_axis, call_te, call_tir, tvm.script), graph pass(Sequential, CanonicalizeBindings, EliminateCommonSubexpr, RewriteDataflowReshape, LiftTransformParams, LegalizeOps, FoldConstant, AnnotateTIROpPattern, FuseOps, FuseOpsByPattern, FuseTIR, DeadCodeElimination), TIR 후처리(LowerInitBlock, PlanAndUpdateBufferAllocationLocation, ConvertBlocksToOpaque, CompactBufferAllocation, LowerMatchBuffer), 메모리(ToNonDataflow, RemovePurityChecking, CallTIRRewrite, StaticPlanBlockMemory, USMP), 검증(structural_hash/equal, relax.build llvm + VirtualMachine)로 나뉩니다. CanonicalizeAttention, FoldRepeatedGemv, 양자화 rewrite, OutputViews는 우리 모듈의 변환이며 TVM 기본 pass가 아닙니다. 선택형 MetaSchedule API(Workload/TuningRecord/JSONDatabase, Schedule.trace, TuneContext/ScheduleFn/ReplayFunc, sample_categorical, extract_tasks)는 6장 절과 같이 기존 NPU 비용 기준을 유지한 채 저장·재생과 검사 순서 변경에만 사용합니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.5, 6.1, [
      ['Graph pass', '하는 일', '현재 사용의 경계'],
      ['LiftTransformParams', 'parameter-only 계산을 별도 함수로', '실행은 host 준비 단계'],
      ['LegalizeOps(map)', 'Relax op → TE/TIR 호출', 'Relax 함수는 남음'],
      ['FoldConstant', '상수만의 계산 미리 평가', 'NPU는 K=64 grouping wrapper 사용'],
      ['AnnotateTIROpPattern', 'TIR 성격 분류', 'SRAM 적합성·속도 판정 아님'],
      ['FuseOps / ByPattern', '의존 그래프·pattern으로 영역', 'TIR 결합·채택 판단은 별도'],
      ['FuseTIR', '영역의 TIR을 한 PrimFunc로', 'SRAM 배치는 후속'],
      ['RewriteDataflowReshape', 'reshape를 view 표현으로', 'transpose는 복사 없이 처리 안 됨'],
      ['DeadCodeElimination', '미사용 함수·값 정리', '읽는 원소 수 감소와 다름'],
    ], { colW: [1.7, 2.4, 2.0], size: 8.5, rowH: 0.32 });
    D.table(s, 6.85, 1.5, 5.98, [
      ['Schedule / TIR / 메모리 API', '역할', '확인할 것'],
      ['split / fuse / reorder', 'loop 분할·평탄화·순서', 'predicate, 의존성, 단일 accumulator'],
      ['cache_read / cache_write', '별도 scope 복사 stage', '어느 소비자가 공유하는지'],
      ['compute_at / reverse_compute_at', '생산·소비 block 이동', '관련 소비가 허용 범위인지'],
      ['set_scope / pad_einsum / decompose_reduction', 'scope 변경 / tile 배수 확장 / init 분리', 'zero padding 의미, 누산 시작 loop'],
      ['TensorIntrin.register / tensorize', 'intrinsic 등록·치환', 'shape/dtype/scope/수식 계약'],
      ['rfactor', 'reduction을 부분합으로', 'CPU reference의 K=64 grouping'],
      ['CompactBufferAllocation', '접근 범위로 allocation 축소', 'cache stripe 크기'],
      ['StaticPlanBlockMemory / USMP', 'DRAM pool / SRAM 충돌 분석', 'flat 주소는 backend가 부여'],
      ['structural_hash / equal', 'IR 동일성', 'cache 키 + equality 확인'],
    ], { colW: [2.1, 1.9, 1.98], size: 8.5, rowH: 0.32 });
    D.table(s, 0.5, 4.5, 6.1, [
      ['MetaSchedule API (선택형)', '제공 기능', '현재 사용 범위'],
      ['Workload / TuningRecord / JSONDatabase', '작업·trace·기록 보존', '선택된 trace만, run_secs=None, 버전별 디렉터리'],
      ['tir.Schedule.trace', '적용한 Schedule 명령 기록', '새 TIR에 재적용 후 lowering·SRAM·codegen 재검증'],
      ['TuneContext / ScheduleFn / ReplayFunc', '탐색 문맥·후보 생성·재실행 전략', '같은 NPU 후보 pool의 검사 순서를 바꾸는 비교 경로'],
      ['sample_categorical', '범주 선택을 확률적으로 기록', '정리된 후보 번호를 같은 확률로 추첨'],
      ['relax_integration.extract_tasks', '같은 TIR 작업 합치기·호출 수 집계', '작업 목록 진단만. TaskScheduler 미사용'],
    ], { colW: [2.1, 1.85, 2.15], size: 8.2, rowH: 0.3 });
    D.callout(s, 6.85, 4.85, 5.98, 1.5, [
      { text: '쓰지 않았다고 문제인 것은 아닌 API: ', bold: true }, { text: 'DLight는 schedule 규칙 체계, MetaSchedule은 탐색·측정 framework입니다. 09-29에 MetaSchedule의 trace DB와 ReplayFunc를 선택형으로 연결했고 DLight는 미사용입니다. 이 API가 장치 계약·SRAM 보관·비용 모델까지 자동 해결하지는 않으며 EvolutionarySearch·학습형 CostModel은 도입하지 않았습니다. CombineParallelMatmul 같은 추가 pass도 concat/split 비용을 측정한 뒤 판단합니다. CanonicalizeAttention, FoldRepeatedGemv, 양자화 rewrite, OutputViews는 우리 모듈의 변환입니다.', size: 8.5 },
    ], { kind: 'faint' });
  });

  // ---------------------------------------------------------------- 85 appendix A4 reproduction
  D.add({ section: '부록 A4 · 재현', title: '재현 환경과 명령', subtitle: '작은 예제부터 시작하고 소스·환경·설정을 기록합니다. 기존 측정값이 현재도 같을 것이라고 가정해 재사용하지 않습니다.', source: 'docs/current/12_operations_limits.md, docs/examples/README.md, d_compiler/toolchain.lock.json',
    notes: 'TVM 0.19.0 (commit c4dc0c29…), LLVM 18.1.8, Python 3.11.15, conda 환경 npu-tvm이 기준입니다. native build 명령은 library가 없거나 소스와 맞지 않을 때 필요합니다. reproduce.py는 작은 compile 7개 구성과 C-model 실행을 하며 전체 Llama나 checkpoint 다운로드는 하지 않습니다. export_llama_ir.py는 큰 weight를 읽거나 codegen하지 않고 IR만 만듭니다. 재현 결과는 보존된 문서 예제를 덮어쓰지 않도록 별도 디렉터리에 둡니다. 예제 출력이 달라졌을 때 문서 숫자만 먼저 바꾸지 말고 source/profile/numeric policy와 선택 후보를 확인합니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 5.2, [
      ['환경', '값'],
      ['TVM', '0.19.0, commit c4dc0c29ff81…'],
      ['LLVM', '18.1.8'],
      ['Python', '3.11.15 (conda env npu-tvm)'],
      ['numpy / torch / transformers', '2.4.6 / 2.12.0+cpu / 5.12.1'],
      ['C-model', '_poc/mysim_v09.cpp (g++ -std=c++17 -O2)'],
      ['native codegen', 'npu_codegen/libnpu_codegen.so (build.sh)'],
      ['lock', 'd_compiler/toolchain.lock.json (conda solver lock 아님)'],
    ], { colW: [2.1, 3.1], size: 9, rowH: 0.32 });
    D.code(s, 5.95, 1.55, 6.88, 2.6, [
      'conda activate npu-tvm',
      'bash d_compiler/npu_codegen/build.sh                 # native library (필요 시)',
      'python d_compiler/docs/examples/reproduce.py --out /tmp/npu-doc-examples',
      '    # E1 pipeline + E2 memory: 작은 compile 7개 + C-model 실행 (--case pipeline|memory)',
      'python d_compiler/docs/examples/export_llama_ir.py --out /tmp/npu-doc-llama-ir',
      '    # 실제 3B 폭 1-layer IR export만 (weight 읽기·codegen 없음)',
      'python d_compiler/docs/examples/schedule_api/reproduce.py   # compute_at 등 교육 dump',
      'python d_compiler/docs/tools/check_docs.py             # 문서 링크·수치 검사',
      'python d_compiler/run_nn_generate.py --model llama --prompt "Hello, NPU compiler!" \\',
      '    --tokens 3 --capacity 9                           # 전체 모델 (checkpoint 필요, 장시간)',
    ], { size: 8.5 });
    D.table(s, 0.5, 4.35, 12.33, [
      ['새 실험을 추가할 때 함께 남길 기록', ''],
      ['identity', 'compiler source hash, TVM/native/C-model identity, contract hash'],
      ['입력', 'shape/dtype, profile, fusion/residency 설정, seed 또는 checkpoint/prompt'],
      ['실행', '실제 실행 여부(executed), 비교 기준(HF/LLVM/NumPy)과 tolerance'],
      ['해석', '같은 입력으로 한 변수만 바꿈. 큰 context는 예상 compile·memory 비용부터 산정. timing 없으면 words/DMA 감소까지만 주장'],
    ], { colW: [1.4, 10.93], size: 9, rowH: 0.32 });
    D.takeaway(s, '작은 예제의 실행 검증, 기존 전체 Llama 기록, 정적 MLP compile, 과거 단일 시간 관측을 구분해 인용합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 86 closing
  D.add({ dark: true, bare: true, notes: '질문을 받습니다. 세부 IR·API·실험 근거는 docs/README.md의 13장과 examples/, evidence/에서 확인할 수 있습니다.' },
  (s, D) => {
    D.text(s, 0.8, 1.6, 11.5, 0.3, 'TVM 기반 LLM NPU COMPILER', { size: 11, color: C.sram, bold: true, charSpacing: 2 });
    D.text(s, 0.8, 2.1, 11.5, 1.2, '계산의 의미 → 실행 배치 → 수명과 주소 → 명령과 runtime', { size: 30, color: C.white, bold: true });
    D.text(s, 0.8, 3.3, 11.5, 1.2, '각 결정은 실제 프로그램의 (DMA bytes, ISA words, SRAM peak)로 확인하고,\n수치는 HF / LLVM 기준으로 따로 검증합니다.', { size: 16, color: C.darkText });
    D.rect(s, 0.8, 4.9, 11.7, 1.4, { fill: '18293F', line: '2B3E58', radius: 0.08 });
    D.text(s, 1.05, 5.05, 11.2, 1.1, '문서: d_compiler/docs/README.md (13장) · 예제: docs/examples/ (실제 IR·명령·실행 결과) · 근거: docs/evidence/ (source hash와 함께 보존)\n코드: d_compiler/npu_compiler/, npu_codegen/ · 검증: d_compiler/run_gate.sh, docs/examples/reproduce.py', { size: 12, color: C.darkMuted });
    D.text(s, 0.8, 6.75, 11.5, 0.3, '2026-09-28 작업 트리 · TVM 0.19.0 · v09 backend · FP16', { size: 10, color: C.darkMuted });
  });
};
