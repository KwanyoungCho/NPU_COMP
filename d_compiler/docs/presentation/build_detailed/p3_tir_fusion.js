'use strict';
const { C, F } = require('./lib');

module.exports = (D) => {
  // ---------------------------------------------------------------- 19 LegalizeOps keeps Relax
  D.add({ section: '4 · LegalizeOps와 수치', title: 'LegalizeOps 뒤에도 Relax는 남는다: call_tir + PrimFunc', subtitle: '작은 예제의 실제 unfused IR. Relax 함수는 호출 순서를, 각 PrimFunc는 원소 계산을 정의합니다. op_pattern attribute가 다음 단계 FuseOps의 입력입니다.', source: 'docs/examples/pipeline/03_unfused_semantic_tir.py, docs/current/04_semantic_tir.md',
    notes: 'LegalizeOps는 고수준 Relax 연산을 원소 단위 계산으로 정의하지만 SRAM 주소나 64×64 실행 순서는 정하지 않습니다. call_tir는 TIR 함수를 호출하고 결과 shape/dtype을 알려주는 표현이지 그래프 연결을 지우는 명령이 아니므로 그 뒤에 FuseOps를 수행할 수 있습니다. TIR은 TVM Tensor IR이며 dump는 TVM Script 문법입니다. 다른 compiler가 TIR이라는 약자를 써도 같은 IR이 아닙니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.3, 'Relax 함수 (호출 순서 유지)', { size: 11, bold: true, color: C.navy });
    D.code(s, 0.5, 1.8, 6.1, 1.75, [
      [{ text: '@R.function', color: C.codeKey }],
      'def prefill(x: (64,128), bias: (64,), lv: (128,64)) -> (64,64):',
      '    R.func_attr({"num_input": 1});  cls = Module',
      '    with R.dataflow():',
      '        lv1 = R.call_tir(cls.matmul, (x, lv),  out_sinfo=R.Tensor((64,64), "float16"))',
      '        lv2 = R.call_tir(cls.add,    (lv1, bias), out_sinfo=...)',
      '        gv  = R.call_tir(cls.silu,   (lv2,),      out_sinfo=...)',
      '        R.output(gv)',
    ], { size: 8.5 });
    D.text(s, 0.5, 3.7, 6, 0.3, 'PrimFunc 목록 (같은 IRModule 안)', { size: 11, bold: true, color: C.navy });
    D.table(s, 0.5, 4.0, 6.1, [
      ['PrimFunc', 'op_pattern', '계산'],
      ['matmul', '4 kOutEWiseFusable', 'FP32 누산 + FP16 cast, npu.numeric_policy attr'],
      ['add', '0 kElemWise', 'T_add[i,j] = lv1[i,j] + bias[j] (broadcast)'],
      ['silu', '0 kElemWise', 'compute = sigmoid(x); T_multiply = x * compute'],
      ['transpose', '2 kInjective', 'host 함수용 (prefill_transform_params가 호출)'],
    ], { colW: [1.1, 1.6, 3.4], size: 9, rowH: 0.32 });
    D.text(s, 6.85, 1.5, 6, 0.3, 'matmul PrimFunc (TVM Script)', { size: 11, bold: true, color: C.navy });
    D.code(s, 6.85, 1.8, 5.98, 3.9, [
      [{ text: '@T.prim_func(private=True)', color: C.codeKey }],
      'def matmul(x: T.Buffer((64, 128), "float16"), lv: T.Buffer((128, 64), "float16"),',
      '           compute: T.Buffer((64, 64), "float16")):',
      '    T.func_attr({"npu.numeric_policy": "fp16_storage_fp32_accum", "op_pattern": 4})',
      [{ text: '    matmul = T.alloc_buffer((64, 64))              # FP32 논리 buffer', color: C.codeStr }],
      '    for i0, i1, k in T.grid(64, 64, 128):',
      '        with T.block("matmul"):',
      [{ text: '            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])   # S,S = spatial, R = reduction', color: C.codeStr }],
      '            T.reads(x[v_i0, v_k], lv[v_k, v_i1]);  T.writes(matmul[v_i0, v_i1])',
      '            with T.init():',
      '                matmul[v_i0, v_i1] = T.float32(0.0)',
      '            matmul[v_i0, v_i1] = matmul[v_i0, v_i1] \\',
      '                + T.Cast("float32", x[v_i0, v_k]) * T.Cast("float32", lv[v_k, v_i1])',
      '    for i0, i1 in T.grid(64, 64):',
      '        with T.block("compute"):',
      [{ text: '            compute[v_i0, v_i1] = T.Cast("float16", matmul[v_i0, v_i1])   # FP16 저장 경계', color: C.codeStr }],
    ], { size: 8 });
    D.callout(s, 6.85, 5.85, 5.98, 0.55, 'TIR의 FP32 buffer "matmul"이 반드시 FP32 SRAM 배열로 materialize되는 것은 아닙니다. tensorize 뒤에는 장치 accumulator가 이 상태를 맡습니다.', { kind: 'warn', size: 9.5 });
    D.takeaway(s, 'Semantic TIR에서 확인할 것: 입출력 shape/dtype, 더하는 축과 초기값, FP32→FP16 cast 위치, 아직 global인 buffer, schedule 뒤에도 같은 계산 의미인지.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 20 reading TIR
  D.add({ section: '4 · LegalizeOps와 수치', title: 'TIR 읽기: block, 축의 종류, reads/writes, TE와 call_te', subtitle: 'TE의 placeholder/compute/reduce_axis는 "이 원소의 값은 어떤 수식인가"를 선언하고, bb.call_te가 그 정의에서 PrimFunc와 call_tir를 만듭니다.', source: 'docs/current/04_semantic_tir.md, reference/tvm_api.md, npu_compiler/npu_legalize.py',
    notes: 'i,j는 결과 위치를 고르는 spatial 축, k는 같은 결과에 여러 값을 더하는 reduction 축입니다. T.init은 reduction 초기값, T.reads/T.writes는 block의 접근 영역으로 schedule이 이동·분할의 합법성을 분석할 때 씁니다. Buffer는 shape/dtype/scope를 가진 배열 표현입니다. Block 이름이 같다고 같은 연산이 아니며, 현재 kernel 분석(npu_kernel.matmul_blocks)도 이름이 아니라 접근·수식·계약을 보고 matmul을 인식합니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 6.3, [
      ['TIR 요소', '뜻', '이 예제'],
      ['T.block("matmul")', '읽기·쓰기 영역과 iteration 관계를 가진 계산 단위', 'C32[i,j] += f32(x[i,k])·f32(lv[k,j])'],
      ['T.axis.remap("SSR")', 'S = spatial(출력 위치), R = reduction(더하는 축)', 'i0, i1 spatial / k reduction'],
      ['T.init()', 'reduction 초기값', 'C32 = 0.0 (FP32)'],
      ['T.reads / T.writes', 'block 접근 영역 → schedule 합법성 분석', 'x[i,k], lv[k,j] / matmul[i,j]'],
      ['T.alloc_buffer', 'PrimFunc 내부 논리 buffer (scope 기본 global)', 'matmul (64,64) FP32'],
      ['T.Cast', 'dtype 변환 위치 = 수치 경계', 'FP16→FP32 곱, FP32→FP16 저장'],
      ['scope', 'buffer 저장 계층 (global / global.sram / local.accumulator)', 'schedule이 global.sram으로 변경'],
    ], { colW: [1.7, 2.7, 1.9], size: 9, rowH: 0.42 });
    D.text(s, 7.05, 1.5, 6, 0.3, 'TE 정의 → PrimFunc (npu_legalize._matmul_te)', { size: 11, bold: true, color: C.navy });
    D.code(s, 7.05, 1.8, 5.78, 2.6, [
      'def _matmul_te(a, b):',
      '    ...  # rank>=2 검사, batch broadcast shape 계산',
      '    axis = te.reduce_axis((0, ashape[-1]), name="k")',
      '    result = te.compute(shape, lambda *idx: te.sum(',
      '        a[..., idx[-2], axis].astype("float32")',
      '        * b[..., axis, idx[-1]].astype("float32"), axis=axis),',
      '        name="matmul")',
      '    return topi.cast(result, "float16")',
      '',
      'def _legalize_matmul(bb, call):   # FP16×FP16→FP16만 NPU 정의',
      '    return bb.call_te(_matmul_te, *call.args, primfunc_name_hint="matmul",',
      '        primfunc_attrs={"npu.numeric_policy": NUMERIC_POLICY})',
    ], { size: 8.5 });
    D.bullets(s, 7.05, 4.55, 5.78, 1.8, [
      ['matmul_te는 실행기가 아님: ', '수식 선언일 뿐, 손으로 쓴 ISA 생성기도 아님'],
      ['확장 지점: ', 'LegalizeOps(custom_legalize_map)에 matmul, sum, mean, rms_norm, softmax, split만 등록. 나머지는 TVM 기본 legalization'],
      ['다른 dtype 조합: ', 'FP16이 아니면 TVM 기본 _matmul로 위임 (범용 FP32 vector 실행은 별도 범위)'],
    ], { size: 9.5, psa: 4 });
    D.takeaway(s, '이 파일에는 아직 최종 DMA량이 없습니다. DMA량은 cache 위치, 보관 선택, 실제 주소와 후단 명령 정리까지 적용한 뒤에야 판단합니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 21 legalize map table
  D.add({ section: '4 · LegalizeOps와 수치', title: 'NPU legalize map: 기본 lowering과 다른 dtype·저장 경계가 필요한 연산만 바꾼다', subtitle: 'TVM 기본 RMSNorm은 FP32 중간값을 유지하지만 우리 vector unit은 FP16을 저장하고 내부 reduction만 FP32입니다. 그래서 per-op recipe만 교체합니다.', source: 'npu_compiler/npu_legalize.py legalize_map(), npu_views.legalize_split',
    notes: 'legalize_map은 relax.matmul, relax.split, relax.sum, relax.mean, relax.nn.rms_norm, relax.nn.softmax 여섯 개를 등록합니다. 각 legalizer는 지원 조건을 검사하고 밖이면 ValueError로 거절합니다. 예를 들어 softmax는 마지막 축, 정적 길이 1..65504, FP16만 허용합니다. RMSNorm은 마지막 축·1차원 weight·epsilon 범위(2^-48 ≤ ε ≤ 65504²)를 검사합니다. split은 axis-0 정적 split을 memory.view + ensure_zero_offset으로 바꿔 decode의 layer별 cache 분리가 복사 없이 되게 합니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 12.33, [
      ['Relax op', 'NPU 정의 (TE)', '지원 조건 검사', 'PrimFunc attr / 이름'],
      ['relax.matmul', 'FP16×FP16 → FP32 sum → FP16 cast. rank≥2, batch broadcast', 'FP16 입력·out_dtype void/float16. 그 외는 TVM 기본 _matmul', 'npu.numeric_policy = fp16_storage_fp32_accum, "matmul"'],
      ['relax.nn.softmax', 'max → exp(x−max) → FP32 sum → FP16 sum → divide', 'FP16, 마지막 축, 정적 길이 1..65504 (exp≤1 → FP16 합 범위)', '"softmax"'],
      ['relax.nn.rms_norm', 'scaled absmax: a=max(absmax,√ε), u=x/a, z=u/√D, s=FP32 Σz²→FP16, y=u·(1/√(s+(√ε/a)²))·w', '마지막 축, 1-D FP16 weight, 2⁻⁴⁸ ≤ ε ≤ 65504², 정적 D', 'npu.rms_algorithm = scaled_absmax_v1'],
      ['relax.sum / mean', 'FP32 sum → FP16 cast (mean은 ×1/count FP16)', 'FP16 입력. 아니면 TVM 기본', '"sum", "mean"'],
      ['relax.split', 'axis-0 정적 split → memory.view + ensure_zero_offset (zero-copy)', '연속 view 가능한 경우만', 'decode cache의 layer별 분리'],
      ['그 외 (add, silu, permute_dims, concat, repeat, reshape …)', 'TVM 기본 legalization', '—', 'op_pattern은 AnnotateTIROpPattern이 부여'],
    ], { colW: [1.75, 4.3, 3.4, 2.88], size: 8.5, rowH: 0.46 });
    D.code(s, 0.5, 5.35, 7.4, 1.1, [
      'def legalize_map(numeric_policy=NUMERIC_POLICY):',
      '    return {"relax.matmul": _legalize_matmul, "relax.split": legalize_split,',
      '            "relax.sum": _legalize_sum, "relax.mean": _legalize_mean,',
      '            "relax.nn.rms_norm": _legalize_rms_norm, "relax.nn.softmax": _legalize_softmax}',
      'stages += [relax.transform.LegalizeOps(custom_legalize), FoldConstants(), AnnotateTIROpPattern()]',
    ], { size: 8.5 });
    D.callout(s, 8.1, 5.35, 4.73, 1.1, [
      { text: '조용히 통과시키지 않는다: ', bold: true }, { text: '지원 subset 밖(예: 길이 70000 softmax, FP32 RMSNorm)은 ValueError로 거절. 임의로 근사하거나 fallback하지 않음.', size: 10 },
    ], { kind: 'warn' });
  });

  // ---------------------------------------------------------------- 22 FP16 storage / FP32 accumulate
  D.add({ section: '4 · LegalizeOps와 수치', title: 'FP16 저장 / FP32 누산: K 전체 동안 accumulator를 끊지 않는다', subtitle: 'K=128 출력 tile 하나의 128개 곱은 64개씩 두 그룹의 FP32 부분합이 되어 같은 accumulator에 더해지고, 마지막에 한 번만 FP16으로 반올림됩니다. 중간에 FP16으로 저장했다 다시 더하면 반올림 위치가 달라집니다.', source: 'docs/current/04_semantic_tir.md, npu_numeric.tile_order_reference, docs/examples/pipeline/08_llvm_reference.py',
    notes: '입력과 출력이 FP16이어도 많은 곱을 FP16으로 계속 더하면 오차가 커집니다. 현재 정책은 FP16 입력을 곱해 FP32로 누산하고 완료된 결과만 FP16으로 저장하는 것입니다. 장치는 K=64 단위 부분합을 만들므로 CPU reference도 split(k,[None,64])+rfactor로 같은 grouping을 표현하고, FoldConstants도 같은 기준을 씁니다. 단순 flat-K CPU loop와 모든 입력에서 bit-exact하다고 약속하지 않습니다(덧셈 결합 순서). E1에서는 같은 grouping의 LLVM 출력과 최대 절대차 0이었습니다.' },
  (s, D) => {
    // k-axis products
    const cells = Array.from({ length: 16 }, (_, i) => ({ label: i === 0 ? 'k=0' : (i === 7 ? '63' : (i === 8 ? '64' : (i === 15 ? '127' : ''))), kind: i < 8 ? 'sram' : 'ir', size: 6.5 }));
    D.lanes(s, 2.0, 1.62, 5.6, 0.32, cells, { title: '곱 x[i,k]·b[k,j] (FP16×FP16)', titleW: 1.9 });
    // group nodes
    const g0 = D.box(s, 2.0, 2.2, 2.8, 0.5, 'g₀ = Σ_{k<64} (FP32 부분합)', { kind: 'sram', size: 9.5 });
    const g1 = D.box(s, 4.8, 2.2, 2.8, 0.5, 'g₁ = Σ_{64≤k<128} (FP32)', { kind: 'ir', size: 9.5 });
    D.arrow(s, 3.4, 1.94, 3.4, 2.2, { width: 1 }); D.arrow(s, 6.2, 1.94, 6.2, 2.2, { width: 1 });
    // accepted chain
    const a1 = D.box(s, 2.0, 2.95, 1.75, 0.55, 'acc = g₀', { kind: 'unit', size: 10, detail: 'M_MUL MAC=0', dsize: 7.5 });
    const a2 = D.box(s, 4.0, 2.95, 1.75, 0.55, 'acc += g₁', { kind: 'unit', size: 10, detail: 'M_MUL MAC=1', dsize: 7.5 });
    const a3 = D.box(s, 6.0, 2.95, 1.6, 0.55, 'FP16 반올림 1회', { kind: 'good', size: 10, detail: 'SAVE → SRAM', dsize: 7.5 });
    D.arrow(s, g0.cx, g0.b, a1.cx, a1.y, { width: 1 }); D.arrow(s, g1.cx, g1.b, a2.cx, a2.y, { width: 1 });
    D.arrow(s, a1.r, a1.cy, a2.x, a2.cy, { width: 1.5 }); D.arrow(s, a2.r, a2.cy, a3.x, a3.cy, { width: 1.5 });
    D.text(s, 0.5, 3.05, 1.45, 0.4, '허용 (현재 계약)', { size: 8.5, bold: true, color: C.good });
    // rejected chain
    const r1 = D.box(s, 2.0, 3.7, 1.75, 0.55, 'g₀ → FP16 저장', { kind: 'red', size: 9.5, detail: '중간 반올림', dsize: 7.5 });
    const r2 = D.box(s, 4.0, 3.7, 1.75, 0.55, 'reload + g₁', { kind: 'red', size: 9.5, detail: '누산 재개', dsize: 7.5 });
    const r3 = D.box(s, 6.0, 3.7, 1.6, 0.55, 'FP16 반올림 2회', { kind: 'red', size: 9.5, detail: '결과가 달라짐', dsize: 7.5 });
    D.arrow(s, r1.r, r1.cy, r2.x, r2.cy, { width: 1.5, color: C.red, dash: 'dash' }); D.arrow(s, r2.r, r2.cy, r3.x, r3.cy, { width: 1.5, color: C.red, dash: 'dash' });
    D.text(s, 0.5, 3.8, 1.45, 0.4, '거절 (chain 중단)', { size: 8.5, bold: true, color: C.red });
    D.text(s, 7.75, 3.72, 5.0, 0.5, '"FP32 matrix chain interrupted: resuming from an FP16 store would change the reduction" (codegen_rejected)', { size: 8, color: C.red, italic: true });
    D.text(s, 0.5, 4.4, 6, 0.28, 'CPU reference: 같은 K=64 grouping (rfactor)', { size: 10.5, bold: true, color: C.navy });
    D.code(s, 0.5, 4.7, 7.1, 1.65, [
      '# npu_numeric.tile_order_reference: split(k,[None,64]) + rfactor(factor_axis=0)',
      'matmul_rf = T.alloc_buffer((2, 64, 64))            # 64개 부분합 × 2 (g₀, g₁)',
      'for i0, i1, k_0, k_1 in T.grid(64, 64, 2, 64):',
      '    matmul_rf[vk_0, i, j] += f32(x[i, vk_0*64+vk_1]) * f32(lv[vk_0*64+vk_1, j])',
      'for i0, i1, k_0 in T.grid(64, 64, 2):            # 부분합 2개를 FP32로 결합',
      '    matmul[i, j] += matmul_rf[vk_0, i, j]',
      'compute[i, j] = T.Cast("float16", matmul[i, j])    # 마지막에 한 번 반올림',
    ], { size: 8.2 });
    D.table(s, 7.85, 1.55, 4.98, [
      ['수치 정책이 적용되는 곳', '방법'],
      ['장치 matmul', 'tensorize 64×64 intrinsic, MAC chain 1개 (matrix_partial_spill forbidden)'],
      ['CPU / LLVM reference', 'tile_order_reference → relax.build(…,"llvm") VM'],
      ['Constant folding', 'FoldConstants: reference module로 fold 후 원래 body 복원'],
      ['검증', 'validate_primitive: sum은 FP32 zero init, max identity는 −65504'],
      ['Fusion 뒤', 'matmul→add→SiLU의 FP16 저장 경계 유지 (SRAM에 둔다고 FP32가 되지 않음)'],
    ], { colW: [1.7, 3.28], size: 8.5, rowH: 0.5 });
    D.callout(s, 7.85, 4.7, 4.98, 1.65, [
      { text: '정밀도 정책은 고정: ', bold: true }, { text: '중간 FP16 반올림을 생략하면 수학적 기준에 더 가까워질 수도 있지만 다른 결과를 만드는 변환이므로 별도 정책이 필요합니다. 현재 compiler는 오차 예산을 받아 FP16/FP32 경계를 자동 탐색하지 않습니다. E1: 같은 grouping의 LLVM과 최대 절대차 0.', size: 9.5 },
    ], { kind: 'faint' });
  });

  // ---------------------------------------------------------------- 23 softmax & rmsnorm
  D.add({ section: '4 · LegalizeOps와 수치', title: 'Softmax와 RMSNorm의 실제 정의: reduction과 FP16 범위', subtitle: '"FP32 reduction"은 softmax 전체가 FP32라는 뜻이 아닙니다. 입력·exp·저장된 합·출력은 FP16이고 합산 내부만 FP32입니다.', source: 'npu_compiler/npu_legalize.py (_softmax_te, _rms_norm_te), docs/current/07_operators.md',
    notes: '최댓값을 빼면 실수 수학에서 결과는 같고 exp의 큰 양수 입력을 피합니다. 유한 logits에서 exp(x−max)≤1이므로 마지막 축 길이가 65504 이하면 FP16 합이 범위 안에 있습니다. RMSNorm은 x=1000이면 x²=10⁶으로 FP16 범위를 넘고 합산만 FP32로 바꿔도 이미 저장한 제곱값의 overflow는 복구할 수 없어서, 입력을 absmax로 먼저 줄입니다. |u|≤1이고 Σz²도 크게 자라지 않지만 FP16 중간 반올림 때문에 별도의 target 근사이며 HF FP32와 동일하지 않습니다.' },
  (s, D) => {
    D.text(s, 0.5, 1.5, 6, 0.3, 'Softmax 한 행 [2, 1, 0]', { size: 11.5, bold: true, color: C.navy });
    D.table(s, 0.5, 1.82, 6.1, [
      ['단계', '값', '장치 계산', 'dtype'],
      ['최댓값', '2', 'REDUCE_MAX (첫 원소로 시작)', 'FP16'],
      ['최댓값 빼기', '[0, −1, −2]', 'BROADCAST + SUB', 'FP16'],
      ['exp', '[1, 0.3679, 0.1353]', 'vector EXP', 'FP16 저장'],
      ['합계', '1.5032', 'REDUCE_SUM (내부 FP32) → FP16 저장', 'FP32→FP16'],
      ['나누기', '[0.6652, 0.2447, 0.0900]', 'BROADCAST + DIV', 'FP16'],
    ], { colW: [1.2, 1.7, 2.3, 0.9], size: 9, rowH: 0.34 });
    D.code(s, 0.5, 4.0, 6.1, 1.5, [
      'def _softmax_te(data):                       # npu_legalize.py',
      '    maximum  = te.compute(shape[:-1], lambda *i: te.max(data[i+(k,)], axis=k))',
      '    exponent = te.compute(shape, lambda *i: te.exp(data[i] - maximum[i[:-1]]))',
      '    total32  = te.compute(shape[:-1], lambda *i: te.sum(exponent[i+(k,)].astype("float32"), axis=k))',
      '    total    = te.compute(shape[:-1], lambda *i: total32[i].astype("float16"))',
      '    return te.compute(shape, lambda *i: exponent[i] / total[i[:-1]])',
    ], { size: 8 });
    D.text(s, 6.85, 1.5, 6, 0.3, 'RMSNorm: scaled_absmax_v1', { size: 11.5, bold: true, color: C.navy });
    D.table(s, 6.85, 1.82, 5.98, [
      ['단계', '수식', '이유'],
      ['a', 'max(absmax(x), √ε)', '|x/a| ≤ 1: 제곱 overflow 방지'],
      ['u', 'x / a', 'FP16 저장'],
      ['z', 'u / √D', '합이 D에 무관하게 ≤ 1'],
      ['s', 'Σ z·z (FP32) → FP16', 'FP32 reduction, FP16 저장 경계'],
      ['inv', '1 / √(s + (√ε/a)²)', 'ε를 같은 scale로'],
      ['y', 'u · inv · weight', '원래 x/√(mean(x²)+ε)·w 와 실수 수학에서 동일'],
    ], { colW: [0.6, 2.2, 3.18], size: 9, rowH: 0.34 });
    D.bullets(s, 6.85, 4.35, 5.98, 1.3, [
      '원래 식 y = x/√(mean(x²)+ε)·w 를 그대로 쓰면 x=1000일 때 x²=10⁶ > 65504 (FP16 overflow)',
      '지원 조건: 마지막 축, 1-D FP16 weight, 정적 D, 2⁻⁴⁸ ≤ ε ≤ 65504²',
      'LegalizeOps에서 정의하는 명시적 target 근사. schedule이 임의로 발명하는 변환이 아님',
    ], { size: 9.5, psa: 4 });
    D.takeaway(s, '행 수를 나누는 것과 한 행의 reduction 길이를 나누는 것은 다릅니다. 임의로 긴 softmax(>65504)는 현재 계약 밖입니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 24 fusion basics
  D.add({ section: '5 · Fusion 후보 선택', title: 'Fusion: 커널 수가 줄어드는 것만으로 이득을 판단하지 않는다', subtitle: '작은 예제에서 분리(off)와 결합(auto)의 실제 데이터 이동. 두 구성 모두 중간 tile P·A가 SRAM에 머물러 DMA는 같고, 커널 경계의 descriptor·LOAD/SAVE 반복과 짧은 vector 행이 words와 SRAM 작업 공간을 바꿉니다.', source: 'docs/current/05_fusion.md, docs/examples/pipeline/stage_summary.json (E1)',
    notes: 'off는 3커널(matmul 43 words, add 391, silu 2,373)이고 auto는 1커널 1,572 words입니다. 양쪽 모두 GLOAD(X, Wᵀ, bias, 상수) 4개와 GSTORE(Y) 1개만 DRAM을 지나므로 DMA는 41,100 B로 같습니다. 그림처럼 중간 tile P, A, sig가 SRAM에서 다음 계산으로 바로 이어지는 것은 커널을 나눠도 가능합니다(8장). Fused 커널은 축 결합으로 vector 행이 128 폭이 되어 명령 수가 절반이 되고, USMP가 buffer offset을 공유해 SRAM peak가 50,700 → 42,508 B로 줄었습니다. 이득의 형태는 예제마다 다르므로 실제 비용으로 확인합니다.' },
  (s, D) => {
    const tileAt = (x, y, label, kind, sub) => { const m = D.matrix(s, x, y, 0.5, 0.5, 1, 1, { kind, regions: [{ r0: 0, r1: 1, c0: 0, c1: 1, kind, label, size: 8 }] }); if (sub) D.text(s, x - 0.3, y + 0.53, 1.1, 0.2, sub, { size: 6.5, color: C.muted, align: 'center' }); return m; };
    // row A: fusion off
    let y = 1.75;
    D.text(s, 0.5, y - 0.05, 1.3, 0.6, 'fusion off\n3 kernels', { size: 10, bold: true, color: C.navy });
    D.box(s, 1.8, y, 0.8, 0.5, 'DRAM', { kind: 'dram', size: 9, detail: 'X, Wᵀ, bias', dsize: 6.5 });
    D.arrow(s, 2.62, y + 0.25, 2.95, y + 0.25, { width: 1.2, color: C.dram }); D.text(s, 2.55, y - 0.25, 0.6, 0.2, 'GLOAD', { size: 6.5, color: C.dram, bold: true, align: 'center' });
    const kA = D.box(s, 3.0, y, 1.3, 0.5, 'matmul', { kind: 'unit', size: 9.5, detail: '43 words', dsize: 7 });
    D.arrow(s, kA.r, y + 0.25, kA.r + 0.3, y + 0.25, { width: 1 });
    tileAt(kA.r + 0.32, y, 'P', 'sram', 'SRAM 보관');
    D.arrow(s, kA.r + 0.84, y + 0.25, kA.r + 1.12, y + 0.25, { width: 1 });
    const kB = D.box(s, kA.r + 1.15, y, 1.3, 0.5, 'add', { kind: 'ir', size: 9.5, detail: '391 words (64행 × VLEN 64)', dsize: 6.5 });
    D.arrow(s, kB.r, y + 0.25, kB.r + 0.3, y + 0.25, { width: 1 });
    tileAt(kB.r + 0.32, y, 'A', 'sram', 'SRAM 보관');
    D.arrow(s, kB.r + 0.84, y + 0.25, kB.r + 1.12, y + 0.25, { width: 1 });
    const kC = D.box(s, kB.r + 1.15, y, 1.55, 0.5, 'silu', { kind: 'ir', size: 9.5, detail: '2,373 words (5단계 × 64행)', dsize: 6.5 });
    D.arrow(s, kC.r, y + 0.25, kC.r + 0.3, y + 0.25, { width: 1 });
    tileAt(kC.r + 0.32, y, 'Y', 'good', '');
    D.arrow(s, kC.r + 0.84, y + 0.25, kC.r + 1.12, y + 0.25, { width: 1.2, color: C.dram }); D.text(s, kC.r + 0.8, y - 0.25, 0.7, 0.2, 'GSTORE', { size: 6.5, color: C.dram, bold: true, align: 'center' });
    D.box(s, kC.r + 1.15, y, 0.8, 0.5, 'DRAM', { kind: 'dram', size: 9, detail: 'Y', dsize: 6.5 });
    D.text(s, 3.0, y + 0.85, 8.5, 0.22, '커널 경계마다: 새 descriptor 설정 + P/A를 SRAM에서 다시 LOAD/SAVE (DRAM 왕복은 없음)', { size: 8, color: C.muted });
    // row C: fusion auto
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
    // SRAM peak bars
    y = 4.75;
    D.text(s, 0.5, y - 0.02, 1.3, 0.5, 'SRAM peak', { size: 10, bold: true, color: C.navy });
    D.membar(s, 1.8, y, 5.0, 0.3, 0, 50700, [{ from: 0, to: 50700, label: 'off 50,700 B', kind: 'ir' }], { size: 8 });
    D.membar(s, 1.8, y + 0.4, 5.0, 0.3, 0, 50700, [{ from: 0, to: 42508, label: 'auto 42,508 B', kind: 'sram' }], { size: 8 });
    D.text(s, 7.0, y - 0.02, 1.2, 0.5, 'words', { size: 10, bold: true, color: C.navy });
    D.membar(s, 7.9, y, 4.9, 0.3, 0, 2812, [{ from: 0, to: 43, label: '', kind: 'unit' }, { from: 43, to: 434, label: '391', kind: 'ir' }, { from: 434, to: 2807, label: 'silu 2,373', kind: 'ir' }, { from: 2807, to: 2812, label: '', kind: 'faint' }], { size: 7.5 });
    D.membar(s, 7.9, y + 0.4, 4.9, 0.3, 0, 2812, [{ from: 0, to: 1572, label: 'fused 1,572 (+5)', kind: 'sram' }], { size: 7.5 });
    D.table(s, 0.5, 5.65, 12.33, [
      ['설정', '커널 수', 'DMA bytes', 'ISA words', 'SRAM peak', 'DRAM image', '출력', 'Fusion이 바꾸지 않는 것'],
      ['off', '3', '41,100', '2,812', '50,700', '41,100', 'bit-exact', 'FP16 저장 경계 (P, A 각 FP16) · vector 명령은 activation bit로 바뀌지 않음 (OFF)'],
      ['auto', '1', '41,100', '1,577', '42,508', '41,100', 'bit-exact', '수치 결과 (fused/unfused bit-exact) · 입력·출력 DMA'],
    ], { colW: [0.7, 0.8, 1.1, 1.0, 1.0, 1.1, 0.9, 5.73], size: 8.5, rowH: 0.28, aligns: ['left', 'right', 'right', 'right', 'right', 'right', 'left', 'left'] });
    D.takeaway(s, '비교 기준은 커널 개수가 아니라 실제 프로그램의 (DMA bytes, ISA words, SRAM peak)입니다. E1은 전송량이 같아도 명령과 작업 공간이 줄어든 사례입니다.', { y: 6.55 });
  });

  // ---------------------------------------------------------------- 25 TVM FuseOps mechanics
  D.add({ section: '5 · Fusion 후보 선택', title: 'TVM은 무엇을 보고 묶는가: op_pattern, FuseOps, FuseTIR', subtitle: 'AnnotateTIROpPattern이 TIR의 연산 성격을 분류하고, FuseOps가 의존 그래프·post-dominator·경로 pattern으로 영역을 만들며, FuseTIR가 TIR 계산을 한 PrimFunc로 결합합니다.', source: 'docs/current/05_fusion.md, TVM 0.19 src/relax/transform/fuse_ops.cc, docs/examples/pipeline/03_unfused_semantic_tir.py',
    notes: 'FuseOps는 pattern 숫자가 같다는 이유로 묶지 않습니다. OutEWiseFusable(matmul) 출력 뒤에 elemwise/broadcast 경로가 이어지면 붙이고, injective와 tuple 연결은 후속 차례에서 처리하며, CommReduce는 다른 복합 연산에 임의로 붙이지 않고, Opaque는 경계입니다. 가지가 나뉘면 중간의 모든 관련 경로가 조건을 만족해야 합니다. 그림의 작은 예제: transpose(2)는 host 함수라 device 그래프에 없고, matmul(4) → add(0) → silu(0)이 한 영역이 되어 FuseTIR가 fused_matmul_add_silu PrimFunc를 만듭니다. NPU는 지원하지 않는 matrix 복합 영역과 W8A8을 opaque(8)로 표시해 경계를 보호합니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.55, 4.9, [
      ['값', '이름', '대표 의미'],
      ['0', 'kElemWise', '같은 위치끼리 원소별 (add, silu)'],
      ['1', 'kBroadcast', '작은 입력을 여러 위치에서 사용'],
      ['2', 'kInjective', 'transpose 등 원소 대응이 명확한 변환'],
      ['3', 'kCommReduce', '합/최댓값 축소 (sum, softmax 내부)'],
      ['4', 'kOutEWiseFusable', 'matmul처럼 출력 뒤 원소별 계산을 붙일 수 있음'],
      ['7', 'kTuple', 'tuple 구성 관련'],
      ['8', 'kOpaque', '일반 fusion의 경계 (NPU가 W8A8·복합 matrix 영역에 부여)'],
    ], { colW: [0.5, 1.6, 2.8], size: 9, rowH: 0.34 });
    D.text(s, 5.65, 1.5, 7, 0.3, 'Partitioner가 검사하는 연결 (주요 분기)', { size: 11, bold: true, color: C.navy });
    D.table(s, 5.65, 1.82, 7.18, [
      ['출발 group', '검사하는 경로 / 도착점', '예'],
      ['OutEWiseFusable', '도착이 elemwise, 사이가 elemwise/broadcast인가', 'matmul 출력에 add/activation'],
      ['Elemwise/Broadcast', '중간이 injective 이하, 도착이 허용 pattern인가', 'add → SiLU'],
      ['Injective/Tuple', '후속 차례에서 injective 이하 경로', 'reshape/transpose + 후속 계산'],
      ['CommReduce', '다른 복합 연산에 임의로 붙이지 않음', 'reduction 둘을 합치지 않음'],
      ['Opaque', '여기서 fusion 진행 없음', 'NPU가 보호한 경계'],
    ], { colW: [1.7, 3.3, 2.18], size: 9, rowH: 0.36 });
    // node graph of the small example with patterns
    const y = 4.6;
    D.text(s, 0.5, y - 0.3, 8, 0.24, '작은 예제의 call graph와 op_pattern → FuseOps 영역 → FuseTIR 결과', { size: 10, bold: true, color: C.navy });
    D.rect(s, 2.3, y - 0.05, 5.55, 0.95, { fill: C.sramFill, line: C.sram, radius: 0.08, lineWidth: 1.25 });
    D.text(s, 2.4, y + 0.66, 5.4, 0.22, 'FuseOps group (Primitive=1) → FuseTIR: fused_matmul_add_silu (중간 buffer는 아직 global)', { size: 7.5, color: C.sram, bold: true });
    const n0 = D.box(s, 0.5, y + 0.05, 1.2, 0.55, 'x, lv, bias', { kind: 'host', size: 9, detail: '함수 입력', dsize: 7 });
    const n1 = D.box(s, 2.45, y + 0.05, 1.5, 0.55, 'matmul', { kind: 'unit', size: 9.5, detail: 'op_pattern 4', dsize: 7 });
    const n2 = D.box(s, 4.25, y + 0.05, 1.4, 0.55, 'add', { kind: 'ir', size: 9.5, detail: 'op_pattern 0', dsize: 7 });
    const n3 = D.box(s, 5.95, y + 0.05, 1.4, 0.55, 'silu', { kind: 'ir', size: 9.5, detail: 'op_pattern 0 (2 blocks)', dsize: 7 });
    const n4 = D.box(s, 8.3, y + 0.05, 1.0, 0.55, 'gv', { kind: 'good', size: 9.5, detail: '출력', dsize: 7 });
    D.arrow(s, n0.r, n0.cy, n1.x, n1.cy, { width: 1.2 }); D.arrow(s, n1.r, n1.cy, n2.x, n2.cy, { width: 1.2 }); D.text(s, n1.r - 0.05, n1.cy - 0.3, 0.4, 0.2, 'lv1', { size: 7, color: C.muted, align: 'center' });
    D.arrow(s, n2.r, n2.cy, n3.x, n3.cy, { width: 1.2 }); D.text(s, n2.r - 0.05, n2.cy - 0.3, 0.4, 0.2, 'lv2', { size: 7, color: C.muted, align: 'center' });
    D.arrow(s, n3.r, n3.cy, n4.x, n4.cy, { width: 1.2 });
    D.text(s, 9.5, y - 0.05, 3.3, 0.9, '4(OutEWiseFusable) → 0 → 0: 출력 뒤 원소별 경로 → 한 영역. transpose(2)는 host 함수(prefill_transform_params)라 device 그래프에 없음. 영역 채택 여부는 다음 슬라이드의 비용 검사가 결정.', { size: 8, color: C.muted });
    D.bullets(s, 0.5, 5.65, 12.33, 0.75, [
      'FuseOps/FuseTIR만으로 중간 buffer가 SRAM에 배치되지는 않음 → 뒤의 schedule과 메모리 계획이 필요',
      'Attention은 Relax pattern이 보일 때 FuseOpsByPattern으로 먼저 그룹을 정하고 LegalizeOps/FuseTIR를 적용. 두 경로는 같은 후보에 공존 가능',
    ], { size: 9.5, psa: 3 });
    D.takeaway(s, '장치에서 빠른지, SRAM에 들어가는지는 이 표준 분류만으로 알 수 없습니다. 영역 생성은 TVM, 채택 판단은 NPU 비용 검사입니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 26 pattern fusion for attention
  D.add({ section: '5 · Fusion 후보 선택', title: 'FuseOpsByPattern: attention 전체 또는 score chain을 composite로', subtitle: '패턴은 pattern 언어(is_op/wildcard)로 쓰고, supported 검사 함수가 정적 rank-3 FP16 64배수 조건을 확인합니다. 조건 밖이면 더 작은 score fusion으로 물러납니다.', source: 'npu_compiler/tvm_pipeline.py (_attention_patterns, _masked_softmax_patterns, _matrix_producer_patterns)',
    notes: '전체 attention 패턴은 matmul → (multiply) → add → softmax → matmul입니다. supported 검사는 q/k/v가 FP16 rank-3 정적 shape이고 softmax axis가 마지막이며 h>0, k[0]==v[0]==h, k[1]==d, v[1]==k[2], 그리고 m, d, t, dv가 모두 64의 배수인지 확인합니다. decode는 m=1이라 조건을 만족하지 않습니다. score chain 패턴(npu.masked_softmax)은 softmax(add(multiply(scores, scale), mask))와 scale이 없는 변형(Gemma)입니다. 코드 주석의 seq 2048 수치는 설계 근거로 남긴 과거 관측이며 이번 실험표가 아닙니다.' },
  (s, D) => {
    D.code(s, 0.5, 1.5, 7.3, 3.3, [
      'def _attention_patterns(tile=64):                       # tvm_pipeline.py',
      '    def supported(context):',
      '        q, k, v = (shape(name) for name in ("q", "k", "v"))   # FP16, rank 3, static',
      '        if int(exprs["softmax"].attrs.axis) not in (-1, 2): return False',
      '        h, m, d = q',
      '        return (h > 0 and k[0] == h and v[0] == h and k[1] == d and v[1] == k[2]',
      '                and all(x > 0 and x % tile == 0 for x in (m, d, k[2], v[2])))',
      '    for scaled in (True, False):',
      '        scores = is_op("relax.matmul")(q, k)',
      '        if scaled: scores = is_op("relax.multiply")(scores, wildcard())',
      '        masked  = is_op("relax.add")(scores, wildcard())',
      '        softmax = is_op("relax.nn.softmax")(masked)',
      '        output  = is_op("relax.matmul")(softmax, v)',
      '        patterns.append(FusionPattern("npu.attention" if scaled else "npu.attention_unscaled",',
      '                        output, {"q": q, "k": k, "v": v, "softmax": softmax}, supported))',
    ], { size: 8.5 });
    D.table(s, 8.0, 1.5, 4.83, [
      ['패턴 이름', '모양 / 조건'],
      ['npu.attention(_unscaled)', 'matmul→(mul)→add→softmax→matmul. rank-3 FP16, m·d·t·dv 64배수'],
      ['npu.masked_softmax(_unscaled)', 'softmax(add(mul(scores,scale),mask)). 전체 패턴이 안 될 때의 fallback'],
      ['npu.matrix_producer_{concat,repeat,expand_dims}', 'matmul(x, concat(…)) (+wrapper). decode KV concat 후보'],
    ], { colW: [1.75, 3.08], size: 8.5, rowH: 0.5 });
    D.bullets(s, 8.0, 3.6, 4.83, 1.3, [
      ['시도 순서: ', '전체 attention 패턴을 먼저, 안 되면 score chain 패턴'],
      ['decode (m=1): ', '64배수 조건 불만족 → full 패턴 대상 아님'],
      ['옵션: ', 'fuse_attention ∈ {False, "score", "full"}'],
    ], { size: 9.5, psa: 4 });
    D.callout(s, 0.5, 5.0, 12.33, 1.3, [
      { text: '왜 score chain만이라도 묶는가 (코드 주석의 설계 근거): ', bold: true }, { text: 'scale·mask·softmax를 분리하면 [heads, seq, seq] score tensor를 op마다 한 번씩 훑습니다. 전체를 묶고 query 행으로 block하면 score matrix가 SRAM을 떠나지 않습니다. 코드 주석의 seq 2048 관측(layer당 3.7 GB 중 1.6 GB, attention 1,057 MB → 235 MB)은 과거 설계 근거이며 이번 실험 표의 수치가 아닙니다.', size: 10 },
    ], { kind: 'faint' });
    D.takeaway(s, '구현된 후보와 실제 선택된 후보는 별개입니다. 전체 Llama 검증 기록에서는 시간 한도 내에 unfused 기준 module이 선택되었습니다.', { y: 6.5 });
  });

  // ---------------------------------------------------------------- 27 npu_fusion.select
  D.add({ section: '5 · Fusion 후보 선택', title: 'npu_fusion.select: 영역 하나를 실제 codegen 비용으로 검사한다', subtitle: 'TVM이 만든 FuseOps 영역마다 unfused(기준)와 fused를 schedule·메모리·명령 생성까지 내려 비교하고, 진 영역은 inline_functions로 다시 펼칩니다.', source: 'npu_compiler/npu_fusion.py select(), _probe(), docs/examples/pipeline/bundle/program.json fusion.regions',
    notes: '먼저 W8A8 PrimFunc와 matmul이 2개 이상인 PrimFunc(또는 matmul_epilogues=False일 때 모든 matmul PrimFunc)에 op_pattern 8(kOpaque)을 붙여 경계로 둡니다. FuseOps로 영역을 만들고 FuseTIR로 결합한 뒤, entry에서 도달 가능한 Primitive 함수마다 unfused 기준을 probe하고 fused wrapper를 probe합니다. unfused probe의 실패는 unsupported 입력이나 compiler 오류이므로 숨기지 않습니다. fused가 사전식 비용에서 작을 때만 채택하고, 나머지는 inline해 다시 FuseTIR·DCE합니다. region_cost cache로 같은 영역의 중복 계산을 줄입니다.' },
  (s, D) => {
    const y = 1.6;
    const st = [['경계 표시', 'W8A8 · multi-matmul PrimFunc\n→ op_pattern 8 (kOpaque)', 'red'], ['FuseOps → FuseTIR', 'outlined (영역 함수)\nfused (결합 PrimFunc)', 'ir'], ['영역별 probe', 'unfused 기준 vs fused wrapper\n각각 compile_program → analyze', 'sram'], ['비교 · 거절', 'cost(fused) < cost(unfused)?\n아니면 rejected에 추가', 'warn'], ['inline · 정리', 'fn.inline_functions(rejected)\n→ FuseTIR → DCE', 'ir']];
    const w = (12.33 - 4 * 0.2) / 5;
    st.forEach((b, i) => {
      const x = 0.5 + i * (w + 0.2);
      D.box(s, x, y, w, 0.95, b[0], { kind: b[2], size: 10.5, detail: b[1], dsize: 8 });
      if (i < 4) D.arrow(s, x + w + 0.02, y + 0.47, x + w + 0.18, y + 0.47, { width: 1.2 });
    });
    D.code(s, 0.5, 2.85, 6.6, 2.7, [
      'def _probe(module, function, profile):          # 영역 하나를 프로그램으로',
      '    mod["__npu_fusion_probe"] = function.without_attr("Primitive")',
      '    asm, plan = npu_link.compile_program(mod, "__npu_fusion_probe", ...)',
      '    return analyze(asm.words, plan, detail=False)   # DMA/words/SRAM',
      '',
      'reference = _probe(outlined, fn, profile)         # unfused: 필수 기준',
      'candidate = _probe(fused, _wrapper(gv, fn), profile)',
      'selected  = cost(candidate) < cost(reference)      # (dma, words, sram) 사전식',
      'if not selected: rejected[gv] = fn',
      '...  outlined[gv] = fn.inline_functions(rejected)   # 진 영역만 펼침',
    ], { size: 8.5 });
    D.text(s, 7.3, 2.85, 5.5, 0.3, '작은 예제의 실제 기록 (program.json → fusion.regions)', { size: 10.5, bold: true, color: C.navy });
    D.table(s, 7.3, 3.15, 5.53, [
      ['항목', 'unfused', 'fused'],
      ['region', 'fused_matmul_add_silu', ''],
      ['dma_bytes', '41,100', '41,100'],
      ['words', '2,812', '1,577'],
      ['sram_peak_bytes', '50,700', '42,508'],
      ['status', '', 'selected ("lexicographic DMA bytes, words, SRAM peak")'],
    ], { colW: [1.45, 1.6, 2.48], size: 9, rowH: 0.32 });
    D.bullets(s, 7.3, 5.1, 5.53, 0.5, ['unfused probe 실패 = unsupported 입력 또는 compiler 오류 → 오류를 숨기지 않음'], { size: 9.5 });
    D.callout(s, 0.5, 5.7, 12.33, 0.65, [
      { text: '부분 검사의 한계: ', bold: true }, { text: '한 영역의 fused/unfused만 봅니다. 주변 그래프의 모든 수명·다른 선택과의 조합은 모릅니다. 그래서 다음 슬라이드의 전체 module 비교가 따로 필요하고, 버린 영역을 모든 조합과 함께 재탐색하는 완전 탐색은 하지 않습니다.', size: 10 },
    ], { kind: 'faint' });
  });

  // ---------------------------------------------------------------- 28 lower_and_select
  D.add({ section: '5 · Fusion 후보 선택', title: 'npu_build.lower_and_select: 같은 IR인 source를 먼저 제거하고, 전체 entry의 비용으로 module 하나를 고른다', subtitle: '최대 27개 조합 중 structural_equal인 source는 generic fusion 검사 전에 버립니다 (E10 prefill: source 9 → 3, 후보 9). 남은 후보는 entry마다 컴파일한 (DMA, words, SRAM)을 사전식으로 비교하며, 첫 합법 module 뒤 추가 탐색은 기본 120초, None이면 제한 없음입니다.', source: 'npu_compiler/npu_build.py lower_and_select(), npu_compilation.optimization_budget, evidence/current/llama_selection.json (7-token 기록), llama_prefill256_full_search_progress.json (E10)',
    notes: 'options()는 attention(False/score/full) × (gqa, producers) 세 구성마다 graph_pipeline으로 source를 만들고, 이미 만든 source와 structural_equal이면 generic 3종을 검사하지 않고 건너뜁니다. E10의 256-token prefill_cache에서는 GQA fold와 producer 변환이 IR을 바꾸지 않아 9개 source 중 3개만 남고, generic off/vectors/matmul을 곱한 9개 구성이 검사 대상입니다(회색 18칸). decode를 함께 요청하는 경우에는 결과가 다를 수 있습니다. generic이 켜지면 npu_fusion.select로 후보를 만들고 이미 본 후보와 같으면 duplicate로 기록합니다. compile_all은 entry마다 EntryCache.compile을 호출하며, 완료된 entry 하나가 incumbent보다 나쁘면 나머지 entry를 건너뛰고 noncompetitive로 기록합니다. 채택 조건은 모든 entry가 비악화이고 하나 이상이 개선인 경우이며, 첫 합법 결과 뒤 graph_search_seconds 예산을 시작합니다(기본 120초, None이면 시간 제한 없음, phase 경계에서만 검사). schedule_database가 설정된 경우 후보 module마다 relax_integration.extract_tasks로 작업 종류·호출 수 목록을 진단용으로 남깁니다. 오른쪽 격자의 ✓/⏱은 과거 7-token 기록입니다: 후보 0만 legal·selected, 후보 1에서 예산 소진.' },
  (s, D) => {
    D.code(s, 0.5, 1.5, 6.9, 3.9, [
      'for attention in (False, "score", "full"):                     # 3',
      '  for gqa, producers in ((F,F), (T,F), (T,T)):                 # × 3',
      '    source = lower(attention, gqa, producers)                  # graph_pipeline',
      '    if any(structural_equal(source, old) for old in sources): continue',
      '    for generic in (False, "vectors", "matmul"):               # × 3 = 27',
      '      candidate, regions = select(source, ...) if generic else (source, None)',
      '      if duplicate(candidate): continue',
      '      programs, stats, regressing = compile_all(candidate, best_costs)',
      '      costs = {entry: cost(stats[entry]) for entry in entries}',
      '      selected = regressing is None and (best_costs is None or',
      '          (all(costs[e] <= best_costs[e] for e in entries)',
      '           and any(costs[e] <  best_costs[e] for e in entries)))',
      '      if selected: winner, best_costs = candidate, costs; arm_budget()',
      '',
      '# E10 (prefill_cache, S=256): source 9 → 3 (GQA/producer 변환이 같은 IR) → 후보 9',
      '# compile_all: 완료된 entry가 incumbent보다 나쁘면 즉시 중단 (noncompetitive)',
      '# cost(stats) = (dma_bytes, words, sram_peak_bytes)   # v09_stats.cost',
    ], { size: 8.5 });
    D.table(s, 7.6, 1.5, 5.23, [
      ['후보 기록 status', '의미'],
      ['legal', '모든 entry 컴파일 완료, 비용 기록 (Pareto 후보)'],
      ['noncompetitive', '한 entry가 이미 더 나쁨 → 나머지 entry 생략'],
      ['unsupported', 'LinkError / V09TirError (지원 밖 schedule 등)'],
      ['duplicate', '동일 IR 후보 (structural_equal). source 중복은 기록 없이 건너뜀'],
      ['budget_exhausted', '추가 탐색 예산 초과 (기본 120 s, None이면 발생하지 않음)'],
    ], { colW: [1.5, 3.73], size: 9, rowH: 0.34 });
    // candidate grid: 3 (attention) x 9 (gqa/producers x generic); gray = identical source (E10 prefill)
    const gx = 8.55, gy = 4.05, gw = 4.28, gh = 1.2;
    D.text(s, 7.6, 3.62, 5.23, 0.24, '27 조합 격자 — 회색 18칸 = 같은 IR인 source (E10 prefill)', { size: 8.5, bold: true, color: C.navy });
    const cellLabel = (r, c) => (r === 0 && c === 0) ? '✓' : ((r === 0 && c === 1) ? '⏱' : (c >= 3 ? '≡' : ''));
    D.matrix(s, gx, gy, gw, gh, 3, 9, { kind: 'ir', tile: 3, cellLabel, cellSize: 8, regions: [
      { r0: 0, r1: 3, c0: 3, c1: 9, kind: 'faint', fill: 'F3F5F8', dash: 'dash' },
      { r0: 0, r1: 1, c0: 0, c1: 1, kind: 'good' }, { r0: 0, r1: 1, c0: 1, c1: 2, kind: 'warn' },
    ] });
    ['off', 'score', 'full'].forEach((t, i) => D.text(s, 7.6, gy + i * gh / 3, 0.9, gh / 3, 'attention ' + t, { size: 7.5, align: 'right', valign: 'middle', color: C.muted }));
    ['gqa off\nprod off', 'gqa on\nprod off', 'gqa on\nprod on'].forEach((t, i) => D.text(s, gx + i * gw / 3, gy + gh + 0.03, gw / 3, 0.3, t, { size: 6.5, align: 'center', color: C.muted }));
    D.text(s, gx, gy - 0.2, gw, 0.18, 'generic: off · vectors · matmul (3칸씩 반복) — 파란 9칸이 E10의 검사 대상', { size: 6.5, color: C.muted, align: 'center' });
    D.text(s, 7.6, 5.6, 5.23, 0.75, '✓ 후보 0 legal·selected (모두 off)  ⏱ 후보 1 (generic vectors) 검사 중 120 s 소진: 과거 7-token 기록(compile 189.7 s). ≡ E10 prefill에서 source가 같은 조합. 예산은 첫 합법 module 뒤 시작하며 phase 경계에서 검사됩니다.', { size: 8, color: C.muted });
    D.callout(s, 0.5, 5.6, 6.9, 0.75, [
      { text: '두 단계가 다른 질문에 답합니다: ', bold: true }, { text: '"이 영역을 합칠 가치가 있는가?"(npu_fusion.select)와 "주변을 포함한 프로그램을 이렇게 선택해도 좋아지는가?"(lower_and_select). 중복 계산은 region/Entry cache로 줄이고, 시간 제한을 풀어도(None) 부분 검사에서 버린 영역의 다른 조합까지 전수 탐색하지는 않습니다.', size: 9.5 },
    ], { kind: 'faint' });
  });

  // ---------------------------------------------------------------- 29 real Llama selection record
  D.add({ section: '5 · Fusion 후보 선택', title: '과거 7-token Llama 선택 기록: 120초 안에서 unfused 기준이 선택되었다', subtitle: 'compile 189.7 s. 후보 0(모두 off)이 legal·selected, 후보 1(generic vectors)에서 추가 탐색 예산 120 s가 소진. "모든 fusion이 더 나쁘다"는 증거가 아닙니다.', source: 'docs/evidence/current/llama_selection.json (과거 7-token 보존 기록; 새 256-token E10과 구분)',
    notes: 'phase별 self seconds를 보면 native codegen 41.5 s, schedule probe codegen 29.3 s, residency 22.4 s, entry compile 자체 20.8 s가 큽니다. cache 통계는 schedule recipe hit 272 / miss 383, trial hit 184 / miss 248, cost probe hit 294 / miss 270, region miss 40입니다. native가 거절한 schedule은 "FP32 matrix chain interrupted" 5건입니다. 후보 1은 vectors generic fusion을 검사하던 중 예산이 끝났으므로 이후 후보(attention score/full 등)는 검사되지 않았습니다.' },
  (s, D) => {
    D.table(s, 0.5, 1.5, 5.6, [
      ['후보', 'attention', 'gqa/producers', 'generic', 'status'],
      ['0', 'off', 'off / off', 'off', 'legal → selected'],
      ['1', 'off', 'off / off', 'vectors', 'budget_exhausted'],
      ['2~26', '(미검사)', '', '', '예산 소진'],
    ], { colW: [0.6, 1.0, 1.4, 1.0, 1.6], size: 9.5, rowH: 0.32 });
    D.table(s, 0.5, 3.0, 5.6, [
      ['entry (후보 0)', 'words', 'DMA bytes', 'SRAM peak', 'image bytes'],
      ['prefill_cache (S=7)', '8,582,352', '6,457,527,556', '8,374,292', '6,429,587,576'],
      ['decode (C=9)', '7,310,801', '6,438,980,676', '7,342,100', '6,427,349,032'],
    ], { colW: [1.35, 0.95, 1.25, 0.85, 1.2], size: 8, rowH: 0.34, aligns: ['left', 'right', 'right', 'right', 'right'] });
    D.text(s, 0.5, 4.25, 5.6, 0.3, 'Cache 통계 (compilation.describe)', { size: 10.5, bold: true, color: C.navy });
    D.table(s, 0.5, 4.55, 5.6, [
      ['cache', 'hit', 'miss'],
      ['schedule recipe (PrimFunc 구조별)', '272', '383'],
      ['schedule trial (mapping별)', '184', '248'],
      ['cost probe', '294', '270'],
      ['region (fusion 영역)', '0', '40'],
      ['native_rejections', 'FP32 matrix chain interrupted', '5'],
    ], { colW: [3.0, 1.6, 1.0], size: 9, rowH: 0.3, aligns: ['left', 'right', 'right'] });
    // native chart: phase self seconds
    s.addChart(D.pres.charts.BAR, [{ name: 'self seconds', labels: ['native_codegen', 'schedule_probe_codegen', 'residency', 'entry_compile(self)', 'entry_statistics', 'schedule_tir', 'bound_kernel_codegen', 'memory_planning', 'schedule_probe_peephole', 'joint_schedule_memory', 'schedule_sram', 'schedule_and_sram'], values: [41.47, 29.32, 22.38, 20.81, 8.52, 4.82, 3.72, 3.08, 2.83, 1.62, 1.44, 1.44] }], {
      x: 6.4, y: 1.5, w: 6.43, h: 4.5, barDir: 'bar', chartColors: ['1F8A8A'], showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 8, dataLabelFormatCode: '0.0',
      showTitle: true, title: 'Compile phase self seconds (합계 ≈ 188 s, 단일 관측)', titleFontSize: 11, titleColor: C.navy,
      catAxisLabelFontSize: 8.5, valAxisLabelFontSize: 8, catAxisLabelColor: C.ink, valAxisLabelColor: C.muted, valGridLine: { color: 'E1E7EF', size: 0.5 }, catGridLine: { style: 'none' }, showLegend: false, valAxisMaxVal: 50, catAxisOrientation: 'maxMin',
    });
    D.takeaway(s, '이 검증에서 채택된 것은 unfused 기준 module입니다. 비용이 큰 phase(native codegen·probe·residency)가 곧 탐색 폭을 제한하는 요인입니다.', { y: 6.5 });
  });
};
