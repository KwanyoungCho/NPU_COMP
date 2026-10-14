'use strict';
const { C, evidence } = require('./lib');

module.exports = (D, { pendingOnly = false } = {}) => {
  const evidence_=evidence('llama_schedule_refinement.json');
  const a=evidence_.summary.find(r=>r.mode==='space'), b=evidence_.summary.find(r=>r.mode==='meta_schedule');
  const stat=(r,key)=>r.statistics[key].toLocaleString('en-US');
  if (!pendingOnly) D.add({ section: '10 · 실험 · 탐색 개선', title: 'E9. 탐색을 보완해도 512-token down의 최종 명령은 같았다',
    subtitle: '실제 Llama layer-0 weight + 합성 activation. 단일 projection의 비교이며 전체 모델 generation 검증이 아닙니다.',
    source: 'docs/current/11_experiments.md E9; evidence/current/llama_schedule_refinement.json',
    notes: '후보 정규화, 직접 FP16 입력에 한정한 SRAM 초과 사전 제외, 기준 words를 반영한 예산을 적용한 뒤 두 방식 모두 추가 합법 후보4개를 검사했습니다. Space10.586초, ReplayFunc33.264초이며 최종 binary는 같습니다. 이 후보 범위에서 기존 schedule보다 나은 정적 비용을 얻지 못했습니다. C-model은 최대 절대차0.000030517578125와 DMA counters 일치를 확인했습니다. DB reuse는 약4초이며 탐색 전략과 예산을 바꾸어도 호환되는 trace를 재사용합니다. 관련 회귀 검사34개가 통과했습니다. 전체 탐색공간의 최적성 또는 실제 장치 latency를 주장하지 않습니다.' }, (s, D) => {
    D.table(s, 0.5, 1.55, 7.5, [
      ['측정 / 512-token down', 'Space', 'TVM ReplayFunc'],
      ['추가 합법 후보', `${a.after_legal}개`, `${b.after_legal}개`],
      ['커널 탐색 시간', `${a.tune_seconds.toFixed(3)} s`, `${b.tune_seconds.toFixed(3)} s`],
      ['DMA bytes', stat(a,'dma_bytes'), stat(b,'dma_bytes')],
      ['ISA words', stat(a,'words'), stat(b,'words')],
      ['SRAM peak', stat(a,'sram_peak_bytes')+' B', stat(b,'sram_peak_bytes')+' B'],
      ['binary / C-model', '동일 binary', '같은 프로그램 출력 검증'],
    ], { colW: [3.0, 2.25, 2.25], rowH: 0.48, size: 12 });
    D.bullets(s, 8.35, 1.65, 4.48, 3.9, [
      ['낭비를 줄이는 변경: ', '같은 순회·복사 주기의 후보를 정규화하고, 증명할 수 있는 용량 초과만 제거합니다.'],
      ['충분한 검사인지 확인: ', '시간만 보지 않고 추가 합법 후보 수와 종료 사유를 기록합니다.'],
      ['관측한 한계: ', '이 커널에서는 ReplayFunc의 비용이 더 컸고 최종 명령은 같았습니다. 학습형 비용 모델은 사용하지 않았습니다.'],
      ['다음 검증: ', 'Fusion 효과는 전체 28-layer·256-token prefill을 별도로 컴파일해 확인합니다.'],
    ], { size: 13, psa: 10 });
    D.callout(s, 0.5, 5.65, 12.33, 0.62, '출력 검증 통과: C-model 최대 절대차 0.000030517578125 · DMA counters 일치 · 관련 회귀 34개 통과', { kind: 'good', size: 12 });
    D.takeaway(s, '불필요한 시도 감소와 trace 재사용을 확인했습니다. 추가 탐색까지 포함한 총 시간이 더 짧아졌다는 비교는 아닙니다.', { y: 6.5 });
  });
  const progress = evidence('llama_prefill256_full_search_progress.json');
  if (progress.graph_inventory.graph_configurations !== 9 || progress.request.prefill_tokens !== 256) throw new Error('Update E10 slide for new experiment configuration');
  D.add({ section: '10 · 실험 · 전체 Llama 재검증', title: 'E10 중단: 큰 cache fusion의 명령 생성 병목을 수정했다',
    subtitle: '28 layers 전체 · prefill_cache 하나 · graph 시간 제한 None · matmul 합법 목표16 / 최대 시도128 · SRAM 8 MiB · 전체 재검증 미실행',
    source: 'docs/current/11_experiments.md E10; evidence/current/llama_prefill256_full_search_progress.json',
    notes: 'Attention off/score/full 3종과 generic off/vectors/matmul 3종입니다. 이 prefill에서는 GQA·producer 설정이 동일 IR이어서 source 중복을 제거한 뒤9개가 남습니다. 표는 일반 fusion 영역 함수 종류이며 layer별 호출 수나 최종 채택 수가 아닙니다. 영역 검사 후 풀리는 fusion과 추가 중복도 가능합니다. 분리된 커널도 SRAM 보관과 직접 출력을 사용하므로 공정하게 비교합니다. Graph 시간제한은 없지만 matmul 시도128·합법목표16·words예산10억, 메모리 beam4이므로 모든 schedule/보관 조합의 완전탐색은 아닙니다. 같은 계산·메모리 조건은 재사용합니다. HF last logits와28-layer K/V reference는 준비됐고 compile은 과도한 메모리로 중단했고 C-model 비교는 미실행입니다. 생산자 반복 배치 거절·직사각형 행 분할·길이1 DMA 처리를 수정하여 독립 검사4개와 회귀54개가 통과했습니다. 비용이나 fusion 채택, NPU token일치 결과는 아직 확정하지 않습니다.' }, (s, D) => {
    D.table(s, 0.5, 1.62, 7.5, [
      ['Attention / Generic', 'Off', 'Vectors', 'Matmul'],
      ['off', '기준 구성', '영역 10종', '영역 11종'],
      ['score', '1개 구성', '영역 9종', '영역 10종'],
      ['full', '1개 구성', '영역 9종', '영역 10종'],
    ], { colW: [2.4, 1.5, 1.8, 1.8], rowH: 0.64, size: 13 });
    D.text(s, 0.55, 4.28, 7.35, 0.5, '9개 = graph 구성 수. 영역 수는 종류이며 최종 채택 수가 아닙니다.\n동일 layer의 판단은 재사용하고 실제 주소와 전체 수명은 다시 연결합니다.', { size: 12, color: C.muted });
    const stages = [ ['HF reference', '마지막 logits + 모든 layer K/V', 'good'], ['전체 compile 중단', 'RSS 91.65 GiB / 병목 수정 후 검증 필요', 'warn'], ['선택된 명령 실행', 'C-model token·cosine·DMA 비교', 'faint'] ];
    stages.forEach((a, i) => {
      D.box(s, 8.55, 1.65 + i * 1.13, 3.95, 0.85, a[0], { kind: a[2], detail: a[1], size: 15, dsize: 11 });
      if (i < 2) D.arrow(s, 10.52, 2.52 + i * 1.13, 10.52, 2.73 + i * 1.13);
    });
    D.callout(s, 0.5, 5.28, 12.33, 0.9, '비교 기준도 SRAM 보관 ON. Fusion이 전체 prefill의 DMA·ISA words·SRAM을 더 줄이는지 확인합니다.\n큰 K/V 출력은 각각14 MiB. Decode를 실행하지 않아도 이 prefill 함수가 만드는 출력입니다.', { kind: 'ir', size: 12 });
    D.takeaway(s, '병목 독립 검사는 통과했습니다. 전체256-token 재컴파일·출력 검증은 아직 수행하지 않았습니다.', { y: 6.5 });
  });
};
