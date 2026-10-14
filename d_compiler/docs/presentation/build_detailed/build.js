'use strict';
const path = require('path');
const { Deck } = require('./lib');
const parts = process.argv.slice(3).length ? process.argv.slice(3) : ['p1_intro','p2_frontend','p3_tir_fusion','p4_schedule','p5_vector_attn','p6_memory','p7_isa','p8_runtime','p9_experiments','p10_limits'];
const out = process.argv[2] || 'deck.pptx';
const D = new Deck({ title: 'TVM 기반 LLM NPU Compiler', author: 'NPU compiler' });
for (const p of parts) {
  let mod;
  try { mod = require('./' + p); } catch (e) { if (e.code === 'MODULE_NOT_FOUND' && e.message.includes(p)) { console.log('skip missing part', p); continue; } throw e; }
  mod(D);
}
const pres = D.build();
// speaker notes markdown next to the deck
const fs = require('fs');
const notesTitle = process.env.NOTES_TITLE || 'NPU compiler 발표자 노트 (상세판)';
const notesIntro = process.env.NOTES_INTRO || '슬라이드별 설명 순서, 세부 역할, 해석 시 주의점과 근거 경로. 본문은 약 90분 분량이며 파트를 선택해 줄일 수 있다.';
const lines = ['# ' + notesTitle, '', notesIntro, ''];
D.specs.forEach((sp, i) => {
  const m = sp.meta || {};
  const title = m.title || (m.dark ? (i === 0 ? '표지' : '마무리') : '');
  lines.push(`## ${String(i + 1).padStart(2, '0')}. ${title}`.trim(), '');
  if (m.section) lines.push(`_${m.section}_`, '');
  if (m.subtitle) lines.push(m.subtitle, '');
  if (m.notes) lines.push(m.notes, '');
  if (m.source) lines.push(`근거: ${m.source}`, '');
});
fs.writeFileSync(out.replace(/\.pptx$/, '') + '.notes.md', lines.join('\n'));
pres.writeFile({ fileName: out }).then(f => console.log('wrote', f, 'slides:', D.specs.length));
