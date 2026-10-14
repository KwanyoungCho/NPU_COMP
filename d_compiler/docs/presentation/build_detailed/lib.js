// Helper library for the NPU compiler deck (pptxgenjs).
// Canvas: LAYOUT_WIDE 13.333 x 7.5 in. All coordinates in inches.
'use strict';
const pptxgen = require('pptxgenjs');

const C = {
  navy: '15243B', ink: '1F2D3D', muted: '5B6B7F', line: 'C9D3DF', faint: 'EEF2F6',
  bg: 'FFFFFF', dark: '0F1B2D', darkText: 'E6EDF5', darkMuted: '9FB3C8',
  dram: 'D98A1E', dramFill: 'FFF1DC',
  sram: '1F8A8A', sramFill: 'DDF3F0',
  unit: '6C4FBF', unitFill: 'ECE6FA',
  ir: '2F6DB5', irFill: 'E3EEFB',
  host: '5B6B7F', hostFill: 'EEF1F5',
  red: 'C0392B', redFill: 'FBE9E7',
  warn: 'B7791F', warnFill: 'FFF6DB',
  good: '2E7D32', goodFill: 'E5F3E7',
  code: 'F5F7FA', codeInk: '1F2D3D', codeKey: '2F6DB5', codeStr: '2E7D32',
  white: 'FFFFFF',
};
const F = { body: 'Malgun Gothic', code: 'Courier New' };
const W = 13.333, H = 7.5;

function kindColors(kind) {
  const m = {
    dram: [C.dramFill, C.dram], sram: [C.sramFill, C.sram], unit: [C.unitFill, C.unit],
    ir: [C.irFill, C.ir], host: [C.hostFill, C.host], red: [C.redFill, C.red],
    warn: [C.warnFill, C.warn], good: [C.goodFill, C.good], plain: [C.white, C.line],
    faint: [C.faint, C.line], navy: [C.navy, C.navy], dark: [C.dark, C.dark],
  };
  return m[kind] || m.plain;
}

// estimate text width in em units (CJK = 1em, others ~0.56em) and shrink font to fit `lines` lines of `width` inches
function textUnits(str) {
  let u = 0;
  for (const ch of String(str)) {
    const c = ch.codePointAt(0);
    if ((c >= 0x1100 && c <= 0x11FF) || (c >= 0x3000 && c <= 0x9FFF) || (c >= 0xAC00 && c <= 0xD7AF) || (c >= 0xFF00 && c <= 0xFFEF)) u += 1.0;
    else if (ch === ' ') u += 0.3; else if (/[A-Z0-9]/.test(ch)) u += 0.66; else u += 0.52;
  }
  return u;
}
function fitSize(str, size, width, lines, minSize) {
  const units = textUnits(str) * 1.06;
  for (let sz = size; sz >= minSize; sz -= 0.5) {
    if (units * sz / 72 <= width * lines) return sz;
  }
  return minSize;
}

class Deck {
  constructor(opts = {}) {
    this.pres = new pptxgen();
    this.pres.layout = 'LAYOUT_WIDE';
    this.pres.author = opts.author || 'NPU compiler';
    this.pres.title = opts.title || 'NPU compiler';
    this.specs = [];
    this.footerLeft = opts.footerLeft || 'TVM 기반 NPU compiler · v09 · FP16';
  }
  // register a slide builder: fn(slide, deck, index, total)
  add(meta, fn) { this.specs.push({ meta, fn }); }
  build() {
    const total = this.specs.length;
    this.specs.forEach((spec, i) => {
      const s = this.pres.addSlide();
      const m = spec.meta || {};
      const dark = !!m.dark;
      s.background = { color: dark ? C.dark : C.bg };
      if (!dark && !m.bare) {
        if (m.section) this.text(s, 0.5, 0.22, 9, 0.28, m.section, { size: 10.5, color: C.sram, bold: true, charSpacing: 1 });
        if (m.title) this.text(s, 0.5, 0.46, 12.3, 0.6, m.title, { size: fitSize(m.title, 24, 12.2, 1, 17), color: C.navy, bold: true });
        if (m.subtitle) this.text(s, 0.5, 1.02, 12.3, 0.45, m.subtitle, { size: fitSize(m.subtitle, 12.5, 12.3, 2, 10), color: C.muted });
      }
      spec.fn(s, this, i + 1, total);
      if (!m.bare) this.footer(s, i + 1, total, m.source, dark);
      if (m.notes) s.addNotes(m.notes);
    });
    return this.pres;
  }
  footer(s, index, total, source, dark) {
    const col = dark ? C.darkMuted : C.muted;
    this.text(s, 0.5, 7.08, 4.5, 0.25, this.footerLeft, { size: 8.5, color: col });
    if (source) this.text(s, 4.2, 7.08, 7.4, 0.25, '근거: ' + source, { size: 8.5, color: col, align: 'center' });
    this.text(s, 11.8, 7.08, 1.03, 0.25, `${index} / ${total}`, { size: 8.5, color: col, align: 'right' });
  }
  // ---- primitives ---------------------------------------------------------
  text(s, x, y, w, h, text, o = {}) {
    const runs = Array.isArray(text) ? text : [{ text: String(text) }];
    const base = {
      x, y, w, h, fontFace: o.font || F.body, fontSize: o.size || 12, color: o.color || C.ink,
      bold: !!o.bold, italic: !!o.italic, align: o.align || 'left', valign: o.valign || 'top',
      margin: o.margin === undefined ? 0 : o.margin, isTextBox: true, fit: o.fit,
      charSpacing: o.charSpacing, lineSpacingMultiple: o.lsm, paraSpaceAfter: o.psa,
    };
    if (o.fill) base.fill = { color: o.fill };
    if (o.lineColor) base.line = { color: o.lineColor, width: o.lineWidth || 0.75 };
    if (o.shape) base.shape = o.shape;
    if (o.rectRadius) base.rectRadius = o.rectRadius;
    if (o.wrap === false) base.wrap = false;
    s.addText(runs, base);
  }
  rich(s, x, y, w, h, runs, o = {}) {
    // runs: [{text, bold, color, size, font, breakLine, bullet}]
    const norm = runs.map(r => ({ text: r.text, options: {
      bold: !!r.bold, color: r.color || o.color || C.ink, fontSize: r.size || o.size || 12,
      fontFace: r.font || o.font || F.body, italic: !!r.italic, breakLine: !!r.breakLine,
      bullet: r.bullet ? (typeof r.bullet === 'object' ? r.bullet : { indent: 12 }) : undefined,
      paraSpaceAfter: r.psa !== undefined ? r.psa : o.psa, indentLevel: r.indent,
    } }));
    s.addText(norm, { x, y, w, h, margin: o.margin === undefined ? 0 : o.margin, isTextBox: true,
      valign: o.valign || 'top', align: o.align || 'left', fill: o.fill ? { color: o.fill } : undefined,
      lineSpacingMultiple: o.lsm });
  }
  bullets(s, x, y, w, h, items, o = {}) {
    const size = o.size || 12;
    const runs = [];
    items.forEach((it, i) => {
      const last = i === items.length - 1;
      if (typeof it === 'string') {
        runs.push({ text: it, bullet: true, breakLine: !last, size, color: o.color, psa: o.psa === undefined ? 6 : o.psa });
      } else if (Array.isArray(it)) {
        // [head, rest] -> bold head + normal rest, same paragraph
        runs.push({ text: it[0], bold: true, bullet: true, size, color: o.color });
        runs.push({ text: it[1], breakLine: !last, size, color: o.color, psa: o.psa === undefined ? 6 : o.psa });
      } else {
        runs.push(Object.assign({ bullet: true, breakLine: !last, size, psa: o.psa === undefined ? 6 : o.psa }, it));
      }
    });
    this.rich(s, x, y, w, h, runs, o);
  }
  rect(s, x, y, w, h, o = {}) {
    const [fill, line] = kindColors(o.kind || 'plain');
    s.addShape(o.radius === 0 ? this.pres.shapes.RECTANGLE : this.pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w, h, fill: { color: o.fill || fill }, line: { color: o.line || line, width: o.lineWidth === undefined ? 1 : o.lineWidth },
      rectRadius: o.radius === undefined ? 0.06 : o.radius, shadow: o.shadow,
    });
  }
  // box with centered title (+ optional detail below)
  box(s, x, y, w, h, title, o = {}) {
    this.rect(s, x, y, w, h, o);
    const size = o.size || 12;
    const dsize = o.dsize || Math.max(8.5, size - 3);
    const [, line] = kindColors(o.kind || 'plain');
    const tcol = o.color || (o.kind === 'navy' || o.kind === 'dark' ? C.white : C.ink);
    if (o.detail) {
      const th = size * 1.5 / 72;
      const dh = h - th - 0.12;
      this.text(s, x + 0.08, y + 0.08, w - 0.16, th, title, { size, bold: o.bold !== false, color: tcol, align: o.align || 'center', valign: 'middle', font: o.font });
      this.text(s, x + 0.08, y + 0.08 + th, w - 0.16, Math.max(dh, 0.2), o.detail, { size: dsize, color: o.dcolor || (o.kind === 'navy' || o.kind === 'dark' ? C.darkMuted : C.muted), align: o.align || 'center', valign: 'top', font: o.dfont });
    } else {
      this.text(s, x + 0.06, y + 0.04, w - 0.12, h - 0.08, title, { size, bold: o.bold !== false, color: tcol, align: o.align || 'center', valign: o.valign || 'middle', font: o.font });
    }
    return { x, y, w, h, cx: x + w / 2, cy: y + h / 2, r: x + w, b: y + h, color: line };
  }
  line(s, x1, y1, x2, y2, o = {}) {
    const x = Math.min(x1, x2), y = Math.min(y1, y2);
    const w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
    s.addShape(this.pres.shapes.LINE, {
      x, y, w: Math.max(w, 0.001), h: Math.max(h, 0.001),
      flipH: x2 < x1, flipV: y2 < y1,
      line: { color: o.color || C.muted, width: o.width || 1.5, dashType: o.dash || 'solid',
        endArrowType: o.arrow === false ? undefined : (o.arrow || 'triangle'), beginArrowType: o.begin },
    });
  }
  arrow(s, x1, y1, x2, y2, o = {}) { this.line(s, x1, y1, x2, y2, Object.assign({ arrow: 'triangle' }, o)); }
  // orthogonal (elbow) arrow: horizontal then vertical then horizontal
  elbow(s, x1, y1, x2, y2, o = {}) {
    const mx = o.mx !== undefined ? o.mx : (x1 + x2) / 2;
    this.line(s, x1, y1, mx, y1, Object.assign({}, o, { arrow: false }));
    this.line(s, mx, y1, mx, y2, Object.assign({}, o, { arrow: false }));
    this.line(s, mx, y2, x2, y2, o);
  }
  velbow(s, x1, y1, x2, y2, o = {}) {
    const my = o.my !== undefined ? o.my : (y1 + y2) / 2;
    this.line(s, x1, y1, x1, my, Object.assign({}, o, { arrow: false }));
    this.line(s, x1, my, x2, my, Object.assign({}, o, { arrow: false }));
    this.line(s, x2, my, x2, y2, o);
  }
  code(s, x, y, w, h, lines, o = {}) {
    const size = o.size || 10;
    this.rect(s, x, y, w, h, { fill: o.fill || C.code, line: o.line || C.line, radius: 0.05, lineWidth: 0.75 });
    const runs = [];
    const arr = Array.isArray(lines) ? lines : String(lines).split('\n');
    arr.forEach((ln, i) => {
      const last = i === arr.length - 1;
      if (typeof ln === 'string') runs.push({ text: ln.length ? ln : ' ', breakLine: !last, size, font: F.code, color: o.color || C.codeInk });
      else {
        // array of segments {text,color,bold}
        ln.forEach((seg, j) => runs.push({ text: seg.text, color: seg.color || C.codeInk, bold: !!seg.bold, size, font: F.code, breakLine: !last && j === ln.length - 1 }));
      }
    });
    this.rich(s, x + 0.12, y + 0.08, w - 0.24, h - 0.16, runs, { size, font: F.code, lsm: o.lsm || 1.05 });
    if (o.caption) this.text(s, x, y + h + 0.03, w, 0.22, o.caption, { size: 8.5, color: C.muted, italic: true });
  }
  callout(s, x, y, w, h, text, o = {}) {
    const [fill, line] = kindColors(o.kind || 'sram');
    this.rect(s, x, y, w, h, { fill, line, radius: 0.06, lineWidth: 1 });
    const runs = Array.isArray(text) ? text : [{ text: String(text), size: o.size || 12, bold: !!o.bold }];
    this.rich(s, x + 0.14, y + 0.06, w - 0.28, h - 0.12, runs.map(r => Object.assign({ size: o.size || 12 }, r)), { valign: 'middle', color: o.color || C.ink });
  }
  takeaway(s, text, o = {}) {
    const y = o.y === undefined ? 6.5 : o.y;
    this.rect(s, 0.5, y, 12.33, 0.46, { fill: C.faint, line: C.faint, radius: 0.05, lineWidth: 0 });
    this.text(s, 0.68, y + 0.06, 12.0, 0.34, text, { size: o.size || 12.5, bold: true, color: C.navy, valign: 'middle' });
  }
  chip(s, x, y, text, o = {}) {
    const [fill, line] = kindColors(o.kind || 'ir');
    const w = o.w || (0.16 + text.length * 0.085);
    this.text(s, x, y, w, o.h || 0.26, text, { size: o.size || 9, bold: true, color: o.color || line, align: 'center', valign: 'middle', fill, lineColor: line, shape: this.pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.13 });
    return w;
  }
  table(s, x, y, w, rows, o = {}) {
    const size = o.size || 10.5;
    const colW = o.colW || rows[0].map(() => w / rows[0].length);
    const data = rows.map((r, ri) => r.map((cell, ci) => {
      const isHead = ri === 0 && o.header !== false;
      const cellObj = typeof cell === 'object' && cell !== null && !Array.isArray(cell) ? cell : { text: String(cell) };
      const opts = Object.assign({
        fontFace: cellObj.font || F.body, fontSize: cellObj.size || size, color: isHead ? C.white : (cellObj.color || C.ink),
        bold: isHead || !!cellObj.bold, fill: { color: isHead ? C.navy : (cellObj.fill || (ri % 2 === 0 ? C.white : C.faint)) },
        align: cellObj.align || (o.aligns ? o.aligns[ci] : 'left'), valign: 'middle',
        margin: [3, 5, 3, 5], border: { type: 'solid', pt: 0.5, color: C.line },
      }, cellObj.options || {});
      return { text: cellObj.text, options: opts };
    }));
    s.addTable(data, { x, y, w, colW, rowH: o.rowH || 0.3, autoPage: false });
  }
  image(s, path, x, y, w, h) { s.addImage({ path, x, y, w, h }); }
  // matrix / tile grid drawing: rows x cols cells
  grid(s, x, y, w, h, rows, cols, o = {}) {
    const cw = w / cols, ch = h / rows;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const hl = o.highlight && o.highlight(r, c);
      const [fill, line] = kindColors(hl || o.kind || 'faint');
      s.addShape(this.pres.shapes.RECTANGLE, { x: x + c * cw, y: y + r * ch, w: cw, h: ch, fill: { color: fill }, line: { color: o.lineColor || line, width: 0.5 } });
      if (o.label) {
        const t = o.label(r, c);
        if (t) this.text(s, x + c * cw, y + r * ch, cw, ch, t, { size: o.size || 8, align: 'center', valign: 'middle', color: C.ink });
      }
    }
  }
  // memory bar: segments [{label, from, to, kind, sub}] over [lo, hi]
  membar(s, x, y, w, h, lo, hi, segs, o = {}) {
    const scale = w / (hi - lo);
    this.rect(s, x, y, w, h, { fill: C.faint, line: C.line, radius: 0, lineWidth: 0.75 });
    segs.forEach(g => {
      const gx = x + (g.from - lo) * scale, gw = (g.to - g.from) * scale;
      const [fill, line] = kindColors(g.kind || 'ir');
      s.addShape(this.pres.shapes.RECTANGLE, { x: gx, y, w: gw, h, fill: { color: fill }, line: { color: line, width: 1 } });
      if (g.label) this.text(s, gx, y, gw, h * (g.sub ? 0.55 : 1), g.label, { size: o.size || 10, bold: true, align: 'center', valign: 'middle', color: C.ink });
      if (g.sub) this.text(s, gx, y + h * 0.5, gw, h * 0.45, g.sub, { size: (o.size || 10) - 2, align: 'center', valign: 'middle', color: C.muted });
    });
    if (o.ticks) o.ticks.forEach(t => {
      const tx = x + (t - lo) * scale;
      this.line(s, tx, y + h, tx, y + h + 0.08, { arrow: false, color: C.muted, width: 0.75 });
      this.text(s, tx - 0.5, y + h + 0.09, 1.0, 0.2, String(t), { size: 8, align: 'center', color: C.muted });
    });
  }
  // bit-field diagram: fields [{name, hi, lo, kind, value}] for a 32-bit word (msb left)
  bitfield(s, x, y, w, h, fields, o = {}) {
    const bits = o.bits || 32;
    const bw = w / bits;
    fields.forEach(f => {
      const width = (f.hi - f.lo + 1) * bw;
      const fx = x + (bits - 1 - f.hi) * bw;
      const [fill, line] = kindColors(f.kind || 'faint');
      s.addShape(this.pres.shapes.RECTANGLE, { x: fx, y, w: width, h, fill: { color: fill }, line: { color: line, width: 0.75 } });
      this.text(s, fx, y + 0.02, width, h * 0.5, f.name, { size: o.size || 8.5, bold: true, align: 'center', valign: 'middle' });
      if (f.value !== undefined) this.text(s, fx, y + h * 0.5, width, h * 0.46, String(f.value), { size: (o.size || 8.5) - 0.5, align: 'center', valign: 'middle', font: F.code, color: C.muted });
      this.text(s, fx, y - 0.2, width, 0.18, f.hi === f.lo ? `${f.hi}` : `${f.hi}..${f.lo}`, { size: 7, align: 'center', color: C.muted });
    });
  }
  // numbered step row: steps [{title, detail, kind}]
  steps(s, x, y, w, h, steps, o = {}) {
    const gap = o.gap === undefined ? 0.28 : o.gap;
    const bw = (w - gap * (steps.length - 1)) / steps.length;
    const boxes = [];
    steps.forEach((st, i) => {
      const bx = x + i * (bw + gap);
      boxes.push(this.box(s, bx, y, bw, h, st.title, { kind: st.kind || 'ir', detail: st.detail, size: o.size || 12, dsize: o.dsize }));
      if (o.numbers !== false) this.chip(s, bx + 0.08, y - 0.14, String(i + 1), { kind: 'navy', w: 0.3, color: C.white, size: 8.5, h: 0.26 });
      if (i < steps.length - 1) this.arrow(s, bx + bw + 0.02, y + h / 2, bx + bw + gap - 0.02, y + h / 2, { width: 1.5 });
    });
    return boxes;
  }
}


// ---- mechanism-figure helpers (tiles, lanes, word strips, lifetime charts) ----
// matrix: rows x cols cells (a cell may stand for a 64x64 tile); regions = [{r0,r1,c0,c1,kind,label,fill,line}]
Deck.prototype.matrix = function (s, x, y, w, h, rows, cols, o = {}) {
  const cw = w / cols, ch = h / rows;
  const [fill, line] = kindColors(o.kind || 'faint');
  s.addShape(this.pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: o.fill || fill }, line: { color: o.line || line, width: o.lineWidth === undefined ? 1 : o.lineWidth } });
  (o.regions || []).forEach(r => {
    const [rf, rl] = kindColors(r.kind || 'sram');
    const rx = x + r.c0 * cw, ry = y + r.r0 * ch, rw = (r.c1 - r.c0) * cw, rh = (r.r1 - r.r0) * ch;
    s.addShape(this.pres.shapes.RECTANGLE, { x: rx, y: ry, w: rw, h: rh, fill: { color: r.fill || rf }, line: { color: r.line || rl, width: r.lineWidth === undefined ? 1.25 : r.lineWidth, dashType: r.dash || 'solid' } });
  });
  if (o.cells !== false) {
    for (let c = 1; c < cols; c++) { const thick = o.tile && c % o.tile === 0; this.line(s, x + c * cw, y, x + c * cw, y + h, { arrow: false, color: thick ? line : (o.cellLine || 'D5DCE5'), width: thick ? 1 : 0.4 }); }
    for (let r = 1; r < rows; r++) { const thick = o.tile && r % o.tile === 0; this.line(s, x, y + r * ch, x + w, y + r * ch, { arrow: false, color: thick ? line : (o.cellLine || 'D5DCE5'), width: thick ? 1 : 0.4 }); }
  }
  (o.regions || []).forEach(r => {
    if (r.label) this.text(s, x + r.c0 * cw, y + r.r0 * ch, (r.c1 - r.c0) * cw, (r.r1 - r.r0) * ch, r.label, { size: r.size || 8, bold: r.bold !== false, align: 'center', valign: 'middle', color: r.color || C.ink });
  });
  if (o.cellLabel) for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { const t = o.cellLabel(r, c); if (t) this.text(s, x + c * cw, y + r * ch, cw, ch, t, { size: o.cellSize || 7, align: 'center', valign: 'middle', color: C.ink, font: o.mono ? F.code : F.body }); }
  if (o.title) this.text(s, x + (o.titleDx || 0), y - 0.25, Math.max(w, o.titleW || 1.4), 0.22, o.title, { size: o.titleSize || 8.5, bold: true, color: C.navy, align: o.titleAlign || 'left' });
  if (o.below) this.text(s, x + (o.belowDx || 0), y + h + 0.04, Math.max(w, o.belowW || 1.4), 0.4, o.below, { size: o.belowSize || 7.5, color: C.muted, align: o.belowAlign || 'left' });
  if (o.left) s.addText([{ text: o.left, options: { fontSize: 7.5, color: C.muted, fontFace: F.body } }], { x: x - 0.12 - h / 2, y: y + h / 2 - 0.1, w: h, h: 0.2, rotate: 270, margin: 0, isTextBox: true, align: 'center', valign: 'middle' });
  return { x, y, w, h, cw, ch, r: x + w, b: y + h, cx: x + w / 2, cy: y + h / 2 };
};
// lanes: a row of cells [{label, kind, fill, color}]
Deck.prototype.lanes = function (s, x, y, w, h, cells, o = {}) {
  const cw = w / cells.length;
  cells.forEach((c, i) => {
    const [fill, line] = kindColors(c.kind || 'faint');
    s.addShape(this.pres.shapes.RECTANGLE, { x: x + i * cw, y, w: cw, h, fill: { color: c.fill || fill }, line: { color: c.line || line, width: 0.5 } });
    if (c.label !== undefined && c.label !== '') this.text(s, x + i * cw, y, cw, h, String(c.label), { size: c.size || o.size || 7.5, align: 'center', valign: 'middle', font: (c.mono || o.mono) ? F.code : F.body, color: c.color || C.ink, bold: !!c.bold });
  });
  if (o.title) this.text(s, x - (o.titleW || 1.6) - 0.06, y, o.titleW || 1.6, h, o.title, { size: o.titleSize || 8, bold: true, align: 'right', valign: 'middle', color: C.navy });
  if (o.right) this.text(s, x + w + 0.08, y, o.rightW || 2.4, h, o.right, { size: o.rightSize || 7.5, valign: 'middle', color: C.muted, font: o.rightMono ? F.code : F.body });
  return { x, y, w, h, cw, r: x + w, b: y + h, cx: x + w / 2, cy: y + h / 2 };
};
// wordStrip: proportional cells [{n, kind, label, tick}]
Deck.prototype.wordStrip = function (s, x, y, w, h, items, o = {}) {
  const total = items.reduce((a, it) => a + it.n, 0);
  let cx = x;
  items.forEach(it => {
    const iw = w * it.n / total;
    const [fill, line] = kindColors(it.kind || 'faint');
    s.addShape(this.pres.shapes.RECTANGLE, { x: cx, y, w: iw, h, fill: { color: it.fill || fill }, line: { color: it.line || line, width: it.lineWidth || 0.5 } });
    if (it.label && iw >= (it.minW || 0.16)) this.text(s, cx, y, iw, h, it.label, { size: it.size || o.size || 6.5, align: 'center', valign: 'middle', font: (it.mono || o.mono) ? F.code : F.body, color: it.color || C.ink, bold: !!it.bold });
    if (it.tick !== undefined) this.text(s, cx - 0.25, y + h + 0.02, 0.5, 0.16, String(it.tick), { size: 6.5, align: 'center', color: C.muted, font: F.code });
    if (it.above) this.text(s, cx - 0.3, y - 0.2, iw + 0.6, 0.18, it.above, { size: 7, align: 'center', color: it.aboveColor || C.navy, bold: true });
    cx += iw;
  });
  return { x, y, w, h, r: x + w, b: y + h };
};
// lifetimeChart: rows [{name, from, to, kind, pinned, label}] over [lo, hi] (x = offset)
Deck.prototype.lifetimeChart = function (s, x, y, w, h, lo, hi, rows, o = {}) {
  const rh = h / rows.length, scale = w / (hi - lo);
  rows.forEach((r, i) => {
    const ry = y + i * rh;
    this.text(s, x - (o.labelW || 2.3) - 0.06, ry, o.labelW || 2.3, rh, r.name, { size: o.size || 8, align: 'right', valign: 'middle', font: o.mono ? F.code : F.body, color: r.pinned ? C.red : C.ink, bold: !!r.pinned });
    s.addShape(this.pres.shapes.RECTANGLE, { x, y: ry + rh * 0.14, w, h: rh * 0.72, fill: { color: C.faint }, line: { color: C.faint, width: 0 } });
    const [fill, line] = kindColors(r.kind || 'sram');
    const bx = x + (r.from - lo) * scale, bw = (r.to - r.from) * scale;
    s.addShape(this.pres.shapes.RECTANGLE, { x: bx, y: ry + rh * 0.14, w: Math.max(bw, 0.02), h: rh * 0.72, fill: { color: fill }, line: { color: r.pinned ? C.red : line, width: r.pinned ? 1.5 : 0.75 } });
    if (r.label) this.text(s, bx, ry + rh * 0.14, Math.max(bw, 1.0), rh * 0.72, r.label, { size: (o.size || 8) - 1, align: bw > 0.8 ? 'center' : 'left', valign: 'middle', color: C.ink });
  });
  (o.ticks || []).forEach(t => {
    const tx = x + (t - lo) * scale;
    this.line(s, tx, y, tx, y + h, { arrow: false, color: 'C9D3DF', width: 0.5, dash: 'dash' });
    this.text(s, tx - 0.45, y + h + 0.02, 0.9, 0.18, String(t), { size: 7, align: 'center', color: C.muted, font: F.code });
  });
  if (o.title) this.text(s, x - (o.labelW || 2.3) - 0.06, y - 0.28, w + (o.labelW || 2.3), 0.24, o.title, { size: 9, bold: true, color: C.navy });
};

// evidence JSON (docs/evidence/current). NPU_EVIDENCE_DIR overrides the repo-relative location.
function evidence(name) {
  const fs = require('fs'), path = require('path');
  const dirs = [process.env.NPU_EVIDENCE_DIR, path.join(__dirname, '../../evidence/current')].filter(Boolean);
  for (const dir of dirs) { const p = path.join(dir, name); if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, 'utf8')); }
  throw new Error('evidence file not found: ' + name + ' (looked in ' + dirs.join(', ') + ')');
}

module.exports = { Deck, C, F, W, H, kindColors, fitSize, textUnits, evidence };
