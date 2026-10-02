// EasyOCR in the browser = EasyOCR 1.7.2 in Python, stage by stage, on 6 recorded pages (scan + phone photo per
// language): tools/easyocr_onnx.py record (notebook 14) -> tools/pack_easyocr_goldens.py -> test/goldens/easyocr*.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { gunzipSync } from "node:zlib";
import * as eo from "../build/easyocr/index.js";
import { readPng } from "./png.mjs";

const G = JSON.parse(readFileSync(new URL("./goldens/easyocr.json", import.meta.url), "utf8"));
const file = (f) => new URL(`./goldens/easyocr/${f}`, import.meta.url).pathname;
const f32s = (f) => { const b = gunzipSync(readFileSync(file(f))); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const u8s = (f) => new Uint8Array(gunzipSync(readFileSync(file(f))));
const gray = (c) => { const p = readPng(file(c.gray_png)); return { data: p.data, width: p.width, height: p.height }; };
const prod = (s) => s.reduce((a, b) => a * b, 1);
function slices(buf, shapes) { const out = []; let o = 0; for (const s of shapes) { out.push(buf.subarray(o, o + prod(s))); o += prod(s); } return out; }
const reader = (lang) => { const r = G.meta.readers[lang]; const keep = new Set([...r.lang_char]);
  return { chars: [...r.character], ignore: new Set([...r.character].map((ch, i) => (keep.has(ch) ? -1 : i + 1)).filter((i) => i > 0)) }; };

test("score maps -> text boxes (connected components, dilation, rectangles) match EasyOCR", () => {
  let exact = 0, total = 0, worst = 0;
  for (const c of G.cases.filter((c) => c.maps)) {
    const [, h2, w2] = c.maps.shape;
    const got = eo.detBoxes(f32s(c.maps.file), h2, w2, 1.0);
    assert.equal(got.length, c.raw_boxes.length, `${c.id}: number of boxes`);
    got.forEach((b, i) => {
      total++;
      const d = Math.max(...b.map((v, k) => Math.abs(v - c.raw_boxes[i][k])));
      exact += d === 0; worst = Math.max(worst, d);
    });
  }
  // OpenCV computes its rectangles in float32; on a few axis-aligned boxes that shifts a corner by one pixel
  assert.ok(exact / total >= 0.99 && worst <= 2, `${exact}/${total} exact, worst ${worst} px`);
  console.log(`   boxes: ${exact}/${total} identical, the rest within ${worst} px`);
});

test("grouping into lines + size filter = EasyOCR on all 6 pages", () => {
  for (const c of G.cases) {
    const g = eo.filterMinSize(eo.groupTextBox(c.raw_boxes));
    assert.deepEqual(g.horizontal, c.horizontal_list, `${c.lang} ${c.id} lines`);
    assert.equal(g.free.length, c.free_list.length);
    g.free.forEach((b, i) => b.forEach((p, k) => p.forEach((v, j) => assert.ok(Math.abs(v - c.free_list[i][k][j]) < 1e-6))));
  }
});

function replay(c) {  // each box: first pass, and the recorded contrast retry when the first pass was below 0.1
  const r = reader(c.lang), g = gray(c);
  const ins = slices(u8s(c.rec.in_file), c.rec.in_shapes), outs = slices(f32s(c.rec.out_file), c.rec.out_shapes);
  const crops = [...c.horizontal_list.map((b) => eo.cropHorizontal(g, b)), ...c.free_list.map((b) => eo.cropFree(g, b))].filter(Boolean);
  const calls = [], results = [];
  let i = 0;
  for (const crop of crops) {
    const dec = (k) => eo.decodeCtc(outs[k], c.rec.out_shapes[k][1], c.rec.out_shapes[k][2], r.chars, r.ignore);
    calls.push({ crop, adjust: 0, rec: ins[i] });
    let res = dec(i); i++;
    if (res.conf < 0.1) { calls.push({ crop, adjust: 0.5, rec: ins[i] }); const r2 = dec(i); i++; res = res.conf > r2.conf ? res : r2; }
    results.push({ box: crop.box, ...res });
  }
  assert.equal(i, ins.length, `${c.id}: every recorded recognizer call accounted for`);
  return { calls, results };
}

test("recognizer inputs (crop, resize, contrast, pad) = EasyOCR's", () => {
  let same = 0, n = 0, vals = 0, diffVals = 0, worst = 0;
  for (const c of G.cases.filter((c) => c.rec)) {
    for (const { crop, adjust, rec } of replay(c).calls) {
      const mine = eo.alignCollate(crop.img, crop.maxWidth, adjust);
      assert.equal(mine.length, rec.length);
      let d = 0;
      for (let k = 0; k < mine.length; k++) {
        const v = Math.round((mine[k] * 0.5 + 0.5) * 255), e = Math.abs(v - rec[k]);
        if (e) diffVals++; d = Math.max(d, e);
      }
      vals += mine.length; n++; same += d === 0; worst = Math.max(worst, d);
    }
  }
  // OpenCV's bilinear enlargement differs by 1 level on some pixels; the contrast boost can stretch that a little
  assert.ok(diffVals / vals < 0.01 && worst <= 8, `${(100 * diffVals / vals).toFixed(2)}% of values differ, worst ${worst}`);
  console.log(`   ${same}/${n} inputs bit-identical; ${(100 * diffVals / vals).toFixed(3)}% of all values differ (worst ${worst})`);
});

test("decoding the recorded recognizer outputs (+ contrast retry) gives readtext's text on 3 pages", () => {
  for (const c of G.cases.filter((c) => c.rec)) {
    const { results } = replay(c);
    assert.deepEqual(results.map((r) => r.text), c.readtext.map((r) => r.text), `${c.lang} ${c.id}`);
    results.forEach((r, i) => assert.ok(Math.abs(r.conf - c.readtext[i].conf) < 1e-5, `${c.id}: confidence ${r.conf} vs ${c.readtext[i].conf}`));
    results.forEach((r, i) => assert.deepEqual(r.box, c.readtext[i].box));
  }
});

test("segments like parda's EasyOCREngine: non-empty text, box = int of corner min / max", () => {
  const segs = eo.toSegments([{ box: [[10.7, 5.2], [40.9, 4.1], [41.5, 20.9], [9.9, 21.3]], text: "PAN", conf: 0.9 },
    { box: [[0, 0], [5, 0], [5, 5], [0, 5]], text: "  ", conf: 0.5 }]);
  assert.deepEqual(segs, [{ text: "PAN", bbox: [9, 4, 41, 21], score: 0.9 }]);
});

test("detector input: padded to multiples of 32, normalised like normalizeMeanVariance", () => {
  for (const c of G.cases.filter((c) => c.maps)) {
    const [h, w] = c.page_hw;
    const inp = eo.detectorInput(new Uint8Array(w * h * 3), w, h, 3);
    assert.deepEqual(inp.dims.slice(2).map((v) => v / 2), c.maps.shape.slice(1, 3), c.id);
  }
  const inp = eo.detectorInput(Uint8Array.from([255, 0, 128]), 1, 1, 3);
  assert.deepEqual(inp.dims, [1, 3, 32, 32]);
  assert.equal(inp.tensor[0], Math.fround(Math.fround(255 - Math.fround(0.485 * 255)) / Math.fround(0.229 * 255)));
  assert.equal(inp.tensor[1], Math.fround(Math.fround(0 - Math.fround(0.485 * 255)) / Math.fround(0.229 * 255))); // padding
});

test("score-map rules the recorded pages never hit: tiny components dropped, near-square regions boxed upright", () => {
  const S = JSON.parse(readFileSync(new URL("./goldens/easyocr_synthetic.json", import.meta.url), "utf8"));
  assert.deepEqual(eo.detBoxes(Float32Array.from(S.maps), S.h2, S.w2, 1.0), S.boxes); // expected from real OpenCV
});
