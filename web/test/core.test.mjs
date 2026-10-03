// Parity with the Python pipeline: every answer in test/goldens/core.json (made by tools/make_goldens.py from the
// real Python functions) must be reproduced EXACTLY by the TypeScript core. Run: npm test  (or: node --test test/)
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import * as core from "../build/core/index.js";

const G = JSON.parse(readFileSync(new URL("./goldens/core.json", import.meta.url), "utf8"));
const I = G.inputs;
const E = G.expected;

test("digits of any script become ASCII (same length)", () => {
  I.texts.forEach((t, i) => assert.equal(core.asciiDigits(t), E.ascii_digits[i], t));
});

test("ID rules: regex + checksums give the same spans", () => {
  let hits = 0;
  I.texts.forEach((t, i) => {
    const got = core.findRules(t);
    hits += got.length;
    assert.deepEqual(got, E.rules[i], t);
  });
  assert.ok(hits > 300, `only ${hits} rule hits`);
});

test("tokenising (GLiNER's word splitter) matches", () => {
  I.texts.forEach((t, i) => assert.deepEqual(core.tokenize(t), E.tokenize[i], t));
});

test("windows match", () => {
  I.windows.forEach(([n, w, s], i) => assert.deepEqual(core.windows(n, w, s), E.windows[i]));
});

test("span merging matches", () => {
  I.span_sets.forEach((s, i) => assert.deepEqual(core.mergeSpans(s.map((x) => ({ ...x }))), E.merge_spans[i]));
});

test("gap filling matches", () => {
  I.region_sets.forEach((r, i) => assert.deepEqual(core.fillGaps(r), E.fill_gaps[i]));
});

test("pixel rounding matches", () => {
  I.pix.forEach(([b, W, H, p], i) => assert.deepEqual(core.pix(b, W, H, p), E.pix[i]));
});

test("Python round() (ties to even) matches", () => {
  I.floats.forEach((x, i) => assert.deepEqual([core.pyRound(x, 1), core.pyRound(x, 3)], E.round[i], String(x)));
});

test("padding per kind matches", () => assert.deepEqual(core.PAD, E.pad));

test("real OCR pages: reading order, rules, span boxes, gap filling all match", () => {
  assert.ok(I.pages.length >= 10);
  I.pages.forEach((segs, pi) => {
    const exp = E.pages[pi];
    const { text, ordered, offs } = core.buildOcrText(segs);
    assert.equal(text, exp.text, `page ${pi} text`);
    assert.deepEqual(ordered.map((s) => [s.text, s.bbox]), exp.order.map((k) => [segs[k].text, segs[k].bbox]), `page ${pi} order`);
    assert.deepEqual(offs, exp.offs, `page ${pi} offsets`);
    const rules = core.findRules(text);
    assert.deepEqual(rules, exp.rules, `page ${pi} rules`);
    assert.deepEqual(rules.map((r) => core.spanBoxes(r.start, r.end, ordered, offs)), exp.span_boxes, `page ${pi} boxes`);
    exp.partial.forEach(([a, b, boxes]) => assert.deepEqual(core.spanBoxes(a, b, ordered, offs), boxes, `page ${pi} partial`));
    const regs = rules.flatMap((r, k) => exp.span_boxes[k].map((bbox) => ({ kind: "text", label: r.label, bbox, score: 1.0, source: "tesseract" })));
    assert.deepEqual(core.fillGaps(regs), exp.fill_gaps, `page ${pi} fill_gaps`);
  });
});

test("text regions + audit entry: no PII text in the audit, rounded like Python", async () => {
  const page = I.pages[0];
  const { regions, text } = await core.textRegions(page, async (t) => core.findRules(t), "tesseract");
  assert.ok(text.length > 0);
  const audit = core.auditEntry(regions);
  for (const a of audit) assert.deepEqual(Object.keys(a).sort(), ["bbox", "kind", "label", "score", "source"]);
});

test("windowed prediction maps chunk offsets back and merges duplicates", async () => {
  const words = Array.from({ length: 450 }, (_, i) => `w${i}`);
  const text = words.join(" ");
  const fake = async (chunk) => {  // a "model" that tags every occurrence of w300 (seen by two windows)
    const k = chunk.indexOf("w300 ");
    return k >= 0 ? [{ label: "person name", start: k, end: k + 4, score: 0.9 }] : [];
  };
  const spans = await core.windowedPredict(text, fake);
  const at = text.indexOf("w300 ");
  assert.deepEqual(spans, [{ label: "PERSON_NAME", start: at, end: at + 4, score: 0.9 }]);
});

test("context rules (dates of birth, unspaced Aadhaar, spans cutting numbers) = Python on 150 cases", () => {
  assert.ok(I.context.length >= 100);
  I.context.forEach(([t, spans], i) => assert.deepEqual(core.filterSpans(t, spans.map((s) => ({ ...s }))), E.context[i], `case ${i}: ${t.slice(0, 40)}`));
});
