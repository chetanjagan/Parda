// The page pipeline glue (src/app/pipeline.ts) = parda/pipeline/redactor.Redactor.analyse: per OCR engine, text ->
// GLiNER + rules spans -> boxes -> gap filling; then YOLO's visual regions (>= 0.25). Checked with real OCR pages.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { analysePage, blackouts, textPredictor } from "../build/app/pipeline.js";
import * as core from "../build/core/index.js";

const G = JSON.parse(readFileSync(new URL("./goldens/core.json", import.meta.url), "utf8"));
const page = G.inputs.pages[0], exp = G.expected.pages[0];
const withSource = (regs, source) => regs.map((r) => ({ ...r, source }));

test("two OCR streams + visual, in Redactor.analyse's order; rules-only = the Python goldens", async () => {
  const dets = [{ xyxy: [10, 10, 60, 80], cls: 0, label: "FACE", conf: 0.9 }, { xyxy: [5, 5, 9, 9], cls: 2, label: "QR_CODE", conf: 0.2 }];
  const steps = [];
  const res = await analysePage({
    ocr: [{ name: "tesseract", run: async () => page }, { name: "easyocr", run: async () => page }],
    yolo: async () => dets,
  }, "en", (s) => steps.push(s));
  const text = withSource(exp.fill_gaps, "tesseract");
  assert.deepEqual(res.regions, [...text, ...withSource(exp.fill_gaps, "easyocr"),
    { kind: "visual", label: "FACE", bbox: [10, 10, 60, 80], score: 0.9, source: "yolo" }]); // the 0.2 QR is dropped
  assert.deepEqual(Object.keys(res.timings), ["ocr_tesseract", "text_pii_tesseract", "ocr_easyocr", "text_pii_easyocr", "visual"]);
  assert.equal(res.segments.tesseract, page);
  assert.equal(steps.length, 5);
});

test("GLiNER + rules = UnionPredictor: anything either flags, merged", async () => {
  const text = "Name Priya Rao PAN ABCPR1234F";
  const gliner = async (chunk) => [{ label: "person name", start: chunk.indexOf("Priya"), end: chunk.indexOf("Priya") + 9, score: 0.8 },
    { label: "pan number", start: chunk.indexOf("ABCPR"), end: chunk.indexOf("ABCPR") + 5, score: 0.6 }];
  const spans = await textPredictor(gliner)(text);
  assert.deepEqual(spans.map((s) => [s.label, text.slice(s.start, s.end)]), [["PERSON_NAME", "Priya Rao"], ["PAN", "ABCPR1234F"]]);
  assert.equal(spans.find((s) => s.label === "PAN").score, 1.0); // the rule's certainty wins the merge
  assert.deepEqual((await textPredictor()(text)).map((s) => s.label), ["PAN"]);   // without GLiNER: rules only
});

test("blackouts use the per-kind safety margin and stay on the page", () => {
  const r = blackouts([{ kind: "text", label: "PAN", bbox: [10, 10, 50, 20], score: 1, source: "t" },
    { kind: "visual", label: "FACE", bbox: [0, 0, 30, 30], score: 1, source: "yolo" }], 100, 100);
  assert.deepEqual(r, [[7, 7, 53, 23], [0, 0, 36, 36]]);
  assert.deepEqual(core.PAD, { text: 3, visual: 6 });
});
