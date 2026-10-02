// Browser OCR (Tesseract.js) wrapper: the same settings and segment format as the Python TesseractEngine.
// Accuracy of Tesseract.js itself is measured on the benchmarks in notebook 13 (it is a different Tesseract build).
import assert from "node:assert/strict";
import { test } from "node:test";
import * as core from "../build/core/index.js";
import { TesseractOcr, tesseractLangs, toSegments, wordsOf } from "../build/ocr/index.js";

const W = (text, confidence, x0, y0, x1, y1) => ({ text, confidence, bbox: { x0, y0, x1, y1 } });
const RESULT = { blocks: [{ paragraphs: [
  { lines: [{ words: [W("Name:", 91, 40, 100, 95, 118), W("Priya", 88, 102, 100, 150, 118), W("Rao", 90, 156, 100, 190, 118)] },
            { words: [W("PAN", 93, 40, 130, 80, 148), W("ABCPR1234F", 96, 88, 130, 210, 148), W("  ", 10, 215, 130, 220, 148)] }] },
  { lines: [{ words: [W("मोबाइल", 71, 40, 160, 110, 180), W("9845012345", 95, 118, 160, 230, 180)] }] },
] }] };

test("language mapping = Python TesseractEngine.LANG", () => {
  assert.equal(tesseractLangs("en"), "eng");
  assert.equal(tesseractLangs("hi-en"), "hin+eng");
  assert.equal(tesseractLangs("kn-en"), "kan+eng");
  assert.equal(tesseractLangs("auto"), "eng+hin+kan");
  assert.equal(tesseractLangs("xx"), "eng");
});

test("words -> segments: every non-empty word, box = word box, score = confidence / 100", () => {
  const segs = toSegments(RESULT);
  assert.equal(segs.length, 7); // the blank word is dropped
  assert.deepEqual(segs[1], { text: "Priya", bbox: [102, 100, 150, 118], score: 0.88 });
  assert.deepEqual(wordsOf({ words: [W("a", 50, 1, 2, 3, 4)] }).map((w) => w.text), ["a"]); // older result format
  const { text } = core.buildOcrText(segs);  // feeds the rest of the pipeline unchanged
  assert.equal(text, "Name: Priya Rao\nPAN ABCPR1234F\nमोबाइल 9845012345");
  assert.deepEqual(core.findRules(text).map((r) => r.label), ["PAN", "PHONE"]);
});

test("one worker; languages switched only when the page language changes; page mode 3", async () => {
  const calls = [];
  const fake = (langs) => ({
    reinitialize: async (l) => calls.push(`reinit ${l}`),
    setParameters: async (p) => calls.push(`psm ${p.tessedit_pageseg_mode}`),
    recognize: async (img, opts, output) => { calls.push(`recognize ${img} blocks=${output.blocks}`); return { data: RESULT }; },
    terminate: async () => calls.push("terminate"),
    _langs: langs,
  });
  const ocr = new TesseractOcr(async (langs) => { calls.push(`create ${langs}`); return fake(langs); });
  await ocr.run("p1.png", "kn-en");
  await ocr.run("p2.png", "kn-en");
  const segs = await ocr.run("p3.png", "en");
  await ocr.close();
  assert.equal(segs.length, 7);
  assert.deepEqual(calls, ["create kan+eng", "psm 3", "recognize p1.png blocks=true", "recognize p2.png blocks=true",
    "reinit eng", "psm 3", "recognize p3.png blocks=true", "terminate"]);
});
