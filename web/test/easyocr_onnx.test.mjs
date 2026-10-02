// The real EasyOCR recognizers (public/easyocr/recognizer_en.onnx, recognizer_kn-en.onnx) in onnxruntime-node, fed by the
// browser's crops, must read the recorded pages like Python EasyOCR (the Hindi one, 205 MB, is checked on Kaggle).
// CI sets REQUIRE_EASYOCR=1 so this can never silently skip.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { EasyOcrReader } from "../build/easyocr/index.js";
import { readPng } from "./png.mjs";

const G = JSON.parse(readFileSync(new URL("./goldens/easyocr.json", import.meta.url), "utf8"));
const model = (lang) => new URL(`../public/easyocr/recognizer_${lang}.onnx`, import.meta.url).pathname;
let ort = null, why = "";
try {
  for (const l of ["en", "kn-en"]) if (!existsSync(model(l))) throw new Error(`model file missing: ${model(l)}`);
  ort = await import("onnxruntime-node");
} catch (e) {
  why = `real EasyOCR recognizers not available (${e.message.split("\n")[0]})`;
  if (process.env.REQUIRE_EASYOCR) throw new Error(`CI requires them: ${why}`);
}

test("real recognizers on the browser's crops read the recorded pages like Python EasyOCR", { skip: !ort && why }, async () => {
  let same = 0, total = 0;
  const diffs = [];
  for (const lang of ["en", "kn-en"]) {
    const session = await ort.InferenceSession.create(model(lang));
    const run = async ({ data, dims }) => {
      const out = await session.run({ image: new ort.Tensor("float32", data, dims) });
      return { data: out.preds.data, dims: out.preds.dims };
    };
    const reader = new EasyOcrReader(async () => { throw new Error("detector not used here"); }, run, G.meta.readers[lang]);
    for (const c of G.cases.filter((c) => c.lang === lang)) {
      const p = readPng(new URL(`./goldens/easyocr/${c.gray_png}`, import.meta.url).pathname);
      const got = await reader.recognizeBoxes({ data: p.data, width: p.width, height: p.height }, c.horizontal_list, c.free_list);
      assert.equal(got.length, c.readtext.length, c.id);
      got.forEach((r, i) => {
        total++;
        if (r.text === c.readtext[i].text) same++;
        else if (diffs.length < 10) diffs.push(`${c.id}: ${JSON.stringify(r.text)} vs ${JSON.stringify(c.readtext[i].text)}`);
      });
    }
  }
  console.log(`   ${same}/${total} text segments identical to Python EasyOCR${diffs.length ? "; e.g.\n   " + diffs.join("\n   ") : ""}`);
  assert.ok(same / total >= 0.97, `only ${same}/${total} identical`);
});
