#!/usr/bin/env node
// Run the browser OCR (Tesseract.js, via src/ocr) over a list of pages and write the SAME jsonl format as the Python
// OCR caches ({"id", "segments", "sec"}), so the Python benchmark (parda.pipeline.evaluate_e2e) can score it.
//   node tools/ocr_pages.mjs --pages pages.jsonl --out ocr_tesseract.jsonl [--workers 4] [--data fast|best_int]
// pages.jsonl: one {"id", "path", "lang"} per line. Resumable: pages already in --out are skipped.
import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { TesseractOcr } from "../build/ocr/index.js";

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const pagesFile = arg("pages"), out = arg("out"), nWorkers = Number(arg("workers", 4)), data = arg("data", "best_int");
if (!pagesFile || !out) { console.error("usage: --pages pages.jsonl --out out.jsonl [--workers N] [--data fast|best_int]"); process.exit(2); }

const { createWorker } = await import("tesseract.js");
// best_int: Tesseract.js' default language files. fast: tessdata_fast, the files Ubuntu's apt packages ship.
const options = data === "fast" ? { langPath: "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main", gzip: false } : {};
const create = (langs) => createWorker(langs, 1, options);

const pages = readFileSync(pagesFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const done = new Set(existsSync(out) ? readFileSync(out, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).id) : []);
const todo = pages.filter((p) => !done.has(p.id)).sort((a, b) => a.lang.localeCompare(b.lang)); // fewer language switches
console.log(`Tesseract.js (${data}): ${todo.length} pages to run (${done.size} done), ${nWorkers} workers`);
let next = 0, n = 0;
const t0 = Date.now();
async function worker() {
  const ocr = new TesseractOcr(create);
  while (next < todo.length) {
    const p = todo[next++];
    const t = Date.now();
    let rec;
    try {
      rec = { id: p.id, segments: await ocr.run(p.path, p.lang), sec: (Date.now() - t) / 1000 };
    } catch (e) {
      rec = { id: p.id, segments: [], sec: (Date.now() - t) / 1000, error: String(e).slice(0, 300) };
    }
    appendFileSync(out, JSON.stringify(rec) + "\n");
    if (++n % 25 === 0) console.log(`  ${n}/${todo.length} pages, ${((Date.now() - t0) / 1000 / n * nWorkers).toFixed(1)} s/page per worker`);
  }
  await ocr.close();
}
await Promise.all(Array.from({ length: nWorkers }, worker));
console.log(`done: ${n} pages in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
