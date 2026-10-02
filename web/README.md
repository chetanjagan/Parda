# Parda web app

The browser version of the Parda pipeline: documents are redacted **on the device**, nothing is uploaded.

## Milestones

| | What | Status |
|---|---|---|
| M1 | Pipeline core in TypeScript (`src/core`): ID rules + checksums, tokenising, windows, reading order of OCR words, text spans → pixel boxes, gap filling, padding, audit log | ✅ identical to Python on 262 texts and 33 pages |
| M2 | GLiNER in the browser (`src/gliner`): input building, padding for the export, decoding; mDeBERTa tokenizer via transformers.js | ✅ identical to Python on 160 real OCR chunks (1,565 spans); real tokenizer checked in CI |
| M3 | YOLO in the browser (`src/vision`): letterbox with OpenCV's integer resize, decoding + per-class overlap removal, boxes back to page pixels | ✅ same boxes as ultralytics on 8 real pages; input within 1 brightness level on ≤ 0.012% of pixels; real ONNX model checked in CI |
| M4a | Browser OCR, part 1: Tesseract.js (`src/ocr`) with the Python engine's settings; `tools/ocr_pages.mjs` writes the Python OCR-cache format | wrapper tested; accuracy measured on both benchmarks in notebook 13 |
| M4b | Browser OCR, part 2: EasyOCR (`src/easyocr`): CRAFT + recognizers as ONNX, detection, grouping, cropping, decoding ported | networks: 2,695/2,695 segments identical to PyTorch (Kaggle); boxes 402/402, grouping and decoding identical on 6 recorded pages; image operations bit-identical to OpenCV/Pillow except bilinear enlargement (±1 on 0.6% of pixels); real en/kn recognizers checked in CI |
| M5 | The app (`index.html`, `src/app`): models downloaded once from Hugging Face and cached, Tesseract.js + optional EasyOCR, GLiNER + rules, YOLO, page pipeline = `Redactor.analyse` | session 1: upload → boxes → redacted PNG; published by `.github/workflows/pages.yml` |
| M6 | Deploy; measure the browser build on the benchmarks | |

## Parity with the Python pipeline

The benchmarks measured the Python pipeline, so the browser must redact exactly the same things.
`tools/make_goldens.py` runs the real Python functions on hundreds of cases (real Tesseract OCR of scans and phone
photos, Indic digits, random OCR-like text, valid and broken IDs, exact edge cases) and stores inputs + answers
in `test/goldens/core.json`. `npm test` must reproduce every answer exactly; CI also re-runs the Python side on every
push. A mutation check (breaking rounding, reading order, checksums, gap filling, same-line rule on purpose) is caught
by the tests.

GLiNER: `tools/make_gliner_goldens.py` (notebook 11, on Kaggle) records the shipped model on real OCR chunks: words,
token ids, network inputs, raw scores and final spans; `tools/pack_gliner_goldens.py` trims that into
`test/goldens/gliner.json` and puts the tokenizer in `public/tokenizer/`. `test/gliner.test.mjs` rebuilds every input
and decodes every recorded score; `test/gliner_tokenizer.test.mjs` checks the real tokenizer word by word (CI sets
`REQUIRE_TOKENIZER=1`, so it can never silently skip).

```bash
cd web
npm install          # TypeScript + transformers.js (tokenizer) + onnxruntime-node (real YOLO test)
# the real-YOLO test needs public/yolo/parda-yolo.onnx (10 MB, from the HF repo parda-onnx-v1)
npm test             # type-check (strict) + 29 parity tests (core, GLiNER, YOLO, real tokenizer, real YOLO)
python ../tools/make_goldens.py --check   # the Python side still gives the stored answers
```

Python behaviours reproduced on purpose: `\w` = `[\p{L}\p{N}_]` (Hindi/Kannada vowel signs are not word
characters), any script's digits → ASCII via Unicode digit values, `round()` with ties to even, stable sorting.
Text positions are JavaScript string indices (identical to Python's for every script Parda handles).
