# Parda web app

The browser version of the Parda pipeline: documents are redacted **on the device**, nothing is uploaded.

## Milestones

| | What | Status |
|---|---|---|
| M1 | Pipeline core in TypeScript (`src/core`): ID rules + checksums, tokenising, windows, reading order of OCR words, text spans → pixel boxes, gap filling, padding, audit log | ✅ identical to Python on 262 texts and 33 pages |
| M2 | GLiNER (555 MB ONNX) + tokenizer in the browser | |
| M3 | YOLO11n (10 MB ONNX) in the browser | |
| M4 | OCR in the browser: Tesseract.js + EasyOCR (ONNX) | |
| M5 | The UI (upload → review → export), all features | |
| M6 | Deploy; measure the browser build on the benchmarks | |

## Parity with the Python pipeline

The benchmarks measured the Python pipeline, so the browser must redact exactly the same things.
`tools/make_goldens.py` runs the real Python functions on hundreds of cases (real Tesseract OCR of scans and phone
photos, Indic digits, random OCR-like text, valid and broken IDs, exact edge cases) and stores inputs + answers
in `test/goldens/core.json`. `npm test` must reproduce every answer exactly; CI also re-runs the Python side on every
push. A mutation check (breaking rounding, reading order, checksums, gap filling, same-line rule on purpose) is caught
by the tests.

```bash
cd web
npm install          # TypeScript only, for now
npm test             # type-check (strict) + 12 parity tests
python ../tools/make_goldens.py --check   # the Python side still gives the stored answers
```

Python behaviours reproduced on purpose: `\w` = `[\p{L}\p{N}_]` (Hindi/Kannada vowel signs are not word
characters), any script's digits → ASCII via Unicode digit values, `round()` with ties to even, stable sorting.
Text positions are JavaScript string indices (identical to Python's for every script Parda handles).
