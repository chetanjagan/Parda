# Parda — hide personal data in Indian documents, in your browser

**Try it: [chetanjagan.github.io/Parda](https://chetanjagan.github.io/Parda/)** — your document never leaves your device.

Parda (परदा, "curtain") finds and blacks out personal data in Indian documents — loan forms, payslips, ID cards,
rent agreements, bank statements, hospital papers — in **English, Hindi and Kannada**, scanned or photographed with a
phone. It hides Aadhaar and PAN numbers, phone numbers, emails, names, addresses, dates of birth, bank and UPI
details, and nine more ID types, plus **faces, signatures, QR codes and stamps**.

It runs **entirely in the browser**: OCR, a fine-tuned GLiNER model, checksum rules and a YOLO detector, all on your
own machine (WebGPU when available), so nothing is uploaded. You review what was found, fix anything, and download a
flattened PDF in which no hidden text survives under the bars.

## Results

On 300 held-out pages from the training templates (**seen**) and 300 pages of **unseen document types
photographed with a phone**. A page counts as protected only when every piece of personal data on it is covered
in the output image (measured on pixels, not text).

| All personal data fully hidden | Seen templates | Unseen layouts + phone photos |
|---|---|---|
| ID rules only (regex + checksums) | 18.7% | 17.9% |
| Off-the-shelf GLiNER + rules | 51.3% | 38.9% |
| Parda v1 | 96.1% | 82.5% |
| **Parda v2** (two OCR engines + fine-tuned GLiNER + rules + YOLO) | **97.2%** | **87.4%** |
| Parda in the browser (Tesseract.js + EasyOCR, same models) | **96.8%** | **86.6%** |

| Parda v2 | Seen | Unseen + photos |
|---|---|---|
| Pages with no leak at all | 74.7% | 36.3% |
| Over-redaction (extra black pixels) | 38% | 45% |

Faces, signatures, QR codes and stamps: **100%** found (YOLO11n, mAP50 1.000) vs 7.2% for an OpenCV baseline;
99.3% of signature ink hidden even on tilted pages.

## What's in the app

- **Upload** a PDF or photo; pick English, हिन्दी + English or ಕನ್ನಡ + English.
- **Review**: highlights by group (ID numbers, people, contact and address, faces and marks); click to keep an item
  visible, drag to hide anything missed, "+N more" hides every other place the same text appears, undo / redo,
  masked Aadhaar (last 4 digits visible), optional "also hide gender", and the raw OCR text to see why something
  was missed.
- **Export**: a flattened PDF or PNG, plus an audit file listing what was hidden and where — never the text itself.
- Models download once (~580 MB; the phone-photo reader adds 93 MB, 284 MB for Hindi) and stay in the browser.

## How the browser version was built and verified

The benchmarks measured the Python pipeline, so the browser had to redact exactly the same things.

| Step | Result |
|---|---|
| GLiNER → ONNX | full precision: 100% the same spans as PyTorch. Plain 8-bit of every layer kept only 36%; quantizing only the word-embedding table halved the model (1.1 GB → 555 MB) with **0.0 points** lost on both benchmarks |
| Export bug | GLiNER passes the word count as a Python list, which the exporter froze; the packed LSTM was replaced, for export only, by an exact equivalent fed by a real length input |
| YOLO → ONNX | identical boxes; the browser letterbox reproduces OpenCV's integer resize (within 1 brightness level on ≤ 0.012% of pixels) |
| EasyOCR → ONNX | **2,695 / 2,695** text segments identical to PyTorch on 60 pages |
| TypeScript port | ID rules, reading order, box mapping, GLiNER input building and decoding, YOLO and EasyOCR pre/post-processing, each checked against **golden files** recorded from the Python code; the real tokenizer matches on every recorded word |
| Browser OCR | Tesseract.js within ~1 point of native Tesseract end to end; browser EasyOCR reads 95% of segments identically (100% on scans) |

**51 tests** run on every push (GitHub Actions), including the real tokenizer, YOLO and EasyOCR networks, and a
second job re-runs the Python side so the two versions cannot drift apart.

**Speed** (MacBook Air, Chrome, WebGPU + 7 CPU threads, with the phone-photo reader on): English form 43.6 s →
11.4 s, Kannada card 15.0 s → 3.8 s, Hindi phone photo 165.2 s → **17.1 s**.

## Tried and rejected

- **Context rules** to cut false bars on bank statements (a date of birth needs a "birth" label, an unspaced 12-digit
  number needs an Aadhaar label, no spans cutting through numbers): they hid **1.5 points less real PII** for a
  negligible drop in over-redaction, so they were removed from the app. See `results/context_rules/`.
- **Naive 8-bit quantization** of GLiNER (36% agreement) and **16-bit conversion** (would not load) — see `results/onnx/`.

## Limitations

- Trained and benchmarked on **synthetic documents**; not yet measured on real scans.
- On unseen document types it **over-redacts** (bank statements: reference numbers, amounts, transaction dates), and on
  phone photos only 36% of pages have no leak at all. **Always review the result before sharing.**
- The Hindi photo reader is large (205 MB) and slow on machines without WebGPU.
- Very large PDFs are processed page by page in memory; phones may run out of memory with the 555 MB text model.

## Licences

- **YOLO11** (Ultralytics) is **AGPL-3.0**, so the app's source is public under AGPL-3.0.
- The face detector was trained with **CelebA**, which is for **non-commercial research only**.
- GLiNER base (`urchade/gliner_multi_pii-v1`) Apache-2.0; mDeBERTa-v3 MIT; EasyOCR Apache-2.0; Tesseract Apache-2.0.
- **Free, non-commercial research and demonstration use.** Commercial use would need a face detector trained
  without CelebA. Models: [huggingface.co/chetan-0804/parda-web-models](https://huggingface.co/chetan-0804/parda-web-models).

---

## The pipeline

```
page image ─┬─ Tesseract ─┐                      ┌─ GLiNER (fine-tuned, 18 labels) ─┐
            │             ├─ text in reading order┤                                  ├─ spans ─ boxes ─ gap fill ─┐
            └─ EasyOCR ───┘   (per OCR engine)   └─ ID rules (regex + checksums) ──┘                            ├─ black bars
            └─ YOLO11n: faces, signatures, QR codes, stamps ────────────────────────────────────────────────────┘
```

Each OCR engine's text is searched separately and every finding is hidden (two engines catch what one misreads).
Text spans are mapped back to word boxes, partial matches cut by character position, and nearby boxes of the same
type joined so no gaps leak between words.

## Phase by phase

| Phase | What | Key result |
|---|---|---|
| 1. Synthetic data | 30,000 labelled documents (5 types, 3 languages, scan and photo damage) | `parda/synth/` |
| 2. OCR benchmark | Tesseract, EasyOCR, combined | PII found 71% → 85.4% with both (Hindi 63% → 79%) |
| 3. Text model | GLiNER fine-tuned on 25k OCR'd pages | 80.4% of pages fully redacted vs 60.0% off-the-shelf (text only) |
| 4. Visual model | YOLO11n on 12k pages with real faces | 100% fully redacted, mAP50 1.000, 0.035 s/page |
| 5. Full pipeline | everything together | 96.1% (v1) |
| Real-world test | 4 unseen document types, phone photos | 82.5% (v1) |
| Text model v2 | + phone-photo and EasyOCR training text | 97.2% / 87.4% |
| 6. Browser | ONNX, TypeScript port, app, WebGPU | see above |

Details and full tables: `results/phase2/` … `results/phase5/`, `results/ood/`, `results/text_v2/`, `results/onnx/`,
`results/browser_ocr/`, `results/context_rules/`, and `web/README.md` for the browser milestones.

## Repository

```
parda/
  synth/      synthetic documents (templates, IDs with valid checksums, names, scan and photo damage)
  ocr/        OCR engines, reading order, OCR benchmark
  pii/        labels, ID rules, GLiNER training and prediction, label transfer onto OCR text, context rules
  vision/     YOLO data, training, evaluation, ink check
  pipeline/   boxes, redactor, CLI, end-to-end benchmark
  export/     ONNX export (version-proof), LSTM replacement, compression
web/
  src/core     ID rules, reading order, boxes, gap fill, audit (TypeScript)
  src/gliner   tokenizer, input building, decoding
  src/vision   letterbox, YOLO decoding
  src/ocr      Tesseract.js
  src/easyocr  CRAFT post-processing, grouping, cropping, CTC decoding
  src/app      the app: background worker, review, export
  test/        golden-file parity tests
tools/        golden-file recorders (Python) for the browser tests
notebooks/    one Kaggle notebook per step (01–15)
results/      reports for every phase
tests/        Python tests
```

## Notebooks (Kaggle, free GPUs)

| Notebook | What | Accelerator |
|---|---|---|
| 01–03, 03a | synthetic data, OCR benchmark, GLiNER v1 | CPU / GPU |
| 04–05 | YOLO, full pipeline benchmark | GPU |
| 06–07 | real-world test, visual ink check | GPU |
| 08a / 08b | phone-photo training pages; GLiNER v2 + both benchmarks | CPU / GPU |
| 09 / 10 | ONNX export; compression and benchmarks | CPU |
| 11 / 12 | GLiNER and YOLO golden recordings for the browser port | CPU |
| 13 | Tesseract.js on both benchmarks | CPU |
| 14 | EasyOCR to ONNX, parity, recordings | CPU |
| 15 | context rules on both benchmarks | CPU |

## Run it yourself

```bash
pip install -r requirements.txt
python tests/test_synth.py                      # every test file prints one "ok" per test
python -m parda.synth.generate --n 200 --out data/synth --workers 4

# redact documents with the Python pipeline
pip install pymupdf pillow-heif
python -m parda.pipeline.redact scan.jpg form.pdf --out_dir redacted/ --lang auto \
       --gliner <you>/parda-gliner-v2 --yolo <you>/parda-yolo-v1 --ocr tesseract,easyocr

# the browser code and its parity tests
cd web && npm install && npm test
```

The web app is plain TypeScript compiled by `tsc` (libraries from a CDN, models from Hugging Face) and published by
`.github/workflows/pages.yml`.

## Labels

Text: `PERSON_NAME AADHAAR PAN PHONE EMAIL ADDRESS DOB BANK_ACCOUNT IFSC UPI_ID GSTIN VOTER_ID PASSPORT VEHICLE_REG
UAN ABHA EMPLOYEE_ID MRN` (+ optional `GENDER` in the app). Visual: `FACE SIGNATURE QR_CODE STAMP`.

Unlabelled on purpose (hard negatives): amounts, non-birth dates, reference numbers (including 12-digit numbers that
fail the Aadhaar checksum), organisation names, hospital phone numbers.

## Annotation format (`annotations.jsonl`, one line per image)

```json
{
  "id": 12, "image": "images/000012.jpg", "width": 827, "height": 1169,
  "doc_type": "loan_application", "lang": "hi-en", "aug": ["rotate(1.20)", "jpeg"],
  "text": "full page text in reading order",
  "words":    [{"text": "रमेश", "bbox": [x0,y0,x1,y1], "start": 120, "end": 124, "label": "PERSON_NAME"}],
  "entities": [{"label": "AADHAAR", "text": "4829 1736 5520", "start": 300, "end": 314, "bbox": [..]}],
  "visuals":  [{"label": "FACE", "bbox": [..]}]
}
```

## Copyright

Copyright (C) 2026 Chetan Jagannatha. The code is licensed under the GNU AGPL v3 (see `LICENSE`).
The models keep their own terms (see "Licences" above).
