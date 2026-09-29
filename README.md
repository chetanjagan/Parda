# Parda — Multilingual PII Redaction for Indian Documents

Parda finds and redacts personal data in scanned Indian documents: Aadhaar, PAN, UPI IDs,
bank accounts, phone numbers, addresses, names in Hindi / Kannada / English, faces,
signatures and QR codes. It runs fully offline in the browser.

> Research / portfolio project. All training documents are **synthetic**. No real personal
> data is collected or used. Generated ID cards are watermarked `SPECIMEN`.

## Status

- [X] **Phase 1 — Synthetic document generator** (30,000 docs, 0 errors)
- [X] **Phase 2 — OCR benchmark** (Tesseract / EasyOCR / PaddleOCR on English, Hindi, Kannada)
- [X] **Phase 3 — Text PII model** (GLiNER fine-tuned on clean + noisy-OCR text, rules baseline)

- [X] **Phase 4 — Visual PII detector** (YOLO11n: 7.2% → 100% of visual PII fully redacted vs OpenCV)

- [ ] Phase 5 — Fusion, evaluation benchmark
- [ ] Phase 6 — ONNX/INT8 in-browser inference + web UI

## Repo layout

```
parda/synth/
  ids.py         fake Indian IDs with real checksum rules (Aadhaar Verhoeff, GSTIN, PAN...)
  names.py       names in English/Hindi/Kannada, places, fake person generator
  canvas.py      draws text word-by-word and records exact boxes + labels
  visuals.py     signatures, QR codes, stamps, photos, SPECIMEN watermark
  templates.py   5 document types: loan form, payslip, ID card, rent agreement, discharge summary
  augment.py     scan / phone-photo degradation (boxes follow rotation)
  generate.py    parallel generator -> images/ + annotations.jsonl
  visualize.py   draws boxes on samples for checking
parda/ocr/
  engines.py     Tesseract, EasyOCR, PaddleOCR (+ oracle for testing) behind one interface
  run_ocr.py     runs an engine on a fixed stratified sample; resumable
  textnorm.py    normalisation, CER, fuzzy substring distance
  evaluate.py    page CER, per-script word accuracy, PII recovery per entity -> report.md + charts
parda/pii/
  labels.py      PII codes <-> natural-language names GLiNER sees
  spans.py       tokenisation, windows, tilt-aware OCR reading order, LABEL TRANSFER onto OCR text
  rules.py       regex + checksum detector (baseline)
  build_data.py  clean + noisy-OCR GLiNER training data (benchmark pages excluded)
  train_gliner.py fine-tuning (dry-run / smoke / full), uploads model to a private HF repo
  predict.py     rules / GLiNER (windowed) / union predictors
  evaluate_pii.py redaction metrics on the 300 benchmark pages
tests/           checksum, labels, OCR metrics, label transfer, rules, evaluator (a perfect system must score 100%)
scripts/         Kaggle setup
```

## Quick start

```bash
bash scripts/setup_kaggle.sh          # fonts + packages (Kaggle); locally: pip install -r requirements.txt
python tests/test_synth.py            # all tests should print ok
python -m parda.synth.generate --n 200 --out data/synth --workers 4
python -m parda.synth.visualize --data data/synth --k 20
```

## Phase 2: OCR benchmark

```bash
bash scripts/setup_ocr.sh
python -m parda.ocr.run_ocr --engine tesseract --per_lang 100 --workers 4 --out outputs/ocr_bench
python -m parda.ocr.run_ocr --engine easyocr   --per_lang 100 --out outputs/ocr_bench
python -m parda.ocr.evaluate --out outputs/ocr_bench
```

Key metric: **PII exact**, the share of personal-data items OCR reads perfectly. Redaction can only
hide what OCR can read, so this is the ceiling for the whole system. Results: `results/phase2/`.

## Phase 3: PII model

The model is trained on **Tesseract's actual (noisy) reading** of the documents, not only clean text.
True labels are transferred onto OCR words by position, so `SBINO364507` at an IFSC position is taught as
an IFSC. Headline metric: **% of all true PII items fully redacted** on the 300 benchmark pages
(items OCR never read count as failures).

```bash
python -m parda.pii.build_data --ocr <ocr_train>/ocr_tesseract.jsonl --clean_docs 10000 --out data/gliner
python -m parda.pii.train_gliner --data_dir data/gliner --out outputs/gliner --smoke   # 2-min check
python -m parda.pii.train_gliner --data_dir data/gliner --out outputs/gliner --epochs 1
python -m parda.pii.evaluate_pii --ocr outputs/bench/ocr_tesseract.jsonl --ft outputs/gliner/final
```

Results: `results/phase3/`.

| System (300 held-out scanned pages)        | Fully redacted  | Over-redaction  | en              | hi-en           | kn-en           |
| ------------------------------------------ | --------------- | --------------- | --------------- | --------------- | --------------- |
| Rules only                                 | 21.4%           | 7.6%            | 26.1%           | 15.1%           | 23.4%           |
| Off-the-shelf GLiNER + rules               | 60.0%           | 29.1%           | 66.9%           | 62.3%           | 51.4%           |
| **Fine-tuned GLiNER (this project)** | **80.4%** | **10.4%** | **79.7%** | **85.5%** | **76.0%** |

Fine-tuned on 25,028 clean + OCR-noisy examples in 54 min on one free Kaggle T4.
Biggest gains: person names 39% → 82%, MRN 38% → 89%, Kannada 51% → 76%.
Remaining gaps (address 27%, employee ID 24%) come mainly from OCR and are addressed in Phase 5.


### Phase 4: visual PII detector (YOLO11n)

Held-out benchmark: the same 300 pages, 600 visual PII boxes. Faces are real photos from a held-out pool never seen in training.

| System                            | Fully redacted | Precision      | mAP50          | Speed                  |
| --------------------------------- | -------------- | -------------- | -------------- | ---------------------- |
| OpenCV (Haar faces + QR detector) | 7.2%           | 17.9%          | 18.9%          | 0.21 s/page            |
| **YOLO11n, fine-tuned**     | **100%** | **100%** | **100%** | **0.035 s/page** |

Faces 0% → 100%, signatures 0.6% → 100%, stamps 0% → 100%, QR codes 97.6% → 100%.
Trained on 12,000 pages in 177 min on one T4. The Phase 1 cartoon ID photos were replaced with real face photos (CelebA)
so the detector works on real faces. Scores saturate on synthetic templates; real-scan performance is tested in Phase 5.
Results: `results/phase4/`.

## Labels

Text: `PERSON_NAME AADHAAR PAN PHONE EMAIL ADDRESS DOB BANK_ACCOUNT IFSC UPI_ID GSTIN VOTER_ID PASSPORT VEHICLE_REG UAN ABHA EMPLOYEE_ID MRN`

Visual: `FACE SIGNATURE QR_CODE STAMP`

Unlabelled on purpose (hard negatives): amounts, non-birth dates, reference numbers
(including 12-digit numbers that fail the Aadhaar checksum), organisation names, hospital phone numbers.

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
