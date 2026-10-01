# Parda — Multilingual PII Redaction for Indian Documents

Parda finds and blacks out personal data in scanned and photographed Indian documents: Aadhaar, PAN, UPI IDs,
bank accounts, phone numbers, addresses, names in Hindi / Kannada / English, plus faces, signatures, QR codes
and stamps. It combines two OCR engines, a fine-tuned multilingual NER model (GLiNER), checksum rules and a
YOLO detector, and is measured end to end on the pixels it actually blacks out.

> Research / portfolio project. All training documents are **synthetic**. No real personal
> data is collected or used. Generated ID cards are watermarked `SPECIMEN`.

## Headline results

Share of all personal-data items fully hidden (text + visual), 300 pages per benchmark, 3 languages:

| System | Seen templates | Unseen layouts + phone photos |
|---|---|---|
| Rules only (regex + checksums) | 18.7% | 17.9% |
| Off-the-shelf GLiNER + rules | 51.3% | 38.9% |
| **Parda v1** (fine-tuned GLiNER, 2 OCR engines, YOLO) | 96.1% | 82.5% |
| **Parda v2** (text model also trained on phone photos + EasyOCR) | **97.2%** | **87.4%** |

- **Unseen** = 4 document types never used in training (bank statement, KYC form, pharmacy bill, offer letter),
  photographed with a phone. Even languages: about 82–83% each for v1 on unseen photos.
- Scored strictly: a text item counts only if **every word** is ≥90% blacked out; a visual item if ≥95% of its box is.
  Measured on the ink instead, **99.3%** of signature ink is hidden on the unseen photos (the box rule under-counts
  tilted signatures), which puts v2 at **≈89.7%** there.
- Pages with **nothing** leaking, v2: 74.7% (seen), 36.3% (unseen). Addresses remain the hardest type (63–67%).
- Over-redaction (blacked-out area that is not PII), v2: 38.3% (seen), 44.9% (unseen). Parda errs on the side of hiding.

## Status

- [x] **Phase 1 — Synthetic document generator** (30,000 docs, 0 errors)
- [x] **Phase 2 — OCR benchmark** (Tesseract / EasyOCR / PaddleOCR on English, Hindi, Kannada)
- [x] **Phase 3 — Text PII model** (GLiNER fine-tuned on clean + noisy-OCR text: 60.0% → 80.4% fully redacted)
- [x] **Phase 4 — Visual PII detector** (YOLO11n: 7.2% → 100% of visual PII fully redacted vs OpenCV)
- [x] **Phase 5 — Full pipeline + end-to-end benchmark** (96.1% of all PII fully redacted; rules only 18.7%)
- [x] **Real-world test** (unseen layouts + phone photos: 82.5%; off-the-shelf 38.9%)
- [x] **Text model v2** (phone-photo + EasyOCR training: 87.4% on unseen photos, 97.2% on seen templates)
- [x] **Phase 6, step 1 — Models for the browser** (ONNX, 0.0-point loss; GLiNER 1.1 GB → 555 MB)
- [ ] Phase 6, step 2 — In-browser web app

## Repo layout

```
parda/synth/
  ids.py          fake Indian IDs with real checksum rules (Aadhaar Verhoeff, GSTIN, PAN...)
  names.py        names in English/Hindi/Kannada, places, fake person generator
  canvas.py       draws text word-by-word and records exact boxes + labels
  visuals.py      signatures, QR codes, stamps, photos, SPECIMEN watermark
  templates.py    5 training document types: loan form, payslip, ID card, rent agreement, discharge summary
  augment.py      scan degradation (boxes follow rotation)
  generate.py     parallel generator -> images/ + annotations.jsonl
  visualize.py    draws boxes on samples for checking
  ood.py          real-world test: 4 unseen document types + phone-photo damage (perspective, shadow, blur)
  photo_pages.py  phone-photo versions of TRAINING pages (for text model v2)
parda/ocr/
  engines.py      Tesseract, EasyOCR, PaddleOCR (+ oracles for testing) behind one interface
  run_ocr.py      runs an engine on a fixed stratified sample; resumable
  textnorm.py     normalisation, CER, fuzzy substring distance
  evaluate.py     page CER, per-script word accuracy, PII recovery per entity -> report.md + charts
parda/pii/
  labels.py       PII codes <-> natural-language names GLiNER sees
  spans.py        tokenisation, windows, reading order, LABEL TRANSFER onto OCR text
                  (word-level by position; line-level EasyOCR by character alignment)
  rules.py        regex + checksum detector (handles Devanagari / Kannada digits)
  build_data.py   clean + noisy-OCR training data from several OCR sources (benchmark pages excluded)
  ocr_pages.py    OCR a chosen set of pages with one engine (resumable)
  train_gliner.py fine-tuning (dry-run / smoke / full), uploads the model to a private HF repo
  predict.py      rules / GLiNER (windowed) / union predictors
  evaluate_pii.py text-level redaction metrics on the 300 benchmark pages
parda/vision/
  classes.py      FACE, SIGNATURE, QR_CODE, STAMP (YOLO class ids)
  faces.py        swaps the synthetic avatars for real face photos (separate train / held-out face pools)
  yolo_data.py    YOLO dataset: train / val / fixed benchmark split, labels, previews
  detectors.py    YOLO, OpenCV baseline (Haar faces + QR), gold/empty behind one interface
  train_yolo.py   smoke / time-capped training, summary, uploads weights to a private HF repo
  evaluate_vis.py fully-redacted %, detection, precision, AP50/mAP50, over-redaction
  ink_check.py    how much of each visual item's actual ink is hidden (vs the strict box rule)
parda/pipeline/
  boxes.py        text spans -> pixel boxes, gap filling, masks, drawing
  redactor.py     OCR -> text PII + visual PII -> regions to redact (+ audit entries, never the PII text)
  redact.py       CLI: redact images / multi-page TIFFs / PDFs / iPhone HEIC -> redacted PDF + audit JSON
  models.py       loads the fine-tuned GLiNER and YOLO from a local path or your private HF repos
  evaluate_e2e.py end-to-end pixel-level benchmark with an ablation (one component at a time)
parda/export/
  onnx_export.py  version-proof ONNX export of GLiNER (records the exact network inputs) and YOLO
  lstm_patch.py   exact, export-friendly replacement for GLiNER's packed LSTM (lengths become a real input)
  onnx_runtime.py runs an exported GLiNER inside the normal GLiNER object (pads to the export size)
  compress.py     compression variants of the exported GLiNER, chosen by agreement with PyTorch
parda/fsutil.py   folder search that never lists huge folders in full (fast on Kaggle mounts)
notebooks/        01–10: one Kaggle notebook per step (see below)
tests/            77 tests in 9 files; a perfect system must score exactly 100%, an empty one 0%
scripts/          Kaggle setup (fonts, Tesseract language packs, EasyOCR)
```

## Quick start

```bash
bash scripts/setup_kaggle.sh          # fonts + packages (Kaggle); locally: pip install -r requirements.txt
python tests/test_synth.py            # every test file prints one "ok" line per test
python -m parda.synth.generate --n 200 --out data/synth --workers 4
python -m parda.synth.visualize --data data/synth --k 20
```

## Redact your own documents

```bash
pip install pymupdf pillow-heif                       # PDF input, iPhone photos
python -m parda.pipeline.redact scan.jpg form.pdf --out_dir redacted/ --lang auto \
       --gliner <you>/parda-gliner-v2 --yolo <you>/parda-yolo-v1 --ocr tesseract,easyocr
```
Writes `<name>_redacted.pdf` (image-only, so no hidden text survives under the boxes) and `<name>_audit.json`
(type, box, confidence and source of every redaction, never the PII text itself). `--preview` draws outlines
instead of black boxes.

## Phase 2: OCR benchmark

```bash
bash scripts/setup_ocr.sh
python -m parda.ocr.run_ocr --engine tesseract --per_lang 100 --workers 4 --out outputs/ocr_bench
python -m parda.ocr.run_ocr --engine easyocr   --per_lang 100 --out outputs/ocr_bench
python -m parda.ocr.evaluate --out outputs/ocr_bench
```
Key metric: **PII exact**, the share of personal-data items OCR reads perfectly. Redaction can only hide what OCR
can read. Combining engines lifts PII found from 71% to 85.4% (Hindi 63% → 79%). Results: `results/phase2/`.

## Phase 3: text PII model

The model is trained on **Tesseract's actual (noisy) reading** of the documents, not only clean text. True labels
are transferred onto OCR words by position, so `SBINO364507` at an IFSC position is taught as an IFSC.

```bash
python -m parda.pii.build_data --ocr <ocr_train>/ocr_tesseract.jsonl --clean_docs 10000 --out data/gliner
python -m parda.pii.train_gliner --data_dir data/gliner --out outputs/gliner --smoke   # 2-min check
python -m parda.pii.train_gliner --data_dir data/gliner --out outputs/gliner --epochs 1
python -m parda.pii.evaluate_pii --ocr outputs/bench/ocr_tesseract.jsonl --ft outputs/gliner/final
```

| System (300 held-out scanned pages, Tesseract text) | Fully redacted | Over-redaction | en | hi-en | kn-en |
|---|---|---|---|---|---|
| Rules only | 21.4% | 7.6% | 26.1% | 15.1% | 23.4% |
| Off-the-shelf GLiNER + rules | 60.0% | 29.1% | 66.9% | 62.3% | 51.4% |
| **Fine-tuned GLiNER** | **80.4%** | **10.4%** | **79.7%** | **85.5%** | **76.0%** |

Fine-tuned on 25,028 clean + OCR-noisy examples in 54 min on one free Kaggle T4. Person names 39% → 82%,
MRN 38% → 89%, Kannada 51% → 76%. Results: `results/phase3/`.

## Phase 4: visual PII detector (YOLO11n)

Phase 1 drew cartoon avatars as ID photos. Before training, `yolo_data.py` pastes a **real face photo** (CelebA)
over every avatar: ~90% of the face photos are used for training, a disjoint ~10% only on the validation and
benchmark pages, so the detector is never scored on a face it saw in training.

```bash
python -m parda.vision.yolo_data --out data/yolo --faces_dir <face photos> --n_train 12000 --n_val 600 --preview 6
python -m parda.vision.train_yolo --data_yaml data/yolo/data.yaml --out outputs/yolo --epochs 40 --hours 4
python -m parda.vision.evaluate_vis --yolo_data data/yolo --systems opencv,yolo --weights outputs/yolo/best.pt --out outputs/vis
```

| System (300 held-out pages, 600 visual boxes) | Fully redacted | Precision | mAP50 | Speed |
|---|---|---|---|---|
| OpenCV (Haar faces + QR detector) | 7.2% | 17.9% | 18.9% | 0.21 s/page |
| **YOLO11n, fine-tuned** | **100%** | **100%** | **100%** | **0.035 s/page** |

Faces 0% → 100%, signatures 0.6% → 100%, stamps 0% → 100%, QR codes 97.6% → 100%. Best epoch 39 of 49,
177 min on one T4. Results: `results/phase4/`.

## Phase 5: full pipeline + end-to-end benchmark

Page → Tesseract + EasyOCR → fine-tuned GLiNER + rules on each reading → pixel boxes (+ gap fill) → YOLO →
black boxes. Scored on the pixels actually blacked out; each row adds one component (notebook 05).

| Configuration (seen templates, v1) | All PII fully hidden | Text | Visual | Pages with no leak |
|---|---|---|---|---|
| Rules only | 18.7% | 23.1% | 0% | 0% |
| Off-the-shelf GLiNER + rules | 51.3% | 62.9% | 1.0% | 2.0% |
| Fine-tuned GLiNER + rules | 69.1% | 85.0% | 0% | 4.0% |
| + gap fill | 70.4% | 86.7% | 0% | 4.0% |
| + EasyOCR (best of both) | 77.3% | 95.1% | 0% | 5.3% |
| **+ YOLO visual = Parda v1** | **96.1%** | **95.1%** | **100%** | **68.0%** |

Even across languages: en 96.1%, hi-en 95.6%, kn-en 96.4%. Employee IDs 24% → 95%, names 39% → 98%.
Results: `results/phase5/`.

## Real-world test: unseen layouts + phone photos

`parda/synth/ood.py` makes **4 document types the models never saw**, photographed with a phone (desk,
perspective tilt, shadow, blur, JPEG), with held-out faces and Aadhaar-lookalike reference numbers as hard
negatives. Same end-to-end scoring (notebook 06).

| Parda v1 | Seen templates | Unseen layouts + phone photos |
|---|---|---|
| All PII fully hidden | 96.1% | **82.5%** |
| Text / visual | 95.1% / 100% | 83.6% / 73.7% |
| Pages with no leak | 68.0% | 30.0% |

On the unseen pages every component still helps: rules 17.9% → off-the-shelf 38.9% → fine-tuned 56.0% →
+ EasyOCR 74.1% → + YOLO 82.5%. **Visual check (notebook 07):** YOLO found all 225 signatures; the 73.7% comes
from the strict box rule on tilted wide signatures, while **99.3% of signature ink is hidden**.
Results: `results/phase5/`.

## Text model v2: trained on phone photos and EasyOCR

v1 learned from clean text and Tesseract's mistakes on scans. v2 also learns from **6,000 phone-photo training
pages** (read by Tesseract) and **2,400 EasyOCR readings**. EasyOCR returns whole lines, so labels are carried
over by **character alignment** with the true text: 100% of labels land on exactly the right words on perfect
text, 98–99% on noisy phone photos (notebooks 08a, 08b).

| Full Parda | Seen templates | Unseen layouts + phone photos |
|---|---|---|
| All PII fully hidden, v1 → v2 | 96.1% → **97.2%** | 82.5% → **87.4%** |
| Text PII, v1 → v2 | 95.1% → 96.5% | 83.6% → 89.1% |
| Pages with no leak, v1 → v2 | 68.0% → 74.7% | 30.0% → 36.3% |
| Over-redaction, v1 → v2 | 33.0% → 38.3% | 41.6% → 44.9% |

On unseen photos: addresses 52% → 67%, UPI IDs 78% → 88%, names 84% → 90%, Aadhaar 96% → 100%.
The test layouts stay unseen; the photo damage matches the training augmentation, so part of the gain is
robustness to phone photos. Model: private HF repo `parda-gliner-v2`. Results: `results/text_v2/`.

## Phase 6, step 1: models for the browser (ONNX)

Both models are exported to ONNX (the format browsers run) and checked against PyTorch on both full benchmarks
(notebooks 09, 10).

| Model | Size | Same PII spans as PyTorch | All PII hidden, seen / unseen (full Parda) |
|---|---|---|---|
| GLiNER v2, PyTorch | 1.16 GB | – | 97.2% / 87.4% |
| GLiNER v2, ONNX fp32 | 1,104 MB | 100% | 97.2% / 87.4% |
| GLiNER v2, plain INT8 (every layer) | 338 MB | 36.2% | 75.7% / 59.8% |
| **GLiNER v2, INT8 word-embedding table only (shipped)** | **555 MB** | **99.0%** | **97.2% / 87.4%** |
| YOLO11n, ONNX | 10.4 MB | (fully redacted 100%, mAP50 1.000, same as PyTorch) | |

What it took:
- **Frozen lengths.** GLiNER passes the word count to its network as a Python list, so a plain export froze it and
  any other text length crashed in the LSTM. The export now replaces GLiNER's packed LSTM with an exact equivalent
  that takes the real lengths as an input (checked against the original before exporting), and pads every text to
  a fixed 256 words, cutting the padding off the output.
- **Compression.** Quantizing the matrix multiplications of mDeBERTa (per-tensor or per-channel) destroys it
  (0–36% agreement); 16-bit conversion produced a model onnxruntime cannot load. Quantizing only the
  word-embedding table (most of the file) halves the size with no measurable loss.
- **Browser build (Tesseract only, no EasyOCR):** 89.7% on seen templates, **67.6%** on unseen phone photos
  (vs 87.4% with both OCR engines). A second OCR engine that runs in the browser is worth ~20 points on photos.

Models: private HF repo `parda-onnx-v1`. Results: `results/onnx/`.

## Notebooks (Kaggle)

| Notebook | What it does | Hardware |
|---|---|---|
| 01 | generate the 30,000 synthetic documents | CPU |
| 02 | OCR benchmark | GPU |
| 03a / 03 | OCR the training pages; train GLiNER v1 | CPU / GPU |
| 04 | build the YOLO dataset (real faces), train YOLO11n | GPU |
| 05 | full pipeline + end-to-end benchmark | GPU |
| 06 | real-world test: unseen layouts + phone photos | GPU |
| 07 | visual ink check | CPU |
| 08a / 08b | phone-photo training pages; train GLiNER v2 + both benchmarks | CPU / GPU |
| 09 | export GLiNER v2 and YOLO to ONNX, check against PyTorch | CPU |
| 10 | compress GLiNER for the browser, pick the smallest accurate variant, both benchmarks | CPU |

## Labels

Text: `PERSON_NAME AADHAAR PAN PHONE EMAIL ADDRESS DOB BANK_ACCOUNT IFSC UPI_ID GSTIN VOTER_ID
PASSPORT VEHICLE_REG UAN ABHA EMPLOYEE_ID MRN`

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
