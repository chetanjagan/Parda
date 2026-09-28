# Parda — Multilingual PII Redaction for Indian Documents

Parda finds and redacts personal data in scanned Indian documents: Aadhaar, PAN, UPI IDs,
bank accounts, phone numbers, addresses, names in Hindi / Kannada / English, faces,
signatures and QR codes. It runs fully offline in the browser.

> Research / portfolio project. All training documents are **synthetic**. No real personal
> data is collected or used. Generated ID cards are watermarked `SPECIMEN`.

## Status

- [x] **Phase 1 — Synthetic document generator** (this commit)
- [ ] Phase 2 — OCR (PaddleOCR) + Indic fine-tuning
- [ ] Phase 3 — Text PII model (GLiNER, 3-stage fine-tuning)
- [ ] Phase 4 — Visual PII detector (faces, signatures, QR, stamps)
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
tests/           checksum + label consistency tests
scripts/         Kaggle setup
```

## Quick start

```bash
bash scripts/setup_kaggle.sh          # fonts + packages (Kaggle); locally: pip install -r requirements.txt
python tests/test_synth.py            # all tests should print ok
python -m parda.synth.generate --n 200 --out data/synth --workers 4
python -m parda.synth.visualize --data data/synth --k 20
```

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
