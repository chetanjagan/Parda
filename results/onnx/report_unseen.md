# Compressed GLiNER (int8_embed) — unseen layouts + phone photos (300 pages)

Scored on the black boxes actually drawn. Text PII item = every word ≥90% blacked out; visual item = ≥95% of its box. Page clean = nothing leaks. Each row adds one component to the row above; the last row is the browser build (no EasyOCR).

| Configuration | All PII fully redacted ↑ | Text ↑ | Visual ↑ | Pages with no leak ↑ | Over-redaction ↓ | s/page |
|---|---|---|---|---|---|---|
| rules only | 17.9% | 20.2% | 0.0% | 0.0% | 30.0% | 1.77 |
| fine-tuned GLiNER + rules | 57.6% | 65.0% | 0.0% | 1.7% | 44.1% | 2.72 |
| + gap fill | 59.2% | 66.8% | 0.0% | 2.3% | 46.3% | 2.72 |
| + EasyOCR (best of both) | 79.1% | 89.2% | 0.2% | 12.3% | 61.1% | 5.79 |
| + YOLO visual = Parda | 87.4% | 89.2% | 73.7% | 36.3% | 45.0% | 6.05 |
| browser build: Tesseract only + YOLO | 67.6% | 66.8% | 73.7% | 11.0% | 30.7% | 2.99 |

## All PII fully redacted, by language

| Configuration | en | hi-en | kn-en |
|---|---|---|---|
| rules only | 19.5% | 13.7% | 20.8% |
| fine-tuned GLiNER + rules | 55.3% | 58.9% | 58.4% |
| + gap fill | 56.7% | 60.7% | 60.3% |
| + EasyOCR (best of both) | 79.0% | 79.9% | 78.4% |
| + YOLO visual = Parda | 87.4% | 88.0% | 86.9% |
| browser build: Tesseract only + YOLO | 65.1% | 68.9% | 68.8% |

## By PII type

| Type | rules only | fine-tuned GLiNER + rules | + gap fill | + EasyOCR (best of both) | + YOLO visual = Parda | browser build: Tesseract only + YOLO |
|---|---|---|---|---|---|---|
| AADHAAR | 48.0% | 65.3% | 69.3% | 100.0% | 100.0% | 69.3% |
| ABHA | 49.3% | 70.7% | 70.7% | 98.7% | 98.7% | 70.7% |
| ADDRESS | 0.0% | 24.0% | 35.0% | 67.7% | 67.7% | 35.0% |
| BANK_ACCOUNT | 1.3% | 76.0% | 76.0% | 96.0% | 96.0% | 76.0% |
| DOB | 0.0% | 61.3% | 61.3% | 97.3% | 97.3% | 61.3% |
| EMAIL | 55.1% | 64.9% | 67.6% | 87.1% | 87.1% | 67.6% |
| EMPLOYEE_ID | 0.0% | 72.0% | 72.0% | 97.3% | 97.3% | 72.0% |
| FACE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| GSTIN | 5.3% | 54.0% | 54.7% | 84.7% | 84.7% | 54.7% |
| IFSC | 28.0% | 65.3% | 65.3% | 91.3% | 91.3% | 65.3% |
| MRN | 0.0% | 68.0% | 68.0% | 93.3% | 93.3% | 68.0% |
| PAN | 35.3% | 67.3% | 68.7% | 98.7% | 98.7% | 68.7% |
| PERSON_NAME | 1.1% | 67.8% | 68.9% | 89.6% | 89.6% | 68.9% |
| PHONE | 50.2% | 71.1% | 72.4% | 93.8% | 93.8% | 72.4% |
| QR_CODE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| SIGNATURE | 0.0% | 0.0% | 0.0% | 0.4% | 44.4% | 44.4% |
| STAMP | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| UAN | 44.0% | 74.7% | 76.0% | 98.7% | 98.7% | 76.0% |
| UPI_ID | 55.8% | 74.4% | 75.4% | 88.2% | 88.2% | 75.4% |
