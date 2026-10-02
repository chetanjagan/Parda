# Browser OCR (Tesseract.js) — seen templates (300 pages)

Scored on the black boxes actually drawn. Text PII item = every word ≥90% blacked out; visual item = ≥95% of its box. Page clean = nothing leaks. Each row adds one component to the row above; the last row is the browser build (no EasyOCR).

| Configuration | All PII fully redacted ↑ | Text ↑ | Visual ↑ | Pages with no leak ↑ | Over-redaction ↓ | s/page |
|---|---|---|---|---|---|---|
| rules only | 19.0% | 23.5% | 0.0% | 0.0% | 30.5% | 2.24 |
| fine-tuned GLiNER + rules | 69.5% | 85.5% | 0.3% | 5.0% | 30.3% | 3.13 |
| + gap fill | 71.3% | 87.6% | 0.3% | 5.0% | 33.8% | 3.13 |
| + EasyOCR (best of both) | 78.1% | 96.1% | 0.3% | 5.0% | 57.7% | 5.37 |
| + YOLO visual = Parda | 96.8% | 96.1% | 100.0% | 73.7% | 37.9% | 5.56 |
| browser build: Tesseract only + YOLO | 90.0% | 87.6% | 100.0% | 46.3% | 22.6% | 3.31 |

## All PII fully redacted, by language

| Configuration | en | hi-en | kn-en |
|---|---|---|---|
| rules only | 23.5% | 12.2% | 21.6% |
| fine-tuned GLiNER + rules | 69.9% | 68.7% | 70.0% |
| + gap fill | 71.4% | 70.7% | 71.7% |
| + EasyOCR (best of both) | 77.0% | 78.9% | 78.5% |
| + YOLO visual = Parda | 97.0% | 96.8% | 96.8% |
| browser build: Tesseract only + YOLO | 91.4% | 88.5% | 90.1% |

## By PII type

| Type | rules only | fine-tuned GLiNER + rules | + gap fill | + EasyOCR (best of both) | + YOLO visual = Parda | browser build: Tesseract only + YOLO |
|---|---|---|---|---|---|---|
| AADHAAR | 71.4% | 90.5% | 95.2% | 99.2% | 99.2% | 95.2% |
| ABHA | 57.8% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ADDRESS | 0.0% | 5.6% | 23.0% | 61.5% | 61.5% | 23.0% |
| BANK_ACCOUNT | 5.6% | 93.7% | 93.7% | 100.0% | 100.0% | 93.7% |
| DOB | 0.0% | 92.6% | 95.8% | 100.0% | 100.0% | 95.8% |
| EMAIL | 83.3% | 88.1% | 88.1% | 100.0% | 100.0% | 88.1% |
| EMPLOYEE_ID | 0.0% | 90.5% | 90.5% | 95.2% | 95.2% | 90.5% |
| FACE | 0.0% | 1.2% | 1.2% | 1.2% | 100.0% | 100.0% |
| GSTIN | 31.0% | 90.5% | 90.5% | 100.0% | 100.0% | 90.5% |
| IFSC | 66.7% | 92.9% | 92.9% | 98.4% | 98.4% | 92.9% |
| MRN | 0.0% | 97.8% | 97.8% | 97.8% | 97.8% | 97.8% |
| PAN | 58.5% | 93.0% | 93.0% | 100.0% | 100.0% | 93.0% |
| PERSON_NAME | 0.0% | 91.7% | 92.1% | 98.8% | 98.8% | 92.1% |
| PHONE | 70.8% | 95.8% | 95.8% | 100.0% | 100.0% | 95.8% |
| QR_CODE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| SIGNATURE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| STAMP | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| UAN | 64.3% | 92.9% | 92.9% | 100.0% | 100.0% | 92.9% |
| UPI_ID | 47.6% | 91.7% | 92.9% | 98.8% | 98.8% | 92.9% |
| VOTER_ID | 81.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
