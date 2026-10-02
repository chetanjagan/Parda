# Browser OCR (Tesseract.js) — unseen layouts + phone photos (300 pages)

Scored on the black boxes actually drawn. Text PII item = every word ≥90% blacked out; visual item = ≥95% of its box. Page clean = nothing leaks. Each row adds one component to the row above; the last row is the browser build (no EasyOCR).

| Configuration | All PII fully redacted ↑ | Text ↑ | Visual ↑ | Pages with no leak ↑ | Over-redaction ↓ | s/page |
|---|---|---|---|---|---|---|
| rules only | 15.9% | 17.9% | 0.0% | 0.0% | 29.2% | 1.93 |
| fine-tuned GLiNER + rules | 56.0% | 63.2% | 0.0% | 2.7% | 43.7% | 2.56 |
| + gap fill | 58.0% | 65.4% | 0.0% | 3.0% | 46.4% | 2.56 |
| + EasyOCR (best of both) | 78.2% | 88.2% | 0.2% | 11.7% | 61.3% | 5.14 |
| + YOLO visual = Parda | 86.6% | 88.2% | 73.7% | 36.0% | 45.1% | 5.31 |
| browser build: Tesseract only + YOLO | 66.4% | 65.4% | 73.7% | 12.3% | 30.6% | 2.73 |

## All PII fully redacted, by language

| Configuration | en | hi-en | kn-en |
|---|---|---|---|
| rules only | 16.8% | 11.5% | 19.5% |
| fine-tuned GLiNER + rules | 52.2% | 57.4% | 58.3% |
| + gap fill | 54.0% | 59.9% | 59.9% |
| + EasyOCR (best of both) | 78.2% | 78.8% | 77.6% |
| + YOLO visual = Parda | 86.7% | 86.9% | 86.1% |
| browser build: Tesseract only + YOLO | 62.5% | 68.1% | 68.4% |

## By PII type

| Type | rules only | fine-tuned GLiNER + rules | + gap fill | + EasyOCR (best of both) | + YOLO visual = Parda | browser build: Tesseract only + YOLO |
|---|---|---|---|---|---|---|
| AADHAAR | 42.7% | 61.3% | 66.7% | 100.0% | 100.0% | 66.7% |
| ABHA | 37.3% | 64.0% | 65.3% | 98.7% | 98.7% | 65.3% |
| ADDRESS | 0.0% | 21.0% | 31.3% | 65.3% | 65.3% | 31.3% |
| BANK_ACCOUNT | 2.0% | 76.7% | 76.7% | 96.0% | 96.0% | 76.7% |
| DOB | 0.0% | 52.0% | 56.0% | 97.3% | 97.3% | 56.0% |
| EMAIL | 55.1% | 65.8% | 69.3% | 87.1% | 87.1% | 69.3% |
| EMPLOYEE_ID | 0.0% | 73.3% | 73.3% | 97.3% | 97.3% | 73.3% |
| FACE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| GSTIN | 2.0% | 53.3% | 53.3% | 85.3% | 85.3% | 53.3% |
| IFSC | 34.7% | 66.0% | 66.0% | 92.0% | 92.0% | 66.0% |
| MRN | 0.0% | 65.3% | 65.3% | 93.3% | 93.3% | 65.3% |
| PAN | 33.3% | 70.7% | 70.7% | 98.7% | 98.7% | 70.7% |
| PERSON_NAME | 0.9% | 67.7% | 68.9% | 88.6% | 88.6% | 68.9% |
| PHONE | 48.4% | 65.8% | 68.0% | 92.0% | 92.0% | 68.0% |
| QR_CODE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| SIGNATURE | 0.0% | 0.0% | 0.0% | 0.4% | 44.4% | 44.4% |
| STAMP | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| UAN | 48.0% | 77.3% | 78.7% | 100.0% | 100.0% | 78.7% |
| UPI_ID | 41.9% | 67.1% | 69.8% | 85.5% | 85.5% | 69.8% |
