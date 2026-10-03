# Context rules — unseen layouts + phone photos (300 pages)

Scored on the black boxes actually drawn. Text PII item = every word ≥90% blacked out; visual item = ≥95% of its box. Page clean = nothing leaks. Each row adds one component to the row above; the last row is the browser build (no EasyOCR).

| Configuration | All PII fully redacted ↑ | Text ↑ | Visual ↑ | Pages with no leak ↑ | Over-redaction ↓ | s/page |
|---|---|---|---|---|---|---|
| rules only | 17.9% | 20.2% | 0.0% | 0.0% | 29.1% | 1.77 |
| fine-tuned GLiNER + rules | 56.4% | 63.6% | 0.0% | 1.7% | 44.2% | 2.79 |
| + gap fill | 58.1% | 65.5% | 0.0% | 2.3% | 46.4% | 2.79 |
| + EasyOCR (best of both) | 77.3% | 87.2% | 0.2% | 12.3% | 61.0% | 5.90 |
| + YOLO visual = Parda | 85.7% | 87.2% | 73.7% | 33.0% | 44.7% | 6.20 |
| browser build: Tesseract only + YOLO | 66.5% | 65.5% | 73.7% | 10.7% | 30.6% | 3.08 |

## All PII fully redacted, by language

| Configuration | en | hi-en | kn-en |
|---|---|---|---|
| rules only | 19.4% | 13.7% | 20.6% |
| fine-tuned GLiNER + rules | 54.2% | 57.7% | 57.2% |
| + gap fill | 55.6% | 59.4% | 59.1% |
| + EasyOCR (best of both) | 77.2% | 78.2% | 76.6% |
| + YOLO visual = Parda | 85.7% | 86.3% | 85.1% |
| browser build: Tesseract only + YOLO | 64.1% | 67.6% | 67.6% |

## By PII type

| Type | rules only | fine-tuned GLiNER + rules | + gap fill | + EasyOCR (best of both) | + YOLO visual = Parda | browser build: Tesseract only + YOLO |
|---|---|---|---|---|---|---|
| AADHAAR | 48.0% | 65.3% | 69.3% | 100.0% | 100.0% | 69.3% |
| ABHA | 49.3% | 70.7% | 70.7% | 98.7% | 98.7% | 70.7% |
| ADDRESS | 0.0% | 24.0% | 35.0% | 67.7% | 67.7% | 35.0% |
| BANK_ACCOUNT | 0.0% | 76.0% | 76.0% | 96.0% | 96.0% | 76.0% |
| DOB | 0.0% | 0.0% | 0.0% | 1.3% | 1.3% | 0.0% |
| EMAIL | 55.1% | 64.9% | 67.6% | 87.1% | 87.1% | 67.6% |
| EMPLOYEE_ID | 0.0% | 72.0% | 72.0% | 97.3% | 97.3% | 72.0% |
| FACE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| GSTIN | 5.3% | 54.0% | 54.7% | 84.7% | 84.7% | 54.7% |
| IFSC | 28.0% | 65.3% | 65.3% | 91.3% | 91.3% | 65.3% |
| MRN | 0.0% | 68.0% | 68.0% | 93.3% | 93.3% | 68.0% |
| PAN | 35.3% | 67.3% | 68.7% | 98.7% | 98.7% | 68.7% |
| PERSON_NAME | 1.1% | 67.5% | 68.7% | 89.6% | 89.6% | 68.7% |
| PHONE | 50.2% | 71.1% | 72.4% | 93.3% | 93.3% | 72.4% |
| QR_CODE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| SIGNATURE | 0.0% | 0.0% | 0.0% | 0.4% | 44.4% | 44.4% |
| STAMP | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| UAN | 42.7% | 74.7% | 76.0% | 98.7% | 98.7% | 76.0% |
| UPI_ID | 55.8% | 74.4% | 75.4% | 88.2% | 88.2% | 75.4% |
