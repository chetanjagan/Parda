# Compressed GLiNER (int8_embed) — seen templates (300 pages)

Scored on the black boxes actually drawn. Text PII item = every word ≥90% blacked out; visual item = ≥95% of its box. Page clean = nothing leaks. Each row adds one component to the row above; the last row is the browser build (no EasyOCR).

| Configuration | All PII fully redacted ↑ | Text ↑ | Visual ↑ | Pages with no leak ↑ | Over-redaction ↓ | s/page |
|---|---|---|---|---|---|---|
| rules only | 18.7% | 23.1% | 0.0% | 0.0% | 31.5% | 2.34 |
| fine-tuned GLiNER + rules | 69.5% | 85.6% | 0.0% | 4.3% | 32.1% | 3.99 |
| + gap fill | 70.9% | 87.3% | 0.0% | 4.3% | 35.7% | 3.99 |
| + EasyOCR (best of both) | 78.4% | 96.5% | 0.0% | 5.3% | 58.4% | 6.71 |
| + YOLO visual = Parda | 97.2% | 96.5% | 100.0% | 74.7% | 38.3% | 6.97 |
| browser build: Tesseract only + YOLO | 89.7% | 87.3% | 100.0% | 41.3% | 23.3% | 4.25 |

## All PII fully redacted, by language

| Configuration | en | hi-en | kn-en |
|---|---|---|---|
| rules only | 23.0% | 12.8% | 20.5% |
| fine-tuned GLiNER + rules | 70.2% | 68.7% | 69.6% |
| + gap fill | 71.5% | 70.7% | 70.6% |
| + EasyOCR (best of both) | 77.2% | 79.4% | 78.5% |
| + YOLO visual = Parda | 97.2% | 97.4% | 96.9% |
| browser build: Tesseract only + YOLO | 91.5% | 88.7% | 89.0% |

## By PII type

| Type | rules only | fine-tuned GLiNER + rules | + gap fill | + EasyOCR (best of both) | + YOLO visual = Parda | browser build: Tesseract only + YOLO |
|---|---|---|---|---|---|---|
| AADHAAR | 72.2% | 94.4% | 96.0% | 100.0% | 100.0% | 96.0% |
| ABHA | 75.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ADDRESS | 0.0% | 7.5% | 22.5% | 62.9% | 62.9% | 22.5% |
| BANK_ACCOUNT | 4.0% | 94.4% | 94.4% | 100.0% | 100.0% | 94.4% |
| DOB | 0.0% | 93.1% | 95.4% | 100.0% | 100.0% | 95.4% |
| EMAIL | 73.8% | 90.5% | 90.5% | 100.0% | 100.0% | 90.5% |
| EMPLOYEE_ID | 0.0% | 85.7% | 85.7% | 97.6% | 97.6% | 85.7% |
| FACE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| GSTIN | 26.2% | 90.5% | 90.5% | 100.0% | 100.0% | 90.5% |
| IFSC | 52.4% | 92.9% | 92.9% | 99.2% | 99.2% | 92.9% |
| MRN | 0.0% | 97.8% | 97.8% | 97.8% | 97.8% | 97.8% |
| PAN | 63.2% | 93.0% | 93.0% | 100.0% | 100.0% | 93.0% |
| PERSON_NAME | 0.0% | 90.6% | 91.2% | 99.1% | 99.1% | 91.2% |
| PHONE | 68.5% | 95.8% | 95.8% | 100.0% | 100.0% | 95.8% |
| QR_CODE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| SIGNATURE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| STAMP | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| UAN | 64.3% | 90.5% | 90.5% | 100.0% | 100.0% | 90.5% |
| UPI_ID | 52.4% | 96.4% | 96.4% | 100.0% | 100.0% | 96.4% |
| VOTER_ID | 81.0% | 97.6% | 97.6% | 100.0% | 100.0% | 97.6% |
