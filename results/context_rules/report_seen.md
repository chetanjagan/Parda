# Context rules — seen templates (300 pages)

Scored on the black boxes actually drawn. Text PII item = every word ≥90% blacked out; visual item = ≥95% of its box. Page clean = nothing leaks. Each row adds one component to the row above; the last row is the browser build (no EasyOCR).

| Configuration | All PII fully redacted ↑ | Text ↑ | Visual ↑ | Pages with no leak ↑ | Over-redaction ↓ | s/page |
|---|---|---|---|---|---|---|
| rules only | 18.5% | 22.8% | 0.0% | 0.0% | 31.4% | 2.34 |
| fine-tuned GLiNER + rules | 67.8% | 83.5% | 0.0% | 4.3% | 32.1% | 3.74 |
| + gap fill | 69.2% | 85.1% | 0.0% | 4.3% | 35.8% | 3.74 |
| + EasyOCR (best of both) | 76.9% | 94.6% | 0.0% | 5.3% | 58.3% | 6.68 |
| + YOLO visual = Parda | 95.7% | 94.6% | 100.0% | 59.3% | 38.1% | 6.93 |
| browser build: Tesseract only + YOLO | 87.9% | 85.1% | 100.0% | 29.7% | 23.3% | 4.00 |

## All PII fully redacted, by language

| Configuration | en | hi-en | kn-en |
|---|---|---|---|
| rules only | 22.7% | 12.6% | 20.4% |
| fine-tuned GLiNER + rules | 68.5% | 67.2% | 67.8% |
| + gap fill | 69.7% | 69.0% | 68.8% |
| + EasyOCR (best of both) | 75.6% | 77.7% | 77.3% |
| + YOLO visual = Parda | 95.6% | 95.7% | 95.6% |
| browser build: Tesseract only + YOLO | 89.7% | 87.0% | 87.1% |

## By PII type

| Type | rules only | fine-tuned GLiNER + rules | + gap fill | + EasyOCR (best of both) | + YOLO visual = Parda | browser build: Tesseract only + YOLO |
|---|---|---|---|---|---|---|
| AADHAAR | 72.2% | 94.4% | 96.0% | 100.0% | 100.0% | 96.0% |
| ABHA | 75.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ADDRESS | 0.0% | 7.5% | 22.5% | 62.9% | 62.9% | 22.5% |
| BANK_ACCOUNT | 0.8% | 94.4% | 94.4% | 100.0% | 100.0% | 94.4% |
| DOB | 0.0% | 69.4% | 70.8% | 77.8% | 77.8% | 70.8% |
| EMAIL | 73.8% | 90.5% | 90.5% | 100.0% | 100.0% | 90.5% |
| EMPLOYEE_ID | 0.0% | 85.7% | 85.7% | 97.6% | 97.6% | 85.7% |
| FACE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| GSTIN | 26.2% | 90.5% | 90.5% | 100.0% | 100.0% | 90.5% |
| IFSC | 52.4% | 92.9% | 92.9% | 99.2% | 99.2% | 92.9% |
| MRN | 0.0% | 97.8% | 97.8% | 97.8% | 97.8% | 97.8% |
| PAN | 63.2% | 93.0% | 93.0% | 100.0% | 100.0% | 93.0% |
| PERSON_NAME | 0.0% | 90.3% | 90.9% | 99.1% | 99.1% | 90.9% |
| PHONE | 68.5% | 95.8% | 95.8% | 100.0% | 100.0% | 95.8% |
| QR_CODE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| SIGNATURE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| STAMP | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| UAN | 59.5% | 90.5% | 90.5% | 100.0% | 100.0% | 90.5% |
| UPI_ID | 52.4% | 96.4% | 96.4% | 100.0% | 100.0% | 96.4% |
| VOTER_ID | 81.0% | 97.6% | 97.6% | 100.0% | 100.0% | 97.6% |
