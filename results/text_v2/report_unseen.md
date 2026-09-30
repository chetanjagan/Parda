# Text model v2 — unseen layouts + phone photos (300 pages)

Scored on the black boxes actually drawn. Text PII item = every word ≥90% blacked out; visual item = ≥95% of its box. Page clean = nothing leaks. Each row adds one component to the row above.

| Configuration | All PII fully redacted ↑ | Text ↑ | Visual ↑ | Pages with no leak ↑ | Over-redaction ↓ | s/page |
|---|---|---|---|---|---|---|
| rules only | 17.9% | 20.2% | 0.0% | 0.0% | 30.0% | 1.77 |
| off-the-shelf GLiNER + rules | 38.9% | 43.9% | 0.0% | 0.3% | 43.8% | 1.82 |
| fine-tuned GLiNER + rules | 57.5% | 64.9% | 0.0% | 1.7% | 44.1% | 1.82 |
| + gap fill | 59.2% | 66.8% | 0.0% | 2.3% | 46.3% | 1.82 |
| + EasyOCR (best of both) | 79.0% | 89.1% | 0.2% | 12.3% | 61.0% | 3.47 |
| + YOLO visual = Parda | 87.4% | 89.1% | 73.7% | 36.3% | 44.9% | 3.50 |

## All PII fully redacted, by language

| Configuration | en | hi-en | kn-en |
|---|---|---|---|
| rules only | 19.5% | 13.7% | 20.8% |
| off-the-shelf GLiNER + rules | 42.8% | 39.3% | 34.7% |
| fine-tuned GLiNER + rules | 55.2% | 58.8% | 58.5% |
| + gap fill | 56.6% | 60.5% | 60.3% |
| + EasyOCR (best of both) | 79.0% | 79.7% | 78.3% |
| + YOLO visual = Parda | 87.4% | 87.8% | 86.8% |

## By PII type

| Type | rules only | off-the-shelf GLiNER + rules | fine-tuned GLiNER + rules | + gap fill | + EasyOCR (best of both) | + YOLO visual = Parda |
|---|---|---|---|---|---|---|
| AADHAAR | 48.0% | 60.0% | 65.3% | 69.3% | 100.0% | 100.0% |
| ABHA | 49.3% | 65.3% | 70.7% | 70.7% | 98.7% | 98.7% |
| ADDRESS | 0.0% | 12.0% | 24.0% | 35.0% | 67.3% | 67.3% |
| BANK_ACCOUNT | 1.3% | 64.7% | 76.0% | 76.0% | 96.0% | 96.0% |
| DOB | 0.0% | 46.7% | 61.3% | 61.3% | 96.0% | 96.0% |
| EMAIL | 55.1% | 63.6% | 64.9% | 67.1% | 86.7% | 86.7% |
| EMPLOYEE_ID | 0.0% | 64.0% | 72.0% | 72.0% | 97.3% | 97.3% |
| FACE | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| GSTIN | 5.3% | 34.0% | 52.7% | 53.3% | 84.0% | 84.0% |
| IFSC | 28.0% | 56.0% | 65.3% | 65.3% | 92.0% | 92.0% |
| MRN | 0.0% | 36.0% | 68.0% | 68.0% | 94.7% | 94.7% |
| PAN | 35.3% | 56.0% | 67.3% | 68.7% | 98.7% | 98.7% |
| PERSON_NAME | 1.1% | 28.4% | 67.8% | 68.9% | 89.6% | 89.6% |
| PHONE | 50.2% | 65.3% | 71.6% | 72.9% | 93.8% | 93.8% |
| QR_CODE | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| SIGNATURE | 0.0% | 0.0% | 0.0% | 0.0% | 0.4% | 44.4% |
| STAMP | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| UAN | 44.0% | 73.3% | 74.7% | 76.0% | 98.7% | 98.7% |
| UPI_ID | 55.8% | 64.3% | 74.2% | 75.2% | 88.0% | 88.0% |
