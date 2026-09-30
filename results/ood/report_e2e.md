# Unseen layouts + phone photos (300 pages)

Scored on the black boxes actually drawn. Text PII item = every word ≥90% blacked out; visual item = ≥95% of its box. Page clean = nothing leaks. Each row adds one component to the row above.

| Configuration | All PII fully redacted ↑ | Text ↑ | Visual ↑ | Pages with no leak ↑ | Over-redaction ↓ | s/page |
|---|---|---|---|---|---|---|
| rules only | 17.9% | 20.2% | 0.0% | 0.0% | 30.0% | 1.77 |
| off-the-shelf GLiNER + rules | 38.9% | 43.9% | 0.0% | 0.3% | 43.8% | 1.82 |
| fine-tuned GLiNER + rules | 56.0% | 63.2% | 0.0% | 1.0% | 42.4% | 1.82 |
| + gap fill | 57.6% | 64.9% | 0.0% | 2.0% | 44.5% | 1.82 |
| + EasyOCR (best of both) | 74.1% | 83.6% | 0.2% | 7.7% | 58.0% | 3.47 |
| + YOLO visual = Parda | 82.5% | 83.6% | 73.7% | 30.0% | 41.6% | 3.50 |

## All PII fully redacted, by language

| Configuration | en | hi-en | kn-en |
|---|---|---|---|
| rules only | 19.5% | 13.7% | 20.8% |
| off-the-shelf GLiNER + rules | 42.8% | 39.3% | 34.7% |
| fine-tuned GLiNER + rules | 54.0% | 57.3% | 56.7% |
| + gap fill | 55.5% | 58.8% | 58.3% |
| + EasyOCR (best of both) | 73.4% | 74.7% | 74.2% |
| + YOLO visual = Parda | 81.9% | 82.8% | 82.7% |

## By PII type

| Type | rules only | off-the-shelf GLiNER + rules | fine-tuned GLiNER + rules | + gap fill | + EasyOCR (best of both) | + YOLO visual = Parda |
|---|---|---|---|---|---|---|
| AADHAAR | 48.0% | 60.0% | 65.3% | 69.3% | 96.0% | 96.0% |
| ABHA | 49.3% | 65.3% | 70.7% | 70.7% | 94.7% | 94.7% |
| ADDRESS | 0.0% | 12.0% | 21.7% | 32.3% | 52.3% | 52.3% |
| BANK_ACCOUNT | 1.3% | 64.7% | 76.7% | 76.7% | 95.3% | 95.3% |
| DOB | 0.0% | 46.7% | 61.3% | 61.3% | 96.0% | 96.0% |
| EMAIL | 55.1% | 63.6% | 64.0% | 66.7% | 82.2% | 82.2% |
| EMPLOYEE_ID | 0.0% | 64.0% | 70.7% | 70.7% | 96.0% | 96.0% |
| FACE | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| GSTIN | 5.3% | 34.0% | 52.0% | 52.7% | 82.7% | 82.7% |
| IFSC | 28.0% | 56.0% | 66.7% | 66.7% | 90.0% | 90.0% |
| MRN | 0.0% | 36.0% | 65.3% | 65.3% | 92.0% | 92.0% |
| PAN | 35.3% | 56.0% | 68.0% | 68.7% | 98.0% | 98.0% |
| PERSON_NAME | 1.1% | 28.4% | 65.0% | 65.8% | 84.3% | 84.3% |
| PHONE | 50.2% | 65.3% | 71.1% | 72.4% | 91.6% | 91.6% |
| QR_CODE | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| SIGNATURE | 0.0% | 0.0% | 0.0% | 0.0% | 0.4% | 44.4% |
| STAMP | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| UAN | 44.0% | 73.3% | 73.3% | 74.7% | 97.3% | 97.3% |
| UPI_ID | 55.8% | 64.3% | 71.7% | 72.7% | 77.7% | 77.7% |
