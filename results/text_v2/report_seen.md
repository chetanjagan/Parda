# Text model v2 — seen templates (300 pages)

Scored on the black boxes actually drawn. Text PII item = every word ≥90% blacked out; visual item = ≥95% of its box. Page clean = nothing leaks. Each row adds one component to the row above.

| Configuration | All PII fully redacted ↑ | Text ↑ | Visual ↑ | Pages with no leak ↑ | Over-redaction ↓ | s/page |
|---|---|---|---|---|---|---|
| rules only | 18.7% | 23.1% | 0.0% | 0.0% | 31.5% | 2.33 |
| off-the-shelf GLiNER + rules | 51.3% | 62.9% | 1.0% | 2.0% | 41.6% | 2.40 |
| fine-tuned GLiNER + rules | 69.5% | 85.5% | 0.0% | 4.3% | 32.0% | 2.40 |
| + gap fill | 70.9% | 87.3% | 0.0% | 4.3% | 35.7% | 2.40 |
| + EasyOCR (best of both) | 78.4% | 96.5% | 0.0% | 5.3% | 58.3% | 3.76 |
| + YOLO visual = Parda | 97.2% | 96.5% | 100.0% | 74.7% | 38.3% | 3.79 |

## All PII fully redacted, by language

| Configuration | en | hi-en | kn-en |
|---|---|---|---|
| rules only | 23.0% | 12.8% | 20.5% |
| off-the-shelf GLiNER + rules | 57.8% | 50.0% | 46.4% |
| fine-tuned GLiNER + rules | 70.2% | 68.7% | 69.5% |
| + gap fill | 71.5% | 70.7% | 70.5% |
| + EasyOCR (best of both) | 77.2% | 79.4% | 78.5% |
| + YOLO visual = Parda | 97.2% | 97.4% | 96.9% |

## By PII type

| Type | rules only | off-the-shelf GLiNER + rules | fine-tuned GLiNER + rules | + gap fill | + EasyOCR (best of both) | + YOLO visual = Parda |
|---|---|---|---|---|---|---|
| AADHAAR | 72.2% | 87.3% | 94.4% | 96.0% | 100.0% | 100.0% |
| ABHA | 75.6% | 97.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| ADDRESS | 0.0% | 6.1% | 7.5% | 22.5% | 62.9% | 62.9% |
| BANK_ACCOUNT | 4.0% | 82.5% | 94.4% | 94.4% | 100.0% | 100.0% |
| DOB | 0.0% | 87.0% | 93.1% | 95.4% | 100.0% | 100.0% |
| EMAIL | 73.8% | 90.5% | 90.5% | 90.5% | 100.0% | 100.0% |
| EMPLOYEE_ID | 0.0% | 78.6% | 85.7% | 85.7% | 97.6% | 97.6% |
| FACE | 0.0% | 1.8% | 0.0% | 0.0% | 0.0% | 100.0% |
| GSTIN | 26.2% | 83.3% | 90.5% | 90.5% | 100.0% | 100.0% |
| IFSC | 52.4% | 90.5% | 92.9% | 92.9% | 99.2% | 99.2% |
| MRN | 0.0% | 40.0% | 97.8% | 97.8% | 97.8% | 97.8% |
| PAN | 63.2% | 90.6% | 93.0% | 93.0% | 100.0% | 100.0% |
| PERSON_NAME | 0.0% | 42.3% | 90.5% | 91.1% | 99.1% | 99.1% |
| PHONE | 68.5% | 89.8% | 95.8% | 95.8% | 100.0% | 100.0% |
| QR_CODE | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| SIGNATURE | 0.0% | 0.9% | 0.0% | 0.0% | 0.0% | 100.0% |
| STAMP | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| UAN | 64.3% | 90.5% | 90.5% | 90.5% | 100.0% | 100.0% |
| UPI_ID | 52.4% | 96.4% | 96.4% | 96.4% | 100.0% | 100.0% |
| VOTER_ID | 81.0% | 90.5% | 97.6% | 97.6% | 100.0% | 100.0% |
