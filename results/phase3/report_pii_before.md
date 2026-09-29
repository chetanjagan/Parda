# Phase 3 — PII detection on 300 benchmark pages (Tesseract OCR text)

*Fully redacted* counts every true PII item, including ones OCR never read. Spaces and punctuation are ignored.

## Overall

| System | Fully redacted ↑ | PII chars covered ↑ | Over-redaction ↓ | Correct type ↑ | Span precision ↑ |
|---|---|---|---|---|---|
| rules | **21.4%** | 22.2% | 7.6% | 22.1% | 91.9% |
| gliner_base@0.3 | **56.8%** | 57.1% | 30.0% | 52.7% | 67.9% |
| gliner_base@0.5 | **49.5%** | 50.6% | 23.0% | 46.5% | 81.0% |
| gliner_base@0.3 + rules | **60.0%** | 60.0% | 29.1% | 58.2% | 69.7% |
| gliner_base@0.5 + rules | **54.4%** | 55.0% | 21.9% | 53.2% | 82.2% |

## Fully redacted, by language

| System | en | hi-en | kn-en |
|---|---|---|---|
| rules | 26.1% | 15.1% | 23.4% |
| gliner_base@0.3 | 64.0% | 59.9% | 47.1% |
| gliner_base@0.5 | 58.5% | 51.0% | 39.6% |
| gliner_base@0.3 + rules | 66.9% | 62.3% | 51.4% |
| gliner_base@0.5 + rules | 63.5% | 54.5% | 45.9% |

## Fully redacted, by PII type

| Type | rules | gliner_base@0.3 | gliner_base@0.5 | gliner_base@0.3 + rules | gliner_base@0.5 + rules |
|---|---|---|---|---|---|
| AADHAAR | 69.0% | 60.3% | 57.1% | 85.7% | 84.1% |
| ABHA | 73.3% | 93.3% | 88.9% | 93.3% | 93.3% |
| ADDRESS | 0.0% | 23.5% | 23.0% | 23.5% | 23.0% |
| BANK_ACCOUNT | 4.0% | 75.4% | 69.8% | 75.4% | 69.8% |
| DOB | 0.0% | 87.5% | 82.9% | 87.5% | 82.9% |
| EMAIL | 83.3% | 92.9% | 92.9% | 95.2% | 95.2% |
| EMPLOYEE_ID | 0.0% | 23.8% | 23.8% | 23.8% | 23.8% |
| GSTIN | 23.8% | 83.3% | 78.6% | 83.3% | 78.6% |
| IFSC | 43.7% | 83.3% | 81.0% | 83.3% | 82.5% |
| MRN | 0.0% | 37.8% | 11.1% | 37.8% | 11.1% |
| PAN | 54.4% | 77.2% | 73.7% | 81.3% | 79.5% |
| PERSON_NAME | 0.0% | 38.8% | 28.5% | 38.8% | 28.5% |
| PHONE | 63.4% | 72.7% | 63.0% | 83.8% | 83.3% |
| UAN | 54.8% | 78.6% | 76.2% | 81.0% | 81.0% |
| UPI_ID | 52.4% | 86.9% | 85.7% | 96.4% | 96.4% |
| VOTER_ID | 83.3% | 66.7% | 28.6% | 90.5% | 85.7% |
