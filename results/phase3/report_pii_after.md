# Phase 3 — PII detection on 300 benchmark pages (Tesseract OCR text)

*Fully redacted* counts every true PII item, including ones OCR never read. Spaces and punctuation are ignored.

## Overall

| System | Fully redacted ↑ | PII chars covered ↑ | Over-redaction ↓ | Correct type ↑ | Span precision ↑ |
|---|---|---|---|---|---|
| rules | **21.4%** | 22.2% | 7.6% | 22.1% | 91.9% |
| gliner_base@0.3 | **56.8%** | 57.1% | 30.0% | 52.7% | 67.9% |
| gliner_base@0.5 | **49.5%** | 50.6% | 23.0% | 46.5% | 81.0% |
| gliner_ft@0.3 | **80.4%** | 77.8% | 10.4% | 80.3% | 89.3% |
| gliner_ft@0.5 | **79.6%** | 77.0% | 9.9% | 79.6% | 90.1% |
| gliner_ft@0.3 + rules | **80.4%** | 77.9% | 10.5% | 80.4% | 89.3% |
| gliner_ft@0.5 + rules | **79.6%** | 77.0% | 10.0% | 79.7% | 90.0% |

## Fully redacted, by language

| System | en | hi-en | kn-en |
|---|---|---|---|
| rules | 26.1% | 15.1% | 23.4% |
| gliner_base@0.3 | 64.0% | 59.9% | 47.1% |
| gliner_base@0.5 | 58.5% | 51.0% | 39.6% |
| gliner_ft@0.3 | 79.7% | 85.5% | 76.0% |
| gliner_ft@0.5 | 79.4% | 84.4% | 75.0% |
| gliner_ft@0.3 + rules | 79.7% | 85.5% | 76.1% |
| gliner_ft@0.5 + rules | 79.4% | 84.4% | 75.1% |

## Fully redacted, by PII type

| Type | rules | gliner_base@0.3 | gliner_base@0.5 | gliner_ft@0.3 | gliner_ft@0.5 | gliner_ft@0.3 + rules | gliner_ft@0.5 + rules |
|---|---|---|---|---|---|---|---|
| AADHAAR | 69.0% | 60.3% | 57.1% | 92.1% | 91.3% | 92.1% | 91.3% |
| ABHA | 73.3% | 93.3% | 88.9% | 93.3% | 93.3% | 93.3% | 93.3% |
| ADDRESS | 0.0% | 23.5% | 23.0% | 27.2% | 25.8% | 27.2% | 25.8% |
| BANK_ACCOUNT | 4.0% | 75.4% | 69.8% | 88.9% | 88.9% | 88.9% | 88.9% |
| DOB | 0.0% | 87.5% | 82.9% | 95.8% | 95.8% | 95.8% | 95.8% |
| EMAIL | 83.3% | 92.9% | 92.9% | 100.0% | 97.6% | 100.0% | 97.6% |
| EMPLOYEE_ID | 0.0% | 23.8% | 23.8% | 23.8% | 23.8% | 23.8% | 23.8% |
| GSTIN | 23.8% | 83.3% | 78.6% | 88.1% | 88.1% | 88.1% | 88.1% |
| IFSC | 43.7% | 83.3% | 81.0% | 82.5% | 82.5% | 83.3% | 83.3% |
| MRN | 0.0% | 37.8% | 11.1% | 88.9% | 88.9% | 88.9% | 88.9% |
| PAN | 54.4% | 77.2% | 73.7% | 82.5% | 81.3% | 82.5% | 81.3% |
| PERSON_NAME | 0.0% | 38.8% | 28.5% | 81.6% | 80.3% | 81.6% | 80.3% |
| PHONE | 63.4% | 72.7% | 63.0% | 88.4% | 88.4% | 88.4% | 88.4% |
| UAN | 54.8% | 78.6% | 76.2% | 81.0% | 81.0% | 81.0% | 81.0% |
| UPI_ID | 52.4% | 86.9% | 85.7% | 96.4% | 96.4% | 96.4% | 96.4% |
| VOTER_ID | 83.3% | 66.7% | 28.6% | 100.0% | 97.6% | 100.0% | 97.6% |
