# Phase 2 — OCR benchmark on synthetic Indian documents

*PII exact* = read character-perfect. *PII found* = read correctly ignoring punctuation/spaces (what redaction needs). Indic digits (६३२२) count as equal to ASCII digits (6322).

## Overall

| Engine | Pages | Page CER ↓ | Word recall ↑ | PII exact ↑ | PII found ↑ | PII ≤10% CER ↑ | PII missed ↓ | Sec/page |
|---|---|---|---|---|---|---|---|---|
| easyocr | 300 | 14.1% | 79.4% | 63.0% | 67.6% | 78.4% | 0.0% | 1.14 |
| tesseract | 300 | 23.9% | 75.3% | 70.6% | 71.4% | 81.6% | 3.7% | 2.01 |
| **best of easyocr + tesseract** | – | – | – | 83.6% | **85.4%** | – | – | – |

## By language

| Engine | Language | Page CER ↓ | Word recall ↑ | PII exact ↑ | PII found ↑ | PII missed ↓ |
|---|---|---|---|---|---|---|
| easyocr | en | 10.8% | 85.8% | 67.8% | 74.0% | 0.0% |
| easyocr | hi-en | 15.9% | 76.8% | 58.9% | 63.3% | 0.1% |
| easyocr | kn-en | 15.6% | 75.7% | 62.5% | 66.0% | 0.0% |
| tesseract | en | 20.0% | 80.8% | 83.1% | 83.7% | 3.9% |
| tesseract | hi-en | 28.1% | 68.5% | 49.9% | 50.1% | 3.3% |
| tesseract | kn-en | 23.6% | 76.7% | 79.5% | 81.1% | 3.9% |
| **best of both** | en | – | – | 87.8% | **88.6%** | – |
| **best of both** | hi-en | – | – | 75.9% | **79.3%** | – |
| **best of both** | kn-en | – | – | 87.4% | **88.6%** | – |

## Word accuracy by script

| Engine | Script | Words | Exact ↑ | Word CER ↓ |
|---|---|---|---|---|
| easyocr | deva | 712 | 68.4% | 13.9% |
| easyocr | knda | 706 | 77.2% | 9.8% |
| easyocr | latin | 30068 | 74.5% | 9.6% |
| tesseract | deva | 712 | 48.3% | 49.9% |
| tesseract | knda | 706 | 78.9% | 20.2% |
| tesseract | latin | 30068 | 82.6% | 13.0% |

## PII by entity type (exact / found)

| Entity | easyocr | tesseract | best of both (found) |
|---|---|---|---|
| AADHAAR | 94.4% / 94.4% | 72.2% / 72.2% | 96.8% |
| ABHA | 68.9% / 75.6% | 66.7% / 66.7% | 88.9% |
| ADDRESS | 6.1% / 46.0% | 58.7% / 62.4% | 75.6% |
| BANK_ACCOUNT | 82.5% / 84.1% | 71.4% / 71.4% | 89.7% |
| DOB | 77.8% / 81.5% | 63.4% / 63.4% | 92.1% |
| EMAIL | 35.7% / 59.5% | 71.4% / 73.8% | 88.1% |
| EMPLOYEE_ID | 45.2% / 45.2% | 16.7% / 16.7% | 45.2% |
| GSTIN | 21.4% / 21.4% | 28.6% / 28.6% | 33.3% |
| IFSC | 38.1% / 38.1% | 43.7% / 43.7% | 61.9% |
| MRN | 28.9% / 28.9% | 57.8% / 57.8% | 64.4% |
| PAN | 36.8% / 36.8% | 59.6% / 59.6% | 67.8% |
| PERSON_NAME | 73.7% / 73.8% | 87.4% / 87.4% | 92.7% |
| PERSON_NAME (deva) | 72.0% / 72.0% | 59.1% / 59.1% |  |
| PERSON_NAME (knda) | 77.6% / 77.6% | 86.7% / 86.7% |  |
| PHONE | 82.9% / 86.6% | 65.7% / 69.9% | 92.1% |
| UAN | 66.7% / 66.7% | 59.5% / 59.5% | 85.7% |
| UPI_ID | 48.8% / 53.6% | 73.8% / 76.2% | 86.9% |
| VOTER_ID | 76.2% / 76.2% | 83.3% / 83.3% | 90.5% |

## PII found, by document type

| Document | easyocr | tesseract |
|---|---|---|
| discharge_summary | 62.8% | 73.4% |
| id_card_tax | 86.9% | 76.2% |
| id_card_uid | 89.0% | 81.8% |
| id_card_voter | 75.3% | 72.7% |
| loan_application | 75.2% | 74.3% |
| payslip | 47.7% | 43.1% |
| rent_agreement | 60.0% | 76.0% |
