# Phase 3 results — PII detection on 300 held-out benchmark pages

Text = Tesseract OCR of 300 scanned pages (100 each: en, hi-en, kn-en). These pages were never used in training.
**Fully redacted** counts every true PII item, including items OCR never read (those count as failures).
Spaces and punctuation are ignored. `base` = off-the-shelf `urchade/gliner_multi_pii-v1`, `ft` = fine-tuned (this project).

## Overall

| System | Fully redacted ↑ | PII chars covered ↑ | Over-redaction ↓ | Correct type ↑ | Span precision ↑ |
|---|---|---|---|---|---|
| rules | 21.4% | 22.2% | 7.6% | 22.1% | 91.9% |
| gliner_base@0.3 | 56.8% | 57.1% | 30.0% | 52.7% | 67.9% |
| gliner_base@0.5 | 49.5% | 50.6% | 23.0% | 46.5% | 81.0% |
| gliner_base@0.3 + rules | 60.0% | 60.0% | 29.1% | 58.2% | 69.7% |
| gliner_base@0.5 + rules | 54.4% | 55.0% | 21.9% | 53.2% | 82.2% |
| **gliner_ft@0.3** | **80.4%** | **77.8%** | **10.4%** | **80.3%** | **89.3%** |
| gliner_ft@0.5 | 79.6% | 77.0% | 9.9% | 79.6% | 90.1% |
| gliner_ft@0.3 + rules | 80.4% | 77.9% | 10.5% | 80.4% | 89.3% |
| gliner_ft@0.5 + rules | 79.6% | 77.0% | 10.0% | 79.7% | 90.0% |

## Fully redacted, by language

| System | en | hi-en | kn-en |
|---|---|---|---|
| rules | 26.1% | 15.1% | 23.4% |
| gliner_base@0.3 | 64.0% | 59.9% | 47.1% |
| gliner_base@0.5 | 58.5% | 51.0% | 39.6% |
| gliner_base@0.3 + rules | 66.9% | 62.3% | 51.4% |
| gliner_base@0.5 + rules | 63.5% | 54.5% | 45.9% |
| **gliner_ft@0.3** | **79.7%** | **85.5%** | **76.0%** |
| gliner_ft@0.5 | 79.4% | 84.4% | 75.0% |
| gliner_ft@0.3 + rules | 79.7% | 85.5% | 76.1% |
| gliner_ft@0.5 + rules | 79.4% | 84.4% | 75.1% |

## Fully redacted, by PII type

| Type | rules | base@0.3 | base@0.3 + rules | **ft@0.3** | ft@0.5 | ft@0.3 + rules |
|---|---|---|---|---|---|---|
| AADHAAR | 69.0% | 60.3% | 85.7% | **92.1%** | 91.3% | 92.1% |
| ABHA | 73.3% | 93.3% | 93.3% | **93.3%** | 93.3% | 93.3% |
| ADDRESS | 0.0% | 23.5% | 23.5% | **27.2%** | 25.8% | 27.2% |
| BANK_ACCOUNT | 4.0% | 75.4% | 75.4% | **88.9%** | 88.9% | 88.9% |
| DOB | 0.0% | 87.5% | 87.5% | **95.8%** | 95.8% | 95.8% |
| EMAIL | 83.3% | 92.9% | 95.2% | **100.0%** | 97.6% | 100.0% |
| EMPLOYEE_ID | 0.0% | 23.8% | 23.8% | **23.8%** | 23.8% | 23.8% |
| GSTIN | 23.8% | 83.3% | 83.3% | **88.1%** | 88.1% | 88.1% |
| IFSC | 43.7% | 83.3% | 83.3% | **82.5%** | 82.5% | 83.3% |
| MRN | 0.0% | 37.8% | 37.8% | **88.9%** | 88.9% | 88.9% |
| PAN | 54.4% | 77.2% | 81.3% | **82.5%** | 81.3% | 82.5% |
| PERSON_NAME | 0.0% | 38.8% | 38.8% | **81.6%** | 80.3% | 81.6% |
| PHONE | 63.4% | 72.7% | 83.8% | **88.4%** | 88.4% | 88.4% |
| UAN | 54.8% | 78.6% | 81.0% | **81.0%** | 81.0% | 81.0% |
| UPI_ID | 52.4% | 86.9% | 96.4% | **96.4%** | 96.4% | 96.4% |
| VOTER_ID | 83.3% | 66.7% | 90.5% | **100.0%** | 97.6% | 100.0% |

## Training (`train_summary.json`)

- 25,028 examples (14,849 clean text + 10,197 Tesseract-noisy with labels transferred by position; 18 broken examples dropped)
- 1 epoch = 3,131 steps, batch 8, fp32, single T4: **54.4 min** (1.04 s/step)
- Train loss 68.7 → 1.0; eval loss 3.17 → 3.69 → 1.55 → 1.23 → 1.46 → 1.26 (plateaus after ~2k steps, no overfitting)

## Reading the results

- **60.0% → 80.4% fully redacted** (best before = base@0.3 + rules), over-redaction **29.1% → 10.4%**.
- Largest gains on the scripts OCR mangles most: **Kannada 51.4% → 76.1%**, Hindi 62.3% → 85.5%.
- Rules add ~0 on top of the fine-tuned model (it learned the ID formats); they stay in the pipeline as a cheap,
  checksum-validated safety net.
- **Threshold 0.3** is chosen: recall matters more than precision for redaction, and 0.3 vs 0.5 differ by <1 pt.
- **EMPLOYEE_ID is 23.8% for every system**, base and fine-tuned alike → the limit is OCR, not the model
  (Tesseract misreads e.g. `E`→`£`; only 265 of ~2,000 employee IDs survived label transfer).
- **ADDRESS 27%**: multi-line spans fail "fully redacted" if any fragment is missed.
- ~9% of PII items are never read by Tesseract at all, so ~90% is the text-model ceiling on this benchmark.
- Both remaining gaps are pipeline problems, handled in Phase 5 (best-of-both OCR, address box merging).
