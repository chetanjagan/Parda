# Phase 5 results — full pipeline, real-world test, visual ink check

All scores are end to end, on the black boxes actually drawn. A text PII item counts as fully hidden only if
every one of its words is ≥90% blacked out; a visual item if ≥95% of its box is. "Pages with no leak" = every
item on the page fully hidden. Each row adds one component to the row above. Text model: v1.

## 1. Seen templates (notebook 05) — 300 held-out benchmark pages

| Configuration | All PII fully hidden | Text | Visual | Pages with no leak | Over-redaction | s/page |
|---|---|---|---|---|---|---|
| Rules only | 18.7% | 23.1% | 0% | 0% | 31.5% | 2.33 |
| Off-the-shelf GLiNER + rules | 51.3% | 62.9% | 1.0% | 2.0% | 41.6% | 2.41 |
| Fine-tuned GLiNER + rules | 69.1% | 85.0% | 0% | 4.0% | 31.8% | 2.41 |
| + gap fill | 70.4% | 86.7% | 0% | 4.0% | 35.3% | 2.41 |
| + EasyOCR (best of both) | 77.3% | 95.1% | 0% | 5.3% | 51.1% | 3.78 |
| **+ YOLO visual = Parda** | **96.1%** | **95.1%** | **100%** | **68.0%** | **33.0%** | **3.81** |

By language (Parda): en 96.1%, hi-en 95.6%, kn-en 96.4%.
Fixed since Phase 3: employee ID 24% → 95.2%, person name 39% → 98.2%, address 7% → 53.1%.

## 2. Real-world test (notebook 06) — unseen layouts + phone photos

300 pages of 4 document types never seen in training (bank statement, KYC form, pharmacy bill, offer letter),
photographed with a phone (desk, perspective, shadow, blur, JPEG). Held-out faces; Aadhaar-lookalike
reference numbers as hard negatives.

| Configuration | All PII fully hidden | Text | Visual | Pages with no leak | Over-redaction | s/page |
|---|---|---|---|---|---|---|
| Rules only | 17.9% | 20.2% | 0% | 0% | 30.0% | 1.77 |
| Off-the-shelf GLiNER + rules | 38.9% | 43.9% | 0% | 0.3% | 43.8% | 1.82 |
| Fine-tuned GLiNER + rules | 56.0% | 63.2% | 0% | 1.0% | 42.4% | 1.82 |
| + gap fill | 57.6% | 64.9% | 0% | 2.0% | 44.5% | 1.82 |
| + EasyOCR (best of both) | 74.1% | 83.6% | 0.2% | 7.7% | 58.0% | 3.47 |
| **+ YOLO visual = Parda** | **82.5%** | **83.6%** | **73.7%** | **30.0%** | **41.6%** | **3.50** |

By language (Parda): en 81.9%, hi-en 82.8%, kn-en 82.7%.
Weakest types: address 52.3%, UPI ID 77.7%, email 82.2%, person name 84.3%.

| Parda | Seen templates | Unseen layouts + phone photos |
|---|---|---|
| All PII fully hidden | 96.1% | 82.5% |
| Pages with no leak | 68.0% | 30.0% |
| Over-redaction | 33.0% | 41.6% |

## 3. Visual PII: ink actually hidden (notebook 07) — the unseen photos

| Class | n | Detected (IoU ≥ 0.5) | Box rule (≥95% of box) | Ink fully hidden (≥98%) | Mean ink hidden |
|---|---|---|---|---|---|
| Face | 75 | 100% | 100% | 100% | 100% |
| Signature | 225 | 100% | 44.4% | 88.9% | 99.3% |
| QR code | 48 | 100% | 100% | 100% | 100% |
| Stamp | 128 | 100% | 100% | 100% | 100% |

YOLO found every signature on pages it never saw. On a tilted photo, the true box of a wide, thin signature is
much bigger than the signature itself, so the box rule fails even when every stroke is hidden (see the
`signature_*.png` examples: original | redacted). Scoring visuals by ink, all PII fully hidden on the unseen
photos is ≈84.9% for v1 (computed from the item counts).

The text model v2 results are in `results/text_v2/`.
