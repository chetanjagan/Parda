# Text model v2 — trained on phone photos and EasyOCR (notebooks 08a, 08b)

v1 learned from clean page text and Tesseract's mistakes on scans. The pipeline reads every page with
**two** OCR engines, and the real world has **phone photos**, so v2 adds:

| Training source | Pages | How the labels get onto the OCR text |
|---|---|---|
| Clean page text (as v1) | 10,000 | exact |
| Tesseract on scans (as v1) | 10,002 | by word position |
| **Tesseract on phone photos** | **6,000** | by word position |
| **EasyOCR on scans** | **1,200** | by character alignment |
| **EasyOCR on phone photos** | **1,200** | by character alignment |

The phone-photo pages are the 5 training templates with the same photo damage as the real-world test
(desk, perspective, shadow, blur, JPEG). The 300 benchmark pages and the 4 real-world-test layouts are never used.

**Character alignment:** EasyOCR returns whole lines ("Name: Ravi Kumar"), so position-based transfer labelled
only 7.7% of entities. v2 aligns each OCR line with the true text of the words it covers, character by
character, and carries the labels across. Measured on 600 pages: 100% of labels exact on perfect text,
98.9% at 15% character noise on phone photos.

## Results (same benchmarks, same OCR, same YOLO; only the text model changes)

| Full Parda | Seen templates | Unseen layouts + phone photos |
|---|---|---|
| All PII fully hidden, v1 → v2 | 96.1% → **97.2%** | 82.5% → **87.4%** |
| Text PII, v1 → v2 | 95.1% → **96.5%** | 83.6% → **89.1%** |
| Pages with no leak, v1 → v2 | 68.0% → 74.7% | 30.0% → 36.3% |
| Over-redaction, v1 → v2 | 33.0% → 38.3% | 41.6% → 44.9% |

| PII type (unseen photos) | v1 | v2 |
|---|---|---|
| Address | 52.3% | **67.3%** |
| UPI ID | 77.7% | **88.0%** |
| Person name | 84.3% | **89.6%** |
| Aadhaar | 96.0% | **100%** |
| Email | 82.2% | 86.7% |
| ABHA | 94.7% | 98.7% |
| Phone | 91.6% | 93.8% |

Seen templates: address 53.1% → 62.9%; every other type 97.6–100%.

- The gain comes mainly from the EasyOCR reading (unseen text 83.6% → 89.1% with both engines, versus
  +1.5 points with Tesseract alone): v2 learned EasyOCR's kind of mistakes, which v1 never saw.
- Cost: v2 hides more that is not PII (over-redaction +3 to +5 points). For a redaction tool, over-hiding is the
  safer error.
- The test layouts remain unseen, but the photo damage matches the training augmentation, so part of the
  unseen-photo gain is robustness to phone photos rather than to new layouts.
- Scoring signatures by ink hidden (99.3%, notebook 07) instead of the box rule puts v2 at ≈89.7% of all PII
  fully hidden on the unseen photos.

Model: private HF repo `parda-gliner-v2`. Files here: `v1_vs_v2.md`, `report_seen.*`, `report_unseen.*`,
`train_summary.json`, `data_stats.json`.
