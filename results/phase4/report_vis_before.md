# Phase 4 results — visual PII detector (YOLO11n)

300 held-out benchmark pages (100 per language; the same pages as Phases 2–3), 600 true boxes.
Faces on these pages are real CelebA photos from a held-out pool never used in training.

| System | Fully redacted | Detected (IoU≥0.5) | Precision | mAP50 | Over-redaction | s/page |
|---|---|---|---|---|---|---|
| OpenCV (Haar faces + QR detector) | 7.2% | 7.0% | 17.9% | 18.9% | 35.4% | 0.21 |
| **YOLO11n, fine-tuned** | **100%** | **100%** | **100%** | **100%** | **15.4%** | **0.035** |

By class (fully redacted), OpenCV → YOLO: FACE 0% → 100%, SIGNATURE 0.6% → 100%, QR_CODE 97.6% → 100%, STAMP 0% → 100%.

- Training: 12,000 pages, YOLO11n at 1024 px, 49 epochs (early stop), 177 min on one T4; val mAP50-95 0.995.
- Over-redaction is the 6 px safety padding (a padded 120×150 photo covers ~16% more area); precision is 100%.
- OpenCV's Haar detector finds the face itself, not the whole ID photo, so it never fully covers the box (0%).
- Caveat: scores saturate because the pages come from 5 synthetic templates. Real-scan performance is tested in Phase 5.