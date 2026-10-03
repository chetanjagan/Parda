### Context rules (dates of birth, unspaced Aadhaar, spans cutting numbers)

| Benchmark | Configuration | All PII fully hidden | Over-redaction | Pages with no leak |
|---|---|---|---|---|
| seen | + YOLO visual = Parda | 97.2% → **95.7%** (-1.5) | 38.3% → **38.1%** (-0.2) | 74.7% → 59.3% |
| seen | browser build: Tesseract only + YOLO | 89.7% → **87.9%** (-1.8) | 23.3% → **23.3%** (-0.1) | 41.3% → 29.7% |
| unseen | + YOLO visual = Parda | 87.4% → **85.7%** (-1.7) | 45.0% → **44.7%** (-0.2) | 36.3% → 33.0% |
| unseen | browser build: Tesseract only + YOLO | 67.6% → **66.5%** (-1.2) | 30.7% → **30.6%** (-0.1) | 11.0% → 10.7% |

**Do not turn them on by default**: they hide less real PII or do not reduce over-redaction. (Per-rule runs: --context dob / aadhaar / cut.)
