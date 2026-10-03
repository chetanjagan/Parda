# Context rules — measured and rejected

Idea: cut false bars on documents the model was never trained on (bank statements) with three rules applied after
the model (`parda/pii/context.py`, mirrored in `web/src/core/context.ts`, identical on 150 golden cases):

- a **date of birth** needs a birth label (Birth, DOB, D.O.B, born, जन्म, ಜನ್ಮ) before it on the same line
- a **12-digit Aadhaar written without spaces** needs an Aadhaar label there (real Aadhaar numbers are printed in groups of four)
- a span that **starts or ends inside a number** (e.g. the "74" of "13,274") is dropped

Notebook 15 ran both benchmarks with the rules (shipped ONNX models, saved OCR):

| Benchmark | Configuration | All PII fully hidden | Over-redaction | Pages with no leak |
|---|---|---|---|---|
| seen | full Parda | 97.2% → 95.7% (−1.5) | 38.3% → 38.1% (−0.2) | 74.7% → 59.3% |
| seen | browser build (Tesseract only) | 89.7% → 87.9% (−1.8) | 23.3% → 23.3% (−0.1) | 41.3% → 29.7% |
| unseen | full Parda | 87.4% → 85.7% (−1.7) | 45.0% → 44.7% (−0.2) | 36.3% → 33.0% |
| unseen | browser build (Tesseract only) | 67.6% → 66.5% (−1.2) | 30.7% → 30.6% (−0.1) | 11.0% → 10.7% |

**Rejected.** The rules remove real PII (labels misread by OCR or on another line), and one missed date of birth leaks
a whole page; the reduction in over-redaction is negligible. A privacy tool should not trade leaks for fewer extra bars,
so the option was removed from the app. The better fix for statements is training data: synthetic bank statements and
tables, so the model learns that reference numbers, amounts and transaction dates are not personal data.
