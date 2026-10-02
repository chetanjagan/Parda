### Browser OCR end to end

| Benchmark | Configuration | All PII fully hidden: native Tesseract → Tesseract.js | Pages with no leak (Tesseract.js) |
|---|---|---|---|
| seen | fine-tuned GLiNER + rules | 69.5% → **69.5%** | 5.0% |
| seen | browser build: Tesseract only + YOLO | 89.7% → **90.0%** | 46.3% |
| seen | + YOLO visual = Parda | 97.2% → **96.8%** | 73.7% |
| unseen | fine-tuned GLiNER + rules | 57.6% → **56.0%** | 2.7% |
| unseen | browser build: Tesseract only + YOLO | 67.6% → **66.4%** | 12.3% |
| unseen | + YOLO visual = Parda | 87.4% → **86.6%** | 36.0% |

Rows with EasyOCR use the saved native EasyOCR text (EasyOCR in the browser is milestone M4b).
