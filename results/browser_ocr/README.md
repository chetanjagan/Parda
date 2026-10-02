
# Browser OCR (Tesseract.js) — web app milestone M4a

The web app reads pages with Tesseract.js (`web/src/ocr`, same settings as the Python TesseractEngine).
Notebook 13 ran it on both benchmarks and scored it against the true text and end to end (shipped ONNX models).

| All PII fully hidden                        | Native Tesseract | Tesseract.js    |
| ------------------------------------------- | ---------------- | --------------- |
| Seen, browser build (Tesseract only + YOLO) | 89.7%            | **90.0%** |
| Seen, full Parda (+ EasyOCR)                | 97.2%            | **96.8%** |
| Unseen photos, browser build                | 67.6%            | **66.4%** |
| Unseen photos, full Parda                   | 87.4%            | **86.6%** |

Reading quality vs the true text: on clean scans Tesseract.js is better (page error 23.7% → 19.8%, word recall
75.3% → 80.7%, Kannada recall 76% → 83%); on phone photos the two are about equal (PII found 46.1% vs 44.6%).
~2 s/page on CPU for both.

On phone photos OCR reads only ~45% of PII items exactly, yet Parda hides 87%: the model was trained on noisy OCR
text, and boxes cover a word's position even when it is misread. The browser build alone (66.4% on photos) shows the
second OCR engine is still worth ~20 points: EasyOCR in the browser is milestone M4b.
