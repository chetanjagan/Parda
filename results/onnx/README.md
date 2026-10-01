# Phase 6, step 1 — models for the browser (ONNX)

Notebook 09 exported GLiNER v2 and YOLO11n to ONNX; notebook 10 compressed GLiNER. Every variant was checked
against PyTorch on 200 (09) or 100 (10) real OCR page texts, and the chosen one on both full benchmarks.
All runs on CPU, like a browser. OCR was reused from the v2 run, so only the models change.

## GLiNER v2

| Variant | Size | Same PII spans as PyTorch | Seen, full Parda | Unseen photos, full Parda |
|---|---|---|---|---|
| PyTorch | 1.16 GB | – | 97.2% | 87.4% |
| ONNX fp32 | 1,103.6 MB | 100% | 97.2% | 87.4% |
| INT8, every layer (QUInt8, dynamic) | 337.5 MB | 36.2% | 75.7% | 59.8% |
| INT8, matrix multiplications only (per-channel) | 887.5 MB | 0.0% | – | – |
| INT8, embeddings + matrix multiplications | 338.0 MB | 0.0% | – | – |
| FP16 | – | does not load (type mismatch from the converter) | – | – |
| **INT8, word-embedding table only (shipped)** | **554.5 MB** | **99.0%** | **97.2%** | **87.4%** |

Largest benchmark drop of the shipped model vs PyTorch: **0.0 points** (pages with no leak unchanged:
74.7% seen, 36.3% unseen).

## YOLO11n

ONNX (10.4 MB, one image per call): fully redacted 100% and mAP50 1.000 on the 300 benchmark pages,
identical to PyTorch.

## Browser build (Tesseract only; EasyOCR has no browser version)

| | Full Parda (Tesseract + EasyOCR) | Browser build (Tesseract only) |
|---|---|---|
| Seen templates | 97.2% | 89.7% |
| Unseen phone photos | 87.4% | 67.6% |

On phone photos the second OCR engine is worth about 20 points, so the web app needs a second OCR engine that
runs in the browser (or an optional server mode).

## How the export works

- GLiNER passes the word count as a Python list; a plain export froze it (47, then 256) and the LSTM crashed on
  other lengths. GLiNER's packed LSTM is replaced, for export only, by an exact equivalent fed by a real length
  input (forward pass over the padded sequence; backward pass on each sequence reversed within its length),
  verified against the original before exporting. Texts are padded to 256 words; the output is cut back.
- The exported network is run inside the normal GLiNER object, so tokenisation and decoding are unchanged.

Files: `export_summary.json`, `compress_summary.json`, `onnx_vs_pytorch.md`, `compressed_vs_pytorch.md`,
`report_seen*.md/json`, `report_unseen*.md/json`. Models: private HF repo `parda-onnx-v1`.
