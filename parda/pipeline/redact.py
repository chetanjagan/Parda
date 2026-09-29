"""Redact documents (images or PDFs) with the full Parda pipeline.

  python -m parda.pipeline.redact scan.jpg form.pdf --out_dir redacted/ --lang auto \
         --gliner <you>/parda-gliner-v1 --yolo <you>/parda-yolo-v1

For every input writes:
  <name>_redacted.pdf   the redacted pages. Image-only on purpose: no hidden text layer survives under the boxes.
  <name>_audit.json     what was redacted (type, box, confidence, which component found it). Never the PII text.
  --png                 also <name>_p<N>_redacted.png per page;  --preview: outlines instead of black boxes.

Without --gliner only the rules run (IDs with checksums); without --yolo nothing visual is redacted.
"""
import argparse
import json
import os
import tempfile
from collections import Counter

from PIL import Image

from ..ocr.engines import make_engine
from ..pii.predict import GLiNERPredictor, RulesPredictor, UnionPredictor
from .boxes import draw_preview
from .models import gliner as load_gliner_model
from .models import yolo_weights
from .redactor import Redactor, audit_entry

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def load_pages(path, dpi=200):
    """-> list of RGB PIL images (one per page)."""
    if path.lower().endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise SystemExit("PDF input needs PyMuPDF: pip install pymupdf")
        pages = []
        with fitz.open(path) as doc:
            for page in doc:
                pm = page.get_pixmap(dpi=dpi, alpha=False)
                pages.append(Image.frombytes("RGB", (pm.width, pm.height), pm.samples))
        return pages
    if path.lower().endswith(IMAGE_EXT):
        img = Image.open(path)
        pages = []
        for i in range(getattr(img, "n_frames", 1)):  # multi-page TIFF
            img.seek(i)
            pages.append(img.convert("RGB"))
        return pages
    raise SystemExit(f"unsupported file type: {path}")


def build_redactor(ocr="tesseract", gliner=None, threshold=0.3, rules=True, yolo=None, visual_conf=0.25, gap_fill=True):
    preds = []
    if gliner:
        preds.append(GLiNERPredictor(load_gliner_model(gliner), threshold=threshold, name="gliner"))
    if rules:
        preds.append(RulesPredictor())
    text_pred = None if not preds else preds[0] if len(preds) == 1 else UnionPredictor(*preds)
    visual = None
    if yolo:
        from ..vision.detectors import YoloDetector
        visual = YoloDetector(yolo_weights(yolo), conf=min(0.05, visual_conf))
    engines = [make_engine(n) for n in ocr.split(",") if n] if text_pred is not None else []
    return Redactor(engines, text_pred, visual, gap_fill=gap_fill, visual_conf=visual_conf)


def redact_file(path, red, out_dir, lang="auto", dpi=200, png=False, preview=False):
    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    pages_out, audit = [], {"input": os.path.basename(path), "lang": lang, "pages": []}
    with tempfile.TemporaryDirectory() as tmp:
        for i, page in enumerate(load_pages(path, dpi), 1):
            p = os.path.join(tmp, f"page{i}.png")
            page.save(p)
            regions, times = red.analyse(p, {"lang": lang})
            out = draw_preview(page, regions) if preview else red.redact(page, regions)
            pages_out.append(out)
            if png:
                out.save(os.path.join(out_dir, f"{name}_p{i}_{'preview' if preview else 'redacted'}.png"))
            audit["pages"].append({"page": i, "size": list(page.size), "regions": audit_entry(regions),
                                   "seconds": {k: round(v, 3) for k, v in times.items()}})
    pdf = os.path.join(out_dir, f"{name}_{'preview' if preview else 'redacted'}.pdf")
    pages_out[0].save(pdf, save_all=True, append_images=pages_out[1:], resolution=dpi)
    json.dump(audit, open(os.path.join(out_dir, f"{name}_audit.json"), "w"), indent=2)
    counts = Counter(r["label"] for pg in audit["pages"] for r in pg["regions"])
    return pdf, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out_dir", default="redacted")
    ap.add_argument("--lang", default="auto", choices=["auto", "en", "hi-en", "kn-en"])
    ap.add_argument("--ocr", default="tesseract", help="comma list: tesseract,easyocr (best of both)")
    ap.add_argument("--gliner", default=None, help="local path or HF repo of the fine-tuned GLiNER")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--no_rules", action="store_true")
    ap.add_argument("--no_gap_fill", action="store_true")
    ap.add_argument("--yolo", default=None, help="local best.pt or HF repo of the YOLO model")
    ap.add_argument("--visual_conf", type=float, default=0.25)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--preview", action="store_true", help="draw outlines instead of black boxes")
    a = ap.parse_args(argv)
    red = build_redactor(a.ocr, a.gliner, a.threshold, not a.no_rules, a.yolo, a.visual_conf, not a.no_gap_fill)
    results = {}
    for path in a.inputs:
        pdf, counts = redact_file(path, red, a.out_dir, a.lang, a.dpi, a.png, a.preview)
        results[path] = dict(counts)
        print(f"{path} -> {pdf}  {dict(counts)}", flush=True)
    return results


if __name__ == "__main__":
    main()
