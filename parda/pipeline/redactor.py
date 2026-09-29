"""The Parda pipeline: page image -> regions to redact.

  OCR engine(s) -> reading-order text -> text PII predictor (GLiNER + rules) -> pixel boxes (+ gap filling)
  visual detector (YOLO)                                                    -> pixel boxes
  union of everything -> black boxes

With several OCR engines ("best of both"), each engine's reading is searched separately and the boxes are
combined: whatever PII either reading reveals gets covered.
"""
import time

from ..pii.spans import build_ocr_text
from .boxes import PAD, draw, fill_gaps, span_boxes


def text_regions(segs, predictor, source, gap_fill=True):
    """OCR segments -> text regions. Returns (regions, n_spans)."""
    text, ordered, offs = build_ocr_text(segs)
    if not text:
        return [], 0
    spans = predictor.predict(text)
    regs = [{"kind": "text", "label": sp["label"], "bbox": b, "score": float(sp.get("score", 1.0)), "source": source}
            for sp in spans for b in span_boxes(sp["start"], sp["end"], ordered, offs)]
    return (fill_gaps(regs) if gap_fill else regs), len(spans)


def visual_regions(dets, conf=0.25):
    return [{"kind": "visual", "label": d["label"], "bbox": list(d["bbox"]), "score": float(d["score"]), "source": "yolo"}
            for d in dets if d["score"] >= conf]


class Redactor:
    def __init__(self, engines, text_predictor=None, visual_detector=None, gap_fill=True, visual_conf=0.25, pad=PAD):
        """engines: OCR engine objects (parda.ocr.engines); text_predictor: .predict(text) -> spans;
        visual_detector: .detect([paths]) -> [[{label, bbox, score}]] (parda.vision.detectors)."""
        self.engines, self.text_predictor, self.visual = list(engines), text_predictor, visual_detector
        self.gap_fill, self.visual_conf, self.pad = gap_fill, visual_conf, pad

    def analyse(self, image_path, rec):
        """rec: {"lang": "en"|"hi-en"|"kn-en"|"auto", ...}. Returns (regions, timings in seconds)."""
        regions, times = [], {}
        if self.text_predictor is not None:
            for eng in self.engines:
                t0 = time.time()
                segs = eng.run(image_path, rec)
                times[f"ocr_{eng.name}"] = time.time() - t0
                t0 = time.time()
                regs, _ = text_regions(segs, self.text_predictor, eng.name, self.gap_fill)
                times[f"text_pii_{eng.name}"] = time.time() - t0
                regions += regs
        if self.visual is not None:
            t0 = time.time()
            regions += visual_regions(self.visual.detect([image_path])[0], self.visual_conf)
            times["visual"] = time.time() - t0
        return regions, times

    def redact(self, image, regions):
        return draw(image, regions, self.pad)


def audit_entry(regions):
    """What was redacted, for the log: type, place, confidence, which component found it. Never the text."""
    return [{"kind": r["kind"], "label": r["label"], "bbox": [round(float(v), 1) for v in r["bbox"]],
             "score": round(float(r["score"]), 3), "source": r["source"]} for r in regions]
