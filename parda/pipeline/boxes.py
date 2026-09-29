"""Geometry: text spans -> pixel boxes, gap filling, masks, drawing.

A region is {"kind": "text"|"visual", "label": str, "bbox": [x0, y0, x1, y1], "score": float, "source": str}.
"""
import numpy as np
from PIL import Image, ImageDraw


def span_boxes(start, end, ordered, offs):
    """Pixel boxes covering text[start:end]. ordered/offs come from spans.build_ocr_text.

    Word-level OCR (Tesseract): whole word boxes. Line-level OCR (EasyOCR) where the span covers only part
    of a segment: the box is cut by character position, widened by one character each side because real
    characters are not all the same width."""
    out = []
    for seg, (a, b) in zip(ordered, offs):
        lo, hi = max(a, start), min(b, end)
        if lo >= hi:
            continue
        x0, y0, x1, y1 = [float(v) for v in seg["bbox"]]
        n = b - a
        if (lo == a and hi == b) or n <= 0:
            out.append([x0, y0, x1, y1])
            continue
        w = (x1 - x0) / n
        out.append([max(x0, x0 + (lo - a) * w - w), y0, min(x1, x0 + (hi - a) * w + w), y1])
    return out


def _same_line(a, b):
    ov = min(a[3], b[3]) - max(a[1], b[1])
    return ov >= 0.5 * min(a[3] - a[1], b[3] - b[1])


def fill_gaps(regions, max_gap=2.5):
    """Join same-label text boxes that sit on one line with a small gap between them (gap <= max_gap x the
    line height). Covers words OCR skipped inside a PII item, e.g. an address word it could not read."""
    text = [dict(r, bbox=list(r["bbox"])) for r in regions if r["kind"] == "text"]
    other = [r for r in regions if r["kind"] != "text"]
    out = []
    for label in sorted({r["label"] for r in text}):
        boxes = sorted((r for r in text if r["label"] == label), key=lambda r: (r["bbox"][1], r["bbox"][0]))
        merged = []
        for r in boxes:
            for m in merged:
                a, b = m["bbox"], r["bbox"]
                h = max(a[3] - a[1], b[3] - b[1])
                gap = max(a[0], b[0]) - min(a[2], b[2])
                if _same_line(a, b) and gap <= max_gap * h:
                    m["bbox"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    m["score"] = max(m["score"], r["score"])
                    break
            else:
                merged.append(r)
        out += merged
    return out + other


def pix(box, W, H, pad=0, eps=1e-3):
    """Float box -> integer pixel range; eps ignores float noise (99.9999 is pixel 100, not 99)."""
    x0 = max(0, int(np.floor(box[0] - pad + eps)))
    y0 = max(0, int(np.floor(box[1] - pad + eps)))
    x1 = min(W, int(np.ceil(box[2] + pad - eps)))
    y1 = min(H, int(np.ceil(box[3] + pad - eps)))
    return x0, y0, max(x0, x1), max(y0, y1)


PAD = {"text": 3, "visual": 6}  # px of safety margin around each redaction box


def _pad(r, pad):
    return pad.get(r["kind"], 0) if isinstance(pad, dict) else pad


def redaction_mask(regions, W, H, pad=PAD):
    m = np.zeros((H, W), bool)
    for r in regions:
        x0, y0, x1, y1 = pix(r["bbox"], W, H, _pad(r, pad))
        m[y0:y1, x0:x1] = True
    return m


def draw(image, regions, pad=PAD, fill=(0, 0, 0)):
    """Black out every region. Returns a new RGB image; the original pixels under the boxes are gone."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    for r in regions:
        x0, y0, x1, y1 = pix(r["bbox"], W, H, _pad(r, pad))
        if x1 > x0 and y1 > y0:
            d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill)
    return img


def draw_preview(image, regions, pad=PAD):
    """Outlines + labels instead of black boxes (for checking what would be redacted)."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    for r in regions:
        x0, y0, x1, y1 = pix(r["bbox"], W, H, _pad(r, pad))
        col = (220, 30, 30) if r["kind"] == "text" else (30, 120, 220)
        d.rectangle([x0, y0, x1, y1], outline=col, width=2)
        d.text((x0 + 2, max(0, y0 - 11)), r["label"], fill=col)
    return img


def as_image(x):
    return x if isinstance(x, Image.Image) else Image.open(x)
