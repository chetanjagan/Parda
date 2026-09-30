"""Did the redaction hide the visual PII's actual INK? (versus the strict box-coverage rule)

  python -m parda.vision.ink_check --data <pages with annotations.jsonl> --yolo <repo or best.pt> --out outputs/ink

For every true visual item (face, signature, QR code, stamp):
  detected     a prediction of the same class overlaps it with IoU >= 0.5
  box rule     >= 95% of its box is blacked out (the Phase 4/5 rule, predictions padded like the pipeline)
  ink hidden   share of its INK pixels (clearly darker than the paper around it) that are blacked out
On a tilted photo the true box of a wide, thin signature is the box *around* the tilted signature, so it is
much bigger than the signature itself: the box rule can fail even when every stroke is hidden. "Ink hidden"
measures what a reader could still see.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from ..pipeline.boxes import PAD, pix, redaction_mask
from ..pipeline.evaluate_e2e import load_pages
from ..pipeline.redactor import visual_regions
from .classes import CLASSES
from .evaluate_vis import iou

BOX_RULE, INK_FULL, DARK = 0.95, 0.98, 0.72


def ink_mask(gray, box, ring=8):
    """Boolean mask (same shape as the box crop) of pixels clearly darker than the paper around the box."""
    H, W = gray.shape
    x0, y0, x1, y1 = pix(box, W, H)
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    ex0, ey0, ex1, ey1 = pix(box, W, H, ring)
    around = gray[ey0:ey1, ex0:ex1].astype(np.float32).copy()
    around[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = np.nan  # keep only the ring
    ring_px = around[~np.isnan(around)]
    paper = np.percentile(ring_px, 90) if ring_px.size else np.percentile(crop, 90) if crop.size else 255.0
    return crop < DARK * paper, (x0, y0, x1, y1)


def score_item(gray, gt_box, black):
    """black: boolean mask of redacted pixels (whole page). -> {box_cover, ink_hidden, ink_px}"""
    ink, (x0, y0, x1, y1) = ink_mask(gray, gt_box)
    region = black[y0:y1, x0:x1]
    box_cover = float(region.mean()) if region.size else 1.0
    n_ink = int(ink.sum())
    ink_hidden = float(region[ink].mean()) if n_ink else 1.0
    return {"box_cover": box_cover, "ink_hidden": ink_hidden, "ink_px": n_ink}


def check(pages, dets, conf=0.25, pad=PAD, examples_dir=None, n_examples=6):
    rows = {c: [] for c in CLASSES}
    saved = 0
    for p, d in zip(pages, dets):
        regions = visual_regions(d, conf)
        img = Image.open(p["path"]).convert("RGB")
        gray = np.asarray(img.convert("L"))
        H, W = gray.shape
        black = redaction_mask(regions, W, H, pad)
        for v in p["rec"]["visuals"]:
            if v["label"] not in rows:
                continue
            s = score_item(gray, v["bbox"], black)
            s["detected"] = any(r["label"] == v["label"] and iou(r["bbox"], v["bbox"]) >= 0.5 for r in regions)
            rows[v["label"]].append(s)
            if (examples_dir and saved < n_examples and v["label"] == "SIGNATURE"
                    and s["box_cover"] < BOX_RULE and s["ink_hidden"] >= INK_FULL):  # box rule fails, ink hidden
                os.makedirs(examples_dir, exist_ok=True)
                x0, y0, x1, y1 = pix(v["bbox"], W, H, 12)
                red = img.copy()
                red.paste((0, 0, 0), mask=Image.fromarray((black * 255).astype(np.uint8)))
                pair = Image.new("RGB", ((x1 - x0) * 2 + 10, y1 - y0), "white")
                pair.paste(img.crop((x0, y0, x1, y1)), (0, 0))
                pair.paste(red.crop((x0, y0, x1, y1)), (x1 - x0 + 10, 0))
                pair.save(os.path.join(examples_dir, f"signature_{p['id']:06d}.png"))
                saved += 1
    out = {}
    for c, rs in rows.items():
        n = len(rs)
        out[c] = None if not n else {
            "n": n,
            "detected": sum(r["detected"] for r in rs) / n,
            "box_rule": sum(r["box_cover"] >= BOX_RULE for r in rs) / n,
            "ink_fully_hidden": sum(r["ink_hidden"] >= INK_FULL for r in rs) / n,
            "mean_ink_hidden": float(np.mean([r["ink_hidden"] for r in rs])),
        }
    return out


def report_md(res, title):
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    L = [f"# {title}", "",
         f"Detected = same class, IoU ≥ 0.5. Box rule = ≥{int(100 * BOX_RULE)}% of the true box blacked out "
         f"(Phase 4/5 rule). Ink fully hidden = ≥{int(100 * INK_FULL)}% of the item's dark pixels blacked out.", "",
         "| Class | n | Detected | Box rule | Ink fully hidden | Mean ink hidden |", "|---|---|---|---|---|---|"]
    for c in CLASSES:
        r = res.get(c)
        if r:
            L.append(f"| {c} | {r['n']} | {pct(r['detected'])} | {pct(r['box_rule'])} | "
                     f"{pct(r['ink_fully_hidden'])} | {pct(r['mean_ink_hidden'])} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--yolo", required=True, help="local best.pt or HF repo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--title", default="Visual PII: box rule vs ink actually hidden")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from ..pipeline.models import yolo_weights
    from .detectors import YoloDetector
    pages = load_pages(a.data, None, a.limit)
    dets = YoloDetector(yolo_weights(a.yolo), conf=0.05).detect([p["path"] for p in pages])
    res = check(pages, dets, a.conf, examples_dir=os.path.join(a.out, "examples"))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "ink_check.json"), "w"), indent=2)
    md = report_md(res, a.title)
    open(os.path.join(a.out, "ink_check.md"), "w", encoding="utf-8").write(md)
    print(md, flush=True)
    return res


if __name__ == "__main__":
    main()
