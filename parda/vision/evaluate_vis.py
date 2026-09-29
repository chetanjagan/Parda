"""Score visual PII detectors on the benchmark split of a YOLO dataset built by yolo_data.py.

  python -m parda.vision.evaluate_vis --yolo_data data/yolo --systems opencv,yolo --weights best.pt \
         --thresholds 0.25,0.5 --out outputs/vis_eval

Metrics (per true box, then averaged):
  Fully redacted   >= 95% of the true box's pixels are covered by predicted boxes (any class) after padding
                   every prediction by --pad px, the same padding the redaction pipeline applies.
  Detected         a prediction of the same class overlaps it with IoU >= 0.5 (unpadded).
  Precision        share of predictions that match a true box of their class (IoU >= 0.5).
  AP50             average precision at IoU 0.5 over all confidences (threshold-free); mAP50 = mean over classes.
  Over-redaction   share of blacked-out pixels (padded predictions) that are not inside any true box.
"""
import argparse
import json
import math
import os
import time

import numpy as np
from PIL import Image

from .classes import CLASSES
from .detectors import make, read_gt

COVER = 0.95


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _pix(box, W, H, pad=0):
    x0 = max(0, int(math.floor(box[0] - pad)))
    y0 = max(0, int(math.floor(box[1] - pad)))
    x1 = min(W, int(math.ceil(box[2] + pad)))
    y1 = min(H, int(math.ceil(box[3] + pad)))
    return x0, y0, max(x0, x1), max(y0, y1)


def ap50(preds, gts, cls):
    """VOC all-point AP at IoU 0.5 for one class. preds/gts: per-image lists. None if the class never occurs."""
    gt_boxes = [[g["bbox"] for g in im if g["label"] == cls] for im in gts]
    npos = sum(len(g) for g in gt_boxes)
    if npos == 0:
        return None
    flat = [(p["score"], i, p["bbox"]) for i, im in enumerate(preds) for p in im if p["label"] == cls]
    flat.sort(key=lambda t: -t[0])
    used = [[False] * len(g) for g in gt_boxes]
    tp, fp = [], []
    for _, i, box in flat:
        best, bj = 0.0, -1
        for j, g in enumerate(gt_boxes[i]):
            o = iou(box, g)
            if o > best:
                best, bj = o, j
        if best >= 0.5 and not used[i][bj]:
            used[i][bj] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = (tp / npos).tolist() if len(tp) else []
    prec = (tp / np.maximum(tp + fp, 1e-9)).tolist() if len(tp) else []
    mrec, mpre = [0.0] + rec + [1.0], [0.0] + prec + [0.0]
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    return float(sum((mrec[k] - mrec[k - 1]) * mpre[k] for k in range(1, len(mrec)) if mrec[k] != mrec[k - 1]))


def score(gts, preds, sizes, metas, thr, pad):
    """Threshold-dependent metrics. gts/preds/sizes/metas are aligned per image."""
    per_cls = {c: {"n": 0, "full": 0, "det": 0, "preds": 0, "tp": 0} for c in CLASSES}
    per_lang = {}
    pix_pred = pix_over = 0
    n_pred = n_tp = 0
    for gt, pr_all, (W, H), meta in zip(gts, preds, sizes, metas):
        pr = [p for p in pr_all if p["score"] >= thr and p["label"] in per_cls]
        pmask = np.zeros((H, W), bool)
        gmask = np.zeros((H, W), bool)
        for p in pr:
            x0, y0, x1, y1 = _pix(p["bbox"], W, H, pad)
            pmask[y0:y1, x0:x1] = True
        for g in gt:
            x0, y0, x1, y1 = _pix(g["bbox"], W, H)
            gmask[y0:y1, x0:x1] = True
        pix_pred += int(pmask.sum())
        pix_over += int((pmask & ~gmask).sum())
        lang = per_lang.setdefault(meta["lang"], {"n": 0, "full": 0})
        for g in gt:
            x0, y0, x1, y1 = _pix(g["bbox"], W, H)
            area = (x1 - x0) * (y1 - y0)
            full = area > 0 and pmask[y0:y1, x0:x1].mean() >= COVER
            det = any(p["label"] == g["label"] and iou(p["bbox"], g["bbox"]) >= 0.5 for p in pr)
            c = per_cls[g["label"]]
            c["n"] += 1
            c["full"] += full
            c["det"] += det
            lang["n"] += 1
            lang["full"] += full
        # precision: greedy one-to-one matching, highest score first
        used = set()
        for p in sorted(pr, key=lambda p: -p["score"]):
            n_pred += 1
            per_cls[p["label"]]["preds"] += 1
            best, bj = 0.0, -1
            for j, g in enumerate(gt):
                if j in used or g["label"] != p["label"]:
                    continue
                o = iou(p["bbox"], g["bbox"])
                if o > best:
                    best, bj = o, j
            if best >= 0.5:
                used.add(bj)
                n_tp += 1
                per_cls[p["label"]]["tp"] += 1
    n = sum(c["n"] for c in per_cls.values())
    ratio = lambda a, b: (a / b) if b else None  # noqa: E731
    return {
        "fully_redacted": ratio(sum(c["full"] for c in per_cls.values()), n),
        "detected": ratio(sum(c["det"] for c in per_cls.values()), n),
        "precision": ratio(n_tp, n_pred),
        "over_redaction": ratio(pix_over, pix_pred) if pix_pred else 0.0,
        "n_true": n, "n_pred": n_pred,
        "by_class": {k: {"n": c["n"], "fully_redacted": ratio(c["full"], c["n"]), "detected": ratio(c["det"], c["n"]),
                         "precision": ratio(c["tp"], c["preds"])} for k, c in per_cls.items()},
        "by_lang": {k: ratio(v["full"], v["n"]) for k, v in sorted(per_lang.items())},
    }


def load_split(yolo_data, split="bench", limit=None):
    metas = [json.loads(ln) for ln in open(os.path.join(yolo_data, split, "meta.jsonl"), encoding="utf-8")]
    if limit:
        metas = metas[:limit]
    paths = [os.path.join(os.path.abspath(yolo_data), "images", split, m["file"]) for m in metas]
    sizes = [Image.open(p).size for p in paths]
    gts = [read_gt(p, s) for p, s in zip(paths, sizes)]
    return metas, paths, sizes, gts


def evaluate(yolo_data, systems, weights=None, thresholds=(0.25, 0.5), pad=6, split="bench", imgsz=1024, limit=None):
    metas, paths, sizes, gts = load_split(yolo_data, split, limit)
    rows = []
    for sysname in systems:
        t0 = time.time()
        det = make(sysname, weights, imgsz)
        preds = det.detect(paths)
        secs = time.time() - t0
        aps = {c: ap50(preds, gts, c) for c in CLASSES}
        present = [v for v in aps.values() if v is not None]
        # systems without real confidences (score 1.0) are scored once
        ths = thresholds if sysname == "yolo" else (0.0,)
        for thr in ths:
            s = score(gts, preds, sizes, metas, thr, pad)
            s.update({"system": f"{sysname}@{thr:g}" if sysname == "yolo" else sysname,
                      "ap50": aps, "map50": sum(present) / len(present) if present else None,
                      "sec_per_page": secs / max(1, len(paths))})
            rows.append(s)
    return {"split": split, "pages": len(paths), "pad": pad, "cover": COVER, "systems": rows}


def _pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def report_md(res):
    rs = res["systems"]
    langs = sorted({k for r in rs for k in r["by_lang"]})
    L = [f"# Phase 4 — visual PII on {res['pages']} benchmark pages ({res['split']})", "",
         f"Fully redacted = ≥{int(100 * res['cover'])}% of the true box covered after padding predictions by "
         f"{res['pad']} px. Detected = same class, IoU ≥ 0.5. AP50 is threshold-free.", "",
         "## Overall", "",
         "| System | Fully redacted ↑ | Detected ↑ | Precision ↑ | mAP50 ↑ | Over-redaction ↓ | s/page |",
         "|---|---|---|---|---|---|---|"]
    for r in rs:
        L.append(f"| {r['system']} | {_pct(r['fully_redacted'])} | {_pct(r['detected'])} | {_pct(r['precision'])} | "
                 f"{_pct(r['map50'])} | {_pct(r['over_redaction'])} | {r['sec_per_page']:.3f} |")
    L += ["", "## Fully redacted, by language", "", "| System | " + " | ".join(langs) + " |",
          "|---|" + "---|" * len(langs)]
    for r in rs:
        L.append(f"| {r['system']} | " + " | ".join(_pct(r["by_lang"].get(k)) for k in langs) + " |")
    L += ["", "## By class: fully redacted / AP50", "",
          "| Class | n | " + " | ".join(r["system"] for r in rs) + " |", "|---|---|" + "---|" * len(rs)]
    for c in CLASSES:
        n = rs[0]["by_class"][c]["n"] if rs else 0
        cells = [f"{_pct(r['by_class'][c]['fully_redacted'])} / {_pct(r['ap50'][c])}" for r in rs]
        L.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo_data", required=True)
    ap.add_argument("--systems", default="opencv,yolo")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--thresholds", default="0.25,0.5")
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--split", default="bench")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = evaluate(a.yolo_data, [s for s in a.systems.split(",") if s], a.weights,
                   [float(t) for t in a.thresholds.split(",")], a.pad, a.split, a.imgsz, a.limit)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "report_vis.json"), "w"), indent=2)
    md = report_md(res)
    open(os.path.join(a.out, "report_vis.md"), "w", encoding="utf-8").write(md)
    print(md, flush=True)
    return res


if __name__ == "__main__":
    main()
