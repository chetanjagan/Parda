"""End-to-end benchmark: redact the 300 fixed benchmark pages and check the PIXELS.

  python -m parda.pipeline.evaluate_e2e --data <synth> --yolo_data <yolo data with bench split> \
         --ocr tesseract,easyocr --gliner_ft <you>/parda-gliner-v1 --gliner_base urchade/gliner_multi_pii-v1 \
         --yolo <you>/parda-yolo-v1 --cache outputs/e2e_cache --out outputs/e2e

Scoring (on the black boxes actually drawn, with the pipeline's padding):
  text PII item    fully redacted if EVERY one of its words has >= 90% of its word box blacked out
  visual PII item  fully redacted if >= 95% of its box is blacked out
  page clean       every PII item on the page is fully redacted (nothing leaks)
  over-redaction   share of blacked-out pixels outside every true PII word/visual box

Ablation: each configuration adds one component, so every component's contribution is visible.
OCR results are cached per engine (resumable); models run once per OCR stream and are combined per config.
"""
import argparse
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

from ..ocr.engines import make_engine
from ..ocr.run_ocr import benchmark_ids, find_data
from ..vision.classes import CLASSES
from .boxes import PAD, fill_gaps, pix, redaction_mask
from .redactor import text_regions, visual_regions

COVER_WORD, COVER_VISUAL = 0.90, 0.95

# (name, OCR engines, text system, gap fill, YOLO)
CONFIGS = [
    ("rules only", ("tesseract",), "rules", False, False),
    ("off-the-shelf GLiNER + rules", ("tesseract",), "base+rules", False, False),
    ("fine-tuned GLiNER + rules", ("tesseract",), "ft+rules", False, False),
    ("+ gap fill", ("tesseract",), "ft+rules", True, False),
    ("+ EasyOCR (best of both)", ("tesseract", "easyocr"), "ft+rules", True, False),
    ("+ YOLO visual = Parda", ("*",), "ft+rules", True, True),  # "*" = every OCR engine given
]


# ------------------------------------------------------------------ OCR (cached, resumable)
_ENG = None


def _init_ocr(name):
    global _ENG
    _ENG = make_engine(name)


def _ocr_one(job):
    pid, path, rec = job
    t0 = time.time()
    try:
        return {"id": pid, "segments": _ENG.run(path, rec), "sec": time.time() - t0}
    except Exception as e:  # keep going; the page is scored with no text found
        return {"id": pid, "segments": [], "sec": time.time() - t0, "error": repr(e)}


def run_ocr(engine, pages, cache, workers=4):
    """engine: a name (parallel CPU workers for tesseract) or an engine object. -> {id: {segments, sec}}"""
    name = engine if isinstance(engine, str) else engine.name
    path = os.path.join(cache, f"ocr_{name}.jsonl")
    done = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            o = json.loads(ln)
            done[o["id"]] = o
    todo = [(p["id"], p["path"], p["ocr_rec"]) for p in pages if p["id"] not in done]
    if todo:
        print(f"OCR {name}: {len(todo)} pages to run ({len(done)} cached)", flush=True)
        os.makedirs(cache, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            if isinstance(engine, str) and engine == "tesseract" and workers > 1:
                with Pool(workers, initializer=_init_ocr, initargs=(name,)) as pool:
                    for o in pool.imap_unordered(_ocr_one, todo):
                        f.write(json.dumps(o, ensure_ascii=False) + "\n")
                        done[o["id"]] = o
            else:
                global _ENG
                _ENG = make_engine(engine) if isinstance(engine, str) else engine
                for job in todo:
                    o = _ocr_one(job)
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
                    f.flush()
                    done[o["id"]] = o
    return done


# ------------------------------------------------------------------ truth + scoring
def truth(rec):
    """[(kind, label, boxes that must be covered, PII area)] per PII item.

    Text: every word box must be covered; the PII area also includes the spaces between the item's words on
    the same line (blacking out "Rekha Bhat" as one bar is not over-redaction). Visual: its box."""
    items = []
    for e in rec["entities"]:
        boxes = [w["bbox"] for w in rec["words"] if w["start"] >= e["start"] and w["end"] <= e["end"]]
        if boxes:
            regs = [{"kind": "text", "label": "x", "bbox": b, "score": 1.0} for b in boxes]
            area = [r["bbox"] for r in fill_gaps(regs, max_gap=1e9)]  # join the item's words line by line
            items.append(("text", e["label"], boxes, area))
    for v in rec["visuals"]:
        if v["label"] in CLASSES:
            items.append(("visual", v["label"], [v["bbox"]], [v["bbox"]]))
    return items


def _covered(mask, box, W, H):
    x0, y0, x1, y1 = pix(box, W, H)
    return float(mask[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 1.0


def score_page(items, regions, W, H, pad=PAD):
    mask = redaction_mask(regions, W, H, pad)
    tmask = np.zeros((H, W), bool)
    res = []
    for kind, label, boxes, area in items:
        for b in area:
            x0, y0, x1, y1 = pix(b, W, H)
            tmask[y0:y1, x0:x1] = True
        need = COVER_WORD if kind == "text" else COVER_VISUAL
        res.append((kind, label, all(_covered(mask, b, W, H) >= need for b in boxes)))
    return res, int(mask.sum()), int((mask & ~tmask).sum())


class Acc:
    def __init__(self):
        self.n = self.full = 0

    def add(self, ok):
        self.n += 1
        self.full += bool(ok)

    def pct(self):
        return self.full / self.n if self.n else None


# ------------------------------------------------------------------ evaluation
def load_pages(data, yolo_data=None, limit=None):
    bench = benchmark_ids(data)
    recs = {}
    with open(os.path.join(data, "annotations.jsonl"), encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            if r["id"] in bench:
                recs[r["id"]] = r
    files = {}
    if yolo_data:  # the Phase 4 benchmark images: real (held-out) faces pasted over the avatars
        for ln in open(os.path.join(yolo_data, "bench", "meta.jsonl"), encoding="utf-8"):
            m = json.loads(ln)
            files[m["id"]] = os.path.join(os.path.abspath(yolo_data), "images", "bench", m["file"])
    pages = []
    for pid in sorted(recs):
        r = recs[pid]
        path = files.get(pid) or os.path.join(data, r["image"])
        pages.append({"id": pid, "path": path, "rec": r, "size": Image.open(path).size,
                      "ocr_rec": {"lang": r["lang"], "text": r["text"], "words": r["words"]}})
    return pages[:limit] if limit else pages


def evaluate(pages, engines, text_systems, visual=None, configs=CONFIGS, cache="e2e_cache", workers=4, pad=PAD):
    """engines: {name: engine name or object}; text_systems: {name: predictor}; visual: detector or None."""
    ocr = {name: run_ocr(eng, pages, cache, workers) for name, eng in engines.items()}
    ocr_sec = {name: float(np.mean([o[p["id"]]["sec"] for p in pages])) for name, o in ocr.items()}
    configs = [(n, tuple(engines) if o == ("*",) else o, t, g, v) for n, o, t, g, v in configs]
    needed = {(e, c[2]) for c in configs for e in c[1] if e in engines and c[2] in text_systems}
    raw, txt_sec = {}, {}
    for eng, tname in sorted(needed):
        pred, per_page, secs = text_systems[tname], {}, []
        for p in pages:
            segs = ocr[eng][p["id"]]["segments"]
            if hasattr(pred, "set_page"):  # test helper: a perfect predictor needs the true page
                pred.set_page(p["rec"], segs)
            t0 = time.time()
            per_page[p["id"]], _ = text_regions(segs, pred, eng, gap_fill=False)
            secs.append(time.time() - t0)
        raw[(eng, tname)], txt_sec[(eng, tname)] = per_page, float(np.mean(secs))
        print(f"text PII {tname} on {eng}: {txt_sec[(eng, tname)]:.3f} s/page", flush=True)
    vis, vis_sec = {}, 0.0
    if visual is not None:
        t0 = time.time()
        dets = visual.detect([p["path"] for p in pages])
        vis_sec = (time.time() - t0) / max(1, len(pages))
        vis = {p["id"]: visual_regions(d) for p, d in zip(pages, dets)}
    truths = {p["id"]: truth(p["rec"]) for p in pages}
    rows = []
    for name, ocr_names, tname, gap, use_vis in configs:
        if any(e not in engines for e in ocr_names) or tname not in text_systems or (use_vis and visual is None):
            print(f"skipping config '{name}' (component not available)", flush=True)
            continue
        overall, text_acc, vis_acc = Acc(), Acc(), Acc()
        by_label, by_lang, clean = defaultdict(Acc), defaultdict(Acc), Acc()
        px_black = px_over = 0
        for p in pages:
            regions = []
            for e in ocr_names:
                r = raw[(e, tname)][p["id"]]
                regions += fill_gaps(r) if gap else r
            if use_vis:
                regions += vis[p["id"]]
            W, H = p["size"]
            res, blk, over = score_page(truths[p["id"]], regions, W, H, pad)
            px_black += blk
            px_over += over
            for kind, label, ok in res:
                overall.add(ok)
                (text_acc if kind == "text" else vis_acc).add(ok)
                by_label[label].add(ok)
                by_lang[p["rec"]["lang"]].add(ok)
            clean.add(all(ok for _, _, ok in res))
        sec = sum(ocr_sec[e] + txt_sec[(e, tname)] for e in ocr_names) + (vis_sec if use_vis else 0.0)
        rows.append({"config": name, "fully_redacted": overall.pct(), "text": text_acc.pct(), "visual": vis_acc.pct(),
                     "pages_clean": clean.pct(), "over_redaction": px_over / px_black if px_black else 0.0,
                     "sec_per_page": sec, "by_label": {k: v.pct() for k, v in sorted(by_label.items())},
                     "by_lang": {k: v.pct() for k, v in sorted(by_lang.items())},
                     "n_items": overall.n, "n_text": text_acc.n, "n_visual": vis_acc.n})
        print(f"{name:34s} fully_redacted={overall.pct():.3f}  pages_clean={clean.pct():.3f}", flush=True)
    return {"pages": len(pages), "pad": pad, "cover_word": COVER_WORD, "cover_visual": COVER_VISUAL,
            "ocr_errors": {n: sum(1 for p in pages if o[p["id"]].get("error")) for n, o in ocr.items()},
            "configs": rows}


def _pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def report_md(res):
    rows = res["configs"]
    L = [f"# Phase 5 — end-to-end redaction on {res['pages']} benchmark pages", "",
         f"Scored on the black boxes actually drawn. Text PII item = every word ≥{int(100 * res['cover_word'])}% "
         f"blacked out; visual item = ≥{int(100 * res['cover_visual'])}% of its box. Page clean = nothing leaks. "
         "Each row adds one component to the row above.", "",
         "| Configuration | All PII fully redacted ↑ | Text ↑ | Visual ↑ | Pages with no leak ↑ | Over-redaction ↓ | s/page |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['config']} | {_pct(r['fully_redacted'])} | {_pct(r['text'])} | {_pct(r['visual'])} | "
                 f"{_pct(r['pages_clean'])} | {_pct(r['over_redaction'])} | {r['sec_per_page']:.2f} |")
    langs = sorted({k for r in rows for k in r["by_lang"]})
    L += ["", "## All PII fully redacted, by language", "", "| Configuration | " + " | ".join(langs) + " |",
          "|---|" + "---|" * len(langs)]
    for r in rows:
        L.append(f"| {r['config']} | " + " | ".join(_pct(r["by_lang"].get(k)) for k in langs) + " |")
    labels = sorted({k for r in rows for k in r["by_label"]})
    L += ["", "## By PII type", "", "| Type | " + " | ".join(r["config"] for r in rows) + " |",
          "|---|" + "---|" * len(rows)]
    for lab in labels:
        L.append(f"| {lab} | " + " | ".join(_pct(r["by_label"].get(lab)) for r in rows) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--yolo_data", default=None, help="folder with the Phase 4 bench split (real faces)")
    ap.add_argument("--ocr", default="tesseract,easyocr")
    ap.add_argument("--gliner_ft", default=None)
    ap.add_argument("--gliner_base", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--yolo", default=None)
    ap.add_argument("--cache", default="outputs/e2e_cache")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from ..pii.predict import GLiNERPredictor, RulesPredictor, UnionPredictor
    from .models import gliner, yolo_weights
    data = find_data(a.data)
    pages = load_pages(data, a.yolo_data, a.limit)
    rules = RulesPredictor()
    text_systems = {"rules": rules}
    for key, path in (("base", a.gliner_base), ("ft", a.gliner_ft)):
        if path:
            text_systems[f"{key}+rules"] = UnionPredictor(GLiNERPredictor(gliner(path), a.threshold, name=key), rules)
    visual = None
    if a.yolo:
        from ..vision.detectors import YoloDetector
        visual = YoloDetector(yolo_weights(a.yolo), conf=0.05)
    engines = {n: n for n in a.ocr.split(",") if n}
    res = evaluate(pages, engines, text_systems, visual, CONFIGS, a.cache, a.workers)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "report_e2e.json"), "w"), indent=2)
    md = report_md(res)
    open(os.path.join(a.out, "report_e2e.md"), "w", encoding="utf-8").write(md)
    print(md, flush=True)
    return res


if __name__ == "__main__":
    main()
