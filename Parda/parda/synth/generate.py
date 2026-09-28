"""Generate synthetic Indian documents with PII labels.

Usage:
  python -m parda.synth.generate --n 2000 --out data/synth --workers 4
Output:
  data/synth/images/000000.jpg ...
  data/synth/annotations.jsonl   (one JSON record per image)
  data/synth/stats.json
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool

from .augment import augment
from .canvas import RAQM, available_scripts
from .templates import TEMPLATES

_CFG = {}


def _init(cfg):
    _CFG.update(cfg)


def make_one(i: int):
    rng = random.Random(_CFG["seed"] * 1_000_003 + i)
    doc_type = rng.choices(list(_CFG["doc_weights"]), weights=list(_CFG["doc_weights"].values()))[0]
    lang = rng.choices(list(_CFG["lang_weights"]), weights=list(_CFG["lang_weights"].values()))[0]
    try:
        doc = TEMPLATES[doc_type](rng, lang, _CFG.get("faces_dir"))
        rec = doc.finalize()
        img, ops = doc.img, []
        if rng.random() < _CFG["aug_prob"]:
            img, rec, ops = augment(img, rec, rng, _CFG["aug_strength"])
        name = f"{i:06d}.jpg"
        img.save(os.path.join(_CFG["out"], "images", name), "JPEG", quality=90)
        rec.update({"id": i, "image": f"images/{name}", "width": img.width, "height": img.height,
                    "doc_type": doc.doc_type, "lang": lang, "aug": ops})
        return rec
    except Exception as e:  # never kill a long run because of one bad sample
        return {"id": i, "error": f"{type(e).__name__}: {e}", "doc_type": doc_type, "lang": lang}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0, help="start index (to add more data later without overlap)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--aug_prob", type=float, default=0.85)
    ap.add_argument("--aug_strength", type=float, default=1.0)
    ap.add_argument("--faces_dir", default=None, help="optional folder of real face crops (e.g. WIDER FACE)")
    ap.add_argument("--langs", default="en:0.4,hi-en:0.3,kn-en:0.3")
    ap.add_argument("--docs", default="loan_application:1,payslip:1,id_card:1,rent_agreement:1,discharge_summary:1")
    a = ap.parse_args(argv)

    scripts = available_scripts()
    lang_w = {k: float(v) for k, v in (x.split(":") for x in a.langs.split(","))}
    if not scripts["deva"]:
        print("WARNING: no Devanagari font found -> dropping hi-en", file=sys.stderr)
        lang_w.pop("hi-en", None)
    if not scripts["knda"]:
        print("WARNING: no Kannada font found -> dropping kn-en", file=sys.stderr)
        lang_w.pop("kn-en", None)
    if not RAQM and (scripts["deva"] or scripts["knda"]):
        print("WARNING: Pillow has no RAQM -> Hindi/Kannada letters will be shaped WRONG. "
              "Run scripts/setup_kaggle.sh", file=sys.stderr)
    doc_w = {k: float(v) for k, v in (x.split(":") for x in a.docs.split(","))}

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    cfg = dict(seed=a.seed, out=a.out, aug_prob=a.aug_prob, aug_strength=a.aug_strength,
               faces_dir=a.faces_dir, lang_weights=lang_w, doc_weights=doc_w)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    t0, n_ok, errors = time.time(), 0, []
    c_doc, c_lang, c_ent, c_vis = Counter(), Counter(), Counter(), Counter()
    ann_path = os.path.join(a.out, "annotations.jsonl")
    with open(ann_path, "a", encoding="utf-8") as fout, Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        idx = range(a.start, a.start + a.n)
        for rec in tqdm(pool.imap_unordered(make_one, idx, chunksize=8), total=a.n):
            if "error" in rec:
                errors.append(rec)
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            c_doc[rec["doc_type"]] += 1
            c_lang[rec["lang"]] += 1
            c_ent.update(e["label"] for e in rec["entities"])
            c_vis.update(v["label"] for v in rec["visuals"])

    stats = {"generated": n_ok, "errors": len(errors), "seconds": round(time.time() - t0, 1),
             "doc_types": c_doc, "langs": c_lang, "entities": c_ent, "visuals": c_vis,
             "raqm": RAQM, "fonts": scripts, "first_errors": errors[:5]}
    with open(os.path.join(a.out, f"stats_{a.start}_{a.start + a.n}.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
