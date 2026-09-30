"""Phone-photo versions of Phase 1 TRAINING pages, to retrain the text model on photo damage.

  python -m parda.synth.photo_pages --data <Phase 1 synth> --out data/photo --per_lang 2000 --workers 4

Takes pages of the 5 training templates (never the 300 benchmark pages), applies the same phone-photo damage
as the real-world test (desk, perspective, shadow, blur, JPEG; see ood.phone_photo) and writes a dataset
(images/ + annotations.jsonl) whose boxes follow the damage exactly. Pages keep their Phase 1 document id,
so the train/val split by document stays the same. The 4 real-world-test layouts are NOT used: that test
stays unseen.
"""
import argparse
import json
import os
import random
import time
from collections import Counter
from multiprocessing import Pool

from PIL import Image

from ..ocr.run_ocr import BENCH_PER_LANG, BENCH_SEED, find_data, pick_ids
from ..pii.build_data import _index_annotations
from .ood import phone_photo

_CFG = {}


def read_meta(ann_path):
    """[(id, lang, doc_type)] in file order, reading only the tail of each line (the full file is ~450 MB)."""
    meta = []
    with open(ann_path, "rb") as f:
        for line in f:
            j = line.rfind(b'"id": ')
            if j < 0:
                continue
            tail = json.loads(b"{" + line[j:])  # the top-level keys are written last: id, image, ..., lang, aug
            meta.append((tail["id"], tail["lang"], tail["doc_type"]))
    return meta


def _init(cfg):
    _CFG.update(cfg)
    _CFG["idx"] = _index_annotations(os.path.join(cfg["data"], "annotations.jsonl"))


def _one(doc_id):
    try:
        with open(os.path.join(_CFG["data"], "annotations.jsonl"), "rb") as f:
            f.seek(_CFG["idx"][doc_id])
            rec = json.loads(f.readline())
        img = Image.open(os.path.join(_CFG["data"], rec["image"])).convert("RGB")
        out, rec, ops = phone_photo(img, rec, random.Random(doc_id * 7919 + 3))
        name = f"{doc_id:06d}.jpg"
        out.save(os.path.join(_CFG["out"], "images", name), quality=90)
        rec.update({"image": f"images/{name}", "width": out.width, "height": out.height,
                    "aug": list(rec.get("aug", [])) + ops, "photo": True})
        return rec
    except Exception as e:  # never kill a long run for one page
        return {"id": doc_id, "error": f"{type(e).__name__}: {e}"}


def build(data, out, per_lang=2000, workers=4, seed=21):
    t0 = time.time()
    meta = read_meta(os.path.join(data, "annotations.jsonl"))
    bench = pick_ids(meta, BENCH_PER_LANG, BENCH_SEED)
    ids = sorted(pick_ids(meta, per_lang, seed, exclude=bench))
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    cfg = {"data": data, "out": out}
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(cfg,)) as p:
            recs = list(p.imap_unordered(_one, ids, chunksize=16))
    else:
        _init(cfg)
        recs = [_one(i) for i in ids]
    ok = sorted((r for r in recs if "error" not in r), key=lambda r: r["id"])
    errors = [r for r in recs if "error" in r]
    with open(os.path.join(out, "annotations.jsonl"), "w", encoding="utf-8") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"n_ok": len(ok), "n_errors": len(errors), "errors": errors[:5], "benchmark_pages_used": len(set(ids) & bench),
             "langs": dict(Counter(r["lang"] for r in ok)), "doc_types": dict(Counter(r["doc_type"] for r in ok)),
             "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=2)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_lang", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args(argv)
    s = build(find_data(a.data), a.out, a.per_lang, a.workers, a.seed)
    print(json.dumps(s, indent=2), flush=True)
    return s


if __name__ == "__main__":
    main()
