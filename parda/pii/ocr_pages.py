"""OCR a chosen set of pages with one engine -> <out_dir>/ocr_<engine>.jsonl (resumable), for build_data.

  python -m parda.pii.ocr_pages --data data/photo --engine tesseract --per_lang 0 --bench_of <synth> --out_dir data/photo
  python -m parda.pii.ocr_pages --data <synth> --engine easyocr --per_lang 400 --seed 31 --out_dir outputs/easy_scan

--per_lang 0 = every page. The Phase 1 benchmark pages (of --bench_of, default --data) are never selected.
Output lines: {"id", "segments", "sec"} (+ "error"), the same format as Phase 3a.
"""
import argparse
import json
import os

from ..ocr.run_ocr import BENCH_PER_LANG, BENCH_SEED, find_data, pick_ids
from ..pipeline.evaluate_e2e import run_ocr
from ..synth.photo_pages import read_meta


def select(data, per_lang, seed, bench_of=None):
    meta = read_meta(os.path.join(data, "annotations.jsonl"))
    bmeta = meta if not bench_of or os.path.abspath(bench_of) == os.path.abspath(data) else \
        read_meta(os.path.join(bench_of, "annotations.jsonl"))
    bench = pick_ids(bmeta, BENCH_PER_LANG, BENCH_SEED)
    if per_lang:
        return sorted(pick_ids(meta, per_lang, seed, exclude=bench))
    return sorted(m[0] for m in meta if m[0] not in bench)


def pages_for(data, ids):
    want, pages = set(ids), []
    with open(os.path.join(data, "annotations.jsonl"), encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            if r["id"] in want:
                pages.append({"id": r["id"], "path": os.path.join(data, r["image"]),
                              "ocr_rec": {"lang": r["lang"], "text": r["text"], "words": r["words"]}})
    return pages


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--engine", required=True, help="tesseract | easyocr (| oracle for tests)")
    ap.add_argument("--per_lang", type=int, default=0)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--bench_of", default=None, help="dataset whose benchmark pages to exclude (default: --data)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    data = find_data(a.data)
    pages = pages_for(data, select(data, a.per_lang, a.seed, a.bench_of))
    done = run_ocr(a.engine, pages, a.out_dir, a.workers)
    errors = sum(1 for p in pages if done[p["id"]].get("error"))
    out = os.path.join(a.out_dir, f"ocr_{a.engine}.jsonl")
    print(json.dumps({"pages": len(pages), "errors": errors, "out": out}), flush=True)
    return out


if __name__ == "__main__":
    main()
