"""Run an OCR engine on a fixed, stratified sample of the synthetic dataset.

  python -m parda.ocr.run_ocr --engine tesseract --per_lang 100 --out outputs/ocr_bench
  python -m parda.ocr.run_ocr --engine easyocr   --per_lang 100 --out outputs/ocr_bench

- Every engine uses the SAME pages (sample_ids.json) so the comparison is fair.
- Resumable: if the session dies, run again and it skips pages already done.
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

from ..fsutil import bounded_walk
from .engines import make_engine


INPUT_ROOT = os.environ.get("PARDA_INPUT_ROOT", "/kaggle/input")
EXTRACT_TO = os.environ.get("PARDA_EXTRACT_TO", "/tmp/parda_synth_v1")


def _n_images(d):
    p = os.path.join(d, "images")
    return len(os.listdir(p)) if os.path.isdir(p) else 0


def _find_files(root, want):
    """Folders under root holding a file for which want(name) is true. Never lists huge folders fully
    (a 200k-photo dataset attached next to ours used to make this take 10+ minutes)."""
    hits = []
    for d, files, subdirs, _ in bounded_walk(root):
        found = [f for f in files if want(f)]
        hits += [os.path.join(d, f) for f in found]
        if "annotations.jsonl" in files:
            subdirs[:] = []  # a Parda dataset: nothing more to find inside it
    return hits


def _candidates():
    found = _find_files(INPUT_ROOT, lambda f: f == "annotations.jsonl")
    found += _find_files(EXTRACT_TO, lambda f: f == "annotations.jsonl")
    dirs = [os.path.dirname(p) for p in found]
    dirs += [d for d in ("data/synth", "/kaggle/working/parda_synth_v1")
             if os.path.isfile(os.path.join(d, "annotations.jsonl"))]
    return list(dict.fromkeys(dirs))


def find_data(data=None, verbose=True):
    """Locate the dataset folder (the one holding annotations.jsonl + images/).

    Handles: extracted Kaggle dataset, a dataset that is still a .zip (extracted to /tmp once),
    and duplicate/partial copies (picks the folder with the most images)."""
    if data:
        if not os.path.isfile(os.path.join(data, "annotations.jsonl")):
            raise FileNotFoundError(f"no annotations.jsonl in {data}")
        return data
    dirs = _candidates()
    if not dirs:  # maybe the dataset is still a zip
        zips = sorted(_find_files(INPUT_ROOT, lambda f: f.lower().endswith(".zip")), key=os.path.getsize)
        if zips:
            import zipfile
            z = zips[-1]  # largest zip
            if verbose:
                print(f"dataset is zipped -> extracting {z} to {EXTRACT_TO} (one-time, a few minutes)", flush=True)
            os.makedirs(EXTRACT_TO, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(EXTRACT_TO)
            dirs = _candidates()
    if not dirs:
        raise FileNotFoundError("annotations.jsonl not found. Attach the parda-synth-v1 dataset or pass --data")
    best = max(dirs, key=_n_images)
    if verbose and len(dirs) > 1:
        print("found several dataset copies, using the one with most images:",
              {d: _n_images(d) for d in dirs}, flush=True)
    return best


def _read_meta(data):
    meta = []
    with open(os.path.join(data, "annotations.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta.append((r["id"], r["lang"], r["doc_type"]))
    return meta


def pick_ids(meta, per_lang, seed, exclude=frozenset()):
    """per_lang pages per language, round-robin over doc types. Deterministic for a given seed."""
    rng = random.Random(seed)
    ids = set()
    for lang in sorted({m[1] for m in meta}):
        pool = [m for m in meta if m[1] == lang]
        rng.shuffle(pool)
        by_doc = {}
        for m in pool:
            if m[0] not in exclude:
                by_doc.setdefault(m[2], []).append(m[0])
        picked, docs = [], sorted(by_doc)
        while len(picked) < per_lang and any(by_doc.values()):
            for d in docs:
                if by_doc[d] and len(picked) < per_lang:
                    picked.append(by_doc[d].pop())
        ids.update(picked)
    return ids


# the Phase 2 benchmark sample (keep fixed forever; env override exists only for tests)
BENCH_PER_LANG, BENCH_SEED = int(os.environ.get("PARDA_BENCH_PER_LANG", 100)), 7


def benchmark_ids(data):
    return pick_ids(_read_meta(data), BENCH_PER_LANG, BENCH_SEED)


def load_sample(data, per_lang, seed, out, exclude=frozenset()):
    """Pick per_lang pages per language (spread over doc types). Saved so all engines share it."""
    path = os.path.join(out, "sample_ids.json")
    if os.path.exists(path):
        ids = set(json.load(open(path)))
    else:
        ids = pick_ids(_read_meta(data), per_lang, seed, exclude)
        os.makedirs(out, exist_ok=True)
        json.dump(sorted(ids), open(path, "w"))
    recs = []
    with open(os.path.join(data, "annotations.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in ids:
                recs.append(r)
    recs.sort(key=lambda r: r["id"])
    return recs


_ENG = None


def _init(name, kw):
    global _ENG
    _ENG = make_engine(name, **kw)


def _run_one(args):
    data, rec = args
    t0 = time.time()
    try:
        segs = _ENG.run(os.path.join(data, rec["image"]), rec)
        err = None
    except Exception as e:
        segs, err = [], f"{type(e).__name__}: {e}"
    return {"id": rec["id"], "engine": _ENG.name, "lang": rec["lang"], "doc_type": rec["doc_type"],
            "seconds": round(time.time() - t0, 3), "segments": segs, "error": err}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["tesseract", "easyocr", "paddleocr", "oracle"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs/ocr_bench")
    ap.add_argument("--per_lang", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (CPU engines only)")
    ap.add_argument("--noise", type=float, default=0.0, help="oracle only")
    ap.add_argument("--exclude_benchmark", action="store_true",
                    help="never pick the 300 Phase 2 benchmark pages (use for TRAINING data)")
    a = ap.parse_args(argv)

    data = find_data(a.data)
    os.makedirs(a.out, exist_ok=True)
    exclude = benchmark_ids(data) if a.exclude_benchmark else frozenset()
    recs = load_sample(data, a.per_lang, a.seed, a.out, exclude)
    if exclude:
        assert not ({r["id"] for r in recs} & exclude), "benchmark pages leaked into training sample"
    out_path = os.path.join(a.out, f"ocr_{a.engine}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"data={data}  sample={len(recs)}  already done={len(done)}  to run={len(todo)}", flush=True)
    if not todo:
        return

    kw = {"noise": a.noise} if a.engine == "oracle" else {}
    workers = a.workers if a.engine in ("tesseract", "oracle") else 1  # GPU engines: one process
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    n_err = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        if workers > 1:
            with Pool(workers, initializer=_init, initargs=(a.engine, kw)) as pool:
                it = pool.imap_unordered(_run_one, [(data, r) for r in todo], chunksize=2)
                for res in tqdm(it, total=len(todo)):
                    n_err += res["error"] is not None
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fout.flush()
        else:
            _init(a.engine, kw)
            for r in tqdm(todo):
                res = _run_one((data, r))
                n_err += res["error"] is not None
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()  # written immediately -> safe if the session dies
    print(f"finished {a.engine}: {len(todo)} pages, {n_err} errors -> {out_path}")
    if n_err:
        print("first error:", next(json.loads(l)["error"] for l in open(out_path) if json.loads(l)["error"]),
              file=sys.stderr)


if __name__ == "__main__":
    main()
