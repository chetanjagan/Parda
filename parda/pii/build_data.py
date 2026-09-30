"""Build GLiNER training data.

  python -m parda.pii.build_data --ocr /kaggle/input/.../ocr_train/ocr_tesseract.jsonl --clean_docs 10000 --out data/gliner
  # v2: several OCR sources, each on its own pages (same document ids as the Phase 1 data)
  python -m parda.pii.build_data --noisy tess_scan=<synth>:<ocr.jsonl> --noisy tess_photo=<photo pages>:<ocr.jsonl> \
         --noisy easy_photo=<photo pages>:<ocr.jsonl>:lines --clean_docs 10000 --out data/gliner_v2

Two kinds of examples:
  clean  the perfect page text from Phase 1 (teaches what PII looks like)
  noisy  an OCR reading of the page with labels transferred onto it
         (teaches PII with real OCR mistakes: 'SBINO364507' is still an IFSC)
         word-level OCR (Tesseract): by position; line-level OCR (EasyOCR, ':lines'): by text alignment
The 300 benchmark pages are never used. Train/val split is by document id, the same for every source, so a
validation document is never trained on in any form.
"""
import argparse
import json
import os
import random
import zlib
from collections import Counter
from multiprocessing import Pool

from ..ocr.run_ocr import benchmark_ids, find_data
from .labels import LABELS
from .spans import char_spans_to_token_ner, chunk_examples, tokenize, transfer_labels, transfer_labels_lines

_CFG = {}


def _init(cfg):
    _CFG.update(cfg)


def _is_val(doc_id, frac):
    return (zlib.crc32(str(doc_id).encode()) % 1000) < frac * 1000


def clean_examples(rec, window, stride):
    spans = [{"label": LABELS[e["label"]], "start": e["start"], "end": e["end"]} for e in rec["entities"]]
    toks = tokenize(rec["text"])
    return chunk_examples(toks, char_spans_to_token_ner(toks, spans), window, stride)


def noisy_examples(rec, segs, window, stride, max_cer, lines=False):
    text, gold, missed = (transfer_labels_lines if lines else transfer_labels)(rec, segs)
    toks = tokenize(text)
    if not toks:  # OCR read nothing on this page: no example (an empty example would only add noise)
        return [], {"gold": 0, "missed": len(missed), "bad": 0, "cer_sum": 0.0, "empty": 1}
    spans = [{"label": LABELS[g["label"]], "start": g["start"], "end": g["end"]} for g in gold]
    exs = chunk_examples(toks, char_spans_to_token_ner(toks, spans), window, stride)
    # drop windows containing an unreadable entity (label would teach 'garbage = PII')
    bad = [g for g in gold if g["cer"] > max_cer]
    if bad:
        bad_tok = char_spans_to_token_ner(toks, [{"label": "x", "start": g["start"], "end": g["end"]} for g in bad])
        keep = []
        for (s, e), ex in zip(_windows_of(toks, window, stride), exs):
            if not any(s <= a <= e - 1 or s <= b <= e - 1 for a, b, _ in bad_tok):
                keep.append(ex)
        exs = keep
    stats = {"gold": len(gold), "missed": len(missed), "bad": len(bad),
             "cer_sum": sum(g["cer"] for g in gold)}
    return exs, stats


def _windows_of(toks, window, stride):
    from .spans import windows
    return windows(len(toks), window, stride)


def _noisy_worker(args):
    rec_line, ocr_line = args
    rec, ocr = json.loads(rec_line), json.loads(ocr_line)
    if ocr.get("error"):
        return rec["id"], [], None
    exs, st = noisy_examples(rec, ocr["segments"], _CFG["window"], _CFG["stride"], _CFG["max_cer"], _CFG["lines"])
    return rec["id"], exs, st


def _index_annotations(path):
    """doc id -> byte offset of its line (so we can fetch single records without loading 450 MB)."""
    idx = {}
    with open(path, "rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            j = line.rfind(b'"id": ')  # top-level id is written last
            if j < 0:
                continue
            k = j + 6
            num = b""
            while line[k:k + 1].isdigit():
                num += line[k:k + 1]
                k += 1
            idx[int(num)] = off
    return idx


def find_ocr_train(root=None):
    from ..fsutil import find_path  # never lists a huge neighbour folder (e.g. 200k face photos) in full
    from ..ocr import run_ocr
    hits = find_path(root or run_ocr.INPUT_ROOT, "ocr_train/ocr_tesseract.jsonl")
    return hits[0] if hits else None


def parse_noisy(spec, default_data):
    """'name=DATA_DIR:OCR_FILE[:lines]' or 'name=OCR_FILE[:lines]' (pages of --data)."""
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    lines = parts[-1] == "lines"
    if lines:
        parts = parts[:-1]
    data, ocr = (parts[0], parts[1]) if len(parts) == 2 else (default_data, parts[0])
    return name, data, ocr, lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--ocr", default=None, help="Tesseract output for training pages (Phase 3a)")
    ap.add_argument("--noisy", action="append", default=[],
                    help="name=DATA_DIR:OCR_FILE[:lines] (repeatable); ':lines' for line-level OCR like EasyOCR")
    ap.add_argument("--clean_docs", type=int, default=10000)
    ap.add_argument("--out", default="data/gliner")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=150)
    ap.add_argument("--max_cer", type=float, default=0.6)
    ap.add_argument("--val_frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    data = find_data(a.data)
    sources = [parse_noisy(sp, data) for sp in a.noisy]
    if not sources:  # Phase 3 behaviour: one Tesseract file on the Phase 1 pages
        ocr_path = a.ocr or find_ocr_train()
        if ocr_path and os.path.exists(ocr_path):
            sources = [("noisy", data, ocr_path, False)]
        else:
            print("WARNING: no OCR training file found -> building clean examples only")
    ann = os.path.join(data, "annotations.jsonl")
    bench = benchmark_ids(data)
    idx = _index_annotations(ann)
    os.makedirs(a.out, exist_ok=True)
    train, val = [], []
    stats = Counter()
    label_counts = Counter()

    def add(doc_id, exs, kind):
        tgt = val if _is_val(doc_id, a.val_frac) else train
        for ex in exs:
            ex["source"] = kind
            tgt.append(ex)
            for _, _, lab in ex["ner"]:
                label_counts[(kind, lab)] += 1
        stats[f"{kind}_docs"] += 1

    # ---- noisy (OCR) examples, one pass per source
    for kind, sdata, ocr_path, lines in sources:
        s_ann = os.path.join(sdata, "annotations.jsonl")
        s_idx = idx if os.path.abspath(sdata) == os.path.abspath(data) else _index_annotations(s_ann)
        jobs = []
        with open(s_ann, "rb") as fa, open(ocr_path, encoding="utf-8") as fo:
            for line in fo:
                if not line.strip():
                    continue
                doc_id = json.loads(line)["id"]
                if doc_id in bench or doc_id not in s_idx:  # the Phase 1 benchmark ids, for every source
                    stats[f"{kind}_skipped"] += 1
                    continue
                fa.seek(s_idx[doc_id])
                jobs.append((fa.readline().decode("utf-8"), line))
        cfg = dict(window=a.window, stride=a.stride, max_cer=a.max_cer, lines=lines)
        with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
            for doc_id, exs, st in pool.imap_unordered(_noisy_worker, jobs, chunksize=16):
                if st is None:
                    stats[f"{kind}_error_pages"] += 1
                    continue
                add(doc_id, exs, kind)
                for k, v in st.items():
                    stats[f"noisy_{k}"] += v
                    stats[f"{kind}_{k}"] += v

    # ---- clean examples (random non-benchmark docs)
    rng = random.Random(a.seed)
    pool_ids = sorted(i for i in idx if i not in bench)
    rng.shuffle(pool_ids)
    with open(ann, "rb") as fa:
        for doc_id in pool_ids[:a.clean_docs]:
            fa.seek(idx[doc_id])
            rec = json.loads(fa.readline())
            add(doc_id, clean_examples(rec, a.window, a.stride), "clean")

    rng.shuffle(train)
    json.dump(train, open(os.path.join(a.out, "train.json"), "w"), ensure_ascii=False)
    json.dump(val, open(os.path.join(a.out, "val.json"), "w"), ensure_ascii=False)
    n_gold = stats.get("noisy_gold", 0)
    summary = {
        "train_examples": len(train), "val_examples": len(val),
        "train_by_source": Counter(e["source"] for e in train),
        "docs": {k: v for k, v in stats.items() if k.endswith("_docs")},
        "ocr_sources": [{"name": k, "data": d, "ocr": o, "mode": "lines" if ln else "words"} for k, d, o, ln in sources],
        "transfer_by_source": {k: {"transferred": stats.get(f"{k}_gold", 0), "missed": stats.get(f"{k}_missed", 0),
                                   "unreadable_dropped": stats.get(f"{k}_bad", 0)} for k, _, _, _ in sources},
        "noisy_label_transfer": {
            "entities_transferred": n_gold,
            "entities_ocr_missed": stats.get("noisy_missed", 0),
            "entities_unreadable_dropped": stats.get("noisy_bad", 0),
            "mean_transferred_cer": round(stats.get("noisy_cer_sum", 0) / max(1, n_gold), 4),
        },
        "labels": {f"{k[0]}:{k[1]}": v for k, v in sorted(label_counts.items())},
        "benchmark_pages_excluded": len(bench),
    }
    json.dump(summary, open(os.path.join(a.out, "stats.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
