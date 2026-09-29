"""Build a YOLO dataset (images/ + labels/ + data.yaml) from the Phase 1 synthetic documents.

  python -m parda.vision.yolo_data --data <synth dir> --out data/yolo --n_train 12000 --n_val 600 \
         --faces_dir <folder of real face photos> --workers 4 --preview 6

Splits (never overlap):
  bench  the fixed Phase 2 benchmark pages (100 per language). Only used for the final comparison.
  val    n_val pages spread over languages and doc types. Used by YOLO during training.
  train  every remaining page with a QR code (the rarest class) first, then random pages up to n_train.

Images without a FACE box are symlinked (no copy, no disk use). Images with a FACE box are re-written with
real face photos pasted over the cartoon avatars (see faces.py). meta.jsonl in each split records the
language, doc type and which face photos were used.
"""
import argparse
import json
import os
import random
import shutil
from collections import Counter
from multiprocessing import Pool

from PIL import Image, ImageDraw

from ..ocr.run_ocr import BENCH_PER_LANG, BENCH_SEED, find_data, pick_ids
from .classes import CLASS_ID, CLASSES
from .faces import FacePool, paste_face

SPLITS = ("train", "val", "bench")


def read_records(data):
    """Only the fields we need, in file order (pick_ids depends on the order)."""
    recs = []
    with open(os.path.join(data, "annotations.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs.append({k: r[k] for k in ("id", "image", "width", "height", "lang", "doc_type", "visuals")})
    return recs


def yolo_lines(rec):
    """'cls cx cy w h' (normalised 0-1) for every visual box; tiny or unknown boxes are skipped."""
    W, H = rec["width"], rec["height"]
    out = []
    for v in rec["visuals"]:
        c = CLASS_ID.get(v["label"])
        if c is None:
            continue
        x0, y0 = max(0.0, float(v["bbox"][0])), max(0.0, float(v["bbox"][1]))
        x1, y1 = min(float(W), float(v["bbox"][2])), min(float(H), float(v["bbox"][3]))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        out.append(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return out


def choose_splits(recs, n_train, n_val, seed=0):
    meta = [(r["id"], r["lang"], r["doc_type"]) for r in recs]
    bench = pick_ids(meta, BENCH_PER_LANG, BENCH_SEED)
    n_langs = max(1, len({m[1] for m in meta}))
    val = pick_ids(meta, max(1, n_val // n_langs), seed + 11, exclude=bench)
    rest = [r for r in recs if r["id"] not in bench and r["id"] not in val]
    random.Random(seed).shuffle(rest)
    has_qr = [r["id"] for r in rest if any(v["label"] == "QR_CODE" for v in r["visuals"])]
    qr_set = set(has_qr)
    others = [r["id"] for r in rest if r["id"] not in qr_set]
    train = has_qr + others
    if n_train:
        train = train[:n_train]
    return {"train": set(train), "val": val, "bench": bench}


# ---- workers ----
_W = {}


def _init(cfg):
    _W.update(cfg)
    _W["pool"] = FacePool(files=cfg["face_files"]) if cfg.get("face_files") else None


def _link_or_copy(src, dst, copy):
    if not copy:
        try:
            os.symlink(os.path.abspath(src), dst)
            return "linked"
        except OSError:
            pass
    shutil.copyfile(src, dst)
    return "copied"


def _write_one(job):
    split, rec = job
    src = os.path.join(_W["data"], rec["image"])
    ext = os.path.splitext(rec["image"])[1] or ".jpg"
    stem = f"{rec['id']:06d}"
    dst = os.path.join(_W["out"], "images", split, stem + ext)
    lines = yolo_lines(rec)
    with open(os.path.join(_W["out"], "labels", split, stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    faces = [v for v in rec["visuals"] if v["label"] == "FACE"]
    used = []
    if _W["pool"] is not None and faces:
        rng = random.Random(rec["id"] * 1000003 + 17)
        img = Image.open(src).convert("RGB")
        for v in faces:
            fp = _W["pool"].pick(rng, heldout=(split != "train"))
            if paste_face(img, v["bbox"], fp, rng):
                used.append(os.path.basename(fp))
        img.save(dst, quality=92)
        how = "face_swapped"
    else:
        how = _link_or_copy(src, dst, _W["copy"])
    meta = {"id": rec["id"], "file": stem + ext, "lang": rec["lang"], "doc_type": rec["doc_type"],
            "n_boxes": len(lines), "faces_used": used}
    return split, how, meta, [int(ln.split()[0]) for ln in lines]


def _prepare_out(out):
    if os.path.exists(out):
        if os.path.isfile(os.path.join(out, "data.yaml")):
            shutil.rmtree(out)  # an older build of this dataset
        elif os.listdir(out):
            raise FileExistsError(f"{out} exists and is not a Parda YOLO dataset; choose another --out")
    for s in SPLITS:
        os.makedirs(os.path.join(out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", s), exist_ok=True)


def write_yaml(out):
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    txt = (f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\ntest: images/bench\n"
           f"nc: {len(CLASSES)}\nnames:\n{names}\n")
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(txt)


def preview(out, split="train", k=6, only_faces=True):
    """Draw the label boxes on k images -> <out>/previews/*.jpg (to eyeball the face swap + labels)."""
    colours = {"FACE": (220, 30, 30), "SIGNATURE": (30, 120, 220), "QR_CODE": (30, 160, 60), "STAMP": (200, 120, 0)}
    metas = [json.loads(ln) for ln in open(os.path.join(out, split, "meta.jsonl"), encoding="utf-8")]
    if only_faces:
        metas = [m for m in metas if m["faces_used"]] or metas
    os.makedirs(os.path.join(out, "previews"), exist_ok=True)
    paths = []
    for m in metas[:k]:
        img = Image.open(os.path.join(out, "images", split, m["file"])).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img)
        for ln in open(os.path.join(out, "labels", split, os.path.splitext(m["file"])[0] + ".txt")):
            c, cx, cy, w, h = ln.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            name = CLASSES[int(c)]
            d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=colours[name], width=3)
            d.text((cx - w / 2 + 3, cy - h / 2 + 3), name, fill=colours[name])
        p = os.path.join(out, "previews", f"{split}_{m['file']}")
        img.save(p, quality=85)
        paths.append(p)
    return paths


def build(data, out, n_train=12000, n_val=600, faces_dir=None, workers=4, copy=False, seed=0, only=SPLITS):
    """only: write just these splits, e.g. only=("bench",) for the Phase 5 evaluation (split choice unchanged)."""
    recs = read_records(data)
    splits = choose_splits(recs, n_train, n_val, seed)
    _prepare_out(out)
    by_id = {r["id"]: r for r in recs}
    jobs = [(s, by_id[i]) for s in SPLITS if s in only for i in sorted(splits[s])]
    pool = FacePool(faces_dir) if faces_dir else None  # listed once here, not once per worker
    cfg = {"data": data, "out": out, "face_files": pool.files if pool else None, "copy": copy}
    stats = {s: {"images": 0, "boxes": Counter(), "how": Counter()} for s in SPLITS}
    metas = {s: [] for s in SPLITS}
    faces_by_split = {s: set() for s in SPLITS}
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(cfg,)) as p:
            results = list(p.imap_unordered(_write_one, jobs, chunksize=32))
    else:
        _init(cfg)
        results = [_write_one(j) for j in jobs]
    for split, how, meta, classes in results:
        st = stats[split]
        st["images"] += 1
        st["how"][how] += 1
        st["boxes"].update(CLASSES[c] for c in classes)
        metas[split].append(meta)
        faces_by_split[split].update(meta["faces_used"])
    for s in SPLITS:
        os.makedirs(os.path.join(out, s), exist_ok=True)
        with open(os.path.join(out, s, "meta.jsonl"), "w", encoding="utf-8") as f:
            for m in sorted(metas[s], key=lambda m: m["id"]):
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    write_yaml(out)
    summary = {
        "data": data, "faces_dir": faces_dir,
        "face_pools": pool.sizes() if pool else None,
        "face_photos_shared_train_vs_eval": len(faces_by_split["train"] & (faces_by_split["val"] | faces_by_split["bench"])),
        "splits": {s: {"images": st["images"], "boxes": dict(st["boxes"]), "how": dict(st["how"])}
                   for s, st in stats.items()},
    }
    if not faces_dir:
        summary["warning"] = "no faces_dir: FACE boxes still show the Phase 1 cartoon avatars"
    json.dump(summary, open(os.path.join(out, "stats.json"), "w"), indent=2)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="synth dataset folder (default: auto-detect)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=12000, help="0 = every non-benchmark, non-val page")
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--faces_dir", default=None, help="folder of real face photos (e.g. CelebA)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0, help="draw boxes on N train images -> <out>/previews")
    ap.add_argument("--only", default=",".join(SPLITS), help="comma list of splits to write, e.g. bench")
    a = ap.parse_args(argv)
    data = find_data(a.data)
    only = tuple(x for x in a.only.split(",") if x)
    summary = build(data, a.out, a.n_train or None, a.n_val, a.faces_dir, a.workers, a.copy, a.seed, only)
    print(json.dumps(summary, indent=2), flush=True)
    if a.preview and "train" in only:
        for p in preview(a.out, "train", a.preview):
            print("preview:", p)
    return summary


if __name__ == "__main__":
    main()
