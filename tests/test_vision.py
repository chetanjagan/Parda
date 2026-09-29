"""Run: python tests/test_vision.py   (CPU only; ultralytics is faked, so no GPU/weights needed)"""
import os

os.environ["PARDA_BENCH_PER_LANG"] = "2"  # tiny benchmark; must be set before parda.ocr.run_ocr is imported

import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image, ImageChops  # noqa: E402

from parda.ocr.run_ocr import benchmark_ids  # noqa: E402
from parda.vision.classes import CLASSES  # noqa: E402
from parda.vision.detectors import label_path  # noqa: E402
from parda.vision.evaluate_vis import ap50, evaluate, iou  # noqa: E402
from parda.vision.faces import FacePool, find_faces_dir  # noqa: E402
from parda.vision.yolo_data import build, read_records, yolo_lines  # noqa: E402

TMP = tempfile.mkdtemp(prefix="parda_vis_test_")
N_TRAIN, N_VAL = 20, 4


def _py(*args, env=None):
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, **(env or {})))
    assert r.returncode == 0, (args, r.stdout[-2000:], r.stderr[-3000:])
    return r


def _data():
    d = os.path.join(TMP, "synth")
    if not os.path.exists(d):  # normal augmentation on, so rotated boxes are covered too
        _py("parda.synth.generate", "--n", "40", "--out", d, "--workers", "2", "--langs", "en:1,hi-en:1")
    return d


def _faces():
    d = os.path.join(TMP, "input", "faces_ds", "img_align_celeba")
    if not os.path.exists(d):
        os.makedirs(d)
        rng = random.Random(1)
        for i in range(80):
            col = tuple(rng.randint(0, 255) for _ in range(3))
            Image.new("RGB", (89, 109), col).save(os.path.join(d, f"{i:06d}.jpg"))
    return d


def _yolo():
    out = os.path.join(TMP, "yolo")
    if not os.path.exists(os.path.join(out, "data.yaml")):
        _py("parda.vision.yolo_data", "--data", _data(), "--out", out, "--n_train", str(N_TRAIN),
            "--n_val", str(N_VAL), "--faces_dir", _faces(), "--workers", "2", "--preview", "2")
    return out


def _metas(split):
    return [json.loads(ln) for ln in open(os.path.join(_yolo(), split, "meta.jsonl"), encoding="utf-8")]


def test_find_faces_dir_skips_synth_data():
    root = os.path.join(TMP, "input")
    _faces()
    link = os.path.join(root, "parda_synth")  # a synth dataset inside the input root must be ignored
    if not os.path.exists(link):
        shutil.copytree(_data(), link)
    assert find_faces_dir(root, min_images=10) == _faces()
    assert find_faces_dir(os.path.join(TMP, "nope")) is None


def test_labels_valid_and_complete():
    recs = {r["id"]: r for r in read_records(_data())}
    n_files = 0
    for split in ("train", "val", "bench"):
        for m in _metas(split):
            lines = open(os.path.join(_yolo(), "labels", split, os.path.splitext(m["file"])[0] + ".txt")).read().split("\n")
            lines = [ln for ln in lines if ln]
            assert lines == yolo_lines(recs[m["id"]]), m
            for ln in lines:
                c, *xywh = ln.split()
                assert 0 <= int(c) < len(CLASSES)
                assert all(0.0 <= float(v) <= 1.0 for v in xywh), ln
            n_files += 1
    assert n_files > 0
    # every visual in the data is one of our classes and survives (boxes are never tiny)
    total = sum(len(r["visuals"]) for r in recs.values())
    assert total == sum(len(yolo_lines(r)) for r in recs.values())


def test_splits_disjoint_and_bench_fixed():
    ids = {s: {m["id"] for m in _metas(s)} for s in ("train", "val", "bench")}
    assert ids["bench"] == benchmark_ids(_data())
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["bench"]) and not (ids["val"] & ids["bench"])
    assert len(ids["train"]) == N_TRAIN and 1 <= len(ids["val"]) <= N_VAL
    stats = json.load(open(os.path.join(_yolo(), "stats.json")))
    assert stats["splits"]["train"]["images"] == N_TRAIN
    # QR pages (rarest class) go into train first
    recs = read_records(_data())
    qr = {r["id"] for r in recs if any(v["label"] == "QR_CODE" for v in r["visuals"])}
    avail = qr - ids["bench"] - ids["val"]
    if len(avail) <= N_TRAIN:
        assert avail <= ids["train"]


def test_face_swap_and_heldout_faces():
    pool = FacePool(_faces())
    assert pool.train and pool.heldout and not set(pool.train) & set(pool.heldout)
    recs = {r["id"]: r for r in read_records(_data())}
    swapped = 0
    used = {s: set() for s in ("train", "val", "bench")}
    for split in used:
        for m in _metas(split):
            used[split].update(m["faces_used"])
            dst = os.path.join(_yolo(), "images", split, m["file"])
            faces = [v for v in recs[m["id"]]["visuals"] if v["label"] == "FACE"]
            if faces:
                assert not os.path.islink(dst) and m["faces_used"], m
                src = Image.open(os.path.join(_data(), recs[m["id"]]["image"])).convert("RGB")
                new = Image.open(dst).convert("RGB")
                b = [int(v) for v in faces[0]["bbox"]]
                crop = (b[0] + 4, b[1] + 4, b[2] - 4, b[3] - 4)
                assert ImageChops.difference(src.crop(crop), new.crop(crop)).getbbox() is not None  # avatar replaced
                swapped += 1
            else:
                assert os.path.islink(dst) and not m["faces_used"]
    assert swapped > 0, "test data has no FACE boxes"
    held = {os.path.basename(p) for p in pool.heldout}
    assert used["train"].isdisjoint(held)
    assert (used["val"] | used["bench"]) <= held
    assert json.load(open(os.path.join(_yolo(), "stats.json")))["face_photos_shared_train_vs_eval"] == 0


def test_label_paths_and_yaml():
    out = _yolo()
    for m in _metas("train")[:5]:
        p = os.path.join(out, "images", "train", m["file"])
        assert os.path.exists(label_path(p)) and Image.open(p).size[0] > 0  # symlinks resolve
    y = open(os.path.join(out, "data.yaml")).read()
    assert f"path: {os.path.abspath(out)}" in y and "nc: 4" in y and "3: STAMP" in y
    assert len(os.listdir(os.path.join(out, "previews"))) == 2


def test_rebuild_and_refuse_foreign_dir():
    out = os.path.join(TMP, "yolo_rebuild")
    for _ in range(2):  # second build replaces the first cleanly
        s = build(_data(), out, n_train=6, n_val=2, faces_dir=None, workers=1)
    assert s["splits"]["train"]["images"] == 6 and "warning" in s
    foreign = os.path.join(TMP, "foreign")
    os.makedirs(foreign, exist_ok=True)
    open(os.path.join(foreign, "keep.txt"), "w").write("x")
    try:
        build(_data(), foreign, n_train=2, n_val=2, workers=1)
        raise AssertionError("should refuse a non-dataset folder")
    except FileExistsError:
        pass
    assert os.path.exists(os.path.join(foreign, "keep.txt"))


def test_iou_and_ap50():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0 and iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-9
    gts = [[{"label": "FACE", "bbox": [0, 0, 10, 10]}], [{"label": "FACE", "bbox": [0, 0, 10, 10]}]]
    perfect = [[{"label": "FACE", "bbox": [0, 0, 10, 10], "score": 0.9}],
               [{"label": "FACE", "bbox": [0, 0, 10, 10], "score": 0.8}]]
    assert ap50(perfect, gts, "FACE") == 1.0
    assert ap50(perfect, gts, "STAMP") is None
    # a confident false positive first: precision 0.5 at recall 0.5, 0.67 at recall 1 -> AP = 0.5*0.667+0.5*0.667
    fp_first = [[{"label": "FACE", "bbox": [50, 50, 60, 60], "score": 0.99},
                 {"label": "FACE", "bbox": [0, 0, 10, 10], "score": 0.9}],
                [{"label": "FACE", "bbox": [0, 0, 10, 10], "score": 0.8}]]
    assert abs(ap50(fp_first, gts, "FACE") - 2 / 3) < 1e-9
    # duplicate detections of one box: the second is a false positive
    dup = [[{"label": "FACE", "bbox": [0, 0, 10, 10], "score": 0.9}, {"label": "FACE", "bbox": [0, 0, 10, 10], "score": 0.8}], []]
    assert abs(ap50(dup, gts, "FACE") - 0.5) < 1e-9


def test_evaluator_gold_and_empty():
    res = evaluate(_yolo(), ["gold", "empty"], pad=0)
    gold, empty = res["systems"]
    assert res["pages"] == len(_metas("bench")) > 0
    assert gold["fully_redacted"] == 1.0 and gold["detected"] == 1.0 and gold["precision"] == 1.0
    assert gold["map50"] == 1.0 and gold["over_redaction"] == 0.0
    assert empty["fully_redacted"] == 0.0 and empty["detected"] == 0.0 and empty["over_redaction"] == 0.0
    assert empty["precision"] is None and empty["map50"] == 0.0
    padded = evaluate(_yolo(), ["gold"], pad=6)["systems"][0]
    assert padded["fully_redacted"] == 1.0 and 0.0 < padded["over_redaction"] < 0.5  # padding costs a little


def test_opencv_baseline_and_cli():
    out = os.path.join(TMP, "eval_cv")
    _py("parda.vision.evaluate_vis", "--yolo_data", _yolo(), "--systems", "opencv,gold", "--out", out)
    res = json.load(open(os.path.join(out, "report_vis.json")))
    cv = res["systems"][0]
    assert cv["system"] == "opencv"
    for k in ("fully_redacted", "detected", "over_redaction"):
        assert 0.0 <= cv[k] <= 1.0, (k, cv[k])
    assert cv["ap50"]["SIGNATURE"] in (None, 0.0) and cv["by_class"]["STAMP"]["detected"] in (None, 0.0)
    md = open(os.path.join(out, "report_vis.md"), encoding="utf-8").read()
    assert "| opencv |" in md and "| gold |" in md and "## By class" in md


_FAKE_ULTRA = r'''
import csv, os
from types import SimpleNamespace
import numpy as np
from PIL import Image

NAMES = {0: "FACE", 1: "SIGNATURE", 2: "QR_CODE", 3: "STAMP"}

class _T:
    def __init__(self, a): self.a = np.asarray(a, dtype=float)
    def cpu(self): return self
    def numpy(self): return self.a

class YOLO:
    def __init__(self, weights):
        self.weights, self.trainer = weights, None

    def train(self, **kw):
        assert not kw.get("resume")
        assert kw["fliplr"] == 0.0 and os.path.exists(kw["data"]) and kw["exist_ok"]
        assert ("time" in kw) != (kw["name"] == "smoke")
        d = os.path.join(kw["project"], kw["name"])
        os.makedirs(os.path.join(d, "weights"), exist_ok=True)
        for w in ("best.pt", "last.pt"):
            open(os.path.join(d, "weights", w), "w").write("fake weights")
        with open(os.path.join(d, "results.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["                  epoch", "      train/box_loss", "   metrics/mAP50(B)", "metrics/mAP50-95(B)"])
            for e in range(1, kw["epochs"] + 1):
                wr.writerow([e, 2.0 / e, 0.5 + 0.1 * e, 0.3])
        self.trainer = SimpleNamespace(save_dir=d)
        return SimpleNamespace(ap_class_index=[0, 1], box=SimpleNamespace(ap50=[0.9, 0.8]), names=NAMES)

    def predict(self, paths, conf=0.25, **kw):
        out = []
        for p in paths:  # a "perfect" model: returns the label boxes
            W, H = Image.open(p).size
            head, tail = p.rsplit(os.sep + "images" + os.sep, 1)
            xyxy, cls = [], []
            for ln in open(os.path.join(head, "labels", os.path.splitext(tail)[0] + ".txt")):
                c, cx, cy, w, h = ln.split()
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                xyxy.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                cls.append(int(c))
            n = len(cls)
            out.append(SimpleNamespace(names=NAMES, boxes=SimpleNamespace(
                xyxy=_T(np.reshape(np.asarray(xyxy, dtype=float), (n, 4))), cls=_T(cls), conf=_T([0.9] * n))))
        return out
'''


def test_train_and_eval_glue_with_fake_ultralytics():
    fake = os.path.join(TMP, "fake_ultra")
    os.makedirs(os.path.join(fake, "ultralytics"), exist_ok=True)
    open(os.path.join(fake, "ultralytics", "__init__.py"), "w").write(_FAKE_ULTRA)
    env = {"PYTHONPATH": fake + os.pathsep + ROOT}
    yaml = os.path.join(_yolo(), "data.yaml")
    for smoke in (True, False):
        out = os.path.join(TMP, "yolo_smoke" if smoke else "yolo_run")
        args = ["parda.vision.train_yolo", "--data_yaml", yaml, "--out", out, "--epochs", "3", "--hours", "0.1"]
        _py(*(args + ["--smoke"] if smoke else args), env=env)
        s = json.load(open(os.path.join(out, "train_summary.json")))
        assert s["epochs_run"] == (1 if smoke else 3) and not s["nan_loss"], s
        assert s["val_ap50_by_class"] == {"FACE": 0.9, "SIGNATURE": 0.8}
        assert os.path.exists(os.path.join(out, "best.pt"))
        assert ("est_full_epoch_min" in s) == smoke
    assert s["best_val_map50"] == 0.8 and s["best_epoch"] == 3.0
    ev = os.path.join(TMP, "eval_yolo")
    _py("parda.vision.evaluate_vis", "--yolo_data", _yolo(), "--systems", "yolo",
        "--weights", os.path.join(TMP, "yolo_run", "best.pt"), "--out", ev, env=env)
    rows = json.load(open(os.path.join(ev, "report_vis.json")))["systems"]
    assert [r["system"] for r in rows] == ["yolo@0.25", "yolo@0.5"]
    assert all(r["fully_redacted"] == 1.0 and r["detected"] == 1.0 and r["map50"] == 1.0 for r in rows)


if __name__ == "__main__":
    try:
        for k, f in list(globals().items()):
            if k.startswith("test_"):
                f()
                print("ok ", k, flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
