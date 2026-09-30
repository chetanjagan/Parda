"""Run: python tests/test_ink.py   (CPU only; YOLO is faked)"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from parda.synth.ood import generate  # noqa: E402
from parda.vision.classes import CLASSES  # noqa: E402
from parda.vision.ink_check import ink_mask, score_item  # noqa: E402

TMP = tempfile.mkdtemp(prefix="parda_ink_test_")
GT = [50, 100, 350, 170]  # a wide box around a tilted signature


def _page(paper=255):
    img = Image.new("L", (400, 300), paper)
    d = ImageDraw.Draw(img)
    pts = [(75 + 25 * k, 120 + 3 * k + (8 if k % 2 else -4)) for k in range(11)]  # zig-zag, tilted
    d.line(pts, fill=30, width=3)
    return np.asarray(img)


def _black(box, shape=(300, 400)):
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def test_tight_box_hides_all_ink_but_fails_the_box_rule():
    g = _page()
    tight = score_item(g, GT, _black([70, 110, 332, 160]))
    assert tight["ink_hidden"] == 1.0 and tight["box_cover"] < 0.7 and tight["ink_px"] > 500  # fails the 95% box rule
    full = score_item(g, GT, _black(GT))
    assert full["ink_hidden"] == 1.0 and full["box_cover"] == 1.0
    none = score_item(g, GT, _black([0, 0, 0, 0]))
    assert none["ink_hidden"] == 0.0 and none["box_cover"] == 0.0
    half = score_item(g, GT, _black([50, 100, 200, 170]))
    assert 0.4 < half["ink_hidden"] < 0.6


def test_ink_is_measured_against_the_paper_colour():
    for paper in (255, 190):  # white and grey (shadowed) paper: only the strokes count as ink
        ink, _ = ink_mask(_page(paper), GT)
        drawn = (_page(paper)[100:170, 50:350] < 100).sum()
        assert abs(int(ink.sum()) - int(drawn)) <= 0.02 * drawn, (paper, ink.sum(), drawn)
    blank = np.full((300, 400), 230, np.uint8)
    ink, _ = ink_mask(blank, GT)
    assert ink.sum() == 0 and score_item(blank, GT, _black([0, 0, 0, 0]))["ink_hidden"] == 1.0  # nothing to hide


_FAKE_ULTRA = r'''
import json, os
from types import SimpleNamespace
import numpy as np
NAMES = {0: "FACE", 1: "SIGNATURE", 2: "QR_CODE", 3: "STAMP"}
BOXES = json.load(open(os.environ["FAKE_BOXES"]))
class _T:
    def __init__(self, a): self.a = np.asarray(a, dtype=float)
    def cpu(self): return self
    def numpy(self): return self.a
class YOLO:
    def __init__(self, weights): pass
    def predict(self, paths, conf=0.25, **kw):
        out = []
        for p in paths:
            b = BOXES.get(os.path.basename(p), [])
            n = len(b)
            out.append(SimpleNamespace(names=NAMES, boxes=SimpleNamespace(
                xyxy=_T(np.reshape(np.asarray([x[1] for x in b], dtype=float), (n, 4))),
                cls=_T([x[0] for x in b]), conf=_T([0.9] * n))))
        return out
'''


def test_cli_on_photographed_pages_with_tight_signature_boxes():
    faces = os.path.join(TMP, "faces")
    os.makedirs(faces)
    rng = random.Random(4)
    for i in range(40):
        Image.new("RGB", (80, 100), tuple(rng.randint(0, 255) for _ in range(3))).save(os.path.join(faces, f"{i:05d}.jpg"))
    data = os.path.join(TMP, "ood")
    generate(8, data, faces_dir=faces, workers=2)
    recs = [json.loads(ln) for ln in open(os.path.join(data, "annotations.jsonl"), encoding="utf-8")]
    boxes = {}
    for r in recs:  # a model that finds everything, with signature boxes shrunk to the ink itself
        g = np.asarray(Image.open(os.path.join(data, r["image"])).convert("L"))
        out = []
        for v in r["visuals"]:
            b = v["bbox"]
            if v["label"] == "SIGNATURE":
                ink, (x0, y0, _, _) = ink_mask(g, b)
                ys, xs = np.nonzero(ink)
                b = [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1]
            out.append([CLASSES.index(v["label"]), b])
        boxes[os.path.basename(r["image"])] = out
    fake = os.path.join(TMP, "fake")
    os.makedirs(os.path.join(fake, "ultralytics"))
    open(os.path.join(fake, "ultralytics", "__init__.py"), "w").write(_FAKE_ULTRA)
    json.dump(boxes, open(os.path.join(TMP, "boxes.json"), "w"))
    weights = os.path.join(TMP, "best.pt")
    open(weights, "w").write("x")
    out = os.path.join(TMP, "ink")
    env = dict(os.environ, PYTHONPATH=fake + os.pathsep + ROOT, FAKE_BOXES=os.path.join(TMP, "boxes.json"))
    r = subprocess.run([sys.executable, "-m", "parda.vision.ink_check", "--data", data, "--yolo", weights, "--out", out],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.load(open(os.path.join(out, "ink_check.json")))
    sig = res["SIGNATURE"]
    assert sig["n"] > 0 and sig["ink_fully_hidden"] >= 0.95 and sig["box_rule"] < sig["ink_fully_hidden"], sig
    for c in ("FACE", "STAMP"):
        if res.get(c):
            assert res[c]["box_rule"] == 1.0 and res[c]["detected"] == 1.0 and res[c]["ink_fully_hidden"] == 1.0, (c, res[c])
    assert "| SIGNATURE |" in open(os.path.join(out, "ink_check.md"), encoding="utf-8").read()
    assert os.listdir(os.path.join(out, "examples"))  # before|after crops of signatures that fail only the box rule


if __name__ == "__main__":
    try:
        for k, f in list(globals().items()):
            if k.startswith("test_"):
                f()
                print("ok ", k, flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
