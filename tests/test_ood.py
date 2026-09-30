"""Run: python tests/test_ood.py   (CPU only)"""
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

from parda.ocr.engines import WordOracleEngine  # noqa: E402
from parda.pii.labels import LABELS  # noqa: E402
from parda.pii.spans import transfer_labels  # noqa: E402
from parda.pipeline.evaluate_e2e import evaluate, load_pages  # noqa: E402
from parda.synth.canvas import available_scripts  # noqa: E402
from parda.synth.ood import TEMPLATES, generate, homography, phone_photo, warp_points  # noqa: E402
from parda.vision.classes import CLASSES  # noqa: E402

TMP = tempfile.mkdtemp(prefix="parda_ood_test_")
LANGS = ["en"] + [lg for lg, sc in (("hi-en", "deva"), ("kn-en", "knda")) if available_scripts()[sc]]


def _faces():
    d = os.path.join(TMP, "faces")
    if not os.path.exists(d):
        os.makedirs(d)
        rng = random.Random(2)
        for i in range(60):
            Image.new("RGB", (80, 100), tuple(rng.randint(0, 255) for _ in range(3))).save(os.path.join(d, f"{i:05d}.jpg"))
    return d


def test_homography_maps_the_four_points():
    src = [(0, 0), (100, 0), (100, 50), (0, 50)]
    dst = [(10, 5), (120, 0), (115, 70), (0, 60)]
    Hm = homography(src, dst)
    assert np.allclose(warp_points(Hm, src), dst, atol=1e-6)
    assert np.allclose(warp_points(homography(src, src), [(37, 12)]), [(37, 12)])


def test_phone_photo_boxes_follow_the_ink():
    """Draw black blocks at known boxes, warp the page, and check the warped boxes hold the ink."""
    img = Image.new("RGB", (600, 800), (255, 255, 255))
    d = ImageDraw.Draw(img)
    rec = {"words": [], "entities": [], "visuals": []}
    for (x, y, w, h) in [(40, 60, 120, 30), (400, 500, 150, 40), (250, 700, 60, 60)]:
        d.rectangle([x, y, x + w - 1, y + h - 1], fill=(0, 0, 0))
        rec["words"].append({"bbox": [x, y, x + w, y + h]})
    for seed in range(5):
        out, r2, ops = phone_photo(img.copy(), json.loads(json.dumps(rec)), random.Random(seed), desk=(255, 255, 255),
                                   tilt=0.07, effects=False)
        dark = np.asarray(out.convert("L")) < 100
        inside = np.zeros_like(dark)
        for w in r2["words"]:
            x0, y0, x1, y1 = w["bbox"]
            inside[y0:y1, x0:x1] = True
        assert dark.sum() > 5000 and (dark & inside).sum() / dark.sum() > 0.995, (seed, ops)
        # and the boxes are tight: most of each box is ink (a warped rectangle fills most of its enclosing box)
        for w in r2["words"]:
            x0, y0, x1, y1 = w["bbox"]
            assert dark[y0:y1, x0:x1].mean() > 0.6


def test_templates_are_valid_in_every_language():
    pii = set(LABELS)
    for name, fn in TEMPLATES.items():
        for lang in LANGS:
            for seed in range(3):
                rec = fn(random.Random(seed), lang, []).finalize()
                assert rec["entities"], (name, lang)
                for e in rec["entities"]:
                    assert e["label"] in pii, (name, e["label"])
                    assert rec["text"][e["start"]:e["end"]] == e["text"]
                    assert any(w["start"] >= e["start"] and w["end"] <= e["end"] for w in rec["words"]), (name, e)
                assert all(v["label"] in CLASSES for v in rec["visuals"])


def test_hard_negatives_are_not_labelled():
    rec = TEMPLATES["bank_statement"](random.Random(1), "en", []).finalize()
    labelled = {w["text"] for e in rec["entities"] for w in rec["words"] if w["start"] >= e["start"] and w["end"] <= e["end"]}
    refs = [w["text"] for w in rec["words"] if w["text"].isdigit() and len(w["text"]) == 12]
    assert len(refs) >= 10 and not (set(refs) & labelled)  # 12-digit references look like Aadhaar but are not PII


class GoldText:
    name = "gold"

    def set_page(self, rec, segs):
        _, gold, _ = transfer_labels(rec, segs)
        self.spans = [{"label": g["label"], "start": g["start"], "end": g["end"], "score": 1.0} for g in gold]

    def predict(self, text):
        return self.spans


class GoldVisual:
    """Perfect visual detector for pages outside a YOLO dataset: reads the annotation."""

    def __init__(self, pages):
        self.by_path = {p["path"]: p["rec"]["visuals"] for p in pages}

    def detect(self, paths):
        return [[dict(v, score=1.0) for v in self.by_path[p]] for p in paths]


def test_generate_is_balanced_and_scores_100_with_perfect_parts():
    out = os.path.join(TMP, "ood")
    n = 4 * len(LANGS) * 2
    s = generate(n, out, faces_dir=_faces(), workers=2)
    assert s["n_ok"] == n and s["n_errors"] == 0, s["errors"]
    assert set(s["langs"].values()) == {n // len(LANGS)} and set(s["doc_types"].values()) == {n // 4}
    assert s["face_photos"] > 0 and "warning" not in s and s["visuals"].get("FACE", 0) > 0
    recs = [json.loads(ln) for ln in open(os.path.join(out, "annotations.jsonl"), encoding="utf-8")]
    assert all(r["ood"] and "perspective(0.05)" in r["aug"] for r in recs)
    pages = load_pages(out)
    assert len(pages) == n
    cfg = [("perfect", ("oracle_words",), "gold", False, True)]
    r = evaluate(pages, {"oracle_words": WordOracleEngine()}, {"gold": GoldText()}, GoldVisual(pages), cfg,
                 os.path.join(TMP, "cache"), workers=1, pad=0)["configs"][0]
    assert r["fully_redacted"] == 1.0 and r["pages_clean"] == 1.0 and r["over_redaction"] == 0.0, r


def test_cli_clean_pages_for_printing():
    out = os.path.join(TMP, "print")
    r = subprocess.run([sys.executable, "-m", "parda.synth.ood", "--n", "4", "--out", out, "--clean", "--workers", "1"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    recs = [json.loads(ln) for ln in open(os.path.join(out, "annotations.jsonl"), encoding="utf-8")]
    assert len(recs) == 4 and all(r["aug"] == [] for r in recs)
    assert json.load(open(os.path.join(out, "stats.json")))["warning"].startswith("no faces_dir")
    assert Image.open(os.path.join(out, recs[0]["image"])).size == (recs[0]["width"], recs[0]["height"])


if __name__ == "__main__":
    try:
        for k, f in list(globals().items()):
            if k.startswith("test_"):
                f()
                print("ok ", k, flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
