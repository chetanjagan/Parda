"""Run: python tests/test_pipeline.py   (CPU; uses real Tesseract; GLiNER and YOLO are faked)"""
import os

os.environ["PARDA_BENCH_PER_LANG"] = "6"  # small benchmark; must be set before parda.ocr.run_ocr is imported

import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from parda.ocr.engines import WordOracleEngine  # noqa: E402
from parda.pii.predict import RulesPredictor  # noqa: E402
from parda.pii.spans import transfer_labels  # noqa: E402
from parda.pipeline.boxes import draw, fill_gaps, redaction_mask, span_boxes  # noqa: E402
from parda.pipeline.evaluate_e2e import evaluate, load_pages, report_md  # noqa: E402
from parda.vision.detectors import GoldDetector  # noqa: E402
from parda.vision.yolo_data import build  # noqa: E402

TMP = tempfile.mkdtemp(prefix="parda_pipe_test_")


def _py(*args, env=None):
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, **(env or {})))
    assert r.returncode == 0, (args, r.stdout[-2000:], r.stderr[-3000:])
    return r


def _data():
    d = os.path.join(TMP, "synth")
    if not os.path.exists(d):  # clean pages so real Tesseract reads them reliably
        _py("parda.synth.generate", "--n", "24", "--out", d, "--workers", "2", "--aug_prob", "0", "--langs", "en:1,hi-en:1")
    return d


def _bench():
    out = os.path.join(TMP, "yolo")
    if not os.path.exists(os.path.join(out, "data.yaml")):
        faces = os.path.join(TMP, "faces")
        os.makedirs(faces, exist_ok=True)
        rng = random.Random(3)
        for i in range(40):
            Image.new("RGB", (60, 75), tuple(rng.randint(0, 255) for _ in range(3))).save(os.path.join(faces, f"{i:04d}.jpg"))
        build(_data(), out, n_train=2, n_val=2, faces_dir=faces, workers=1, only=("bench",))
    return out


def _pages():
    return load_pages(_data(), _bench())


class GoldText:
    """A perfect text model: the true PII spans on whatever the OCR read."""
    name = "gold"

    def set_page(self, rec, segs):
        _, gold, _ = transfer_labels(rec, segs)
        self.spans = [{"label": g["label"], "start": g["start"], "end": g["end"], "score": 1.0} for g in gold]

    def predict(self, text):
        return self.spans


class Nothing:
    name = "nothing"

    def predict(self, text):
        return []


def test_span_boxes_word_and_line_segments():
    words = [{"bbox": [0, 0, 40, 10]}, {"bbox": [50, 0, 90, 10]}]
    offs = [(0, 4), (5, 9)]  # "abcd efgh"
    assert span_boxes(5, 9, words, offs) == [[50.0, 0.0, 90.0, 10.0]]
    assert span_boxes(0, 9, words, offs) == [[0.0, 0.0, 40.0, 10.0], [50.0, 0.0, 90.0, 10.0]]
    assert span_boxes(10, 12, words, offs) == []
    line = [{"bbox": [0, 0, 100, 10]}]  # one line-level segment of 10 chars: 10 px per char
    assert span_boxes(2, 4, line, [(0, 10)]) == [[10.0, 0.0, 50.0, 10.0]]  # chars 2-4, widened by 1 char each side
    assert span_boxes(0, 2, line, [(0, 10)]) == [[0.0, 0.0, 30.0, 10.0]]   # clipped to the segment


def test_fill_gaps_only_joins_same_label_same_line_close_boxes():
    R = lambda lab, b, kind="text": {"kind": kind, "label": lab, "bbox": b, "score": 0.5, "source": "t"}  # noqa: E731
    regs = [R("ADDRESS", [0, 0, 40, 10]), R("ADDRESS", [60, 0, 90, 10]),      # gap 20 <= 2.5 x 10 -> join
            R("ADDRESS", [300, 0, 340, 10]),                                    # far away -> separate
            R("PHONE", [95, 0, 120, 10]),                                       # other label -> separate
            R("ADDRESS", [0, 40, 40, 50]),                                      # next line -> separate
            R("FACE", [0, 0, 5, 5], kind="visual")]                             # visual untouched
    out = fill_gaps(regs)
    boxes = sorted((r["label"], tuple(r["bbox"])) for r in out)
    assert ("ADDRESS", (0, 0, 90, 10)) in boxes and ("ADDRESS", (300, 0, 340, 10)) in boxes
    assert ("ADDRESS", (0, 40, 40, 50)) in boxes and ("PHONE", (95, 0, 120, 10)) in boxes
    assert ("FACE", (0, 0, 5, 5)) in boxes and len(out) == 5


def test_draw_blacks_out_exactly_the_padded_boxes():
    img = Image.new("RGB", (100, 60), (255, 255, 255))
    regs = [{"kind": "text", "label": "PAN", "bbox": [20, 20, 40, 30], "score": 1, "source": "t"}]
    out = np.asarray(draw(img, regs, pad=2))
    assert (out[18:32, 18:42] == 0).all()                    # box + 2 px padding is black
    assert (out[:17] == 255).all() and (out[:, 43:] == 255).all()
    assert redaction_mask(regs, 100, 60, pad=2).sum() == 14 * 24
    assert np.asarray(img).min() == 255                      # original untouched


def test_perfect_system_scores_100():
    pages = _pages()
    assert len(pages) == 12
    cfgs = [("gold text", ("oracle_words",), "gold", False, False), ("gold all", ("*",), "gold", True, True)]
    res = evaluate(pages, {"oracle_words": WordOracleEngine()}, {"gold": GoldText()}, GoldDetector(), cfgs,
                   os.path.join(TMP, "cache_gold"), workers=1, pad=0)
    t, a = res["configs"]
    assert t["text"] == 1.0 and t["visual"] == 0.0 and t["over_redaction"] == 0.0
    assert a["fully_redacted"] == 1.0 and a["pages_clean"] == 1.0 and a["n_visual"] > 0 and a["n_text"] > 0
    # the only extra black: gap fill also covers the space between two same-label items on one line
    assert a["over_redaction"] < 0.05
    padded = evaluate(pages, {"oracle_words": WordOracleEngine()}, {"gold": GoldText()}, GoldDetector(), cfgs[1:],
                      os.path.join(TMP, "cache_gold"), workers=1)["configs"][0]
    assert padded["fully_redacted"] == 1.0 and 0.0 < padded["over_redaction"] < 0.6  # padding costs a little


def test_nothing_scores_0():
    res = evaluate(_pages(), {"oracle_words": WordOracleEngine()}, {"nothing": Nothing()}, None,
                   [("nothing", ("oracle_words",), "nothing", True, False)], os.path.join(TMP, "cache_none"), workers=1)
    r = res["configs"][0]
    assert r["fully_redacted"] == 0.0 and r["pages_clean"] == 0.0 and r["over_redaction"] == 0.0


def test_real_tesseract_with_rules_and_ocr_cache():
    cache = os.path.join(TMP, "cache_tess")
    cfgs = [("rules only", ("tesseract",), "rules", False, False), ("+ gap fill", ("tesseract",), "rules", True, False)]
    res = evaluate(_pages(), {"tesseract": "tesseract"}, {"rules": RulesPredictor()}, None, cfgs, cache, workers=2)
    assert res["ocr_errors"] == {"tesseract": 0}
    plain, gapped = res["configs"]
    assert 0.0 < plain["text"] < 1.0                       # rules catch the checksum IDs, not names/addresses
    assert gapped["text"] >= plain["text"]                 # gap filling never loses a redaction
    assert plain["over_redaction"] < 0.5
    lines = open(os.path.join(cache, "ocr_tesseract.jsonl")).read().count("\n")
    evaluate(_pages(), {"tesseract": "tesseract"}, {"rules": RulesPredictor()}, None, cfgs, cache, workers=2)
    assert open(os.path.join(cache, "ocr_tesseract.jsonl")).read().count("\n") == lines == 12  # cached, not re-run
    assert "| rules only |" in report_md(res)


def test_redact_cli_image_multipage_and_audit():
    page = _pages()[0]
    src = Image.open(page["path"]).convert("RGB")
    tif = os.path.join(TMP, "two_pages.tif")
    src.save(tif, save_all=True, append_images=[src])
    out = os.path.join(TMP, "redacted")
    _py("parda.pipeline.redact", page["path"], tif, "--out_dir", out, "--lang", page["rec"]["lang"], "--png")
    name = os.path.splitext(os.path.basename(page["path"]))[0]
    pdf = open(os.path.join(out, f"{name}_redacted.pdf"), "rb").read()
    assert pdf.startswith(b"%PDF")
    audit = json.load(open(os.path.join(out, "two_pages_audit.json")))
    assert len(audit["pages"]) == 2 and audit["lang"] == page["rec"]["lang"]
    regions = [r for pg in audit["pages"] for r in pg["regions"]]
    assert regions and all(set(r) == {"kind", "label", "bbox", "score", "source"} for r in regions)  # no PII text
    red = np.asarray(Image.open(os.path.join(out, f"{name}_p1_redacted.png")))
    first = json.load(open(os.path.join(out, f"{name}_audit.json")))["pages"][0]["regions"][0]["bbox"]
    cx, cy = int((first[0] + first[2]) / 2), int((first[1] + first[3]) / 2)
    assert (red[cy, cx] == 0).all()  # the region really is black in the output


def test_redact_cli_pdf_input_if_pymupdf():
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        try:
            import fitz  # noqa: F401
        except ImportError:
            print("   (skipped: pymupdf not installed)")
            return
    page = _pages()[0]
    pdf_in = os.path.join(TMP, "scan.pdf")
    Image.open(page["path"]).convert("RGB").save(pdf_in, resolution=100)
    out = os.path.join(TMP, "redacted_pdf")
    _py("parda.pipeline.redact", pdf_in, "--out_dir", out, "--lang", page["rec"]["lang"], "--dpi", "100")
    assert len(json.load(open(os.path.join(out, "scan_audit.json")))["pages"]) == 1


_FAKE_GLINER = r'''
from parda.pii.labels import LABELS
from parda.pii.rules import find_rules

class GLiNER:
    @classmethod
    def from_pretrained(cls, path):
        return cls()
    def to(self, device):
        return self
    def eval(self):
        return self
    def predict_entities(self, text, labels, threshold=0.5):  # a "model" that knows only the rules
        return [{"start": s["start"], "end": s["end"], "label": LABELS[s["label"]], "score": 0.9}
                for s in find_rules(text) if LABELS.get(s["label"]) in labels]
'''

_FAKE_ULTRA = r'''
import os
from types import SimpleNamespace
import numpy as np
from PIL import Image
NAMES = {0: "FACE", 1: "SIGNATURE", 2: "QR_CODE", 3: "STAMP"}
class _T:
    def __init__(self, a): self.a = np.asarray(a, dtype=float)
    def cpu(self): return self
    def numpy(self): return self.a
class YOLO:
    def __init__(self, weights): pass
    def predict(self, paths, conf=0.25, **kw):
        out = []
        for p in paths:  # a "perfect" model: returns the label boxes (none for files outside a YOLO dataset)
            W, H = Image.open(p).size
            xyxy, cls, lines = [], [], []
            if os.sep + "images" + os.sep in p:
                head, tail = p.rsplit(os.sep + "images" + os.sep, 1)
                lp = os.path.join(head, "labels", os.path.splitext(tail)[0] + ".txt")
                lines = open(lp).read().split("\n") if os.path.exists(lp) else []
            for ln in (l for l in lines if l.strip()):
                c, cx, cy, w, h = ln.split()
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                xyxy.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]); cls.append(int(c))
            n = len(cls)
            out.append(SimpleNamespace(names=NAMES, boxes=SimpleNamespace(
                xyxy=_T(np.reshape(np.asarray(xyxy, dtype=float), (n, 4))), cls=_T(cls), conf=_T([0.9] * n))))
        return out
'''


def test_evaluate_e2e_cli_with_fake_models():
    fake = os.path.join(TMP, "fakes")
    for mod, src in (("gliner", _FAKE_GLINER), ("ultralytics", _FAKE_ULTRA)):
        os.makedirs(os.path.join(fake, mod), exist_ok=True)
        open(os.path.join(fake, mod, "__init__.py"), "w").write(src)
    weights = os.path.join(TMP, "best.pt")
    open(weights, "w").write("fake")
    out = os.path.join(TMP, "e2e")
    env = {"PYTHONPATH": fake + os.pathsep + ROOT}
    _py("parda.pipeline.evaluate_e2e", "--data", _data(), "--yolo_data", _bench(), "--ocr", "tesseract",
        "--gliner_ft", "fake/ft", "--gliner_base", "fake/base", "--yolo", weights,
        "--cache", os.path.join(TMP, "cache_cli"), "--workers", "2", "--out", out, env=env)
    res = json.load(open(os.path.join(out, "report_e2e.json")))
    names = [r["config"] for r in res["configs"]]
    assert names == ["rules only", "off-the-shelf GLiNER + rules", "fine-tuned GLiNER + rules", "+ gap fill",
                     "+ YOLO visual = Parda", "browser build: Tesseract only + YOLO"], names  # EasyOCR row skipped
    full = res["configs"][-2]
    assert full["visual"] == 1.0 and full["text"] == res["configs"][-3]["text"]
    assert res["configs"][-1]["fully_redacted"] == full["fully_redacted"]  # only Tesseract given: same system
    md = open(os.path.join(out, "report_e2e.md"), encoding="utf-8").read()
    assert "| + YOLO visual = Parda |" in md and "## By PII type" in md
    # the redact CLI loads the same fake models from "HF repos"
    red = os.path.join(TMP, "redacted_models")
    page = _pages()[0]
    _py("parda.pipeline.redact", page["path"], "--out_dir", red, "--gliner", "fake/ft", "--yolo", weights, env=env)
    regions = json.load(open(os.path.join(red, os.path.splitext(os.path.basename(page["path"]))[0] + "_audit.json")))["pages"][0]["regions"]
    assert "tesseract" in {r["source"] for r in regions} and {r["source"] for r in regions} <= {"tesseract", "yolo"}


if __name__ == "__main__":
    try:
        for k, f in list(globals().items()):
            if k.startswith("test_"):
                f()
                print("ok ", k, flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
