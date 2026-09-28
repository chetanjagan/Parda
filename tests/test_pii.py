"""Run: python tests/test_pii.py"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from parda.ocr.textnorm import loose  # noqa: E402
from parda.pii.labels import LABELS, to_code  # noqa: E402
from parda.pii.predict import GLiNERPredictor, GoldPredictor, RulesPredictor, UnionPredictor, merge_spans  # noqa: E402
from parda.pii.rules import find_rules  # noqa: E402
from parda.pii.spans import (build_ocr_text, char_spans_to_token_ner, chunk_examples, tokenize,  # noqa: E402
                             transfer_labels, windows)
from parda.synth import ids  # noqa: E402

TMP = tempfile.mkdtemp(prefix="parda_pii_test_")


def _n_langs():
    # the generator drops hi-en when no Devanagari font is installed (e.g. an OCR/GPU-only Kaggle session)
    from parda.synth.canvas import available_scripts
    return 2 if available_scripts()["deva"] else 1
ENV = {"PARDA_BENCH_PER_LANG": "3"}  # tiny benchmark for the tiny test dataset


def _py(*args):
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, **ENV))
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return r.stdout


def _data():
    d = os.path.join(TMP, "synth")
    if not os.path.exists(d):
        _py("parda.synth.generate", "--n", "30", "--out", d, "--workers", "2", "--langs", "en:1,hi-en:1")
    return d


def _records():
    return [json.loads(l) for l in open(os.path.join(_data(), "annotations.jsonl"), encoding="utf-8")]


def _word_ocr(rec, noise=0.0, rng=None):
    """Word-level fake OCR: the true words (optionally corrupted), like Tesseract's output format."""
    rng = rng or random.Random(0)
    segs = []
    for w in rec["words"]:
        t = w["text"]
        if noise and rng.random() < noise and len(t) > 2:
            i = rng.randrange(len(t))
            t = t[:i] + rng.choice("0O1Il") + t[i + 1:]
        segs.append({"text": t, "bbox": w["bbox"], "score": 1.0})
    return segs


# ---------------------------------------------------------------- spans
def test_tokenize_and_token_spans():
    text = "Name: Ramesh Kumar, PAN: ABCPK1234F."
    toks = tokenize(text)
    assert [t for t, _, _ in toks] == ["Name", ":", "Ramesh", "Kumar", ",", "PAN", ":", "ABCPK1234F", "."]
    spans = [{"label": "person name", "start": 6, "end": 18}, {"label": "pan number", "start": 25, "end": 35}]
    ner = char_spans_to_token_ner(toks, spans)
    assert ner == [[2, 3, "person name"], [7, 7, "pan number"]]  # inclusive end


def test_windows_and_chunks():
    assert windows(10, 200, 150) == [(0, 10)]
    w = windows(500, 200, 150)
    assert w[0] == (0, 200) and w[-1][1] == 500 and all(b - a <= 200 for a, b in w)
    toks = [(f"t{i}", 0, 0) for i in range(400)]
    ner = [[10, 12, "x"], [195, 205, "y"], [390, 391, "z"]]
    ch = chunk_examples(toks, ner, 200, 150)
    for ex in ch:
        for s, e, _ in ex["ner"]:
            assert 0 <= s <= e < len(ex["tokenized_text"])
    labs = [l for ex in ch for _, _, l in ex["ner"]]
    assert "x" in labs and "y" in labs and "z" in labs  # 'y' crosses a boundary but fits the next window


def test_label_transfer_perfect_words():
    for rec in _records()[:12]:
        text, gold, missed = transfer_labels(rec, _word_ocr(rec))
        assert not missed, missed[:2]
        assert len(gold) == len(rec["entities"])
        for g in gold:
            assert loose(g["ocr"]) == loose(g["gt"]) and g["cer"] == 0, g
            assert text[g["start"]:g["end"]] == g["ocr"]


def test_label_transfer_noisy_and_tesseract():
    rng = random.Random(1)
    rec = _records()[0]
    text, gold, missed = transfer_labels(rec, _word_ocr(rec, noise=0.5, rng=rng))
    assert gold and any(g["cer"] > 0 for g in gold) and not missed
    if shutil.which("tesseract"):
        out = os.path.join(TMP, "tess")
        _py("parda.ocr.run_ocr", "--engine", "tesseract", "--data", _data(), "--out", out, "--per_lang", "3")
        n_gold = 0
        for line in open(os.path.join(out, "ocr_tesseract.jsonl"), encoding="utf-8"):
            o = json.loads(line)
            r = next(x for x in _records() if x["id"] == o["id"])
            text, gold, missed = transfer_labels(r, o["segments"])
            n_gold += len(gold)
            for g in gold:
                assert text[g["start"]:g["end"]] == g["ocr"]
        assert n_gold > 10


def test_build_ocr_text_offsets():
    segs = [{"text": "PAN:", "bbox": [10, 10, 50, 25]}, {"text": "ABCPK1234F", "bbox": [60, 11, 160, 26]},
            {"text": "Mobile:", "bbox": [10, 50, 70, 65]}]
    text, ordered, offs = build_ocr_text(segs)
    assert text == "PAN: ABCPK1234F\nMobile:"
    assert [text[a:b] for a, b in offs] == ["PAN:", "ABCPK1234F", "Mobile:"]


# ---------------------------------------------------------------- rules
def test_rules_find_valid_ids():
    rng = random.Random(3)
    for _ in range(200):
        a, p, g = ids.aadhaar(rng), ids.pan(rng, "Rao"), ids.gstin(rng)
        f = ids.ifsc(rng, "HDFC")
        txt = f"Aadhaar {a} PAN {p} GSTIN {g} IFSC {f} email ravi.k@gmail.com UPI ravi@okaxis Ph +91 98450 12345"
        got = {(s["label"], txt[s["start"]:s["end"]]) for s in find_rules(txt)}
        for lab, val in [("AADHAAR", a), ("PAN", p), ("GSTIN", g), ("IFSC", f), ("EMAIL", "ravi.k@gmail.com"),
                         ("UPI_ID", "ravi@okaxis"), ("PHONE", "+91 98450 12345")]:
            assert (lab, val) in got, (lab, val, got)


def test_rules_reject_invalid():
    rng = random.Random(4)
    for _ in range(200):
        bad = ids.fake_12_digit_non_aadhaar(rng)
        assert not any(s["label"] == "AADHAAR" for s in find_rules(f"Ref No: {bad}"))
    assert not find_rules("Loan Amount: Rs. 2,00,000 Tenure 24 months")


# ---------------------------------------------------------------- prediction wrapper
class FakeGLiNER:
    """Finds 10-digit phone numbers with a regex, like a model would return them."""

    def __init__(self):
        self.calls = 0

    def predict_entities(self, text, labels, threshold=0.5):
        self.calls += 1
        assert "phone number" in labels and len(text) > 0
        return [{"start": m.start(), "end": m.end(), "text": m.group(), "label": "phone number", "score": 0.9}
                for m in re.finditer(r"[6-9]\d{9}", text)]


def test_gliner_predictor_windowing():
    parts, truth = [], []
    for i in range(120):  # long text -> several windows
        parts.append(f"word{i} filler text here")
        if i % 17 == 0:
            parts.append(f"call {9000000000 + i}")
    text = " ".join(parts)
    truth = [(m.start(), m.end()) for m in re.finditer(r"[6-9]\d{9}", text)]
    fake = FakeGLiNER()
    pred = GLiNERPredictor(fake, threshold=0.5).predict(text)
    assert fake.calls > 1, "should have used several windows"
    assert [(p["start"], p["end"]) for p in pred] == truth  # no duplicates, correct global offsets
    assert all(p["label"] == "PHONE" for p in pred)


def test_merge_and_labels():
    m = merge_spans([{"label": "A", "start": 0, "end": 5, "score": .5}, {"label": "A", "start": 3, "end": 9, "score": .9},
                     {"label": "B", "start": 2, "end": 4, "score": .1}])
    assert {(s["label"], s["start"], s["end"]) for s in m} == {("A", 0, 9), ("B", 2, 4)}
    assert to_code("Aadhaar Number") == "AADHAAR" and len(LABELS) == 18


# ---------------------------------------------------------------- evaluation + data building
def _bench_ocr():
    """Word-level OCR for every page (the evaluator only scores the benchmark subset)."""
    p = os.path.join(TMP, "bench_ocr.jsonl")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            for r in _records():
                f.write(json.dumps({"id": r["id"], "segments": _word_ocr(r), "error": None}, ensure_ascii=False) + "\n")
    return p


def test_evaluator_gold_and_empty():
    os.environ.update(ENV)
    import importlib
    import parda.ocr.run_ocr as R
    importlib.reload(R)
    import parda.pii.evaluate_pii as E
    importlib.reload(E)

    class Empty:
        name = "empty"

        def predict(self, text):
            return []

    rep, md = E.evaluate(_data(), _bench_ocr(), [GoldPredictor(), Empty(), RulesPredictor(),
                                                 UnionPredictor(RulesPredictor(), GoldPredictor())],
                         os.path.join(TMP, "pii_eval"))
    g, e, r = rep["gold"]["overall"], rep["empty"]["overall"], rep["rules"]["overall"]
    assert g["fully_redacted"] == 1.0 and g["over_redaction"] == 0.0 and g["typed_recall"] == 1.0, g
    assert e["fully_redacted"] == 0.0 and e["char_recall"] == 0.0, e
    assert 0 < r["fully_redacted"] < 1.0, r          # rules can't find names/addresses
    assert rep["rules + gold"]["overall"]["fully_redacted"] == 1.0
    assert rep["rules"]["by_label"]["PERSON_NAME"]["fully_redacted"] == 0.0
    assert rep["_meta"]["pages"] == 3 * _n_langs()   # only the (tiny) benchmark pages
    assert "Fully redacted" in md


def test_build_data_and_dry_run():
    d = _data()
    ocr = os.path.join(TMP, "train_ocr.jsonl")
    rng = random.Random(7)
    with open(ocr, "w", encoding="utf-8") as f:
        for r in _records():
            f.write(json.dumps({"id": r["id"], "segments": _word_ocr(r, 0.2, rng), "error": None},
                               ensure_ascii=False) + "\n")
    out = os.path.join(TMP, "gliner_data")
    _py("parda.pii.build_data", "--data", d, "--ocr", ocr, "--clean_docs", "10", "--out", out, "--workers", "2",
        "--val_frac", "0.2")
    st = json.load(open(os.path.join(out, "stats.json")))
    tr = json.load(open(os.path.join(out, "train.json")))
    va = json.load(open(os.path.join(out, "val.json")))
    nb = 3 * _n_langs()
    assert st["benchmark_pages_excluded"] == nb
    assert st["docs"]["noisy_docs"] == 30 - nb and st["docs"]["clean_docs"] == 10, st   # 30 docs - benchmark
    assert tr and va and {e["source"] for e in tr} == {"clean", "noisy"}
    import parda.ocr.run_ocr as R
    bench = R.pick_ids(R._read_meta(d), 3, 7)
    assert len(bench) == nb
    # every entity token span is valid and uses a known label
    for ex in tr + va:
        for s, e, lab in ex["ner"]:
            assert 0 <= s <= e < len(ex["tokenized_text"]) and lab in LABELS.values()
    out_txt = _py("parda.pii.train_gliner", "--data_dir", out, "--dry_run")
    assert '"entities"' in out_txt


# ---------------------------------------------------------------- training glue vs fake GLiNER APIs
_FAKE_COMMON = """
import os, types
class _State:
    def __init__(self): self.global_step = 0; self.log_history = []
class _Trainerish:
    def __init__(self, callbacks=()):
        self.state = _State(); self.callbacks = list(callbacks or [])
    def run(self, steps=3):
        control = types.SimpleNamespace(should_save=False, should_training_stop=False)
        for i in range(steps):
            self.state.global_step += 1
            self.state.log_history.append({"loss": 1.0 / (i + 1)})
            for cb in self.callbacks:
                cb.on_step_end(None, self.state, control)
        self.state.log_history.append({"eval_loss": 0.5})
        return self
class GLiNER:
    def __init__(self):
        self.config = object()
        self.data_processor = types.SimpleNamespace(transformer_tokenizer="tok")
    @classmethod
    def from_pretrained(cls, p): return cls()
    def to(self, d): return self
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True); open(os.path.join(path, "saved"), "w").write("1")
    def predict_entities(self, text, labels, threshold=0.5):
        return [{"label": "pan number", "text": "ABCPK1234F", "score": 0.9, "start": 0, "end": 1}]
"""

_FAKE_NEW = _FAKE_COMMON + """
__version__ = "9.9-new"
def _train_model(self, train_dataset, eval_dataset, output_dir, **kw):
    if "callbacks" in kw: raise TypeError("TrainingArguments got an unexpected keyword argument 'callbacks'")
    if "eval_strategy" in kw: raise TypeError("unexpected keyword argument 'eval_strategy'")
    assert "evaluation_strategy" in kw and train_dataset and eval_dataset
    return _Trainerish().run()
GLiNER.train_model = _train_model
"""

_FAKE_OLD_TRAINING = """
from gliner import _Trainerish
class TrainingArguments:
    def __init__(self, **kw): self.kw = kw
class Trainer(_Trainerish):
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator, callbacks=(), tokenizer=None):
        assert tokenizer == "tok" and data_collator.ok
        super().__init__(callbacks)
    def train(self, resume_from_checkpoint=None): return self.run(4)
"""
_FAKE_OLD_COLLATOR = """
class BaseDataCollator: pass
class DataCollator:
    def __init__(self, config, data_processor=None, prepare_labels=False):
        self.ok = prepare_labels and data_processor is not None
"""


def _make_fakes(root, kind):
    os.makedirs(os.path.join(root, "gliner", "data_processing"), exist_ok=True)
    os.makedirs(os.path.join(root, "torch"), exist_ok=True)
    os.makedirs(os.path.join(root, "transformers"), exist_ok=True)
    open(os.path.join(root, "torch", "__init__.py"), "w").write(
        "class cuda:\n    @staticmethod\n    def is_available(): return False\n")
    open(os.path.join(root, "transformers", "__init__.py"), "w").write(
        "class TrainerCallback:\n    pass\n")
    open(os.path.join(root, "gliner", "data_processing", "__init__.py"), "w").write("")
    if kind == "new":
        open(os.path.join(root, "gliner", "__init__.py"), "w").write(_FAKE_NEW)
        open(os.path.join(root, "gliner", "data_processing", "collator.py"), "w").write(
            "class BaseDataCollator: pass\n")
    else:
        open(os.path.join(root, "gliner", "__init__.py"), "w").write(_FAKE_COMMON + '__version__ = "0.1-old"\n')
        open(os.path.join(root, "gliner", "training.py"), "w").write(_FAKE_OLD_TRAINING)
        open(os.path.join(root, "gliner", "data_processing", "collator.py"), "w").write(_FAKE_OLD_COLLATOR)


def test_train_glue_old_and_new_gliner_api():
    data_dir = os.path.join(TMP, "gliner_data")
    if not os.path.exists(os.path.join(data_dir, "train.json")):
        test_build_data_and_dry_run()
    for kind, strategy in (("new", "model.train_model"), ("old", "Trainer+DataCollator")):
        fake_root = os.path.join(TMP, f"fake_{kind}")
        _make_fakes(fake_root, kind)
        out = os.path.join(TMP, f"train_{kind}")
        r = subprocess.run([sys.executable, "-m", "parda.pii.train_gliner", "--data_dir", data_dir, "--out", out,
                            "--smoke"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONPATH=fake_root + os.pathsep + ROOT))
        assert r.returncode == 0, (kind, r.stdout[-1500:], r.stderr[-2500:])
        summ = json.load(open(os.path.join(out, "train_summary.json")))
        assert summ["strategy"] == strategy, summ
        assert summ["steps"] >= 3 and summ["last_loss"] < summ["first_loss"], summ
        assert os.path.exists(os.path.join(out, "final", "saved"))
        assert "pan number" in r.stdout  # sanity prediction printed


if __name__ == "__main__":
    try:
        for k, f in list(globals().items()):
            if k.startswith("test_"):
                f()
                print("ok ", k)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
