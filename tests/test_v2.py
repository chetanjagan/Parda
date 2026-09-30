"""Run: python tests/test_v2.py   (CPU only) — text model v2: photo pages, EasyOCR-style labels, multi-source data"""
import os

os.environ["PARDA_BENCH_PER_LANG"] = "3"  # small benchmark; must be set before parda.ocr.run_ocr is imported

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from parda.fsutil import find_path  # noqa: E402
from parda.ocr.engines import OracleEngine  # noqa: E402
from parda.ocr.run_ocr import benchmark_ids  # noqa: E402
from parda.pii.build_data import parse_noisy  # noqa: E402
from parda.pii.spans import transfer_labels_lines  # noqa: E402
from parda.synth.photo_pages import build as build_photo  # noqa: E402
from parda.synth.photo_pages import read_meta  # noqa: E402

TMP = tempfile.mkdtemp(prefix="parda_v2_test_")


def _py(*args):
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (args, r.stdout[-2000:], r.stderr[-3000:])
    return r


def _synth():
    d = os.path.join(TMP, "synth")
    if not os.path.exists(d):  # normal augmentation (tilt etc.) on
        _py("parda.synth.generate", "--n", "30", "--out", d, "--workers", "2", "--langs", "en:1,hi-en:1")
    return d


def _photo():
    d = os.path.join(TMP, "photo")
    if not os.path.exists(os.path.join(d, "annotations.jsonl")):
        build_photo(_synth(), d, per_lang=6, workers=2)
    return d


def _recs(d):
    return [json.loads(ln) for ln in open(os.path.join(d, "annotations.jsonl"), encoding="utf-8")]


def test_line_ocr_labels_have_exact_boundaries():
    """EasyOCR returns whole lines; labels are carried over by text alignment, with exact word boundaries."""
    for d in (_synth(), _photo()):
        for noise in (0.0, 0.15):
            eng = OracleEngine(noise=noise, seed=2)  # true lines, like EasyOCR, with character noise
            n_ent = n_found = n_exact_words = n_spans = n_exact = 0
            for r in _recs(d):
                text, gold, missed = transfer_labels_lines(r, eng.run(None, r))
                n_ent += len(r["entities"])
                n_found += len({g["entity"] for g in gold})
                n_spans += len(gold)
                n_exact_words += sum(len(g["ocr"].split()) == len(g["gt"].split()) for g in gold)
                n_exact += sum(g["ocr"].split() == g["gt"].split() for g in gold)
                assert all(text[g["start"]:g["end"]] == g["ocr"] for g in gold)
            if noise == 0:  # perfect text: labels are exactly the true values (tilted pages: rare line overlaps)
                assert n_exact >= 0.99 * n_spans, (d, n_exact, n_spans)
            assert n_found >= 0.98 * n_ent, (d, noise, n_found, n_ent)
            assert n_exact_words >= 0.98 * n_spans and n_spans <= 1.05 * n_found, (d, noise)


def test_photo_pages_keep_ids_and_skip_benchmark():
    synth, photo = _synth(), _photo()
    stats = json.load(open(os.path.join(photo, "stats.json")))
    assert stats["n_ok"] == 12 and stats["n_errors"] == 0 and stats["benchmark_pages_used"] == 0
    orig = {r["id"]: r for r in _recs(synth)}
    bench = benchmark_ids(synth)
    for r in _recs(photo):
        o = orig[r["id"]]
        assert r["id"] not in bench and r["photo"] and r["text"] == o["text"] and r["entities"][0]["label"] == o["entities"][0]["label"]
        assert any(op.startswith("perspective") for op in r["aug"])
        im = Image.open(os.path.join(photo, r["image"]))
        assert im.size == (r["width"], r["height"]) and r["width"] > o["width"]  # desk margin added
        for w in r["words"]:
            assert 0 <= w["bbox"][0] <= w["bbox"][2] <= r["width"] and 0 <= w["bbox"][1] <= w["bbox"][3] <= r["height"]
    assert [m[0] for m in read_meta(os.path.join(synth, "annotations.jsonl"))] == [r["id"] for r in _recs(synth)]


def test_ocr_pages_cli_resumes():
    out = os.path.join(TMP, "ocr_photo")
    _py("parda.pii.ocr_pages", "--data", _photo(), "--engine", "oracle", "--bench_of", _synth(), "--out_dir", out, "--workers", "1")
    path = os.path.join(out, "ocr_oracle.jsonl")
    lines = open(path).read().count("\n")
    assert lines == 12
    _py("parda.pii.ocr_pages", "--data", _photo(), "--engine", "oracle", "--bench_of", _synth(), "--out_dir", out, "--workers", "1")
    assert open(path).read().count("\n") == lines  # nothing re-run
    sel = os.path.join(TMP, "ocr_scan")
    _py("parda.pii.ocr_pages", "--data", _synth(), "--engine", "oracle_words", "--per_lang", "4", "--out_dir", sel, "--workers", "1")
    ids = [json.loads(ln)["id"] for ln in open(os.path.join(sel, "ocr_oracle_words.jsonl"))]
    assert len(ids) == 8 and not set(ids) & benchmark_ids(_synth())


def test_build_data_from_several_ocr_sources():
    synth, photo = _synth(), _photo()
    words = os.path.join(TMP, "ocr_scan", "ocr_oracle_words.jsonl")
    lines = os.path.join(TMP, "ocr_photo", "ocr_oracle.jsonl")
    if not (os.path.exists(words) and os.path.exists(lines)):
        test_ocr_pages_cli_resumes()
    out = os.path.join(TMP, "gliner_v2")
    _py("parda.pii.build_data", "--data", synth, "--noisy", f"scan_words={synth}:{words}",
        "--noisy", f"photo_lines={photo}:{lines}:lines", "--clean_docs", "10", "--out", out, "--workers", "2")
    st = json.load(open(os.path.join(out, "stats.json")))
    assert [s["name"] for s in st["ocr_sources"]] == ["scan_words", "photo_lines"]
    assert st["ocr_sources"][1]["mode"] == "lines" and st["ocr_sources"][0]["mode"] == "words"
    for name in ("scan_words", "photo_lines"):
        t = st["transfer_by_source"][name]
        assert t["transferred"] > 0 and t["missed"] <= 0.02 * (t["transferred"] + t["missed"]), (name, t)
    assert set(st["train_by_source"]) == {"scan_words", "photo_lines", "clean"}, st["train_by_source"]
    train = json.load(open(os.path.join(out, "train.json")))
    assert all(ex["ner"] and all(0 <= s <= e < len(ex["tokenized_text"]) for s, e, _ in ex["ner"])
               for ex in train if ex["source"] == "photo_lines")
    _py("parda.pii.train_gliner", "--data_dir", out, "--dry_run")


def test_pages_where_ocr_read_nothing_give_no_examples():
    from parda.pii.build_data import noisy_examples
    rec = _recs(_synth())[0]
    for lines in (False, True):
        exs, st = noisy_examples(rec, [], 200, 150, 0.6, lines)
        assert exs == [] and st["empty"] == 1 and st["missed"] == len(rec["entities"])


def test_parse_noisy_and_find_path():
    assert parse_noisy("a=/d:/o.jsonl:lines", "/x") == ("a", "/d", "/o.jsonl", True)
    assert parse_noisy("b=/o.jsonl", "/x") == ("b", "/x", "/o.jsonl", False)
    root = os.path.join(TMP, "input")
    os.makedirs(os.path.join(root, "nb1", "e2e_cache"), exist_ok=True)
    open(os.path.join(root, "nb1", "e2e_cache", "ocr_tesseract.jsonl"), "w").close()
    assert find_path(root, "e2e_cache/ocr_tesseract.jsonl") == [os.path.join(root, "nb1", "e2e_cache", "ocr_tesseract.jsonl")]
    assert find_path(root, "nope.json") == []


if __name__ == "__main__":
    try:
        for k, f in list(globals().items()):
            if k.startswith("test_"):
                f()
                print("ok ", k, flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
