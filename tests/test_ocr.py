"""Run: python tests/test_ocr.py"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from parda.ocr.textnorm import cer, levenshtein, loose, norm, squash, substring_distance  # noqa: E402
from parda.synth.canvas import available_scripts  # noqa: E402

TMP = tempfile.mkdtemp(prefix="parda_ocr_test_")


def _py(*args):
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True, capture_output=True, text=True)


def _data():
    d = os.path.join(TMP, "synth")
    if not os.path.exists(d):
        # clean pages (no augmentation) so the oracle's simple line grouping is exact
        _py("parda.synth.generate", "--n", "24", "--out", d, "--workers", "2", "--aug_prob", "0", "--langs", "en:1,hi-en:1")
    return d


def test_edit_distance():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert substring_distance("9050", "aadhaar 8757 6329 9050 xyz") == 0
    assert substring_distance("9O50", "8757 6329 9050") == 1
    assert substring_distance("abc", "") == 3
    assert squash(" 8757 6329\n9050 ") == "875763299050"
    assert cer("abcd", "abcd") == 0 and abs(cer("abcd", "abxd") - 0.25) < 1e-9
    assert cer("रमेश", "रमेश") == 0


def test_indic_digits_and_loose():
    assert norm("६३२२ ३७३० ८७६४") == "6322 3730 8764"      # Devanagari digits
    assert norm("೧೨೩") == "123"                              # Kannada digits
    assert loose("#173, 1st Main Rd,") == "1731stmainrd"
    assert loose("रमेश कुमार") == "रमेशकुमार"                 # vowel signs (matras) must survive
    assert loose("ಅಮಿತ್ ಕುಲಕರ್ಣಿ") == "ಅಮಿತ್ಕುಲಕರ್ಣಿ"
    assert loose("Rekha31@Yahoo.co.in") == "rekha31yahoocoin"


def _expected_pages():
    # the generator drops hi-en when no Devanagari font is installed (e.g. an OCR-only Kaggle session)
    return 6 * (2 if available_scripts()["deva"] else 1)


def _bench(engine, noise=0.0, tag=None):
    out = os.path.join(TMP, tag or engine)
    args = ["parda.ocr.run_ocr", "--engine", engine, "--data", _data(), "--out", out, "--per_lang", "6"]
    if engine == "oracle":
        args += ["--noise", str(noise)]
    _py(*args)
    _py("parda.ocr.evaluate", "--out", out, "--data", _data(), "--per_lang", "6")
    return json.load(open(os.path.join(out, "report.json")))


def test_oracle_perfect():
    r = _bench("oracle", 0.0, "oracle_clean")["oracle"]["overall"]
    assert r["pages"] == _expected_pages(), r
    assert r["word_exact"] == 1.0, r
    assert r["pii_exact"] == 1.0 and r["pii_found"] == 1.0 and r["pii_missed"] == 0.0, r
    assert r["word_recall"] > 0.99 and r["page_cer"] < 0.01, r


def test_oracle_noisy_is_worse():
    r = _bench("oracle", 0.15, "oracle_noisy")["oracle"]["overall"]
    assert r["pii_exact"] < 0.9 and r["page_cer"] > 0.03, r


def test_resume_skips_done():
    out = os.path.join(TMP, "oracle_clean")
    before = sum(1 for _ in open(os.path.join(out, "ocr_oracle.jsonl")))
    _py("parda.ocr.run_ocr", "--engine", "oracle", "--data", _data(), "--out", out, "--per_lang", "6")
    after = sum(1 for _ in open(os.path.join(out, "ocr_oracle.jsonl")))
    assert before == after == _expected_pages()


def test_find_data_zip_and_duplicates():
    import zipfile
    import parda.ocr.run_ocr as R
    src = _data()
    root = os.path.join(TMP, "kaggle_input")
    os.makedirs(os.path.join(root, "parda-synth-v1"), exist_ok=True)
    R.INPUT_ROOT, R.EXTRACT_TO = root, os.path.join(TMP, "extracted")

    # case 1: dataset is still a zip (with a top folder inside, like Kaggle's output zip)
    zpath = os.path.join(root, "parda-synth-v1", "parda_synth_v1.zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        for dp, _, files in os.walk(src):
            for f in files:
                full = os.path.join(dp, f)
                zf.write(full, os.path.join("parda_synth_v1", os.path.relpath(full, src)))
    d = R.find_data(verbose=False)
    assert R._n_images(d) == 24 and os.path.isfile(os.path.join(d, "annotations.jsonl")), d

    # case 2: a partial copy (few images) also exists -> must pick the full one
    part = os.path.join(root, "parda-synth-v1", "partial")
    os.makedirs(os.path.join(part, "images"), exist_ok=True)
    shutil.copy(os.path.join(src, "annotations.jsonl"), part)
    first = sorted(os.listdir(os.path.join(src, "images")))[0]
    shutil.copy(os.path.join(src, "images", first), os.path.join(part, "images"))
    assert R.find_data(verbose=False) == d

    # case 3: explicit --data wins
    assert R.find_data(src, verbose=False) == src


def test_tesseract_end_to_end():
    if shutil.which("tesseract") is None:
        print("   (skipped: tesseract not installed)")
        return
    rep = _bench("tesseract")["tesseract"]
    en = rep["by_lang"]["en"]
    assert rep["errors"] == 0, rep
    assert en["pii_exact"] > 0.5 and en["page_cer"] < 0.5, en
    for f in ("report.md", "failures.jsonl"):
        assert os.path.exists(os.path.join(TMP, "tesseract", f))


def test_best_of_both():
    out = os.path.join(TMP, "combo")
    for eng, extra in (("oracle", ["--noise", "0.15"]), ("tesseract", [])):
        if eng == "tesseract" and shutil.which("tesseract") is None:
            print("   (skipped: tesseract not installed)")
            return
        _py("parda.ocr.run_ocr", "--engine", eng, "--data", _data(), "--out", out, "--per_lang", "6", *extra)
    _py("parda.ocr.evaluate", "--out", out, "--data", _data(), "--per_lang", "6")
    rep = json.load(open(os.path.join(out, "report.json")))
    best = rep["_best_of_all"]
    assert best["engines"] == ["oracle", "tesseract"]
    for e in ("oracle", "tesseract"):
        assert best["pii_found"] >= rep[e]["overall"]["pii_found"] - 1e-9, (best, e)
    md = open(os.path.join(out, "report.md"), encoding="utf-8").read()
    assert "best of oracle + tesseract" in md


if __name__ == "__main__":
    try:
        for k, f in list(globals().items()):
            if k.startswith("test_"):
                f()
                print("ok ", k)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
