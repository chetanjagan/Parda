"""Run: python -m pytest -q tests/   (or: python tests/test_synth.py)"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parda.synth import ids  # noqa: E402
from parda.synth.canvas import available_scripts  # noqa: E402
from parda.synth.templates import TEMPLATES  # noqa: E402


def test_aadhaar_checksum():
    rng = random.Random(0)
    for _ in range(2000):
        a = ids.aadhaar(rng, fmt="plain")
        assert len(a) == 12 and a[0] in "23456789" and ids.verhoeff_valid(a)
        assert not ids.verhoeff_valid(ids.fake_12_digit_non_aadhaar(rng))


def test_verhoeff_detects_single_digit_errors():
    rng = random.Random(1)
    for _ in range(300):
        a = list(ids.aadhaar(rng, fmt="plain"))
        i = rng.randrange(12)
        a[i] = str((int(a[i]) + rng.randint(1, 9)) % 10)
        assert not ids.verhoeff_valid("".join(a))


def test_gstin_known_example():
    assert ids.gstin_check_char("27AAPFU0939F1Z") == "V"  # widely published sample GSTIN


def test_formats():
    rng = random.Random(2)
    for _ in range(500):
        assert re.fullmatch(r"[A-Z]{3}P[A-Z]\d{4}[A-Z]", ids.pan(rng, "Kumar"))
        assert re.fullmatch(r"[A-Z]{4}0\d{6}", ids.ifsc(rng, "HDFC"))
        g = ids.gstin(rng)
        assert len(g) == 15 and ids.gstin_check_char(g[:14]) == g[14]
        assert re.fullmatch(r"[A-Z]{3}\d{7}", ids.voter_epic(rng))


def test_templates_labels_consistent():
    langs = ["en"] + [l for l, k in (("hi-en", "deva"), ("kn-en", "knda")) if available_scripts()[k]]
    for name, fn in TEMPLATES.items():
        for lang in langs:
            for seed in range(5):
                doc = fn(random.Random(seed), lang)
                rec = doc.finalize()
                text = rec["text"]
                assert rec["entities"], f"{name}/{lang}: no entities"
                for e in rec["entities"]:
                    # the span must point at exactly the entity text
                    assert text[e["start"]:e["end"]] == e["text"], (name, lang, e)
                    x0, y0, x1, y1 = e["bbox"]
                    assert 0 <= x0 < x1 <= doc.img.width and 0 <= y0 < y1 <= doc.img.height, (name, e)
                    if e["label"] == "AADHAAR":
                        assert ids.verhoeff_valid(e["text"]), e
                for w in rec["words"]:
                    assert text[w["start"]:w["end"]] == w["text"]
                for v in rec["visuals"]:
                    assert v["label"] in {"FACE", "SIGNATURE", "QR_CODE", "STAMP"}


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok ", k)
