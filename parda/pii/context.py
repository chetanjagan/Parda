"""Context rules applied after the model, to cut false bars on documents the model was not trained on (bank
statements: reference numbers, amounts and transaction dates). Each rule only removes spans; it never adds.

  dob      a DATE OF BIRTH needs a birth label before it on the same line (Birth, DOB, D.O.B, born, जन्म, ಜನ್ಮ)
  aadhaar  a 12-digit AADHAAR written WITHOUT spaces needs an Aadhaar label before it (Aadhaar, UID, आधार, ಆಧಾರ);
           real Aadhaar numbers are printed in groups of four, table reference numbers are not
  cut      a span that starts or ends in the middle of a number (e.g. the "74" of "13,274") is dropped

Python is the reference: web/src/core/context.ts must give the same answers (tools/make_goldens.py checks it).
"""
import re

from .rules import ascii_digits

RULES = ("dob", "aadhaar", "cut")
WINDOW = 60
DOB_WORDS = ("birth", "dob", "d.o.b", "born", "जन्म", "ಜನ್ಮ")
AADHAAR_WORDS = ("aadhaar", "aadhar", "adhaar", "uid", "आधार", "ಆಧಾರ")
DIGITS = "0123456789"
_TWELVE = re.compile(r"[0-9]{12}")


def _before(text, start):
    """Up to 60 characters before the span, on its own line (a label belongs to the line it is on)."""
    line_start = text.rfind("\n", 0, start) + 1
    return text[max(line_start, start - WINDOW):start].lower()


def cuts_number(norm, start, end):
    """Does [start, end) begin or end inside a number? (digits joined by nothing, a comma or a dot count as one)"""
    n = len(norm)
    left = (0 < start < n and norm[start] in DIGITS and
            (norm[start - 1] in DIGITS or (norm[start - 1] in ",." and start >= 2 and norm[start - 2] in DIGITS)))
    right = (0 < end < n and norm[end - 1] in DIGITS and
             (norm[end] in DIGITS or (norm[end] in ",." and end + 1 < n and norm[end + 1] in DIGITS)))
    return left or right


def keep(text, norm, sp, rules=RULES):
    s, e, lab = sp["start"], sp["end"], sp["label"]
    if "cut" in rules and cuts_number(norm, s, e):
        return False
    if "dob" in rules and lab == "DOB" and not any(w in _before(text, s) for w in DOB_WORDS):
        return False
    if ("aadhaar" in rules and lab == "AADHAAR" and _TWELVE.fullmatch(norm[s:e])
            and not any(w in _before(text, s) for w in AADHAAR_WORDS)):
        return False
    return True


def filter_spans(text, spans, rules=RULES):
    norm = ascii_digits(text)
    return [sp for sp in spans if keep(text, norm, sp, rules)]


class ContextPredictor:
    """Any predictor + the context rules."""

    def __init__(self, inner, rules=RULES):
        self.inner, self.rules = inner, tuple(rules)
        self.name = f"{getattr(inner, 'name', 'model')}+context"

    def set_page(self, *a, **k):
        if hasattr(self.inner, "set_page"):
            self.inner.set_page(*a, **k)

    def predict(self, text):
        return filter_spans(text, self.inner.predict(text), self.rules)
