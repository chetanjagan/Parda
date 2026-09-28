"""Rules-only PII detection: regular expressions + checksum validation.

This is the classic approach (what Presidio's Indian recognizers do). It is precise for
well-formed IDs but cannot find names or addresses, and breaks when OCR changes one character.
"""
import re

from ..synth.ids import gstin_check_char, verhoeff_valid

_B = r"(?<![A-Za-z0-9])"   # left boundary
_E = r"(?![A-Za-z0-9])"    # right boundary

PATTERNS = [
    ("ABHA", _B + r"\d{2}-\d{4}-\d{4}-\d{4}" + _E, None),
    ("AADHAAR", _B + r"[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}" + _E,
     lambda m: verhoeff_valid(re.sub(r"\D", "", m))),
    ("GSTIN", _B + r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]" + _E,
     lambda m: gstin_check_char(m[:14]) == m[14]),
    ("PAN", _B + r"[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]" + _E, None),
    ("IFSC", _B + r"[A-Z]{4}0[A-Z0-9]{6}" + _E, None),
    ("EMAIL", _B + r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+" + _E, None),
    ("UPI_ID", _B + r"[A-Za-z0-9._-]{2,}@[A-Za-z]{2,}" + r"(?![A-Za-z0-9.])", None),
    ("PHONE", r"(?:\+91[ -]?)?" + _B + r"[6-9]\d{4}[ -]?\d{5}" + _E, None),
    ("VOTER_ID", _B + r"[A-Z]{3}\d{7}" + _E, None),
    ("PASSPORT", _B + r"[A-PR-WYZ][1-9]\d{6}" + _E, None),
    ("UAN", _B + r"10\d{10}" + _E, None),
    ("VEHICLE_REG", _B + r"[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,2}[ -]?\d{4}" + _E, None),
]
_COMPILED = [(lab, re.compile(p), check) for lab, p, check in PATTERNS]


def find_rules(text):
    """-> [{label, start, end, score}] non-overlapping (earlier patterns win ties, longer spans win)."""
    cands = []
    for lab, rx, check in _COMPILED:
        for m in rx.finditer(text):
            if check is None or check(m.group()):
                cands.append({"label": lab, "start": m.start(), "end": m.end(), "score": 1.0})
    cands.sort(key=lambda c: (c["start"], -(c["end"] - c["start"])))
    out, last_end = [], -1
    for c in cands:
        if c["start"] >= last_end:
            out.append(c)
            last_end = c["end"]
    return out
