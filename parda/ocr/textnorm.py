"""Text normalisation and edit-distance helpers for OCR evaluation."""
import re
import unicodedata

try:
    from rapidfuzz.distance import Levenshtein as _RF
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)  # zero-width chars OCR often adds/drops


def norm(s: str) -> str:
    """NFC + remove zero-width chars + collapse whitespace."""
    s = unicodedata.normalize("NFC", s or "").translate(_ZW)
    return re.sub(r"\s+", " ", s).strip()


def squash(s: str) -> str:
    """Remove ALL whitespace: '8757 6329 9050' == '875763299050'."""
    return re.sub(r"\s+", "", norm(s))


def levenshtein(a: str, b: str) -> int:
    if HAS_RAPIDFUZZ:
        return _RF.distance(a, b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate = edits / len(reference)."""
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def substring_distance(needle: str, hay: str) -> int:
    """Smallest edit distance between `needle` and ANY substring of `hay` (Sellers' algorithm).
    Used to ask: 'does this Aadhaar number appear somewhere in what OCR read here?'"""
    if not needle:
        return 0
    if not hay:
        return len(needle)
    prev = [0] * (len(hay) + 1)  # empty needle matches anywhere for free
    for i, cn in enumerate(needle, 1):
        cur = [i]
        for j, ch in enumerate(hay, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cn != ch)))
        prev = cur
    return min(prev)


def is_punct(word: str) -> bool:
    return all(unicodedata.category(c).startswith(("P", "S")) for c in word)
