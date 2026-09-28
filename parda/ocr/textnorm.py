"""Text normalisation and edit-distance helpers for OCR evaluation."""
import re
import unicodedata

try:
    from rapidfuzz.distance import Levenshtein as _RF
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)  # zero-width chars OCR often adds/drops
# Indic digits -> ASCII. EasyOCR's Hindi model writes '6322' as '६३२२' (same number, different glyphs).
for _base in (0x0966, 0x09E6, 0x0BE6, 0x0C66, 0x0CE6):  # Devanagari, Bengali, Tamil, Telugu, Kannada
    for _i in range(10):
        _ZW[_base + _i] = str(_i)


def norm(s: str) -> str:
    """NFC + remove zero-width chars + Indic digits to ASCII + collapse whitespace."""
    s = unicodedata.normalize("NFC", s or "").translate(_ZW)
    return re.sub(r"\s+", " ", s).strip()


def loose(s: str) -> str:
    """Letters, combining marks (Hindi/Kannada vowel signs) and digits only, casefolded.
    Drops punctuation, symbols and spaces: '#173, 1st Main Rd,' -> '1731stmainrd'."""
    return "".join(c for c in norm(s) if unicodedata.category(c)[0] in "LMN").casefold()


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
