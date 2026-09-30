"""Span utilities shared by data building, prediction and evaluation.

Key idea (label transfer): we know WHERE every true PII word is on the page (Phase 1 boxes).
Tesseract gives us its own words + boxes. Any Tesseract word sitting on a PII word inherits that
label, so 'SBINO364507' at the IFSC position becomes a training example of an IFSC with an OCR error.
"""
import re
import unicodedata
from statistics import median

from ..ocr.textnorm import levenshtein, loose

# Same splitting rule GLiNER uses internally ("whitespace" splitter), so train and inference agree.
TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")


def tokenize(text):
    """-> list of (token, start_char, end_char)"""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def char_spans_to_token_ner(tokens, spans):
    """spans: [{label, start, end}] in chars -> [[tok_start, tok_end_INCLUSIVE, label]] (GLiNER format)."""
    ner = []
    for sp in spans:
        idx = [i for i, (_, s, e) in enumerate(tokens) if s >= sp["start"] and e <= sp["end"]]
        if idx:
            ner.append([idx[0], idx[-1], sp["label"]])
    return ner


def windows(n_tokens, window=200, stride=150):
    """Overlapping [start, end) token windows covering n_tokens."""
    if n_tokens <= window:
        return [(0, n_tokens)]
    out, s = [], 0
    while True:
        e = min(s + window, n_tokens)
        out.append((s, e))
        if e == n_tokens:
            return out
        s += stride


def chunk_examples(tokens, ner, window=200, stride=150):
    """Split a page into GLiNER examples; keep only entities fully inside each window."""
    exs = []
    for s, e in windows(len(tokens), window, stride):
        ents = [[a - s, b - s, lab] for a, b, lab in ner if a >= s and b < e]
        exs.append({"tokenized_text": [t for t, _, _ in tokens[s:e]], "ner": ents})
    return exs


# ------------------------------------------------------------------ OCR text with offsets
def _yc(s):
    return (s["bbox"][1] + s["bbox"][3]) / 2


def _xc(s):
    return (s["bbox"][0] + s["bbox"][2]) / 2


def _chain_lines(items, h, slope):
    """Walk left->right; attach each word to the line whose last word predicts its height
    (following the page tilt `slope`), else start a new line."""
    tall = lambda x: (x["bbox"][3] - x["bbox"][1]) >= 0.6 * h  # noqa: E731  (commas/dots are tiny & low)
    lines, anchors = [], []  # anchor = last full-height word of each line (predicts where the line goes)
    for s in items:  # items sorted by x0
        best, best_d = None, None
        for li, ln in enumerate(lines):
            last, anc = ln[-1], anchors[li]
            if s["bbox"][0] < last["bbox"][2] - 0.5 * h:  # overlaps horizontally -> not the next word
                continue
            pred = _yc(anc) + slope * (_xc(s) - _xc(anc))
            d = abs(_yc(s) - pred)
            if d < (0.5 * h if tall(s) else 0.8 * h) and (best_d is None or d < best_d):
                best, best_d = li, d
        if best is None:
            lines.append([s])
            anchors.append(s)
        else:
            lines[best].append(s)
            if tall(s) or not tall(anchors[best]):
                anchors[best] = s
    return lines


def _estimate_slope(lines):
    slopes = []
    for ln in lines:
        if len(ln) >= 3:
            dx = _xc(ln[-1]) - _xc(ln[0])
            if dx > 50:
                slopes.append((_yc(ln[-1]) - _yc(ln[0])) / dx)
    return median(slopes) if slopes else 0.0


def build_ocr_text(segs):
    """Order OCR segments into lines (top->bottom, left->right) and join them. Handles tilted scans.
    Returns (text, ordered_segs, [(start, end)] char span of each ordered seg)."""
    segs = [s for s in segs if s.get("text", "").strip()]
    if not segs:
        return "", [], []
    h = median(max(1, s["bbox"][3] - s["bbox"][1]) for s in segs)
    items = sorted(segs, key=lambda s: (s["bbox"][0], _yc(s)))
    slope = _estimate_slope(_chain_lines(items, h, 0.0))  # pass 1: rough lines -> page tilt
    lines = _chain_lines(items, h, slope)                   # pass 2: follow the tilt
    def key(ln):  # top->bottom after removing tilt, using the line's full-height words
        t = [x for x in ln if (x["bbox"][3] - x["bbox"][1]) >= 0.6 * h] or ln
        return median(_yc(x) - slope * _xc(x) for x in t), _xc(ln[0])
    lines.sort(key=key)
    text, ordered, offs = [], [], []
    pos = 0
    for li, ln in enumerate(lines):
        if li:
            text.append("\n")
            pos += 1
        for j, s in enumerate(ln):
            if j:
                text.append(" ")
                pos += 1
            t = s["text"].strip()
            ordered.append(s)
            offs.append((pos, pos + len(t)))
            text.append(t)
            pos += len(t)
    return "".join(text), ordered, offs


# ------------------------------------------------------------------ label transfer
class _Grid:
    """Tiny spatial index so we don't compare every OCR word with every true word."""

    def __init__(self, boxes, cell=64):
        self.cell, self.cells = cell, {}
        for i, b in enumerate(boxes):
            for cx in range(int(b[0]) // cell, int(b[2]) // cell + 1):
                for cy in range(int(b[1]) // cell, int(b[3]) // cell + 1):
                    self.cells.setdefault((cx, cy), []).append(i)

    def query(self, b):
        found = set()
        for cx in range(int(b[0]) // self.cell, int(b[2]) // self.cell + 1):
            for cy in range(int(b[1]) // self.cell, int(b[3]) // self.cell + 1):
                found.update(self.cells.get((cx, cy), ()))
        return found


def _inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(b):
    return max(1, (b[2] - b[0]) * (b[3] - b[1]))


def _snap(t, a, b):
    """Grow [a, b) in string t to whole whitespace-separated tokens."""
    while a > 0 and not t[a - 1].isspace():
        a -= 1
    while b < len(t) and not t[b].isspace():
        b += 1
    return a, b


def _trim(t, a, b, gt):
    """Drop punctuation glued to the outer edges ("Bhat," -> "Bhat", "(ICIC0740688)" -> "ICIC0740688")
    unless the true value itself has it."""
    while b > a and t[b - 1] in ",;:)]}\"'" and not gt.endswith(t[b - 1]):
        b -= 1
    while a < b and t[a] in "([{\"'" and not gt.startswith(t[a]):
        a += 1
    if b > a and t[b - 1] == "." and not gt.endswith("."):
        b -= 1
    return a, b


def transfer_labels_lines(rec, segs, min_overlap=0.5):
    """Like transfer_labels, for LINE-level OCR (EasyOCR returns "Name: Ravi Kumar" as one segment).

    Geometry can't split a line into words exactly, so: (1) each true word is assigned to the OCR line that
    covers most of it, (2) the OCR line text is aligned character-by-character with the true text of its
    words (a diff), (3) every entity's characters are carried across the alignment. Label boundaries stay
    exact even when OCR misreads letters. Returns (text, gold, missed) like transfer_labels."""
    import difflib
    text, ordered, offs = build_ocr_text(segs)
    words, rtext = rec["words"], rec["text"]
    grid = _Grid([s["bbox"] for s in ordered])
    seg_words = [[] for _ in ordered]
    for wi, w in enumerate(words):
        best, best_ov = None, 0.0
        for si in grid.query(w["bbox"]):
            ov = _inter(w["bbox"], ordered[si]["bbox"]) / _area(w["bbox"])
            if ov > best_ov:
                best, best_ov = si, ov
        if best is not None and best_ov >= min_overlap:
            seg_words[best].append(wi)
    char_ent = {}
    for ei, e in enumerate(rec["entities"]):
        for c in range(e["start"], e["end"]):
            char_ent[c] = ei
    pieces = {}  # entity -> [(start, end) in text]
    for si, wis in enumerate(seg_words):
        if not wis:
            continue
        wis.sort(key=lambda i: words[i]["start"])
        tchars, tpos = [], []  # true line text + its offset in rec["text"]
        for k, wi in enumerate(wis):
            if k:
                tchars.append(" ")
                tpos.append(None)
            for c in range(words[wi]["start"], words[wi]["end"]):
                tchars.append(rtext[c])
                tpos.append(c)
        a0 = offs[si][0]
        ocr = text[a0:offs[si][1]]
        t2o = [None] * len(tchars)
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, "".join(tchars), ocr, autojunk=False).get_opcodes():
            if tag in ("equal", "replace"):
                for k in range(i2 - i1):  # replace blocks: spread proportionally
                    t2o[i1 + k] = j1 + min(j2 - j1 - 1, int(k * (j2 - j1) / max(1, i2 - i1)))
        by_ent = {}
        for ti, rc in enumerate(tpos):
            if rc is not None and rc in char_ent and t2o[ti] is not None:
                by_ent.setdefault(char_ent[rc], []).append(t2o[ti])
        for ei, pos in by_ent.items():
            a, b = _snap(ocr, min(pos), max(pos) + 1)
            pieces.setdefault(ei, []).append((a0 + a, a0 + b))
    gold = []
    for ei, ps in sorted(pieces.items()):
        ps.sort()
        merged = [list(ps[0])]
        for a, b in ps[1:]:
            if text[merged[-1][1]:a].strip() == "":  # only whitespace / line break between -> one span
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        e = rec["entities"][ei]
        for a, b in merged:
            a, b = _trim(text, a, b, e["text"])  # after merging: only the outer edges
            ocr_txt = text[a:b]
            la, lb = loose(e["text"]), loose(ocr_txt)
            gold.append({"label": e["label"], "start": a, "end": b, "gt": e["text"], "ocr": ocr_txt,
                         "cer": min(1.0, levenshtein(la, lb) / max(1, len(la))), "entity": ei})
    gold.sort(key=lambda g: g["start"])
    found = {g["entity"] for g in gold}
    missed = [dict(e, entity=ei) for ei, e in enumerate(rec["entities"]) if ei not in found]
    return text, gold, missed


def transfer_labels(rec, segs, min_overlap=0.5):
    """Map true PII entities onto an OCR reading of the page.

    Works for WORD-level OCR output (Tesseract). Returns:
      text      OCR text in reading order
      gold      [{label, start, end, gt, ocr, cer, entity}] spans in `text`
      missed    true entities OCR produced no text for (can't be redacted from text)
    """
    text, ordered, offs = build_ocr_text(segs)
    words = rec["words"]
    word_ent = {}
    for ei, e in enumerate(rec["entities"]):
        for wi, w in enumerate(words):
            if w["start"] >= e["start"] and w["end"] <= e["end"]:
                word_ent[wi] = ei
    grid = _Grid([w["bbox"] for w in words])

    seg_ent = []
    for s in ordered:
        best, best_ov = None, 0.0
        for wi in grid.query(s["bbox"]):
            ov = _inter(s["bbox"], words[wi]["bbox"]) / _area(s["bbox"])
            if ov > best_ov:
                best, best_ov = wi, ov
        seg_ent.append(word_ent.get(best) if best is not None and best_ov >= min_overlap else None)

    gold, found = [], set()
    i = 0
    while i < len(ordered):
        ei = seg_ent[i]
        if ei is None:
            i += 1
            continue
        j = i
        while j + 1 < len(ordered) and seg_ent[j + 1] == ei:
            j += 1
        e = rec["entities"][ei]
        st, en = offs[i][0], offs[j][1]
        ocr_txt = text[st:en]
        a, b = loose(e["text"]), loose(ocr_txt)
        gold.append({"label": e["label"], "start": st, "end": en, "gt": e["text"], "ocr": ocr_txt,
                     "cer": min(1.0, levenshtein(a, b) / max(1, len(a))), "entity": ei})
        found.add(ei)
        i = j + 1
    missed = [dict(e, entity=ei) for ei, e in enumerate(rec["entities"]) if ei not in found]
    return text, gold, missed


def content_positions(text, start, end):
    """Char positions in [start, end) that are letters/marks/digits (ignore spaces & punctuation)."""
    return {i for i in range(start, end) if unicodedata.category(text[i])[0] in "LMN"}
