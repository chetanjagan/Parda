"""Drawing surface that remembers WHERE every word is and WHAT it is.

Every word is drawn separately so we know its exact bounding box. Words that
belong to a PII entity share an entity id. `finalize()` turns this into:
  - full page text in reading order
  - per-word boxes + labels (for OCR / layout training)
  - per-entity character spans + boxes (for NER training)
"""
import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont, features

# ---------------- fonts ----------------
FONT_CANDIDATES = {
    "latin": ["NotoSans-Regular.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf"],
    "latin_bold": ["NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf"],
    "serif": ["NotoSerif-Regular.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf", "FreeSerif.ttf"],
    "mono": ["NotoSansMono-Regular.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "FreeMono.ttf"],
    "deva": ["NotoSansDevanagari-Regular.ttf", "Lohit-Devanagari.ttf", "FreeSans.ttf"],
    "deva_bold": ["NotoSansDevanagari-Bold.ttf", "NotoSansDevanagari-Regular.ttf", "FreeSansBold.ttf"],
    "knda": ["NotoSansKannada-Regular.ttf", "Lohit-Kannada.ttf"],
    "knda_bold": ["NotoSansKannada-Bold.ttf", "NotoSansKannada-Regular.ttf"],
}
SEARCH_DIRS = [os.environ.get("PARDA_FONT_DIR", ""), "assets/fonts", "/usr/share/fonts",
               "/usr/local/share/fonts", "/kaggle/working/fonts"]

RAQM = features.check("raqm")  # needed for correct Hindi/Kannada shaping


@lru_cache(maxsize=1)
def _font_index():
    idx = {}
    for d in SEARCH_DIRS:
        if d and os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        idx.setdefault(f, os.path.join(root, f))
    return idx


def find_font(kind: str):
    idx = _font_index()
    for name in FONT_CANDIDATES[kind]:
        if name in idx:
            return idx[name]
    return None


def available_scripts():
    return {k: find_font(k) is not None for k in ("latin", "deva", "knda")}


@lru_cache(maxsize=256)
def get_font(kind: str, size: int):
    path = find_font(kind)
    if path is None:
        raise RuntimeError(f"No font found for '{kind}'. Run scripts/setup_kaggle.sh or put fonts in assets/fonts/")
    engine = ImageFont.Layout.RAQM if RAQM else ImageFont.Layout.BASIC
    return ImageFont.truetype(path, size, layout_engine=engine)


def script_of(word: str) -> str:
    for ch in word:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "deva"
        if 0x0C80 <= o <= 0x0CFF:
            return "knda"
    return "latin"


def font_for(word: str, size: int, style: str = "regular"):
    sc = script_of(word)
    if sc == "latin":
        kind = {"regular": "latin", "bold": "latin_bold", "serif": "serif", "mono": "mono"}[style]
    else:
        kind = sc + ("_bold" if style == "bold" else "")
    return get_font(kind, size)


# ---------------- document ----------------
class Doc:
    def __init__(self, w: int, h: int, bg=(255, 255, 255), doc_type="", lang="en"):
        self.img = Image.new("RGB", (w, h), bg)
        self.bg = bg
        self.draw = ImageDraw.Draw(self.img)
        self.words = []      # {text, bbox, line}
        self.entities = {}   # eid -> {label, word_ids}
        self.visuals = []    # {label, bbox}
        self._next_eid = 0
        self.doc_type, self.lang = doc_type, lang

    # ---- text ----
    def text(self, x, y, s, size=18, style="regular", fill=(20, 20, 20), label=None, ent=None):
        """Draw string s at (x, y) word by word. Returns (x_end, entity_id)."""
        if label is not None and ent is None:
            ent = self._new_entity(label)
        space = font_for(" ", size, style).getlength(" ")
        for i, word in enumerate(s.split()):
            f = font_for(word, size, style)
            self.draw.text((x, y), word, font=f, fill=fill)
            bb = self.draw.textbbox((x, y), word, font=f)
            wid = len(self.words)
            self.words.append({"text": word, "bbox": [int(b) for b in bb], "line": int(y)})
            if ent is not None:
                self.entities[ent]["word_ids"].append(wid)
            x += f.getlength(word) + space
        return x, ent

    def text_width(self, s, size=18, style="regular"):
        space = font_for(" ", size, style).getlength(" ")
        return sum(font_for(w, size, style).getlength(w) for w in s.split()) + space * max(0, len(s.split()) - 1)

    def paragraph(self, x, y, max_w, segments, size=17, style="regular", line_h=None, fill=(20, 20, 20)):
        """Wrap text made of (text, label_or_None) segments. Entities stay intact across line breaks."""
        line_h = line_h or int(size * 1.6)
        space = font_for(" ", size, style).getlength(" ")
        cx = x
        for seg_text, label in segments:
            ent = self._new_entity(label) if label else None
            for k, word in enumerate(seg_text.split()):
                f = font_for(word, size, style)
                wlen = f.getlength(word)
                # punctuation that starts a segment sticks to the previous word ("Kumar," not "Kumar ,")
                glue = k == 0 and word[0] in ",.;:)'\"" and cx > x
                if glue:
                    cx -= space
                if cx + wlen > x + max_w and cx > x and not glue:
                    cx, y = x, y + line_h
                self.draw.text((cx, y), word, font=f, fill=fill)
                bb = self.draw.textbbox((cx, y), word, font=f)
                wid = len(self.words)
                self.words.append({"text": word, "bbox": [int(b) for b in bb], "line": int(y), "glue": glue})
                if ent is not None:
                    self.entities[ent]["word_ids"].append(wid)
                cx += wlen + space
        return y + line_h

    def _new_entity(self, label):
        eid = self._next_eid
        self._next_eid += 1
        self.entities[eid] = {"label": label, "word_ids": []}
        return eid

    # ---- graphics ----
    def paste(self, im, x, y, label=None):
        if im.mode == "RGBA":
            self.img.paste(im, (int(x), int(y)), im)
        else:
            self.img.paste(im, (int(x), int(y)))
        if label:
            self.visuals.append({"label": label, "bbox": [int(x), int(y), int(x + im.width), int(y + im.height)]})

    def hline(self, x0, x1, y, fill=(90, 90, 90), width=1):
        self.draw.line([(x0, y), (x1, y)], fill=fill, width=width)

    def rect(self, box, outline=(90, 90, 90), width=1, fill=None):
        self.draw.rectangle(box, outline=outline, width=width, fill=fill)

    # ---- output ----
    def finalize(self):
        """Reading-order text + char spans for every entity."""
        lines = {}
        for wid, w in enumerate(self.words):
            lines.setdefault(w["line"], []).append(wid)
        order = []
        for ly in sorted(lines):
            order.append(sorted(lines[ly], key=lambda i: self.words[i]["bbox"][0]))

        text_parts, pos = [], 0
        starts, ends = {}, {}
        for li, wids in enumerate(order):
            if li > 0:
                text_parts.append("\n")
                pos += 1
            for j, wid in enumerate(wids):
                if j > 0 and not self.words[wid].get("glue"):
                    text_parts.append(" ")
                    pos += 1
                starts[wid] = pos
                text_parts.append(self.words[wid]["text"])
                pos += len(self.words[wid]["text"])
                ends[wid] = pos
        full = "".join(text_parts)
        reading = [wid for wids in order for wid in wids]
        rank = {wid: r for r, wid in enumerate(reading)}

        word_label = ["O"] * len(self.words)
        entities = []
        for eid, e in self.entities.items():
            wids = sorted(e["word_ids"], key=lambda i: rank[i])
            if not wids:
                continue
            # split into contiguous runs in reading order (safety for odd layouts)
            runs, cur = [], [wids[0]]
            for a, b in zip(wids, wids[1:]):
                if rank[b] == rank[a] + 1:
                    cur.append(b)
                else:
                    runs.append(cur)
                    cur = [b]
            runs.append(cur)
            for run in runs:
                s, t = starts[run[0]], ends[run[-1]]
                bbs = [self.words[i]["bbox"] for i in run]
                entities.append({
                    "label": e["label"], "text": full[s:t], "start": s, "end": t, "group": eid,
                    "bbox": [min(b[0] for b in bbs), min(b[1] for b in bbs),
                             max(b[2] for b in bbs), max(b[3] for b in bbs)],
                })
            for i in wids:
                word_label[i] = e["label"]

        words_out = [{"text": self.words[i]["text"], "bbox": self.words[i]["bbox"],
                      "start": starts[i], "end": ends[i], "label": word_label[i]} for i in reading]
        entities.sort(key=lambda e: e["start"])
        return {"text": full, "words": words_out, "entities": entities, "visuals": list(self.visuals)}
