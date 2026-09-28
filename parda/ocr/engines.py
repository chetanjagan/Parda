"""OCR engines behind one interface.

engine.run(image_path, record) -> list of segments:
    {"text": str, "bbox": [x0, y0, x1, y1], "score": float}
A segment can be a word (Tesseract) or a line/phrase (EasyOCR, PaddleOCR); the
evaluator handles both.

`record["lang"]` ("en" / "hi-en" / "kn-en") picks the language model. In the real
product we'd detect the script automatically; for the benchmark we use the known language.
"""
import os
import random
import subprocess


def _xyxy(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


# ---------------------------------------------------------------- Tesseract (CPU)
class TesseractEngine:
    name = "tesseract"
    LANG = {"en": "eng", "hi-en": "hin+eng", "kn-en": "kan+eng"}

    def __init__(self, psm: int = 3):
        self.psm = psm
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
        self.available = set(out.stdout.split()[1:]) if out.returncode == 0 else set()
        if not self.available:
            raise RuntimeError("tesseract not installed (apt-get install tesseract-ocr)")

    def _langs(self, lang):
        want = self.LANG.get(lang, "eng").split("+")
        have = [l for l in want if l in self.available]
        return "+".join(have) if have else "eng"

    def run(self, path, rec):
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        cmd = ["tesseract", path, "stdout", "-l", self._langs(rec["lang"]), "--psm", str(self.psm), "tsv"]
        out = subprocess.run(cmd, capture_output=True, text=True, env=env)
        segs = []
        for line in out.stdout.splitlines()[1:]:
            p = line.split("\t")
            if len(p) < 12 or p[0] != "5" or not p[11].strip():
                continue
            x, y, w, h = map(int, p[6:10])
            segs.append({"text": p[11], "bbox": [x, y, x + w, y + h], "score": float(p[10]) / 100})
        return segs


# ---------------------------------------------------------------- EasyOCR (GPU)
class EasyOCREngine:
    name = "easyocr"
    # EasyOCR can only mix a non-Latin script with English, so one reader per document language
    LANG = {"en": ["en"], "hi-en": ["hi", "en"], "kn-en": ["kn", "en"]}

    def __init__(self, gpu: bool = True):
        import easyocr  # noqa: F401
        self.gpu, self.readers = gpu, {}

    def _reader(self, lang):
        if lang not in self.readers:
            import easyocr
            self.readers[lang] = easyocr.Reader(self.LANG.get(lang, ["en"]), gpu=self.gpu, verbose=False)
        return self.readers[lang]

    def run(self, path, rec):
        res = self._reader(rec["lang"]).readtext(path, detail=1, paragraph=False)
        return [{"text": t, "bbox": _xyxy(box), "score": float(c)} for box, t, c in res if str(t).strip()]


# ---------------------------------------------------------------- PaddleOCR (GPU) — optional
class PaddleEngine:
    """Supports PaddleOCR 3.x (.predict) and 2.x (.ocr). Kannada model code is 'ka'."""
    name = "paddleocr"
    LANG = {"en": "en", "hi-en": "hi", "kn-en": "ka"}

    def __init__(self):
        import paddleocr  # noqa: F401
        self.models, self.failed = {}, {}

    def _model(self, lang):
        code = self.LANG.get(lang, "en")
        if code in self.failed:
            raise RuntimeError(self.failed[code])
        if code not in self.models:
            from paddleocr import PaddleOCR
            try:
                try:  # 3.x
                    self.models[code] = PaddleOCR(lang=code, use_doc_orientation_classify=False,
                                                  use_doc_unwarping=False, use_textline_orientation=False)
                except (TypeError, ValueError):  # 2.x
                    self.models[code] = PaddleOCR(lang=code, use_angle_cls=False, show_log=False)
            except Exception as e:
                self.failed[code] = f"PaddleOCR could not load lang '{code}': {e}"
                raise RuntimeError(self.failed[code])
        return self.models[code]

    def run(self, path, rec):
        m = self._model(rec["lang"])
        segs = []
        if hasattr(m, "predict"):  # 3.x
            for res in m.predict(path):
                d = res
                j = getattr(res, "json", None)
                if isinstance(j, dict):
                    d = j.get("res", j)
                texts, scores = list(d.get("rec_texts", [])), list(d.get("rec_scores", []))
                boxes = d.get("rec_boxes")
                polys = d.get("rec_polys")
                for i, t in enumerate(texts):
                    if not str(t).strip():
                        continue
                    if boxes is not None and len(boxes) > i:
                        bb = [int(v) for v in list(boxes[i])[:4]]
                    else:
                        bb = _xyxy(polys[i])
                    segs.append({"text": t, "bbox": bb, "score": float(scores[i]) if i < len(scores) else 1.0})
        else:  # 2.x
            result = m.ocr(path, cls=False)
            for line in (result[0] if result and result[0] else []):
                box, (t, s) = line
                if str(t).strip():
                    segs.append({"text": t, "bbox": _xyxy(box), "score": float(s)})
        return segs


# ---------------------------------------------------------------- Oracle (testing only)
class OracleEngine:
    """Returns the ground-truth words grouped into lines, optionally with character noise.
    noise=0 must give perfect scores -> proves the evaluator is correct."""
    name = "oracle"

    def __init__(self, noise: float = 0.0, seed: int = 0):
        self.noise, self.rng = noise, random.Random(seed)

    def _corrupt(self, w):
        if self.noise <= 0:
            return w
        out = []
        for ch in w:
            r = self.rng.random()
            if r < self.noise / 3:
                continue  # deletion
            if r < 2 * self.noise / 3:
                out.append(self.rng.choice("0123456789abcdefghijklmnopqrstuvwxyz"))  # substitution
                continue
            out.append(ch)
        return "".join(out) or w

    def run(self, path, rec):
        # follow the true line breaks of the ground-truth text; keep the true gap (" " or "") between words
        text, lines, prev_end = rec["text"], [], None
        for w in rec["words"]:
            gap = "" if prev_end is None else text[prev_end:w["start"]]
            if prev_end is None or "\n" in gap:
                lines.append([])
                gap = ""
            lines[-1].append((gap, w))
            prev_end = w["end"]
        segs = []
        for ln in lines:
            bbs = [w["bbox"] for _, w in ln]
            segs.append({"text": "".join(g + self._corrupt(w["text"]) for g, w in ln),
                         "bbox": [min(b[0] for b in bbs), min(b[1] for b in bbs),
                                  max(b[2] for b in bbs), max(b[3] for b in bbs)], "score": 1.0})
        return segs


ENGINES = {"tesseract": TesseractEngine, "easyocr": EasyOCREngine, "paddleocr": PaddleEngine,
           "oracle": OracleEngine}


def make_engine(name: str, **kw):
    return ENGINES[name](**kw)
