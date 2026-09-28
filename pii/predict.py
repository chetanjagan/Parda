"""Predictors: text -> [{label, start, end, score}] character spans.

GLiNERPredictor slides a window over long pages (GLiNER has a length limit), maps every
prediction back to page offsets, and merges duplicates from overlapping windows.
"""
from .labels import LABELS, to_code
from .rules import find_rules
from .spans import tokenize, windows


def merge_spans(spans):
    """Merge overlapping/adjacent spans with the same label; keep the best score."""
    out = []
    for sp in sorted(spans, key=lambda s: (s["label"], s["start"], s["end"])):
        if out and out[-1]["label"] == sp["label"] and sp["start"] <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], sp["end"])
            out[-1]["score"] = max(out[-1]["score"], sp["score"])
        else:
            out.append(dict(sp))
    return sorted(out, key=lambda s: (s["start"], s["end"]))


class RulesPredictor:
    name = "rules"

    def predict(self, text):
        return find_rules(text)


class GLiNERPredictor:
    def __init__(self, model, threshold=0.5, labels=None, window=200, stride=150, name="gliner"):
        """model: a loaded GLiNER model, or a path / Hugging Face id to load."""
        if isinstance(model, str):
            model = load_gliner(model)
        self.model, self.threshold = model, threshold
        self.labels = [LABELS[c] for c in (labels or LABELS)]
        self.window, self.stride = window, stride
        self.name = f"{name}@{threshold}"

    def predict(self, text):
        toks = tokenize(text)
        spans = []
        for s, e in windows(len(toks), self.window, self.stride):
            if s >= e:
                continue
            base, stop = toks[s][1], toks[e - 1][2]
            chunk = text[base:stop]
            for ent in self.model.predict_entities(chunk, self.labels, threshold=self.threshold):
                a, b = base + int(ent["start"]), base + int(ent["end"])
                if 0 <= a < b <= len(text):
                    spans.append({"label": to_code(ent["label"]), "start": a, "end": b,
                                  "score": float(ent.get("score", 1.0))})
        return merge_spans(spans)


class UnionPredictor:
    """Model + rules: anything either flags gets redacted."""

    def __init__(self, *preds):
        self.preds = preds
        self.name = " + ".join(p.name for p in preds)

    def predict(self, text):
        return merge_spans([s for p in self.preds for s in p.predict(text)])


class GoldPredictor:
    """Testing only: returns the true spans (a perfect system)."""
    name = "gold"

    def __init__(self):
        self.gold = []

    def predict(self, text):
        return [dict(g, score=1.0) for g in self.gold]


def load_gliner(path_or_id):
    import torch
    from gliner import GLiNER
    m = GLiNER.from_pretrained(path_or_id)
    if torch.cuda.is_available():
        m = m.to("cuda")
    m.eval()
    return m
