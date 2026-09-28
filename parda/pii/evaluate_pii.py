"""Evaluate PII detectors on the 300 benchmark pages (Tesseract text of each page).

  python -m parda.pii.evaluate_pii --ocr outputs/bench/ocr_tesseract.jsonl --systems rules,base,ft,ft+rules \\
      --base urchade/gliner_multi_pii-v1 --ft outputs/gliner/final --thresholds 0.3,0.5 --out outputs/pii_eval

Metrics (all ignore spaces/punctuation, i.e. count only letters/marks/digits):
  fully_redacted   % of ALL true PII items completely covered by predictions  <- headline
                   (items OCR never produced text for count as NOT redacted)
  char_recall      % of PII characters (in OCR text) covered
  over_redaction   % of redacted characters that were NOT PII (lower = less document damage)
  typed_recall     % of PII items found with the correct type (>=50% overlap)
  span_precision   % of predicted spans that are mostly PII
"""
import argparse
import json
import os
from collections import defaultdict

from ..ocr.run_ocr import benchmark_ids, find_data
from .predict import GLiNERPredictor, RulesPredictor, UnionPredictor, load_gliner
from .spans import content_positions, transfer_labels


class Acc:
    def __init__(self):
        self.items = self.full = self.typed = 0
        self.gold_chars = self.gold_cov = self.pred_chars = self.pred_bad = 0
        self.spans = self.spans_ok = 0

    def row(self):
        d = lambda a, b: a / b if b else float("nan")  # noqa: E731
        return {"items": self.items, "fully_redacted": d(self.full, self.items),
                "char_recall": d(self.gold_cov, self.gold_chars),
                "over_redaction": d(self.pred_bad, self.pred_chars),
                "typed_recall": d(self.typed, self.items), "span_precision": d(self.spans_ok, self.spans)}


def score_page(text, gold, missed, preds, accs):
    """accs: list of Acc objects to update (overall, by lang, ...); label-level handled via dict."""
    pred_pos = set()
    for p in preds:
        pred_pos |= content_positions(text, p["start"], p["end"])
    gold_pos = set()
    per_entity = defaultdict(set)
    for g in gold:
        pos = content_positions(text, g["start"], g["end"])
        per_entity[g["entity"]] |= pos
        gold_pos |= pos
    labels_by_entity = {g["entity"]: g["label"] for g in gold}
    for m in missed:
        labels_by_entity[m["entity"]] = m["label"]

    results = []  # (label, fully, typed)
    for ei, lab in labels_by_entity.items():
        pos = per_entity.get(ei, set())
        fully = bool(pos) and pos <= pred_pos
        typed = False
        if pos:
            for p in preds:
                if p["label"] == lab and len(pos & content_positions(text, p["start"], p["end"])) >= 0.5 * len(pos):
                    typed = True
                    break
        results.append((lab, fully, typed, len(pos), len(pos & pred_pos)))

    for acc, filt in accs:
        for lab, fully, typed, n, cov in results:
            if filt is None or filt == lab:
                acc.items += 1
                acc.full += fully
                acc.typed += typed
                acc.gold_chars += n
                acc.gold_cov += cov
        if filt is None:
            acc.pred_chars += len(pred_pos)
            acc.pred_bad += len(pred_pos - gold_pos)
            for p in preds:
                pp = content_positions(text, p["start"], p["end"])
                if pp:
                    acc.spans += 1
                    acc.spans_ok += len(pp & gold_pos) >= 0.5 * len(pp)


def _set_gold(p, gold):
    """Testing helper: hand the true spans to any GoldPredictor, including inside a UnionPredictor."""
    if hasattr(p, "gold"):
        p.gold = gold
    for q in getattr(p, "preds", ()):
        _set_gold(q, gold)


def build_systems(names, base, ft, thresholds):
    systems, cache = [], {}

    def model(path):
        if path not in cache:
            cache[path] = load_gliner(path)
        return cache[path]

    rules = RulesPredictor()
    for n in names:
        if n == "rules":
            systems.append(rules)
        elif n in ("base", "ft", "base+rules", "ft+rules"):
            path = base if n.startswith("base") else ft
            if not path:
                print(f"skipping {n}: no model path given")
                continue
            for t in thresholds:
                g = GLiNERPredictor(model(path), threshold=t, name="gliner_" + n.split("+")[0])
                systems.append(UnionPredictor(g, rules) if n.endswith("+rules") else g)
    return systems


def evaluate(data, ocr_path, systems, out):
    bench = benchmark_ids(data)  # only ever score the fixed benchmark pages
    recs = {}
    with open(os.path.join(data, "annotations.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in bench:
                recs[r["id"]] = r
    pages = []
    with open(ocr_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["id"] in recs and not o.get("error"):
                pages.append((recs[o["id"]], o["segments"]))
    if not pages:
        raise SystemExit("no benchmark pages found in the OCR file")
    report = {}
    for sysm in systems:
        overall, by_lang, by_label = Acc(), defaultdict(Acc), defaultdict(Acc)
        for rec, segs in pages:
            text, gold, missed = transfer_labels(rec, segs)
            _set_gold(sysm, gold)
            preds = sysm.predict(text)
            labels = {g["label"] for g in gold} | {m["label"] for m in missed}
            accs = [(overall, None), (by_lang[rec["lang"]], None)] + [(by_label[l], l) for l in labels]
            score_page(text, gold, missed, preds, accs)
        report[sysm.name] = {"overall": overall.row(), "by_lang": {k: v.row() for k, v in sorted(by_lang.items())},
                             "by_label": {k: v.row() for k, v in sorted(by_label.items())}}
        print(f"{sysm.name:32s} fully_redacted={report[sysm.name]['overall']['fully_redacted']:.3f}", flush=True)
    report["_meta"] = {"pages": len(pages), "ocr": ocr_path}
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, "report_pii.json"), "w"), indent=2)
    md = to_markdown(report)
    open(os.path.join(out, "report_pii.md"), "w", encoding="utf-8").write(md)
    return report, md


def _p(v):
    return "–" if v != v else f"{100 * v:.1f}%"


def to_markdown(report):
    meta = report.get("_meta", {})
    systems = [k for k in report if not k.startswith("_")]
    L = [f"# Phase 3 — PII detection on {meta.get('pages', '?')} benchmark pages (Tesseract OCR text)\n",
         "*Fully redacted* counts every true PII item, including ones OCR never read. "
         "Spaces and punctuation are ignored.\n",
         "## Overall\n", "| System | Fully redacted ↑ | PII chars covered ↑ | Over-redaction ↓ | Correct type ↑ "
         "| Span precision ↑ |", "|---|---|---|---|---|---|"]
    for s in systems:
        o = report[s]["overall"]
        L.append(f"| {s} | **{_p(o['fully_redacted'])}** | {_p(o['char_recall'])} | {_p(o['over_redaction'])} | "
                 f"{_p(o['typed_recall'])} | {_p(o['span_precision'])} |")
    L += ["\n## Fully redacted, by language\n", "| System | " + " | ".join(
        sorted({l for s in systems for l in report[s]["by_lang"]})) + " |"]
    langs = sorted({l for s in systems for l in report[s]["by_lang"]})
    L.append("|---|" + "---|" * len(langs))
    for s in systems:
        L.append(f"| {s} | " + " | ".join(_p(report[s]["by_lang"].get(l, {}).get("fully_redacted", float("nan")))
                                          for l in langs) + " |")
    labels = sorted({l for s in systems for l in report[s]["by_label"]})
    L += ["\n## Fully redacted, by PII type\n", "| Type | " + " | ".join(systems) + " |",
          "|---|" + "---|" * len(systems)]
    for lab in labels:
        L.append(f"| {lab} | " + " | ".join(_p(report[s]["by_label"].get(lab, {}).get("fully_redacted", float("nan")))
                                            for s in systems) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--ocr", required=True, help="ocr_tesseract.jsonl of the benchmark pages")
    ap.add_argument("--systems", default="rules,base,ft,ft+rules")
    ap.add_argument("--base", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--ft", default=None)
    ap.add_argument("--thresholds", default="0.5")
    ap.add_argument("--out", default="outputs/pii_eval")
    a = ap.parse_args(argv)
    systems = build_systems(a.systems.split(","), a.base, a.ft, [float(t) for t in a.thresholds.split(",")])
    _, md = evaluate(find_data(a.data), a.ocr, systems, a.out)
    print(md)


if __name__ == "__main__":
    main()
