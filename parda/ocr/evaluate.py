"""Score OCR results against the synthetic ground truth.

  python -m parda.ocr.evaluate --out outputs/ocr_bench

Metrics (per engine, and split by language / script / entity type / document type):
  page_cer            character error rate of the whole page (reading-order sensitive)
  word_recall         % of ground-truth words found anywhere on the page (order-free)
  word_exact          % of words read exactly at the right place (split by script: latin/deva/knda)
  pii_exact           % of PII entities whose text OCR reproduced exactly at the right place
  pii_found           same, ignoring punctuation/spaces/case (what redaction needs: FIND the item)
  pii_near            same, allowing <=10% character errors
  pii_missed          % of PII entities where OCR detected no text at all
  best-of-both        an entity counts as found if ANY engine found it (upper bound for combining engines)
  sec_per_page        speed
The PII numbers matter most: redaction can only hide what OCR can read.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from statistics import median

from ..synth.canvas import script_of
from .run_ocr import find_data, load_sample
from .textnorm import cer, is_punct, loose, norm, squash, substring_distance


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def overlapping(box, segs, min_frac):
    """Segments covering at least min_frac of `box`, in reading order."""
    a = _area(box) or 1
    hits = [s for s in segs if _inter(box, s["bbox"]) / a >= min_frac]
    return sorted(hits, key=lambda s: (s["bbox"][1], s["bbox"][0]))


def reading_order_text(segs):
    """Group segments into lines by vertical position, then left-to-right."""
    if not segs:
        return ""
    h = median(max(1, s["bbox"][3] - s["bbox"][1]) for s in segs)
    items = sorted(segs, key=lambda s: (s["bbox"][1] + s["bbox"][3]) / 2)
    lines, cur, cur_y = [], [], None
    for s in items:
        yc = (s["bbox"][1] + s["bbox"][3]) / 2
        if cur and abs(yc - cur_y) > 0.6 * h:
            lines.append(cur)
            cur = []
        if not cur:
            cur_y = yc
        cur.append(s)
    lines.append(cur)
    return " ".join(" ".join(x["text"] for x in sorted(l, key=lambda s: s["bbox"][0])) for l in lines)


_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def _tokens(text):
    out = []
    for t in norm(text).split():
        t = _EDGE_PUNCT.sub("", t).casefold()
        if t:
            out.append(t)
    return out


def score_page(rec, segs):
    gt_text = norm(rec["text"].replace("\n", " "))
    ocr_text = reading_order_text(segs)
    page = {"page_cer": min(1.0, cer(gt_text, ocr_text))}

    gt_tok, ocr_tok = Counter(_tokens(gt_text)), Counter(_tokens(ocr_text))
    tot = sum(gt_tok.values()) or 1
    page["word_recall"] = sum(min(c, ocr_tok[t]) for t, c in gt_tok.items()) / tot

    words = []
    for w in rec["words"]:
        if is_punct(w["text"]):
            continue
        hay = squash(" ".join(s["text"] for s in overlapping(w["bbox"], segs, 0.5))).casefold()
        needle = squash(w["text"]).casefold()
        d = substring_distance(needle, hay)
        words.append({"script": script_of(w["text"]), "exact": d == 0, "cer": min(1.0, d / max(1, len(needle)))})

    # Entities are scored word by word, each word against the OCR text at ITS OWN location.
    # This handles entities split across lines (addresses) and engines that return lines out of order.
    ents = []
    for e in rec["entities"]:
        ews = [w for w in rec["words"] if w["start"] >= e["start"] and w["end"] <= e["end"]
               and not is_punct(w["text"])]
        if not ews:
            continue
        dist, total, dist_loose, seen, any_hit = 0, 0, 0, [], False
        for w in ews:
            hits = overlapping(w["bbox"], segs, 0.5)
            any_hit |= bool(hits)
            for h in hits:
                if h not in seen:
                    seen.append(h)
            hay_text = " ".join(h["text"] for h in hits)
            needle = squash(w["text"]).casefold()
            dist += substring_distance(needle, squash(hay_text).casefold())
            total += len(needle)
            nl = loose(w["text"])
            if nl:
                dist_loose += substring_distance(nl, loose(hay_text))
        c = min(1.0, dist / max(1, total))
        ents.append({"k": len(ents), "label": e["label"], "script": script_of(e["text"]), "exact": dist == 0,
                     "found": dist_loose == 0, "near": c <= 0.10, "missed": not any_hit, "cer": c,
                     "gt": e["text"], "ocr": " | ".join(h["text"] for h in seen)})
    return page, words, ents


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def evaluate(out, data=None, per_lang=100, seed=7):
    data = find_data(data)
    recs = {r["id"]: r for r in load_sample(data, per_lang, seed, out)}
    report, failures, outcomes = {}, [], {}  # outcomes[engine][(page_id, k)] = entity dict
    for path in sorted(glob.glob(os.path.join(out, "ocr_*.jsonl"))):
        engine = os.path.basename(path)[4:-6]
        pages, words, ents = [], [], []
        n_err = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rec = recs.get(r["id"])
                if rec is None:
                    continue
                if r.get("error"):
                    n_err += 1
                p, w, e = score_page(rec, r["segments"])
                p.update(lang=rec["lang"], doc_type=rec["doc_type"], sec=r["seconds"])
                pages.append(p)
                for x in w:
                    x["lang"] = rec["lang"]
                words += w
                for x in e:
                    x.update(lang=rec["lang"], doc_type=rec["doc_type"], id=rec["id"])
                    outcomes.setdefault(engine, {})[(rec["id"], x["k"])] = x
                    if not x["exact"]:
                        failures.append(dict(x, engine=engine))
                ents += e
        if not pages:
            continue

        def summarise(pg, wd, en):
            return {"pages": len(pg), "page_cer": _mean(p["page_cer"] for p in pg),
                    "word_recall": _mean(p["word_recall"] for p in pg),
                    "word_exact": _mean(x["exact"] for x in wd),
                    "pii_exact": _mean(x["exact"] for x in en), "pii_found": _mean(x["found"] for x in en),
                    "pii_near": _mean(x["near"] for x in en),
                    "pii_missed": _mean(x["missed"] for x in en), "sec_per_page": _mean(p["sec"] for p in pg)}

        rep = {"overall": summarise(pages, words, ents), "errors": n_err, "by_lang": {}, "by_doc_type": {},
               "by_script": {}, "by_label": {}}
        for lang in sorted({p["lang"] for p in pages}):
            rep["by_lang"][lang] = summarise([p for p in pages if p["lang"] == lang],
                                             [x for x in words if x["lang"] == lang],
                                             [x for x in ents if x["lang"] == lang])
        for dt in sorted({p["doc_type"] for p in pages}):
            rep["by_doc_type"][dt] = summarise([p for p in pages if p["doc_type"] == dt], [],
                                               [x for x in ents if x["doc_type"] == dt])
        for sc in sorted({x["script"] for x in words}):
            ws = [x for x in words if x["script"] == sc]
            rep["by_script"][sc] = {"words": len(ws), "word_exact": _mean(x["exact"] for x in ws),
                                    "word_cer": _mean(x["cer"] for x in ws)}
        by_label = defaultdict(list)
        for x in ents:
            key = x["label"] + ("" if x["script"] == "latin" else f" ({x['script']})")
            by_label[key].append(x)
        for k in sorted(by_label):
            xs = by_label[k]
            rep["by_label"][k] = {"count": len(xs), "pii_exact": _mean(x["exact"] for x in xs),
                                  "pii_found": _mean(x["found"] for x in xs),
                                  "pii_near": _mean(x["near"] for x in xs), "pii_missed": _mean(x["missed"] for x in xs)}
        report[engine] = rep

    if len(outcomes) >= 2:  # best-of-both: found by ANY engine (only pages every engine processed)
        common = set.intersection(*(set(o) for o in outcomes.values()))
        merged = []
        for key in common:
            xs = [o[key] for o in outcomes.values()]
            merged.append(dict(xs[0], exact=any(x["exact"] for x in xs), found=any(x["found"] for x in xs)))
        best = {"engines": sorted(outcomes), "entities": len(merged),
                "pii_exact": _mean(x["exact"] for x in merged), "pii_found": _mean(x["found"] for x in merged),
                "by_lang": {}, "by_label": {}}
        for lang in sorted({x["lang"] for x in merged}):
            xs = [x for x in merged if x["lang"] == lang]
            best["by_lang"][lang] = {"pii_exact": _mean(x["exact"] for x in xs), "pii_found": _mean(x["found"] for x in xs)}
        for lab in sorted({x["label"] for x in merged}):
            xs = [x for x in merged if x["label"] == lab]
            best["by_label"][lab] = {"pii_found": _mean(x["found"] for x in xs)}
        report["_best_of_all"] = best

    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2, ensure_ascii=False)
    with open(os.path.join(out, "failures.jsonl"), "w", encoding="utf-8") as f:
        for x in sorted(failures, key=lambda x: -x["cer"]):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    md = to_markdown(report)
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write(md)
    make_charts(report, out)
    return report, md


def _pct(v):
    return "–" if v != v else f"{100 * v:.1f}%"


def to_markdown(report):
    best = report.get("_best_of_all")
    report = {k: v for k, v in report.items() if not k.startswith("_")}
    engines = list(report)
    L = ["# Phase 2 — OCR benchmark on synthetic Indian documents\n",
         "*PII exact* = read character-perfect. *PII found* = read correctly ignoring punctuation/spaces "
         "(what redaction needs). Indic digits (६३२२) count as equal to ASCII digits (6322).\n"]
    L.append("## Overall\n\n| Engine | Pages | Page CER ↓ | Word recall ↑ | PII exact ↑ | PII found ↑ | PII ≤10% CER ↑ "
             "| PII missed ↓ | Sec/page |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for eng, r in report.items():
        o = r["overall"]
        L.append(f"| {eng} | {o['pages']} | {_pct(o['page_cer'])} | {_pct(o['word_recall'])} | {_pct(o['pii_exact'])} | "
                 f"{_pct(o['pii_found'])} | {_pct(o['pii_near'])} | {_pct(o['pii_missed'])} | {o['sec_per_page']:.2f} |")
    if best:
        L.append(f"| **best of {' + '.join(best['engines'])}** | – | – | – | {_pct(best['pii_exact'])} | "
                 f"**{_pct(best['pii_found'])}** | – | – | – |")
    L.append("\n## By language\n\n| Engine | Language | Page CER ↓ | Word recall ↑ | PII exact ↑ | PII found ↑ | PII missed ↓ |")
    L.append("|---|---|---|---|---|---|---|")
    for eng, r in report.items():
        for lang, o in r["by_lang"].items():
            L.append(f"| {eng} | {lang} | {_pct(o['page_cer'])} | {_pct(o['word_recall'])} | {_pct(o['pii_exact'])} | "
                     f"{_pct(o['pii_found'])} | {_pct(o['pii_missed'])} |")
    if best:
        for lang, o in best["by_lang"].items():
            L.append(f"| **best of both** | {lang} | – | – | {_pct(o['pii_exact'])} | **{_pct(o['pii_found'])}** | – |")
    L.append("\n## Word accuracy by script\n\n| Engine | Script | Words | Exact ↑ | Word CER ↓ |")
    L.append("|---|---|---|---|---|")
    for eng, r in report.items():
        for sc, o in r["by_script"].items():
            L.append(f"| {eng} | {sc} | {o['words']} | {_pct(o['word_exact'])} | {_pct(o['word_cer'])} |")
    labels = sorted({k for r in report.values() for k in r["by_label"]})
    L.append("\n## PII by entity type (exact / found)\n")
    L.append("| Entity | " + " | ".join(engines) + (" | best of both (found)" if best else "") + " |")
    L.append("|---|" + "---|" * (len(engines) + (1 if best else 0)))
    for k in labels:
        cells = []
        for e in engines:
            o = report[e]["by_label"].get(k)
            cells.append(f"{_pct(o['pii_exact'])} / {_pct(o['pii_found'])}" if o else "–")
        if best:
            base = k.split(" (")[0]
            cells.append(_pct(best["by_label"].get(base, {}).get("pii_found", float("nan"))) if k == base else "")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("\n## PII found, by document type\n")
    L.append("| Document | " + " | ".join(engines) + " |")
    L.append("|---|" + "---|" * len(engines))
    for dt in sorted({d for r in report.values() for d in r["by_doc_type"]}):
        L.append(f"| {dt} | " + " | ".join(_pct(report[e]["by_doc_type"].get(dt, {}).get("pii_found", float("nan")))
                                           for e in engines) + " |")
    return "\n".join(L) + "\n"


def make_charts(report, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    report = {k: v for k, v in report.items() if not k.startswith("_")}
    engines = list(report)
    if not engines:
        return
    langs = sorted({l for r in report.values() for l in r["by_lang"]})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    width = 0.8 / len(engines)
    for i, e in enumerate(engines):
        xs = [j + i * width for j in range(len(langs))]
        axes[0].bar(xs, [100 * report[e]["by_lang"].get(l, {}).get("pii_found", 0) for l in langs], width, label=e)
        axes[1].bar(xs, [100 * report[e]["by_lang"].get(l, {}).get("page_cer", 0) for l in langs], width, label=e)
    for ax, t in zip(axes, ["PII found (%) ↑  (ignoring punctuation)", "Page character error rate (%) ↓"]):
        ax.set_xticks([j + width * (len(engines) - 1) / 2 for j in range(len(langs))])
        ax.set_xticklabels(langs)
        ax.set_title(t)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "chart_by_language.png"), dpi=120)
    plt.close(fig)

    labels = sorted({k for r in report.values() for k in r["by_label"]})
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(labels))))
    height = 0.8 / len(engines)
    for i, e in enumerate(engines):
        ys = [j + i * height for j in range(len(labels))]
        ax.barh(ys, [100 * report[e]["by_label"].get(k, {}).get("pii_found", 0) for k in labels], height, label=e)
    ax.set_yticks([j + height * (len(engines) - 1) / 2 for j in range(len(labels))])
    ax.set_yticklabels(labels)
    ax.set_xlabel("PII found (%)  (ignoring punctuation)")
    ax.set_xlim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "chart_by_entity.png"), dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ocr_bench")
    ap.add_argument("--data", default=None)
    ap.add_argument("--per_lang", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    _, md = evaluate(a.out, a.data, a.per_lang, a.seed)
    print(md)


if __name__ == "__main__":
    main()
