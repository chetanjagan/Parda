"""Compare two OCR outputs of the same pages against the TRUE page text (same scoring as Phase 2's OCR benchmark).

  python tools/compare_ocr.py --data <pages with annotations.jsonl> --ocr native=cache/ocr_tesseract.jsonl \
         --ocr browser=tjs/ocr_tesseract.jsonl --out outputs/ocr_compare.md

Per OCR and language: page character error rate, word recall, and the share of PII items read exactly / found
(loose match) / missed entirely. PII found is what limits redaction: text PII that OCR never reads cannot be hidden.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from parda.ocr.evaluate import score_page  # noqa: E402


def load(path):
    return {o["id"]: o for o in map(json.loads, open(path, encoding="utf-8"))}


def compare(data, ocrs):
    want = set.intersection(*[set(o) for o in ocrs.values()])
    recs = {}
    with open(os.path.join(data, "annotations.jsonl"), encoding="utf-8") as f:
        for ln in f:  # the Phase 1 file has 30,000 pages: keep only the compared ones
            j = ln.rfind('"id": ')
            if j >= 0 and json.loads("{" + ln[j:])["id"] not in want:
                continue
            r = json.loads(ln)
            recs[r["id"]] = r
    ids = sorted(want & set(recs))
    rows = {}
    for name, o in ocrs.items():
        acc = defaultdict(lambda: defaultdict(list))
        for i in ids:
            r = recs[i]
            page, _, ents = score_page(r, o[i].get("segments") or [])
            for lang in (r["lang"], "all"):
                a = acc[lang]
                a["cer"].append(page["page_cer"])
                a["recall"].append(page["word_recall"])
                for e in ents:
                    a["exact"].append(e["exact"])
                    a["found"].append(e["found"])
                    a["missed"].append(e["missed"])
                a["sec"].append(o[i].get("sec", 0.0))
        mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
        rows[name] = {lang: {k: mean([float(x) for x in v]) for k, v in a.items()} | {"pii": len(a["exact"])}
                      for lang, a in acc.items()}
    return {"pages": len(ids), "rows": rows}


def report_md(res, title):
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    langs = sorted({lang for r in res["rows"].values() for lang in r} - {"all"}) + ["all"]
    L = [f"### {title} ({res['pages']} pages)", "",
         "| OCR | Language | Page error rate ↓ | Word recall ↑ | PII read exactly ↑ | PII found ↑ | PII missed ↓ | s/page |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in res["rows"].items():
        for lang in langs:
            if lang in r:
                x = r[lang]
                L.append(f"| {name} | {lang} | {pct(x['cer'])} | {pct(x['recall'])} | {pct(x['exact'])} | {pct(x['found'])} | "
                         f"{pct(x['missed'])} | {x['sec']:.2f} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ocr", action="append", required=True, help="name=path/to/ocr.jsonl (repeat)")
    ap.add_argument("--title", default="OCR compared with the true page text")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ocrs = {s.split("=", 1)[0]: load(s.split("=", 1)[1]) for s in a.ocr}
    res = compare(a.data, ocrs)
    md = report_md(res, a.title)
    open(a.out, "w", encoding="utf-8").write(md)
    json.dump(res, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=2)
    print(md, flush=True)
    return res


if __name__ == "__main__":
    main()
