"""Golden test cases for the web app's TypeScript port of the pipeline core (web/src/core).

  python tools/make_goldens.py --ocr outputs/ocr_tesseract.jsonl ...   # (re)build: inputs + Python answers
  python tools/make_goldens.py --check                                  # CI: Python still gives the stored answers

The file stores INPUTS and the Python pipeline's OUTPUTS for: ID rules (regex + checksums, Indic digits),
tokenising, windows, span merging, reading order of real OCR pages, text-span -> pixel boxes, gap filling,
pixel rounding and audit rounding. web/test/core.test.ts must reproduce every output exactly, so the browser
app redacts exactly what the Python pipeline (the one the benchmarks measured) redacts.
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from parda.pii.context import filter_spans  # noqa: E402
from parda.pii.predict import merge_spans  # noqa: E402
from parda.pii.rules import ascii_digits, find_rules  # noqa: E402
from parda.pii.spans import build_ocr_text, tokenize, windows  # noqa: E402
from parda.pipeline.boxes import PAD, fill_gaps, pix, span_boxes  # noqa: E402
from parda.synth import ids  # noqa: E402

GOLDEN = os.path.join(ROOT, "web", "test", "goldens", "core.json")

HAND_TEXTS = [
    "Aadhaar 4829 1736 5520 and PAN ABCPK1234F, mobile +91 98450 12345 or 98450-12345.",
    "IFSC SBIN0364507, UPI ravi.kumar@okaxis, email ravi.k@gmail.com, GSTIN 29ABCDE1234F1Z5.",
    "Voter ID ABC1234567, passport K1234567, vehicle KA 01 AB 1234, UAN 100660595745.",
    "ABHA 12-3456-7890-1234; employee EMP12334; xABCPK1234Fx stays; 4829173655201 too long.",
    "आधार २०५३ ९७७९ ०२५१ मोबाइल ९८४५० १२३४५",  # Devanagari digits
    "ಆಧಾರ್ ೨೦೫೩ ೯೭೭೯ ೦೨೫೧ PAN ABCPK1234F",       # Kannada digits
    "superscript ² digits ³ and ①②③",
    "SBINO364507 (OCR read 0 as O) and SBIN0364507.",
    "+91-98450 12345, 9845012345, 0984501234 (not a mobile), 12345 67890",
    "mail a.b-c_d%e+f@sub.example.co.in end. upi 6866839508@cnrb end",
    "",
    "   ",
]


def _rand_text(rng):
    pool = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789  -.,@/+_:" +
            "०१२३४५६७८९" + "೦೧೨೩೪೫೬೭೮೯" + "अकमಅಕಮ")
    parts = []
    for _ in range(rng.randint(3, 12)):
        r = rng.random()
        if r < 0.12:
            parts.append(ids.aadhaar(rng))
        elif r < 0.2:
            parts.append(ids.pan(rng))
        elif r < 0.27:
            parts.append(ids.gstin(rng))
        elif r < 0.33:
            parts.append(ids.fake_12_digit_non_aadhaar(rng))
        else:
            parts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 14))))
    return " ".join(parts)


def _rand_spans(rng, n_text=200):
    labs = ["PAN", "PHONE", "PERSON_NAME"]
    out = []
    for _ in range(rng.randint(0, 9)):
        a = rng.randint(0, n_text - 1)
        out.append({"label": rng.choice(labs), "start": a, "end": a + rng.randint(1, 15), "score": round(rng.random(), 4)})
    return out


def _rand_regions(rng):
    regs = []
    for _ in range(rng.randint(0, 12)):
        x, y = rng.uniform(0, 700), rng.choice([100.0, 100.5, 140.0, 141.0, 300.0])
        h = rng.choice([14.0, 16.0, 20.0])
        regs.append({"kind": rng.choice(["text", "text", "text", "visual"]), "label": rng.choice(["ADDRESS", "PHONE"]),
                     "bbox": [x, y, x + rng.uniform(5, 90), y + h], "score": round(rng.random(), 4), "source": "t"})
    return regs


CONTEXT_TEXTS = [
    "Date of Birth: 10/11/2002\n13/03 SMS CHARGES 20,160\nजन्म तिथि: 01/02/1990\nಜನ್ಮ ದಿನಾಂಕ: 15/08/1985",
    "Aadhaar No: 619752410246\n08/02 UPI/DR 383598852085 13,274\nआधार: 205397790251\nRef 6197 5241 0246",
    "D.O.B 1990-01-02 | dob: 02/03/1991 | Born on 3 Jan 1992 | Period: 01 Jan 2026 to 30 Mar 2026",
    "UID 205397790251, fluid 205397790251 and ಆಧಾರ್ ೨೦೫೩೯೭೭೯೦೨೫೧ then 12,345.67 debit",
    "राशि ०१,२३४ जमा  Mobile 9844512345, 98450-12345",
]


def _context_cases(texts, rng):
    """(text, spans) pairs: the ID-rule spans plus random spans labelled DOB / AADHAAR / PHONE / NAME, many of them
    starting or ending inside numbers, so every context rule is exercised."""
    cases = []
    for t in texts:
        spans = [dict(x) for x in find_rules(t)]
        for _ in range(rng.randint(2, 8)):
            if len(t) < 3:
                break
            a = rng.randrange(0, len(t) - 1)
            b = min(len(t), a + rng.randint(1, 16))
            spans.append({"label": rng.choice(["DOB", "AADHAAR", "PHONE", "PERSON_NAME"]), "start": a, "end": b,
                          "score": round(rng.random(), 3)})
        cases.append([t, spans])
    return cases


def _edge_regions():
    """Boxes exactly at the gap-fill limit (gap == 2.5 x height) and at the same-line limit (overlap == half)."""
    R = lambda lab, b, kind="text": {"kind": kind, "label": lab, "bbox": b, "score": 0.5, "source": "t"}  # noqa: E731
    return [
        [R("ADDRESS", [0.0, 100.0, 40.0, 120.0]), R("ADDRESS", [90.0, 100.0, 130.0, 120.0])],     # gap 50 == 2.5 x 20
        [R("ADDRESS", [0.0, 100.0, 40.0, 120.0]), R("ADDRESS", [90.01, 100.0, 130.0, 120.0])],    # just over
        [R("PHONE", [0.0, 100.0, 40.0, 120.0]), R("PHONE", [45.0, 110.0, 80.0, 130.0])],          # overlap 10 == half
        [R("PHONE", [0.0, 100.0, 40.0, 120.0]), R("PHONE", [45.0, 110.01, 80.0, 130.01])],        # just under half
        [R("PHONE", [0.0, 100.0, 40.0, 120.0]), R("FACE", [45.0, 100.0, 80.0, 120.0], "visual"),
         R("PHONE", [60.0, 100.0, 90.0, 120.0]), R("ADDRESS", [95.0, 100.0, 120.0, 120.0])],
    ]


def py_round(x, nd):
    return round(float(x), nd)


def compute(inputs):
    """All Python answers for the stored inputs."""
    out = {"ascii_digits": [ascii_digits(t) for t in inputs["texts"]],
           "rules": [find_rules(t) for t in inputs["texts"]],
           "tokenize": [[list(t) for t in tokenize(t)] for t in inputs["texts"]],
           "windows": [windows(n, w, s) for n, w, s in inputs["windows"]],
           "merge_spans": [merge_spans(s) for s in inputs["span_sets"]],
           "fill_gaps": [fill_gaps(r) for r in inputs["region_sets"]],
           "pix": [list(pix(b, W, H, p)) for b, W, H, p in inputs["pix"]],
           "round": [[py_round(x, 1), py_round(x, 3)] for x in inputs["floats"]],
           "pad": PAD,
           "context": [filter_spans(t, sps) for t, sps in inputs.get("context", [])]}
    pages = []
    for segs in inputs["pages"]:
        text, ordered, offs = build_ocr_text(segs)
        order = [segs.index(s) for s in ordered]
        rules = find_rules(text)
        boxes = [span_boxes(r["start"], r["end"], ordered, offs) for r in rules]
        partial = []  # spans that cover only part of a segment (the line-level OCR case)
        for i, (a, b) in enumerate(offs[:12]):
            if b - a >= 4:
                partial.append([a + 1, b - 1, span_boxes(a + 1, b - 1, ordered, offs)])
        regs = [{"kind": "text", "label": r["label"], "bbox": bx, "score": 1.0, "source": "tesseract"}
                for r, bxs in zip(rules, boxes) for bx in bxs]
        pages.append({"text": text, "order": order, "offs": [list(o) for o in offs], "rules": rules,
                      "span_boxes": boxes, "partial": partial, "fill_gaps": fill_gaps(regs)})
    out["pages"] = pages
    return out


def build(ocr_files, n_pages=28, seed=7):
    rng = random.Random(seed)
    pages = []
    for f in ocr_files:
        rows = [json.loads(ln) for ln in open(f, encoding="utf-8")]
        rng.shuffle(rows)
        for r in rows[: n_pages // max(1, len(ocr_files))]:
            segs = [{"text": s["text"], "bbox": [round(float(v), 2) for v in s["bbox"]]} for s in r.get("segments") or []]
            if segs:
                pages.append(segs)
    for slope in (0.0, 0.03, -0.05):  # punctuation marks at many distances from their line (0.4-0.9 x height)
        page = []
        for li in range(5):
            for wi in range(6):
                x = 40 + wi * 110
                y = 100 + li * 40 + slope * x
                page.append({"text": f"t{li}{wi}", "bbox": [x, y, x + 60, y + 20]})
                d = (0.4 + 0.05 * ((li * 6 + wi) % 11)) * 20  # mark centre this far below the word centre
                page.append({"text": ",", "bbox": [x + 66, y + 10 + d - 3, x + 70, y + 10 + d + 3]})
        rng.shuffle(page)
        pages.append(page)
    # line order decided by which words count as "tall" (>= 0.6 x height) in the line's position: line A = a word at
    # y-centre 110 + a 0.55-height mark at 125.9 (joins A); line B = a word at 115 that overlaps A horizontally.
    # Counting only tall words, A (110) comes before B (115); counting the mark too would put A at 117.95, after B.
    order_page = [{"text": "alpha", "bbox": [0, 100, 60, 120]}, {"text": "beta", "bbox": [40, 105, 100, 125]},
                  {"text": "~", "bbox": [66, 120.4, 72, 131.4]}]
    order_page += [{"text": f"z{k}", "bbox": [0, 400 + 60 * k, 60, 420 + 60 * k]} for k in range(4)]
    pages.append(order_page)
    tilted = []  # a synthetic tilted page: 6 lines x 8 words, slope 0.04
    for li in range(6):
        for wi in range(8):
            x = 40 + wi * 90
            y = 100 + li * 30 + 0.04 * x
            tilted.append({"text": f"w{li}{wi}", "bbox": [x, y, x + 70, y + 16]})
    rng.shuffle(tilted)
    pages.append(tilted)
    texts = HAND_TEXTS + [_rand_text(rng) for _ in range(250)]
    inputs = {
        "texts": texts,
        "windows": [[n, w, s] for n in (0, 1, 5, 199, 200, 201, 350, 351, 1000) for w, s in ((200, 150), (60, 40))],
        "span_sets": [_rand_spans(rng) for _ in range(120)],
        "region_sets": [_rand_regions(rng) for _ in range(120)] + _edge_regions(),
        "pix": [[[rng.uniform(-5, 800), rng.uniform(-5, 1100), rng.uniform(0, 900), rng.uniform(0, 1200)],
                 827, 1169, rng.choice([0, 3, 6])] for _ in range(150)]
               + [[[99.9999, 10.0001, 200.0, 20.5], 827, 1169, 0]],
        "floats": [rng.uniform(0, 1000) for _ in range(200)] + [12.25, 0.125, 2.675, 0.5, 1.5, 2.5, 1003.45, 0.0005],
        "pages": pages,
        "context": _context_cases(CONTEXT_TEXTS * 6 + texts[:120], rng),
    }
    return {"about": "inputs + Python answers; see tools/make_goldens.py", "inputs": inputs, "expected": compute(inputs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ocr", nargs="*", default=[], help="OCR jsonl files to sample real pages from (build mode)")
    ap.add_argument("--check", action="store_true", help="recompute from the stored inputs and compare")
    a = ap.parse_args(argv)
    if a.check:
        g = json.load(open(GOLDEN, encoding="utf-8"))
        now = json.loads(json.dumps(compute(g["inputs"])))
        bad = [k for k in g["expected"] if now.get(k) != g["expected"][k]]
        if bad:
            print("Python answers changed for:", bad, "-> rebuild the goldens and re-run the web tests")
            sys.exit(1)
        print("goldens match the Python pipeline:", len(g["expected"]), "groups")
        return
    g = build(a.ocr)
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    json.dump(g, open(GOLDEN, "w", encoding="utf-8"), ensure_ascii=False)
    e = g["expected"]
    print(f"wrote {GOLDEN}: {len(e['rules'])} texts, {sum(len(r) for r in e['rules'])} rule hits, "
          f"{len(e['pages'])} pages, {os.path.getsize(GOLDEN) // 1024} KB")


if __name__ == "__main__":
    main()
