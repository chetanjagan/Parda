"""Draw labelled boxes on a few generated images to check they line up.

Usage: python -m parda.synth.visualize --data data/synth --k 8
Writes data/synth/previews/*.jpg
"""
import argparse
import json
import os

from PIL import Image, ImageDraw

COLORS = {
    "PERSON_NAME": (30, 110, 255), "AADHAAR": (230, 30, 30), "PAN": (255, 140, 0), "PHONE": (0, 170, 80),
    "EMAIL": (0, 150, 150), "ADDRESS": (150, 60, 200), "DOB": (200, 0, 120), "BANK_ACCOUNT": (160, 110, 0),
    "IFSC": (120, 120, 0), "UPI_ID": (0, 120, 200), "GSTIN": (100, 40, 40), "VOTER_ID": (255, 80, 80),
    "UAN": (90, 90, 200), "ABHA": (0, 200, 200), "EMPLOYEE_ID": (140, 140, 140), "MRN": (200, 100, 50),
    "FACE": (255, 0, 255), "SIGNATURE": (0, 0, 0), "QR_CODE": (255, 200, 0), "STAMP": (100, 0, 200),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args(argv)
    out = os.path.join(a.data, "previews")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(a.data, "annotations.jsonl"), encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n >= a.k:
                break
            r = json.loads(line)
            im = Image.open(os.path.join(a.data, r["image"])).convert("RGB")
            d = ImageDraw.Draw(im)
            for w in r["words"]:  # thin grey box around every word
                d.rectangle(w["bbox"], outline=(190, 190, 190))
            for e in r["entities"] + r["visuals"]:
                c = COLORS.get(e["label"], (255, 0, 0))
                d.rectangle(e["bbox"], outline=c, width=3)
                d.text((e["bbox"][0], max(0, e["bbox"][1] - 11)), e["label"], fill=c)
            im.save(os.path.join(out, f"{r['id']:06d}_{r['doc_type']}_{r['lang']}.jpg"), quality=85)
    print("previews written to", out)


if __name__ == "__main__":
    main()
