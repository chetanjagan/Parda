"""Trim the EasyOCR recording (tools/easyocr_onnx.py record, notebook 14) into the web test goldens.

  python tools/pack_easyocr_goldens.py --src easyocr_goldens/ [--jpg_dir first_recording/pages]

web/test/goldens/easyocr.json + easyocr/: per page the grouped boxes and readtext's result; EasyOCR's own gray page
(decoded from the JPEG, as EasyOCR does); for some pages the detector's score maps (float32) and the recognizer's
inputs (uint8) and outputs (float32), gzip-compressed. Every stage of the TypeScript port is checked against these.
"""
import argparse
import gzip
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET_PAGES = ("en_scan1475", "hi-en_scan672", "kn-en_photo2")  # score maps kept (float32 maps are large)
REC_PAGES = ("en_scan1475", "kn-en_scan282", "hi-en_scan672")  # every recognizer call kept


def gz(path, arr):
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(np.ascontiguousarray(arr).tobytes())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--jpg_dir", required=True, help="the page JPEGs (EasyOCR reads its gray image from these)")
    a = ap.parse_args(argv)
    g = json.load(open(os.path.join(a.src, "easyocr_goldens.json")))
    out = os.path.join(ROOT, "web", "test", "goldens", "easyocr")
    os.makedirs(out, exist_ok=True)
    cases = []
    for c in g["cases"]:
        stem = f"{c['lang']}_{c['id']}"
        z = np.load(os.path.join(a.src, "pages", c["npz"]))
        grey = cv2.imread(os.path.join(a.jpg_dir, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
        assert grey.shape == tuple(c["page_hw"]), (stem, grey.shape, c["page_hw"])
        cv2.imwrite(os.path.join(out, stem + "_gray.png"), grey)
        case = {k: c[k] for k in ("id", "lang", "page_hw", "raw_boxes", "horizontal_list", "free_list", "readtext")}
        case["gray_png"] = stem + "_gray.png"
        if stem in DET_PAGES:
            gz(os.path.join(out, stem + "_maps.f32.gz"), z["det_out"].astype(np.float32))
            case["maps"] = {"file": stem + "_maps.f32.gz", "shape": list(z["det_out"].shape)}
        if stem in REC_PAGES:
            ins = sorted([k for k in z.files if k.startswith("rec_in_")], key=lambda k: int(k.split("_")[-1]))
            outs = [k.replace("rec_in_", "rec_out_") for k in ins]
            gz(os.path.join(out, stem + "_rec_in.u8.gz"), np.concatenate([z[k].ravel() for k in ins]))
            gz(os.path.join(out, stem + "_rec_out.f32.gz"), np.concatenate([z[k].astype(np.float32).ravel() for k in outs]))
            case["rec"] = {"in_file": stem + "_rec_in.u8.gz", "out_file": stem + "_rec_out.f32.gz",
                           "in_shapes": [list(z[k].shape) for k in ins], "out_shapes": [list(z[k].shape) for k in outs]}
        cases.append(case)
    meta = {"readers": {k: {"character": v["character"], "lang_char": v["lang_char"]} for k, v in g["meta"]["readers"].items()},
            "defaults": g["meta"]["readtext_defaults"], "easyocr_version": g["meta"].get("easyocr_version")}
    json.dump({"meta": meta, "cases": cases}, open(os.path.join(ROOT, "web", "test", "goldens", "easyocr.json"), "w"),
              ensure_ascii=False)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"{len(cases)} pages | {sum(len(c['readtext']) for c in cases)} text segments | files {size / 2 ** 20:.1f} MB")


if __name__ == "__main__":
    main()
