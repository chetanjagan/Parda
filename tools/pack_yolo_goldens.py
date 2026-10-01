"""Trim the YOLO recording (tools/make_yolo_goldens.py, run on Kaggle) into the web test goldens.

  python tools/pack_yolo_goldens.py --src yolo_goldens/ [--n_pixels 3]

web/test/goldens/yolo.json: per page the size, the final boxes and the raw network output, stored sparsely (only the
candidates whose best class score is above 0.01; the rest are far below the 0.05 threshold and decode to nothing).
web/test/goldens/yolo/: for n_pixels pages the original page (decoded to PNG) and ultralytics' 1024x1024 input (PNG),
to check the browser letterbox pixel by pixel.
"""
import argparse
import base64
import json
import os
import shutil

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--n_pixels", type=int, default=3)
    ap.add_argument("--keep_above", type=float, default=0.01)
    a = ap.parse_args(argv)
    g = json.load(open(os.path.join(a.src, "yolo_goldens.json")))
    out_dir = os.path.join(ROOT, "web", "test", "goldens", "yolo")
    os.makedirs(out_dir, exist_ok=True)
    cases = []
    for i, c in enumerate(g["cases"]):
        raw = np.frombuffer(base64.b64decode(c["raw"]["b64"]), np.float32).reshape(c["raw"]["shape"])
        cols = np.where(raw[0, 4:].max(0) > a.keep_above)[0]
        case = {"image": c["image"], "orig_hw": c["orig_hw"], "gain": c["gain"], "boxes": c["boxes"],
                "raw_shape": c["raw"]["shape"], "raw_cols": cols.tolist(),
                "raw_vals": base64.b64encode(np.ascontiguousarray(raw[0][:, cols]).astype(np.float32).tobytes()).decode()}
        if i < a.n_pixels:
            stem = os.path.splitext(c["image"])[0]
            Image.open(os.path.join(a.src, "images", c["image"])).convert("RGB").save(os.path.join(out_dir, f"{stem}_page.png"))
            shutil.copyfile(os.path.join(a.src, "letterbox", c["letterbox_png"]), os.path.join(out_dir, f"{stem}_input.png"))
            case["page_png"], case["input_png"] = f"yolo/{stem}_page.png", f"yolo/{stem}_input.png"
        cases.append(case)
    out = os.path.join(ROOT, "web", "test", "goldens", "yolo.json")
    json.dump({"meta": g["meta"], "cases": cases}, open(out, "w"))
    sizes = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"{out}: {len(cases)} pages, {sum(len(c['boxes']) for c in cases)} boxes, "
          f"{sum(len(c['raw_cols']) for c in cases)} raw candidates kept, {os.path.getsize(out) / 1024:.0f} KB | "
          f"pixel goldens {sizes / 2 ** 20:.1f} MB")


if __name__ == "__main__":
    main()
