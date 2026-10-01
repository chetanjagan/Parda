"""Golden cases for the browser port of the YOLO detector (web app milestone M3). Runs the SHIPPED model
(parda-yolo.onnx, through ultralytics, exactly as parda.vision.detectors.YoloDetector does) on real benchmark pages
and records, per page, every step ultralytics takes around the network:

  image          the page itself (copied, so the browser test can start from the same pixels)
  letterbox      the network input ultralytics built (resize + gray padding to 1024x1024), as an RGB PNG
  raw            the network output (float32, base64)
  boxes          the final boxes (xyxy in page pixels, class, confidence) after filtering and overlap removal
  steps          ratio / padding ultralytics used, and its settings (conf, iou, max_det, agnostic, imgsz)

plus ultralytics' own source code for these steps, so the port follows the real code.

  python tools/make_yolo_goldens.py --yolo outputs/onnx/yolo/parda-yolo.onnx --images data/yolo/images/bench --n 8 \
         --out outputs/yolo_goldens
"""
import argparse
import base64
import glob
import json
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONF, IOU, IMGSZ = 0.05, 0.5, 1024  # what parda's YoloDetector passes to ultralytics


def b64_float32(arr):
    import numpy as np
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def pick_images(folder, labels_dir, n):
    """Pages with as many different visual classes as possible first (faces, signatures, QR codes, stamps)."""
    scored = []
    for p in sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))):
        lab = os.path.join(labels_dir, os.path.splitext(os.path.basename(p))[0] + ".txt") if labels_dir else None
        classes = set()
        if lab and os.path.exists(lab):
            classes = {ln.split()[0] for ln in open(lab) if ln.strip()}
        scored.append((-len(classes), p))
    scored.sort()
    return [p for _, p in scored[:n]]


def record(yolo_path, images, out):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO
    model = YOLO(yolo_path)
    model.predict(images[0], conf=CONF, iou=IOU, imgsz=IMGSZ, device="cpu", verbose=False)  # sets up the predictor
    pred = model.predictor
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "letterbox"), exist_ok=True)
    cases = []
    for p in images:
        name = os.path.basename(p)
        shutil.copyfile(p, os.path.join(out, "images", name))
        im0 = cv2.imread(p)  # what ultralytics loads (BGR)
        with torch.no_grad():
            im = pred.preprocess([im0])  # letterbox + BGR->RGB + /255 + CHW: exactly the network input
            raw = pred.inference(im)
            raw0 = raw[0] if isinstance(raw, (list, tuple)) else raw
            results = pred.postprocess(raw, im, [im0])
        lb = (im[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(lb).save(os.path.join(out, "letterbox", os.path.splitext(name)[0] + ".png"))
        r = results[0]
        boxes = [{"xyxy": [float(v) for v in b], "cls": int(c), "label": r.names[int(c)], "conf": float(s)}
                 for b, c, s in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.cls.cpu().numpy().tolist(),
                                    r.boxes.conf.cpu().numpy().tolist())]
        # what YoloDetector.detect returns for the same page (the pipeline's view), as a cross-check
        direct = model.predict(p, conf=CONF, iou=IOU, imgsz=IMGSZ, device="cpu", verbose=False)[0]
        same = np.allclose(direct.boxes.xyxy.cpu().numpy(), r.boxes.xyxy.cpu().numpy(), atol=1e-3)
        h, w = im0.shape[:2]
        H, W = im.shape[2], im.shape[3]
        gain = min(H / h, W / w)
        cases.append({"image": name, "orig_hw": [h, w], "input_hw": [H, W], "gain": gain,
                      "letterbox_png": os.path.splitext(name)[0] + ".png",
                      "raw": b64_float32(torch.as_tensor(raw0).cpu().numpy()), "boxes": boxes,
                      "same_as_predict": bool(same)})
        print(f"{name}: {w}x{h} -> {W}x{H}, raw {list(torch.as_tensor(raw0).shape)}, {len(boxes)} boxes, "
              f"same as predict(): {same}", flush=True)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    settings = {"conf": CONF, "iou": IOU, "imgsz": IMGSZ, "max_det": getattr(pred.args, "max_det", None),
                "agnostic_nms": getattr(pred.args, "agnostic_nms", None), "classes": getattr(pred.args, "classes", None),
                "names": {int(k): v for k, v in names.items()}}
    return cases, settings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo", required=True, help="parda-yolo.onnx")
    ap.add_argument("--images", required=True, help="folder of page images (e.g. the YOLO bench split)")
    ap.add_argument("--labels", default=None, help="matching YOLO labels folder (to pick varied pages)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import ultralytics
    os.makedirs(a.out, exist_ok=True)
    images = pick_images(a.images, a.labels, a.n)
    cases, settings = record(a.yolo, images, a.out)
    meta = {"ultralytics_version": ultralytics.__version__, "settings": settings, "n_cases": len(cases),
            "n_boxes": sum(len(c["boxes"]) for c in cases)}
    json.dump({"meta": meta, "cases": cases}, open(os.path.join(a.out, "yolo_goldens.json"), "w"))
    src = os.path.dirname(ultralytics.__file__)  # ultralytics' own code for these steps
    keep = ("engine/predictor.py", "models/yolo/detect/predict.py", "data/augment.py", "utils/ops.py", "utils/nms.py",
            "nn/autobackend.py", "nn/backends/onnx.py", "engine/results.py")
    with zipfile.ZipFile(os.path.join(a.out, "ultralytics_src.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for rel in keep:
            p = os.path.join(src, rel)
            if os.path.exists(p):
                z.write(p, os.path.join("ultralytics", rel))
    print(json.dumps(meta, indent=2), flush=True)
    return meta


if __name__ == "__main__":
    main()
