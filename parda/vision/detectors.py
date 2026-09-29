"""Visual PII detectors with one interface: detect(paths) -> list (per image) of
[{"label": "FACE", "bbox": [x0, y0, x1, y1], "score": 0.93}, ...] in pixel coordinates.

  YoloDetector    the Phase 4 model (ultralytics)
  OpenCVBaseline  classic "before" system: Haar-cascade faces + OpenCV QR detector (no signatures/stamps)
  GoldDetector    returns the true boxes (tests / sanity checks)
  EmptyDetector   returns nothing (tests)
"""
import os

from PIL import Image

from .classes import CLASSES


def label_path(image_path):
    """YOLO convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt"""
    head, tail = image_path.rsplit(os.sep + "images" + os.sep, 1)
    return os.path.join(head, "labels", os.path.splitext(tail)[0] + ".txt")


def read_gt(image_path, size=None):
    W, H = size or Image.open(image_path).size
    out = []
    p = label_path(image_path)
    if os.path.exists(p):
        for ln in open(p):
            parts = ln.split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            out.append({"label": CLASSES[c], "bbox": [(cx - w / 2) * W, (cy - h / 2) * H,
                                                       (cx + w / 2) * W, (cy + h / 2) * H]})
    return out


class GoldDetector:
    name = "gold"

    def detect(self, paths):
        return [[dict(b, score=1.0) for b in read_gt(p)] for p in paths]


class EmptyDetector:
    name = "empty"

    def detect(self, paths):
        return [[] for _ in paths]


class OpenCVBaseline:
    """What you get without training anything. Scores are 1.0 (these detectors give no confidence)."""
    name = "opencv"

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.face = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        self.qr = cv2.QRCodeDetector()

    def _qr_boxes(self, gray):
        pts = None
        try:
            ok, pts = self.qr.detectMulti(gray)
            if not ok:
                pts = None
        except Exception:
            pts = None
        if pts is None:
            try:
                ok, p1 = self.qr.detect(gray)
                pts = [p1] if ok and p1 is not None else []
            except Exception:
                pts = []
        boxes = []
        for p in pts:
            p = p.reshape(-1, 2)
            boxes.append([float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())])
        return boxes

    def detect(self, paths):
        cv2 = self.cv2
        out = []
        for path in paths:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = []
            for (x, y, w, h) in self.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
                dets.append({"label": "FACE", "bbox": [float(x), float(y), float(x + w), float(y + h)], "score": 1.0})
            for b in self._qr_boxes(gray):
                dets.append({"label": "QR_CODE", "bbox": b, "score": 1.0})
            out.append(dets)
        return out


class YoloDetector:
    name = "yolo"

    def __init__(self, weights, conf=0.01, imgsz=1024, iou=0.5, device=None, batch=16):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.iou, self.batch = conf, imgsz, iou, batch
        if device is None:
            try:
                import torch
                device = 0 if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

    def detect(self, paths):
        out = []
        for i in range(0, len(paths), self.batch):
            res = self.model.predict(list(paths[i:i + self.batch]), conf=self.conf, imgsz=self.imgsz, iou=self.iou,
                                     device=self.device, verbose=False)
            for r in res:
                b = r.boxes
                xyxy = b.xyxy.cpu().numpy().tolist()
                cls = b.cls.cpu().numpy().tolist()
                conf = b.conf.cpu().numpy().tolist()
                names = r.names
                out.append([{"label": names[int(c)], "bbox": [float(v) for v in box], "score": float(s)}
                            for box, c, s in zip(xyxy, cls, conf)])
        return out


def make(system, weights=None, imgsz=1024):
    if system == "gold":
        return GoldDetector()
    if system == "empty":
        return EmptyDetector()
    if system == "opencv":
        return OpenCVBaseline()
    if system == "yolo":
        if not weights:
            raise ValueError("--weights is required for the yolo system")
        return YoloDetector(weights, imgsz=imgsz)
    raise ValueError(f"unknown system {system}")
