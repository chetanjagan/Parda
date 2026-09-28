"""Make clean renders look like real scans / phone photos.

Geometric changes (rotation) move the boxes, so we transform every box too.
Photometric changes (blur, noise, JPEG, lighting) don't move anything.
"""
import io
import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _rotate_box(box, angle_deg, cx, cy, W, H):
    """PIL rotates counter-clockwise around the centre. Map the 4 corners, take the enclosing box."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    xs, ys = [], []
    for (x, y) in [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]:
        dx, dy = x - cx, y - cy
        xs.append(cx + dx * ca + dy * sa)
        ys.append(cy - dx * sa + dy * ca)
    return [max(0, int(min(xs))), max(0, int(min(ys))), min(W, int(math.ceil(max(xs)))),
            min(H, int(math.ceil(max(ys))))]


def augment(img: Image.Image, record: dict, rng: random.Random, strength: float = 1.0):
    """Returns (augmented image, record with updated boxes, list of applied ops)."""
    W, H = img.size
    ops = []

    # ---- geometric: small rotation (scanner skew / phone angle) ----
    if rng.random() < 0.6 * strength:
        angle = rng.uniform(-3.0, 3.0) * strength
        fill = img.getpixel((2, 2))
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)
        cx, cy = W / 2, H / 2
        for w in record["words"]:
            w["bbox"] = _rotate_box(w["bbox"], angle, cx, cy, W, H)
        for e in record["entities"]:
            e["bbox"] = _rotate_box(e["bbox"], angle, cx, cy, W, H)
        for v in record["visuals"]:
            v["bbox"] = _rotate_box(v["bbox"], angle, cx, cy, W, H)
        ops.append(f"rotate({angle:.2f})")

    # ---- photometric ----
    if rng.random() < 0.5 * strength:  # uneven lighting / yellowed paper
        arr = np.asarray(img).astype(np.float32)
        gx = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.95, 1.05), W)[None, :, None]
        gy = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.95, 1.05), H)[:, None, None]
        tint = np.array([1.0, rng.uniform(0.96, 1.0), rng.uniform(0.85, 1.0)])[None, None, :]
        img = Image.fromarray(np.clip(arr * gx * gy * tint, 0, 255).astype(np.uint8))
        ops.append("lighting")
    if rng.random() < 0.5 * strength:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.2))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.1))
        ops.append("contrast")
    if rng.random() < 0.35 * strength:  # low-resolution scan
        f = rng.uniform(0.5, 0.8)
        img = img.resize((int(W * f), int(H * f)), Image.BILINEAR).resize((W, H), Image.BILINEAR)
        ops.append(f"lowres({f:.2f})")
    if rng.random() < 0.4 * strength:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
        ops.append("blur")
    if rng.random() < 0.5 * strength:
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 12), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        ops.append("noise")
    if rng.random() < 0.2 * strength:  # photocopy streak
        arr = np.asarray(img).astype(np.int16)
        x, wdt = rng.randint(0, W - 4), rng.randint(1, 3)
        arr[:, x:x + wdt] -= rng.randint(40, 110)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        ops.append("streak")
    if rng.random() < 0.7:  # JPEG artefacts (almost every real scan is a JPEG)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(35, 92))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        ops.append("jpeg")
    return img, record, ops
