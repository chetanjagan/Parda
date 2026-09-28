"""Non-text PII: signature, QR code, stamp, face photo, SPECIMEN watermark."""
import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter

try:
    import qrcode
    HAS_QR = True
except ImportError:
    HAS_QR = False

INKS = [(20, 30, 120), (15, 15, 15), (30, 30, 90), (10, 60, 140)]


def signature(rng: random.Random, w=200, h=70):
    """Random cursive-looking scribble made of sine-wave strokes."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ink = rng.choice(INKS) + (255,)
    x = rng.uniform(5, 15)
    for _ in range(rng.randint(2, 4)):  # 2-4 "letters groups"
        seg_w = rng.uniform(w * 0.2, w * 0.4)
        amp, freq, phase = rng.uniform(8, h * 0.35), rng.uniform(2, 5), rng.uniform(0, 6.28)
        base = h / 2 + rng.uniform(-8, 8)
        pts = []
        steps = 40
        for i in range(steps + 1):
            t = i / steps
            px = x + t * seg_w
            py = base + amp * math.sin(freq * 2 * math.pi * t + phase) * (1 - 0.5 * t)
            pts.append((px, py))
        d.line(pts, fill=ink, width=rng.choice([2, 2, 3]), joint="curve")
        x += seg_w * rng.uniform(0.7, 1.0)
        if x > w - 20:
            break
    if rng.random() < 0.6:  # underline flourish
        y = h * rng.uniform(0.7, 0.85)
        d.line([(10, y), (x, y + rng.uniform(-6, 6))], fill=ink, width=2)
    return im


def qr_image(rng: random.Random, data: str, size=150):
    if HAS_QR:  # draw from the raw matrix (stable across qrcode versions)
        q = qrcode.QRCode(border=1)
        q.add_data(data)
        q.make(fit=True)
        m = q.get_matrix()
        n = len(m)
        im = Image.new("RGB", (n, n), "white")
        px = im.load()
        for i in range(n):
            for j in range(n):
                if m[i][j]:
                    px[j, i] = (0, 0, 0)
    else:  # fallback: QR-looking random matrix with finder patterns
        n = 29
        im = Image.new("RGB", (n, n), "white")
        px = im.load()
        for i in range(n):
            for j in range(n):
                if rng.random() < 0.5:
                    px[i, j] = (0, 0, 0)
        d = ImageDraw.Draw(im)
        for (ox, oy) in [(0, 0), (n - 7, 0), (0, n - 7)]:
            d.rectangle([ox, oy, ox + 6, oy + 6], fill="black")
            d.rectangle([ox + 1, oy + 1, ox + 5, oy + 5], fill="white")
            d.rectangle([ox + 2, oy + 2, ox + 4, oy + 4], fill="black")
    return im.resize((size, size), Image.NEAREST)


def stamp(rng: random.Random, text_font, size=140):
    """Round rubber stamp with a short text; returned as RGBA."""
    col = rng.choice([(40, 60, 160), (110, 40, 140), (160, 30, 40)])
    alpha = rng.randint(140, 210)
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([3, 3, size - 3, size - 3], outline=col + (alpha,), width=4)
    d.ellipse([16, 16, size - 16, size - 16], outline=col + (alpha,), width=2)
    for i, line in enumerate(["AUTHORISED", "SIGNATORY"]):
        tw = d.textlength(line, font=text_font)
        d.text(((size - tw) / 2, size / 2 - 16 + i * 18), line, font=text_font, fill=col + (alpha,))
    return im.rotate(rng.uniform(-25, 25), resample=Image.BICUBIC)


def _load_face_pool(faces_dir):
    if not faces_dir or not os.path.isdir(faces_dir):
        return []
    return [os.path.join(faces_dir, f) for f in os.listdir(faces_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]


_FACE_POOL = None


def photo(rng: random.Random, w=120, h=150, faces_dir=None):
    """ID photo. Uses real face crops from faces_dir if given (e.g. WIDER FACE crops),
    otherwise draws a simple synthetic avatar."""
    global _FACE_POOL
    if _FACE_POOL is None:
        _FACE_POOL = _load_face_pool(faces_dir)
    if _FACE_POOL:
        try:
            return Image.open(rng.choice(_FACE_POOL)).convert("RGB").resize((w, h))
        except Exception:
            pass
    bg = rng.choice([(210, 225, 240), (235, 235, 235), (200, 215, 200), (240, 230, 215)])
    skin = rng.choice([(224, 180, 140), (198, 150, 110), (170, 120, 85), (140, 95, 65)])
    hair = rng.choice([(20, 20, 20), (45, 30, 20), (70, 60, 55)])
    shirt = rng.choice([(40, 60, 110), (120, 30, 40), (60, 90, 60), (230, 230, 230), (90, 90, 90)])
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    cx = w / 2 + rng.uniform(-5, 5)
    d.ellipse([cx - w * 0.45, h * 0.72, cx + w * 0.45, h * 1.3], fill=shirt)          # shoulders
    d.rectangle([cx - w * 0.08, h * 0.55, cx + w * 0.08, h * 0.78], fill=skin)       # neck
    d.ellipse([cx - w * 0.24, h * 0.14, cx + w * 0.24, h * 0.64], fill=skin)         # face
    d.chord([cx - w * 0.26, h * 0.1, cx + w * 0.26, h * 0.5], 180, 360, fill=hair)   # hair
    for ex in (-0.09, 0.09):
        d.ellipse([cx + w * ex - 3, h * 0.36, cx + w * ex + 3, h * 0.40], fill=(30, 30, 30))
    d.arc([cx - w * 0.08, h * 0.44, cx + w * 0.08, h * 0.54], 20, 160, fill=(120, 60, 60), width=2)
    return im.filter(ImageFilter.GaussianBlur(0.6))


def watermark(img, font, text="SPECIMEN", alpha=45):
    """Diagonal SPECIMEN watermark so no synthetic ID ever looks like a real one."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tw = d.textlength(text, font=font)
    d.text(((img.width - tw) / 2, img.height / 2 - font.size / 2), text, font=font, fill=(200, 0, 0, alpha))
    layer = layer.rotate(25, resample=Image.BICUBIC)
    base = img.convert("RGBA")
    base.alpha_composite(layer)
    return base.convert("RGB")
