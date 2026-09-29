"""Real face photos for FACE boxes.

Phase 1 drew cartoon avatars as ID photos (it ran without a face dataset). A detector trained only on
those would not find real faces, so when building the YOLO dataset we paste a real face photo over every
avatar. The avatar's box is known exactly, so labels stay correct.

Face photos are split into two fixed pools by a hash of the file name: ~90% for training, ~10% held out
for the validation and benchmark pages. The model is never scored on a face it was trained on.
"""
import os
import zlib

from PIL import Image, ImageFilter, ImageOps

IMG_EXT = (".jpg", ".jpeg", ".png")


def find_faces_dir(root, exclude=(), min_images=500):
    """Folder under root with the most images (e.g. CelebA's img_align_celeba). Skips Parda datasets."""
    if not root or not os.path.isdir(root):
        return None
    ex = [os.path.abspath(e) for e in exclude if e]
    best, best_n = None, 0
    for d, subdirs, files in os.walk(root):
        ad = os.path.abspath(d)
        if "annotations.jsonl" in files or any(ad == e or ad.startswith(e + os.sep) for e in ex):
            subdirs[:] = []  # a Parda synth dataset (its images are documents, not faces)
            continue
        n = sum(f.lower().endswith(IMG_EXT) for f in files)
        if n > best_n:
            best, best_n = d, n
    return best if best_n >= min_images else None


def _heldout(path):
    return zlib.crc32(os.path.basename(path).encode("utf-8")) % 10 == 0


class FacePool:
    def __init__(self, faces_dir, max_faces=30000):
        files = []
        for d, _, fs in os.walk(faces_dir):
            files += [os.path.join(d, f) for f in fs if f.lower().endswith(IMG_EXT)]
        files = sorted(files)[:max_faces]
        if not files:
            raise FileNotFoundError(f"no face images in {faces_dir}")
        self.train = [f for f in files if not _heldout(f)]
        self.heldout = [f for f in files if _heldout(f)]
        # tiny folders (tests) may put everything on one side
        self.train = self.train or self.heldout
        self.heldout = self.heldout or self.train

    def pick(self, rng, heldout):
        return rng.choice(self.heldout if heldout else self.train)

    def sizes(self):
        return {"train_pool": len(self.train), "heldout_pool": len(self.heldout)}


def paste_face(img, box, face_path, rng):
    """Cover box with a real face photo cropped to the box's shape. Returns True if pasted."""
    W, H = img.size
    x0, y0 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x1, y1 = min(W, int(round(box[2]))), min(H, int(round(box[3])))
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return False
    face = Image.open(face_path).convert("RGB")
    face = ImageOps.fit(face, (w, h), method=Image.BICUBIC, centering=(0.5, 0.4))
    if rng.random() < 0.2:  # photocopied / black-and-white ID photo
        face = ImageOps.grayscale(face).convert("RGB")
    blur = rng.uniform(0.0, 0.9)
    if blur > 0.2:  # match the softness of a scanned page
        face = face.filter(ImageFilter.GaussianBlur(blur))
    img.paste(face, (x0, y0))
    return True
