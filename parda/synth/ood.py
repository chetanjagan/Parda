"""Out-of-distribution (OOD) test documents: layouts the models never saw + phone-photo damage.

  python -m parda.synth.ood --n 300 --out data/ood --faces_dir <face photos> --workers 4
  python -m parda.synth.ood --n 12  --out data/ood_print --clean      # clean pages, e.g. to print and photograph

Four NEW document types (training used loan form, payslip, ID card, rent agreement, discharge summary):
  bank_statement   small-font transaction table; UPI IDs and payee names inside narrations, next to
                   12-digit reference numbers that look like Aadhaar numbers but are not (hard negatives)
  kyc_form         boxed grid with the label ABOVE each value, photo, signature, QR code
  pharmacy_bill    patient block (UHID, ABHA, doctor), batch-number table, stamp
  offer_letter     letter prose: recipient address block, "Dear <first name>", IDs inside sentences
Photo damage (phone_photo): page on a desk, perspective tilt, shadow, defocus, noise, JPEG. Every word /
entity / visual box goes through the same perspective transform, so the labels stay exact.
Languages cycle en / hi-en / kn-en and document types cycle, so the set is balanced.
Faces come from the HELD-OUT face pool (never used to train the detector).
"""
import argparse
import io
import json
import os
import random
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from . import ids
from .canvas import Doc, available_scripts, get_font
from .names import HOSPITALS, ORGS, PLACES, address_lines, make_person
from .templates import inr, lbl, name_value, paper
from .visuals import photo, qr_image, signature, stamp

LANGS = ["en", "hi-en", "kn-en"]


# ------------------------------------------------------------------ faces
def face(rng, w, h, files):
    """A held-out real face photo cropped to w x h, or the synthetic avatar if no photos were given."""
    if files:
        try:
            return ImageOps.fit(Image.open(rng.choice(files)).convert("RGB"), (w, h), Image.BICUBIC, centering=(0.5, 0.4))
        except OSError:
            pass
    return photo(rng, w, h, None)


def _addr_block(doc, x, y, lines, size, line_h, style="regular"):
    """Multi-line address as ONE entity."""
    ent = None
    for ln in lines:
        _, ent = doc.text(x, y, ln, size, style, label="ADDRESS" if ent is None else None, ent=ent)
        y += line_h
    return y


# ------------------------------------------------------------------ templates
def bank_statement(rng, lang, files):
    W, H, M = 827, 1169, 40
    doc = Doc(W, H, paper(rng), "bank_statement", lang)
    p = make_person(rng)
    col = rng.choice([(0, 70, 140), (140, 20, 40), (0, 110, 90), (90, 40, 120)])
    doc.rect([0, 0, W, 70], outline=col, fill=col)
    doc.text(M, 20, p["bank_name"].upper(), 24, "bold", fill=(255, 255, 255))
    t = "STATEMENT OF ACCOUNT"
    doc.text(W - M - doc.text_width(t, 15, "bold"), 28, t, 15, "bold", fill=(255, 255, 255))
    y = 95
    x = doc.text(M, y, lbl("name", lang, rng) + ":", 13, fill=(90, 90, 90))[0] + 6
    name_value(doc, x, y, p["name"], lang, rng, 13, "bold")
    doc.text(470, y, "Account No:", 13, fill=(90, 90, 90))
    doc.text(570, y, p["account"], 13, "bold", label="BANK_ACCOUNT")
    y += 22
    _addr_block(doc, M, y, p["address"], 13, 20)
    doc.text(470, y, "IFSC:", 13, fill=(90, 90, 90))
    doc.text(570, y, p["ifsc"], 13, label="IFSC")
    doc.text(470, y + 20, "Customer ID:", 13, fill=(90, 90, 90))
    doc.text(570, y + 20, str(rng.randint(10 ** 7, 10 ** 8 - 1)), 13)  # an ID, but not personal data
    y += 44
    doc.text(M, y, lbl("mobile", lang, rng) + ":", 13, fill=(90, 90, 90))
    doc.text(M + 115, y, p["phone"], 13, label="PHONE")
    doc.text(470, y, "Email:", 13, fill=(90, 90, 90))
    doc.text(570, y, p["email"], 12, label="EMAIL")
    y += 22
    doc.text(M, y, f"Period: 01 {rng.choice(['Jan', 'Apr', 'Jul', 'Oct'])} 2026 to 30 "
                   f"{rng.choice(['Mar', 'Jun', 'Sep', 'Dec'])} 2026", 13, fill=(90, 90, 90))
    y += 35
    cols = [M, M + 72, M + 405, M + 545, M + 625, W - M]
    heads = ["Date", "Narration", "Ref No", "Debit", "Credit"]
    doc.rect([M, y, W - M, y + 24], outline=col, fill=(235, 238, 245))
    for i, hd in enumerate(heads):
        doc.text(cols[i] + 4, y + 5, hd, 12, "bold")
    y += 30
    bal = rng.randint(20, 400) * 1000
    for _ in range(rng.randint(12, 18)):
        doc.text(cols[0] + 4, y, f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}", 12)
        kind = rng.random()
        x = cols[1] + 4
        amt = rng.randint(50, 25000)
        if kind < 0.45:  # UPI payment to a person: their UPI ID and name are PII
            other = make_person(rng)
            x = doc.text(x, y, "UPI/DR", 12)[0]
            x = doc.text(x, y, other["upi"], 12, label="UPI_ID")[0]
            doc.text(x, y, other["name"]["en"], 12, label="PERSON_NAME")
            debit = True
        elif kind < 0.6:
            other = make_person(rng)
            x = doc.text(x, y, "NEFT CR", 12)[0]
            doc.text(x, y, other["name"]["en"], 12, label="PERSON_NAME")
            debit = False
        else:  # merchants, ATM, charges: not personal data
            doc.text(x, y, rng.choice(["POS BIG BAZAAR", "ATM WDL MG ROAD", "BILLPAY BESCOM", "SMS CHARGES",
                                       "POS SWIGGY", "INT CREDIT", "EMI HOME LOAN", "POS RELIANCE"]), 12)
            debit = rng.random() < 0.7
        doc.text(cols[2] + 4, y, ids.fake_12_digit_non_aadhaar(rng).replace(" ", ""), 12)  # looks like Aadhaar, isn't
        bal += -amt if debit else amt
        doc.text(cols[3 if debit else 4] + 4, y, inr(amt), 12)
        doc.hline(M, W - M, y + 19, fill=(215, 215, 215))
        y += 25
    doc.text(M, y + 20, f"Closing balance: Rs. {inr(max(bal, 0))}", 13, "bold")
    doc.text(M, y + 45, "This is a computer generated statement and does not require a signature.", 11,
             fill=(110, 110, 110))
    return doc


def kyc_form(rng, lang, files):
    W, H, M = 827, 1100, 45
    doc = Doc(W, H, paper(rng), "kyc_form", lang)
    p = make_person(rng)
    t = "KYC APPLICATION FORM" if lang == "en" else rng.choice(["KYC APPLICATION FORM", "KNOW YOUR CUSTOMER (KYC)"])
    doc.text(M, 35, t, 22, "bold")
    doc.text(M, 68, f"{rng.choice(ORGS)} - Individual customer", 13, fill=(100, 100, 100))
    doc.paste(face(rng, 130, 160, files), W - M - 135, 30, label="FACE")
    doc.rect([W - M - 138, 27, W - M + 3, 193], outline=(120, 120, 120))

    def cell(x0, y0, x1, key, value, label, names=None):
        """A box with the label on top and the value below (training only had 'Label: value' rows)."""
        doc.rect([x0, y0, x1, y0 + 58], outline=(150, 150, 150))
        doc.text(x0 + 6, y0 + 5, lbl(key, lang, rng), 12, fill=(110, 110, 110))
        if names is not None:
            name_value(doc, x0 + 6, y0 + 27, names, lang, rng, 16)
        else:
            doc.text(x0 + 6, y0 + 27, value, 16, label=label)

    y = 205
    cell(M, y, W - M, "name", None, "PERSON_NAME", p["name"])
    y += 66
    cell(M, y, W - M, "father", None, "PERSON_NAME", p["father"])
    y += 66
    cell(M, y, 300, "dob", p["dob"], "DOB")
    cell(308, y, 520, "gender", "Male" if p["gender"] == "M" else "Female", None)
    cell(528, y, W - M, "mobile", p["phone"], "PHONE")
    y += 66
    cell(M, y, 420, "aadhaar", p["aadhaar"], "AADHAAR")
    cell(428, y, W - M, "pan", p["pan"], "PAN")
    y += 66
    cell(M, y, W - M, "email", p["email"], "EMAIL")
    y += 66
    doc.rect([M, y, W - M, y + 82], outline=(150, 150, 150))
    doc.text(M + 6, y + 5, lbl("address", lang, rng), 12, fill=(110, 110, 110))
    _addr_block(doc, M + 6, y + 27, p["address"], 16, 25)
    y += 110
    doc.text(M, y, "Declaration: I confirm that the information given above is true and correct.", 13)
    y += 45
    doc.paste(signature(rng, 200, 65), M, y, label="SIGNATURE")
    doc.text(M, y + 72, lbl("signature", lang, rng), 12, fill=(110, 110, 110))
    doc.text(360, y + 20, f"{lbl('place', lang, rng)}: {p['city']}", 14)
    doc.text(360, y + 45, f"{lbl('date', lang, rng)}: {ids.date_str(rng, 2025, 2026)}", 14)
    if rng.random() < 0.6:
        doc.paste(qr_image(rng, f"KYC|{p['name']['en']}|{p['pan']}", 120), W - M - 125, y - 10, label="QR_CODE")
    return doc


def pharmacy_bill(rng, lang, files):
    W, H, M = 827, 1050, 45
    doc = Doc(W, H, paper(rng), "pharmacy_bill", lang)
    p, dr = make_person(rng), make_person(rng)
    hosp = rng.choice(HOSPITALS)
    loc, city, _, pin = rng.choice(PLACES)
    doc.text(M, 30, hosp, 21, "bold", fill=(0, 90, 80))
    doc.text(M, 60, f"Pharmacy | {loc}, {city} - {pin} | Ph: 080-{rng.randint(2000000, 9999999)}", 12,
             fill=(90, 90, 90))
    x = doc.text(M, 80, "GSTIN:", 12, fill=(90, 90, 90))[0]
    doc.text(x, 80, ids.gstin(rng), 12, label="GSTIN")
    t = "PHARMACY BILL / CASH MEMO"
    doc.text(W - M - doc.text_width(t, 15, "bold"), 80, t, 15, "bold")
    doc.hline(M, W - M, 105, width=2)
    y = 118
    rows = [
        (("Patient", None), ("name", p)),
        (("UHID", ids.mrn(rng)), ("MRN", None)),
        (("Mobile", p["phone"]), ("PHONE", None)),
        (("ABHA No", p["abha"]), ("ABHA", None)),
        (("Age / Sex", f"{p['age']} / {p['gender']}"), (None, None)),
        (("Doctor", None), ("doctor", dr)),
    ]
    for i, ((lab, val), (plab, person)) in enumerate(rows):
        x0 = M if i % 2 == 0 else 440
        yy = y + (i // 2) * 26
        x = doc.text(x0, yy, lab + ":", 14, fill=(90, 90, 90))[0] + 4
        if plab == "name":
            name_value(doc, x, yy, person["name"], lang, rng, 14, "bold")
        elif plab == "doctor":
            x = doc.text(x, yy, "Dr.", 14)[0]
            doc.text(x, yy, person["name"]["en"], 14, label="PERSON_NAME")
        else:
            doc.text(x, yy, val, 14, label=plab)
    y += 3 * 26 + 18
    cols = [M, M + 250, M + 380, M + 470, M + 540, W - M]
    doc.rect([M, y, W - M, y + 26], outline=(120, 120, 120), fill=(230, 242, 238))
    for i, hd in enumerate(["Medicine", "Batch", "Expiry", "Qty", "Amount"]):
        doc.text(cols[i] + 5, y + 6, hd, 13, "bold")
    y += 32
    total = 0
    for _ in range(rng.randint(5, 9)):
        med = rng.choice(["Paracetamol 650", "Pantoprazole 40", "Azithromycin 500", "Cetirizine 10", "Metformin 500",
                          "Amlodipine 5", "ORS Sachet", "Vitamin D3 60K", "Montelukast 10", "Atorvastatin 20"])
        qty, amt = rng.randint(1, 30), rng.randint(20, 900)
        total += amt
        doc.text(cols[0] + 5, y, med, 13)
        doc.text(cols[1] + 5, y, f"{rng.choice('ABCDEFGH')}{rng.randint(10000, 99999)}", 13)  # batch: not PII
        doc.text(cols[2] + 5, y, f"{rng.randint(1, 12):02d}/{rng.randint(27, 30)}", 13)
        doc.text(cols[3] + 5, y, str(qty), 13)
        doc.text(cols[4] + 5, y, inr(amt), 13)
        doc.hline(M, W - M, y + 20, fill=(220, 220, 220))
        y += 27
    doc.text(cols[3] - 60, y + 10, f"Total: Rs. {inr(total)}", 15, "bold")
    y += 60
    doc.paste(stamp(rng, get_font("latin_bold", 11)), M + 20, y, label="STAMP")
    doc.paste(signature(rng, 170, 55), W - M - 220, y + 20, label="SIGNATURE")
    doc.text(W - M - 220, y + 85, "Pharmacist", 13)
    return doc


def offer_letter(rng, lang, files):
    W, H, M = 827, 1169, 70
    doc = Doc(W, H, paper(rng), "offer_letter", lang)
    p, hr = make_person(rng), make_person(rng)
    org = rng.choice(ORGS)
    loc, city, state, pin = rng.choice(PLACES)
    doc.text(M, 40, org, 22, "bold", fill=(30, 30, 90))
    doc.text(M, 70, f"{rng.randint(1, 300)}, Tech Park, {loc}, {city}, {state} - {pin}", 12, fill=(100, 100, 100))
    x = doc.text(M, 88, "GSTIN:", 12, fill=(100, 100, 100))[0]
    doc.text(x, 88, ids.gstin(rng), 12, label="GSTIN")
    doc.hline(M, W - M, 112, fill=(30, 30, 90), width=2)
    date = ids.date_str(rng, 2025, 2026)
    doc.text(W - M - doc.text_width(date, 15), 130, date, 15)
    y = 165
    doc.text(M, y, "To,", 15)
    y += 24
    name_value(doc, M, y, p["name"], lang, rng, 15, "bold")
    y = _addr_block(doc, M, y + 24, p["address"], 15, 24)
    y += 20
    doc.text(M, y, "Subject: Offer of employment", 15, "bold")
    y += 38
    x = doc.text(M, y, "Dear", 15)[0]
    doc.text(x, y, p["first"] + ",", 15, label="PERSON_NAME")
    y += 36
    role = rng.choice(["Software Engineer", "Data Analyst", "Accounts Executive", "Product Designer", "HR Associate"])
    ctc = rng.randint(4, 30) * 100000
    y = doc.paragraph(M, y, W - 2 * M, [
        (f"We are pleased to offer you the position of {role} at {org.rstrip('.')}. Your employee ID will be", None),
        (ids.employee_id(rng), "EMPLOYEE_ID"),
        (f"and your annual cost to company will be Rs. {inr(ctc)}. Please report to our {loc} office on", None),
        (ids.date_str(rng, 2025, 2026), None), (".", None),
    ], size=15) + 14
    y = doc.paragraph(M, y, W - 2 * M, [
        ("Your salary will be credited to account", None), (p["account"], "BANK_ACCOUNT"),
        ("(IFSC", None), (p["ifsc"], "IFSC"),
        ("). Your provident fund will continue under UAN", None), (p["uan"], "UAN"),
        (". Kindly bring your PAN card", None), (p["pan"], "PAN"),
        ("and address proof on your first day.", None),
    ], size=15) + 14
    y = doc.paragraph(M, y, W - 2 * M, [
        ("For any questions, please call us or write to", None),
        (ids.email(rng, hr["first"], hr["last"]).split("@")[0] + "@" + org.split()[0].lower() + ".in", "EMAIL"),
        (". We look forward to welcoming you to the team.", None),
    ], size=15) + 30
    doc.text(M, y, f"For {org}", 15)
    y += 30
    doc.paste(signature(rng, 190, 60), M, y, label="SIGNATURE")
    if rng.random() < 0.7:
        doc.paste(stamp(rng, get_font("latin_bold", 11)), M + 230, y - 10, label="STAMP")
    doc.text(M, y + 70, hr["name"]["en"], 15, "bold", label="PERSON_NAME")
    doc.text(M, y + 94, "Manager, Human Resources", 14)
    return doc


TEMPLATES = {"bank_statement": bank_statement, "kyc_form": kyc_form,
             "pharmacy_bill": pharmacy_bill, "offer_letter": offer_letter}


# ------------------------------------------------------------------ phone photo
def homography(src, dst):
    """3x3 H with dst ~ H @ src, from 4 point pairs."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = np.linalg.solve(np.asarray(A, float), np.asarray(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def warp_points(Hm, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:3]


def _warp_box(Hm, box, dx, dy, W, H):
    x0, y0, x1, y1 = box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy
    q = warp_points(Hm, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return [max(0, int(np.floor(q[:, 0].min()))), max(0, int(np.floor(q[:, 1].min()))),
            min(W, int(np.ceil(q[:, 0].max()))), min(H, int(np.ceil(q[:, 1].max())))]


def phone_photo(img, rec, rng, desk=None, tilt=0.05, effects=True):
    """Page photographed on a desk: margin, perspective, shadow, defocus, noise, JPEG. Boxes follow exactly.
    effects=False: geometry only (for testing that the boxes follow the ink)."""
    W, H = img.size
    m = int(0.06 * max(W, H) * rng.uniform(0.6, 1.2))
    desk = desk or rng.choice([(96, 72, 52), (140, 110, 80), (60, 60, 65), (200, 200, 195), (35, 35, 40)])
    canvas = Image.new("RGB", (W + 2 * m, H + 2 * m), desk)
    canvas.paste(img, (m, m))
    CW, CH = canvas.size
    src = [(m, m), (m + W, m), (m + W, m + H), (m, m + H)]
    dst = [(x + rng.uniform(-tilt, tilt) * min(W, H), y + rng.uniform(-tilt, tilt) * min(W, H)) for x, y in src]
    Hf = homography(src, dst)
    Hi = np.linalg.inv(Hf)
    out = canvas.transform((CW, CH), Image.PERSPECTIVE, tuple((Hi / Hi[2, 2]).flatten()[:8]), Image.BICUBIC,
                           fillcolor=desk)
    for key in ("words", "entities", "visuals"):
        for it in rec[key]:
            it["bbox"] = _warp_box(Hf, it["bbox"], m, m, CW, CH)
    ops = [f"perspective({tilt:.2f})"]
    if not effects:
        return out, rec, ops
    arr = np.asarray(out).astype(np.float32)
    gx = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.1), CW)[None, :, None]
    gy = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.9, 1.1), CH)[:, None, None]
    arr = arr * gx * gy
    if rng.random() < 0.6:  # soft shadow of the phone / hand
        sh = Image.new("L", (CW, CH), 0)
        cx, cy = rng.uniform(0, CW), rng.uniform(0, CH)
        r = rng.uniform(0.25, 0.5) * max(CW, CH)
        ImageDraw.Draw(sh).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(r * 0.35))).astype(np.float32)[:, :, None] / 255.0
        arr = arr * (1 - rng.uniform(0.2, 0.45) * sh)
        ops.append("shadow")
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.7:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.3)))
        ops.append("defocus")
    arr = np.asarray(out).astype(np.float32)
    arr += np.random.default_rng(rng.randint(0, 2 ** 31)).normal(0, rng.uniform(2, 8), arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=rng.randint(45, 85))
    out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    ops += ["noise", "jpeg"]
    return out, rec, ops


# ------------------------------------------------------------------ generator
_CFG = {}


def _init(cfg):
    _CFG.update(cfg)


def make_one(i):
    rng = random.Random(_CFG["seed"] * 1_000_003 + i)
    langs = _CFG["langs"]
    lang = langs[i % len(langs)]
    doc_type = list(TEMPLATES)[(i // len(langs)) % len(TEMPLATES)]
    try:
        doc = TEMPLATES[doc_type](rng, lang, _CFG["faces"])
        rec = doc.finalize()
        img, ops = doc.img, []
        if not _CFG["clean"]:
            img, rec, ops = phone_photo(img, rec, rng)
        name = f"{i:06d}.jpg"
        img.save(os.path.join(_CFG["out"], "images", name), "JPEG", quality=92)
        rec.update({"id": i, "image": f"images/{name}", "width": img.width, "height": img.height,
                    "doc_type": doc_type, "lang": lang, "aug": ops, "ood": True})
        return rec
    except Exception as e:  # never kill a run for one bad page
        return {"id": i, "error": f"{type(e).__name__}: {e}", "doc_type": doc_type, "lang": lang}


def generate(n, out, faces_dir=None, workers=4, clean=False, seed=7):
    from ..vision.faces import FacePool
    faces = FacePool(faces_dir).heldout if faces_dir else []
    scripts = available_scripts()
    langs = [lg for lg in LANGS if lg == "en" or scripts[{"hi-en": "deva", "kn-en": "knda"}[lg]]]
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    cfg = {"seed": seed, "out": out, "faces": faces, "clean": clean, "langs": langs}
    t0, recs, errors = time.time(), [], []
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
            results = list(pool.imap_unordered(make_one, range(n), chunksize=4))
    else:
        _init(cfg)
        results = [make_one(i) for i in range(n)]
    for r in sorted(results, key=lambda r: r["id"]):
        (errors if "error" in r else recs).append(r)
    with open(os.path.join(out, "annotations.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"n_ok": len(recs), "n_errors": len(errors), "errors": errors[:5], "clean": clean,
             "face_photos": len(faces), "seconds": round(time.time() - t0, 1),
             "doc_types": dict(Counter(r["doc_type"] for r in recs)), "langs": dict(Counter(r["lang"] for r in recs)),
             "entities": dict(Counter(e["label"] for r in recs for e in r["entities"])),
             "visuals": dict(Counter(v["label"] for r in recs for v in r["visuals"]))}
    if not faces_dir:
        stats["warning"] = "no faces_dir: ID photos are synthetic avatars"
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=2)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--faces_dir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clean", action="store_true", help="no photo damage (e.g. pages to print)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    s = generate(a.n, a.out, a.faces_dir, a.workers, a.clean, a.seed)
    print(json.dumps(s, indent=2), flush=True)
    return s


if __name__ == "__main__":
    main()
