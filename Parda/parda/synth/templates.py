"""Document templates. Each builder returns a Doc with all PII labelled.

PII labels:
  PERSON_NAME AADHAAR PAN PHONE EMAIL ADDRESS DOB BANK_ACCOUNT IFSC UPI_ID
  GSTIN VOTER_ID PASSPORT VEHICLE_REG UAN ABHA EMPLOYEE_ID MRN
Visual labels:
  FACE SIGNATURE QR_CODE STAMP
Everything else (amounts, dates that aren't DOB, reference numbers, org names)
is intentionally left unlabelled -> these are the "hard negatives".
"""
import random

from PIL import ImageDraw

from . import ids
from .canvas import Doc, get_font
from .names import HOSPITALS, ORGS, PLACES, address_lines, make_person
from .visuals import photo, qr_image, signature, stamp, watermark

# (english, hindi, kannada)
LBL = {
    "name": ("Name", "नाम", "ಹೆಸರು"),
    "father": ("Father's Name", "पिता का नाम", "ತಂದೆಯ ಹೆಸರು"),
    "dob": ("Date of Birth", "जन्म तिथि", "ಜನ್ಮ ದಿನಾಂಕ"),
    "gender": ("Gender", "लिंग", "ಲಿಂಗ"),
    "address": ("Address", "पता", "ವಿಳಾಸ"),
    "mobile": ("Mobile", "मोबाइल", "ಮೊಬೈಲ್"),
    "email": ("Email", "ईमेल", "ಇಮೇಲ್"),
    "aadhaar": ("Aadhaar No", "आधार संख्या", "ಆಧಾರ್ ಸಂಖ್ಯೆ"),
    "pan": ("PAN", "पैन", "ಪ್ಯಾನ್"),
    "account": ("Bank A/C No", "बैंक खाता संख्या", "ಬ್ಯಾಂಕ್ ಖಾತೆ ಸಂಖ್ಯೆ"),
    "loan": ("Loan Amount", "ऋण राशि", "ಸಾಲದ ಮೊತ್ತ"),
    "purpose": ("Purpose", "उद्देश्य", "ಉದ್ದೇಶ"),
    "signature": ("Signature", "हस्ताक्षर", "ಸಹಿ"),
    "date": ("Date", "दिनांक", "ದಿನಾಂಕ"),
    "place": ("Place", "स्थान", "ಸ್ಥಳ"),
}
GENDER = {"M": ("Male", "पुरुष", "ಪುರುಷ"), "F": ("Female", "महिला", "ಮಹಿಳೆ")}
LANG_IDX = {"en": 0, "hi-en": 1, "kn-en": 2}
NATIVE = {"hi-en": "hi", "kn-en": "kn"}


def inr(n) -> str:
    """Indian digit grouping: 1234567 -> 12,34,567"""
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def lbl(key, lang, rng):
    en = LBL[key][0]
    if lang == "en":
        return en
    native = LBL[key][LANG_IDX[lang]]
    r = rng.random()
    return f"{native} / {en}" if r < 0.6 else (native if r < 0.8 else en)


def name_value(doc, x, y, names, lang, rng, size=18, style="regular"):
    """Write a person's name as English, native script, or both. Each script = separate entity."""
    if lang == "en":
        return doc.text(x, y, names["en"], size, style, label="PERSON_NAME")[0]
    native = names[NATIVE[lang]]
    r = rng.random()
    if r < 0.4:
        x = doc.text(x, y, native, size, style, label="PERSON_NAME")[0]
        x = doc.text(x, y, "/", size, style)[0]
        return doc.text(x, y, names["en"], size, style, label="PERSON_NAME")[0]
    if r < 0.7:
        return doc.text(x, y, names["en"], size, style, label="PERSON_NAME")[0]
    return doc.text(x, y, native, size, style, label="PERSON_NAME")[0]


def label_at(doc, x, y, key, lang, rng, vx=None, size=18):
    """Draw 'Label:' and return where the value should start (never overlapping a long label)."""
    x_end = doc.text(x, y, lbl(key, lang, rng) + ":", size, "bold" if rng.random() < 0.3 else "regular",
                     fill=(60, 60, 60))[0]
    return max(vx, x_end + 8) if vx is not None else x_end + 8


def name_row(doc, x, y, key, names, lang, rng, vx=None, size=18):
    vx = label_at(doc, x, y, key, lang, rng, vx, size)
    name_value(doc, vx, y, names, lang, rng, size)
    return y + int(size * 1.7)


def field(doc, x, y, key, value, label, lang, rng, vx=None, size=18, underline=False, right=None):
    """'Label: value' row. vx = x of value column. value may be a list of lines (address)."""
    vx = label_at(doc, x, y, key, lang, rng, vx, size)
    lines = value if isinstance(value, list) else [value]
    ent, yy = None, y
    for ln in lines:
        _, ent = doc.text(vx, yy, ln, size, label=label if ent is None else None, ent=ent)
        if underline and right:
            doc.hline(vx, right, yy + size + 6, fill=(170, 170, 170))
        yy += int(size * 1.7)
    return yy


def raw_field(doc, x, y, label_text, value, label, vx=None, size=18, style="regular"):
    x_end = doc.text(x, y, label_text + ":", size, "regular", fill=(60, 60, 60))[0]
    doc.text(max(vx, x_end + 8) if vx is not None else x_end + 8, y, value, size, style, label=label)
    return y + int(size * 1.7)


def paper(rng):
    return rng.choice([(255, 255, 255), (252, 251, 246), (248, 247, 240), (250, 250, 250)])


# ============================================================ LOAN APPLICATION
def loan_application(rng, lang, faces_dir=None):
    W, H, M = 827, 1169, 50
    doc = Doc(W, H, paper(rng), "loan_application", lang)
    p = make_person(rng)
    org = rng.choice(ORGS)

    tw = doc.text_width(org, 26, "bold")
    doc.text((W - tw) / 2, 35, org, 26, "bold", fill=(20, 40, 100))
    title = "LOAN APPLICATION FORM"
    doc.text((W - doc.text_width(title, 20, "bold")) / 2, 78, title, 20, "bold")
    doc.hline(M, W - M, 112, width=2)
    doc.text(M, 125, f"Application No: LA/{rng.randint(2024, 2026)}/{rng.randint(1000, 99999):05d}", 15)
    doc.text(W - M - 190, 125, f"Date: {ids.date_str(rng, 2024, 2026)}", 15)

    # photo box (top right)
    px, py = W - M - 130, 165
    doc.rect([px - 4, py - 4, px + 124, py + 154])
    doc.paste(photo(rng, 120, 150, faces_dir), px, py, label="FACE")

    vx, y, right = 285, 175, W - M - 150
    y = name_row(doc, M, y, "name", p["name"], lang, rng, vx)
    y = name_row(doc, M, y, "father", p["father"], lang, rng, vx)
    y = field(doc, M, y, "dob", p["dob"], "DOB", lang, rng, vx)
    g = GENDER[p["gender"]][LANG_IDX[lang] if rng.random() < 0.5 else 0]
    y = field(doc, M, y, "gender", g, None, lang, rng, vx)
    right = W - M
    y = field(doc, M, y, "mobile", p["phone"], "PHONE", lang, rng, vx, underline=True, right=right)
    y = field(doc, M, y, "email", p["email"], "EMAIL", lang, rng, vx, underline=True, right=right)
    y = field(doc, M, y, "aadhaar", p["aadhaar"], "AADHAAR", lang, rng, vx, underline=True, right=right)
    y = field(doc, M, y, "pan", p["pan"], "PAN", lang, rng, vx, underline=True, right=right)
    y = field(doc, M, y, "address", p["address"], "ADDRESS", lang, rng, vx, underline=True, right=right)
    y = field(doc, M, y, "account", p["account"], "BANK_ACCOUNT", lang, rng, vx)
    y = raw_field(doc, M, y, "Bank / IFSC", "", None, vx)
    x2 = doc.text(vx, y - int(18 * 1.7), p["bank_name"] + " /", 18)[0]
    doc.text(x2, y - int(18 * 1.7), p["ifsc"], 18, label="IFSC")
    y = raw_field(doc, M, y, "UPI ID", p["upi"], "UPI_ID", vx)
    y = field(doc, M, y, "loan", f"Rs. {inr(rng.choice([50, 75, 100, 150, 200, 300, 500]) * 1000)}", None, lang, rng, vx)
    y = field(doc, M, y, "purpose", rng.choice(["Shop renovation", "Education", "Medical expenses",
                                                  "Two-wheeler purchase", "Agriculture inputs",
                                                  "Home repair"]), None, lang, rng, vx)
    y = raw_field(doc, M, y, "Tenure", f"{rng.choice([12, 18, 24, 36, 48])} months", None, vx)
    # hard negative: 12-digit number that is NOT an Aadhaar
    y = raw_field(doc, M, y, "Existing Loan Ref No", ids.fake_12_digit_non_aadhaar(rng), None, vx)

    y += 10
    doc.text(M, y, "DECLARATION", 17, "bold")
    y += 32
    y = doc.paragraph(M, y, W - 2 * M, [
        ("I,", None), (p["name"]["en"], "PERSON_NAME"),
        (", hereby declare that the information furnished above is true and correct to the best of my "
         "knowledge. I authorise", None), (org, None),
        ("to verify my details and contact me on", None), (p["phone"], "PHONE"),
        (". I understand that any false information may lead to rejection of my application.", None),
    ], size=16)

    y += 25
    x = doc.text(M, y + 30, lbl("signature", lang, rng) + ":", 17)[0]
    doc.paste(signature(rng, 200, 70), x + 10, y, label="SIGNATURE")
    doc.text(W - M - 260, y + 10, lbl("place", lang, rng) + ": " + p["city"], 16)
    doc.text(W - M - 260, y + 40, lbl("date", lang, rng) + ": " + ids.date_str(rng, 2025, 2026), 16)

    if rng.random() < 0.6:
        doc.paste(stamp(rng, get_font("latin_bold", 11)), W - M - 150, H - 190, label="STAMP")
        doc.text(W - M - 175, H - 45, "FOR OFFICE USE ONLY", 13, fill=(90, 90, 90))
    return doc


# ============================================================ PAYSLIP
def payslip(rng, lang, faces_dir=None):
    W, H, M = 827, 1000, 45
    doc = Doc(W, H, paper(rng), "payslip", lang)
    p = make_person(rng)
    org = rng.choice(ORGS)
    loc, city, state, pin = rng.choice(PLACES)

    doc.text(M, 35, org, 24, "bold", fill=(20, 40, 100))
    doc.text(M, 72, f"Regd. Office: {rng.randint(1, 300)}, Industrial Area, {loc}, {city} - {pin}", 14,
             fill=(80, 80, 80))
    raw_field(doc, M, 96, "GSTIN", ids.gstin(rng), "GSTIN", size=14)
    month = rng.choice(["January", "February", "March", "April", "May", "June", "July", "August",
                        "September", "October", "November", "December"])
    t = f"Payslip for the month of {month} {rng.choice([2025, 2026])}"
    doc.hline(M, W - M, 130, width=2)
    doc.text((W - doc.text_width(t, 18, "bold")) / 2, 142, t, 18, "bold")

    lx, lv, rx, rv, y = M, M + 175, 440, 585, 190
    name_row(doc, lx, y, "name", p["name"], lang, rng, lv, size=15)
    raw_field(doc, rx, y, "PAN", p["pan"], "PAN", rv, size=15)
    y += 30
    raw_field(doc, lx, y, "Employee ID", ids.employee_id(rng), "EMPLOYEE_ID", lv, size=15)
    raw_field(doc, rx, y, "UAN", p["uan"], "UAN", rv, size=15)
    y += 30
    raw_field(doc, lx, y, "Designation", rng.choice(["Software Engineer", "Accountant", "Sales Executive",
                                                        "Operations Manager", "HR Associate", "Supervisor"]),
              None, lv, size=15)
    raw_field(doc, rx, y, "Bank A/C", p["account"], "BANK_ACCOUNT", rv, size=15)
    y += 30
    raw_field(doc, lx, y, "Department", rng.choice(["Engineering", "Finance", "Sales", "Operations", "HR"]),
              None, lv, size=15)
    raw_field(doc, rx, y, "IFSC", p["ifsc"], "IFSC", rv, size=15)
    y += 30
    raw_field(doc, lx, y, "Date of Joining", ids.date_str(rng, 2012, 2025), None, lv, size=15)
    raw_field(doc, rx, y, "Days Paid", str(rng.randint(26, 31)), None, rv, size=15)

    # earnings / deductions table
    y += 55
    basic = rng.randint(15, 90) * 1000
    earn = [("Basic", basic), ("HRA", int(basic * 0.4)), ("Special Allowance", int(basic * rng.uniform(0.2, 0.6))),
            ("Conveyance", 1600)]
    ded = [("Provident Fund", int(basic * 0.12)), ("Professional Tax", 200),
           ("TDS", int(basic * rng.uniform(0.0, 0.15))), ("Other", rng.choice([0, 500, 1000]))]
    cols = [M, M + 230, 430, 430 + 230, W - M]
    doc.rect([M, y, W - M, y + 36 * 6], width=1)
    for cx in cols[1:-1]:
        doc.draw.line([(cx, y), (cx, y + 36 * 6)], fill=(90, 90, 90))
    for i, head in enumerate(["Earnings", "Amount (Rs.)", "Deductions", "Amount (Rs.)"]):
        doc.text(cols[i] + 8, y + 8, head, 15, "bold")
    for r in range(4):
        yy = y + 36 * (r + 1)
        doc.hline(M, W - M, yy)
        doc.text(cols[0] + 8, yy + 8, earn[r][0], 15)
        doc.text(cols[1] + 8, yy + 8, inr(earn[r][1]), 15)
        doc.text(cols[2] + 8, yy + 8, ded[r][0], 15)
        doc.text(cols[3] + 8, yy + 8, inr(ded[r][1]), 15)
    yy = y + 36 * 5
    doc.hline(M, W - M, yy)
    ge, gd = sum(v for _, v in earn), sum(v for _, v in ded)
    doc.text(cols[0] + 8, yy + 8, "Gross Earnings", 15, "bold")
    doc.text(cols[1] + 8, yy + 8, inr(ge), 15, "bold")
    doc.text(cols[2] + 8, yy + 8, "Total Deductions", 15, "bold")
    doc.text(cols[3] + 8, yy + 8, inr(gd), 15, "bold")

    y = y + 36 * 6 + 30
    doc.text(M, y, f"Net Pay: Rs. {inr(ge - gd)}", 19, "bold")
    doc.text(M, y + 45, "This is a computer generated payslip and does not require a signature.", 13,
             fill=(100, 100, 100))
    if rng.random() < 0.5:
        doc.paste(signature(rng, 180, 60), W - M - 220, y + 80, label="SIGNATURE")
        doc.text(W - M - 220, y + 150, "Authorised Signatory", 14)
        if rng.random() < 0.6:
            doc.paste(stamp(rng, get_font("latin_bold", 11)), W - M - 380, y + 70, label="STAMP")
    return doc


# ============================================================ ID CARD (generic, SPECIMEN)
def id_card(rng, lang, faces_dir=None):
    W, H = 1012, 638
    bg = rng.choice([(236, 244, 250), (250, 244, 232), (238, 248, 238), (245, 240, 250)])
    variant = rng.choice(["uid", "tax", "voter"])
    doc = Doc(W, H, bg, f"id_card_{variant}", lang)
    p = make_person(rng)
    band = rng.choice([(30, 70, 140), (150, 60, 30), (40, 110, 70), (90, 50, 120)])
    doc.draw.rectangle([0, 0, W, 95], fill=band)
    head = {"uid": "RESIDENT IDENTITY CARD", "tax": "TAX IDENTITY CARD", "voter": "ELECTOR IDENTITY CARD"}[variant]
    doc.text(40, 18, head, 30, "bold", fill=(255, 255, 255))
    doc.text(40, 60, "Demo Registry  -  SPECIMEN FOR RESEARCH USE ONLY", 15, fill=(230, 230, 230))

    if variant == "uid":
        doc.paste(photo(rng, 180, 220, faces_dir), 40, 130, label="FACE")
        x, y = 260, 140
        if lang != "en":
            doc.text(x, y, p["name"][NATIVE[lang]], 26, label="PERSON_NAME")
            y += 45
        doc.text(x, y, p["name"]["en"], 26, "bold", label="PERSON_NAME")
        y += 55
        y = field(doc, x, y, "dob", p["dob"], "DOB", lang, rng, size=20)
        g = GENDER[p["gender"]]
        doc.text(x, y, f"{g[LANG_IDX[lang]]} / {g[0]}" if lang != "en" else g[0], 20)
        num = p["aadhaar"]
        doc.text((W - doc.text_width(num, 42, "bold")) / 2, 470, num, 42, "bold", label="AADHAAR")
        doc.paste(qr_image(rng, f"SPECIMEN|{p['name']['en']}|{p['dob']}|{p['aadhaar'][-4:]}", 170), 800, 140,
                  label="QR_CODE")
        doc.hline(40, W - 40, 560, fill=band, width=3)
        doc.text(40, 575, "Identity proof specimen. Not a government document.", 16, fill=(90, 90, 90))

    elif variant == "tax":
        x, y = 40, 130
        doc.text(x, y, lbl("name", lang, rng), 16, fill=(90, 90, 90))
        name_value(doc, x, y + 26, p["name"], lang, rng, size=24, style="bold")
        y += 80
        doc.text(x, y, lbl("father", lang, rng), 16, fill=(90, 90, 90))
        name_value(doc, x, y + 26, p["father"], lang, rng, size=22)
        y += 80
        doc.text(x, y, lbl("dob", lang, rng), 16, fill=(90, 90, 90))
        doc.text(x, y + 26, p["dob"], 22, label="DOB")
        y += 80
        doc.text(x, y, "Permanent Account Number", 16, fill=(90, 90, 90))
        doc.text(x, y + 26, p["pan"], 36, "bold", label="PAN")
        doc.paste(photo(rng, 170, 210, faces_dir), 790, 130, label="FACE")
        doc.paste(signature(rng, 210, 70), 770, 380, label="SIGNATURE")
        doc.text(800, 455, lbl("signature", lang, rng), 15, fill=(90, 90, 90))

    else:  # voter
        doc.paste(photo(rng, 170, 210, faces_dir), 40, 130, label="FACE")
        x, y = 250, 125
        doc.text(x, y, p["voter"], 30, "bold", label="VOTER_ID")
        y += 55
        y = name_row(doc, x, y, "name", p["name"], lang, rng, x + 200, size=19)
        y = name_row(doc, x, y, "father", p["father"], lang, rng, x + 200, size=19)
        g = GENDER[p["gender"]]
        y = field(doc, x, y, "gender", g[LANG_IDX[lang]], None, lang, rng, size=19)
        y = field(doc, x, y, "dob", p["dob"], "DOB", lang, rng, size=19)
        field(doc, x, y, "address", p["address"], "ADDRESS", lang, rng, size=17)

    doc.img = watermark(doc.img, get_font("latin_bold", 110))
    doc.draw = ImageDraw.Draw(doc.img)
    return doc


# ============================================================ RENT AGREEMENT
def rent_agreement(rng, lang, faces_dir=None):
    W, H, M = 827, 1169, 60
    doc = Doc(W, H, paper(rng), "rent_agreement", lang)
    owner, tenant = make_person(rng), make_person(rng)
    prop_addr, city = address_lines(rng)
    st = "serif" if rng.random() < 0.6 else "regular"
    t = "RENTAL AGREEMENT"
    doc.text((W - doc.text_width(t, 24, "bold")) / 2, 45, t, 24, "bold")
    rent = rng.choice([8, 10, 12, 15, 18, 22, 25, 30, 40]) * 1000
    y = 110
    y = doc.paragraph(M, y, W - 2 * M, [
        ("This Rental Agreement is made and executed on", None), (ids.date_str(rng, 2025, 2026), None),
        ("at", None), (city, None), ("between", None), (owner["name"]["en"], "PERSON_NAME"),
        (", S/o" if owner["gender"] == "M" else rng.choice([", D/o", ", W/o"]), None),
        (owner["father"]["en"], "PERSON_NAME"), (", residing at", None),
        (" ".join(owner["address"]), "ADDRESS"),
        ('(hereinafter called the "Owner") AND', None), (tenant["name"]["en"], "PERSON_NAME"),
        (f", aged {tenant['age']} years, holding PAN", None), (tenant["pan"], "PAN"),
        (", Aadhaar No.", None), (tenant["aadhaar"], "AADHAAR"), (", mobile", None), (tenant["phone"], "PHONE"),
        ('(hereinafter called the "Tenant").', None),
    ], size=16, style=st) + 12
    y = doc.paragraph(M, y, W - 2 * M, [
        ("1. The Owner agrees to let out the residential premises situated at", None),
        (" ".join(prop_addr), "ADDRESS"),
        ("to the Tenant for a period of 11 months commencing from", None), (ids.date_str(rng, 2025, 2026), None),
        (".", None),
    ], size=16, style=st) + 12
    y = doc.paragraph(M, y, W - 2 * M, [
        (f"2. The monthly rent is Rs. {inr(rent)} and the Tenant has paid a security deposit of "
         f"Rs. {inr(rent * rng.choice([2, 3, 5, 6]))}. Rent shall be paid on or before the 5th of every month to "
         f"the Owner's account no.", None), (owner["account"], "BANK_ACCOUNT"),
        (f"({owner['bank_name']}, IFSC", None), (owner["ifsc"], "IFSC"), (") or via UPI to", None),
        (owner["upi"], "UPI_ID"), (".", None),
    ], size=16, style=st) + 12
    for clause in [
        "3. The Tenant shall not sublet the premises or any part thereof without written consent of the Owner.",
        "4. The Tenant shall pay electricity and water charges as per actual consumption.",
        "5. Either party may terminate this agreement by giving one month's written notice to the other party.",
        "6. The Tenant shall hand over vacant possession of the premises in good condition on expiry.",
    ]:
        y = doc.paragraph(M, y, W - 2 * M, [(clause, None)], size=16, style=st) + 8

    y += 25
    doc.text(M, y, "IN WITNESS WHEREOF the parties have signed this agreement.", 16, st)
    y += 50
    for i, (who, person) in enumerate([("OWNER", owner), ("TENANT", tenant)]):
        x = M + i * 380
        doc.paste(signature(rng, 190, 65), x, y, label="SIGNATURE")
        doc.text(x, y + 75, who, 15, "bold")
        doc.text(x, y + 100, person["name"]["en"], 15, label="PERSON_NAME")
    y += 160
    doc.text(M, y, "WITNESSES:", 15, "bold")
    y += 35
    for i in range(2):
        w = make_person(rng)
        x = M + i * 380
        doc.text(x, y, f"{i + 1}.", 15)
        doc.text(x + 25, y, w["name"]["en"], 15, label="PERSON_NAME")
        doc.paste(signature(rng, 160, 55), x + 20, y + 30, label="SIGNATURE")
    return doc


# ============================================================ DISCHARGE SUMMARY
DIAGNOSES = ["Community acquired pneumonia", "Acute gastroenteritis with moderate dehydration",
             "Type 2 diabetes mellitus with hyperglycaemia", "Dengue fever with thrombocytopenia",
             "Right distal radius fracture", "Acute appendicitis, post laparoscopic appendectomy"]
MEDS = ["Tab. Paracetamol 650 mg 1-0-1 x 5 days", "Tab. Pantoprazole 40 mg 1-0-0 x 14 days",
        "Cap. Amoxicillin 500 mg 1-1-1 x 7 days", "Tab. Metformin 500 mg 1-0-1 continue",
        "ORS sachets as advised", "Tab. Cetirizine 10 mg 0-0-1 x 5 days"]


def discharge_summary(rng, lang, faces_dir=None):
    W, H, M = 827, 1169, 50
    doc = Doc(W, H, paper(rng), "discharge_summary", lang)
    p, kin, drp = make_person(rng), make_person(rng), make_person(rng)
    hosp = rng.choice(HOSPITALS)
    loc, city, state, pin = rng.choice(PLACES)
    doc.text((W - doc.text_width(hosp, 24, "bold")) / 2, 35, hosp, 24, "bold", fill=(120, 20, 40))
    sub = f"{loc}, {city} - {pin}  |  Ph: 080-{rng.randint(2000, 4999)}{rng.randint(1000, 9999)}"
    doc.text((W - doc.text_width(sub, 14)) / 2, 72, sub, 14, fill=(90, 90, 90))  # hospital phone: NOT personal
    doc.hline(M, W - M, 100, width=2)
    doc.text((W - doc.text_width("DISCHARGE SUMMARY", 20, "bold")) / 2, 112, "DISCHARGE SUMMARY", 20, "bold")

    lx, lv, rx, rv, y = M, M + 160, 440, 570, 165
    name_row(doc, lx, y, "name", p["name"], lang, rng, lv, size=15)
    raw_field(doc, rx, y, "UHID", ids.mrn(rng), "MRN", rv, size=15)
    y += 30
    raw_field(doc, lx, y, "Age / Sex", f"{p['age']} Y / {p['gender']}", None, lv, size=15)
    raw_field(doc, rx, y, "ABHA No", p["abha"], "ABHA", rv, size=15)
    y += 30
    field(doc, lx, y, "dob", p["dob"], "DOB", lang, rng, lv, size=15)
    field(doc, rx, y, "mobile", p["phone"], "PHONE", lang, rng, rv, size=15)
    y += 30
    y = field(doc, lx, y, "address", p["address"], "ADDRESS", lang, rng, lv, size=15)
    x = doc.text(lx, y, "Emergency Contact:", 15, fill=(60, 60, 60))[0]
    x = doc.text(lv, y, kin["name"]["en"], 15, label="PERSON_NAME")[0]
    x = doc.text(x, y, f"({rng.choice(['Spouse', 'Son', 'Daughter', 'Brother', 'Father'])})", 15)[0]
    doc.text(x + 5, y, kin["phone"], 15, label="PHONE")
    y += 30
    raw_field(doc, lx, y, "Admitted On", ids.date_str(rng, 2025, 2026), None, lv, size=15)
    raw_field(doc, rx, y, "Discharged On", ids.date_str(rng, 2025, 2026), None, rv, size=15)
    y += 30
    x = doc.text(lx, y, "Consultant:", 15, fill=(60, 60, 60))[0]
    x = doc.text(lv, y, "Dr.", 15)[0]
    doc.text(x, y, drp["name"]["en"], 15, label="PERSON_NAME")
    raw_field(doc, rx, y, "Ward / Bed", f"{rng.choice(['General', 'Semi-Private', 'ICU'])} / {rng.randint(1, 60)}",
              None, rv, size=15)
    doc.hline(M, W - M, y + 35)

    y += 55
    diag = rng.choice(DIAGNOSES)
    doc.text(M, y, "Final Diagnosis:", 16, "bold")
    doc.text(M + 160, y, diag, 16)
    y += 40
    doc.text(M, y, "Course in Hospital:", 16, "bold")
    y += 28
    y = doc.paragraph(M, y, W - 2 * M, [
        ("The patient", None), (p["name"]["en"], "PERSON_NAME"),
        (f"was admitted with complaints of fever, weakness and reduced appetite for {rng.randint(2, 10)} days. "
         "Relevant investigations were done and the patient was managed conservatively with IV fluids, "
         "antibiotics and supportive care. The patient improved symptomatically and is being discharged "
         "in a haemodynamically stable condition.", None),
    ], size=15) + 10
    doc.text(M, y, "Medications on Discharge:", 16, "bold")
    y += 30
    for i, med in enumerate(rng.sample(MEDS, rng.randint(3, 5))):
        doc.text(M + 15, y, f"{i + 1}. {med}", 15)
        y += 26
    y += 10
    doc.text(M, y, "Advice:", 16, "bold")
    doc.text(M + 80, y, rng.choice(["Plenty of oral fluids. Review with CBC report.",
                                    "Diabetic diet. Monitor blood sugar twice daily.",
                                    "Rest for one week. Avoid lifting heavy weights."]), 15)
    y += 32
    raw_field(doc, M, y, "Follow-up", ids.date_str(rng, 2025, 2026) + " in OPD", None, M + 110, size=15)

    y = max(y + 60, H - 220)
    doc.paste(signature(rng, 190, 65), W - M - 260, y, label="SIGNATURE")
    x = doc.text(W - M - 260, y + 75, "Dr.", 15)[0]
    doc.text(x, y + 75, drp["name"]["en"], 15, label="PERSON_NAME")
    doc.text(W - M - 260, y + 100, "MBBS, MD (General Medicine)", 14)
    doc.text(W - M - 260, y + 122, f"Reg. No. KMC {rng.randint(10000, 99999)}", 14)
    if rng.random() < 0.6:
        doc.paste(stamp(rng, get_font("latin_bold", 11)), M + 40, y, label="STAMP")
    return doc


TEMPLATES = {
    "loan_application": loan_application,
    "payslip": payslip,
    "id_card": id_card,
    "rent_agreement": rent_agreement,
    "discharge_summary": discharge_summary,
}
