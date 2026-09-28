"""Fake-but-valid-format Indian identifiers.

Everything here is randomly generated. Numbers follow the public format and
checksum rules so the model learns realistic patterns, but they are NOT linked
to any real person. Never mix these with real data.
"""
import random
import string

# ---------------- Verhoeff checksum (used by Aadhaar) ----------------
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_check_digit(num: str) -> int:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return _INV[c]


def verhoeff_valid(num: str) -> bool:
    num = num.replace(" ", "").replace("-", "")
    if not num.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


# ---------------- Aadhaar ----------------
def aadhaar(rng: random.Random, fmt: str = "spaced") -> str:
    """12 digits, first digit 2-9, last digit Verhoeff checksum."""
    body = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    num = body + str(verhoeff_check_digit(body))
    if fmt == "spaced":
        return f"{num[:4]} {num[4:8]} {num[8:]}"
    if fmt == "dashed":
        return f"{num[:4]}-{num[4:8]}-{num[8:]}"
    if fmt == "masked":
        return f"XXXX XXXX {num[8:]}"
    return num


def fake_12_digit_non_aadhaar(rng: random.Random) -> str:
    """12-digit number that FAILS Verhoeff -> used as a hard negative (e.g. reference no.)."""
    while True:
        num = "".join(str(rng.randint(0, 9)) for _ in range(12))
        if not verhoeff_valid(num):
            return f"{num[:4]} {num[4:8]} {num[8:]}"


# ---------------- PAN ----------------
def pan(rng: random.Random, surname: str = "", entity: str = "P") -> str:
    """AAAPL1234C : 3 letters, entity type, surname initial, 4 digits, letter."""
    L = string.ascii_uppercase
    first3 = "".join(rng.choice(L) for _ in range(3))
    initial = surname[0].upper() if surname and surname[0].isalpha() and surname[0].isascii() else rng.choice(L)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(4))
    return f"{first3}{entity}{initial}{digits}{rng.choice(L)}"


# ---------------- GSTIN ----------------
_GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GST_STATE_CODES = ["29", "27", "07", "33", "36", "24", "09", "32", "19", "06"]


def gstin_check_char(s14: str) -> str:
    total = 0
    for i, ch in enumerate(s14):
        prod = _GST_CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += prod // 36 + prod % 36
    return _GST_CHARS[(36 - total % 36) % 36]


def gstin(rng: random.Random) -> str:
    s14 = rng.choice(GST_STATE_CODES) + pan(rng, entity=rng.choice("CFP")) + str(rng.randint(1, 9)) + "Z"
    return s14 + gstin_check_char(s14)


# ---------------- Banking ----------------
BANKS = [  # (IFSC prefix, display name, UPI handle)
    ("SBIN", "State Bank of India", "oksbi"),
    ("HDFC", "HDFC Bank", "okhdfcbank"),
    ("ICIC", "ICICI Bank", "okicici"),
    ("UTIB", "Axis Bank", "okaxis"),
    ("KKBK", "Kotak Mahindra Bank", "kotak"),
    ("CNRB", "Canara Bank", "cnrb"),
    ("BARB", "Bank of Baroda", "barodampay"),
    ("PUNB", "Punjab National Bank", "pnb"),
]
UPI_GENERIC = ["ybl", "ibl", "axl", "paytm", "upi"]


def bank(rng: random.Random):
    return rng.choice(BANKS)


def ifsc(rng: random.Random, prefix: str) -> str:
    return prefix + "0" + "".join(rng.choice(string.digits) for _ in range(6))


def account_number(rng: random.Random) -> str:
    n = rng.choice([9, 11, 12, 14, 15, 16])
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


def upi_id(rng: random.Random, first: str, last: str, handle: str) -> str:
    base = rng.choice([first.lower(), f"{first.lower()}.{last.lower()}", f"{first.lower()}{last.lower()[:1]}"])
    if rng.random() < 0.5:
        base += str(rng.randint(1, 999))
    if rng.random() < 0.3:
        base = "".join(str(rng.randint(0, 9)) for _ in range(10))  # phone-number UPI
    return f"{base}@{handle if rng.random() < 0.6 else rng.choice(UPI_GENERIC)}"


# ---------------- Contact ----------------
def phone(rng: random.Random) -> str:
    num = str(rng.randint(6, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(9))
    style = rng.random()
    if style < 0.35:
        return f"{num[:5]} {num[5:]}"
    if style < 0.6:
        return f"+91 {num}"
    if style < 0.75:
        return f"+91-{num[:5]}-{num[5:]}"
    return num


def email(rng: random.Random, first: str, last: str) -> str:
    dom = rng.choice(["gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com", "hotmail.com"])
    user = rng.choice([f"{first}.{last}", f"{first}{last}", f"{first}_{last[:1]}", f"{first}{rng.randint(1, 99)}"])
    return f"{user.lower()}@{dom}"


# ---------------- Other IDs ----------------
def voter_epic(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(3)) + "".join(
        str(rng.randint(0, 9)) for _ in range(7))


def passport(rng: random.Random) -> str:
    return rng.choice("ABCDEFGHJKLMNPRSTUVWZ") + str(rng.randint(1, 9)) + "".join(
        str(rng.randint(0, 9)) for _ in range(6))


def vehicle_reg(rng: random.Random) -> str:
    st = rng.choice(["KA", "MH", "DL", "TN", "TS", "KL", "AP", "GJ"])
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.choice([1, 2])))
    return f"{st} {rng.randint(1, 60):02d} {letters} {rng.randint(1, 9999):04d}"


def uan(rng: random.Random) -> str:  # EPFO Universal Account Number
    return "10" + "".join(str(rng.randint(0, 9)) for _ in range(10))


def abha_number(rng: random.Random) -> str:  # ABHA: 14 digits, shown as XX-XXXX-XXXX-XXXX
    d = "".join(str(rng.randint(0, 9)) for _ in range(14))
    return f"{d[:2]}-{d[2:6]}-{d[6:10]}-{d[10:]}"


def employee_id(rng: random.Random) -> str:
    return rng.choice(["EMP", "E", "KNC", "ID"]) + str(rng.randint(1000, 99999))


def mrn(rng: random.Random) -> str:  # hospital record / UHID
    return rng.choice(["UHID", "MRN", "IP"]) + "/" + str(rng.choice([2024, 2025, 2026])) + "/" + str(
        rng.randint(10000, 999999))


# ---------------- Dates ----------------
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def date_str(rng: random.Random, y0: int, y1: int) -> str:
    y, m, d = rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28)
    s = rng.random()
    if s < 0.45:
        return f"{d:02d}/{m:02d}/{y}"
    if s < 0.7:
        return f"{d:02d}-{m:02d}-{y}"
    return f"{d:02d} {MONTHS[m - 1]} {y}"


def dob(rng: random.Random) -> str:
    return date_str(rng, 1955, 2005)
