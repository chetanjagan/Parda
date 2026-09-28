"""Seed name/place lists in English, Hindi (Devanagari) and Kannada.

Starter list only. Later we'll expand names using AI4Bharat Naamapadam
(PER entities) so the model sees thousands of real-looking names.
"""
import random

from . import ids

# (english, hindi, kannada, gender)
FIRST_NAMES = [
    ("Ramesh", "रमेश", "ರಮೇಶ್", "M"), ("Suresh", "सुरेश", "ಸುರೇಶ್", "M"),
    ("Mahesh", "महेश", "ಮಹೇಶ್", "M"), ("Ganesh", "गणेश", "ಗಣೇಶ್", "M"),
    ("Rajesh", "राजेश", "ರಾಜೇಶ್", "M"), ("Arjun", "अर्जुन", "ಅರ್ಜುನ್", "M"),
    ("Vikram", "विक्रम", "ವಿಕ್ರಮ್", "M"), ("Rahul", "राहुल", "ರಾಹುಲ್", "M"),
    ("Amit", "अमित", "ಅಮಿತ್", "M"), ("Kiran", "किरण", "ಕಿರಣ್", "M"),
    ("Manoj", "मनोज", "ಮನೋಜ್", "M"), ("Deepak", "दीपक", "ದೀಪಕ್", "M"),
    ("Anil", "अनिल", "ಅನಿಲ್", "M"), ("Sunil", "सुनील", "ಸುನೀಲ್", "M"),
    ("Prakash", "प्रकाश", "ಪ್ರಕಾಶ್", "M"), ("Naveen", "नवीन", "ನವೀನ್", "M"),
    ("Chetan", "चेतन", "ಚೇತನ್", "M"), ("Harish", "हरीश", "ಹರೀಶ್", "M"),
    ("Priya", "प्रिया", "ಪ್ರಿಯಾ", "F"), ("Anita", "अनीता", "ಅನಿತಾ", "F"),
    ("Sunita", "सुनीता", "ಸುನೀತಾ", "F"), ("Kavya", "काव्या", "ಕಾವ್ಯಾ", "F"),
    ("Lakshmi", "लक्ष्मी", "ಲಕ್ಷ್ಮಿ", "F"), ("Meena", "मीना", "ಮೀನಾ", "F"),
    ("Pooja", "पूजा", "ಪೂಜಾ", "F"), ("Divya", "दिव्या", "ದಿವ್ಯಾ", "F"),
    ("Asha", "आशा", "ಆಶಾ", "F"), ("Rekha", "रेखा", "ರೇಖಾ", "F"),
    ("Shreya", "श्रेया", "ಶ್ರೇಯಾ", "F"), ("Neha", "नेहा", "ನೇಹಾ", "F"),
    ("Geetha", "गीता", "ಗೀತಾ", "F"), ("Savitha", "सविता", "ಸವಿತಾ", "F"),
]

SURNAMES = [
    ("Kumar", "कुमार", "ಕುಮಾರ್"), ("Sharma", "शर्मा", "ಶರ್ಮಾ"), ("Rao", "राव", "ರಾವ್"),
    ("Reddy", "रेड्डी", "ರೆಡ್ಡಿ"), ("Gowda", "गौड़ा", "ಗೌಡ"), ("Patil", "पाटिल", "ಪಾಟೀಲ್"),
    ("Nair", "नायर", "ನಾಯರ್"), ("Iyer", "अय्यर", "ಅಯ್ಯರ್"), ("Singh", "सिंह", "ಸಿಂಗ್"),
    ("Verma", "वर्मा", "ವರ್ಮಾ"), ("Gupta", "गुप्ता", "ಗುಪ್ತಾ"), ("Joshi", "जोशी", "ಜೋಶಿ"),
    ("Shetty", "शेट्टी", "ಶೆಟ್ಟಿ"), ("Hegde", "हेगड़े", "ಹೆಗ್ಡೆ"), ("Naik", "नाइक", "ನಾಯಕ್"),
    ("Kulkarni", "कुलकर्णी", "ಕುಲಕರ್ಣಿ"), ("Bhat", "भट", "ಭಟ್"), ("Desai", "देसाई", "ದೇಸಾಯಿ"),
]

# (locality, city, state, pincode)
PLACES = [
    ("Jayanagar", "Bengaluru", "Karnataka", "560041"), ("Koramangala", "Bengaluru", "Karnataka", "560034"),
    ("Indiranagar", "Bengaluru", "Karnataka", "560038"), ("Malleshwaram", "Bengaluru", "Karnataka", "560003"),
    ("Whitefield", "Bengaluru", "Karnataka", "560066"), ("HSR Layout", "Bengaluru", "Karnataka", "560102"),
    ("Basavanagudi", "Bengaluru", "Karnataka", "560004"), ("Rajajinagar", "Bengaluru", "Karnataka", "560010"),
    ("Vijayanagar", "Mysuru", "Karnataka", "570017"), ("Vidyanagar", "Hubballi", "Karnataka", "580021"),
    ("Andheri West", "Mumbai", "Maharashtra", "400053"), ("Kothrud", "Pune", "Maharashtra", "411038"),
    ("Lajpat Nagar", "New Delhi", "Delhi", "110024"), ("T Nagar", "Chennai", "Tamil Nadu", "600017"),
    ("Banjara Hills", "Hyderabad", "Telangana", "500034"), ("Navrangpura", "Ahmedabad", "Gujarat", "380009"),
]
STREETS = ["Main Road", "Cross", "Street", "Layout", "Nagar", "Road", "Colony", "Extension"]

# Invented organisation names (non-PII context words)
ORGS = [
    "Kaveri Nidhi Finance Ltd.", "Sahyog Credit Pvt. Ltd.", "Nandi Microfinance Services",
    "Tunga Finserv Pvt. Ltd.", "Brindavan Softworks Pvt. Ltd.", "Deccan Logistics Pvt. Ltd.",
    "Sahyadri Textiles Ltd.", "Malnad Agro Foods Pvt. Ltd.",
]
HOSPITALS = ["Sanjeevani Multispeciality Hospital", "Arogya Nursing Home", "Chaitanya Hospital & Research Centre",
             "Shushrusha Medical Centre"]


def ordinal(n: int) -> str:
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def address_lines(rng: random.Random):
    loc, city, state, pin = rng.choice(PLACES)
    house = rng.choice([f"#{rng.randint(1, 450)}", f"No. {rng.randint(1, 450)}", f"Flat {rng.randint(101, 1204)}"])
    street = f"{ordinal(rng.randint(1, 18))} {rng.choice(STREETS)}" if rng.random() < 0.6 else rng.choice(STREETS)
    line1 = f"{house}, {street}, {loc},"
    line2 = f"{city}, {state} - {pin}"
    return [line1, line2], city


def make_person(rng: random.Random) -> dict:
    f_en, f_hi, f_kn, gender = rng.choice(FIRST_NAMES)
    s_en, s_hi, s_kn = rng.choice(SURNAMES)
    fa_en, fa_hi, fa_kn, _ = rng.choice([n for n in FIRST_NAMES if n[3] == "M"])
    b = ids.bank(rng)
    addr, city = address_lines(rng)
    return {
        "first": f_en, "last": s_en, "gender": gender,
        "name": {"en": f"{f_en} {s_en}", "hi": f"{f_hi} {s_hi}", "kn": f"{f_kn} {s_kn}"},
        "father": {"en": f"{fa_en} {s_en}", "hi": f"{fa_hi} {s_hi}", "kn": f"{fa_kn} {s_kn}"},
        "dob": ids.dob(rng),
        "aadhaar": ids.aadhaar(rng),
        "pan": ids.pan(rng, surname=s_en),
        "phone": ids.phone(rng),
        "email": ids.email(rng, f_en, s_en),
        "address": addr, "city": city,
        "bank_name": b[1],
        "account": ids.account_number(rng),
        "ifsc": ids.ifsc(rng, b[0]),
        "upi": ids.upi_id(rng, f_en, s_en, b[2]),
        "voter": ids.voter_epic(rng),
        "passport": ids.passport(rng),
        "vehicle": ids.vehicle_reg(rng),
        "uan": ids.uan(rng),
        "abha": ids.abha_number(rng),
        "age": rng.randint(21, 75),
    }
