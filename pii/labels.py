"""PII label codes and the natural-language names GLiNER is trained/prompted with.
GLiNER reads the label text, so descriptive names work better than codes."""

LABELS = {
    "PERSON_NAME": "person name",
    "AADHAAR": "aadhaar number",
    "PAN": "pan number",
    "PHONE": "phone number",
    "EMAIL": "email address",
    "ADDRESS": "address",
    "DOB": "date of birth",
    "BANK_ACCOUNT": "bank account number",
    "IFSC": "ifsc code",
    "UPI_ID": "upi id",
    "GSTIN": "gstin number",
    "VOTER_ID": "voter id number",
    "PASSPORT": "passport number",
    "VEHICLE_REG": "vehicle registration number",
    "UAN": "uan number",
    "ABHA": "abha health id",
    "EMPLOYEE_ID": "employee id",
    "MRN": "medical record number",
}
NAT2CODE = {v: k for k, v in LABELS.items()}


def to_code(name: str) -> str:
    return NAT2CODE.get(name.strip().lower(), name.strip().upper())
