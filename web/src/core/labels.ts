/** PII label codes and the natural-language names GLiNER is prompted with (parda/pii/labels.py). */
export const LABELS: Record<string, string> = {
  PERSON_NAME: "person name", AADHAAR: "aadhaar number", PAN: "pan number", PHONE: "phone number",
  EMAIL: "email address", ADDRESS: "address", DOB: "date of birth", BANK_ACCOUNT: "bank account number",
  IFSC: "ifsc code", UPI_ID: "upi id", GSTIN: "gstin number", VOTER_ID: "voter id number",
  PASSPORT: "passport number", VEHICLE_REG: "vehicle registration number", UAN: "uan number",
  ABHA: "abha health id", EMPLOYEE_ID: "employee id", MRN: "medical record number",
};
const NAT2CODE: Record<string, string> = Object.fromEntries(Object.entries(LABELS).map(([k, v]) => [v, k]));

export function toCode(name: string): string {
  return NAT2CODE[name.trim().toLowerCase()] ?? name.trim().toUpperCase();
}
