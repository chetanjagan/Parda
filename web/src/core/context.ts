/** Context rules after the model (parda/pii/context.py, the reference; identical answers checked by the goldens).
 *  dob: a date of birth needs a birth label before it on its own line. aadhaar: a 12-digit Aadhaar without spaces
 *  needs an Aadhaar label there. cut: a span that starts or ends inside a number is dropped. They only remove spans. */
import { asciiDigits } from "./pycompat.js";
import type { Span } from "./rules.js";

export type ContextRule = "dob" | "aadhaar" | "cut";
export const CONTEXT_RULES: ContextRule[] = ["dob", "aadhaar", "cut"];
const WINDOW = 60;
const DOB_WORDS = ["birth", "dob", "d.o.b", "born", "जन्म", "ಜನ್ಮ"];
const AADHAAR_WORDS = ["aadhaar", "aadhar", "adhaar", "uid", "आधार", "ಆಧಾರ"];
const isDigit = (c: string | undefined) => c !== undefined && c >= "0" && c <= "9" && c.length === 1;

function before(text: string, start: number): string {
  const lineStart = text.lastIndexOf("\n", start - 1) + 1;
  return text.slice(Math.max(lineStart, start - WINDOW), start).toLowerCase();
}

export function cutsNumber(norm: string, start: number, end: number): boolean {
  const n = norm.length;
  const left = 0 < start && start < n && isDigit(norm[start]) &&
    (isDigit(norm[start - 1]) || ((norm[start - 1] === "," || norm[start - 1] === ".") && start >= 2 && isDigit(norm[start - 2])));
  const right = 0 < end && end < n && isDigit(norm[end - 1]) &&
    (isDigit(norm[end]) || ((norm[end] === "," || norm[end] === ".") && end + 1 < n && isDigit(norm[end + 1])));
  return left || right;
}

export function filterSpans(text: string, spans: Span[], rules: ContextRule[] = CONTEXT_RULES): Span[] {
  const norm = asciiDigits(text);
  return spans.filter((sp) => {
    const { start: s, end: e, label } = sp;
    if (rules.includes("cut") && cutsNumber(norm, s, e)) return false;
    if (rules.includes("dob") && label === "DOB" && !DOB_WORDS.some((w) => before(text, s).includes(w))) return false;
    if (rules.includes("aadhaar") && label === "AADHAAR" && /^[0-9]{12}$/.test(norm.slice(s, e))
      && !AADHAAR_WORDS.some((w) => before(text, s).includes(w))) return false;
    return true;
  });
}
