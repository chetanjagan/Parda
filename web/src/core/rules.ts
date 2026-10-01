/** Regex + checksum PII detection (parda/pii/rules.py), on text whose digits are made ASCII first. */
import { asciiDigits } from "./pycompat.js";

export interface Span { label: string; start: number; end: number; score: number }

const D = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
  [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
  [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];
const P = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
  [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
  [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

/** Aadhaar's Verhoeff checksum (spaces and hyphens ignored). */
export function verhoeffValid(num: string): boolean {
  const n = num.replace(/ /g, "").replace(/-/g, "");
  if (!/^[0-9]+$/.test(n)) return false;
  let c = 0;
  const digits = n.split("").reverse();
  for (let i = 0; i < digits.length; i++) c = D[c][P[i % 8][digits[i].charCodeAt(0) - 48]];
  return c === 0;
}

const GST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/** GSTIN check character for the first 14 characters; null if a character is not allowed. */
export function gstinCheckChar(s14: string): string | null {
  let total = 0;
  for (let i = 0; i < s14.length; i++) {
    const v = GST.indexOf(s14[i]);
    if (v < 0) return null;
    const prod = v * (i % 2 === 0 ? 1 : 2);
    total += Math.floor(prod / 36) + (prod % 36);
  }
  return GST[(36 - (total % 36)) % 36];
}

const B = "(?<![A-Za-z0-9])";
const E = "(?![A-Za-z0-9])";
type Check = ((m: string) => boolean) | null;
const PATTERNS: Array<[string, string, Check]> = [
  ["ABHA", B + "\\d{2}-\\d{4}-\\d{4}-\\d{4}" + E, null],
  ["AADHAAR", B + "[2-9]\\d{3}[ -]?\\d{4}[ -]?\\d{4}" + E, (m) => verhoeffValid(m.replace(/\D/g, ""))],
  ["GSTIN", B + "\\d{2}[A-Z]{5}\\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]" + E, (m) => gstinCheckChar(m.slice(0, 14)) === m[14]],
  ["PAN", B + "[A-Z]{3}[ABCFGHLJPT][A-Z]\\d{4}[A-Z]" + E, null],
  ["IFSC", B + "[A-Z]{4}0[A-Z0-9]{6}" + E, null],
  ["EMAIL", B + "[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)+" + E, null],
  ["UPI_ID", B + "[A-Za-z0-9._-]{2,}@[A-Za-z]{2,}" + "(?![A-Za-z0-9.])", null],
  ["PHONE", "(?:\\+91[ -]?)?" + B + "[6-9]\\d{4}[ -]?\\d{5}" + E, null],
  ["VOTER_ID", B + "[A-Z]{3}\\d{7}" + E, null],
  ["PASSPORT", B + "[A-PR-WYZ][1-9]\\d{6}" + E, null],
  ["UAN", B + "10\\d{10}" + E, null],
  ["VEHICLE_REG", B + "[A-Z]{2}[ -]?\\d{1,2}[ -]?[A-Z]{1,2}[ -]?\\d{4}" + E, null],
];
const COMPILED = PATTERNS.map(([lab, p, check]) => [lab, new RegExp(p, "g"), check] as const);

/** Non-overlapping matches: earlier position first, longer span wins, then pattern order. */
export function findRules(text: string): Span[] {
  const norm = asciiDigits(text);
  const cands: Span[] = [];
  for (const [label, rx, check] of COMPILED) {
    rx.lastIndex = 0;
    for (const m of norm.matchAll(rx)) {
      if (check && !check(m[0])) continue;
      cands.push({ label, start: m.index!, end: m.index! + m[0].length, score: 1.0 });
    }
  }
  cands.sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start)); // stable: ties keep pattern order
  const out: Span[] = [];
  let lastEnd = -1;
  for (const c of cands) {
    if (c.start >= lastEnd) {
      out.push(c);
      lastEnd = c.end;
    }
  }
  return out;
}
