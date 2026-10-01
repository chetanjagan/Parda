/**
 * Small pieces of Python behaviour the pipeline depends on, reproduced exactly.
 * (Text positions are JavaScript string indices; for the Basic Multilingual Plane - all our scripts - they are
 * the same as Python's.)
 */

/** Unicode decimal digits (category Nd) come in runs of ten starting at 0, so the value is the offset in the run. */
const ND = /\p{Nd}/u;

/** Characters Python's str.isdigit() accepts outside category Nd (superscripts, subscripts, circled...). */
const EXTRA_DIGITS: Record<number, number> = (() => {
  const t: Record<number, number> = { 0xb2: 2, 0xb3: 3, 0xb9: 1, 0x2070: 0, 0x24ea: 0 };
  const runs: Array<[number, number, number]> = [ // [first code point, first value, count]
    [0x2074, 4, 6], [0x2080, 0, 10], [0x2460, 1, 9], [0x2474, 1, 9], [0x2488, 1, 9], [0x24f5, 1, 9],
    [0x2776, 1, 9], [0x2780, 1, 9], [0x278a, 1, 9], [0x1369, 1, 9],
  ];
  for (const [cp, v, n] of runs) for (let i = 0; i < n; i++) t[cp + i] = v + i;
  return t;
})();

/** Python's unicodedata.digit(c) for a single character, or null. */
export function digitValue(ch: string): number | null {
  const cp = ch.codePointAt(0)!;
  if (cp >= 0x30 && cp <= 0x39) return cp - 0x30;
  if (cp in EXTRA_DIGITS) return EXTRA_DIGITS[cp];
  if (!ND.test(ch)) return null;
  let k = 0;
  while (k < 10 && ND.test(String.fromCodePoint(cp - k - 1))) k++;
  return k % 10;
}

/** rules.ascii_digits: any script's digits -> 0-9, same length (so positions do not move). */
export function asciiDigits(s: string): string {
  let out = "";
  for (const ch of s) {
    const cp = ch.codePointAt(0)!;
    if (cp > 0x7f) {
      const v = digitValue(ch);
      if (v !== null) {
        out += String(v);
        continue;
      }
    }
    out += ch;
  }
  return out;
}

/** statistics.median */
export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) throw new Error("median of no data");
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** Python's round(x, nd): the exact binary value rounded to nd decimals, ties to even. */
export function pyRound(x: number, nd: number): number {
  if (!Number.isFinite(x) || Math.abs(x) >= 1e21) return x;
  const neg = x < 0;
  const exact = Math.abs(x).toFixed(100); // the exact decimal expansion of the double
  const dot = exact.indexOf(".");
  const intPart = exact.slice(0, dot);
  const frac = exact.slice(dot + 1);
  const keep = frac.slice(0, nd);
  const rest = frac.slice(nd);
  let digits = intPart + keep; // integer string scaled by 10^nd
  const first = rest.charCodeAt(0) - 48;
  const tail = rest.slice(1);
  const tie = first === 5 && /^0*$/.test(tail);
  const lastEven = (digits.charCodeAt(digits.length - 1) - 48) % 2 === 0;
  if (first > 5 || (first === 5 && !tie) || (tie && !lastEven)) digits = addOne(digits);
  const s = nd > 0 ? digits.slice(0, digits.length - nd) + "." + digits.slice(digits.length - nd) : digits;
  const v = parseFloat(s);
  return neg ? -v : v;
}

function addOne(d: string): string {
  const a = d.split("");
  let i = a.length - 1;
  while (i >= 0) {
    if (a[i] === "9") {
      a[i] = "0";
      i--;
    } else {
      a[i] = String.fromCharCode(a[i].charCodeAt(0) + 1);
      return a.join("");
    }
  }
  return "1" + a.join("");
}

/** Stable sort by a tuple key (Python's sorted(key=...)). */
export function sortedBy<T>(xs: T[], key: (x: T) => Array<number | string>): T[] {
  const cmp = (a: Array<number | string>, b: Array<number | string>): number => {
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      if (a[i] < b[i]) return -1;
      if (a[i] > b[i]) return 1;
    }
    return a.length - b.length;
  };
  return xs.map((x, i) => ({ x, k: key(x), i }))
    .sort((p, q) => cmp(p.k, q.k) || p.i - q.i)
    .map((p) => p.x);
}
