/** Gender (optional, off by default): only a value right after a gender label on the same line
 *  ("Gender: Female", "लिंग: महिला", "ಲಿಂಗ: ಮಹಿಳೆ"), so the word elsewhere in a sentence is left alone. */
import type { Span } from "../core/rules.js";

const LABEL = /(?<![\p{L}\p{M}])(gender|sex|लिंग|ಲಿಂಗ)(?![\p{L}\p{M}])/giu;
const VALUE = /(?<![\p{L}\p{M}])(transgender|female|male|other|m|f|t|पुरुष|महिला|स्त्री|अन्य|ಪುರುಷ|ಮಹಿಳೆ|ಸ್ತ್ರೀ|ಇತರೆ)(?![\p{L}\p{M}])/giu;
const LOOK = 40; // characters after the label, on its line

export function genderSpans(text: string): Span[] {
  const out = new Map<number, Span>();
  for (const m of text.matchAll(LABEL)) {
    const from = m.index! + m[0].length;
    const lineEnd = text.indexOf("\n", from);
    const seg = text.slice(from, Math.min(lineEnd < 0 ? text.length : lineEnd, from + LOOK));
    for (const v of seg.matchAll(VALUE)) {
      const start = from + v.index!;
      out.set(start, { label: "GENDER", start, end: start + v[0].length, score: 1.0 });
    }
  }
  return [...out.values()].sort((a, b) => a.start - b.start);
}
