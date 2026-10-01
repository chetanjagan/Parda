/**
 * GLiNER's span decoding (gliner 0.2.29, BaseSpanDecoder with flat_ner, batch size 1), verified on recorded scores:
 * sigmoid (float32) -> keep scores > threshold, in (start, width, class) order -> drop spans past the last word ->
 * sort by score (highest first, stable) -> keep a span only if it overlaps none already kept -> sort by start.
 */
export interface WordSpan { start: number; end: number; label: string; score: number } // word indices, end inclusive

const f32 = Math.fround;

export function decodeSpans(logits: Float32Array | number[], dims: number[], nWords: number, labels: string[],
  threshold = 0.3): WordSpan[] {
  const [, Lp, K, C] = dims;
  const thr = f32(threshold);
  const cands: WordSpan[] = [];
  for (let s = 0; s < Math.min(Lp, nWords); s++) {
    for (let k = 0; k < K; k++) {
      for (let c = 0; c < C; c++) {
        const x = logits[(s * K + k) * C + c];
        const p = f32(1 / (1 + Math.exp(-x)));
        if (p > thr && s + k + 1 <= nWords) cands.push({ start: s, end: s + k, label: labels[c], score: p });
      }
    }
  }
  const byScore = cands.map((c, i) => ({ c, i })).sort((a, b) => b.c.score - a.c.score || a.i - b.i).map((x) => x.c);
  const kept: WordSpan[] = [];
  for (const sp of byScore) {
    const overlaps = kept.some((t) => (sp.start === t.start && sp.end === t.end) || !(sp.start > t.end || t.start > sp.end));
    if (!overlaps) kept.push(sp);
  }
  return kept.map((c, i) => ({ c, i })).sort((a, b) => a.c.start - b.c.start || a.i - b.i).map((x) => x.c);
}

/** Word spans -> character spans of the chunk (start of the first word, end of the last). */
export function toCharSpans(spans: WordSpan[], words: Array<[string, number, number]>) {
  return spans.map((s) => ({ start: words[s.start][1], end: words[s.end][2], label: s.label, score: s.score }));
}
