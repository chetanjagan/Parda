/** Span merging, windowed prediction and union of predictors (parda/pii/predict.py). */
import { toCode } from "./labels.js";
import { sortedBy } from "./pycompat.js";
import type { Span } from "./rules.js";
import { tokenize, windows } from "./spans.js";

/** Merge overlapping/adjacent spans with the same label; keep the best score. */
export function mergeSpans(spans: Span[]): Span[] {
  const out: Span[] = [];
  for (const sp of sortedBy(spans, (s) => [s.label, s.start, s.end])) {
    const last = out[out.length - 1];
    if (last && last.label === sp.label && sp.start <= last.end) {
      last.end = Math.max(last.end, sp.end);
      last.score = Math.max(last.score, sp.score);
    } else {
      out.push({ ...sp });
    }
  }
  return sortedBy(out, (s) => [s.start, s.end]);
}

/** A model's raw entity on a chunk of text (natural-language label, chunk offsets). */
export interface RawEntity { label: string; start: number; end: number; score?: number }
export type ChunkModel = (chunk: string) => Promise<RawEntity[]>;

/** GLiNERPredictor.predict: slide a window over the page, map entities back to page offsets, merge. */
export async function windowedPredict(text: string, model: ChunkModel, window = 200, stride = 150): Promise<Span[]> {
  const toks = tokenize(text);
  const spans: Span[] = [];
  for (const [s, e] of windows(toks.length, window, stride)) {
    if (s >= e) continue;
    const base = toks[s][1];
    const chunk = text.slice(base, toks[e - 1][2]);
    for (const ent of await model(chunk)) {
      const a = base + Math.trunc(ent.start);
      const b = base + Math.trunc(ent.end);
      if (a >= 0 && a < b && b <= text.length) {
        spans.push({ label: toCode(ent.label), start: a, end: b, score: ent.score ?? 1.0 });
      }
    }
  }
  return mergeSpans(spans);
}

/** Model + rules: anything either flags gets redacted. */
export function unionSpans(...lists: Span[][]): Span[] {
  return mergeSpans(lists.flat());
}
