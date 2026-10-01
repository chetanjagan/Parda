/** Tokens, windows and reading order of OCR words (parda/pii/spans.py). */
import { median, sortedBy } from "./pycompat.js";

export type BBox = [number, number, number, number];
export interface Segment { text: string; bbox: BBox; score?: number }

/** GLiNER's whitespace splitter: Python's \w is [\p{L}\p{N}_] (marks such as Hindi vowel signs are not \w). */
const TOKEN_RE = /[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|\S/gu;

export function tokenize(text: string): Array<[string, number, number]> {
  return [...text.matchAll(TOKEN_RE)].map((m) => [m[0], m.index!, m.index! + m[0].length]);
}

/** Overlapping [start, end) token windows covering n tokens. */
export function windows(n: number, window = 200, stride = 150): Array<[number, number]> {
  if (n <= window) return [[0, n]];
  const out: Array<[number, number]> = [];
  let s = 0;
  for (;;) {
    const e = Math.min(s + window, n);
    out.push([s, e]);
    if (e === n) return out;
    s += stride;
  }
}

const yc = (s: Segment) => (s.bbox[1] + s.bbox[3]) / 2;
const xc = (s: Segment) => (s.bbox[0] + s.bbox[2]) / 2;

function chainLines(items: Segment[], h: number, slope: number): Segment[][] {
  const tall = (x: Segment) => x.bbox[3] - x.bbox[1] >= 0.6 * h;
  const lines: Segment[][] = [];
  const anchors: Segment[] = [];
  for (const s of items) {
    let best: number | null = null;
    let bestD: number | null = null;
    lines.forEach((ln, li) => {
      const last = ln[ln.length - 1];
      const anc = anchors[li];
      if (s.bbox[0] < last.bbox[2] - 0.5 * h) return;
      const pred = yc(anc) + slope * (xc(s) - xc(anc));
      const d = Math.abs(yc(s) - pred);
      if (d < (tall(s) ? 0.5 * h : 0.8 * h) && (bestD === null || d < bestD)) {
        best = li;
        bestD = d;
      }
    });
    if (best === null) {
      lines.push([s]);
      anchors.push(s);
    } else {
      lines[best].push(s);
      if (tall(s) || !tall(anchors[best])) anchors[best] = s;
    }
  }
  return lines;
}

function estimateSlope(lines: Segment[][]): number {
  const slopes: number[] = [];
  for (const ln of lines) {
    if (ln.length >= 3) {
      const dx = xc(ln[ln.length - 1]) - xc(ln[0]);
      if (dx > 50) slopes.push((yc(ln[ln.length - 1]) - yc(ln[0])) / dx);
    }
  }
  return slopes.length ? median(slopes) : 0.0;
}

/** Order OCR segments into lines (top->bottom, left->right, following the page tilt) and join them.
 *  Returns the text, the segments in reading order, and each one's [start, end) in the text. */
export function buildOcrText(segsIn: Segment[]): { text: string; ordered: Segment[]; offs: Array<[number, number]> } {
  const segs = segsIn.filter((s) => (s.text ?? "").trim());
  if (!segs.length) return { text: "", ordered: [], offs: [] };
  const h = median(segs.map((s) => Math.max(1, s.bbox[3] - s.bbox[1])));
  const items = sortedBy(segs, (s) => [s.bbox[0], yc(s)]);
  const slope = estimateSlope(chainLines(items, h, 0.0));
  const lines = chainLines(items, h, slope);
  const key = (ln: Segment[]) => {
    const t = ln.filter((x) => x.bbox[3] - x.bbox[1] >= 0.6 * h);
    const use = t.length ? t : ln;
    return [median(use.map((x) => yc(x) - slope * xc(x))), xc(ln[0])];
  };
  const sortedLines = sortedBy(lines, key);
  let text = "";
  const ordered: Segment[] = [];
  const offs: Array<[number, number]> = [];
  sortedLines.forEach((ln, li) => {
    if (li) text += "\n";
    ln.forEach((s, j) => {
      if (j) text += " ";
      const t = s.text.trim();
      ordered.push(s);
      offs.push([text.length, text.length + t.length]);
      text += t;
    });
  });
  return { text, ordered, offs };
}
