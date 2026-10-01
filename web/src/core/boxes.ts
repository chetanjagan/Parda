/** Text spans -> pixel boxes, gap filling, pixel rounding, redaction rectangles (parda/pipeline/boxes.py). */
import { sortedBy } from "./pycompat.js";
import type { BBox, Segment } from "./spans.js";

export interface Region { kind: "text" | "visual"; label: string; bbox: BBox; score: number; source: string }

/** Safety margin (px) around each redaction box. */
export const PAD: Record<string, number> = { text: 3, visual: 6 };

/** Pixel boxes covering text[start:end]; a partial line-level segment is cut by character position (+1 char each side). */
export function spanBoxes(start: number, end: number, ordered: Segment[], offs: Array<[number, number]>): BBox[] {
  const out: BBox[] = [];
  ordered.forEach((seg, i) => {
    const [a, b] = offs[i];
    const lo = Math.max(a, start);
    const hi = Math.min(b, end);
    if (lo >= hi) return;
    const [x0, y0, x1, y1] = seg.bbox.map(Number) as BBox;
    const n = b - a;
    if ((lo === a && hi === b) || n <= 0) {
      out.push([x0, y0, x1, y1]);
      return;
    }
    const w = (x1 - x0) / n;
    out.push([Math.max(x0, x0 + (lo - a) * w - w), y0, Math.min(x1, x0 + (hi - a) * w + w), y1]);
  });
  return out;
}

function sameLine(a: BBox, b: BBox): boolean {
  const ov = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  return ov >= 0.5 * Math.min(a[3] - a[1], b[3] - b[1]);
}

/** Join same-label text boxes on one line with a small gap (<= maxGap x line height). Visual regions untouched. */
export function fillGaps(regions: Region[], maxGap = 2.5): Region[] {
  const text = regions.filter((r) => r.kind === "text").map((r) => ({ ...r, bbox: [...r.bbox] as BBox }));
  const other = regions.filter((r) => r.kind !== "text");
  const out: Region[] = [];
  for (const label of [...new Set(text.map((r) => r.label))].sort()) {
    const boxes = sortedBy(text.filter((r) => r.label === label), (r) => [r.bbox[1], r.bbox[0]]);
    const merged: Region[] = [];
    for (const r of boxes) {
      let joined = false;
      for (const m of merged) {
        const a = m.bbox;
        const b = r.bbox;
        const h = Math.max(a[3] - a[1], b[3] - b[1]);
        const gap = Math.max(a[0], b[0]) - Math.min(a[2], b[2]);
        if (sameLine(a, b) && gap <= maxGap * h) {
          m.bbox = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
          m.score = Math.max(m.score, r.score);
          joined = true;
          break;
        }
      }
      if (!joined) merged.push(r);
    }
    out.push(...merged);
  }
  return [...out, ...other];
}

/** Float box -> integer pixel range [x0, y0, x1, y1]; eps ignores float noise (99.9999 is pixel 100). */
export function pix(box: BBox, W: number, H: number, pad = 0, eps = 1e-3): BBox {
  const x0 = Math.max(0, Math.floor(box[0] - pad + eps));
  const y0 = Math.max(0, Math.floor(box[1] - pad + eps));
  const x1 = Math.min(W, Math.ceil(box[2] + pad - eps));
  const y1 = Math.min(H, Math.ceil(box[3] + pad - eps));
  return [x0, y0, Math.max(x0, x1), Math.max(y0, y1)];
}

/** The pixel rectangles to black out (with the per-kind safety margin). */
export function redactionRects(regions: Region[], W: number, H: number, pad: Record<string, number> = PAD): BBox[] {
  return regions.map((r) => pix(r.bbox, W, H, pad[r.kind] ?? 0)).filter(([x0, y0, x1, y1]) => x1 > x0 && y1 > y0);
}
