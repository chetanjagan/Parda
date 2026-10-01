/** OCR segments + a span predictor -> text regions to redact (parda/pipeline/redactor.text_regions). */
import { fillGaps, type Region, spanBoxes } from "./boxes.js";
import type { Span } from "./rules.js";
import { buildOcrText, type Segment } from "./spans.js";

export async function textRegions(segs: Segment[], predict: (text: string) => Promise<Span[]>, source: string,
  gapFill = true): Promise<{ regions: Region[]; spans: Span[]; text: string }> {
  const { text, ordered, offs } = buildOcrText(segs);
  if (!text) return { regions: [], spans: [], text };
  const spans = await predict(text);
  const regions: Region[] = spans.flatMap((sp) => spanBoxes(sp.start, sp.end, ordered, offs).map((bbox) => ({
    kind: "text" as const, label: sp.label, bbox, score: Number(sp.score ?? 1.0), source,
  })));
  return { regions: gapFill ? fillGaps(regions) : regions, spans, text };
}
