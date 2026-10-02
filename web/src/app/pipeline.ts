/** One page through Parda, as parda/pipeline/redactor.Redactor.analyse does it: for each OCR engine, its text ->
 *  PII spans (GLiNER + ID rules) -> pixel boxes (+ gap filling); then YOLO's visual regions. DOM-free and injected. */
import { type Region, redactionRects } from "../core/boxes.js";
import type { BBox } from "../core/spans.js";
import { type ChunkModel, unionSpans, windowedPredict } from "../core/predict.js";
import { textRegions } from "../core/regions.js";
import { findRules, type Span } from "../core/rules.js";
import type { Segment } from "../core/spans.js";
import { visualRegions } from "../vision/detector.js";
import type { Detection } from "../vision/yolo.js";

export interface OcrEngine { name: string; run(lang: string): Promise<Segment[]> }
export interface PageEngines {
  ocr: OcrEngine[];
  gliner?: ChunkModel;                   // GLiNER on one chunk (natural-language labels); omitted = rules only
  yolo?: () => Promise<Detection[]>;      // faces, signatures, QR codes, stamps
}
export interface PageResult { regions: Region[]; segments: Record<string, Segment[]>; timings: Record<string, number> }

/** UnionPredictor(GLiNER, rules): anything either flags. */
export function textPredictor(gliner?: ChunkModel): (text: string) => Promise<Span[]> {
  return async (text) => unionSpans(gliner ? await windowedPredict(text, gliner) : [], findRules(text));
}

export async function analysePage(engines: PageEngines, lang: string, onStep?: (step: string) => void,
  gapFill = true, visualConf = 0.25): Promise<PageResult> {
  const regions: Region[] = [];
  const segments: Record<string, Segment[]> = {};
  const timings: Record<string, number> = {};
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const predict = textPredictor(engines.gliner);
  for (const eng of engines.ocr) {
    onStep?.(`reading text (${eng.name})`);
    let t = now();
    const segs = await eng.run(lang);
    timings[`ocr_${eng.name}`] = (now() - t) / 1000;
    segments[eng.name] = segs;
    onStep?.(`finding personal data (${eng.name})`);
    t = now();
    regions.push(...(await textRegions(segs, predict, eng.name, gapFill)).regions);
    timings[`text_pii_${eng.name}`] = (now() - t) / 1000;
  }
  if (engines.yolo) {
    onStep?.("finding faces, signatures, QR codes, stamps");
    const t = now();
    regions.push(...visualRegions(await engines.yolo(), visualConf));
    timings.visual = (now() - t) / 1000;
  }
  return { regions, segments, timings };
}

/** The rectangles to black out on a page (padding per kind, as in Redactor.redact). */
export function blackouts(regions: Region[], width: number, height: number): BBox[] {
  return redactionRects(regions, width, height);
}
