/**
 * Tesseract in the browser (Tesseract.js), set up like the Python pipeline's TesseractEngine: page segmentation mode 3,
 * the same language mapping, every word with non-empty text, score = confidence / 100, box = the word's pixel box.
 * Tesseract.js is injected (createWorker), so this file has no dependencies and the conversion is testable.
 */
import type { Segment } from "../core/spans.js";

/** Same mapping as parda/ocr/engines.py TesseractEngine.LANG. */
export const TESS_LANGS: Record<string, string> = { "en": "eng", "hi-en": "hin+eng", "kn-en": "kan+eng", "auto": "eng+hin+kan" };

export function tesseractLangs(lang: string): string {
  return TESS_LANGS[lang] ?? "eng";
}

interface TjsWord { text: string; confidence: number; bbox: { x0: number; y0: number; x1: number; y1: number } }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TjsData = any;

/** All words of a Tesseract.js result (v5+ nests them in blocks -> paragraphs -> lines; v4 had data.words). */
export function wordsOf(data: TjsData): TjsWord[] {
  if (Array.isArray(data?.words) && data.words.length) return data.words;
  const out: TjsWord[] = [];
  for (const b of data?.blocks ?? []) {
    for (const p of b.paragraphs ?? []) for (const l of p.lines ?? []) for (const w of l.words ?? []) out.push(w);
  }
  return out;
}

/** Tesseract.js result -> OCR segments (the format build_ocr_text / span_boxes use). */
export function toSegments(data: TjsData): Segment[] {
  return wordsOf(data)
    .filter((w) => (w.text ?? "").trim())
    .map((w) => ({ text: w.text, bbox: [w.bbox.x0, w.bbox.y0, w.bbox.x1, w.bbox.y1], score: w.confidence / 100 }));
}

export interface TjsWorker {
  reinitialize?(langs: string, oem?: number): Promise<unknown>;
  setParameters(p: Record<string, string>): Promise<unknown>;
  recognize(image: unknown, opts?: unknown, output?: unknown): Promise<{ data: TjsData }>;
  terminate(): Promise<unknown>;
}
export type CreateWorker = (langs: string) => Promise<TjsWorker>;

/** One Tesseract.js worker; switches languages only when the page's language changes. */
export class TesseractOcr {
  readonly name = "tesseract";
  private worker: TjsWorker | null = null;
  private langs = "";

  constructor(private readonly create: CreateWorker, private readonly psm = 3) {}

  private async ready(langs: string): Promise<TjsWorker> {
    if (this.worker && this.langs === langs) return this.worker;
    if (this.worker?.reinitialize) {
      await this.worker.reinitialize(langs);
    } else {
      await this.worker?.terminate();
      this.worker = await this.create(langs);
    }
    this.langs = langs;
    await this.worker!.setParameters({ tessedit_pageseg_mode: String(this.psm) });
    return this.worker!;
  }

  /** image: anything Tesseract.js accepts (canvas, ImageData, Blob, URL, Buffer in Node). */
  async run(image: unknown, lang: string): Promise<Segment[]> {
    const w = await this.ready(tesseractLangs(lang));
    const { data } = await w.recognize(image, {}, { blocks: true });
    return toSegments(data);
  }

  async close(): Promise<void> {
    await this.worker?.terminate();
    this.worker = null;
    this.langs = "";
  }
}
