/** EasyOCR's readtext in the browser: CRAFT detector + one recognizer, both as ONNX (injected runners). */
import type { Segment } from "../core/spans.js";
import { detBoxes, detectorInput, filterMinSize, type FreeBox, groupTextBox, type HBox } from "./detect.js";
import { type Gray, rgbToGray } from "./imgops.js";
import { alignCollate, type Crop, cropFree, cropHorizontal, decodeCtc } from "./recognize.js";

export type TensorRunner = (input: { data: Float32Array; dims: number[] }) => Promise<{ data: Float32Array | number[]; dims: number[] }>;

export interface ReaderMeta { character: string; lang_char: string }
export interface TextResult { box: number[][]; text: string; conf: number }

export class EasyOcrReader {
  private readonly chars: string[];
  private readonly ignore: Set<number>;

  constructor(private readonly detect: TensorRunner, private readonly recognize: TensorRunner, meta: ReaderMeta,
    private readonly contrastThs = 0.1, private readonly adjustContrast = 0.5) {
    this.chars = [...meta.character];
    const keep = new Set([...meta.lang_char]);
    this.ignore = new Set(this.chars.map((ch, i) => (keep.has(ch) ? -1 : i + 1)).filter((i) => i > 0));
  }

  /** The grouped text boxes (detect()). */
  async boxes(pixels: Uint8Array | Uint8ClampedArray, w: number, h: number, channels = 4) {
    const inp = detectorInput(pixels, w, h, channels);
    const out = await this.detect({ data: inp.tensor, dims: inp.dims });
    const [, h2, w2] = out.dims;
    return filterMinSize(groupTextBox(detBoxes(out.data, h2, w2, inp.ratio)));
  }

  private async read(crop: Crop): Promise<{ text: string; conf: number }> {
    const run = async (adjust: number) => {
      const out = await this.recognize({ data: alignCollate(crop.img, crop.maxWidth, adjust), dims: [1, 1, 64, crop.maxWidth] });
      const [, T, C] = out.dims;
      return decodeCtc(out.data, T, C, this.chars, this.ignore);
    };
    const r1 = await run(0);
    if (r1.conf >= this.contrastThs) return r1;
    const r2 = await run(this.adjustContrast); // low confidence: retry with boosted contrast, keep the better one
    return r1.conf > r2.conf ? r1 : r2;
  }

  /** readtext(image, detail=1, paragraph=False). grey: the gray page (default: computed from the colour pixels). */
  async readtext(pixels: Uint8Array | Uint8ClampedArray, w: number, h: number, channels = 4, grey?: Gray,
    onLine?: (done: number, total: number) => void): Promise<TextResult[]> {
    const g = grey ?? rgbToGray(pixels, w, h, channels);
    const { horizontal, free } = await this.boxes(pixels, w, h, channels);
    return this.recognizeBoxes(g, horizontal, free, onLine);
  }

  /** recognize(): each line box, then each tilted box, cropped and read on its own. */
  async recognizeBoxes(g: Gray, horizontal: HBox[], free: FreeBox[], onLine?: (done: number, total: number) => void): Promise<TextResult[]> {
    const out: TextResult[] = [];
    const total = horizontal.length + free.length;
    let done = 0;
    for (const b of horizontal) {
      const c = cropHorizontal(g, b);
      if (c) out.push({ box: c.box, ...(await this.read(c)) });
      onLine?.(++done, total);
    }
    for (const b of free) {
      const c = cropFree(g, b);
      if (c) out.push({ box: c.box, ...(await this.read(c)) });
      onLine?.(++done, total);
    }
    return out;
  }
}

/** Results -> OCR segments, like parda's EasyOCREngine (non-empty text, box = corner min/max, score = confidence). */
export function toSegments(results: TextResult[]): Segment[] {
  return results.filter((r) => r.text.trim()).map((r) => {
    const xs = r.box.map((p) => p[0]), ys = r.box.map((p) => p[1]);
    const t = Math.trunc; // parda's _xyxy: int() of the corner min / max
    return { text: r.text, bbox: [t(Math.min(...xs)), t(Math.min(...ys)), t(Math.max(...xs)), t(Math.max(...ys))], score: r.conf };
  });
}
