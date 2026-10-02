/** The real models, loaded on demand: GLiNER (+ tokenizer), YOLO, Tesseract.js, and optionally EasyOCR. */
import { EasyOcrReader, toSegments } from "../easyocr/reader.js";
import { GlinerModel } from "../gliner/model.js";
import type { OnnxMeta, WordTokenizer } from "../gliner/processor.js";
import { loadTokenizer } from "../gliner/tokenizer.js";
import { TesseractOcr } from "../ocr/tesseract.js";
import { YoloDetector } from "../vision/detector.js";
import { fetchCached, fetchJson, floatRunner, MODEL_REPO, type Progress, type Runner, session, toTensors } from "./models.js";
import type { PageImage } from "./pages.js";
import type { PageEngines } from "./pipeline.js";

interface EasyMeta { detector: { file: string }; readers: Record<string, { file: string; character: string; lang_char: string }> }

/** Download sizes, for the UI. */
export const SIZES_MB = { core: 555 + 10 + 16, easyBase: 79, easy: { "en": 14, "hi-en": 205, "kn-en": 15 } as Record<string, number> };

const yieldToUi = () => new Promise((r) => setTimeout(r, 0));

export class Models {
  private gliner?: GlinerModel;
  private yolo?: YoloDetector;
  private tess?: TesseractOcr;
  private easyMeta?: EasyMeta;
  private detector?: Runner;
  private readers = new Map<string, EasyOcrReader>();

  constructor(private readonly progress: Progress) {}

  get coreReady(): boolean { return !!(this.gliner && this.yolo && this.tess); }

  async loadCore(): Promise<void> {
    if (this.coreReady) return;
    const local = (f: string) => new URL(`public/tokenizer/${f}`, document.baseURI).href;
    const tok: WordTokenizer = await loadTokenizer(
      await fetchJson(local("tokenizer.json"), "tokenizer", this.progress),
      await fetchJson(local("tokenizer_config.json"), "tokenizer", this.progress));
    const meta = await fetchJson<OnnxMeta>(`${MODEL_REPO}/gliner/parda_onnx.json`, "GLiNER settings", this.progress);
    const gl = await session(await fetchCached(`${MODEL_REPO}/gliner/model_int8_embed.onnx`, "GLiNER (personal data)", this.progress));
    this.gliner = new GlinerModel(async (feeds) => {
      const r = await gl.run(await toTensors(feeds));
      return { data: r.logits.data as Float32Array, dims: r.logits.dims as number[] };
    }, tok, meta);
    const yolo = await session(await fetchCached(`${MODEL_REPO}/yolo/parda-yolo.onnx`, "YOLO (faces, signatures)", this.progress));
    this.yolo = new YoloDetector(await floatRunner(yolo));
    const name = "tesseract.js";
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const t: any = await import(/* @vite-ignore */ name);
    const createWorker = t.createWorker ?? t.default?.createWorker;
    this.tess = new TesseractOcr((langs: string) => createWorker(langs, 1));
  }

  async loadEasyOcr(lang: string): Promise<EasyOcrReader> {
    const have = this.readers.get(lang);
    if (have) return have;
    this.easyMeta ??= await fetchJson<EasyMeta>(`${MODEL_REPO}/easyocr/easyocr_onnx.json`, "EasyOCR settings", this.progress);
    this.detector ??= await floatRunner(await session(await fetchCached(`${MODEL_REPO}/easyocr/${this.easyMeta.detector.file}`,
      "EasyOCR text finder", this.progress)));
    const r = this.easyMeta.readers[lang] ?? this.easyMeta.readers.en;
    const rec = await floatRunner(await session(await fetchCached(`${MODEL_REPO}/easyocr/${r.file}`,
      `EasyOCR reader (${lang})`, this.progress)));
    const reader = new EasyOcrReader(this.detector, rec, r);
    this.readers.set(lang, reader);
    return reader;
  }

  /** The engines for one page. */
  async engines(page: PageImage, lang: string, useEasyOcr: boolean): Promise<PageEngines> {
    await this.loadCore();
    const ocr: PageEngines["ocr"] = [{ name: "tesseract", run: async (l) => { await yieldToUi(); return this.tess!.run(page.canvas, l); } }];
    if (useEasyOcr) {
      const reader = await this.loadEasyOcr(lang);
      ocr.push({ name: "easyocr", run: async () => { await yieldToUi(); return toSegments(await reader.readtext(page.rgba, page.width, page.height, 4)); } });
    }
    return {
      ocr,
      gliner: async (chunk) => { await yieldToUi(); return this.gliner!.predictChunk(chunk); },
      yolo: async () => { await yieldToUi(); return this.yolo!.detect(page.rgba, page.width, page.height, 4); },
    };
  }
}
