/** The app's background worker: GLiNER (+ tokenizer), YOLO and EasyOCR run here, so the page never freezes.
 *  In:  { type: "analyse", id, rgba, width, height, lang, easyocr, tesseract: Segment[] }
 *  Out: { type: "progress" | "step" | "result" | "error", id, ... }   result: { streams, dets, timings } */
import { buildOcrText, type Segment } from "../core/spans.js";
import { EasyOcrReader, toSegments } from "../easyocr/reader.js";
import { GlinerModel } from "../gliner/model.js";
import type { OnnxMeta } from "../gliner/processor.js";
import { loadTokenizer } from "../gliner/tokenizer.js";
import { YoloDetector } from "../vision/detector.js";
import { LIBS } from "./libs.js";
import { fetchCached, fetchJson, floatRunner, MODEL_REPO, type Progress, type Runner, session, toTensors } from "./models.js";
import { textPredictor } from "./pipeline.js";
import type { Stream } from "./review.js";

interface EasyMeta { detector: { file: string }; readers: Record<string, { file: string; character: string; lang_char: string }> }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const post = (m: any, transfer: Transferable[] = []) => (self as any).postMessage(m, transfer);

let gliner: GlinerModel | undefined, yolo: YoloDetector | undefined;
let easyMeta: EasyMeta | undefined, detector: Runner | undefined;
const readers = new Map<string, EasyOcrReader>();

async function loadCore(progress: Progress, base: string) {
  if (gliner && yolo) return;
  const tok = await loadTokenizer(await fetchJson(`${base}public/tokenizer/tokenizer.json`, "Tokenizer", progress),
    await fetchJson(`${base}public/tokenizer/tokenizer_config.json`, "Tokenizer", progress), LIBS.transformers);
  const meta = await fetchJson<OnnxMeta>(`${MODEL_REPO}/gliner/parda_onnx.json`, "Text model settings", progress);
  const gl = await session(await fetchCached(`${MODEL_REPO}/gliner/model_int8_embed.onnx`, "Text model (GLiNER)", progress));
  const wanted = new Set<string>(gl.inputNames); // the export dropped inputs mDeBERTa ignores (token_type_ids)
  gliner = new GlinerModel(async (feeds) => {
    const r = await gl.run(await toTensors(Object.fromEntries(Object.entries(feeds).filter(([k]) => wanted.has(k)))));
    return { data: r.logits.data as Float32Array, dims: r.logits.dims as number[] };
  }, tok, meta);
  yolo = new YoloDetector(await floatRunner(await session(await fetchCached(`${MODEL_REPO}/yolo/parda-yolo.onnx`,
    "Face and signature model (YOLO)", progress))));
}

async function loadEasy(lang: string, progress: Progress): Promise<EasyOcrReader> {
  const have = readers.get(lang);
  if (have) return have;
  easyMeta ??= await fetchJson<EasyMeta>(`${MODEL_REPO}/easyocr/easyocr_onnx.json`, "Photo reader settings", progress);
  detector ??= await floatRunner(await session(await fetchCached(`${MODEL_REPO}/easyocr/${easyMeta.detector.file}`,
    "Photo reader: text finder", progress)));
  const r = easyMeta.readers[lang] ?? easyMeta.readers.en;
  const rec = await floatRunner(await session(await fetchCached(`${MODEL_REPO}/easyocr/${r.file}`, `Photo reader: ${lang}`, progress)));
  const reader = new EasyOcrReader(detector, rec, r);
  readers.set(lang, reader);
  return reader;
}

async function stream(source: string, segs: Segment[]): Promise<Stream> {
  const { text, ordered, offs } = buildOcrText(segs);
  const spans = text ? await textPredictor((chunk) => gliner!.predictChunk(chunk))(text) : [];
  return { source, text, ordered, offs, spans };
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  if (m.type !== "analyse") return;
  const progress: Progress = (label, done, total) => post({ type: "progress", id: m.id, label, done, total });
  const step = (text: string) => post({ type: "step", id: m.id, text });
  const timings: Record<string, number> = {};
  const timed = async <T>(k: string, f: () => Promise<T>) => { const t = performance.now(); const r = await f(); timings[k] = (performance.now() - t) / 1000; return r; };
  try {
    step("Getting the models ready");
    await loadCore(progress, m.base);
    const rgba = new Uint8ClampedArray(m.rgba);
    step("Finding personal data in the text");
    const streams: Stream[] = [await timed("text_pii_tesseract", () => stream("tesseract", m.tesseract))];
    if (m.easyocr) {
      const reader = await loadEasy(m.lang, progress);
      step("Reading the page again with the photo reader");
      const segs = await timed("ocr_easyocr", async () => toSegments(await reader.readtext(rgba, m.width, m.height, 4)));
      step("Finding personal data in what the photo reader saw");
      streams.push(await timed("text_pii_easyocr", () => stream("easyocr", segs)));
    }
    step("Looking for faces, signatures, QR codes and stamps");
    const dets = await timed("visual", () => yolo!.detect(rgba, m.width, m.height, 4));
    post({ type: "result", id: m.id, streams, dets, timings });
  } catch (err) {
    post({ type: "error", id: m.id, message: String((err as Error)?.message ?? err) });
  }
};
