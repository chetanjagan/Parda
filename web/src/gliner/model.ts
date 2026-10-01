/** GLiNER in the browser: words -> network inputs -> ONNX -> spans. The ONNX runner is injected (onnxruntime-web in
 *  the app, a recorded fake in tests), so this file has no dependencies. */
import { LABELS } from "../core/labels.js";
import type { RawEntity } from "../core/predict.js";
import { decodeSpans, toCharSpans } from "./decoder.js";
import { buildInputs, type Feed, type OnnxMeta, promptIds, toFeeds, type WordTokenizer } from "./processor.js";

export type Runner = (feeds: Record<string, Feed>) => Promise<{ data: Float32Array | number[]; dims: number[] }>;

export class GlinerModel {
  readonly labels: string[];
  private readonly prompt: number[];

  constructor(private readonly run: Runner, private readonly tok: WordTokenizer, private readonly meta: OnnxMeta,
    labels: string[] = Object.values(LABELS), private readonly threshold = 0.3, private readonly maxWidth = 12) {
    this.labels = labels;
    this.prompt = promptIds(tok, labels);
  }

  /** predict_entities for one chunk: natural-language labels and chunk character offsets. */
  async predictChunk(chunk: string): Promise<RawEntity[]> {
    if (!chunk.trim()) return [];
    const x = buildInputs(chunk, this.tok, this.labels, this.maxWidth, this.prompt);
    if (!x.textLength) return [];
    const out = await this.run(toFeeds(x, this.meta));
    return toCharSpans(decodeSpans(out.data, out.dims, x.textLength, this.labels, this.threshold), x.words);
  }
}
