/**
 * GLiNER's input building (gliner 0.2.29, UniEncoderSpanProcessor), verified against 160 recorded cases:
 *   [CLS] + for each label: <<ENT>>, label + <<SEP>> + the chunk's words + [SEP]
 * every word tokenized on its own; words_mask marks the FIRST sub-token of each chunk word with its number
 * (1, 2, ...; a word that yields no tokens is not counted), candidate spans = every start word x widths 0..K-1.
 */
import { tokenize } from "../core/spans.js";

export const ENT_TOKEN = "<<ENT>>";
export const SEP_TOKEN = "<<SEP>>";

/** Tokenizes ONE word with no special tokens (what GLiNER's tokenizer does per pre-split word). */
export interface WordTokenizer {
  encodeWord(word: string): number[];
  clsId: number;
  sepId: number;
}

export interface GlinerInputs {
  words: Array<[string, number, number]>; // GLiNER's word split of the chunk, with character offsets
  inputIds: number[];
  wordsMask: number[];
  textLength: number;
  spanIdx: Array<[number, number]>;
  spanMask: boolean[];
}

export function promptIds(tok: WordTokenizer, labels: string[]): number[] {
  return [...labels.flatMap((l) => [...tok.encodeWord(ENT_TOKEN), ...tok.encodeWord(l)]), ...tok.encodeWord(SEP_TOKEN)];
}

export function buildInputs(chunk: string, tok: WordTokenizer, labels: string[], maxWidth = 12,
  prompt?: number[]): GlinerInputs {
  const words = tokenize(chunk);
  const inputIds = [tok.clsId];
  const wordsMask = [0];
  for (const id of prompt ?? promptIds(tok, labels)) {
    inputIds.push(id);
    wordsMask.push(0);
  }
  let n = 0;
  for (const [w] of words) {
    const ids = tok.encodeWord(w);
    if (!ids.length) continue; // no tokens: GLiNER never sees this word, so it is not counted
    n++;
    ids.forEach((id, i) => {
      inputIds.push(id);
      wordsMask.push(i === 0 ? n : 0);
    });
  }
  inputIds.push(tok.sepId);
  wordsMask.push(0);
  const L = words.length;
  const spanIdx: Array<[number, number]> = [];
  const spanMask: boolean[] = [];
  for (let s = 0; s < L; s++) {
    for (let k = 0; k < maxWidth; k++) {
      spanIdx.push([s, s + k]);
      spanMask.push(s + k <= L - 1);
    }
  }
  return { words, inputIds, wordsMask, textLength: L, spanIdx, spanMask };
}

/** What the exported model expects (parda_onnx.json written by parda/export/onnx_export.py). */
export interface OnnxMeta {
  inputs: string[];
  fixed?: { max_words: number; max_width: number; pad_to: Record<string, number>; fake_text_lengths?: boolean } | null;
}

export interface Feed { type: "int64" | "bool"; data: number[] | boolean[]; dims: number[] }

/** Network feeds, padded the way the export needs: candidate spans padded to max_words x max_width (masked out),
 *  text_lengths = max_words for the network, the REAL word count goes to the patched LSTM (lstm_lengths). */
export function toFeeds(x: GlinerInputs, meta: OnnxMeta): Record<string, Feed> {
  const T = x.inputIds.length;
  const fixed = meta.fixed;
  if (fixed && x.textLength > fixed.max_words) {
    throw new Error(`chunk has ${x.textLength} words; the exported model takes at most ${fixed.max_words}`);
  }
  let spanIdx = x.spanIdx;
  let spanMask = x.spanMask;
  const padTo = fixed?.pad_to ?? {};
  if (padTo.span_idx && spanIdx.length < padTo.span_idx) {
    spanIdx = [...spanIdx, ...Array.from({ length: padTo.span_idx - spanIdx.length }, () => [0, 0] as [number, number])];
  }
  if (padTo.span_mask && spanMask.length < padTo.span_mask) {
    spanMask = [...spanMask, ...Array.from({ length: padTo.span_mask - spanMask.length }, () => false)];
  }
  const all: Record<string, Feed> = {
    input_ids: { type: "int64", data: x.inputIds, dims: [1, T] },
    token_type_ids: { type: "int64", data: new Array(T).fill(0), dims: [1, T] },
    attention_mask: { type: "int64", data: new Array(T).fill(1), dims: [1, T] },
    words_mask: { type: "int64", data: x.wordsMask, dims: [1, T] },
    span_idx: { type: "int64", data: spanIdx.flat(), dims: [1, spanIdx.length, 2] },
    span_mask: { type: "bool", data: spanMask, dims: [1, spanMask.length] },
    text_lengths: { type: "int64", data: [fixed?.fake_text_lengths ? fixed.max_words : x.textLength], dims: [1, 1] },
    lstm_lengths: { type: "int64", data: [x.textLength], dims: [1] },
  };
  const feeds: Record<string, Feed> = {};
  for (const name of meta.inputs) {
    if (!(name in all)) throw new Error(`the model wants an input this app does not know: ${name}`);
    feeds[name] = all[name];
  }
  return feeds;
}
