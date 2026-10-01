// GLiNER in the browser = GLiNER in Python, on 160 real OCR chunks recorded from the shipped model
// (tools/make_gliner_goldens.py on Kaggle -> tools/pack_gliner_goldens.py -> test/goldens/gliner.json).
// The tokenizer here is replaced by the recorded token ids; test/gliner_tokenizer.test.mjs checks the real one.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import * as core from "../build/core/index.js";
import { buildInputs, decodeSpans, GlinerModel, toCharSpans, toFeeds } from "../build/gliner/index.js";

const G = JSON.parse(readFileSync(new URL("./goldens/gliner.json", import.meta.url), "utf8"));
const LABELS = G.meta.labels;
const ENT = 250103, SEP = 250104;

function recordedTokenizer() {
  const words = new Map();
  for (const c of G.cases) c.words.forEach((w, i) => {
    const ids = c.word_token_ids[i];
    if (words.has(w)) assert.deepEqual(words.get(w), ids, `word ${w} tokenized two ways`);
    words.set(w, ids);
  });
  const ids = G.cases[0].input_ids;
  const prompt = ids.slice(1, ids.indexOf(SEP) + 1); // <<ENT>> label ... <<SEP>>
  let li = -1;
  const labelIds = LABELS.map(() => []);
  for (const t of prompt) {
    if (t === ENT) li++;
    else if (t !== SEP) labelIds[li].push(t);
  }
  LABELS.forEach((l, i) => words.set(l, labelIds[i]));
  words.set("<<ENT>>", [ENT]);
  words.set("<<SEP>>", [SEP]);
  return { encodeWord: (w) => { if (!words.has(w)) throw new Error(`no recording for ${w}`); return words.get(w); }, clsId: 1, sepId: 2 };
}

function logitsOf(c) {
  const b = Buffer.from(c.logits.b64, "base64");
  return { data: new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4), dims: c.logits.shape };
}

function sameEntities(got, exp, msg) {
  assert.deepEqual(got.map((e) => [e.start, e.end, e.label]), exp.map((e) => [e.start, e.end, e.label]), msg);
  got.forEach((e, i) => assert.ok(Math.abs(e.score - exp[i].score) < 1e-6, `${msg}: score ${e.score} vs ${exp[i].score}`));
}

test("GLiNER's word split = the core tokenizer, on every chunk", () => {
  for (const c of G.cases) assert.deepEqual(core.tokenize(c.chunk).map((t) => t[0]), c.words);
});

test("network inputs (token ids, word mask, word count) match all 160 chunks", () => {
  const tok = recordedTokenizer();
  for (const [i, c] of G.cases.entries()) {
    const x = buildInputs(c.chunk, tok, LABELS);
    assert.deepEqual(x.inputIds, c.input_ids, `chunk ${i} input_ids`);
    assert.deepEqual(x.wordsMask, c.words_mask, `chunk ${i} words_mask`);
    assert.equal(x.textLength, c.words.length);
    assert.equal(x.spanIdx.length, c.words.length * 12);
    assert.equal(x.spanMask.filter(Boolean).length, x.spanIdx.filter(([s, e]) => e <= c.words.length - 1).length);
  }
});

test("decoding the recorded network scores gives the recorded spans", () => {
  const withLogits = G.cases.filter((c) => c.logits);
  assert.ok(withLogits.length >= 10);
  for (const c of withLogits) {
    const { data, dims } = logitsOf(c);
    const spans = toCharSpans(decodeSpans(data, dims, c.words.length, LABELS, G.meta.threshold), core.tokenize(c.chunk));
    sameEntities(spans, c.entities, c.chunk.slice(0, 40));
  }
});

test("feeds follow the export: spans padded to 256 x 12, network told 256 words, LSTM told the real count", () => {
  const c = G.cases[0];
  const x = buildInputs(c.chunk, recordedTokenizer(), LABELS);
  const f = toFeeds(x, G.meta.onnx);
  assert.deepEqual(Object.keys(f), G.meta.onnx.inputs);
  assert.deepEqual(f.span_idx.dims, [1, 3072, 2]);
  assert.deepEqual(f.span_mask.dims, [1, 3072]);
  assert.equal(f.span_mask.data.filter(Boolean).length, x.spanMask.filter(Boolean).length); // padding is masked out
  assert.deepEqual(f.text_lengths.data, [256]);
  assert.deepEqual(f.lstm_lengths.data, [c.words.length]);
  assert.deepEqual(f.input_ids.data, c.input_ids);
  assert.ok(f.token_type_ids.data.every((v) => v === 0) && f.attention_mask.data.every((v) => v === 1));
  const tooLong = { ...x, textLength: 257 };
  assert.throws(() => toFeeds(tooLong, G.meta.onnx), /at most 256/);
});

test("GlinerModel end to end (recorded network) reproduces predict_entities", async () => {
  for (const c of G.cases.filter((c) => c.logits)) {
    const { data, dims } = logitsOf(c);
    const [, L, K, C] = dims;
    const padded = new Float32Array(256 * K * C); // the export returns 256 word rows; rows past the text are ignored
    padded.set(data);
    const run = async (feeds) => {
      assert.deepEqual(feeds.input_ids.data, c.input_ids);
      assert.equal(feeds.lstm_lengths.data[0], L);
      return { data: padded, dims: [1, 256, K, C] };
    };
    const model = new GlinerModel(run, recordedTokenizer(), G.meta.onnx, LABELS, G.meta.threshold);
    sameEntities(await model.predictChunk(c.chunk), c.entities, c.chunk.slice(0, 40));
  }
});

test("labels and windows: the model plugs into the core's windowed predictor", async () => {
  const c = G.cases.find((c) => c.logits && c.entities.length);
  const { data, dims } = logitsOf(c);
  const model = new GlinerModel(async () => ({ data, dims }), recordedTokenizer(), G.meta.onnx, LABELS, G.meta.threshold);
  const spans = await core.windowedPredict(c.chunk, (t) => model.predictChunk(t));
  assert.deepEqual(spans, core.mergeSpans(c.entities.map((e) => ({ label: core.toCode(e.label), start: e.start, end: e.end,
    score: spans.find((s) => s.start === e.start)?.score ?? e.score }))));
});

test("edge cases the recordings never hit: threshold boundary, spans past the end, words with no tokens", () => {
  // 3 words, 2 widths, 1 class. sigmoid(x) = 0.3 at x = ln(0.3/0.7) = -0.8473
  const lab = ["person name"];
  const z = -0.8472978603872037;
  const logits = new Float32Array([
    z + 1e-3, -9,   // word 0: width 0 just ABOVE 0.3 -> kept;  width 1 low
    z - 1e-3, -9,   // word 1: width 0 just BELOW 0.3 -> dropped
    -9, 5,          // word 2: width 1 would end past the last word -> dropped even with a high score
  ]);
  const spans = decodeSpans(logits, [1, 3, 2, 1], 3, lab, 0.3);
  assert.deepEqual(spans.map((s) => [s.start, s.end]), [[0, 0]]);
  // exactly at the threshold is not "> threshold"
  const exact = new Float32Array([Math.log(0.3 / 0.7), -9]);
  assert.equal(decodeSpans(exact, [1, 1, 2, 1], 1, lab, 0.3).length, Math.fround(1 / (1 + Math.exp(-exact[0]))) > Math.fround(0.3) ? 1 : 0);
  // a word that the tokenizer turns into no tokens is skipped and NOT counted in the word mask (as GLiNER does)
  const tok = { encodeWord: (w) => (w === "\u200c" ? [] : w.startsWith("<<") ? [9] : [w.length + 10, 99].slice(0, w.length > 2 ? 2 : 1)), clsId: 1, sepId: 2 };
  const x = buildInputs("ab \u200c cde f", tok, ["x"]);
  assert.deepEqual(x.words.map((w) => w[0]), ["ab", "\u200c", "cde", "f"]);
  const textPart = x.wordsMask.slice(1 + 3); // after [CLS] + prompt (<<ENT>>, x, <<SEP>>)
  assert.deepEqual(textPart, [1, 2, 0, 3, 0]); // ab -> 1, (no tokens), cde -> 2 + continuation, f -> 3, [SEP]
  assert.equal(x.textLength, 4); // the word count still includes it (text_lengths = number of words)
});
