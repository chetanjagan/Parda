// The real mDeBERTa tokenizer (transformers.js + public/tokenizer/tokenizer.json) must give exactly the token ids GLiNER
// gave on Kaggle, for every word of the 160 recorded chunks and for the label prompt. Needs `npm install` (runs in CI).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { buildInputs, loadTokenizer, promptIds } from "../build/gliner/index.js";

const G = JSON.parse(readFileSync(new URL("./goldens/gliner.json", import.meta.url), "utf8"));
const read = (f) => JSON.parse(readFileSync(new URL(`../public/tokenizer/${f}`, import.meta.url), "utf8"));

let tok = null;
let why = "";
try {
  tok = await loadTokenizer(read("tokenizer.json"), read("tokenizer_config.json"));
} catch (e) {
  why = `transformers.js not available (${e.message.split("\n")[0]}); run npm install`;
  if (process.env.REQUIRE_TOKENIZER) throw new Error(`CI requires the real tokenizer: ${why}`); // never a silent skip
}

test("special tokens", { skip: !tok && why }, () => {
  assert.equal(tok.clsId, 1);
  assert.equal(tok.sepId, 2);
  assert.deepEqual(tok.encodeWord("<<ENT>>"), [250103]);
  assert.deepEqual(tok.encodeWord("<<SEP>>"), [250104]);
});

test("the label prompt is tokenized exactly like GLiNER did", { skip: !tok && why }, () => {
  const ids = G.cases[0].input_ids;
  assert.deepEqual(promptIds(tok, G.meta.labels), ids.slice(1, ids.indexOf(250104) + 1));
});

test("every recorded word gets exactly the recorded token ids", { skip: !tok && why }, () => {
  const seen = new Map();
  for (const c of G.cases) c.words.forEach((w, i) => seen.set(w, c.word_token_ids[i]));
  const wrong = [];
  for (const [w, ids] of seen) {
    const got = tok.encodeWord(w);
    if (JSON.stringify(got) !== JSON.stringify(ids)) wrong.push(`${JSON.stringify(w)}: ${JSON.stringify(got)} vs ${JSON.stringify(ids)}`);
  }
  assert.equal(wrong.length, 0, `${wrong.length}/${seen.size} words differ, e.g.\n${wrong.slice(0, 15).join("\n")}`);
});

test("full network inputs from the real tokenizer match all 160 chunks", { skip: !tok && why }, () => {
  for (const [i, c] of G.cases.entries()) assert.deepEqual(buildInputs(c.chunk, tok, G.meta.labels).inputIds, c.input_ids, `chunk ${i}`);
});
