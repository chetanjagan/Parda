/** The mDeBERTa tokenizer (tokenizer.json saved with GLiNER) via transformers.js. Loaded lazily, so the core and the
 *  tests that do not need it never import the library. */
import type { WordTokenizer } from "./processor.js";

export async function loadTokenizer(tokenizerJson: unknown, tokenizerConfig: unknown,
  name = "@huggingface/transformers"): Promise<WordTokenizer> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mod: any = await import(/* @vite-ignore */ name);
  const Cls = mod.DebertaV2Tokenizer ?? mod.PreTrainedTokenizer;
  const tok = new Cls(tokenizerJson, tokenizerConfig);
  const id = (t: string): number => {
    const ids: number[] = tok.encode(t, { add_special_tokens: false });
    if (ids.length !== 1) throw new Error(`special token ${t} is not a single token`);
    return ids[0];
  };
  return {
    encodeWord: (w: string) => Array.from(tok.encode(w, { add_special_tokens: false }) as number[]),
    clsId: id("[CLS]"),
    sepId: id("[SEP]"),
  };
}
