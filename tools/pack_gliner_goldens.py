"""Trim the GLiNER recording (tools/make_gliner_goldens.py, run on Kaggle) into the web test goldens.

  python tools/pack_gliner_goldens.py --src gliner_goldens/ [--n_logits 15]

Writes web/test/goldens/gliner.json (inputs + final spans for every chunk, raw scores for n_logits chunks) and copies
the tokenizer files to web/public/tokenizer/ (the app serves them; the tokenizer test reads them).
"""
import argparse
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# What the exported model expects: input order and the fixed word count (parda/export/onnx_export.py, MAX_WORDS=256,
# max_width 12, LSTM lengths as a real input). Same as the shipped model's parda_onnx.json.
ONNX_META = {"inputs": ["input_ids", "token_type_ids", "attention_mask", "words_mask", "span_idx", "span_mask",
                        "text_lengths", "lstm_lengths"],
             "fixed": {"max_words": 256, "max_width": 12, "pad_to": {"span_idx": 3072, "span_mask": 3072},
                       "fake_text_lengths": True}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder unpacked from gliner_goldens.zip")
    ap.add_argument("--n_logits", type=int, default=15)
    a = ap.parse_args(argv)
    g = json.load(open(os.path.join(a.src, "gliner_goldens.json"), encoding="utf-8"))
    cases = []
    with_logits = 0
    for c in g["cases"]:
        case = {"source": c["source"], "chunk": c["chunk"], "words": c["words"], "word_token_ids": c["word_token_ids"],
                "input_ids": c["inputs"]["input_ids"]["values"][0], "words_mask": c["inputs"]["words_mask"]["values"][0],
                "entities": [{k: e[k] for k in ("start", "end", "label", "score")} for e in c["entities"]]}
        if "logits" in c and with_logits < a.n_logits:
            case["logits"] = c["logits"]
            with_logits += 1
        cases.append(case)
    meta = {k: g["meta"][k] for k in ("gliner_version", "labels", "threshold", "config")}
    meta["onnx"] = ONNX_META
    out = os.path.join(ROOT, "web", "test", "goldens", "gliner.json")
    json.dump({"meta": meta, "cases": cases}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    tdir = os.path.join(ROOT, "web", "public", "tokenizer")
    os.makedirs(tdir, exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(os.path.join(a.src, "tokenizer", f), os.path.join(tdir, f))
    print(f"{out}: {len(cases)} chunks, {sum(len(c['entities']) for c in cases)} spans, {with_logits} with raw scores, "
          f"{os.path.getsize(out) / 2 ** 20:.1f} MB | tokenizer -> {tdir}")


if __name__ == "__main__":
    main()
