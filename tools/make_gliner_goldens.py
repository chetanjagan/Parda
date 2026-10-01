"""Golden cases for the browser port of GLiNER (web app milestone M2). Runs the SHIPPED model (the ONNX
int8_embed export, via the normal GLiNER object) on real OCR text chunks and records, per chunk:

  words          GLiNER's word split of the chunk, with character offsets
  word_token_ids the tokenizer's ids for every word on its own (no special tokens)
  inputs         exactly what the network receives: input_ids, attention_mask, words_mask, text_lengths,
                 span_idx / span_mask (summarised; full for the first chunks), token_type_ids, + constant inputs
  logits         the network output (float32, base64) for the first --n_logits chunks
  entities       what predict_entities returns (start, end, label, score) at the pipeline's threshold

plus the tokenizer files, gliner_config.json and GLiNER's own source code (so the port follows the real code).

  python tools/make_gliner_goldens.py --model outputs/onnx/gliner --ocr cache/ocr_tesseract.jsonl cache/ocr_easyocr.jsonl \
         --n 160 --out outputs/gliner_goldens
"""
import argparse
import base64
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def b64_float32(arr):
    import numpy as np
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def chunks_from_ocr(files, n, window=200, stride=150):
    """The exact text chunks the pipeline sends to GLiNER (GLiNERPredictor windows over each page's OCR text)."""
    from parda.pii.spans import build_ocr_text, tokenize, windows
    out = []
    per_file = max(1, n // max(1, len(files)))
    for f in files:
        k = 0
        for ln in open(f, encoding="utf-8"):
            text = build_ocr_text(json.loads(ln).get("segments") or [])[0]
            toks = tokenize(text)
            for s, e in windows(len(toks), window, stride):
                if s < e:
                    src = os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f))  # e.g. cache_ood/ocr_easyocr.jsonl
                    out.append({"source": src, "chunk": text[toks[s][1]:toks[e - 1][2]]})
                    k += 1
            if k >= per_file:
                break
    return out[:n]


def find_tokenizer(model):
    for path in ("data_processor.transformer_tokenizer", "data_processor.tokenizer", "tokenizer"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, path
        except AttributeError:
            continue
    return None, None


def record(model, chunks, labels, threshold, n_full=12, n_logits=30):
    import torch
    seen = {}

    def pre(mod, args, kwargs):
        kw = dict(kwargs)
        if args and isinstance(args[0], dict):
            kw.update(args[0])
        seen["kw"] = kw

    def post(mod, args, kwargs, out):
        from parda.export.onnx_runtime import logits_of
        seen["logits"] = logits_of(out)

    h1 = model.model.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = model.model.register_forward_hook(post, with_kwargs=True)
    tok, tok_path = find_tokenizer(model)
    cases = []
    try:
        for i, c in enumerate(chunks):
            seen.clear()
            with torch.no_grad():
                ents = model.predict_entities(c["chunk"], labels, threshold=threshold)
            kw = seen.get("kw", {})
            t = {k: v for k, v in kw.items() if torch.is_tensor(v)}
            const = {k: v for k, v in kw.items() if not torch.is_tensor(v)}
            words = const.get("tokens", [[]])[0] if isinstance(const.get("tokens"), list) else None
            case = {"source": c["source"], "chunk": c["chunk"], "words": words,
                    "entities": [{"start": int(e["start"]), "end": int(e["end"]), "label": e["label"],
                                  "score": float(e["score"]), "text": e.get("text")} for e in ents],
                    "inputs": {}, "constant_inputs": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                                          else repr(v)[:300]) for k, v in const.items() if k != "tokens"}}
            for k, v in t.items():
                v = v.detach().cpu()
                if k in ("span_idx", "span_mask") and i >= n_full:
                    case["inputs"][k] = {"shape": list(v.shape), "sum": float(v.float().sum())}
                else:
                    case["inputs"][k] = {"shape": list(v.shape), "values": v.tolist()}
            if tok is not None and words:
                case["word_token_ids"] = [tok(w, add_special_tokens=False)["input_ids"] for w in words]
            if i < n_logits and "logits" in seen:
                case["logits"] = b64_float32(seen["logits"].detach().cpu().numpy())
            cases.append(case)
    finally:
        h1.remove()
        h2.remove()
    return cases, tok, tok_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="exported ONNX folder (uses its chosen variant) or a GLiNER path")
    ap.add_argument("--ocr", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import gliner
    from parda.pii.labels import LABELS
    from parda.pipeline.models import gliner as load_model
    os.makedirs(a.out, exist_ok=True)
    model = load_model(a.model)
    labels = list(LABELS.values())
    chunks = chunks_from_ocr(a.ocr, a.n)
    cases, tok, tok_path = record(model, chunks, labels, a.threshold)
    meta = {"gliner_version": gliner.__version__, "labels": labels, "threshold": a.threshold,
            "model": a.model, "onnx_file": getattr(model, "onnx_file", None), "tokenizer_attr": tok_path,
            "config": {k: getattr(model.config, k, None) for k in ("max_len", "max_width", "span_mode", "model_name",
                                                                   "max_types", "ent_token", "sep_token", "words_splitter_type")},
            "n_cases": len(cases), "n_entities": sum(len(c["entities"]) for c in cases)}
    json.dump({"meta": meta, "cases": cases}, open(os.path.join(a.out, "gliner_goldens.json"), "w"), ensure_ascii=False)
    if tok is not None:  # the tokenizer files the browser needs (tokenizer.json etc.)
        tok.save_pretrained(os.path.join(a.out, "tokenizer"))
    try:
        json.dump(model.config.to_dict(), open(os.path.join(a.out, "gliner_config.json"), "w"), indent=2, default=str)
    except Exception as e:
        meta["config_error"] = repr(e)
    src = os.path.dirname(gliner.__file__)  # GLiNER's own code: the port follows it, not guesses
    with zipfile.ZipFile(os.path.join(a.out, "gliner_src.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, fs in os.walk(src):
            for f in fs:
                if f.endswith(".py"):
                    p = os.path.join(d, f)
                    z.write(p, os.path.relpath(p, os.path.dirname(src)))
    print(json.dumps(meta, indent=2), flush=True)
    sizes = {f: f"{os.path.getsize(os.path.join(a.out, f)) / 2 ** 20:.1f} MB" for f in os.listdir(a.out)
             if os.path.isfile(os.path.join(a.out, f))}
    print("files:", sizes, "| tokenizer folder:", sorted(os.listdir(os.path.join(a.out, "tokenizer"))) if tok else None)
    return meta


if __name__ == "__main__":
    main()
