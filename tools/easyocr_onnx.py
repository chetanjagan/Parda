"""EasyOCR for the browser, step A (web app milestone M4b): export its two networks to ONNX and prove the converted
networks give EXACTLY the same OCR as PyTorch EasyOCR, then record what the browser port must reproduce.

  python tools/easyocr_onnx.py export --out outputs/easyocr_onnx
  python tools/easyocr_onnx.py check  --onnx outputs/easyocr_onnx --pages pages.jsonl --n 10
  python tools/easyocr_onnx.py record --onnx outputs/easyocr_onnx --pages pages.jsonl --n 2 --out outputs/easyocr_goldens

EasyOCR = CRAFT text detector (score maps -> boxes grouped into lines) + one recognizer per language set (CNN + LSTM,
CTC-decoded), set up exactly like the pipeline's EasyOCREngine (one reader per document language, readtext defaults).
The check swaps ONLY the networks inside the real EasyOCR Reader for onnxruntime sessions, so any difference comes
from the conversion. pages.jsonl: {"id", "path", "lang"} per line.
"""
import argparse
import json
import os
import shutil
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

READER_LANGS = {"en": ["en"], "hi-en": ["hi", "en"], "kn-en": ["kn", "en"]}  # = parda.ocr.engines.EasyOCREngine.LANG


def reader_for(lang):
    import easyocr
    return easyocr.Reader(READER_LANGS[lang], gpu=False, verbose=False)


def net(m):
    return getattr(m, "module", m)  # unwrap DataParallel


def first_call(module, fn):
    """Run fn() and return the (args, kwargs) of the first forward call of module."""
    seen = {}

    def hook(mod, args, kwargs):
        seen.setdefault("call", (args, kwargs))

    h = module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        fn()
    finally:
        h.remove()
    return seen["call"]


def sample_image(pages, lang):
    p = next((p for p in pages if p["lang"] == lang), pages[0])
    return p["path"]


# ------------------------------------------------------------------ export
def export(out, pages, opset=17):
    import torch
    os.makedirs(out, exist_ok=True)
    meta = {"readers": {}, "opset": opset}
    for lang in READER_LANGS:
        reader = reader_for(lang)
        img = sample_image(pages, lang)
        det, rec = net(reader.detector), net(reader.recognizer)
        det.eval()
        rec.eval()
        info = {"langs": READER_LANGS[lang], "character": reader.character,
                "lang_char": "".join(reader.lang_char) if isinstance(reader.lang_char, list) else reader.lang_char,
                "model_lang": getattr(reader, "model_lang", None),
                "recognizer_network": type(rec).__name__}
        if lang == "en":  # one detector (CRAFT) for every language
            (x,), _ = first_call(det, lambda: reader.readtext(img))

            class Det(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = det

                def forward(self, x):
                    return self.net(x)[0]

            with torch.no_grad():
                torch.onnx.export(Det().eval(), (x,), os.path.join(out, "craft.onnx"), input_names=["image"],
                                  output_names=["maps"], opset_version=opset, dynamo=False,
                                  dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                                "maps": {0: "batch", 1: "h2", 2: "w2"}})
            meta["detector"] = {"file": "craft.onnx", "input_shape": list(x.shape)}
        (img_t, text), _ = first_call(rec, lambda: reader.readtext(img))
        text_len = int(text.shape[1])

        class Rec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = rec

            def forward(self, image):  # the CTC recognizer ignores its text argument
                return self.net(image, torch.zeros(image.shape[0], text_len, dtype=torch.long))

        f = f"recognizer_{lang}.onnx"
        with torch.no_grad():
            torch.onnx.export(Rec().eval(), (img_t,), os.path.join(out, f), input_names=["image"],
                              output_names=["preds"], opset_version=opset, dynamo=False,
                              dynamic_axes={"image": {0: "batch", 3: "width"}, "preds": {0: "batch", 1: "steps"}})
        info.update(file=f, input_shape=list(img_t.shape), text_len=text_len)
        meta["readers"][lang] = info
        print(f"{lang}: {info['recognizer_network']} -> {f}, input {list(img_t.shape)}, {len(reader.character)} characters",
              flush=True)
    meta["mb"] = {f: round(os.path.getsize(os.path.join(out, f)) / 2 ** 20, 1) for f in os.listdir(out) if f.endswith(".onnx")}
    json.dump(meta, open(os.path.join(out, "easyocr_onnx.json"), "w"), indent=2, ensure_ascii=False)
    return meta


# ------------------------------------------------------------------ swap the networks for onnxruntime
def attach(reader, onnx_dir, lang):
    import onnxruntime as ort
    import torch
    meta = json.load(open(os.path.join(onnx_dir, "easyocr_onnx.json")))
    so = ort.SessionOptions()
    det_s = ort.InferenceSession(os.path.join(onnx_dir, meta["detector"]["file"]), so, providers=["CPUExecutionProvider"])
    rec_s = ort.InferenceSession(os.path.join(onnx_dir, meta["readers"][lang]["file"]), so, providers=["CPUExecutionProvider"])
    det, rec = net(reader.detector), net(reader.recognizer)

    def det_forward(x, *a, **k):
        y = det_s.run(["maps"], {"image": x.detach().cpu().numpy().astype("float32")})[0]
        return torch.from_numpy(y), None

    def rec_forward(image, *a, **k):
        return torch.from_numpy(rec_s.run(["preds"], {"image": image.detach().cpu().numpy().astype("float32")})[0])

    det.forward = det_forward
    rec.forward = rec_forward
    return reader


def readtext(reader, path):
    return [{"box": [[float(v) for v in pt] for pt in box], "text": t, "conf": float(c)}
            for box, t, c in reader.readtext(path, detail=1, paragraph=False)]


def check(onnx_dir, pages, n):
    """PyTorch EasyOCR vs the same EasyOCR with ONNX networks, page by page."""
    by_lang = {}
    for p in pages:
        by_lang.setdefault(p["lang"], []).append(p)
    res = {}
    for lang, ps in by_lang.items():
        ps = ps[:n]
        torch_r, onnx_r = reader_for(lang), attach(reader_for(lang), onnx_dir, lang)
        same_text = same_box = total = 0
        conf_diff = 0.0
        t_torch = t_onnx = 0.0
        for p in ps:
            t0 = time.time()
            a = readtext(torch_r, p["path"])
            t1 = time.time()
            b = readtext(onnx_r, p["path"])
            t_torch, t_onnx = t_torch + t1 - t0, t_onnx + time.time() - t1
            total += max(len(a), len(b))
            for x, y in zip(a, b):
                same_text += x["text"] == y["text"]
                same_box += x["box"] == y["box"]
                conf_diff = max(conf_diff, abs(x["conf"] - y["conf"]))
        res[lang] = {"pages": len(ps), "segments": total, "same_text": same_text, "same_box": same_box,
                     "max_conf_diff": conf_diff, "s_per_page_torch": round(t_torch / len(ps), 2),
                     "s_per_page_onnx": round(t_onnx / len(ps), 2)}
        print(f"{lang}: {same_text}/{total} segments same text, {same_box}/{total} same box, "
              f"max confidence diff {conf_diff:.1e} | {res[lang]['s_per_page_torch']} s/page PyTorch, "
              f"{res[lang]['s_per_page_onnx']} s/page ONNX", flush=True)
    return res


# ------------------------------------------------------------------ recording for the TypeScript port
def record(onnx_dir, pages, n, out, max_crops=40):
    """Per page: the image, the detector's input and output, the boxes EasyOCR groups, the recognizer's inputs and
    outputs (first max_crops crops), and readtext's result. Arrays go to one compressed .npz per page (float16)."""
    import easyocr
    import numpy as np
    os.makedirs(os.path.join(out, "pages"), exist_ok=True)
    by_lang = {}
    for p in pages:
        by_lang.setdefault(p["lang"], []).append(p)
    cases = []
    for lang, ps in by_lang.items():
        reader = attach(reader_for(lang), onnx_dir, lang)  # the shipped (ONNX) networks
        det, rec = net(reader.detector), net(reader.recognizer)
        for p in ps[:n]:
            stem = f"{lang}_{p['id']}"
            shutil.copyfile(p["path"], os.path.join(out, "pages", stem + os.path.splitext(p["path"])[1]))
            calls = {"det_in": [], "det_out": [], "rec_in": [], "rec_out": []}
            df, rf = det.forward, rec.forward

            def det_rec(x, *a, **k):
                y, f = df(x)
                calls["det_in"].append(x.numpy())
                calls["det_out"].append(y.numpy())
                return y, f

            def rec_rec(image, *a, **k):
                y = rf(image)
                if sum(len(v) for v in calls["rec_in"]) < max_crops:
                    calls["rec_in"].append(image.numpy())
                    calls["rec_out"].append(y.numpy())
                return y

            det.forward, rec.forward = det_rec, rec_rec
            horizontal, free = reader.detect(p["path"])
            calls["det_in"], calls["det_out"] = [], []  # keep the detector call of readtext below
            result = readtext(reader, p["path"])
            det.forward, rec.forward = df, rf
            arrays = {"det_in": calls["det_in"][0].astype(np.float16), "det_out": calls["det_out"][0].astype(np.float16)}
            for k, v in enumerate(calls["rec_in"]):
                arrays[f"rec_in_{k}"] = v.astype(np.float16)
                arrays[f"rec_out_{k}"] = calls["rec_out"][k].astype(np.float16)
            np.savez_compressed(os.path.join(out, "pages", stem + ".npz"), **arrays)
            cases.append({"id": p["id"], "lang": lang, "image": stem + os.path.splitext(p["path"])[1], "npz": stem + ".npz",
                          "horizontal_list": json.loads(json.dumps(horizontal[0], default=float)),
                          "free_list": json.loads(json.dumps(free[0], default=float)),
                          "rec_calls": len(calls["rec_in"]), "readtext": result})
            print(f"{stem}: {len(result)} text segments, det input {list(arrays['det_in'].shape)}, "
                  f"{len(calls['rec_in'])} recognizer calls recorded", flush=True)
    meta = json.load(open(os.path.join(onnx_dir, "easyocr_onnx.json")))
    import inspect
    meta["readtext_defaults"] = {k: (v.default if v.default is not inspect.Parameter.empty else None)
                                 for k, v in inspect.signature(easyocr.Reader.readtext).parameters.items() if k != "self"}
    meta["easyocr_version"] = getattr(easyocr, "__version__", None)
    json.dump({"meta": meta, "cases": cases}, open(os.path.join(out, "easyocr_goldens.json"), "w"), ensure_ascii=False,
              default=str)
    src = os.path.dirname(easyocr.__file__)  # EasyOCR's own code: the port follows it
    with zipfile.ZipFile(os.path.join(out, "easyocr_src.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, fs in os.walk(src):
            for f in fs:
                if f.endswith(".py") or f.endswith(".yaml"):
                    z.write(os.path.join(d, f), os.path.relpath(os.path.join(d, f), os.path.dirname(src)))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "check", "record"])
    ap.add_argument("--pages", required=True)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(argv)
    pages = [json.loads(ln) for ln in open(a.pages, encoding="utf-8") if ln.strip()]
    if a.cmd == "export":
        print(json.dumps(export(a.out, pages)["mb"], indent=2))
    elif a.cmd == "check":
        res = check(a.onnx, pages, a.n)
        json.dump(res, open(os.path.join(a.onnx, "parity.json"), "w"), indent=2)
    else:
        record(a.onnx, pages, a.n, a.out)


if __name__ == "__main__":
    main()
