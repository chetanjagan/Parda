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
    """quantize=False: on CPU EasyOCR would otherwise compress its networks to 8-bit (which ONNX export cannot take).
    The benchmarks ran EasyOCR on GPU, where it never quantizes, so the full-precision networks are the ones to match."""
    import easyocr
    return easyocr.Reader(READER_LANGS[lang], gpu=False, quantize=False, verbose=False)


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


def exportable(model):
    """AdaptiveAvgPool2d((None, 1)) ("keep this axis, average the last one to 1") cannot be exported with a dynamic
    width; it is exactly a mean over the last axis, so swap it for that (in place; only used for the export)."""
    import torch

    class MeanLast(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=-1, keepdim=True)

    swapped = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.AdaptiveAvgPool2d) and tuple(child.output_size) == (None, 1):
                setattr(mod, child_name, MeanLast())
                swapped.append(f"{name}.{child_name}".strip("."))
    return swapped


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
        with torch.no_grad():
            before = rec(img_t, text)
        info["swapped_for_export"] = exportable(rec)
        with torch.no_grad():  # the swap must not change anything
            diff = float((rec(img_t, text) - before).abs().max())
        if diff > 1e-5:
            raise RuntimeError(f"replacing the pooling changed the recognizer output (max diff {diff})")
        info["swap_max_diff"] = diff

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
def record(onnx_dir, pages, n, out, max_crops=300):
    """Per page, every stage separately (one compressed .npz per page):
      page        the image exactly as EasyOCR decoded it (RGB uint8, also saved as PNG)
      det_out     the detector's score maps, float32 (thresholds at 0.4 / 0.7 need full precision)
      raw_boxes   get_textbox's boxes before grouping; horizontal_list / free_list after grouping (json)
      rec_in_k    each recognizer input, as uint8 (exactly recoverable from the normalised [-1, 1] values)
      rec_out_k   each recognizer output, float32
    and readtext's result."""
    import cv2
    import easyocr
    import numpy as np
    from easyocr.detection import get_textbox
    from easyocr.utils import reformat_input
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
            img, grey = reformat_input(p["path"])  # what EasyOCR works on
            cv2.imwrite(os.path.join(out, "pages", stem + ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            calls = {"det_out": [], "rec_in": [], "rec_out": []}
            df, rf = det.forward, rec.forward

            def det_rec(x, *a, **k):
                y, f = df(x)
                calls["det_out"].append(y.numpy().astype(np.float32))
                return y, f

            def rec_rec(image, *a, **k):
                y = rf(image)
                if sum(len(v) for v in calls["rec_in"]) < max_crops:
                    calls["rec_in"].append(np.rint((image.numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8))
                    calls["rec_out"].append(y.numpy().astype(np.float32))
                return y

            d = reader_defaults()
            raw = get_textbox(reader.detector, img, canvas_size=d["canvas_size"], mag_ratio=d["mag_ratio"],
                              text_threshold=d["text_threshold"], link_threshold=d["link_threshold"], low_text=d["low_text"],
                              poly=False, device="cpu", optimal_num_chars=None, threshold=d["threshold"],
                              bbox_min_score=d["bbox_min_score"], bbox_min_size=d["bbox_min_size"],
                              max_candidates=d["max_candidates"])
            horizontal, free = reader.detect(p["path"])
            det.forward, rec.forward = det_rec, rec_rec
            result = readtext(reader, p["path"])
            det.forward, rec.forward = df, rf
            arrays = {"page": img, "det_out": calls["det_out"][0]}
            for k, v in enumerate(calls["rec_in"]):
                arrays[f"rec_in_{k}"] = v
                arrays[f"rec_out_{k}"] = calls["rec_out"][k]
            np.savez_compressed(os.path.join(out, "pages", stem + ".npz"), **arrays)
            j = lambda x: json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v)))  # noqa: E731
            cases.append({"id": p["id"], "lang": lang, "page_png": stem + ".png", "npz": stem + ".npz",
                          "page_hw": list(img.shape[:2]), "det_out_shape": list(arrays["det_out"].shape),
                          "raw_boxes": j(raw[0]), "horizontal_list": j(horizontal[0]), "free_list": j(free[0]),
                          "rec_calls": [list(v.shape) for v in calls["rec_in"]], "readtext": result})
            print(f"{stem}: {len(j(raw[0]))} raw boxes -> {len(horizontal[0])} lines + {len(free[0])} tilted -> "
                  f"{len(result)} text segments | {len(calls['rec_in'])} recognizer calls", flush=True)
    meta = json.load(open(os.path.join(onnx_dir, "easyocr_onnx.json")))
    meta["readtext_defaults"] = reader_defaults()
    meta["easyocr_version"] = getattr(easyocr, "__version__", None)
    meta["opencv_version"] = cv2.__version__
    json.dump({"meta": meta, "cases": cases}, open(os.path.join(out, "easyocr_goldens.json"), "w"), ensure_ascii=False,
              default=str)
    src = os.path.dirname(easyocr.__file__)  # EasyOCR's own code: the port follows it
    with zipfile.ZipFile(os.path.join(out, "easyocr_src.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for d_, _, fs in os.walk(src):
            for f in fs:
                if f.endswith(".py") or f.endswith(".yaml"):
                    z.write(os.path.join(d_, f), os.path.relpath(os.path.join(d_, f), os.path.dirname(src)))
    return cases


def reader_defaults():
    import inspect
    import easyocr
    return {k: (v.default if v.default is not inspect.Parameter.empty else None)
            for k, v in inspect.signature(easyocr.Reader.readtext).parameters.items() if k != "self"}


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
