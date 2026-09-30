"""Phase 6, step 1: export YOLO and GLiNER to ONNX (+ an 8-bit GLiNER) and check the predictions still agree.

  python -m parda.export.onnx_export --gliner <you>/parda-gliner-v2 --yolo <you>/parda-yolo-v1 --out outputs/onnx \
         --parity_ocr outputs/cache/ocr_tesseract.jsonl outputs/cache/ocr_easyocr.jsonl

Writes
  <out>/gliner/model_fp32.onnx, model_int8.onnx, parda_onnx.json (+ tokenizer/ and gliner_config.json for the web app)
  <out>/yolo/parda-yolo.onnx
  <out>/export_summary.json   sizes, and how often the ONNX models find exactly the same PII spans as PyTorch

The GLiNER export is version-proof: it records the exact named tensors GLiNER feeds its network during one normal
prediction and exports the network for those inputs (sequence lengths stay dynamic).
"""
import argparse
import json
import os
import shutil
import time

SAMPLE = ("Name: Priya Rao, PAN ABCPR1234F, Aadhaar 4829 1736 5520, mobile +91 98450 12345, UPI priya.rao@okaxis, "
          "lives at Flat 12, 4th Cross, Jayanagar, Bengaluru - 560041. Employee ID EMP12334, UAN 100660595745.")


MAX_WORDS = 256  # the exported GLiNER always sees this many words; shorter texts are padded (windows are <= 200)


def dummy_text(n_words):
    """n one-word, one-token words: the export runs on a text of exactly the maximum size."""
    return " ".join(["a"] * n_words)


def fixed_shapes(names, tensors):
    """GLiNER sizes some internal tables by the number of words, and export records that number as a constant.
    Returns what the runtime must pad to that size (None if the network has no word-count input)."""
    d = dict(zip(names, tensors))
    if "text_lengths" not in d:
        return None
    W = int(d["text_lengths"].max())
    K = d["span_idx"].shape[1] // W if "span_idx" in d and W else None
    pad = {}
    for n, t in d.items():
        if n in ("input_ids", "attention_mask", "words_mask") or t.dim() < 2:
            continue  # token-level inputs stay dynamic
        if t.shape[1] == W or (K and t.shape[1] == W * K):
            pad[n] = int(t.shape[1])
    return {"max_words": W, "max_width": K, "pad_to": pad}


def _labels():
    from ..pii.labels import LABELS
    return list(LABELS.values())


def capture(model, text=SAMPLE, labels=None, threshold=0.3):
    """Run one normal prediction; return (tensor input names, tensors, constant non-tensor inputs)."""
    import torch
    seen = {}

    def hook(mod, args, kwargs):
        if "kw" in seen:
            return
        kw = dict(kwargs)
        if args and isinstance(args[0], dict):
            kw.update(args[0])
        elif args:
            seen["positional"] = len(args)
        seen["kw"] = kw

    h = model.model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model.predict_entities(text, labels or _labels(), threshold=threshold)
    finally:
        h.remove()
    if "kw" not in seen:
        raise RuntimeError("GLiNER never called its network during predict_entities")
    if seen.get("positional"):
        raise RuntimeError("GLiNER passes positional tensors to its network; the export needs named inputs")
    kw = seen["kw"]
    names = [k for k, v in kw.items() if torch.is_tensor(v)]
    extra = {k: v for k, v in kw.items() if not torch.is_tensor(v)}
    return names, [kw[k] for k in names], extra


def make_wrapper(net, names, extra):
    import torch
    from .onnx_runtime import logits_of

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, *xs):
            return logits_of(self.net(**dict(zip(names, xs)), **extra))

    return Wrapper().eval()


def export_gliner_model(model, out_dir, source, opset=14, quantize=True, max_words=MAX_WORDS):
    """model: a loaded GLiNER(-like) object on CPU. Returns info about the exported files.
    Exported on a text of exactly `max_words` words (see fixed_shapes); onnx_runtime pads shorter texts."""
    import torch
    os.makedirs(out_dir, exist_ok=True)
    try:
        model.eval()
    except Exception:
        pass
    names, tensors, extra = capture(model, text=dummy_text(max_words))
    fixed = fixed_shapes(names, tensors)
    wrapper = make_wrapper(model.model, names, extra)
    with torch.no_grad():
        n_dims = wrapper(*tensors).dim()
    # batch and length dims are dynamic; deeper dims (e.g. span_idx's [start, end] pair) stay fixed
    dyn = {n: ({0: "batch"} if t.dim() == 1 else {0: "batch", 1: f"{n}_len"}) for n, t in zip(names, tensors)}
    dyn["logits"] = {i: f"logits_{i}" for i in range(n_dims)}
    fp32 = os.path.join(out_dir, "model_fp32.onnx")
    kw = dict(input_names=names, output_names=["logits"], dynamic_axes=dyn, opset_version=opset, do_constant_folding=True)
    t0 = time.time()
    exporter = "torchscript"
    with torch.no_grad():
        try:
            torch.onnx.export(wrapper, tuple(tensors), fp32, dynamo=False, **kw)
        except TypeError:  # older torch: no `dynamo` argument (TorchScript is its only exporter)
            torch.onnx.export(wrapper, tuple(tensors), fp32, **kw)
        except Exception as first:  # a torch without the TorchScript exporter: use the new one
            print(f"TorchScript export failed ({type(first).__name__}: {first}); trying the dynamo exporter", flush=True)
            torch.onnx.export(wrapper, tuple(tensors), fp32, dynamo=True, external_data=False, **kw)
            exporter = "dynamo"
    info = {"source": source, "inputs": names, "fixed": fixed, "constant_inputs": {k: repr(v) for k, v in extra.items()},
            "opset": opset, "exporter": exporter, "files": {"fp32": "model_fp32.onnx"},
            "export_seconds": round(time.time() - t0, 1)}
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(fp32, os.path.join(out_dir, "model_int8.onnx"), weight_type=QuantType.QUInt8)
        info["files"]["int8"] = "model_int8.onnx"
    info["mb"] = {k: round(os.path.getsize(os.path.join(out_dir, f)) / 2 ** 20, 1) for k, f in info["files"].items()}
    for what, fn in (("tokenizer", lambda: model.data_processor.transformer_tokenizer.save_pretrained(
                         os.path.join(out_dir, "tokenizer"))),
                     ("config", lambda: json.dump(model.config.to_dict(), open(os.path.join(out_dir, "gliner_config.json"), "w"), indent=2))):
        try:  # the web app needs these; best effort (field names differ across GLiNER versions)
            fn()
            info[f"saved_{what}"] = True
        except Exception as e:
            info[f"saved_{what}"] = f"no ({type(e).__name__})"
    json.dump(info, open(os.path.join(out_dir, "parda_onnx.json"), "w"), indent=2)
    return info


def span_agreement(pred_a, pred_b, texts):
    """How often two predictors find exactly the same (start, end, label) spans. F1 of the two span sets."""
    a_n = b_n = both = 0
    for t in texts:
        a = {(s["start"], s["end"], s["label"]) for s in pred_a.predict(t)}
        b = {(s["start"], s["end"], s["label"]) for s in pred_b.predict(t)}
        a_n, b_n, both = a_n + len(a), b_n + len(b), both + len(a & b)
    return {"texts": len(texts), "spans_torch": a_n, "spans_onnx": b_n, "same": both,
            "agreement_f1": round(2 * both / (a_n + b_n), 4) if a_n + b_n else 1.0}


def parity_texts(ocr_files, n=150):
    """Up to n real OCR page texts from each file (varied lengths, real OCR noise)."""
    from ..pii.spans import build_ocr_text
    texts = []
    for f in ocr_files:
        k = 0
        for ln in open(f, encoding="utf-8"):
            text = build_ocr_text(json.loads(ln).get("segments") or [])[0]
            if text.strip():
                texts.append(text)
                k += 1
            if k >= n:
                break
    return texts


def _cpu(model):
    try:
        return model.to("cpu")
    except Exception:
        return model


def export_yolo(src, out_dir, imgsz=1024):
    from ultralytics import YOLO
    from ..pipeline.models import yolo_weights
    os.makedirs(out_dir, exist_ok=True)
    path = YOLO(yolo_weights(src)).export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False)
    dst = os.path.join(out_dir, "parda-yolo.onnx")
    shutil.copyfile(str(path), dst)
    return {"file": dst, "imgsz": imgsz, "mb": round(os.path.getsize(dst) / 2 ** 20, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gliner", required=True, help="local path or HF repo of the (v2) GLiNER")
    ap.add_argument("--yolo", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--max_words", type=int, default=MAX_WORDS)
    ap.add_argument("--no_quantize", action="store_true")
    ap.add_argument("--parity_ocr", nargs="*", default=[], help="OCR jsonl files whose page texts are used for the check")
    ap.add_argument("--parity_n", type=int, default=150, help="texts per OCR file")
    ap.add_argument("--threshold", type=float, default=0.3)
    a = ap.parse_args(argv)
    from ..pii.predict import GLiNERPredictor, load_gliner
    from .onnx_runtime import attach_onnx
    summary = {}
    if a.yolo:
        summary["yolo"] = export_yolo(a.yolo, os.path.join(a.out, "yolo"))
        print("YOLO:", summary["yolo"], flush=True)
    gdir = os.path.join(a.out, "gliner")
    torch_model = _cpu(load_gliner(a.gliner))
    summary["gliner"] = export_gliner_model(torch_model, gdir, a.gliner, a.opset, not a.no_quantize, a.max_words)
    print("GLiNER:", json.dumps(summary["gliner"]["mb"]), flush=True)
    texts = parity_texts(a.parity_ocr, a.parity_n) if a.parity_ocr else [SAMPLE]
    ref = GLiNERPredictor(torch_model, a.threshold, name="torch")
    summary["agreement"] = {}
    for variant, fname in summary["gliner"]["files"].items():
        m = attach_onnx(_cpu(load_gliner(a.gliner)), os.path.join(gdir, fname))  # a fresh copy per variant
        t0 = time.time()
        summary["agreement"][variant] = span_agreement(ref, GLiNERPredictor(m, a.threshold, name=variant), texts)
        summary["agreement"][variant]["seconds"] = round(time.time() - t0, 1)
        summary["agreement"][variant]["too_long_fallbacks"] = getattr(m, "onnx_fallbacks", 0)
        print(f"{variant}: same spans as PyTorch on {len(texts)} OCR texts:",
              summary["agreement"][variant]["agreement_f1"], flush=True)
    json.dump(summary, open(os.path.join(a.out, "export_summary.json"), "w"), indent=2)
    return summary


if __name__ == "__main__":
    main()
