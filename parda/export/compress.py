"""Shrink the exported GLiNER for the browser without losing accuracy: try several compressions, keep the
smallest one that still finds the same PII as PyTorch.

  python -m parda.export.compress --gliner_dir outputs/onnx/gliner --parity_ocr cache/ocr_tesseract.jsonl ... \
         --parity_n 100 --min_agreement 0.99

Plain 8-bit compression of every layer (QUInt8 on activations) kept only 36% of PyTorch's spans for this model
(mDeBERTa), so the variants below compress more carefully:
  int8_embed            only the word-embedding table (Gather), weights to int8   (most of the model's size)
  int8_matmul_pc        matrix multiplications, signed int8, per output channel; embeddings untouched
  int8_embed_matmul_pc  both of the above
  fp16                  every weight in 16-bit floats (inputs/outputs stay 32-bit)
Writes model_<variant>.onnx next to model_fp32.onnx, compress_summary.json, and records the chosen variant as
"default" in parda_onnx.json (the pipeline loads that one for the folder).
"""
import argparse
import json
import os
import time

VARIANTS = ("int8_embed", "int8_matmul_pc", "int8_embed_matmul_pc", "fp16")


def make_variant(fp32, out, variant):
    """Write one compressed copy of fp32 to out."""
    if variant == "fp16":
        import onnx
        from onnxconverter_common import float16
        model = float16.convert_float_to_float16(onnx.load(fp32), keep_io_types=True)
        onnx.save(model, out)
        return
    from onnxruntime.quantization import QuantType, quantize_dynamic
    ops = {"int8_embed": ["Gather"], "int8_matmul_pc": ["MatMul"], "int8_embed_matmul_pc": ["Gather", "MatMul"]}[variant]
    quantize_dynamic(fp32, out, weight_type=QuantType.QInt8, per_channel="matmul" in variant,
                     op_types_to_quantize=ops, extra_options={"MatMulConstBOnly": True})


def choose(results, min_agreement):
    """results: {variant: {"mb": .., "agreement_f1": ..}}. The smallest variant at or above min_agreement;
    if none qualifies, the most accurate one (and the caller should keep fp32)."""
    good = {k: v for k, v in results.items() if v.get("agreement_f1", 0) >= min_agreement}
    if good:
        return min(good, key=lambda k: good[k]["mb"]), True
    return max(results, key=lambda k: results[k].get("agreement_f1", 0)), False


def compress(gliner_dir, texts, variants=VARIANTS, min_agreement=0.99, threshold=0.3, model_loader=None):
    """model_loader(): a fresh GLiNER-like object (defaults to the torch source in parda_onnx.json)."""
    from ..pii.predict import GLiNERPredictor, load_gliner
    from .onnx_export import _cpu, span_agreement
    from .onnx_runtime import attach_onnx
    meta_path = os.path.join(gliner_dir, "parda_onnx.json")
    meta = json.load(open(meta_path))
    fp32 = os.path.join(gliner_dir, meta["files"]["fp32"])
    load = model_loader or (lambda: _cpu(load_gliner(meta["source"])))
    ref = GLiNERPredictor(load(), threshold, name="torch")
    results = {}
    for v in variants:
        out = os.path.join(gliner_dir, f"model_{v}.onnx")
        t0 = time.time()
        try:
            make_variant(fp32, out, v)
        except Exception as e:  # e.g. a missing package: report it and try the next variant
            results[v] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"{v}: could not be made ({results[v]['error']})", flush=True)
            continue
        made = time.time() - t0
        t0 = time.time()
        r = span_agreement(ref, GLiNERPredictor(attach_onnx(load(), out), threshold, name=v), texts)
        results[v] = {"mb": round(os.path.getsize(out) / 2 ** 20, 1), "agreement_f1": r["agreement_f1"],
                      "spans_torch": r["spans_torch"], "spans_onnx": r["spans_onnx"],
                      "make_seconds": round(made, 1), "check_seconds": round(time.time() - t0, 1)}
        meta["files"][v] = os.path.basename(out)
        print(f"{v}: {results[v]['mb']} MB, same spans as PyTorch {r['agreement_f1']:.1%}", flush=True)
    ok = {k: v for k, v in results.items() if "error" not in v}
    best, passed = choose(ok, min_agreement) if ok else (None, False)
    meta["default"] = best if passed else "fp32"
    json.dump(meta, open(meta_path, "w"), indent=2)
    summary = {"variants": results, "min_agreement": min_agreement, "texts": len(texts),
               "chosen": meta["default"], "chosen_passed": passed, "fp32_mb": round(os.path.getsize(fp32) / 2 ** 20, 1)}
    json.dump(summary, open(os.path.join(gliner_dir, "compress_summary.json"), "w"), indent=2)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gliner_dir", required=True, help="folder with model_fp32.onnx and parda_onnx.json")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--parity_ocr", nargs="+", required=True)
    ap.add_argument("--parity_n", type=int, default=50, help="texts per OCR file")
    ap.add_argument("--min_agreement", type=float, default=0.99)
    a = ap.parse_args(argv)
    from .onnx_export import parity_texts
    s = compress(a.gliner_dir, parity_texts(a.parity_ocr, a.parity_n), [v for v in a.variants.split(",") if v],
                 a.min_agreement)
    print(json.dumps(s, indent=2), flush=True)
    return s


if __name__ == "__main__":
    main()
