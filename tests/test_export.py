"""Run: python tests/test_export.py
The round-trip tests need torch + onnx + onnxruntime (installed in the Kaggle notebook); without them they skip."""
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from parda.export.onnx_export import span_agreement  # noqa: E402
from parda.export.onnx_runtime import logits_of  # noqa: E402
from parda.pipeline import models  # noqa: E402

TMP = tempfile.mkdtemp(prefix="parda_export_test_")


def _have(*mods):
    try:
        for m in mods:
            __import__(m)
        return True
    except ImportError:
        return False


class _Fixed:
    def __init__(self, spans):
        self.spans = spans

    def predict(self, text):
        return self.spans


def test_span_agreement():
    a = [{"start": 0, "end": 4, "label": "PAN"}, {"start": 10, "end": 14, "label": "PHONE"}]
    b = [{"start": 0, "end": 4, "label": "PAN"}, {"start": 10, "end": 15, "label": "PHONE"}]
    r = span_agreement(_Fixed(a), _Fixed(b), ["x", "y"])
    assert r["same"] == 2 and r["spans_torch"] == 4 and r["agreement_f1"] == 0.5
    assert span_agreement(_Fixed([]), _Fixed([]), ["x"])["agreement_f1"] == 1.0


def test_logits_of_every_output_kind():
    @dataclass
    class Out:
        logits: object = None
    assert logits_of(Out(logits=7)) == 7 and logits_of({"logits": 8}) == 8 and logits_of((9, 1)) == 9 and logits_of(5) == 5


def test_loader_recognises_exported_models():
    d = os.path.join(TMP, "exp")
    os.makedirs(d)
    json.dump({"source": "me/parda-gliner-v2", "files": {"fp32": "model_fp32.onnx", "int8": "model_int8.onnx"}},
              open(os.path.join(d, "parda_onnx.json"), "w"))
    for f in ("model_fp32.onnx", "model_int8.onnx"):
        open(os.path.join(d, f), "w").write("x")
    assert models.onnx_target(d) == (os.path.join(d, "model_int8.onnx"), "me/parda-gliner-v2")  # folder -> 8-bit
    fp = os.path.join(d, "model_fp32.onnx")
    assert models.onnx_target(fp) == (fp, "me/parda-gliner-v2")
    assert models.onnx_target("me/parda-gliner-v2") is None and models.onnx_target(TMP) is None
    import parda.export.onnx_runtime as R
    real, seen = R.load_onnx, []
    R.load_onnx = lambda f, src: seen.append((f, src)) or "onnx-model"
    try:
        assert models.gliner(d) == "onnx-model" and seen == [(os.path.join(d, "model_int8.onnx"), "me/parda-gliner-v2")]
    finally:
        R.load_onnx = real


# ---------------------------------------------------------------- real round trip on a tiny GLiNER-like model
def _toy(positional=False):
    import torch

    @dataclass
    class ToyOut:
        logits: object = None
        extra: object = None

    class ToyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(300, 16)
            self.lab = torch.nn.Linear(16, 18)

        def forward(self, input_ids, attention_mask, span_idx, span_mask, mode="span"):
            h = self.emb(input_ids) * attention_mask.unsqueeze(-1)
            d = h.size(-1)
            start = torch.gather(h, 1, span_idx[..., 0:1].expand(-1, -1, d))
            end = torch.gather(h, 1, span_idx[..., 1:2].expand(-1, -1, d))
            return ToyOut(logits=self.lab(start + end) * span_mask.unsqueeze(-1), extra=mode)

    class ToyGLiNER:
        """Same interface as GLiNER: .model is the network, predict_entities decodes its logits."""

        def __init__(self):
            self.model = ToyNet().eval()

        def to(self, device):
            return self

        def eval(self):
            return self

        def predict_entities(self, text, labels, threshold=0.5):
            words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
            if not words:
                return []
            ids = torch.tensor([[sum(map(ord, text[a:b])) % 299 + 1 for a, b in words]])
            L = ids.size(1)
            spans = [(i, j) for i in range(L) for j in range(i, min(L, i + 2))]
            args = dict(input_ids=ids, attention_mask=torch.ones(1, L), span_idx=torch.tensor([spans]),
                        span_mask=torch.ones(1, len(spans)))
            with torch.no_grad():
                out = self.model(*args.values()) if positional else self.model(**args, mode="span")
            probs = torch.sigmoid(out.logits)[0]
            res = []
            for k, (i, j) in enumerate(spans):
                c = int(probs[k].argmax())
                if float(probs[k, c]) >= threshold:
                    res.append({"start": words[i][0], "end": words[j][1], "label": labels[c], "score": float(probs[k, c])})
            return res

    return ToyGLiNER()


def test_export_round_trip_is_exact_for_any_text_length():
    if not _have("torch", "onnx", "onnxruntime"):
        print("   (skipped: torch/onnx/onnxruntime not installed)")
        return
    from parda.export.onnx_export import capture, export_gliner_model
    from parda.export.onnx_runtime import attach_onnx
    from parda.pii.predict import GLiNERPredictor
    names, tensors, extra = capture(_toy())
    assert names == ["input_ids", "attention_mask", "span_idx", "span_mask"] and extra == {"mode": "span"}
    out = os.path.join(TMP, "onnx")
    info = export_gliner_model(_toy(), out, "toy", quantize=True)
    assert set(info["files"]) == {"fp32", "int8"} and all(os.path.getsize(os.path.join(out, f)) > 0 for f in info["files"].values())
    assert json.load(open(os.path.join(out, "parda_onnx.json")))["inputs"] == names
    texts = ["Priya", "Name Priya Rao PAN ABCPR1234F", " ".join(f"w{i}" for i in range(260)),  # 1 word .. 2 windows
             "मोबाइल +91 98450 12345 पता Flat 12 Jayanagar"]
    ref = GLiNERPredictor(_toy(), threshold=0.5)
    fp32 = GLiNERPredictor(attach_onnx(_toy(), os.path.join(out, "model_fp32.onnx")), threshold=0.5)
    r = span_agreement(ref, fp32, texts)
    assert r["spans_torch"] > 0 and r["agreement_f1"] == 1.0, r  # conversion alone changes nothing
    int8 = GLiNERPredictor(attach_onnx(_toy(), os.path.join(out, "model_int8.onnx")), threshold=0.5)
    assert span_agreement(ref, int8, texts)["agreement_f1"] > 0.5  # 8-bit runs; its accuracy is measured on Kaggle


def test_capture_refuses_positional_networks():
    if not _have("torch"):
        print("   (skipped: torch not installed)")
        return
    from parda.export.onnx_export import capture
    try:
        capture(_toy(positional=True))
        raise AssertionError("should refuse")
    except RuntimeError as e:
        assert "positional" in str(e)


if __name__ == "__main__":
    try:
        for k, f in list(globals().items()):
            if k.startswith("test_"):
                f()
                print("ok ", k, flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
