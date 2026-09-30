"""Run an exported GLiNER network with onnxruntime inside the normal GLiNER object.

Only the network call is swapped: GLiNER's own tokenisation, windows and span decoding stay exactly as they are,
so any difference in the predictions comes from the ONNX conversion / 8-bit compression alone.
"""
import copy
import dataclasses


def logits_of(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def like(template, logits):
    """An output of the same kind the network returns, carrying the new logits."""
    import torch
    if torch.is_tensor(template):
        return logits
    if dataclasses.is_dataclass(template):
        return dataclasses.replace(template, logits=logits)
    if isinstance(template, dict):
        out = dict(template)
        out["logits"] = logits
        return out
    if isinstance(template, tuple):
        return (logits,) + tuple(template[1:])
    out = copy.copy(template)
    out.logits = logits
    return out


def _meta(onnx_file):
    import json
    import os
    p = os.path.join(os.path.dirname(onnx_file), "parda_onnx.json")
    return json.load(open(p)) if os.path.isfile(p) else {}


def pad_dim1(t, size):
    """Pad a tensor with zeros along dim 1 up to `size` (padded spans are masked out)."""
    import torch
    if t.shape[1] >= size:
        return t
    pad = torch.zeros((t.shape[0], size - t.shape[1]) + tuple(t.shape[2:]), dtype=t.dtype)
    return torch.cat([t.cpu(), pad], 1)


def cut_logits(logits, n_words, fixed):
    """Drop the padded words again: logits laid out (batch, words, ...) or (batch, words*width, ...)."""
    W, K = fixed["max_words"], fixed.get("max_width")
    if logits.shape[1] == W:
        return logits[:, :n_words]
    if K and logits.shape[1] == W * K:
        return logits[:, :n_words * K]
    return logits


def attach_onnx(model, onnx_file, threads=None):
    """model: a loaded GLiNER-like object (has .model, the network). Its network now runs in onnxruntime (CPU).
    If the export fixed the word count (parda_onnx.json "fixed"), inputs are padded to it and outputs cut back."""
    import onnxruntime as ort
    import torch
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_file, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    fixed = _meta(onnx_file).get("fixed")
    net = model.model
    orig = net.forward
    state = {}
    model.onnx_fallbacks = 0

    def forward(*args, **kwargs):
        kw = dict(kwargs)
        if args and isinstance(args[0], dict):
            kw.update(args[0])
        if "template" not in state:  # one real call to learn what kind of object the network returns
            with torch.no_grad():
                state["template"] = orig(**kw)
        feed_src = dict(kw)
        n_words = None
        if fixed:
            n_words = int(kw["text_lengths"].max())
            if n_words > fixed["max_words"]:  # longer than the exported size (never with our windows)
                model.onnx_fallbacks += 1
                with torch.no_grad():
                    return orig(**kw)
            for n, size in fixed["pad_to"].items():
                feed_src[n] = pad_dim1(kw[n], size)
        feed = {n: feed_src[n].detach().cpu().numpy() for n in names}
        logits = torch.from_numpy(sess.run(["logits"], feed)[0])
        if fixed:
            logits = cut_logits(logits, n_words, fixed)
        return like(state["template"], logits)

    net.forward = forward
    model.onnx_session = sess
    model.onnx_file = onnx_file
    return model


def load_onnx(onnx_file, source):
    """GLiNER from `source` (for its tokenizer and decoding) with the network from `onnx_file`."""
    from ..pii.predict import load_gliner
    model = load_gliner(source)
    try:
        model = model.to("cpu")
    except Exception:
        pass
    return attach_onnx(model, onnx_file)
