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


def attach_onnx(model, onnx_file, threads=None):
    """model: a loaded GLiNER-like object (has .model, the network). Its network now runs in onnxruntime (CPU)."""
    import onnxruntime as ort
    import torch
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_file, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    net = model.model
    orig = net.forward
    state = {}

    def forward(*args, **kwargs):
        kw = dict(kwargs)
        if args and isinstance(args[0], dict):
            kw.update(args[0])
        feed = {n: kw[n].detach().cpu().numpy() for n in names}
        logits = torch.from_numpy(sess.run(["logits"], feed)[0])
        if "template" not in state:  # one real call to learn what kind of object the network returns
            with torch.no_grad():
                state["template"] = orig(**kw)
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
