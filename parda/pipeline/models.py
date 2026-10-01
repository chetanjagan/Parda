"""Load the trained models from a local path or your private Hugging Face repos (needs HF_TOKEN)."""
import os

GLINER_BASE = "urchade/gliner_multi_pii-v1"


def onnx_target(path):
    """(onnx_file, torch_source) if `path` is an exported model (a .onnx file or its folder), else None.
    A folder means the variant chosen by parda.export.compress ("default"), else the full-precision model."""
    import json
    if path.endswith(".onnx") and os.path.isfile(path):
        folder, onnx_file = os.path.dirname(path), path
    elif os.path.isdir(path) and os.path.isfile(os.path.join(path, "parda_onnx.json")):
        folder, onnx_file = path, None
    else:
        return None
    meta = json.load(open(os.path.join(folder, "parda_onnx.json")))
    if onnx_file is None:
        onnx_file = os.path.join(folder, meta["files"].get(meta.get("default", "fp32")) or meta["files"]["fp32"])
    return onnx_file, meta["source"]


def gliner(path_or_repo):
    """A GLiNER model; an exported ONNX model runs in onnxruntime with GLiNER's own pre/post-processing."""
    target = onnx_target(path_or_repo)
    if target:
        from ..export.onnx_runtime import load_onnx
        return load_onnx(*target)
    from ..pii.predict import load_gliner
    return load_gliner(path_or_repo)


def yolo_weights(path_or_repo, filename="best.pt"):
    """A local .pt file, or '<user>/<repo>' on Hugging Face (downloads <repo>/best.pt once, then cached)."""
    if os.path.exists(path_or_repo):
        return path_or_repo
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=path_or_repo, filename=filename, token=os.environ.get("HF_TOKEN"))
