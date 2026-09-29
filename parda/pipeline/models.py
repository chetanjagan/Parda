"""Load the trained models from a local path or your private Hugging Face repos (needs HF_TOKEN)."""
import os

GLINER_BASE = "urchade/gliner_multi_pii-v1"


def gliner(path_or_repo):
    from ..pii.predict import load_gliner
    return load_gliner(path_or_repo)


def yolo_weights(path_or_repo, filename="best.pt"):
    """A local .pt file, or '<user>/<repo>' on Hugging Face (downloads <repo>/best.pt once, then cached)."""
    if os.path.exists(path_or_repo):
        return path_or_repo
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=path_or_repo, filename=filename, token=os.environ.get("HF_TOKEN"))
