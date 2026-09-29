"""Train the Phase 4 visual PII detector (YOLO) on the dataset built by yolo_data.py.

  # 1) ~3-minute smoke test: 1 epoch on 5% of the training images
  python -m parda.vision.train_yolo --data_yaml data/yolo/data.yaml --out outputs/yolo_smoke --smoke
  # 2) real run, hard-capped at --hours (ultralytics re-plans the epochs to fit the time)
  python -m parda.vision.train_yolo --data_yaml data/yolo/data.yaml --out outputs/yolo --epochs 40 --hours 3

Writes <out>/best.pt, <out>/train_summary.json and the ultralytics run folder <out>/<name>/ (results.csv,
plots). Optionally uploads the weights + summary to a private Hugging Face repo.
"""
import os

# single GPU: Kaggle "T4 x2" exposes 2 GPUs and multi-GPU (DDP) would relaunch this script
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402


def _device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def read_results_csv(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k.strip()] = float(v)
                except (TypeError, ValueError):
                    row[k.strip()] = v
            rows.append(row)
    return rows


def _per_class_ap50(metrics):
    """Per-class val AP50 from the ultralytics metrics object (best effort across versions)."""
    try:
        idx = list(metrics.ap_class_index)
        ap = list(metrics.box.ap50)
        names = metrics.names
        return {names[int(i)]: round(float(a), 4) for i, a in zip(idx, ap)}
    except Exception:
        return None


def summarise(run_dir, minutes, a, metrics=None):
    rows = read_results_csv(os.path.join(run_dir, "results.csv"))
    key = "metrics/mAP50(B)"
    best = max(rows, key=lambda r: r.get(key, -1)) if rows else {}
    last = rows[-1] if rows else {}
    losses = [r.get("train/box_loss") for r in rows if isinstance(r.get("train/box_loss"), float)]
    return {
        "model": a.model, "imgsz": a.imgsz, "batch": a.batch, "smoke": a.smoke,
        "epochs_run": len(rows), "minutes": round(minutes, 1),
        "sec_per_epoch": round(60 * minutes / len(rows), 1) if rows else None,
        "best_epoch": best.get("epoch"),
        "best_val_map50": best.get(key), "best_val_map50_95": best.get("metrics/mAP50-95(B)"),
        "last_val_map50": last.get(key),
        "first_box_loss": losses[0] if losses else None, "last_box_loss": losses[-1] if losses else None,
        "nan_loss": any(math.isnan(v) for v in losses),
        "val_ap50_by_class": _per_class_ap50(metrics) if metrics is not None else None,
    }


def upload(repo, files, token=None):
    from huggingface_hub import HfApi
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo, private=True, exist_ok=True)
    for f in files:
        if os.path.exists(f):
            api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=repo)
    print(f"uploaded to https://huggingface.co/{repo}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11n.pt", help="nano: small enough to run in the browser (Phase 6)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hours", type=float, default=3.0, help="hard time cap for the whole run")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue <out>/train/weights/last.pt")
    ap.add_argument("--hf_repo", default=None, help="e.g. yourname/parda-yolo-v1 (private)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 5%% of train images")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    name = "smoke" if a.smoke else "train"
    run_dir = os.path.join(out, name)
    last = os.path.join(run_dir, "weights", "last.pt")
    t0 = time.time()
    if a.resume and os.path.exists(last):
        print(f"resuming {last}", flush=True)
        model = YOLO(last)
        metrics = model.train(resume=True)
    else:
        model = YOLO(a.model)
        kw = dict(data=a.data_yaml, epochs=1 if a.smoke else a.epochs, imgsz=a.imgsz, batch=a.batch,
                  patience=a.patience, device=_device(), workers=a.workers, project=out, name=name, exist_ok=True,
                  seed=a.seed, deterministic=False, fliplr=0.0,  # documents are never mirrored
                  plots=not a.smoke, fraction=0.05 if a.smoke else 1.0, cache=False, amp=True)
        if a.hours and not a.smoke:
            kw["time"] = a.hours
        print("train args:", json.dumps({k: v for k, v in kw.items() if k != "data"}), flush=True)
        metrics = model.train(**kw)
    minutes = (time.time() - t0) / 60

    trainer = getattr(model, "trainer", None)
    if trainer is not None and getattr(trainer, "save_dir", None):
        run_dir = str(trainer.save_dir)
    weights = os.path.join(run_dir, "weights")
    best = os.path.join(weights, "best.pt")
    if not os.path.exists(best):
        best = os.path.join(weights, "last.pt")
    if os.path.exists(best):
        shutil.copyfile(best, os.path.join(out, "best.pt"))
    summ = summarise(run_dir, minutes, a, metrics)
    if a.smoke and summ["sec_per_epoch"] is not None:
        summ["est_full_epoch_min"] = round(summ["sec_per_epoch"] / 0.05 / 60, 1)  # rough upper bound
    json.dump(summ, open(os.path.join(out, "train_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2), flush=True)
    if summ["nan_loss"]:
        print("WARNING: NaN loss seen. Try --batch 8 or check the labels.", flush=True)
    if a.hf_repo and not a.smoke:
        upload(a.hf_repo, [os.path.join(out, "best.pt"), os.path.join(out, "train_summary.json"),
                           os.path.join(run_dir, "results.csv"), os.path.join(run_dir, "args.yaml"), a.data_yaml])
    return summ


if __name__ == "__main__":
    main()
