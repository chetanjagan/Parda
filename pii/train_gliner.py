"""Fine-tune GLiNER on Parda's clean + noisy-OCR PII examples.

  # 1) check data only (no GPU / no gliner needed)
  python -m parda.pii.train_gliner --data_dir data/gliner --dry_run
  # 2) 2-minute smoke test on GPU (10 steps)
  python -m parda.pii.train_gliner --data_dir data/gliner --out outputs/gliner_smoke --smoke
  # 3) real run
  python -m parda.pii.train_gliner --data_dir data/gliner --out outputs/gliner --epochs 1

Saves the final model to <out>/final (GLiNER format), optionally uploads it to a private HF repo.
"""
import argparse
import json
import math
import os
import random
import time

from .labels import LABELS

VALID_LABELS = set(LABELS.values())


def load_examples(path, limit=None, seed=0):
    exs = json.load(open(path, encoding="utf-8"))
    if limit and len(exs) > limit:
        exs = random.Random(seed).sample(exs, limit)
    return exs


def validate(exs, name):
    """Raise on anything GLiNER would choke on; return summary."""
    n_ent, empty, max_len = 0, 0, 0
    for i, ex in enumerate(exs):
        toks, ner = ex.get("tokenized_text"), ex.get("ner")
        assert isinstance(toks, list) and toks and all(isinstance(t, str) and t for t in toks), f"{name}[{i}] tokens"
        assert isinstance(ner, list), f"{name}[{i}] ner"
        for s, e, lab in ner:
            assert 0 <= s <= e < len(toks), f"{name}[{i}] span {s},{e} outside {len(toks)} tokens"
            assert lab in VALID_LABELS, f"{name}[{i}] unknown label {lab}"
        n_ent += len(ner)
        empty += not ner
        max_len = max(max_len, len(toks))
    return {"examples": len(exs), "entities": n_ent, "no_entity_examples": empty, "max_tokens": max_len}


def strip(exs):
    return [{"tokenized_text": e["tokenized_text"], "ner": [list(x) for x in e["ner"]]} for e in exs]


def train(a, tr, va):
    import torch
    from gliner import GLiNER
    from gliner.data_processing.collator import DataCollator
    from gliner.training import Trainer, TrainingArguments
    from transformers import TrainerCallback

    class TimeGuard(TrainerCallback):
        def __init__(self, hours):
            self.deadline = time.time() + hours * 3600

        def on_step_end(self, args, state, control, **kw):
            if time.time() > self.deadline:
                print(f"time guard: stopping at step {state.global_step}", flush=True)
                control.should_save = True
                control.should_training_stop = True
            return control

    print("loading base model:", a.base, flush=True)
    model = GLiNER.from_pretrained(a.base)
    collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    use_fp16 = torch.cuda.is_available() and not a.no_fp16
    steps_per_epoch = math.ceil(len(tr) / a.bs)
    kw = dict(
        output_dir=os.path.join(a.out, "checkpoints"), learning_rate=a.lr, weight_decay=0.01,
        others_lr=a.others_lr, others_weight_decay=0.01, lr_scheduler_type="linear", warmup_ratio=0.1,
        per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs, focal_loss_alpha=0.75,
        focal_loss_gamma=2, num_train_epochs=a.epochs, max_steps=a.max_steps if a.max_steps > 0 else -1,
        save_strategy="steps", save_steps=a.save_steps, save_total_limit=2, eval_steps=a.save_steps,
        logging_steps=a.log_steps, dataloader_num_workers=0, fp16=use_fp16,
        report_to=["wandb"] if a.wandb else "none", eval_strategy="steps",
    )
    if a.wandb:
        kw["run_name"] = os.path.basename(a.out.rstrip("/"))
    try:
        targs = TrainingArguments(**kw)
    except TypeError:  # older transformers name
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
        targs = TrainingArguments(**kw)
    base_kw = dict(model=model, args=targs, train_dataset=tr, eval_dataset=va, data_collator=collator,
                   callbacks=[TimeGuard(a.max_hours)])
    tok = model.data_processor.transformer_tokenizer
    trainer = None
    for key in ("processing_class", "tokenizer"):  # newer / older transformers
        try:
            trainer = Trainer(**base_kw, **{key: tok})
            break
        except TypeError:
            continue
    if trainer is None:
        trainer = Trainer(**base_kw)

    print(f"train={len(tr)} val={len(va)} batch={a.bs} steps/epoch={steps_per_epoch} fp16={use_fp16}", flush=True)
    last = None
    ck = os.path.join(a.out, "checkpoints")
    if a.resume and os.path.isdir(ck):
        cks = sorted((d for d in os.listdir(ck) if d.startswith("checkpoint-")), key=lambda d: int(d.split("-")[1]))
        last = os.path.join(ck, cks[-1]) if cks else None
        print("resuming from", last)
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=last)
    dt = time.time() - t0
    steps = max(1, result.global_step)
    print(f"trained {steps} steps in {dt / 60:.1f} min ({dt / steps:.2f} s/step)", flush=True)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    evals = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
    if losses and any(x != x for x in losses):
        print("WARNING: NaN loss seen. Re-run with --no_fp16", flush=True)
    final = os.path.join(a.out, "final")
    model.save_pretrained(final)
    summary = {"steps": steps, "minutes": round(dt / 60, 1), "sec_per_step": round(dt / steps, 3),
               "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
               "eval_losses": evals, "base": a.base, "train_examples": len(tr), "fp16": use_fp16}
    json.dump(summary, open(os.path.join(a.out, "train_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    # quick sanity prediction on a noisy sentence
    m = GLiNER.from_pretrained(final)
    if torch.cuda.is_available():
        m = m.to("cuda")
    demo = "Name: Ramesh Kumar PAN: ABCPK1234F Aadhaar No: 4829 1736 5520 IFSC: SBINO364507 Mobile: 98450 12345"
    for ent in m.predict_entities(demo, list(LABELS.values()), threshold=0.3):
        print(f"   {ent['label']:22s} {ent['text']!r} {ent['score']:.2f}")

    if a.hf_repo:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(a.hf_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=final, repo_id=a.hf_repo)
        print("uploaded model to", a.hf_repo)
    return final


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/gliner")
    ap.add_argument("--out", default="outputs/gliner")
    ap.add_argument("--base", default="urchade/gliner_multi_pii-v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--others_lr", type=float, default=3e-5)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--log_steps", type=int, default=50)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_val", type=int, default=600)
    ap.add_argument("--max_hours", type=float, default=8.0)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--hf_repo", default=None, help="e.g. yourname/parda-gliner (private)")
    ap.add_argument("--smoke", action="store_true", help="10 steps on 64 examples to check everything works")
    ap.add_argument("--dry_run", action="store_true", help="validate data only, no training")
    a = ap.parse_args(argv)

    if a.smoke:
        a.max_train, a.max_val, a.max_steps, a.save_steps, a.log_steps = 64, 16, 10, 5, 2
    tr = load_examples(os.path.join(a.data_dir, "train.json"), a.max_train)
    va = load_examples(os.path.join(a.data_dir, "val.json"), a.max_val)
    info = {"train": validate(tr, "train"), "val": validate(va, "val")}
    print(json.dumps(info, indent=2), flush=True)
    if a.dry_run:
        return None
    os.makedirs(a.out, exist_ok=True)
    return train(a, strip(tr), strip(va))


if __name__ == "__main__":
    main()
