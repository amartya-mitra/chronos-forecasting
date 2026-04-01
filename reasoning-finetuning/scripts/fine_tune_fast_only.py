#!/usr/bin/env python3
"""
Fine-tune chronos-t5-small on sarsim0-fast-only.arrow.

All 10,000 rows are fast-mode: the model learns that the fast-mode
prefix token (4096) maps to standard forecasting behaviour.

Inherits all v3 optimisations:
  - Batch size 128
  - Pre-tokenised + pre-concatenated input_ids (no per-step CPU cat)
  - Batched eval inference (EVAL_BATCH=32)
  - Eval every 500 steps

tqdm fix vs v3: per-step loss is embedded in the bar's postfix via
set_postfix() — no tqdm.write() call per step, so no extra newlines.
tqdm.write() is used only for eval milestones (~20 lines total).

Usage:
  python fine_tune_fast_only.py [--job-id JOB_ID]

Outputs (figures/finetune_fast_only_{JOB_ID}/):
  config.txt
  loss.png
  eval_fast_samples.png

Checkpoint: checkpoints/fast-only-finetune-{JOB_ID}/
"""

import sys
import argparse
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import random
import numpy as np
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, GenerationConfig

from chronos import ChronosPipeline

# ── hyperparameters ────────────────────────────────────────────────────────────
FAST_TOKEN_ID  = 4096
EXTENDED_VOCAB = 4098   # keep vocab size consistent with v3 (room for trend token)

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 64

BATCH_SIZE    = 128
N_STEPS       = 3_000
WARMUP_STEPS  = 500
LR            = 1e-4
GRAD_CLIP     = 1.0
LOG_EVERY     = 50
EVAL_EVERY    = 500
EVAL_BATCH    = 32
EVAL_SAMPLES  = 20
EVAL_N_SHOW   = 5
EVAL_POOL     = 100     # held-out fast rows

SEED     = 42
MODEL_ID = "amazon/chronos-t5-small"
DATA_PATH = FINETUNING_ROOT / "data" / "sarsim0-fast-only.arrow"

_CTX_COL  = "steelblue"
_GT_COL   = "darkorange"
_PRED_COL = "crimson"


# ── reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── config record ──────────────────────────────────────────────────────────────

def write_config(path: Path, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Experiment Configuration",
        "=" * 40,
        f"Job ID          : {job_id}",
        f"Model           : {MODEL_ID}",
        f"Dataset         : sarsim0-fast-only.arrow",
        f"Dataset split   : 100% fast-mode",
        f"Fast token ID   : {FAST_TOKEN_ID}",
        f"Extended vocab  : {EXTENDED_VOCAB}",
        f"Context length  : {CONTEXT_LENGTH}",
        f"Pred length     : {PREDICTION_LENGTH}",
        f"Batch size      : {BATCH_SIZE}",
        f"Steps           : {N_STEPS:,}",
        f"Warmup steps    : {WARMUP_STEPS}",
        f"Learning rate   : {LR}",
        f"Grad clip       : {GRAD_CLIP}",
        f"Eval every      : {EVAL_EVERY} steps",
        f"Eval batch size : {EVAL_BATCH}",
        f"Eval pool       : {EVAL_POOL}",
        f"Seed            : {SEED}",
        "",
        "Motivation:",
        "  Train the model to associate fast-mode prefix (4096) with",
        "  standard 64-step forecasting. Pure fast-only dataset so the",
        "  model cannot confuse the two prefix tokens.",
    ]
    path.write_text("\n".join(lines) + "\n")
    tqdm.write(f"  Config written: {path.name}", file=sys.stdout)


# ── data ───────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> tuple[list[dict], list[dict]]:
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()

    rows = []
    for item_id, mode, ctx, tgt in zip(
        table.column("item_id").to_pylist(),
        table.column("mode").to_pylist(),
        table.column("context").to_pylist(),
        table.column("target").to_pylist(),
    ):
        rows.append({
            "item_id": item_id,
            "mode":    mode,
            "context": np.array(ctx, dtype=np.float32),
            "target":  np.array(tgt, dtype=np.float32),
        })

    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)

    eval_rows  = rows[:EVAL_POOL]
    train_rows = rows[EVAL_POOL:]
    rng.shuffle(train_rows)

    return train_rows, eval_rows


def pretokenise(rows: list[dict], tokenizer) -> list[dict]:
    """
    Pre-tokenise all training rows once on CPU.

    Stores fully assembled encoder input (mode prefix pre-concatenated):
      input_ids : (514,) int tensor   [fast_token | context_tokens | EOS]
      attn_mask : (514,) bool tensor
      labels    : (65,)  int tensor   [-100 for padding]
    """
    N = len(rows)
    tqdm.write(f"  Pre-tokenising {N:,} training rows on CPU...", file=sys.stdout)

    contexts = torch.tensor(np.stack([d["context"] for d in rows]), dtype=torch.float32)
    targets  = torch.tensor(np.stack([d["target"]  for d in rows]), dtype=torch.float32)

    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    label_ids, label_mask       = tokenizer.label_input_transform(targets, scale)
    labels = label_ids.masked_fill(~label_mask.bool(), -100)

    # All rows are fast-mode — pre-fill mode prefix column
    mode_col      = torch.full((N, 1), FAST_TOKEN_ID, dtype=token_ids.dtype)
    mode_mask_col = torch.ones(N, 1, dtype=torch.bool)

    full_input_ids = torch.cat([mode_col,      token_ids], dim=1)   # (N, 514)
    full_attn_mask = torch.cat([mode_mask_col, attn_mask], dim=1)   # (N, 514)

    tokenised = []
    for i, d in enumerate(rows):
        tokenised.append({
            "item_id":   d["item_id"],
            "context":   d["context"],
            "target":    d["target"],
            "input_ids": full_input_ids[i],
            "attn_mask": full_attn_mask[i],
            "labels":    labels[i],
        })

    tqdm.write("  Pre-tokenisation complete.", file=sys.stdout)
    return tokenised


class CyclicLoader:
    def __init__(self, rows: list[dict], batch_size: int, seed: int = 0):
        self.rows       = rows
        self.batch_size = batch_size
        self.rng        = np.random.default_rng(seed)
        self._indices   = np.arange(len(rows))
        self._pos       = len(rows)

    def next_batch(self) -> list[dict]:
        if self._pos + self.batch_size > len(self._indices):
            self.rng.shuffle(self._indices)
            self._pos = 0
        batch     = [self.rows[i] for i in self._indices[self._pos:self._pos + self.batch_size]]
        self._pos += self.batch_size
        return batch


# ── training step assembly ─────────────────────────────────────────────────────

def assemble_batch(
    batch: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack pre-assembled CPU tensors and issue one .to(device) per tensor."""
    input_ids = torch.stack([d["input_ids"] for d in batch]).to(device)
    attn_mask = torch.stack([d["attn_mask"] for d in batch]).to(device)
    labels    = torch.stack([d["labels"]    for d in batch]).to(device)
    return input_ids, attn_mask, labels


# ── batched eval inference ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(
    pipeline,
    contexts: list[np.ndarray],
    device: torch.device,
    prediction_length: int = PREDICTION_LENGTH,
    num_samples: int = EVAL_SAMPLES,
) -> list[np.ndarray]:
    """One model.generate() call for a full batch, all fast-mode."""
    B   = len(contexts)
    ctx = torch.tensor(np.stack(contexts), dtype=torch.float32)

    token_ids, attn_mask, scale = pipeline.tokenizer.context_input_transform(ctx)

    mode_prefix = torch.full((B, 1), FAST_TOKEN_ID, dtype=token_ids.dtype)
    mode_mask   = torch.ones(B, 1, dtype=torch.bool)

    input_ids_dev = torch.cat([mode_prefix, token_ids], dim=1).to(device)
    attn_mask_dev = torch.cat([mode_mask,   attn_mask], dim=1).to(device)

    cfg = pipeline.model.config
    raw = pipeline.model.model.generate(
        input_ids=input_ids_dev,
        attention_mask=attn_mask_dev,
        generation_config=GenerationConfig(
            min_new_tokens=prediction_length,
            max_new_tokens=prediction_length,
            do_sample=True,
            num_return_sequences=num_samples,
            eos_token_id=cfg.eos_token_id,
            pad_token_id=cfg.pad_token_id,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        ),
    )
    raw     = raw[:, 1:].reshape(B, num_samples, prediction_length)
    preds   = pipeline.tokenizer.output_transform(raw.cpu(), scale)
    medians = preds.median(dim=1).values.numpy()
    return [medians[i] for i in range(B)]


# ── evaluation ─────────────────────────────────────────────────────────────────

def evaluate(pipeline, eval_rows: list[dict], device: torch.device, step: int) -> float:
    pipeline.model.model.eval()
    maes = []
    bar  = tqdm(
        range(0, len(eval_rows), EVAL_BATCH),
        desc=f"  eval step={step:,}",
        file=sys.stdout,
        leave=False,
        dynamic_ncols=True,
    )
    for i in bar:
        batch    = eval_rows[i:i + EVAL_BATCH]
        contexts = [r["context"] for r in batch]
        preds    = predict_batch(pipeline, contexts, device)
        for r, pred in zip(batch, preds):
            maes.append(float(np.mean(np.abs(pred - r["target"]))))
    pipeline.model.model.train()
    return float(np.mean(maes)) if maes else float("nan")


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_loss(
    loss_steps, train_losses,
    eval_steps, fast_maes,
    path: Path, job_id: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(loss_steps, train_losses, color="steelblue", lw=0.8, alpha=0.6)
    if len(train_losses) >= 50:
        sm   = np.convolve(train_losses, np.ones(50) / 50, mode="valid")
        sm_x = loss_steps[49:]
        axes[0].plot(sm_x, sm, color="steelblue", lw=1.8, label="Training loss (MA-50)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(eval_steps, fast_maes, color=_GT_COL, lw=1.8, marker="o", ms=3, label="Fast-mode MAE")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Eval MAE (held-out, fast-mode)", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Fine-tuning: chronos-t5-small  ·  fast-only dataset  ·  job {job_id}",
        fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    tqdm.write(f"  Saved: {path.name}", file=sys.stdout)


def plot_eval_samples(
    pipeline,
    eval_rows: list[dict],
    device: torch.device,
    path: Path,
    job_id: str,
    n: int = EVAL_N_SHOW,
) -> None:
    rng   = np.random.default_rng(0)
    picks = rng.choice(len(eval_rows), size=min(n, len(eval_rows)), replace=False)
    picks.sort()

    selected = [eval_rows[i] for i in picks]
    preds    = predict_batch(pipeline, [r["context"] for r in selected], device)

    ctx_x = np.arange(CONTEXT_LENGTH)
    tgt_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"Fine-tuned chronos-t5-small  ·  fast-mode eval  ·  job {job_id}\n"
        f"darkorange = ground truth   crimson = model prediction",
        fontsize=11, fontweight="bold",
    )

    pipeline.model.model.eval()
    for row_idx, (row, pred) in enumerate(zip(selected, preds)):
        ax = axes[row_idx]
        ax.fill_between(ctx_x, row["context"], alpha=0.07, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL,  lw=0.7, label="Input context")
        ax.plot(tgt_x, row["target"],  color=_GT_COL,   lw=1.8, label="Target forecast")
        ax.plot(tgt_x, pred,           color=_PRED_COL, lw=1.8, ls="--", label="Model prediction")
        ax.axvline(CONTEXT_LENGTH - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        mae = float(np.mean(np.abs(pred - row["target"])))
        ax.set_title(f"{row['item_id']}  MAE={mae:.4f}", fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.7)
    axes[-1].set_xlabel("Time step", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    pipeline.model.model.train()
    tqdm.write(f"  Saved: {path.name}", file=sys.stdout)


# ── main training loop ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="nojobid")
    args   = parser.parse_args()
    job_id = args.job_id

    FIGURES_DIR = FINETUNING_ROOT / "figures" / "finetunes" / f"finetune_fast_only_{job_id}"
    CKPT_DIR    = FINETUNING_ROOT / "checkpoints" / f"fast-only-finetune-{job_id}"

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Fine-tuning: chronos-t5-small  |  fast-only dataset")
    print(f"  Job ID    : {job_id}")
    print(f"  Device    : {device}")
    print(f"  Steps     : {N_STEPS:,}  |  batch={BATCH_SIZE}")
    print(f"  LR        : {LR}  |  warmup={WARMUP_STEPS}")
    print(f"  Eval every: {EVAL_EVERY} steps  |  eval_batch={EVAL_BATCH}")
    print(f"  Figures   : {FIGURES_DIR}")
    print(f"  Checkpoint: {CKPT_DIR}")
    print("=" * 60)

    write_config(FIGURES_DIR / "config.txt", job_id)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.bfloat16,
    )
    inner = pipeline.model.model

    current_vocab = inner.get_input_embeddings().weight.shape[0]
    if current_vocab < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
        print(f"  Vocab extended: {current_vocab} → {EXTENDED_VOCAB}")

    # ── load & pre-tokenise data ───────────────────────────────────────────────
    print("\nLoading dataset...")
    train_rows, eval_rows = load_dataset(DATA_PATH)
    print(f"  Train : {len(train_rows):,}  |  Eval : {len(eval_rows):,}")

    tokenised_train = pretokenise(train_rows, pipeline.tokenizer)
    loader = CyclicLoader(tokenised_train, BATCH_SIZE, seed=SEED)

    # ── move model to GPU ──────────────────────────────────────────────────────
    print(f"\nMoving model to {device}...")
    inner = inner.to(device)
    inner.train()

    # ── optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(inner.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=N_STEPS,
    )

    # ── training loop ─────────────────────────────────────────────────────────
    loss_steps, train_losses = [], []
    eval_steps, fast_maes    = [], []

    running_loss = 0.0

    print("\nTraining...")
    pbar = tqdm(
        range(1, N_STEPS + 1),
        total=N_STEPS,
        desc="Training",
        file=sys.stdout,         # bar goes to stdout → same file as tail -f
        dynamic_ncols=True,
        smoothing=0.05,
    )

    for step in pbar:
        batch = loader.next_batch()
        input_ids, attn_mask, labels = assemble_batch(batch, device)

        out  = inner(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % LOG_EVERY == 0:
            avg    = running_loss / LOG_EVERY
            lr_now = scheduler.get_last_lr()[0]
            train_losses.append(avg)
            loss_steps.append(step)
            running_loss = 0.0
            # Embed stats in bar suffix — no tqdm.write() → no extra newlines
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")

        if step % EVAL_EVERY == 0:
            pipeline.model.model = inner
            mae_f = evaluate(pipeline, eval_rows, device, step)
            eval_steps.append(step)
            fast_maes.append(mae_f)
            # Eval milestones are worth a dedicated line
            tqdm.write(
                f"  [eval] step={step:,}  fast MAE={mae_f:.4f}",
                file=sys.stdout,
            )
            inner.train()

    pbar.close()

    # ── save checkpoint ───────────────────────────────────────────────────────
    print(f"\nSaving checkpoint to {CKPT_DIR} ...")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    inner.save_pretrained(str(CKPT_DIR))
    np.save(str(CKPT_DIR / "train_losses.npy"), np.array(train_losses))
    np.save(str(CKPT_DIR / "loss_steps.npy"),   np.array(loss_steps))
    np.save(str(CKPT_DIR / "eval_steps.npy"),   np.array(eval_steps))
    np.save(str(CKPT_DIR / "fast_maes.npy"),    np.array(fast_maes))
    print("  Done.")

    # ── figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    pipeline.model.model = inner
    plot_loss(
        np.array(loss_steps), np.array(train_losses),
        np.array(eval_steps), np.array(fast_maes),
        FIGURES_DIR / "loss.png", job_id,
    )
    plot_eval_samples(
        pipeline, eval_rows, device,
        FIGURES_DIR / "eval_fast_samples.png", job_id,
    )
    print(f"  All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
