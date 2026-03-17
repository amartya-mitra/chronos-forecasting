#!/usr/bin/env python3
"""
Fine-tune chronos-t5-small on sarsim0-trend-test-v2.arrow.

Dataset split: 70% fast-mode / 30% trend-mode.

Two prefix tokens prepended to the encoder input distinguish the task:
  4096  (<fast_mode>)  → predict 64-step future forecast
  4097  (<trend_mode>) → predict 64-step trend extracted from the input context

Training:
  10 000 steps · batch 32 · AdamW lr=1e-4 · warmup 500 steps · cosine LR decay
  Eval on held-out split every 250 steps

Usage:
  python fine_tune_trend_v2.py [--job-id JOB_ID]

Outputs (all under figures/finetune_trend_{JOB_ID}/):
  config.txt                  experiment configuration record
  loss.png                    training loss + eval MAE curves
  eval_fast_samples.png       5 held-out fast-mode predictions
  eval_trend_samples.png      5 held-out trend-mode predictions

Checkpoint saved to:
  checkpoints/trend-finetune-{JOB_ID}/
"""

import sys
import argparse
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import math
import random
import numpy as np
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup, GenerationConfig

from chronos import ChronosPipeline

# ── hyperparameters ───────────────────────────────────────────────────────────
FAST_TOKEN_ID  = 4096
TREND_TOKEN_ID = 4097
EXTENDED_VOCAB = 4098

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 64

BATCH_SIZE    = 32
N_STEPS       = 10_000
WARMUP_STEPS  = 500
LR            = 1e-4
GRAD_CLIP     = 1.0
EVAL_EVERY    = 250
EVAL_SAMPLES  = 20
EVAL_N_SHOW   = 5
EVAL_POOL_PER_MODE = 100

SEED          = 42
MODEL_ID      = "amazon/chronos-t5-small"
DATA_PATH     = FINETUNING_ROOT / "data" / "sarsim0-trend-test-v2.arrow"

# ── colours ───────────────────────────────────────────────────────────────────
_CTX_COL   = "steelblue"
_GT_COL    = "darkorange"
_PRED_COL  = "crimson"
_TREND_COL = "seagreen"


# ── reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── config record ─────────────────────────────────────────────────────────────

def write_config(path: Path, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Experiment Configuration",
        "=" * 40,
        f"Job ID          : {job_id}",
        f"Model           : {MODEL_ID}",
        f"Dataset         : sarsim0-trend-test-v2.arrow",
        f"Dataset split   : 70% fast-mode / 30% trend-mode",
        f"Fast token ID   : {FAST_TOKEN_ID}",
        f"Trend token ID  : {TREND_TOKEN_ID}",
        f"Extended vocab  : {EXTENDED_VOCAB}",
        f"Context length  : {CONTEXT_LENGTH}",
        f"Pred length     : {PREDICTION_LENGTH}",
        f"Batch size      : {BATCH_SIZE}",
        f"Steps           : {N_STEPS:,}",
        f"Warmup steps    : {WARMUP_STEPS}",
        f"Learning rate   : {LR}",
        f"Grad clip       : {GRAD_CLIP}",
        f"Eval every      : {EVAL_EVERY} steps",
        f"Eval pool/mode  : {EVAL_POOL_PER_MODE}",
        f"Seed            : {SEED}",
        "",
        "Motivation: prior run (70% trend / 30% fast) showed the model",
        "predicting trend even in fast-mode. Rebalancing to 70% fast / 30%",
        "trend to bias the model toward retaining its base forecast behaviour.",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"  Config written: {path.name}")


# ── data ──────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> tuple[list[dict], list[dict]]:
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()

    fast_rows, trend_rows = [], []
    for item_id, mode, ctx, tgt in zip(
        table.column("item_id").to_pylist(),
        table.column("mode").to_pylist(),
        table.column("context").to_pylist(),
        table.column("target").to_pylist(),
    ):
        row = {
            "item_id": item_id,
            "mode":    mode,
            "context": np.array(ctx, dtype=np.float32),
            "target":  np.array(tgt, dtype=np.float32),
        }
        if mode == "fast":
            fast_rows.append(row)
        else:
            trend_rows.append(row)

    rng = np.random.default_rng(SEED)
    rng.shuffle(fast_rows)
    rng.shuffle(trend_rows)

    eval_rows  = fast_rows[:EVAL_POOL_PER_MODE] + trend_rows[:EVAL_POOL_PER_MODE]
    train_rows = fast_rows[EVAL_POOL_PER_MODE:]  + trend_rows[EVAL_POOL_PER_MODE:]
    rng.shuffle(train_rows)

    return train_rows, eval_rows


class CyclicLoader:
    """Infinite cyclic iterator over a shuffled list of dicts."""
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
        batch = [self.rows[i] for i in self._indices[self._pos:self._pos + self.batch_size]]
        self._pos += self.batch_size
        return batch


# ── tokenisation helpers ──────────────────────────────────────────────────────

def tokenise_batch(
    batch: list[dict],
    tokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts = torch.tensor(
        np.stack([d["context"] for d in batch]), dtype=torch.float32
    )
    targets = torch.tensor(
        np.stack([d["target"] for d in batch]), dtype=torch.float32
    )

    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)

    mode_ids = torch.tensor(
        [[FAST_TOKEN_ID if d["mode"] == "fast" else TREND_TOKEN_ID] for d in batch],
        dtype=token_ids.dtype,
    )
    mode_mask = torch.ones(len(batch), 1, dtype=torch.bool)

    input_ids = torch.cat([mode_ids,   token_ids], dim=1).to(device)
    attn_mask = torch.cat([mode_mask, attn_mask], dim=1).to(device)

    label_ids, label_mask = tokenizer.label_input_transform(targets, scale)
    labels = label_ids.masked_fill(~label_mask.bool(), -100).to(device)

    return input_ids, attn_mask, labels


# ── inference for evaluation ──────────────────────────────────────────────────

@torch.no_grad()
def predict_with_mode(
    pipeline,
    context_np: np.ndarray,
    mode_token_id: int,
    device: torch.device,
    prediction_length: int = PREDICTION_LENGTH,
    num_samples: int = EVAL_SAMPLES,
) -> np.ndarray:
    ctx = torch.tensor(context_np, dtype=torch.float32).unsqueeze(0)

    token_ids, attn_mask, scale = pipeline.tokenizer.context_input_transform(ctx)

    mode_prefix = torch.full((1, 1), mode_token_id, dtype=token_ids.dtype)
    mode_mask   = torch.ones(1, 1, dtype=torch.bool)

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
    raw = raw[:, 1:]
    raw = raw.reshape(1, num_samples, prediction_length)

    preds = pipeline.tokenizer.output_transform(raw.cpu(), scale)
    return preds[0].median(dim=0).values.numpy()


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pipeline,
    eval_rows: list[dict],
    device: torch.device,
) -> tuple[float, float]:
    pipeline.model.model.eval()

    fast_maes, trend_maes = [], []
    for row in eval_rows:
        mode_id = FAST_TOKEN_ID if row["mode"] == "fast" else TREND_TOKEN_ID
        pred    = predict_with_mode(pipeline, row["context"], mode_id, device)
        mae     = float(np.mean(np.abs(pred - row["target"])))
        if row["mode"] == "fast":
            fast_maes.append(mae)
        else:
            trend_maes.append(mae)

    pipeline.model.model.train()
    mean_fast  = float(np.mean(fast_maes))  if fast_maes  else float("nan")
    mean_trend = float(np.mean(trend_maes)) if trend_maes else float("nan")
    return mean_fast, mean_trend


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_loss(
    loss_steps: list[int],
    train_losses: list[float],
    eval_steps: list[int],
    fast_maes: list[float],
    trend_maes: list[float],
    path: Path,
    job_id: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(loss_steps, train_losses, color="steelblue", lw=0.8, alpha=0.6)
    if len(train_losses) >= 50:
        sm   = np.convolve(train_losses, np.ones(50)/50, mode="valid")
        sm_x = loss_steps[49:]
        axes[0].plot(sm_x, sm, color="steelblue", lw=1.8, label="Training loss (MA-50)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(eval_steps, fast_maes,  color=_GT_COL,    lw=1.8, marker="o", ms=3, label="Fast-mode MAE")
    axes[1].plot(eval_steps, trend_maes, color=_TREND_COL, lw=1.8, marker="s", ms=3, label="Trend-mode MAE")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Eval MAE (held-out)", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Fine-tuning: chronos-t5-small  ·  trend-test-v2 (70% fast / 30% trend)  ·  job {job_id}",
        fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_eval_samples(
    pipeline,
    eval_rows: list[dict],
    mode: str,
    device: torch.device,
    path: Path,
    job_id: str,
    n: int = EVAL_N_SHOW,
) -> None:
    rows  = [r for r in eval_rows if r["mode"] == mode]
    rng   = np.random.default_rng(0)
    picks = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    picks.sort()

    mode_token_id = FAST_TOKEN_ID if mode == "fast" else TREND_TOKEN_ID
    colour        = _GT_COL if mode == "fast" else _TREND_COL
    label_gt      = "Target forecast" if mode == "fast" else "Target trend"

    ctx_x = np.arange(CONTEXT_LENGTH)
    tgt_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"Fine-tuned chronos-t5-small  ·  {mode}-mode evaluation  ·  job {job_id}\n"
        f"{colour} = ground truth   crimson = model prediction",
        fontsize=11, fontweight="bold",
    )

    pipeline.model.model.eval()
    for row_idx, pick in enumerate(picks):
        row  = rows[pick]
        pred = predict_with_mode(pipeline, row["context"], mode_token_id, device)

        ax = axes[row_idx]
        ax.fill_between(ctx_x, row["context"], alpha=0.07, color=_CTX_COL)
        ax.plot(ctx_x, row["context"],  color=_CTX_COL,  lw=0.7, label="Input context")
        ax.plot(tgt_x, row["target"],   color=colour,    lw=1.8, label=label_gt)
        ax.plot(tgt_x, pred,            color=_PRED_COL, lw=1.8, ls="--", label="Model prediction")
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
    print(f"  Saved: {path.name}")


# ── main training loop ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="nojobid", help="SLURM job ID for output naming")
    args = parser.parse_args()
    job_id = args.job_id

    FIGURES_DIR = FINETUNING_ROOT / "figures" / f"finetune_trend_{job_id}"
    CKPT_DIR    = FINETUNING_ROOT / "checkpoints" / f"trend-finetune-{job_id}"

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Fine-tuning: chronos-t5-small on trend-test-v2 dataset")
    print(f"  Job ID    : {job_id}")
    print(f"  Device    : {device}")
    print(f"  Steps     : {N_STEPS:,}")
    print(f"  Batch     : {BATCH_SIZE}")
    print(f"  LR        : {LR}")
    print(f"  Dataset   : 70% fast / 30% trend")
    print(f"  Figures   : {FIGURES_DIR}")
    print(f"  Checkpoint: {CKPT_DIR}")
    print("=" * 60)

    # Write experiment config immediately
    write_config(FIGURES_DIR / "config.txt", job_id)

    # ── load data ─────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_rows, eval_rows = load_dataset(DATA_PATH)
    n_fast_eval  = sum(1 for r in eval_rows  if r["mode"] == "fast")
    n_trend_eval = sum(1 for r in eval_rows  if r["mode"] == "trend")
    print(f"  Train rows : {len(train_rows):,}")
    print(f"  Eval rows  : {len(eval_rows):,}  "
          f"(fast={n_fast_eval}, trend={n_trend_eval})")

    loader = CyclicLoader(train_rows, BATCH_SIZE, seed=SEED)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.bfloat16,
    )
    inner = pipeline.model.model

    current_vocab = inner.get_input_embeddings().weight.shape[0]
    print(f"  Current vocab size : {current_vocab}")
    if current_vocab < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
        print(f"  Vocab extended to  : {EXTENDED_VOCAB}")

    inner = inner.to(device)
    inner.train()
    # Tokenizer stays on CPU — contexts are tokenised as CPU tensors,
    # only input_ids / labels are moved to device after tokenisation.

    # ── optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(inner.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=N_STEPS,
    )

    # ── training loop ─────────────────────────────────────────────────────────
    loss_steps, train_losses = [], []
    eval_steps, fast_maes, trend_maes = [], [], []

    print("\nTraining...")
    running_loss = 0.0
    LOG_EVERY    = 50

    for step in range(1, N_STEPS + 1):
        batch = loader.next_batch()

        input_ids, attn_mask, labels = tokenise_batch(
            batch, pipeline.tokenizer, device
        )

        out  = inner(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            train_losses.append(avg)
            loss_steps.append(step)
            running_loss = 0.0
            print(f"  step {step:>6,} / {N_STEPS:,}  "
                  f"loss={avg:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if step % EVAL_EVERY == 0:
            pipeline.model.model = inner
            mae_f, mae_t = evaluate(pipeline, eval_rows, device)
            eval_steps.append(step)
            fast_maes.append(mae_f)
            trend_maes.append(mae_t)
            print(f"    [eval] step={step:,}  "
                  f"fast MAE={mae_f:.4f}  trend MAE={mae_t:.4f}")
            inner.train()

    # ── save model ────────────────────────────────────────────────────────────
    print(f"\nSaving model to {CKPT_DIR} ...")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    inner.save_pretrained(str(CKPT_DIR))
    np.save(str(CKPT_DIR / "train_losses.npy"),  np.array(train_losses))
    np.save(str(CKPT_DIR / "loss_steps.npy"),    np.array(loss_steps))
    np.save(str(CKPT_DIR / "eval_steps.npy"),    np.array(eval_steps))
    np.save(str(CKPT_DIR / "fast_maes.npy"),     np.array(fast_maes))
    np.save(str(CKPT_DIR / "trend_maes.npy"),    np.array(trend_maes))
    print("  Done.")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    inner.eval()
    pipeline.model.model = inner

    plot_loss(
        loss_steps, train_losses,
        eval_steps, fast_maes, trend_maes,
        path=FIGURES_DIR / "loss.png",
        job_id=job_id,
    )
    plot_eval_samples(
        pipeline, eval_rows, mode="fast",  device=device,
        path=FIGURES_DIR / "eval_fast_samples.png",
        job_id=job_id,
    )
    plot_eval_samples(
        pipeline, eval_rows, mode="trend", device=device,
        path=FIGURES_DIR / "eval_trend_samples.png",
        job_id=job_id,
    )

    print(f"\nDone.  Figures in {FIGURES_DIR}/")
    print(f"       Checkpoint in {CKPT_DIR}/")

    print("\n" + "=" * 60)
    print("Final eval metrics:")
    print(f"  Fast-mode  MAE : {fast_maes[-1]:.4f}")
    print(f"  Trend-mode MAE : {trend_maes[-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
