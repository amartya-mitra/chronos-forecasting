#!/usr/bin/env python3
"""
Fine-tune chronos-t5-small on sarsim0-trend-test-v2.arrow.  (v4 — decoder-side conditioning)

Dataset split: 70% fast-mode / 30% trend-mode.

Key change from v3: mode token moved from encoder input to decoder input.
  v3: encoder = [mode_token | ctx_tokens | EOS]   (514 tokens)
  v4: encoder = [ctx_tokens | EOS]                (513 tokens, clean)
      decoder_input_ids = [mode_token | tgt_0 .. tgt_62]  (65 tokens, explicit)

Motivation: the mode token at encoder position 0 competes with 512 dense
context tokens for the decoder cross-attention.  Placing it as the first
decoder token means every autoregressive step self-attends to it directly
via causal self-attention — a much stronger mode-conditioning signal.

Two mode tokens:
  4096  (<fast_mode>)  → predict 64-step future forecast
  4097  (<trend_mode>) → predict 64-step trend of the input context

Usage:
  python fine_tune_trend_v4.py [--job-id JOB_ID]

Outputs (all under figures/finetune_trend_{JOB_ID}/):
  config.txt
  loss.png
  eval_fast_samples.png
  eval_trend_samples.png

Checkpoint: checkpoints/trend-finetune-{JOB_ID}/
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

# ── hyperparameters ───────────────────────────────────────────────────────────
FAST_TOKEN_ID  = 4096
TREND_TOKEN_ID = 4097
EXTENDED_VOCAB = 4098

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 64

BATCH_SIZE    = 128
N_STEPS       = 10_000
WARMUP_STEPS  = 500
LR            = 1e-4
GRAD_CLIP     = 1.0
EVAL_EVERY    = 500
EVAL_BATCH    = 32
EVAL_SAMPLES  = 20
EVAL_N_SHOW   = 5
EVAL_POOL_PER_MODE = 100

SEED     = 42
MODEL_ID = "amazon/chronos-t5-small"
DATA_PATH = FINETUNING_ROOT / "data" / "sarsim0-trend-test-v2.arrow"

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
        f"Script          : fine_tune_trend_v4.py",
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
        f"Eval batch size : {EVAL_BATCH}",
        f"Eval pool/mode  : {EVAL_POOL_PER_MODE}",
        f"Seed            : {SEED}",
        "",
        "Key change vs v3 — decoder-side mode conditioning:",
        "  v3: encoder = [mode_token | ctx_tokens | EOS]  (514 tokens)",
        "  v4: encoder = [ctx_tokens | EOS]               (513 tokens, clean)",
        "      decoder_input_ids = [mode_token | tgt_0..tgt_62]  (65 tokens)",
        "",
        "Motivation: encoder position-0 mode token competed with 512 dense",
        "context tokens for decoder cross-attention. Decoder-first token is",
        "attended to by all subsequent decoder steps via causal self-attention,",
        "providing a much harder mode-conditioning signal.",
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


def pretokenise(rows: list[dict], tokenizer) -> list[dict]:
    """
    Pre-tokenise all training rows once on CPU.

    v4 — decoder-side conditioning:
      input_ids    : (513,) int tensor  [context_tokens | EOS]   — no mode prefix
      attn_mask    : (513,) bool tensor
      dec_input_ids: (65,)  int tensor  [mode_token | tgt_0 .. tgt_62]
      labels       : (65,)  int tensor  [tgt_0 .. tgt_63 | EOS]  (-100 for padding)

    assemble_batch just stacks and moves to device — no per-step CPU work.
    """
    N = len(rows)
    print(f"  Pre-tokenising {N:,} training rows on CPU...")

    contexts = torch.tensor(
        np.stack([d["context"] for d in rows]), dtype=torch.float32
    )   # (N, 512)
    targets = torch.tensor(
        np.stack([d["target"]  for d in rows]), dtype=torch.float32
    )   # (N, 64)

    # Tokenise context — no mode prefix on encoder side
    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    # token_ids : (N, 513)  [context tokens + EOS]
    # attn_mask : (N, 513)  bool

    # Tokenise targets
    label_ids, label_mask = tokenizer.label_input_transform(targets, scale)
    # label_ids : (N, 65)   [tgt_0 .. tgt_63 | EOS]
    labels = label_ids.masked_fill(~label_mask.bool(), -100)   # (N, 65)

    # Build decoder_input_ids: [mode_token | tgt_0 .. tgt_62]  (right-shift of labels)
    mode_col = torch.tensor(
        [FAST_TOKEN_ID if d["mode"] == "fast" else TREND_TOKEN_ID for d in rows],
        dtype=token_ids.dtype,
    ).unsqueeze(1)   # (N, 1)
    dec_input_ids = torch.cat([mode_col, label_ids[:, :-1]], dim=1)  # (N, 65)

    tokenised = []
    for i, d in enumerate(rows):
        tokenised.append({
            "item_id":      d["item_id"],
            "mode":         d["mode"],
            "context":      d["context"],       # kept for eval plots (numpy, CPU)
            "target":       d["target"],        # kept for eval plots (numpy, CPU)
            "input_ids":    token_ids[i],       # (513,) CPU tensor — encoder, no mode
            "attn_mask":    attn_mask[i],       # (513,) CPU tensor
            "dec_input_ids": dec_input_ids[i],  # (65,)  CPU tensor — decoder start
            "labels":       labels[i],          # (65,)  CPU tensor
        })

    print("  Pre-tokenisation complete.")
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
        batch = [self.rows[i] for i in self._indices[self._pos:self._pos + self.batch_size]]
        self._pos += self.batch_size
        return batch


# ── training step assembly ────────────────────────────────────────────────────

def assemble_batch(
    batch: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stack pre-assembled CPU tensors and issue a single .to(device) per tensor.
    No cat, no reshape — all CPU work was done once in pretokenise().
    """
    input_ids     = torch.stack([d["input_ids"]     for d in batch]).to(device)  # (B, 513)
    attn_mask     = torch.stack([d["attn_mask"]     for d in batch]).to(device)  # (B, 513)
    dec_input_ids = torch.stack([d["dec_input_ids"] for d in batch]).to(device)  # (B, 65)
    labels        = torch.stack([d["labels"]        for d in batch]).to(device)  # (B, 65)
    return input_ids, attn_mask, dec_input_ids, labels


# ── batched eval inference ────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch_with_mode(
    pipeline,
    contexts: list[np.ndarray],
    mode_token_id: int,
    device: torch.device,
    prediction_length: int = PREDICTION_LENGTH,
    num_samples: int = EVAL_SAMPLES,
) -> list[np.ndarray]:
    """
    Batched inference over a list of contexts sharing the same mode token.
    One model.generate() call produces B * num_samples sequences.
    """
    B   = len(contexts)
    ctx = torch.tensor(np.stack(contexts), dtype=torch.float32)   # (B, T) — CPU

    # Tokenise on CPU (tokenizer stays on CPU throughout)
    token_ids, attn_mask, scale = pipeline.tokenizer.context_input_transform(ctx)

    # v4: no mode prefix on encoder — clean context only
    input_ids_dev = token_ids.to(device)   # (B, 513)
    attn_mask_dev = attn_mask.to(device)   # (B, 513)

    cfg = pipeline.model.config
    raw = pipeline.model.model.generate(
        input_ids=input_ids_dev,
        attention_mask=attn_mask_dev,
        generation_config=GenerationConfig(
            min_new_tokens=prediction_length,
            max_new_tokens=prediction_length,
            do_sample=True,
            num_return_sequences=num_samples,
            decoder_start_token_id=mode_token_id,   # mode conditions decoder start
            eos_token_id=cfg.eos_token_id,
            pad_token_id=cfg.pad_token_id,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        ),
    )
    raw = raw[:, 1:]                                         # drop decoder start token
    raw = raw.reshape(B, num_samples, prediction_length)     # (B, S, T)

    # Transfer back to CPU for output_transform (required — tokenizer is CPU-only)
    preds   = pipeline.tokenizer.output_transform(raw.cpu(), scale)   # (B, S, T)
    medians = preds.median(dim=1).values.numpy()                      # (B, T)
    return [medians[i] for i in range(B)]


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pipeline,
    eval_rows: list[dict],
    device: torch.device,
    step: int,
) -> tuple[float, float]:
    """Batched eval — ~7 GPU generate() calls instead of 200."""
    pipeline.model.model.eval()

    fast_rows  = [r for r in eval_rows if r["mode"] == "fast"]
    trend_rows = [r for r in eval_rows if r["mode"] == "trend"]

    def run_mode(rows, mode_token_id, label: str):
        maes = []
        bar  = tqdm(
            range(0, len(rows), EVAL_BATCH),
            desc=f"    eval {label} step={step:,}",
            file=sys.stdout,
            leave=False,
            dynamic_ncols=True,
        )
        for i in bar:
            batch    = rows[i:i + EVAL_BATCH]
            contexts = [r["context"] for r in batch]
            preds    = predict_batch_with_mode(
                pipeline, contexts, mode_token_id, device
            )
            for r, pred in zip(batch, preds):
                maes.append(float(np.mean(np.abs(pred - r["target"]))))
        return float(np.mean(maes)) if maes else float("nan")

    mean_fast  = run_mode(fast_rows,  FAST_TOKEN_ID,  "fast ")
    mean_trend = run_mode(trend_rows, TREND_TOKEN_ID, "trend")

    pipeline.model.model.train()
    return mean_fast, mean_trend


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_loss(
    loss_steps, train_losses,
    eval_steps, fast_maes, trend_maes,
    path: Path, job_id: str,
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
        f"Fine-tuning v4: chronos-t5-small  ·  70% fast / 30% trend  ·  decoder-mode conditioning  ·  job {job_id}",
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

    selected = [rows[i] for i in picks]
    preds    = predict_batch_with_mode(
        pipeline, [r["context"] for r in selected], mode_token_id, device
    )

    ctx_x = np.arange(CONTEXT_LENGTH)
    tgt_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"Fine-tuned chronos-t5-small (v4)  ·  {mode}-mode eval  ·  job {job_id}\n"
        f"{colour} = ground truth   crimson = model prediction",
        fontsize=11, fontweight="bold",
    )

    pipeline.model.model.eval()
    for row_idx, (row, pred) in enumerate(zip(selected, preds)):
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
    parser.add_argument("--job-id", default="nojobid")
    args   = parser.parse_args()
    job_id = args.job_id

    FIGURES_DIR = FINETUNING_ROOT / "figures"     / f"finetune_trend_v4_{job_id}"
    CKPT_DIR    = FINETUNING_ROOT / "checkpoints" / f"trend-finetune-v4-{job_id}"

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Fine-tuning v4: chronos-t5-small  |  trend-test-v2  |  decoder-mode conditioning")
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
    n_fast_eval  = sum(1 for r in eval_rows if r["mode"] == "fast")
    n_trend_eval = sum(1 for r in eval_rows if r["mode"] == "trend")
    print(f"  Train : {len(train_rows):,}  |  Eval : {len(eval_rows):,}  "
          f"(fast={n_fast_eval}, trend={n_trend_eval})")

    tokenised_train = pretokenise(train_rows, pipeline.tokenizer)
    loader = CyclicLoader(tokenised_train, BATCH_SIZE, seed=SEED)

    # ── move model to GPU ──────────────────────────────────────────────────────
    print(f"\nMoving model to {device}...")
    inner = inner.to(device)
    inner.train()
    # Tokenizer stays on CPU for context_input_transform / output_transform

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

    running_loss = 0.0
    LOG_EVERY    = 50

    print("\nTraining...")
    pbar = tqdm(
        range(1, N_STEPS + 1),
        total=N_STEPS,
        desc="Training",
        file=sys.stdout,
        dynamic_ncols=True,
        smoothing=0.05,
        miniters=LOG_EVERY,   # render bar in sync with step logs (every 50 steps)
        mininterval=0.0,
    )

    for step in pbar:
        batch = loader.next_batch()
        input_ids, attn_mask, dec_input_ids, labels = assemble_batch(batch, device)

        out  = inner(
            input_ids=input_ids,
            attention_mask=attn_mask,
            decoder_input_ids=dec_input_ids,
            labels=labels,
        )
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
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")
            tqdm.write(
                f"  step {step:>6,}/{N_STEPS:,}  loss={avg:.4f}  lr={lr_now:.2e}"
            )

        if step % EVAL_EVERY == 0:
            pipeline.model.model = inner
            mae_f, mae_t = evaluate(pipeline, eval_rows, device, step)
            eval_steps.append(step)
            fast_maes.append(mae_f)
            trend_maes.append(mae_t)
            tqdm.write(
                f"  [eval] step={step:,}  fast MAE={mae_f:.4f}  trend MAE={mae_t:.4f}"
            )
            inner.train()

    pbar.close()

    # ── save model ────────────────────────────────────────────────────────────
    print(f"\nSaving checkpoint to {CKPT_DIR} ...")
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
        path=FIGURES_DIR / "eval_fast_samples.png", job_id=job_id,
    )
    plot_eval_samples(
        pipeline, eval_rows, mode="trend", device=device,
        path=FIGURES_DIR / "eval_trend_samples.png", job_id=job_id,
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
