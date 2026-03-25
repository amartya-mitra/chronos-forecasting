#!/usr/bin/env python3
"""
Fine-tune chronos-t5-small on sarsim0-seasonal-only.arrow.

All 6,000 rows use the SEASONAL mode token (4098).
  encoder           = [ctx_tokens | EOS]                  (513 tokens)
  decoder_input_ids = [SEASONAL_TOKEN | tgt_0 .. tgt_62]  (65 tokens)

Vocab is extended from 4096 → 4099 (adds fast=4096, trend=4097, seasonal=4098).
Embedding init: EOS + noise (same as v5).

Outputs (figures/finetune_seasonal_only_{JOB_ID}/):
  config.txt  loss.png  eval_samples.png

Checkpoint: checkpoints/finetune-seasonal-only-{JOB_ID}/
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
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, GenerationConfig

from chronos import ChronosPipeline

# ── constants ─────────────────────────────────────────────────────────────────
SEASONAL_TOKEN_ID = 4098
EXTENDED_VOCAB    = 4099   # 4096 + fast(4096) + trend(4097) + seasonal(4098)

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 64

BATCH_SIZE   = 128
N_STEPS      = 10_000
WARMUP_STEPS = 500
LR           = 1e-4
GRAD_CLIP    = 1.0
EVAL_EVERY   = 500
EVAL_BATCH   = 32
EVAL_SAMPLES = 20
EVAL_N_SHOW  = 5
EVAL_POOL    = 100

EMBED_INIT_STD_MULTIPLIER = 2.0
SEED     = 42
MODEL_ID = "amazon/chronos-t5-small"
DATA_PATH = FINETUNING_ROOT / "data" / "sarsim0-seasonal-only.arrow"

_CTX_COL      = "steelblue"
_SEASONAL_COL = "mediumpurple"
_PRED_COL     = "crimson"


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def init_mode_embeddings(inner) -> None:
    w = inner.get_input_embeddings().weight
    existing_std = w[:4096].std().item()
    noise_std    = EMBED_INIT_STD_MULTIPLIER * existing_std
    with torch.no_grad():
        base = w[1].clone()   # EOS token (ID=1)
        gen  = torch.Generator(); gen.manual_seed(SEED)
        for tok_id in (4096, 4097, SEASONAL_TOKEN_ID):
            w[tok_id] = base + noise_std * torch.randn(w.shape[1], generator=gen, dtype=w.dtype)
    s_emb = w[SEASONAL_TOKEN_ID].float()
    e_emb = w[1].float()
    cos = F.cosine_similarity(s_emb.unsqueeze(0), e_emb.unsqueeze(0)).item()
    print(f"  Embed init : eos_plus_noise,  noise_std={noise_std:.5f}")
    print(f"  Cosine sim : seasonal↔EOS={cos:.3f}")


def write_config(path: Path, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "Experiment Configuration", "=" * 40,
        f"Job ID   : {job_id}",
        f"Script   : fine_tune_seasonal_only.py",
        f"Model    : {MODEL_ID}",
        f"Dataset  : sarsim0-seasonal-only.arrow  (6k rows, all seasonal-mode)",
        f"Token ID : SEASONAL_TOKEN_ID={SEASONAL_TOKEN_ID}",
        f"Vocab    : extended to {EXTENDED_VOCAB}",
        f"Steps    : {N_STEPS:,}  |  batch={BATCH_SIZE}  |  LR={LR}",
        "",
        "Architecture: decoder-side mode conditioning (v5-style)",
        "  encoder           = [ctx_tokens | EOS]                 (513 tokens)",
        "  decoder_input_ids = [SEASONAL_TOKEN | tgt_0..tgt_62]  (65 tokens)",
        f"Embed init: eos_plus_noise  (multiplier={EMBED_INIT_STD_MULTIPLIER})",
    ]) + "\n")
    print(f"  Config written: {path.name}")


# ── data ──────────────────────────────────────────────────────────────────────

def load_dataset(path: Path):
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()
    rows = []
    for item_id, ctx, tgt in zip(
        table.column("item_id").to_pylist(),
        table.column("context").to_pylist(),
        table.column("target").to_pylist(),
    ):
        rows.append({
            "item_id": item_id,
            "context": np.array(ctx, dtype=np.float32),
            "target":  np.array(tgt, dtype=np.float32),
        })
    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[EVAL_POOL:], rows[:EVAL_POOL]


def pretokenise(rows, tokenizer):
    N = len(rows)
    print(f"  Pre-tokenising {N:,} rows on CPU...")
    contexts = torch.tensor(np.stack([d["context"] for d in rows]), dtype=torch.float32)
    targets  = torch.tensor(np.stack([d["target"]  for d in rows]), dtype=torch.float32)
    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    label_ids, label_mask       = tokenizer.label_input_transform(targets, scale)
    labels        = label_ids.masked_fill(~label_mask.bool(), -100)
    seas_col      = torch.full((N, 1), SEASONAL_TOKEN_ID, dtype=token_ids.dtype)
    dec_input_ids = torch.cat([seas_col, label_ids[:, :-1]], dim=1)   # (N, 65)
    out = []
    for i, d in enumerate(rows):
        out.append({
            "item_id":       d["item_id"],
            "context":       d["context"],
            "target":        d["target"],
            "input_ids":     token_ids[i],
            "attn_mask":     attn_mask[i],
            "dec_input_ids": dec_input_ids[i],
            "labels":        labels[i],
        })
    print("  Pre-tokenisation complete.")
    return out


class CyclicLoader:
    def __init__(self, rows, batch_size, seed=0):
        self.rows, self.batch_size = rows, batch_size
        self.rng = np.random.default_rng(seed)
        self._idx = np.arange(len(rows)); self._pos = len(rows)

    def next_batch(self):
        if self._pos + self.batch_size > len(self._idx):
            self.rng.shuffle(self._idx); self._pos = 0
        b = [self.rows[i] for i in self._idx[self._pos:self._pos+self.batch_size]]
        self._pos += self.batch_size
        return b


def assemble_batch(batch, device):
    return (
        torch.stack([d["input_ids"]     for d in batch]).to(device),
        torch.stack([d["attn_mask"]     for d in batch]).to(device),
        torch.stack([d["dec_input_ids"] for d in batch]).to(device),
        torch.stack([d["labels"]        for d in batch]).to(device),
    )


# ── eval & plots ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(pipeline, contexts, device):
    ctx = torch.tensor(np.stack(contexts), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)
    cfg = pipeline.model.config
    raw = pipeline.model.model.generate(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        generation_config=GenerationConfig(
            min_new_tokens=PREDICTION_LENGTH, max_new_tokens=PREDICTION_LENGTH,
            do_sample=True, num_return_sequences=EVAL_SAMPLES,
            decoder_start_token_id=SEASONAL_TOKEN_ID,
            eos_token_id=cfg.eos_token_id, pad_token_id=cfg.pad_token_id,
            temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p,
        ),
    )
    B   = len(contexts)
    raw = raw[:, 1:].reshape(B, EVAL_SAMPLES, PREDICTION_LENGTH)
    return pipeline.tokenizer.output_transform(raw.cpu(), scale).median(dim=1).values.numpy()


def evaluate(pipeline, eval_rows, device, step):
    pipeline.model.model.eval()
    maes = []
    bar  = tqdm(range(0, len(eval_rows), EVAL_BATCH),
                desc=f"    eval step={step:,}", file=sys.stdout, leave=False, dynamic_ncols=True)
    for i in bar:
        batch = eval_rows[i:i+EVAL_BATCH]
        preds = predict_batch(pipeline, [r["context"] for r in batch], device)
        for r, pred in zip(batch, preds):
            maes.append(float(np.mean(np.abs(pred - r["target"]))))
    pipeline.model.model.train()
    return float(np.mean(maes))


def plot_loss(loss_steps, losses, eval_steps, maes, path, job_id):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(loss_steps, losses, color="steelblue", lw=0.8, alpha=0.6)
    if len(losses) >= 50:
        sm = np.convolve(losses, np.ones(50)/50, mode="valid")
        axes[0].plot(loss_steps[49:], sm, color="steelblue", lw=1.8, label="Loss (MA-50)")
    axes[0].set_title("Training Loss (CE)", fontweight="bold")
    axes[0].set_xlabel("Step"); axes[0].legend(fontsize=8)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[1].plot(eval_steps, maes, color=_SEASONAL_COL, lw=1.8, marker="s", ms=3, label="Seasonal MAE")
    axes[1].set_title("Eval MAE — Seasonal Mode", fontweight="bold")
    axes[1].set_xlabel("Step"); axes[1].legend(fontsize=8)
    axes[1].spines[["top","right"]].set_visible(False)
    fig.suptitle(f"Fine-tuning seasonal-only: chronos-t5-small  ·  job {job_id}", fontweight="bold")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_eval_samples(pipeline, eval_rows, device, path, job_id, n=EVAL_N_SHOW):
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(eval_rows), size=min(n, len(eval_rows)), replace=False))
    sel   = [eval_rows[i] for i in picks]
    pipeline.model.model.eval()
    preds = predict_batch(pipeline, [r["context"] for r in sel], device)
    pipeline.model.model.train()

    ctx_x = np.arange(CONTEXT_LENGTH)
    tgt_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)
    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"fine_tune_seasonal_only  ·  job {job_id}\n"
        "mediumpurple = target seasonal   crimson = model prediction",
        fontsize=11, fontweight="bold",
    )
    for idx, (row, pred) in enumerate(zip(sel, preds)):
        ax = axes[idx]
        ax.fill_between(ctx_x, row["context"], alpha=0.07, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL,      lw=0.7, label="Input context")
        ax.plot(tgt_x, row["target"],  color=_SEASONAL_COL, lw=1.8, label="Target seasonal")
        ax.plot(tgt_x, pred,           color=_PRED_COL,     lw=1.8, ls="--", label="Model prediction")
        ax.axvline(CONTEXT_LENGTH - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        mae = float(np.mean(np.abs(pred - row["target"])))
        ax.set_title(f"{row['item_id']}  MAE={mae:.4f}", fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7); ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.7)
    axes[-1].set_xlabel("Time step", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="nojobid")
    job_id = parser.parse_args().job_id

    FIGURES_DIR = FINETUNING_ROOT / "figures"     / f"finetune_seasonal_only_{job_id}"
    CKPT_DIR    = FINETUNING_ROOT / "checkpoints" / f"finetune-seasonal-only-{job_id}"

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Fine-tuning seasonal-only: chronos-t5-small")
    print(f"  Job ID    : {job_id}")
    print(f"  Device    : {device}")
    print(f"  Steps     : {N_STEPS:,}  |  batch={BATCH_SIZE}  |  LR={LR}")
    print(f"  Token ID  : SEASONAL={SEASONAL_TOKEN_ID}")
    print(f"  Figures   : {FIGURES_DIR}")
    print("=" * 60)
    write_config(FIGURES_DIR / "config.txt", job_id)

    print(f"\nLoading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    inner    = pipeline.model.model
    if inner.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
        print(f"  Vocab extended → {EXTENDED_VOCAB}")
    init_mode_embeddings(inner)

    print("\nLoading dataset...")
    train_rows, eval_rows = load_dataset(DATA_PATH)
    print(f"  Train : {len(train_rows):,}  |  Eval : {len(eval_rows):,}")
    tokenised = pretokenise(train_rows, pipeline.tokenizer)
    loader    = CyclicLoader(tokenised, BATCH_SIZE, seed=SEED)

    print(f"\nMoving model to {device}...")
    inner = inner.to(device); inner.train()

    optimizer = torch.optim.AdamW(inner.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, N_STEPS)

    loss_steps, losses = [], []
    eval_steps, maes   = [], []
    running = 0.0; LOG_EVERY = 50

    print("\nTraining...")
    pbar = tqdm(range(1, N_STEPS+1), total=N_STEPS, desc="Training",
                file=sys.stdout, dynamic_ncols=True, smoothing=0.05,
                miniters=LOG_EVERY, mininterval=0.0)

    for step in pbar:
        ids, mask, dec, lbl = assemble_batch(loader.next_batch(), device)
        out  = inner(input_ids=ids, attention_mask=mask, decoder_input_ids=dec, labels=lbl)
        loss = out.loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), GRAD_CLIP)
        optimizer.step(); scheduler.step()
        running += loss.item()

        if step % LOG_EVERY == 0:
            avg = running / LOG_EVERY; running = 0.0
            lr_now = scheduler.get_last_lr()[0]
            loss_steps.append(step); losses.append(avg)
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")
            tqdm.write(f"  step {step:>6,}/{N_STEPS:,}  loss={avg:.4f}  lr={lr_now:.2e}")

        if step % EVAL_EVERY == 0:
            pipeline.model.model = inner
            mae = evaluate(pipeline, eval_rows, device, step)
            eval_steps.append(step); maes.append(mae)
            tqdm.write(f"  [eval] step={step:,}  seasonal MAE={mae:.4f}")
            inner.train()

    pbar.close()

    print(f"\nSaving checkpoint to {CKPT_DIR} ...")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    inner.save_pretrained(str(CKPT_DIR))
    for name, arr in [("loss_steps", loss_steps), ("losses", losses),
                      ("eval_steps", eval_steps), ("maes", maes)]:
        np.save(str(CKPT_DIR / f"{name}.npy"), np.array(arr))

    print("\nGenerating plots...")
    inner.eval(); pipeline.model.model = inner
    plot_loss(loss_steps, losses, eval_steps, maes,
              path=FIGURES_DIR / "loss.png", job_id=job_id)
    plot_eval_samples(pipeline, eval_rows, device,
                      path=FIGURES_DIR / "eval_samples.png", job_id=job_id)

    print(f"\nDone.  Figures in {FIGURES_DIR}/")
    print(f"Final seasonal MAE : {maes[-1]:.4f}")


if __name__ == "__main__":
    main()
