#!/usr/bin/env python3
"""
Fine-tune chronos-t5-small on sarsim0-contrastive-v1.arrow.  (v6 — contrastive loss)

Dataset split: 30% fast-mode / 70% trend-mode.
Each row stores both correct_target and wrong_target — no on-the-fly computation.

Architecture (same as v4/v5):
  encoder           = [ctx_tokens | EOS]                  (513 tokens)
  decoder_input_ids = [mode_token | tgt_0 .. tgt_62]      (65 tokens)

Key change from v5: adds a soft-prediction contrastive loss on top of CE.

Contrastive loss:
  For each sample, compute a differentiable soft prediction by taking the
  expectation of bin centres under the decoder output distribution:

      soft_norm[t] = softmax(logits[t, bin_slice]) · bin_centres    (normalised space)

  Then apply a hinge loss that pushes MAE(soft, correct) < MAE(soft, wrong):

      L_contrast = mean_B( relu( MARGIN + MAE(soft, correct_norm) - MAE(soft, wrong_norm) ) )

  All comparisons are done in the tokeniser's normalised space (divided by scale)
  so the margin is scale-invariant.

Total loss:
  L = L_ce + LAMBDA_CONTRASTIVE * L_contrast

Two mode tokens (same as v4/v5):
  4096  (<fast_mode>)  → predict 64-step forecast
  4097  (<trend_mode>) → predict 64-step trend of context

Usage:
  python fine_tune_trend_v6.py [--job-id JOB_ID]

Outputs (all under figures/finetune_trend_v6_{JOB_ID}/):
  config.txt  loss.png  eval_fast_samples.png  eval_trend_samples.png

Checkpoint: checkpoints/trend-finetune-v6-{JOB_ID}/
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

# ── mode tokens ───────────────────────────────────────────────────────────────
FAST_TOKEN_ID  = 4096
TREND_TOKEN_ID = 4097
EXTENDED_VOCAB = 4098

# ── sequence lengths ──────────────────────────────────────────────────────────
CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 64

# ── training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE   = 128
N_STEPS      = 10_000
WARMUP_STEPS = 500
LR           = 1e-4
GRAD_CLIP    = 1.0
EVAL_EVERY   = 500

# ── contrastive loss hyperparameters ─────────────────────────────────────────
LAMBDA_CONTRASTIVE  = 0.5    # weight of contrastive loss relative to CE
CONTRASTIVE_MARGIN  = 0.05   # scale-invariant margin (in normalised bin space)

# ── eval hyperparameters ──────────────────────────────────────────────────────
EVAL_BATCH         = 32
EVAL_SAMPLES       = 20
EVAL_N_SHOW        = 5
EVAL_POOL_PER_MODE = 100

# ── embedding init (same as v5) ───────────────────────────────────────────────
EMBED_INIT_MODE           = "eos_plus_noise"
EMBED_INIT_STD_MULTIPLIER = 2.0

SEED     = 42
MODEL_ID = "amazon/chronos-t5-small"
DATA_PATH = FINETUNING_ROOT / "data" / "sarsim0-contrastive-v1.arrow"

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


# ── mode-token embedding init (same as v5) ───────────────────────────────────

def init_mode_embeddings(inner, eos_token_id: int = 1) -> None:
    embed_weight = inner.get_input_embeddings().weight
    existing_std = embed_weight[:FAST_TOKEN_ID].std().item()
    noise_std    = EMBED_INIT_STD_MULTIPLIER * existing_std

    with torch.no_grad():
        if EMBED_INIT_MODE == "eos_plus_noise":
            base = embed_weight[eos_token_id].clone()
        else:
            base = torch.zeros(embed_weight.shape[1], dtype=embed_weight.dtype,
                               device=embed_weight.device)
        gen = torch.Generator()
        gen.manual_seed(SEED)
        embed_weight[FAST_TOKEN_ID]  = base + noise_std * torch.randn(
            embed_weight.shape[1], generator=gen, dtype=embed_weight.dtype)
        embed_weight[TREND_TOKEN_ID] = base + noise_std * torch.randn(
            embed_weight.shape[1], generator=gen, dtype=embed_weight.dtype)

    fast_emb  = embed_weight[FAST_TOKEN_ID].float()
    trend_emb = embed_weight[TREND_TOKEN_ID].float()
    eos_emb   = embed_weight[eos_token_id].float()
    cos_ft = F.cosine_similarity(fast_emb.unsqueeze(0), trend_emb.unsqueeze(0)).item()
    cos_fe = F.cosine_similarity(fast_emb.unsqueeze(0), eos_emb.unsqueeze(0)).item()
    cos_te = F.cosine_similarity(trend_emb.unsqueeze(0), eos_emb.unsqueeze(0)).item()
    print(f"  Embed init : {EMBED_INIT_MODE},  noise_std={noise_std:.5f}")
    print(f"  Cosine sim : fast↔trend={cos_ft:.3f}  fast↔EOS={cos_fe:.3f}  trend↔EOS={cos_te:.3f}")


# ── config record ─────────────────────────────────────────────────────────────

def write_config(path: Path, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Experiment Configuration",
        "=" * 40,
        f"Job ID          : {job_id}",
        f"Script          : fine_tune_trend_v6.py",
        f"Model           : {MODEL_ID}",
        f"Dataset         : sarsim0-contrastive-v1.arrow",
        f"Dataset split   : 30% fast-mode / 70% trend-mode",
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
        "Architecture (same as v4/v5) — decoder-side mode conditioning:",
        "  encoder           = [ctx_tokens | EOS]              (513 tokens)",
        "  decoder_input_ids = [mode_token | tgt_0..tgt_62]   (65 tokens)",
        "",
        "Key change vs v5 — contrastive loss:",
        f"  LAMBDA_CONTRASTIVE : {LAMBDA_CONTRASTIVE}",
        f"  CONTRASTIVE_MARGIN : {CONTRASTIVE_MARGIN}  (scale-invariant, normalised bin space)",
        "  L = L_ce + lambda * relu(margin + MAE(soft,correct_norm) - MAE(soft,wrong_norm))",
        "",
        f"Embedding init : {EMBED_INIT_MODE}  (multiplier={EMBED_INIT_STD_MULTIPLIER})",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"  Config written: {path.name}")


# ── data ──────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> tuple[list[dict], list[dict]]:
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()

    fast_rows, trend_rows = [], []
    for item_id, mode, ctx, correct, wrong in zip(
        table.column("item_id").to_pylist(),
        table.column("mode").to_pylist(),
        table.column("context").to_pylist(),
        table.column("correct_target").to_pylist(),
        table.column("wrong_target").to_pylist(),
    ):
        row = {
            "item_id":        item_id,
            "mode":           mode,
            "context":        np.array(ctx,     dtype=np.float32),
            "correct_target": np.array(correct, dtype=np.float32),
            "wrong_target":   np.array(wrong,   dtype=np.float32),
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

    Stores per row:
      input_ids     : (513,)  encoder tokens [ctx | EOS]
      attn_mask     : (513,)  encoder attention mask
      dec_input_ids : (65,)   [mode_token | tgt_0..tgt_62]
      labels        : (65,)   [tgt_0..tgt_63 | EOS]  (-100 for padding)
      scale         : ()      normalisation scale (mean abs of context)
      correct_t     : (64,)   correct target in original value space
      wrong_t       : (64,)   wrong  target in original value space
    """
    N = len(rows)
    print(f"  Pre-tokenising {N:,} training rows on CPU...")

    contexts = torch.tensor(np.stack([d["context"]        for d in rows]), dtype=torch.float32)
    targets  = torch.tensor(np.stack([d["correct_target"] for d in rows]), dtype=torch.float32)

    # Encoder: context only — no mode prefix
    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    # token_ids : (N, 513)  attn_mask : (N, 513)  scale : (N,)

    # Decoder targets
    label_ids, label_mask = tokenizer.label_input_transform(targets, scale)
    # label_ids : (N, 65) = [tgt_0..tgt_63 | EOS]
    labels = label_ids.masked_fill(~label_mask.bool(), -100)

    # Decoder input: [mode_token | tgt_0..tgt_62]
    mode_col = torch.tensor(
        [FAST_TOKEN_ID if d["mode"] == "fast" else TREND_TOKEN_ID for d in rows],
        dtype=token_ids.dtype,
    ).unsqueeze(1)
    dec_input_ids = torch.cat([mode_col, label_ids[:, :-1]], dim=1)  # (N, 65)

    # Contrastive targets as tensors (kept in original float space)
    correct_t = torch.tensor(np.stack([d["correct_target"] for d in rows]), dtype=torch.float32)
    wrong_t   = torch.tensor(np.stack([d["wrong_target"]   for d in rows]), dtype=torch.float32)

    tokenised = []
    for i, d in enumerate(rows):
        tokenised.append({
            "item_id":       d["item_id"],
            "mode":          d["mode"],
            "context":       d["context"],        # numpy, kept for eval plots
            "correct_target": d["correct_target"], # numpy, kept for eval plots
            "wrong_target":   d["wrong_target"],   # numpy, kept for eval plots
            "input_ids":     token_ids[i],         # (513,) CPU tensor
            "attn_mask":     attn_mask[i],         # (513,) CPU tensor
            "dec_input_ids": dec_input_ids[i],     # (65,)  CPU tensor
            "labels":        labels[i],            # (65,)  CPU tensor
            "scale":         scale[i],             # ()     CPU tensor
            "correct_t":     correct_t[i],         # (64,)  CPU tensor
            "wrong_t":       wrong_t[i],           # (64,)  CPU tensor
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


# ── batch assembly ────────────────────────────────────────────────────────────

def assemble_batch(batch: list[dict], device: torch.device):
    input_ids     = torch.stack([d["input_ids"]     for d in batch]).to(device)
    attn_mask     = torch.stack([d["attn_mask"]     for d in batch]).to(device)
    dec_input_ids = torch.stack([d["dec_input_ids"] for d in batch]).to(device)
    labels        = torch.stack([d["labels"]        for d in batch]).to(device)
    scale         = torch.stack([d["scale"]         for d in batch]).to(device)   # (B,)
    correct_t     = torch.stack([d["correct_t"]     for d in batch]).to(device)   # (B, 64)
    wrong_t       = torch.stack([d["wrong_t"]       for d in batch]).to(device)   # (B, 64)
    return input_ids, attn_mask, dec_input_ids, labels, scale, correct_t, wrong_t


# ── contrastive loss ──────────────────────────────────────────────────────────

def compute_contrastive_loss(
    logits:       torch.Tensor,   # (B, 65, vocab_extended)  — full decoder logits
    scale:        torch.Tensor,   # (B,)                     — per-sample scale
    correct_t:    torch.Tensor,   # (B, 64) float32          — correct target values
    wrong_t:      torch.Tensor,   # (B, 64) float32          — wrong  target values
    bin_centers:  torch.Tensor,   # (n_centers,)             — centres in normalised space
    n_special:    int,            # tokenizer config.n_special_tokens
) -> torch.Tensor:
    """
    Soft-prediction contrastive loss (scale-invariant).

    Converts decoder logits → soft value predictions via:
        soft_norm[b,t] = softmax(logits[b, t, bin_slice]) · bin_centres

    Then in normalised space (divided by scale):
        L = mean_B( relu( MARGIN + MAE(soft_norm, correct_norm) - MAE(soft_norm, wrong_norm) ) )
    """
    n_centers = bin_centers.shape[0]

    # Bin tokens occupy IDs [n_special+1 .. n_special+n_centers] (mirrors output_transform)
    # Use first 64 decoder positions (value predictions; position 64 predicts EOS)
    bin_logits = logits[:, :64, n_special + 1 : n_special + 1 + n_centers].float()  # (B, 64, C)
    probs      = F.softmax(bin_logits, dim=-1)                                        # (B, 64, C)
    ctrs       = bin_centers.to(logits.device).float()                                # (C,)
    soft_norm  = (probs * ctrs).sum(dim=-1)                                           # (B, 64)

    # Convert targets to normalised space for scale-invariant comparison
    sc = scale.float().unsqueeze(1).clamp(min=1e-8)   # (B, 1)
    correct_norm = correct_t.float() / sc              # (B, 64)
    wrong_norm   = wrong_t.float()   / sc              # (B, 64)

    mae_correct = (soft_norm - correct_norm).abs().mean(dim=1)   # (B,)
    mae_wrong   = (soft_norm - wrong_norm  ).abs().mean(dim=1)   # (B,)

    return F.relu(CONTRASTIVE_MARGIN + mae_correct - mae_wrong).mean()


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
    B   = len(contexts)
    ctx = torch.tensor(np.stack(contexts), dtype=torch.float32)
    token_ids, attn_mask, scale = pipeline.tokenizer.context_input_transform(ctx)

    raw = pipeline.model.model.generate(
        input_ids=token_ids.to(device),
        attention_mask=attn_mask.to(device),
        generation_config=GenerationConfig(
            min_new_tokens=prediction_length,
            max_new_tokens=prediction_length,
            do_sample=True,
            num_return_sequences=num_samples,
            decoder_start_token_id=mode_token_id,
            eos_token_id=pipeline.model.config.eos_token_id,
            pad_token_id=pipeline.model.config.pad_token_id,
            temperature=pipeline.model.config.temperature,
            top_k=pipeline.model.config.top_k,
            top_p=pipeline.model.config.top_p,
        ),
    )
    raw     = raw[:, 1:].reshape(B, num_samples, prediction_length)
    preds   = pipeline.tokenizer.output_transform(raw.cpu(), scale)
    medians = preds.median(dim=1).values.numpy()
    return [medians[i] for i in range(B)]


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(pipeline, eval_rows, device, step) -> tuple[float, float]:
    pipeline.model.model.eval()

    fast_rows  = [r for r in eval_rows if r["mode"] == "fast"]
    trend_rows = [r for r in eval_rows if r["mode"] == "trend"]

    def run_mode(rows, mode_token_id, label):
        maes = []
        bar  = tqdm(
            range(0, len(rows), EVAL_BATCH),
            desc=f"    eval {label} step={step:,}",
            file=sys.stdout, leave=False, dynamic_ncols=True,
        )
        for i in bar:
            batch    = rows[i:i + EVAL_BATCH]
            contexts = [r["context"] for r in batch]
            preds    = predict_batch_with_mode(pipeline, contexts, mode_token_id, device)
            for r, pred in zip(batch, preds):
                maes.append(float(np.mean(np.abs(pred - r["correct_target"]))))
        return float(np.mean(maes)) if maes else float("nan")

    mean_fast  = run_mode(fast_rows,  FAST_TOKEN_ID,  "fast ")
    mean_trend = run_mode(trend_rows, TREND_TOKEN_ID, "trend")

    pipeline.model.model.train()
    return mean_fast, mean_trend


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_loss(
    loss_steps, train_ce, train_contrast, train_total,
    eval_steps, fast_maes, trend_maes,
    path: Path, job_id: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # CE loss
    axes[0].plot(loss_steps, train_ce, color="steelblue", lw=0.8, alpha=0.6)
    if len(train_ce) >= 50:
        sm = np.convolve(train_ce, np.ones(50)/50, mode="valid")
        axes[0].plot(loss_steps[49:], sm, color="steelblue", lw=1.8, label="CE loss (MA-50)")
    axes[0].set_title("CE Loss", fontweight="bold")
    axes[0].set_xlabel("Step")
    axes[0].legend(fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Contrastive loss
    axes[1].plot(loss_steps, train_contrast, color="darkorange", lw=0.8, alpha=0.6)
    if len(train_contrast) >= 50:
        sm = np.convolve(train_contrast, np.ones(50)/50, mode="valid")
        axes[1].plot(loss_steps[49:], sm, color="darkorange", lw=1.8, label="Contrastive loss (MA-50)")
    axes[1].set_title("Contrastive Loss", fontweight="bold")
    axes[1].set_xlabel("Step")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    # Eval MAE
    axes[2].plot(eval_steps, fast_maes,  color=_GT_COL,    lw=1.8, marker="o", ms=3, label="Fast-mode MAE")
    axes[2].plot(eval_steps, trend_maes, color=_TREND_COL, lw=1.8, marker="s", ms=3, label="Trend-mode MAE")
    axes[2].set_title("Eval MAE (held-out)", fontweight="bold")
    axes[2].set_xlabel("Step")
    axes[2].legend(fontsize=8)
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Fine-tuning v6: chronos-t5-small  ·  30% fast / 70% trend  ·  contrastive loss  ·  job {job_id}",
        fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_eval_samples(pipeline, eval_rows, mode, device, path, job_id, n=EVAL_N_SHOW) -> None:
    rows          = [r for r in eval_rows if r["mode"] == mode]
    rng           = np.random.default_rng(0)
    picks         = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    picks.sort()
    mode_token_id = FAST_TOKEN_ID if mode == "fast" else TREND_TOKEN_ID
    colour        = _GT_COL if mode == "fast" else _TREND_COL
    label_gt      = "Target forecast" if mode == "fast" else "Target trend"

    selected = [rows[i] for i in picks]
    preds    = predict_batch_with_mode(pipeline, [r["context"] for r in selected], mode_token_id, device)

    ctx_x = np.arange(CONTEXT_LENGTH)
    tgt_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"Fine-tuned chronos-t5-small (v6)  ·  {mode}-mode eval  ·  job {job_id}\n"
        f"{colour} = ground truth   crimson = model prediction",
        fontsize=11, fontweight="bold",
    )

    pipeline.model.model.eval()
    for row_idx, (row, pred) in enumerate(zip(selected, preds)):
        ax = axes[row_idx]
        ax.fill_between(ctx_x, row["context"], alpha=0.07, color=_CTX_COL)
        ax.plot(ctx_x, row["context"],         color=_CTX_COL,  lw=0.7, label="Input context")
        ax.plot(tgt_x, row["correct_target"],  color=colour,    lw=1.8, label=label_gt)
        ax.plot(tgt_x, pred,                   color=_PRED_COL, lw=1.8, ls="--", label="Model prediction")
        ax.axvline(CONTEXT_LENGTH - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        mae = float(np.mean(np.abs(pred - row["correct_target"])))
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

    FIGURES_DIR = FINETUNING_ROOT / "figures"     / f"finetune_trend_v6_{job_id}"
    CKPT_DIR    = FINETUNING_ROOT / "checkpoints" / f"trend-finetune-v6-{job_id}"

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Fine-tuning v6: chronos-t5-small  |  contrastive-v1  |  decoder-mode + contrastive loss")
    print(f"  Job ID    : {job_id}")
    print(f"  Device    : {device}")
    print(f"  Steps     : {N_STEPS:,}  |  batch={BATCH_SIZE}")
    print(f"  LR        : {LR}  |  warmup={WARMUP_STEPS}")
    print(f"  Lambda    : {LAMBDA_CONTRASTIVE}  |  margin={CONTRASTIVE_MARGIN}")
    print(f"  Eval every: {EVAL_EVERY} steps  |  eval_batch={EVAL_BATCH}")
    print(f"  Figures   : {FIGURES_DIR}")
    print(f"  Checkpoint: {CKPT_DIR}")
    print("=" * 60)

    write_config(FIGURES_DIR / "config.txt", job_id)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    inner    = pipeline.model.model

    current_vocab = inner.get_input_embeddings().weight.shape[0]
    if current_vocab < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
        print(f"  Vocab extended: {current_vocab} → {EXTENDED_VOCAB}")
    init_mode_embeddings(inner, eos_token_id=1)

    # Cache tokeniser bin info for contrastive loss (stays on CPU until needed)
    bin_centers = pipeline.tokenizer.centers.clone()   # (n_centers,) normalised
    n_special   = pipeline.tokenizer.config.n_special_tokens
    print(f"  Bin centers : {len(bin_centers):,} bins  |  n_special_tokens={n_special}")

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

    # ── optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(inner.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=N_STEPS,
    )

    # ── training loop ─────────────────────────────────────────────────────────
    loss_steps    = []
    train_ce      = []
    train_contrast = []
    train_total   = []
    eval_steps, fast_maes, trend_maes = [], [], []

    running_ce = running_contrast = 0.0
    LOG_EVERY  = 50

    print("\nTraining...")
    pbar = tqdm(
        range(1, N_STEPS + 1), total=N_STEPS, desc="Training",
        file=sys.stdout, dynamic_ncols=True, smoothing=0.05,
        miniters=LOG_EVERY, mininterval=0.0,
    )

    for step in pbar:
        batch = loader.next_batch()
        input_ids, attn_mask, dec_input_ids, labels, scale, correct_t, wrong_t = \
            assemble_batch(batch, device)

        out     = inner(
            input_ids=input_ids,
            attention_mask=attn_mask,
            decoder_input_ids=dec_input_ids,
            labels=labels,
        )
        ce_loss = out.loss

        contrast_loss = compute_contrastive_loss(
            out.logits, scale, correct_t, wrong_t, bin_centers, n_special,
        )
        loss = ce_loss + LAMBDA_CONTRASTIVE * contrast_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        running_ce       += ce_loss.item()
        running_contrast += contrast_loss.item()

        if step % LOG_EVERY == 0:
            avg_ce  = running_ce       / LOG_EVERY
            avg_cc  = running_contrast / LOG_EVERY
            avg_tot = avg_ce + LAMBDA_CONTRASTIVE * avg_cc
            loss_steps.append(step)
            train_ce.append(avg_ce)
            train_contrast.append(avg_cc)
            train_total.append(avg_tot)
            running_ce = running_contrast = 0.0
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix(ce=f"{avg_ce:.4f}", cc=f"{avg_cc:.4f}", lr=f"{lr_now:.2e}")
            tqdm.write(
                f"  step {step:>6,}/{N_STEPS:,}"
                f"  ce={avg_ce:.4f}  contrast={avg_cc:.4f}  total={avg_tot:.4f}"
                f"  lr={lr_now:.2e}"
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

    # ── save checkpoint ───────────────────────────────────────────────────────
    print(f"\nSaving checkpoint to {CKPT_DIR} ...")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    inner.save_pretrained(str(CKPT_DIR))
    np.save(str(CKPT_DIR / "loss_steps.npy"),    np.array(loss_steps))
    np.save(str(CKPT_DIR / "train_ce.npy"),      np.array(train_ce))
    np.save(str(CKPT_DIR / "train_contrast.npy"), np.array(train_contrast))
    np.save(str(CKPT_DIR / "eval_steps.npy"),    np.array(eval_steps))
    np.save(str(CKPT_DIR / "fast_maes.npy"),     np.array(fast_maes))
    np.save(str(CKPT_DIR / "trend_maes.npy"),    np.array(trend_maes))
    print("  Done.")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    inner.eval()
    pipeline.model.model = inner

    plot_loss(
        loss_steps, train_ce, train_contrast, train_total,
        eval_steps, fast_maes, trend_maes,
        path=FIGURES_DIR / "loss.png", job_id=job_id,
    )
    plot_eval_samples(pipeline, eval_rows, mode="fast",  device=device,
                      path=FIGURES_DIR / "eval_fast_samples.png",  job_id=job_id)
    plot_eval_samples(pipeline, eval_rows, mode="trend", device=device,
                      path=FIGURES_DIR / "eval_trend_samples.png", job_id=job_id)

    print(f"\nDone.  Figures in {FIGURES_DIR}/")
    print(f"       Checkpoint in {CKPT_DIR}/")
    print("\n" + "=" * 60)
    print("Final eval metrics:")
    print(f"  Fast-mode  MAE : {fast_maes[-1]:.4f}")
    print(f"  Trend-mode MAE : {trend_maes[-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
