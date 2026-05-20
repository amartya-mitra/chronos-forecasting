#!/usr/bin/env python3
"""
SarSim0 fine-tuning with STL-estimated Parallel Auxiliary Conditioning.

Replaces oracle trend/seasonal (fine_tune_sarsim0_aux_cond.py) with
STL-estimated components derived entirely from the observed context.
This makes the conditioning deployable at inference time — no future
ground truth is required.

Decomposition pipeline (per sample, computed once before training):
  1. Run FFT on the 512-step context → detect dominant period.
  2. Run STL(context, period=detected, robust=True).
  3. Take the last 64 steps of STL trend and seasonal as aux inputs.

The aux inputs are then scale-normalised by context_scale and projected
via the same AuxiliaryConditioningModule (Linear(64→512)+Tanh ×2).

Training data: ALL 10,000 rows in sarsim0-10k.arrow (fast + reasoning),
since STL estimation from context does not depend on the mode label.

Figures:
  figures/finetunes/finetune_sarsim0_stlcond_{job_id}/
    config.txt
    loss.png
    eval_samples.png  — context | STL trend (ctx tail) | STL seasonal (ctx tail)

Usage (single GPU):
  /opt/conda/bin/python3.12 reasoning-finetuning/scripts/fine_tune_sarsim0_stl_cond.py \\
      --job-id $SLURM_JOB_ID [--freeze-backbone]

Usage (multi-GPU via torchrun):
  torchrun --nproc_per_node=4 --master_port=29505 \\
      reasoning-finetuning/scripts/fine_tune_sarsim0_stl_cond.py \\
      --job-id $SLURM_JOB_ID
"""

import os
import sys
import argparse
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.seasonal import STL
from transformers import get_cosine_schedule_with_warmup, GenerationConfig
from transformers.modeling_outputs import BaseModelOutput

from chronos import ChronosPipeline

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = FINETUNING_ROOT / "data"    / "sarsim0-10k.arrow"
BASELINE_REF = FINETUNING_ROOT / "figures" / "evals" / "eval_pretrained_sarsim0" / "baseline_mae.txt"
MODEL_ID     = "amazon/chronos-t5-small"

# ── dataset constants ─────────────────────────────────────────────────────────
CONTEXT_LEN = 512
HORIZON     = 64
DECOMP_LEN  = 64   # last N steps of STL output used as aux input
D_MODEL     = 512  # T5-small hidden dimension

# ── special / extended vocab ──────────────────────────────────────────────────
FAST_TOKEN_ID  = 4096
EXTENDED_VOCAB = 4105

# ── hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 64
N_STEPS      = 10_000
WARMUP_STEPS = 500
LR_AUX       = 1e-3
LR_BACKBONE  = 1e-4
GRAD_CLIP    = 1.0
EVAL_EVERY   = 500
EVAL_BATCH   = 32
EVAL_SAMPLES = 20
EVAL_N_SHOW  = 5
EVAL_POOL    = 200   # larger pool — all rows available

EARLY_STOP_PATIENCE  = 5
EARLY_STOP_MIN_DELTA = 1e-4

EMBED_INIT_STD_MULTIPLIER = 2.0
SCALE_EPS = 1e-4
SEED      = 42

_CTX_COL  = "steelblue"
_PRED_COL = "crimson"
_GT_COL   = "darkorange"
_TRND_COL = "seagreen"
_SEAS_COL = "mediumpurple"


# ── STL decomposition ─────────────────────────────────────────────────────────

def stl_decompose(
    context: np.ndarray,
    decomp_len: int = DECOMP_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate trend and seasonal components of `context` via STL.
    Returns the last `decomp_len` steps of each component.

    Period detection: dominant frequency from FFT of the de-meaned context.
    Clamped to [2, len(context) // 2 - 1] to satisfy STL requirements.

    Args:
        context:    (CONTEXT_LEN,) float32 array of observed signal values.
        decomp_len: number of tail steps to return (= HORIZON = 64).

    Returns:
        trend_tail:    (decomp_len,) float32 — STL trend, last N steps of context.
        seasonal_tail: (decomp_len,) float32 — STL seasonal, last N steps of context.
    """
    n = len(context)

    # FFT-based dominant period detection
    fft_mag = np.abs(np.fft.rfft(context - context.mean()))
    fft_mag[0] = 0.0  # suppress DC
    freqs = np.fft.rfftfreq(n)
    # Find peak ignoring DC (index 0)
    peak_idx = int(np.argmax(fft_mag[1:])) + 1
    dominant_freq = freqs[peak_idx]
    period = int(round(1.0 / dominant_freq)) if dominant_freq > 1e-6 else 12
    # Clamp: STL requires nobs > 2 * period
    period = int(np.clip(period, 2, n // 2 - 1))

    res = STL(context, period=period, robust=True).fit()
    trend_tail    = res.trend[-decomp_len:].astype(np.float32)
    seasonal_tail = res.seasonal[-decomp_len:].astype(np.float32)
    return trend_tail, seasonal_tail


# ── Model ─────────────────────────────────────────────────────────────────────

class AuxiliaryConditioningModule(nn.Module):
    """
    Linear(DECOMP_LEN→D_MODEL) + Tanh, one projection per component.
    Total new parameters: 2 × (64×512 + 512) = 66,560.
    """
    def __init__(self, decomp_len: int = DECOMP_LEN, d_model: int = D_MODEL):
        super().__init__()
        self.trend_proj    = nn.Linear(decomp_len, d_model)
        self.seasonal_proj = nn.Linear(decomp_len, d_model)
        self.act = nn.Tanh()

    def forward(self, trend: torch.Tensor, seasonal: torch.Tensor) -> torch.Tensor:
        # trend, seasonal: (B, DECOMP_LEN)  → output: (B, 2, D_MODEL)
        t = self.act(self.trend_proj(trend))
        s = self.act(self.seasonal_proj(seasonal))
        return torch.stack([t, s], dim=1)


class ChronosWithAuxCond(nn.Module):
    """Wraps T5ForConditionalGeneration with auxiliary conditioning tokens."""

    def __init__(self, inner, decomp_len: int = DECOMP_LEN, d_model: int = D_MODEL):
        super().__init__()
        self.inner      = inner
        self.aux_module = AuxiliaryConditioningModule(decomp_len, d_model)

    def _augment_encoder(self, input_ids, attention_mask, aux_trend, aux_seasonal):
        enc_out  = self.inner.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc_dtype = enc_out.last_hidden_state.dtype
        aux_tokens = self.aux_module(aux_trend.to(enc_dtype), aux_seasonal.to(enc_dtype))
        aug_hidden = torch.cat([enc_out.last_hidden_state, aux_tokens], dim=1)
        aux_mask   = torch.ones(attention_mask.size(0), 2,
                                dtype=attention_mask.dtype, device=attention_mask.device)
        aug_mask   = torch.cat([attention_mask, aux_mask], dim=1)
        return BaseModelOutput(last_hidden_state=aug_hidden), aug_mask

    def forward(self, input_ids, attention_mask, aux_trend, aux_seasonal,
                decoder_input_ids=None, labels=None):
        aug_enc_out, aug_mask = self._augment_encoder(
            input_ids, attention_mask, aux_trend, aux_seasonal)
        return self.inner(
            attention_mask=aug_mask,
            encoder_outputs=aug_enc_out,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    @torch.no_grad()
    def generate_with_aux(self, input_ids, attention_mask, aux_trend, aux_seasonal,
                          **gen_kwargs):
        aug_enc_out, aug_mask = self._augment_encoder(
            input_ids, attention_mask, aux_trend, aux_seasonal)
        return self.inner.generate(
            encoder_outputs=aug_enc_out,
            attention_mask=aug_mask,
            **gen_kwargs,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def init_mode_embeddings(inner) -> None:
    w = inner.get_input_embeddings().weight
    noise_std = EMBED_INIT_STD_MULTIPLIER * w[:4096].std().item()
    with torch.no_grad():
        base = w[1].clone()
        gen  = torch.Generator(); gen.manual_seed(SEED)
        for tok_id in range(4096, EXTENDED_VOCAB):
            w[tok_id] = base + noise_std * torch.randn(
                w.shape[1], generator=gen, dtype=w.dtype)
    print(f"  Embed init : eos_plus_noise  noise_std={noise_std:.5f}")
    cos = F.cosine_similarity(
        w[FAST_TOKEN_ID].float().unsqueeze(0), w[1].float().unsqueeze(0)).item()
    print(f"  Cosine sim : FAST↔EOS = {cos:.3f}")


def load_baseline_mae() -> dict:
    ref = {"mean_mae": None, "median_mae": None}
    if BASELINE_REF.exists():
        for line in BASELINE_REF.read_text().splitlines():
            if line.startswith("mean_mae"):
                ref["mean_mae"]   = float(line.split(":")[-1].strip())
            elif line.startswith("median_mae"):
                ref["median_mae"] = float(line.split(":")[-1].strip())
    return ref


def write_config(path: Path, job_id: str, freeze_backbone: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    phase = "1 (aux only, backbone frozen)" if freeze_backbone else "2 (full finetune)"
    path.write_text("\n".join([
        "Experiment Configuration", "=" * 40,
        f"Job ID         : {job_id}",
        f"Script         : fine_tune_sarsim0_stl_cond.py",
        f"Model          : {MODEL_ID}",
        f"Dataset        : sarsim0-10k.arrow  (ALL rows — fast + reasoning)",
        f"Phase          : {phase}",
        f"LR_aux         : {LR_AUX}  |  LR_backbone : {LR_BACKBONE}",
        f"Batch size     : {BATCH_SIZE}",
        f"Steps          : {N_STEPS:,}  |  warmup : {WARMUP_STEPS}",
        f"Ctx len        : {CONTEXT_LEN}  |  Horizon : {HORIZON}",
        f"Decomp source  : STL on full 512-step context, last {DECOMP_LEN} steps",
        f"Period detect  : FFT dominant frequency, clamped to [2, {CONTEXT_LEN//2-1}]",
        f"Aux tokens     : 2 (trend + seasonal, appended to encoder → L+2 cross-attn)",
        f"Aux params     : 66,560  (Linear×2 + Tanh)",
        f"Vocab          : extended to {EXTENDED_VOCAB}",
        f"Early stop     : patience={EARLY_STOP_PATIENCE}  delta={EARLY_STOP_MIN_DELTA}  (FAST MAE)",
        "",
        "Key difference from fine_tune_sarsim0_aux_cond.py:",
        "  - Aux inputs are STL-estimated from observed context (deployable at inference).",
        "  - Oracle GT trend/seasonal are NOT used.",
        "  - ALL 10k rows used for training (not just 6k reasoning-mode rows).",
    ]) + "\n")
    print(f"  Config written: {path.name}")


# ── data ──────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> tuple[list, list]:
    """
    Load ALL rows (fast + reasoning) from sarsim0-10k.arrow.
    STL estimation from context requires no mode-specific columns.
    Returns (train_rows, eval_rows) after shuffle.
    Each row: {item_id, context(512,), forecast(64,)}
    """
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()

    rows = []
    for item_id, ctx, fct in zip(
        table.column("item_id").to_pylist(),
        table.column("context").to_pylist(),
        table.column("forecast").to_pylist(),
    ):
        rows.append({
            "item_id":  item_id,
            "context":  np.array(ctx, dtype=np.float32),   # (512,)
            "forecast": np.array(fct, dtype=np.float32),   # (64,)
        })

    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[EVAL_POOL:], rows[:EVAL_POOL]   # (train, eval)


def compute_stl_aux(rows: list, desc: str = "STL decomposition") -> list:
    """
    Run STL on each row's context and attach stl_trend / stl_seasonal.
    Returns the same list with two new keys added per row (in-place).
    """
    for r in tqdm(rows, desc=desc, unit="row", file=sys.stdout):
        t, s = stl_decompose(r["context"])
        r["stl_trend"]    = t   # (DECOMP_LEN,) — last 64 steps of context STL trend
        r["stl_seasonal"] = s   # (DECOMP_LEN,) — last 64 steps of context STL seasonal
    return rows


def pretokenise(rows: list, tokenizer) -> list:
    """
    Tokenise all rows.  STL estimates must already be attached (stl_trend, stl_seasonal).
    Scale-normalises aux arrays by context_scale per sample.
    """
    N = len(rows)
    contexts  = torch.tensor(np.stack([r["context"]  for r in rows]), dtype=torch.float32)
    forecasts = torch.tensor(np.stack([r["forecast"] for r in rows]), dtype=torch.float32)
    stl_t     = np.stack([r["stl_trend"]    for r in rows])   # (N, 64)
    stl_s     = np.stack([r["stl_seasonal"] for r in rows])   # (N, 64)

    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    label_ids, label_mask, _    = tokenizer._input_transform(
        context=forecasts, scale=scale)

    eos_col  = torch.full((N, 1), tokenizer.config.eos_token_id, dtype=label_ids.dtype)
    ones_col = torch.ones((N, 1), dtype=label_ids.dtype)
    labels_seq    = torch.cat([label_ids, eos_col],   dim=1)
    mask_seq      = torch.cat([label_mask, ones_col], dim=1)
    labels        = labels_seq.masked_fill(~mask_seq.bool(), -100)
    mode_col      = torch.full((N, 1), FAST_TOKEN_ID, dtype=token_ids.dtype)
    dec_input_ids = torch.cat([mode_col, labels_seq[:, :-1]], dim=1)

    scale_np     = scale.numpy().reshape(-1, 1)
    aux_trend    = torch.tensor(stl_t / np.maximum(scale_np, SCALE_EPS), dtype=torch.float32)
    aux_seasonal = torch.tensor(stl_s / np.maximum(scale_np, SCALE_EPS), dtype=torch.float32)

    return [{
        "item_id":       rows[i]["item_id"],
        "context":       rows[i]["context"],
        "forecast":      rows[i]["forecast"],
        "stl_trend":     rows[i]["stl_trend"],
        "stl_seasonal":  rows[i]["stl_seasonal"],
        "input_ids":     token_ids[i],
        "attn_mask":     attn_mask[i],
        "dec_input_ids": dec_input_ids[i],
        "labels":        labels[i],
        "aux_trend":     aux_trend[i],
        "aux_seasonal":  aux_seasonal[i],
        "scale":         scale[i],
    } for i in range(N)]


class CyclicLoader:
    def __init__(self, rows, batch_size, seed=0):
        self.rows, self.batch_size = rows, batch_size
        self.rng  = np.random.default_rng(seed)
        self._idx = np.arange(len(rows)); self._pos = len(rows)

    def next_batch(self):
        if self._pos + self.batch_size > len(self._idx):
            self.rng.shuffle(self._idx); self._pos = 0
        b = [self.rows[i] for i in self._idx[self._pos:self._pos + self.batch_size]]
        self._pos += self.batch_size
        return b


def assemble_batch(batch: list, device) -> tuple:
    ids  = torch.stack([d["input_ids"]     for d in batch]).to(device)
    mask = torch.stack([d["attn_mask"]     for d in batch]).to(device)
    dec  = torch.stack([d["dec_input_ids"] for d in batch]).to(device)
    lbl  = torch.stack([d["labels"]        for d in batch]).to(device)
    atr  = torch.stack([d["aux_trend"]     for d in batch]).to(device)
    ase  = torch.stack([d["aux_seasonal"]  for d in batch]).to(device)
    return ids, mask, dec, lbl, atr, ase


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_fast_with_aux(
    model, pipeline,
    contexts:  list,
    stl_trends:    list,
    stl_seasonals: list,
    device,
) -> np.ndarray:
    """FAST-mode generation with STL-estimated aux conditioning."""
    B  = len(contexts)
    mc = pipeline.model.config
    ctx = torch.tensor(np.stack(contexts), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)
    scale_np     = scale.numpy().reshape(-1, 1)
    aux_trend    = torch.tensor(
        np.stack(stl_trends)    / np.maximum(scale_np, SCALE_EPS),
        dtype=torch.float32).to(device)
    aux_seasonal = torch.tensor(
        np.stack(stl_seasonals) / np.maximum(scale_np, SCALE_EPS),
        dtype=torch.float32).to(device)
    raw = model.generate_with_aux(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        aux_trend=aux_trend, aux_seasonal=aux_seasonal,
        generation_config=GenerationConfig(
            min_new_tokens=HORIZON, max_new_tokens=HORIZON,
            do_sample=True, num_return_sequences=EVAL_SAMPLES,
            decoder_start_token_id=FAST_TOKEN_ID,
            eos_token_id=mc.eos_token_id, pad_token_id=mc.pad_token_id,
            temperature=mc.temperature, top_k=mc.top_k, top_p=mc.top_p,
        ),
    )
    raw     = raw[:, 1:]
    raw_bins = raw[:, :HORIZON].reshape(B, EVAL_SAMPLES, HORIZON)
    return pipeline.tokenizer.output_transform(raw_bins.cpu(), scale).median(dim=1).values.numpy()


def evaluate_fast(model, pipeline, eval_rows, device) -> float:
    model.eval()
    maes = []
    for i in range(0, len(eval_rows), EVAL_BATCH):
        batch = eval_rows[i:i + EVAL_BATCH]
        preds = predict_fast_with_aux(
            model, pipeline,
            contexts=      [r["context"]      for r in batch],
            stl_trends=    [r["stl_trend"]    for r in batch],
            stl_seasonals= [r["stl_seasonal"] for r in batch],
            device=device,
        )
        for r, pred in zip(batch, preds):
            maes.append(float(np.mean(np.abs(pred - r["forecast"]))))
    model.train()
    return float(np.mean(maes))


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_loss(loss_steps, losses, eval_steps, fast_maes, baseline_mae,
              path, job_id, freeze_backbone) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    phase_tag = "Phase 1 (frozen backbone)" if freeze_backbone else "Phase 2 (full)"

    ax = axes[0]
    ax.plot(loss_steps, losses, color=_CTX_COL, lw=0.8, alpha=0.6)
    if len(losses) >= 50:
        sm = np.convolve(losses, np.ones(50) / 50, mode="valid")
        ax.plot(loss_steps[49:], sm, color=_CTX_COL, lw=1.8, label="Loss (MA-50)")
    ax.set_title("Training Loss (CE — FAST tokens)", fontweight="bold")
    ax.set_xlabel("Step"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(eval_steps, fast_maes, color=_GT_COL, lw=1.8, marker="o", ms=3,
            label="FAST MAE (original units)")
    if baseline_mae.get("mean_mae") is not None:
        ax.axhline(baseline_mae["mean_mae"], color="grey", lw=1.2, ls="--",
                   label=f"Pretrained mean  {baseline_mae['mean_mae']:.4f}")
    if baseline_mae.get("median_mae") is not None:
        ax.axhline(baseline_mae["median_mae"], color="black", lw=1.0, ls=":",
                   label=f"Pretrained median {baseline_mae['median_mae']:.4f}")
    ax.set_title(f"Eval MAE — FAST+STL-Aux  [{phase_tag}]", fontweight="bold")
    ax.set_xlabel("Step"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"SarSim0 STL Aux Conditioning  ·  chronos-t5-small  ·  job {job_id}  ·  {phase_tag}",
        fontweight="bold")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_eval_samples(model, pipeline, eval_rows, device, path, job_id,
                      freeze_backbone, n: int = EVAL_N_SHOW) -> None:
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(eval_rows), size=min(n, len(eval_rows)), replace=False))
    sel   = [eval_rows[i] for i in picks]

    model.eval()
    preds = predict_fast_with_aux(
        model, pipeline,
        contexts=      [r["context"]      for r in sel],
        stl_trends=    [r["stl_trend"]    for r in sel],
        stl_seasonals= [r["stl_seasonal"] for r in sel],
        device=device,
    )
    model.train()

    ctx_x  = np.arange(CONTEXT_LEN)
    fct_x  = np.arange(CONTEXT_LEN, CONTEXT_LEN + HORIZON)
    # STL aux inputs are last DECOMP_LEN steps of the context window
    tail_x = np.arange(CONTEXT_LEN - DECOMP_LEN, CONTEXT_LEN)

    fig, axes_raw = plt.subplots(n, 3, figsize=(21, 3 * n))
    axes = np.asarray(axes_raw)

    phase_tag = "Phase 1 (aux only)" if freeze_backbone else "Phase 2 (full)"
    fig.suptitle(
        f"SarSim0 STL Aux Cond · job {job_id} · {phase_tag}\n"
        "Left: context + forecast  |  Centre: STL trend (ctx tail)"
        "  |  Right: STL seasonal (ctx tail)",
        fontsize=11, fontweight="bold",
    )

    for idx, (row, pred) in enumerate(zip(sel, preds)):
        mae = float(np.mean(np.abs(pred - row["forecast"])))

        # col 0: context + GT forecast + STL-cond prediction
        ax = axes[idx, 0]
        ax.fill_between(ctx_x, row["context"], alpha=0.06, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL, lw=0.6,
                label="Context" if idx == 0 else "_")
        ax.plot(fct_x, row["forecast"], color=_GT_COL, lw=1.8,
                label="GT forecast" if idx == 0 else "_")
        ax.plot(fct_x, pred, color=_PRED_COL, lw=1.4, ls="--", alpha=0.85,
                label=f"STL-cond pred (MAE={mae:.3f})" if idx == 0 else "_")
        ax.axvline(CONTEXT_LEN - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(CONTEXT_LEN - DECOMP_LEN - 0.5, color="grey", lw=0.6, ls=":", alpha=0.4)
        ax.set_title(f"{row['item_id']}  MAE={mae:.4f}", fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

        # col 1: STL trend estimate over context tail
        ax = axes[idx, 1]
        ax.plot(tail_x, row["context"][CONTEXT_LEN - DECOMP_LEN:],
                color=_CTX_COL, lw=0.8, alpha=0.35, ls="--",
                label="Raw context tail (ref)" if idx == 0 else "_")
        ax.plot(tail_x, row["stl_trend"], color=_TRND_COL, lw=1.8,
                label="STL trend estimate (aux)" if idx == 0 else "_")
        ax.axvline(CONTEXT_LEN - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

        # col 2: STL seasonal estimate over context tail
        ax = axes[idx, 2]
        ax.plot(tail_x, row["context"][CONTEXT_LEN - DECOMP_LEN:] - row["stl_trend"],
                color=_CTX_COL, lw=0.8, alpha=0.35, ls="--",
                label="Detrended context tail (ref)" if idx == 0 else "_")
        ax.plot(tail_x, row["stl_seasonal"], color=_SEAS_COL, lw=1.8,
                label="STL seasonal estimate (aux)" if idx == 0 else "_")
        ax.axvline(CONTEXT_LEN - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

    for col in range(3):
        axes[-1, col].set_xlabel("Time step (context index)", fontsize=9)
    axes[0, 0].set_title("Context + Forecast", fontsize=9, fontweight="bold")
    axes[0, 1].set_title("STL Trend Estimate  (context tail, aux input)", fontsize=9, fontweight="bold")
    axes[0, 2].set_title("STL Seasonal Estimate  (context tail, aux input)", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id",          default="nojobid")
    parser.add_argument("--freeze-backbone", action="store_true")
    args            = parser.parse_args()
    job_id          = args.job_id
    freeze_backbone = args.freeze_backbone

    FIGURES_DIR = (FINETUNING_ROOT / "figures" / "finetunes"
                   / f"finetune_sarsim0_stlcond_{job_id}")
    CKPT_DIR    = (FINETUNING_ROOT / "checkpoints"
                   / f"finetune-sarsim0-stlcond-{job_id}")

    # ── DDP setup ─────────────────────────────────────────────────────────────
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
    IS_DDP     = LOCAL_RANK != -1
    if IS_DDP:
        dist.init_process_group(backend="nccl")
        WORLD_SIZE = dist.get_world_size()
        IS_MAIN    = (LOCAL_RANK == 0)
        device     = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(LOCAL_RANK)
    else:
        WORLD_SIZE = 1
        IS_MAIN    = True
        device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(SEED + (LOCAL_RANK if IS_DDP else 0))

    if IS_MAIN:
        baseline_mae = load_baseline_mae()
        phase_tag    = "Phase 1 (aux only, backbone frozen)" if freeze_backbone \
                       else "Phase 2 (full finetune)"
        print("=" * 65)
        print("SarSim0 STL Aux Conditioning  ·  chronos-t5-small")
        print(f"  Job ID         : {job_id}")
        print(f"  Phase          : {phase_tag}")
        print(f"  Device         : {device}  |  DDP={IS_DDP}  world={WORLD_SIZE}")
        print(f"  Steps          : {N_STEPS:,}  |  batch={BATCH_SIZE}")
        print(f"  Aux source     : STL on full 512-step context, last {DECOMP_LEN} steps")
        print(f"  LR_aux={LR_AUX}  LR_backbone={LR_BACKBONE}")
        if baseline_mae.get("mean_mae") is not None:
            print(f"  Baseline MAE   : mean={baseline_mae['mean_mae']:.4f}"
                  f"  median={baseline_mae['median_mae']:.4f}")
        print("=" * 65)
        write_config(FIGURES_DIR / "config.txt", job_id, freeze_backbone)
    else:
        baseline_mae = {}

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"\n[rank {LOCAL_RANK if IS_DDP else 0}] Loading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    inner    = pipeline.model.model
    if inner.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
    init_mode_embeddings(inner)

    model = ChronosWithAuxCond(inner, decomp_len=DECOMP_LEN, d_model=D_MODEL)

    if freeze_backbone:
        for p in model.inner.parameters():
            p.requires_grad_(False)
        if IS_MAIN:
            print("  Backbone frozen: only aux_module parameters will be optimised.")
    else:
        if IS_MAIN:
            print("  Full finetune: all parameters will be optimised.")

    # ── data ──────────────────────────────────────────────────────────────────
    if IS_MAIN:
        print("\nLoading dataset (ALL rows — fast + reasoning) ...")
    train_rows, eval_rows = load_rows(DATA_PATH)
    if IS_MAIN:
        print(f"  Train : {len(train_rows):,}  |  Eval : {len(eval_rows):,}")
        print("  Computing STL decompositions ...")
    compute_stl_aux(train_rows, desc="  STL train")
    compute_stl_aux(eval_rows,  desc="  STL eval ")
    if IS_MAIN:
        print("  Pre-tokenising ...")
    tok_rows = pretokenise(train_rows, pipeline.tokenizer)
    if IS_MAIN:
        print(f"  Tokenised {len(tok_rows):,} rows.")

    def shard(rows):
        if IS_DDP:
            return [rows[i] for i in range(len(rows)) if i % WORLD_SIZE == LOCAL_RANK]
        return rows

    per_rank_batch = max(1, BATCH_SIZE // WORLD_SIZE)
    loader = CyclicLoader(shard(tok_rows), per_rank_batch, seed=SEED)

    # ── training ──────────────────────────────────────────────────────────────
    print(f"\n[rank {LOCAL_RANK if IS_DDP else 0}] Moving model to {device} ...")
    model = model.to(device)
    model.aux_module = model.aux_module.to(torch.bfloat16)
    model.train()

    if IS_DDP:
        dp_model = DDP(model, device_ids=[LOCAL_RANK])
        if IS_MAIN:
            print(f"  DDP across {WORLD_SIZE} GPUs")
    else:
        dp_model = model

    param_groups = [{"params": model.aux_module.parameters(), "lr": LR_AUX}]
    if not freeze_backbone:
        param_groups.append({"params": model.inner.parameters(), "lr": LR_BACKBONE})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, N_STEPS)

    loss_steps, losses = [], []
    eval_steps, fast_maes = [], []
    running   = 0.0
    LOG_EVERY = 50

    best_fast_mae  = float("inf")
    no_improve_cnt = 0
    stopped_early  = False

    if IS_MAIN:
        print("\nTraining...")
    pbar = tqdm(range(1, N_STEPS + 1), total=N_STEPS,
                desc="Training (SarSim0 STL-Cond)", file=sys.stdout,
                dynamic_ncols=True, smoothing=0.05,
                miniters=LOG_EVERY, mininterval=0.0, disable=not IS_MAIN)

    for step in pbar:
        batch = loader.next_batch()
        ids, mask, dec, lbl, aux_t, aux_s = assemble_batch(batch, device)

        out  = dp_model(input_ids=ids, attention_mask=mask,
                        aux_trend=aux_t, aux_seasonal=aux_s,
                        decoder_input_ids=dec, labels=lbl)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step(); scheduler.step()
        running += loss.item()

        if step % LOG_EVERY == 0 and IS_MAIN:
            avg    = running / LOG_EVERY; running = 0.0
            lr_now = scheduler.get_last_lr()[0]
            loss_steps.append(step); losses.append(avg)
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")
            tqdm.write(f"  step {step:>6,}/{N_STEPS:,}  loss={avg:.4f}  lr={lr_now:.2e}")

        if step % EVAL_EVERY == 0:
            if IS_DDP:
                dist.barrier()
            if IS_MAIN:
                fast_mae = evaluate_fast(model, pipeline, eval_rows, device)
                eval_steps.append(step); fast_maes.append(fast_mae)
                bl_str = (f"  (baseline mean={baseline_mae['mean_mae']:.4f})"
                          if baseline_mae.get("mean_mae") else "")
                tqdm.write(f"  [eval] step={step:,}  FAST MAE={fast_mae:.4f}{bl_str}")

                if fast_mae < best_fast_mae - EARLY_STOP_MIN_DELTA:
                    best_fast_mae = fast_mae; no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
                    tqdm.write(f"  [early stop] No improvement  "
                               f"{no_improve_cnt}/{EARLY_STOP_PATIENCE}"
                               f"  (best={best_fast_mae:.4f})")
                    if no_improve_cnt >= EARLY_STOP_PATIENCE:
                        tqdm.write(f"  [early stop] Triggered at step {step:,}.")
                        stopped_early = True

            if IS_DDP:
                stop_tensor = torch.tensor(
                    [1 if stopped_early else 0], dtype=torch.long, device=device)
                dist.broadcast(stop_tensor, src=0)
                if not IS_MAIN:
                    stopped_early = bool(stop_tensor.item())
                dist.barrier()

            model.train()
            if stopped_early:
                break

    pbar.close()
    if IS_MAIN and stopped_early:
        print(f"\nEarly stop at step {step:,}  (best FAST MAE={best_fast_mae:.4f})")

    if IS_MAIN:
        print(f"\nSaving checkpoint → {CKPT_DIR}")
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        model.inner.save_pretrained(str(CKPT_DIR))
        torch.save(model.aux_module.state_dict(), str(CKPT_DIR / "aux_module.pt"))
        print(f"  aux_module.pt saved")
        for name, arr in [
            ("loss_steps", loss_steps), ("losses",    losses),
            ("eval_steps", eval_steps), ("fast_maes", fast_maes),
        ]:
            np.save(str(CKPT_DIR / f"{name}.npy"), np.array(arr))

        print("\nGenerating plots...")
        model.eval()
        plot_loss(loss_steps, losses, eval_steps, fast_maes, baseline_mae,
                  path=FIGURES_DIR / "loss.png", job_id=job_id,
                  freeze_backbone=freeze_backbone)
        plot_eval_samples(model, pipeline, eval_rows, device,
                          path=FIGURES_DIR / "eval_samples.png",
                          job_id=job_id, freeze_backbone=freeze_backbone)
        print(f"\nDone.  Figures → {FIGURES_DIR}/")
        if fast_maes:
            print(f"Final FAST MAE : {fast_maes[-1]:.4f}")
            if baseline_mae.get("mean_mae"):
                print(f"FAST retention : {fast_maes[-1]/baseline_mae['mean_mae']:.3f}x pretrained mean MAE")

    if IS_DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
