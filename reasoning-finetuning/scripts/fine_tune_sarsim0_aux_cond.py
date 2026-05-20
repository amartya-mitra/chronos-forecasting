#!/usr/bin/env python3
"""
SarSim0 fine-tuning with Parallel Auxiliary Conditioning.

Mechanism: precomputed trend and seasonal arrays (ground-truth from SarSim0)
are projected into the T5 d_model space and *appended* to the encoder output
before the decoder's cross-attention.  The decoder attends to them at every
layer without any output budget cost — only forecast tokens are generated.

Dataset: sarsim0-10k.arrow  (reasoning-mode rows only, ~6000 samples)
  • context  : 512 raw signal values
  • forecast : 64 raw signal values  (ground-truth future)
  • trend    : 64 raw signal values  (GT over forecast window)
  • seasonal : 64 raw signal values  (GT over forecast window)

AuxiliaryConditioningModule (new params only — 66,560 total):
  trend_proj   : Linear(DECOMP_LEN=64, D_MODEL=512) + Tanh  →  (B, 1, 512)
  seasonal_proj: Linear(DECOMP_LEN=64, D_MODEL=512) + Tanh  →  (B, 1, 512)
  concat → augmented_encoder_output shape (B, L+2, 512)

Training phases:
  Phase 1 (--freeze-backbone): only aux_module trained  → isolates conditioning
  Phase 2 (full finetune):     all params trained        → optional follow-up

Early stop: FAST MAE (original units) — since phase 1 has no TSF head, we
track the held-out FAST MAE as the primary signal.  A lower FAST MAE indicates
the conditioning is improving forecast quality.

Checkpointing:
  checkpoints/finetune-sarsim0-auxcond-{job_id}/
    config.json + pytorch_model.bin   ← inner T5 (save_pretrained)
    aux_module.pt                     ← AuxiliaryConditioningModule state_dict

Figures:
  figures/finetunes/finetune_sarsim0_auxcond_{job_id}/
    config.txt
    loss.png          — 1×3: training CE loss / FAST MAE / ablation trace
    eval_samples.png  — 5 samples: context | GT forecast | aux-cond prediction

Usage (single GPU via sbatch --wrap):
  /opt/conda/bin/python3.12 reasoning-finetuning/scripts/fine_tune_sarsim0_aux_cond.py \\
      --job-id $SLURM_JOB_ID [--freeze-backbone] [--fast-ratio 0.0]

Usage (multi-GPU via torchrun):
  torchrun --nproc_per_node=2 --master_port=29504 \\
      reasoning-finetuning/scripts/fine_tune_sarsim0_aux_cond.py \\
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
from transformers import get_cosine_schedule_with_warmup, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput

from chronos import ChronosPipeline

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = FINETUNING_ROOT / "data"    / "sarsim0-10k.arrow"
BASELINE_REF = FINETUNING_ROOT / "figures" / "evals" / "eval_pretrained_sarsim0" / "baseline_mae.txt"
MODEL_ID     = "amazon/chronos-t5-small"

# ── dataset constants ─────────────────────────────────────────────────────────
CONTEXT_LEN   = 512
HORIZON       = 64    # forecast length
DECOMP_LEN    = 64    # trend / seasonal length (= HORIZON for SarSim0)
D_MODEL       = 512   # T5-small hidden dimension

# ── special / extended vocab ──────────────────────────────────────────────────
# FAST mode start token — unchanged from joint training convention.
# No TSF tokens are needed here; only FAST-mode decoding is used at eval time.
FAST_TOKEN_ID  = 4096
EXTENDED_VOCAB = 4105   # keep vocab consistent with joint runs

# ── hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 64      # smaller than joint runs (two extra encoder tokens per sample)
N_STEPS      = 10_000
WARMUP_STEPS = 500
LR_AUX       = 1e-3   # auxiliary module — higher LR, starting from scratch
LR_BACKBONE  = 1e-4   # T5 backbone — used only when backbone is unfrozen
GRAD_CLIP    = 1.0
EVAL_EVERY   = 500
EVAL_BATCH   = 32
EVAL_SAMPLES = 20
EVAL_N_SHOW  = 5
EVAL_POOL    = 100

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


# ── Auxiliary conditioning module ─────────────────────────────────────────────

class AuxiliaryConditioningModule(nn.Module):
    """
    Projects precomputed trend and seasonal vectors into the T5 encoder hidden
    dimension so they can be appended to the encoder output as two extra
    cross-attention key-value pairs.

    Architecture (per component):
        (B, DECOMP_LEN=64)  →  Linear(64, 512)  →  Tanh  →  (B, 512)

    Combined output: (B, 2, D_MODEL) — ready to cat with encoder_hidden_states.

    Both projections are the ONLY new parameters trained in Phase 1.
    Total: 2 × (64×512 + 512) = 66,560 parameters.
    """

    def __init__(self, decomp_len: int = DECOMP_LEN, d_model: int = D_MODEL):
        super().__init__()
        self.trend_proj   = nn.Linear(decomp_len, d_model)
        self.seasonal_proj = nn.Linear(decomp_len, d_model)
        self.act = nn.Tanh()

    def forward(
        self,
        trend:    torch.Tensor,  # (B, DECOMP_LEN) — scale-normalised
        seasonal: torch.Tensor,  # (B, DECOMP_LEN) — scale-normalised
    ) -> torch.Tensor:           # (B, 2, D_MODEL)
        t = self.act(self.trend_proj(trend))       # (B, D_MODEL)
        s = self.act(self.seasonal_proj(seasonal)) # (B, D_MODEL)
        return torch.stack([t, s], dim=1)          # (B, 2, D_MODEL)


class ChronosWithAuxCond(nn.Module):
    """
    Wraps the inner T5ForConditionalGeneration (pipeline.model.model) to inject
    auxiliary conditioning tokens into the encoder cross-attention memory.

    Training forward:
        1. Run T5 encoder → encoder_hidden_states (B, L, D)
        2. Project trend/seasonal → aux_tokens (B, 2, D)
        3. Augment: cat([enc_hidden, aux_tokens], dim=1) → (B, L+2, D)
        4. Extend attention mask → (B, L+2)
        5. Pass augmented encoder output + extended mask to decoder.

    Inference (generate):
        Steps 1–4 identical; then pass augmented encoder_outputs directly to
        T5.generate() via the encoder_outputs= argument.  HuggingFace skips
        re-running the encoder when encoder_outputs is provided.

    NOTE: self.inner is T5ForConditionalGeneration (not ChronosPipeline).
    Access pattern: pipeline.model.model → self.inner.
    """

    def __init__(self, inner, decomp_len: int = DECOMP_LEN, d_model: int = D_MODEL):
        super().__init__()
        # inner = T5ForConditionalGeneration (pipeline.model.model)
        assert hasattr(inner, "encoder"), (
            "ChronosWithAuxCond expects T5ForConditionalGeneration "
            f"(pipeline.model.model), got {type(inner)}"
        )
        self.inner      = inner
        self.aux_module = AuxiliaryConditioningModule(decomp_len, d_model)

    def _augment_encoder(
        self,
        input_ids:      torch.Tensor,  # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        aux_trend:      torch.Tensor,  # (B, DECOMP_LEN) — scale-normalised
        aux_seasonal:   torch.Tensor,  # (B, DECOMP_LEN) — scale-normalised
    ):
        """
        Returns (augmented_encoder_outputs, augmented_attention_mask).
        Shape: BaseModelOutput with last_hidden_state (B, L+2, D), mask (B, L+2).
        """
        # Step 1: run the T5 encoder
        enc_out = self.inner.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # last_hidden_state: (B, L, D) — dtype follows backbone (bfloat16)

        # Step 2: compute aux tokens (B, 2, D).
        # Cast aux inputs to match the encoder hidden state dtype (bfloat16 in practice)
        # before they pass through aux_module projections, preventing dtype mismatch.
        enc_dtype = enc_out.last_hidden_state.dtype
        aux_tokens = self.aux_module(
            aux_trend.to(enc_dtype),
            aux_seasonal.to(enc_dtype),
        )  # (B, 2, D) — same dtype as encoder output

        # Step 3: augment encoder hidden states
        aug_hidden = torch.cat([enc_out.last_hidden_state, aux_tokens], dim=1)  # (B, L+2, D)

        # Step 4: extend attention mask to cover aux positions (always attended to)
        aux_mask   = torch.ones(
            attention_mask.size(0), 2,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        aug_mask   = torch.cat([attention_mask, aux_mask], dim=1)  # (B, L+2)

        aug_enc_out = BaseModelOutput(
            last_hidden_state=aug_hidden,
            hidden_states=None,
            attentions=None,
        )
        return aug_enc_out, aug_mask

    def forward(
        self,
        input_ids:           torch.Tensor,
        attention_mask:      torch.Tensor,
        aux_trend:           torch.Tensor,
        aux_seasonal:        torch.Tensor,
        decoder_input_ids:   torch.Tensor = None,
        labels:              torch.Tensor = None,
    ):
        aug_enc_out, aug_mask = self._augment_encoder(
            input_ids, attention_mask, aux_trend, aux_seasonal
        )
        return self.inner(
            attention_mask=aug_mask,
            encoder_outputs=aug_enc_out,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    @torch.no_grad()
    def generate_with_aux(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        aux_trend:      torch.Tensor,
        aux_seasonal:   torch.Tensor,
        **gen_kwargs,
    ) -> torch.Tensor:
        """
        Autoregressive generation with augmented cross-attention memory.
        Passes encoder_outputs= to model.generate() so the encoder is NOT
        re-run internally — augmented hidden states are used throughout.
        Also passes the extended attention_mask (B, L+2) so aux tokens are
        visible to the decoder at every generation step.
        """
        aug_enc_out, aug_mask = self._augment_encoder(
            input_ids, attention_mask, aux_trend, aux_seasonal
        )
        return self.inner.generate(
            encoder_outputs=aug_enc_out,
            attention_mask=aug_mask,   # extended mask — critical, do NOT use original
            **gen_kwargs,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def init_mode_embeddings(inner) -> None:
    """Initialise extended vocab tokens (4096–4104) as EOS + noise."""
    w = inner.get_input_embeddings().weight
    noise_std = EMBED_INIT_STD_MULTIPLIER * w[:4096].std().item()
    with torch.no_grad():
        base = w[1].clone()   # EOS token
        gen  = torch.Generator(); gen.manual_seed(SEED)
        for tok_id in range(4096, EXTENDED_VOCAB):
            w[tok_id] = base + noise_std * torch.randn(
                w.shape[1], generator=gen, dtype=w.dtype
            )
    print(f"  Embed init : eos_plus_noise  noise_std={noise_std:.5f}")
    cos = F.cosine_similarity(
        w[FAST_TOKEN_ID].float().unsqueeze(0), w[1].float().unsqueeze(0)
    ).item()
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


def write_config(
    path: Path, job_id: str, freeze_backbone: bool, fast_ratio: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    phase = "1 (aux only, backbone frozen)" if freeze_backbone else "2 (full finetune)"
    path.write_text("\n".join([
        "Experiment Configuration", "=" * 40,
        f"Job ID         : {job_id}",
        f"Script         : fine_tune_sarsim0_aux_cond.py",
        f"Model          : {MODEL_ID}",
        f"Dataset        : sarsim0-10k.arrow  (reasoning-mode rows only)",
        f"Phase          : {phase}",
        f"LR_aux         : {LR_AUX}  |  LR_backbone : {LR_BACKBONE}",
        f"Batch size     : {BATCH_SIZE}",
        f"Steps          : {N_STEPS:,}  |  warmup : {WARMUP_STEPS}",
        f"Ctx len        : {CONTEXT_LEN}  |  Horizon : {HORIZON}",
        f"Decomp len     : {DECOMP_LEN}  (trend + seasonal, GT from SarSim0)",
        f"Aux tokens     : 2 (appended to encoder output → L+2 cross-attn memory)",
        f"Aux params     : 66,560  (Linear×2 + Tanh, no bias removed)",
        f"Vocab          : extended to {EXTENDED_VOCAB}  (consistent with joint runs)",
        f"Early stop     : patience={EARLY_STOP_PATIENCE}  delta={EARLY_STOP_MIN_DELTA}  (FAST MAE)",
        "",
        "Design: trend/seasonal injected into decoder cross-attention memory.",
        "No decomposition tokens in output. Forecast tokens only.",
        f"Embed init     : eos_plus_noise  (multiplier={EMBED_INIT_STD_MULTIPLIER})",
    ]) + "\n")
    print(f"  Config written: {path.name}")


# ── data ──────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> tuple[list, list]:
    """
    Load reasoning-mode rows from sarsim0-10k.arrow.
    Returns (train_rows, eval_rows) after shuffle.
    Each row: {item_id, context(512,), forecast(64,), trend(64,), seasonal(64,)}
    All arrays are in raw signal units (not context-scale-normalised).
    """
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()

    item_ids  = table.column("item_id").to_pylist()
    modes     = table.column("mode").to_pylist()
    contexts  = table.column("context").to_pylist()
    forecasts = table.column("forecast").to_pylist()
    trends    = table.column("trend").to_pylist()
    seasonals = table.column("seasonal").to_pylist()

    rows = []
    for item_id, mode, ctx, fct, trnd, seas in zip(
        item_ids, modes, contexts, forecasts, trends, seasonals
    ):
        if mode != "reasoning":
            continue   # only reasoning rows carry trend/seasonal
        rows.append({
            "item_id":  item_id,
            "context":  np.array(ctx,  dtype=np.float32),   # (512,)
            "forecast": np.array(fct,  dtype=np.float32),   # (64,)  raw units
            "trend":    np.array(trnd, dtype=np.float32),   # (64,)  raw units
            "seasonal": np.array(seas, dtype=np.float32),   # (64,)  raw units
        })

    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[EVAL_POOL:], rows[:EVAL_POOL]   # (train, eval)


def pretokenise(rows: list, tokenizer) -> list:
    """
    Tokenise contexts and forecast targets for FAST-mode training.

    Scale-normalise trend and seasonal by context_scale *per sample* here
    (inside the batch tokenisation) so they live in the same numerical space
    as the tokenised context bins.

      labels        = [forecast_tokens(64) | EOS]
      dec_input_ids = [FAST_TOKEN(4096)    | forecast_tokens(64)]
      aux_trend     = trend   / context_scale   (64,)  per sample
      aux_seasonal  = seasonal / context_scale  (64,)  per sample
    """
    N = len(rows)

    contexts  = torch.tensor(np.stack([r["context"]  for r in rows]), dtype=torch.float32)
    forecasts = torch.tensor(np.stack([r["forecast"] for r in rows]), dtype=torch.float32)
    trends    = np.stack([r["trend"]    for r in rows])   # (N, 64) raw
    seasonals = np.stack([r["seasonal"] for r in rows])   # (N, 64) raw

    # Tokenise context + forecast
    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    label_ids, label_mask, _    = tokenizer._input_transform(
        context=forecasts, scale=scale
    )

    # Build labels and decoder input ids (FAST mode)
    eos_col  = torch.full((N, 1), tokenizer.config.eos_token_id, dtype=label_ids.dtype)
    ones_col = torch.ones((N, 1), dtype=label_ids.dtype)

    labels_seq    = torch.cat([label_ids, eos_col],   dim=1)   # (N, 65)
    mask_seq      = torch.cat([label_mask, ones_col], dim=1)   # (N, 65)
    labels        = labels_seq.masked_fill(~mask_seq.bool(), -100)
    mode_col      = torch.full((N, 1), FAST_TOKEN_ID, dtype=token_ids.dtype)
    dec_input_ids = torch.cat([mode_col, labels_seq[:, :-1]], dim=1)  # (N, 65)

    # Scale-normalise aux arrays: divide by context_scale per sample.
    # scale from tokenizer is shape (N,) — reshape to (N,1) for broadcasting against (N,64).
    scale_np   = scale.numpy().reshape(-1, 1)        # (N, 1)
    aux_trend  = torch.tensor(
        trends / np.maximum(scale_np, SCALE_EPS), dtype=torch.float32
    )   # (N, 64)
    aux_seasonal = torch.tensor(
        seasonals / np.maximum(scale_np, SCALE_EPS), dtype=torch.float32
    )   # (N, 64)

    return [{
        "item_id":       rows[i]["item_id"],
        "context":       rows[i]["context"],
        "forecast":      rows[i]["forecast"],
        "trend":         rows[i]["trend"],
        "seasonal":      rows[i]["seasonal"],
        "input_ids":     token_ids[i],
        "attn_mask":     attn_mask[i],
        "dec_input_ids": dec_input_ids[i],
        "labels":        labels[i],
        "aux_trend":     aux_trend[i],
        "aux_seasonal":  aux_seasonal[i],
        "scale":         scale[i],          # keep for eval denormalisation
    } for i in range(N)]


class CyclicLoader:
    """Infinite cyclic dataloader with per-epoch shuffle."""
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
    """Stack pre-tokenised tensors onto device."""
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
    model:     ChronosWithAuxCond,
    pipeline,
    contexts:  list,      # list of (512,) np.float32 arrays — raw units
    trends:    list,      # list of (64,)  np.float32 arrays — raw units
    seasonals: list,      # list of (64,)  np.float32 arrays — raw units
    device,
) -> np.ndarray:
    """
    FAST-mode generation with auxiliary conditioning.
    Returns (B, HORIZON) array in original signal units.
    """
    B  = len(contexts)
    mc = pipeline.model.config

    ctx  = torch.tensor(np.stack(contexts), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)

    # Scale-normalise aux arrays using per-sample context_scale.
    # scale from tokenizer is shape (B,) — reshape to (B,1) for broadcasting against (B,64).
    scale_np     = scale.numpy().reshape(-1, 1)   # (B, 1)
    aux_trend    = torch.tensor(
        np.stack(trends)    / np.maximum(scale_np, SCALE_EPS), dtype=torch.float32
    ).to(device)
    aux_seasonal = torch.tensor(
        np.stack(seasonals) / np.maximum(scale_np, SCALE_EPS), dtype=torch.float32
    ).to(device)

    raw = model.generate_with_aux(
        input_ids=tids.to(device),
        attention_mask=amask.to(device),
        aux_trend=aux_trend,
        aux_seasonal=aux_seasonal,
        generation_config=GenerationConfig(
            min_new_tokens=HORIZON,
            max_new_tokens=HORIZON,
            do_sample=True,
            num_return_sequences=EVAL_SAMPLES,
            decoder_start_token_id=FAST_TOKEN_ID,
            eos_token_id=mc.eos_token_id,
            pad_token_id=mc.pad_token_id,
            temperature=mc.temperature,
            top_k=mc.top_k,
            top_p=mc.top_p,
        ),
    )
    raw = raw[:, 1:]   # strip decoder start token
    raw_bins = raw[:, :HORIZON].reshape(B, EVAL_SAMPLES, HORIZON)
    return pipeline.tokenizer.output_transform(raw_bins.cpu(), scale).median(dim=1).values.numpy()


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_fast(
    model:     ChronosWithAuxCond,
    pipeline,
    eval_rows: list,
    device,
) -> float:
    """
    Original-unit MAE of FAST+aux predictions vs ground-truth forecast.
    eval_rows contain raw-unit forecast, trend, seasonal arrays.
    """
    model.eval()
    maes = []
    for i in range(0, len(eval_rows), EVAL_BATCH):
        batch = eval_rows[i:i + EVAL_BATCH]
        preds = predict_fast_with_aux(
            model, pipeline,
            contexts=  [r["context"]  for r in batch],
            trends=    [r["trend"]    for r in batch],
            seasonals= [r["seasonal"] for r in batch],
            device=device,
        )
        for r, pred in zip(batch, preds):
            maes.append(float(np.mean(np.abs(pred - r["forecast"]))))
    model.train()
    return float(np.mean(maes))


def check_gradients(model: ChronosWithAuxCond) -> None:
    """Verification check: print grad status for aux module weights."""
    for name, p in model.aux_module.named_parameters():
        print(f"  grad check  {name}: grad={'OK' if p.grad is not None else 'NONE'}")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_loss(
    loss_steps, losses, eval_steps, fast_maes,
    baseline_mae: dict, path: Path,
    job_id: str, freeze_backbone: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

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
    phase_tag = "Phase 1 (frozen backbone)" if freeze_backbone else "Phase 2 (full)"
    ax.set_title(f"Eval MAE — FAST+Aux  [{phase_tag}]", fontweight="bold")
    ax.set_xlabel("Step"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"SarSim0 Aux Conditioning  ·  chronos-t5-small  ·  job {job_id}  ·  {phase_tag}",
        fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_eval_samples(
    model:     ChronosWithAuxCond,
    pipeline,
    eval_rows: list,
    device,
    path:      Path,
    job_id:    str,
    freeze_backbone: bool,
    n: int = EVAL_N_SHOW,
) -> None:
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(eval_rows), size=min(n, len(eval_rows)), replace=False))
    sel   = [eval_rows[i] for i in picks]
    model.eval()
    preds = predict_fast_with_aux(
        model, pipeline,
        contexts=  [r["context"]  for r in sel],
        trends=    [r["trend"]    for r in sel],
        seasonals= [r["seasonal"] for r in sel],
        device=device,
    )
    model.train()

    ctx_x = np.arange(CONTEXT_LEN)
    fct_x = np.arange(CONTEXT_LEN, CONTEXT_LEN + HORIZON)
    dec_x = np.arange(fct_x[0], fct_x[0] + DECOMP_LEN)

    fig, axes_raw = plt.subplots(n, 3, figsize=(21, 3 * n))
    axes = np.asarray(axes_raw)

    phase_tag = "Phase 1 (aux only)" if freeze_backbone else "Phase 2 (full)"
    fig.suptitle(
        f"SarSim0 Aux Cond · job {job_id} · {phase_tag}\n"
        f"Left: context+forecast  |  Centre: GT trend  |  Right: GT seasonal",
        fontsize=11, fontweight="bold",
    )

    for idx, (row, pred) in enumerate(zip(sel, preds)):
        mae = float(np.mean(np.abs(pred - row["forecast"])))

        # col 0: context + GT forecast + prediction
        ax = axes[idx, 0]
        ax.fill_between(ctx_x, row["context"], alpha=0.06, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL, lw=0.6,
                label="Context" if idx == 0 else "_")
        ax.plot(fct_x, row["forecast"], color=_GT_COL, lw=1.8,
                label="GT forecast" if idx == 0 else "_")
        ax.plot(fct_x, pred, color=_PRED_COL, lw=1.4, ls="--", alpha=0.85,
                label=f"Prediction (MAE={mae:.3f})" if idx == 0 else "_")
        ax.axvline(CONTEXT_LEN - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(f"{row['item_id']}  MAE={mae:.4f}", fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

        # col 1: GT trend (aux input) over forecast window; ref = deseasonalised signal
        ax = axes[idx, 1]
        ax.plot(dec_x, row["forecast"] - row["seasonal"], color=_GT_COL, lw=0.9, alpha=0.3, ls="--",
                label="GT deseasonalised (ref)" if idx == 0 else "_")
        ax.plot(dec_x, row["trend"], color=_TRND_COL, lw=1.8,
                label="GT trend (aux)" if idx == 0 else "_")
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

        # col 2: GT seasonal (aux input) over forecast window; ref = detrended signal
        ax = axes[idx, 2]
        ax.plot(dec_x, row["forecast"] - row["trend"], color=_GT_COL, lw=0.9, alpha=0.3, ls="--",
                label="GT detrended (ref)" if idx == 0 else "_")
        ax.plot(dec_x, row["seasonal"], color=_SEAS_COL, lw=1.8,
                label="GT seasonal (aux)" if idx == 0 else "_")
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

    for col in range(3):
        axes[-1, col].set_xlabel("Time step", fontsize=9)
    axes[0, 0].set_title("Context + Forecast", fontsize=9, fontweight="bold")
    axes[0, 1].set_title("GT Trend (aux input)", fontsize=9, fontweight="bold")
    axes[0, 2].set_title("GT Seasonal (aux input)", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id",          default="nojobid")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Phase 1: freeze T5 weights, train aux_module only.")
    args            = parser.parse_args()
    job_id          = args.job_id
    freeze_backbone = args.freeze_backbone

    FIGURES_DIR = (
        FINETUNING_ROOT / "figures"     / "finetunes"
        / f"finetune_sarsim0_auxcond_{job_id}"
    )
    CKPT_DIR = (
        FINETUNING_ROOT / "checkpoints"
        / f"finetune-sarsim0-auxcond-{job_id}"
    )

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
        print("SarSim0 Aux Conditioning  ·  chronos-t5-small")
        print(f"  Job ID         : {job_id}")
        print(f"  Phase          : {phase_tag}")
        print(f"  Device         : {device}  |  DDP={IS_DDP}  world={WORLD_SIZE}")
        print(f"  Steps          : {N_STEPS:,}  |  batch={BATCH_SIZE}")
        print(f"  LR_aux         : {LR_AUX}  |  LR_backbone : {LR_BACKBONE}")
        if baseline_mae["mean_mae"] is not None:
            print(f"  Baseline MAE   : mean={baseline_mae['mean_mae']:.4f}"
                  f"  median={baseline_mae['median_mae']:.4f}  (pretrained)")
        print("=" * 65)
        write_config(FIGURES_DIR / "config.txt", job_id, freeze_backbone, fast_ratio=0.0)
    else:
        baseline_mae = {}

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"\n[rank {LOCAL_RANK if IS_DDP else 0}] Loading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    # pipeline.model       → ChronosModel  (wraps T5ForConditionalGeneration)
    # pipeline.model.model → T5ForConditionalGeneration  ← this is our `inner`
    inner = pipeline.model.model
    if inner.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
    init_mode_embeddings(inner)

    # Wrap with auxiliary conditioning
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
        print("\nLoading dataset (reasoning-mode rows only) ...")
    train_rows, eval_rows = load_rows(DATA_PATH)
    if IS_MAIN:
        print(f"  Train : {len(train_rows):,}  |  Eval : {len(eval_rows):,}")
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
    # The T5 backbone is bfloat16 (loaded with dtype=torch.bfloat16).
    # aux_module initialises in float32 by default. Cast it explicitly to match,
    # so the Linear weight dtype matches the bfloat16 encoder hidden states.
    model.aux_module = model.aux_module.to(torch.bfloat16)
    model.train()

    if IS_DDP:
        dp_model = DDP(model, device_ids=[LOCAL_RANK])
        if IS_MAIN:
            print(f"  DDP across {WORLD_SIZE} GPUs")
    else:
        dp_model = model

    # Two param groups: aux_module at LR_AUX, backbone at LR_BACKBONE (or frozen)
    param_groups = [
        {"params": model.aux_module.parameters(), "lr": LR_AUX},
    ]
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
    grad_checked   = False   # run gradient check once after first backward

    if IS_MAIN:
        print("\nTraining...")
    pbar = tqdm(range(1, N_STEPS + 1), total=N_STEPS,
                desc="Training (SarSim0 AuxCond)", file=sys.stdout,
                dynamic_ncols=True, smoothing=0.05,
                miniters=LOG_EVERY, mininterval=0.0, disable=not IS_MAIN)

    for step in pbar:
        batch = loader.next_batch()
        ids, mask, dec, lbl, aux_t, aux_s = assemble_batch(batch, device)

        out  = dp_model(
            input_ids=ids, attention_mask=mask,
            aux_trend=aux_t, aux_seasonal=aux_s,
            decoder_input_ids=dec, labels=lbl,
        )
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()

        # ── gradient check (once, after first backward) ──────────────────────
        if IS_MAIN and not grad_checked:
            check_gradients(model)
            grad_checked = True

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step(); scheduler.step()
        running += loss.item()

        if step % LOG_EVERY == 0 and IS_MAIN:
            avg    = running / LOG_EVERY; running = 0.0
            lr_now = scheduler.get_last_lr()[0]
            loss_steps.append(step); losses.append(avg)
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")
            tqdm.write(
                f"  step {step:>6,}/{N_STEPS:,}  loss={avg:.4f}  lr={lr_now:.2e}"
            )

        if step % EVAL_EVERY == 0:
            if IS_DDP:
                dist.barrier()
            if IS_MAIN:
                fast_mae = evaluate_fast(model, pipeline, eval_rows, device)
                eval_steps.append(step)
                fast_maes.append(fast_mae)
                bl_str = (
                    f"  (baseline mean={baseline_mae['mean_mae']:.4f})"
                    if baseline_mae.get("mean_mae") else ""
                )
                tqdm.write(
                    f"  [eval] step={step:,}  FAST MAE={fast_mae:.4f}{bl_str}"
                )

                if fast_mae < best_fast_mae - EARLY_STOP_MIN_DELTA:
                    best_fast_mae  = fast_mae
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
                    tqdm.write(
                        f"  [early stop] No FAST improvement  "
                        f"{no_improve_cnt}/{EARLY_STOP_PATIENCE}"
                        f"  (best={best_fast_mae:.4f})"
                    )
                    if no_improve_cnt >= EARLY_STOP_PATIENCE:
                        tqdm.write(
                            f"  [early stop] Triggered at step {step:,}."
                            f"  Best FAST MAE={best_fast_mae:.4f}"
                        )
                        stopped_early = True

            if IS_DDP:
                stop_tensor = torch.tensor(
                    [1 if stopped_early else 0], dtype=torch.long, device=device
                )
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
        # Save T5 backbone (even if frozen, so checkpoint is self-contained)
        model.inner.save_pretrained(str(CKPT_DIR))
        # Save aux module separately for easy re-attachment
        torch.save(model.aux_module.state_dict(), str(CKPT_DIR / "aux_module.pt"))
        print(f"  aux_module.pt saved  ({sum(p.numel() for p in model.aux_module.parameters()):,} params)")
        for name, arr in [
            ("loss_steps", loss_steps), ("losses",    losses),
            ("eval_steps", eval_steps), ("fast_maes", fast_maes),
        ]:
            np.save(str(CKPT_DIR / f"{name}.npy"), np.array(arr))

        print("\nGenerating plots...")
        model.eval()
        plot_loss(
            loss_steps, losses, eval_steps, fast_maes,
            baseline_mae,
            path=FIGURES_DIR / "loss.png",
            job_id=job_id, freeze_backbone=freeze_backbone,
        )
        plot_eval_samples(
            model, pipeline, eval_rows, device,
            path=FIGURES_DIR / "eval_samples.png",
            job_id=job_id, freeze_backbone=freeze_backbone,
        )

        print(f"\nDone.  Figures → {FIGURES_DIR}/")
        if fast_maes:
            print(f"Final FAST MAE : {fast_maes[-1]:.4f}")
            if baseline_mae.get("mean_mae"):
                ratio = fast_maes[-1] / baseline_mae["mean_mae"]
                print(f"FAST retention : {ratio:.3f}x pretrained mean MAE")

    if IS_DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
