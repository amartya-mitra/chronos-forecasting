#!/usr/bin/env python3
"""
Joint TSF + FAST fine-tuning on SarSim0 data.

For each training step a batch is drawn randomly from either:
  • TSF  mode (token 4102): Trend(64) | SEP_1 | Seasonal(64) | SEP_2 | Forecast(64)  → gen_len=194
  • FAST mode (token 4096): direct 64-step forecast, no SEPs                           → gen_len=64

The split is controlled by --fast-ratio (default 0.20 = 20% FAST batches).
All rows within a batch share the same mode (homogeneous batches).

Objectives
----------
  TSF rows  : teach trend / seasonal / forecast decomposition
  FAST rows : anchor zero-shot forecasting ability (prevent catastrophic forgetting)

Evaluation (every EVAL_EVERY steps, rank 0 only)
-------------------------------------------------
  TSF MAE  : normalised MAE vs 3-part stored target  (primary early-stop metric)
  FAST MAE : original-unit MAE vs ground-truth forecast, compared to pretrained baseline

Usage
-----
  torchrun --nproc_per_node=2 --master_port=29502 fine_tune_sarsim0_joint.py --job-id $SLURM_JOB_ID
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

from chronos import ChronosPipeline

# ── paths ───────────────────────────────────────────────────────────────────────
DATA_PATH    = FINETUNING_ROOT / "data"    / "sarsim0-tsf.arrow"
BASELINE_REF = FINETUNING_ROOT / "figures" / "evals" / "eval_pretrained_sarsim0" / "baseline_mae.txt"
MODEL_ID     = "amazon/chronos-t5-small"

# ── SarSim0 dimensions ──────────────────────────────────────────────────────────
CONTEXT_LEN = 512
HORIZON     = 64   # segment length for both modes

# ── mode configs ────────────────────────────────────────────────────────────────
TSF_CFG = {
    "token_id":    4102,
    "n_parts":     3,
    "segment_len": HORIZON,
    "ctx_len":     CONTEXT_LEN,
    "pred_len":    192,           # 3 × 64
    "gen_len":     194,           # 192 + 2 SEPs
    "forced":      [(65, 4103), (130, 4104)],
    "part_names":  ["Trend", "Seasonal", "Forecast"],
    "part_cols":   ["seagreen", "mediumpurple", "darkorange"],
}

FAST_CFG = {
    "token_id":    4096,
    "n_parts":     1,
    "segment_len": HORIZON,
    "ctx_len":     CONTEXT_LEN,
    "pred_len":    HORIZON,       # 64 direct forecast tokens
    "gen_len":     HORIZON,       # no SEPs
    "forced":      [],
    "part_names":  ["Forecast"],
    "part_cols":   ["darkorange"],
}

# ── special tokens ──────────────────────────────────────────────────────────────
SEP_1_TOKEN_ID = 4103
SEP_2_TOKEN_ID = 4104
EXTENDED_VOCAB = 4105

# ── hyperparameters ─────────────────────────────────────────────────────────────
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

EARLY_STOP_PATIENCE  = 5
EARLY_STOP_MIN_DELTA = 1e-4

EMBED_INIT_STD_MULTIPLIER = 2.0
SCALE_EPS = 1e-4
SEED      = 42

_CTX_COL  = "steelblue"
_PRED_COL = "crimson"


# ── helpers ─────────────────────────────────────────────────────────────────────

class SEPForcer(LogitsProcessor):
    """Force SEP tokens at fixed positions during autoregressive generation."""
    def __init__(self, forced: list):
        self._forced = {pos - 1: tok for pos, tok in forced}

    def __call__(self, input_ids, scores):
        cur = input_ids.shape[-1] - 1
        if cur in self._forced:
            scores[:] = -float("inf")
            scores[:, self._forced[cur]] = 0.0
        return scores


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def init_mode_embeddings(inner) -> None:
    """Initialise all new tokens (4096-4104) as EOS + noise."""
    w = inner.get_input_embeddings().weight
    noise_std = EMBED_INIT_STD_MULTIPLIER * w[:4096].std().item()
    with torch.no_grad():
        base = w[1].clone()   # EOS token
        gen  = torch.Generator(); gen.manual_seed(SEED)
        for tok_id in range(4096, EXTENDED_VOCAB):
            w[tok_id] = base + noise_std * torch.randn(w.shape[1], generator=gen, dtype=w.dtype)
    print(f"  Embed init : eos_plus_noise  noise_std={noise_std:.5f}")
    for name, tid in [("FAST", 4096), ("TSF", 4102), ("SEP_1", 4103), ("SEP_2", 4104)]:
        cos = F.cosine_similarity(w[tid].float().unsqueeze(0), w[1].float().unsqueeze(0)).item()
        print(f"  Cosine sim : {name}↔EOS = {cos:.3f}")


def load_baseline_mae() -> dict:
    ref = {"mean_mae": None, "median_mae": None}
    if BASELINE_REF.exists():
        for line in BASELINE_REF.read_text().splitlines():
            if line.startswith("mean_mae"):
                ref["mean_mae"]   = float(line.split(":")[-1].strip())
            elif line.startswith("median_mae"):
                ref["median_mae"] = float(line.split(":")[-1].strip())
    return ref


def write_config(path: Path, job_id: str, fast_ratio: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "Experiment Configuration", "=" * 40,
        f"Job ID      : {job_id}",
        f"Script      : fine_tune_sarsim0_joint.py",
        f"Model       : {MODEL_ID}",
        f"Dataset     : sarsim0-tsf.arrow",
        f"TSF token   : 4102  |  FAST token : 4096",
        f"Fast ratio  : {fast_ratio:.0%} FAST  /  {1-fast_ratio:.0%} TSF  (per batch)",
        f"Vocab       : extended to {EXTENDED_VOCAB}",
        f"Steps       : {N_STEPS:,}  |  batch={BATCH_SIZE}  |  LR={LR}",
        f"Ctx len     : {CONTEXT_LEN}  |  Horizon : {HORIZON}",
        f"TSF pred    : 192 tokens (3×64) + 2 SEPs → gen_len=194",
        f"FAST pred   : 64 tokens  (direct forecast, no SEPs)",
        f"Early stop  : patience={EARLY_STOP_PATIENCE}  min_delta={EARLY_STOP_MIN_DELTA}  (TSF MAE)",
        "",
        "Objective: TSF rows teach decomposition; FAST rows anchor zero-shot ability.",
        f"Embed init  : eos_plus_noise  (multiplier={EMBED_INIT_STD_MULTIPLIER})",
    ]) + "\n")
    print(f"  Config written: {path.name}")


# ── data ────────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> tuple[list, list]:
    """Load arrow, shuffle with SEED=42, return (train_rows, eval_rows)."""
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()

    item_ids  = table.column("item_id").to_pylist()
    contexts  = table.column("context").to_pylist()
    targets   = table.column("target").to_pylist()
    s_firsts  = table.column("s_first").to_pylist()
    s_seconds = table.column("s_second").to_pylist()
    s_thirds  = table.column("s_third").to_pylist()

    rows = []
    for item_id, ctx, tgt, s1, s2, s3 in zip(
        item_ids, contexts, targets, s_firsts, s_seconds, s_thirds
    ):
        rows.append({
            "item_id": item_id,
            "context": np.array(ctx, dtype=np.float32),
            "target":  np.array(tgt, dtype=np.float32),   # (192,) context-scale-normalised
            "s_parts": [float(s1), float(s2), float(s3)],
        })

    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[EVAL_POOL:], rows[:EVAL_POOL]   # train, eval


def pretokenise_tsf(rows: list, tokenizer) -> list:
    """
    Tokenise rows for TSF mode.
      labels        = [trend(64) | SEP_1 | seasonal(64) | SEP_2 | forecast(64) | EOS]
      dec_input_ids = [4102 | trend(64) | SEP_1 | seasonal(64) | SEP_2 | forecast(64)]
    """
    N   = len(rows)
    seg = HORIZON

    contexts = torch.tensor(np.stack([r["context"] for r in rows]), dtype=torch.float32)
    targets  = torch.tensor(np.stack([r["target"]  for r in rows]), dtype=torch.float32)

    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    label_ids, _,          _   = tokenizer._input_transform(context=targets, scale=scale)

    parts    = [label_ids[:, i*seg:(i+1)*seg] for i in range(3)]
    sep1_col = torch.full((N, 1), SEP_1_TOKEN_ID, dtype=label_ids.dtype)
    sep2_col = torch.full((N, 1), SEP_2_TOKEN_ID, dtype=label_ids.dtype)
    eos_col  = torch.full((N, 1), tokenizer.config.eos_token_id, dtype=label_ids.dtype)

    labels_seq    = torch.cat([parts[0], sep1_col, parts[1], sep2_col, parts[2], eos_col], dim=1)
    mask_seq      = torch.ones_like(labels_seq)
    labels        = labels_seq.masked_fill(~mask_seq.bool(), -100)
    mode_col      = torch.full((N, 1), TSF_CFG["token_id"], dtype=token_ids.dtype)
    dec_input_ids = torch.cat([mode_col, labels_seq[:, :-1]], dim=1)

    return [{
        "item_id":       rows[i]["item_id"],
        "context":       rows[i]["context"],
        "target":        rows[i]["target"],
        "s_parts":       rows[i]["s_parts"],
        "input_ids":     token_ids[i],
        "attn_mask":     attn_mask[i],
        "dec_input_ids": dec_input_ids[i],
        "labels":        labels[i],
    } for i in range(N)]


def pretokenise_fast(rows: list, tokenizer) -> list:
    """
    Tokenise rows for FAST mode.
      target        = forecast_norm only  (last 64 of the 192-token target)
      labels        = [forecast(64) | EOS]
      dec_input_ids = [4096 | forecast(64)]
    """
    N = len(rows)

    contexts     = torch.tensor(np.stack([r["context"]            for r in rows]), dtype=torch.float32)
    fast_targets = torch.tensor(np.stack([r["target"][2*HORIZON:] for r in rows]), dtype=torch.float32)

    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    label_ids, label_mask, _   = tokenizer._input_transform(context=fast_targets, scale=scale)

    ones    = torch.ones((N, 1), dtype=label_ids.dtype)
    eos_col = torch.full((N, 1), tokenizer.config.eos_token_id, dtype=label_ids.dtype)

    labels_seq    = torch.cat([label_ids, eos_col], dim=1)
    mask_seq      = torch.cat([label_mask, ones],   dim=1)
    labels        = labels_seq.masked_fill(~mask_seq.bool(), -100)
    mode_col      = torch.full((N, 1), FAST_CFG["token_id"], dtype=token_ids.dtype)
    dec_input_ids = torch.cat([mode_col, labels_seq[:, :-1]], dim=1)

    return [{
        "item_id":       rows[i]["item_id"],
        "context":       rows[i]["context"],
        "target":        rows[i]["target"],
        "s_parts":       rows[i]["s_parts"],
        "input_ids":     token_ids[i],
        "attn_mask":     attn_mask[i],
        "dec_input_ids": dec_input_ids[i],
        "labels":        labels[i],
    } for i in range(N)]


class CyclicLoader:
    def __init__(self, rows, batch_size, seed=0):
        self.rows, self.batch_size = rows, batch_size
        self.rng  = np.random.default_rng(seed)
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


# ── inference ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_tsf(pipeline, contexts: list, device) -> np.ndarray:
    """Generate TSF predictions; return (B, 192) normalised values."""
    gen_len   = TSF_CFG["gen_len"]
    seg       = HORIZON
    model_cfg = pipeline.model.config

    ctx  = torch.tensor(np.stack(contexts), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)

    lp  = LogitsProcessorList([SEPForcer(TSF_CFG["forced"])])
    raw = pipeline.model.model.generate(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        logits_processor=lp,
        generation_config=GenerationConfig(
            min_new_tokens=gen_len, max_new_tokens=gen_len,
            do_sample=True, num_return_sequences=EVAL_SAMPLES,
            decoder_start_token_id=TSF_CFG["token_id"],
            eos_token_id=model_cfg.eos_token_id,
            pad_token_id=model_cfg.pad_token_id,
            temperature=model_cfg.temperature,
            top_k=model_cfg.top_k, top_p=model_cfg.top_p,
        ),
    )
    B   = len(contexts)
    raw = raw[:, 1:]   # strip mode start token
    # Strip SEPs: [part0(64) | SEP_1(1) | part1(64) | SEP_2(1) | part2(64)]
    raw_bins = torch.cat([raw[:, :seg], raw[:, seg+1:2*seg+1], raw[:, 2*seg+2:]], dim=1)
    raw_bins = raw_bins.reshape(B, EVAL_SAMPLES, TSF_CFG["pred_len"])
    return pipeline.tokenizer.output_transform(raw_bins.cpu(), scale).median(dim=1).values.numpy()


@torch.no_grad()
def predict_fast(pipeline, contexts: list, device) -> np.ndarray:
    """Generate FAST predictions; return (B, 64) in original units."""
    model_cfg = pipeline.model.config

    ctx  = torch.tensor(np.stack(contexts), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)

    raw = pipeline.model.model.generate(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        generation_config=GenerationConfig(
            min_new_tokens=HORIZON, max_new_tokens=HORIZON,
            do_sample=True, num_return_sequences=EVAL_SAMPLES,
            decoder_start_token_id=FAST_CFG["token_id"],
            eos_token_id=model_cfg.eos_token_id,
            pad_token_id=model_cfg.pad_token_id,
            temperature=model_cfg.temperature,
            top_k=model_cfg.top_k, top_p=model_cfg.top_p,
        ),
    )
    B        = len(contexts)
    raw      = raw[:, 1:]
    raw_bins = raw[:, :HORIZON].reshape(B, EVAL_SAMPLES, HORIZON)
    return pipeline.tokenizer.output_transform(raw_bins.cpu(), scale).median(dim=1).values.numpy()


# ── evaluation ──────────────────────────────────────────────────────────────────

def evaluate_tsf(pipeline, eval_rows: list, device) -> float:
    """Normalised MAE of TSF predictions vs stored 3-part target."""
    pipeline.model.model.eval()
    maes = []
    for i in range(0, len(eval_rows), EVAL_BATCH):
        batch = eval_rows[i:i+EVAL_BATCH]
        preds = predict_tsf(pipeline, [r["context"] for r in batch], device)
        for r, pred in zip(batch, preds):
            maes.append(float(np.mean(np.abs(pred - r["target"]))))
    pipeline.model.model.train()
    return float(np.mean(maes))


def evaluate_fast(pipeline, eval_rows: list, device) -> float:
    """Original-unit MAE of FAST predictions vs denormalised ground-truth forecast."""
    pipeline.model.model.eval()
    maes = []
    for i in range(0, len(eval_rows), EVAL_BATCH):
        batch = eval_rows[i:i+EVAL_BATCH]
        preds = predict_fast(pipeline, [r["context"] for r in batch], device)
        for r, pred in zip(batch, preds):
            s_ctx = max(float(np.mean(np.abs(r["context"]))), SCALE_EPS)
            gt    = r["target"][2*HORIZON:] * (r["s_parts"][2] / s_ctx)
            maes.append(float(np.mean(np.abs(pred - gt))))
    pipeline.model.model.train()
    return float(np.mean(maes))


# ── plots ────────────────────────────────────────────────────────────────────────

def plot_loss(loss_steps, losses, eval_steps, tsf_maes, fast_maes,
              baseline_mae: dict, path: Path, job_id: str, fast_ratio: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    axes[0].plot(loss_steps, losses, color="steelblue", lw=0.8, alpha=0.6)
    if len(losses) >= 50:
        sm = np.convolve(losses, np.ones(50)/50, mode="valid")
        axes[0].plot(loss_steps[49:], sm, color="steelblue", lw=1.8, label="Loss (MA-50)")
    axes[0].set_title("Training Loss (CE)", fontweight="bold")
    axes[0].set_xlabel("Step"); axes[0].legend(fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(eval_steps, tsf_maes, color="darkorange", lw=1.8, marker="o", ms=3,
                 label="TSF MAE (normalised)")
    axes[1].set_title("Eval MAE — TSF Mode", fontweight="bold")
    axes[1].set_xlabel("Step"); axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].plot(eval_steps, fast_maes, color="seagreen", lw=1.8, marker="o", ms=3,
                 label="FAST MAE (original units)")
    if baseline_mae.get("mean_mae") is not None:
        axes[2].axhline(baseline_mae["mean_mae"], color="grey", lw=1.2, ls="--",
                        label=f"Pretrained mean  {baseline_mae['mean_mae']:.4f}")
    if baseline_mae.get("median_mae") is not None:
        axes[2].axhline(baseline_mae["median_mae"], color="black", lw=1.0, ls=":",
                        label=f"Pretrained median {baseline_mae['median_mae']:.4f}")
    axes[2].set_title("Eval MAE — FAST Mode  (retention check)", fontweight="bold")
    axes[2].set_xlabel("Step"); axes[2].legend(fontsize=8)
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Joint TSF+FAST  ·  chronos-t5-small  ·  SarSim0  ·  job {job_id}"
        f"  ·  fast_ratio={fast_ratio:.0%}",
        fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_eval_samples_tsf(pipeline, eval_rows: list, device, path: Path,
                          job_id: str, fast_ratio: float,
                          n: int = EVAL_N_SHOW) -> None:
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(eval_rows), size=min(n, len(eval_rows)), replace=False))
    sel   = [eval_rows[i] for i in picks]
    pipeline.model.model.eval()
    preds = predict_tsf(pipeline, [r["context"] for r in sel], device)
    pipeline.model.model.train()

    seg     = HORIZON
    ctx_x   = np.arange(CONTEXT_LEN)
    parts_x = [np.arange(CONTEXT_LEN + i*seg, CONTEXT_LEN + (i+1)*seg) for i in range(3)]

    fig, axes_raw = plt.subplots(n, 1, figsize=(18, 3*n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"Joint TSF_SarSim0  ·  job {job_id}  ·  fast_ratio={fast_ratio:.0%}\n"
        "seagreen=Trend  mediumpurple=Seasonal  darkorange=Forecast  crimson=Prediction",
        fontsize=11, fontweight="bold",
    )
    for idx, (row, pred) in enumerate(zip(sel, preds)):
        ax    = axes[idx]
        s_ctx = float(np.mean(np.abs(row["context"])))
        inv   = [s / s_ctx for s in row["s_parts"]]

        ax.fill_between(ctx_x, row["context"], alpha=0.05, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL, lw=0.5, label="Context")
        for i in range(3):
            tgt_i  = row["target"][i*seg:(i+1)*seg] * inv[i]
            pred_i = pred[i*seg:(i+1)*seg]           * inv[i]
            ax.plot(parts_x[i], tgt_i,  color=TSF_CFG["part_cols"][i], lw=1.8,
                    label=f"Target {TSF_CFG['part_names'][i]}")
            ax.plot(parts_x[i], pred_i, color=_PRED_COL, lw=1.4, ls="--", alpha=0.85)
        ax.axvline(CONTEXT_LEN - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        for i in range(1, 3):
            ax.axvline(CONTEXT_LEN + i*seg - 0.5, color="grey", lw=0.6, ls=":", alpha=0.5)
        mae_norm = float(np.mean(np.abs(pred - row["target"])))
        ax.set_title(f"{row['item_id']}  MAE(norm)={mae_norm:.4f}", fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.7)
    axes[-1].set_xlabel("Time step", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id",     default="nojobid")
    parser.add_argument("--fast-ratio", type=float, default=0.20)
    args       = parser.parse_args()
    job_id     = args.job_id
    fast_ratio = args.fast_ratio

    FIGURES_DIR = FINETUNING_ROOT / "figures" / "finetunes" / f"finetune_sarsim0_joint_{job_id}"
    CKPT_DIR    = FINETUNING_ROOT / "checkpoints" / f"finetune-sarsim0-joint-{job_id}"

    # ── DDP setup ───────────────────────────────────────────────────────────────
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
        print("=" * 60)
        print("Joint TSF+FAST fine-tuning: chronos-t5-small  (SarSim0)")
        print(f"  Job ID      : {job_id}")
        print(f"  Device      : {device}  |  DDP={IS_DDP}  world={WORLD_SIZE}")
        print(f"  Fast ratio  : {fast_ratio:.0%} FAST  /  {1-fast_ratio:.0%} TSF")
        print(f"  Steps       : {N_STEPS:,}  |  batch={BATCH_SIZE}  |  LR={LR}")
        if baseline_mae["mean_mae"] is not None:
            print(f"  Baseline MAE: mean={baseline_mae['mean_mae']:.4f}"
                  f"  median={baseline_mae['median_mae']:.4f}  (pretrained)")
        print("=" * 60)
        write_config(FIGURES_DIR / "config.txt", job_id, fast_ratio)
    else:
        baseline_mae = {}

    # ── model ───────────────────────────────────────────────────────────────────
    print(f"\n[rank {LOCAL_RANK if IS_DDP else 0}] Loading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    inner    = pipeline.model.model
    if inner.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
    init_mode_embeddings(inner)

    # ── data ────────────────────────────────────────────────────────────────────
    if IS_MAIN:
        print("\nLoading dataset...")
    train_rows, eval_rows = load_rows(DATA_PATH)
    if IS_MAIN:
        print(f"  Train : {len(train_rows):,}  |  Eval : {len(eval_rows):,}")
        print("  Pre-tokenising TSF rows ...")
    tsf_tok  = pretokenise_tsf(train_rows,  pipeline.tokenizer)
    if IS_MAIN:
        print("  Pre-tokenising FAST rows ...")
    fast_tok = pretokenise_fast(train_rows, pipeline.tokenizer)

    per_rank_batch = max(1, BATCH_SIZE // WORLD_SIZE)
    if IS_DDP:
        tsf_shard  = [tsf_tok[i]  for i in range(len(tsf_tok))  if i % WORLD_SIZE == LOCAL_RANK]
        fast_shard = [fast_tok[i] for i in range(len(fast_tok)) if i % WORLD_SIZE == LOCAL_RANK]
    else:
        tsf_shard, fast_shard = tsf_tok, fast_tok

    tsf_loader  = CyclicLoader(tsf_shard,  per_rank_batch, seed=SEED)
    fast_loader = CyclicLoader(fast_shard, per_rank_batch, seed=SEED+1000)

    # Pre-generate per-step mode assignments — deterministic, same across all ranks
    mode_schedule = np.random.default_rng(SEED).random(N_STEPS) < fast_ratio  # True → FAST

    # ── training ────────────────────────────────────────────────────────────────
    print(f"\n[rank {LOCAL_RANK if IS_DDP else 0}] Moving model to {device} ...")
    inner = inner.to(device); inner.train()

    if IS_DDP:
        dp_model = DDP(inner, device_ids=[LOCAL_RANK])
        if IS_MAIN:
            print(f"  DDP across {WORLD_SIZE} GPUs")
    else:
        dp_model = inner

    optimizer = torch.optim.AdamW(inner.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, N_STEPS)

    loss_steps, losses  = [], []
    eval_steps          = []
    tsf_maes, fast_maes = [], []
    running   = 0.0
    LOG_EVERY = 50

    best_tsf_mae   = float("inf")
    no_improve_cnt = 0
    stopped_early  = False

    if IS_MAIN:
        print("\nTraining...")
    pbar = tqdm(range(1, N_STEPS+1), total=N_STEPS, desc="Training (joint TSF+FAST)",
                file=sys.stdout, dynamic_ncols=True, smoothing=0.05,
                miniters=LOG_EVERY, mininterval=0.0, disable=not IS_MAIN)

    for step in pbar:
        use_fast = bool(mode_schedule[step - 1])
        batch    = (fast_loader if use_fast else tsf_loader).next_batch()

        ids, mask, dec, lbl = assemble_batch(batch, device)
        out  = dp_model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec, labels=lbl)
        loss = out.loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), GRAD_CLIP)
        optimizer.step(); scheduler.step()
        running += loss.item()

        if step % LOG_EVERY == 0 and IS_MAIN:
            avg    = running / LOG_EVERY; running = 0.0
            lr_now = scheduler.get_last_lr()[0]
            loss_steps.append(step); losses.append(avg)
            mode_tag = "FAST" if use_fast else "TSF "
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}", mode=mode_tag)
            tqdm.write(
                f"  step {step:>6,}/{N_STEPS:,}  loss={avg:.4f}"
                f"  lr={lr_now:.2e}  mode={mode_tag}"
            )

        if step % EVAL_EVERY == 0:
            if IS_DDP:
                dist.barrier()
            if IS_MAIN:
                pipeline.model.model = inner
                tsf_mae  = evaluate_tsf( pipeline, eval_rows, device)
                fast_mae = evaluate_fast(pipeline, eval_rows, device)
                eval_steps.append(step)
                tsf_maes.append(tsf_mae)
                fast_maes.append(fast_mae)
                tqdm.write(
                    f"  [eval] step={step:,}"
                    f"  TSF MAE={tsf_mae:.4f}"
                    f"  FAST MAE={fast_mae:.4f}"
                    + (f"  (baseline mean={baseline_mae['mean_mae']:.4f})"
                       if baseline_mae.get("mean_mae") else "")
                )

                if tsf_mae < best_tsf_mae - EARLY_STOP_MIN_DELTA:
                    best_tsf_mae   = tsf_mae
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
                    tqdm.write(
                        f"  [early stop] No TSF improvement  {no_improve_cnt}/{EARLY_STOP_PATIENCE}"
                        f"  (best={best_tsf_mae:.4f})"
                    )
                    if no_improve_cnt >= EARLY_STOP_PATIENCE:
                        tqdm.write(
                            f"  [early stop] Triggered at step {step:,}."
                            f"  Best TSF MAE={best_tsf_mae:.4f}"
                        )
                        stopped_early = True

            if IS_DDP:
                stop_tensor = torch.tensor([1 if stopped_early else 0],
                                           dtype=torch.long, device=device)
                dist.broadcast(stop_tensor, src=0)
                if not IS_MAIN:
                    stopped_early = bool(stop_tensor.item())
                dist.barrier()

            inner.train()
            if stopped_early:
                break

    pbar.close()
    if IS_MAIN and stopped_early:
        print(f"\nEarly stop at step {step:,}  (best TSF MAE={best_tsf_mae:.4f})")

    if IS_MAIN:
        print(f"\nSaving checkpoint → {CKPT_DIR}")
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        inner.save_pretrained(str(CKPT_DIR))
        for name, arr in [("loss_steps", loss_steps), ("losses",   losses),
                          ("eval_steps", eval_steps), ("tsf_maes", tsf_maes),
                          ("fast_maes",  fast_maes)]:
            np.save(str(CKPT_DIR / f"{name}.npy"), np.array(arr))

        print("\nGenerating plots...")
        inner.eval(); pipeline.model.model = inner
        plot_loss(loss_steps, losses, eval_steps, tsf_maes, fast_maes,
                  baseline_mae, path=FIGURES_DIR / "loss.png",
                  job_id=job_id, fast_ratio=fast_ratio)
        plot_eval_samples_tsf(pipeline, eval_rows, device,
                              path=FIGURES_DIR / "eval_tsf_samples.png",
                              job_id=job_id, fast_ratio=fast_ratio)

        print(f"\nDone.  Figures → {FIGURES_DIR}/")
        if tsf_maes:
            print(f"Final TSF MAE  : {tsf_maes[-1]:.4f}")
            print(f"Final FAST MAE : {fast_maes[-1]:.4f}")
            if baseline_mae.get("mean_mae"):
                ratio = fast_maes[-1] / baseline_mae["mean_mae"]
                print(f"FAST retention : {ratio:.3f}x pretrained mean MAE")

    if IS_DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
