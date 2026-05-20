#!/usr/bin/env python3
"""
SarSim0 FAST-only fine-tuning baseline.

Trains Chronos-T5-Small in FAST-mode (direct 64-step forecast) on all
SarSim0 rows with no auxiliary conditioning.  Used as the "finetuned,
no conditioning" control condition in the STL-conditioning ablation.

Key differences from fine_tune_sarsim0_stl_cond.py:
  - No AuxiliaryConditioningModule — standard T5 forward pass.
  - No STL decomposition.
  - All 10,000 rows used (fast + reasoning).
  - 10,000 training steps to match the aux-cond runs.

Checkpointing:
  checkpoints/finetune-sarsim0-fastonly-{job_id}/
    config.json + model.safetensors   ← T5 backbone (save_pretrained)

Figures:
  figures/finetunes/finetune_sarsim0_fastonly_{job_id}/
    config.txt
    loss.png
    eval_samples.png

Usage (single GPU):
  /opt/conda/bin/python3.12 reasoning-finetuning/scripts/fine_tune_sarsim0_fast_only.py \\
      --job-id $SLURM_JOB_ID

Usage (multi-GPU via torchrun):
  torchrun --nproc_per_node=4 --master_port=29506 \\
      reasoning-finetuning/scripts/fine_tune_sarsim0_fast_only.py \\
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

from chronos import ChronosPipeline

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = FINETUNING_ROOT / "data"    / "sarsim0-10k.arrow"
BASELINE_REF = FINETUNING_ROOT / "figures" / "evals" / "eval_pretrained_sarsim0" / "baseline_mae.txt"
MODEL_ID     = "amazon/chronos-t5-small"

# ── dataset constants ─────────────────────────────────────────────────────────
CONTEXT_LEN = 512
HORIZON     = 64

# ── special / extended vocab ──────────────────────────────────────────────────
FAST_TOKEN_ID  = 4096
EXTENDED_VOCAB = 4105

# ── hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 128   # larger than aux-cond (no extra encoder tokens)
N_STEPS      = 10_000
WARMUP_STEPS = 500
LR           = 1e-4
GRAD_CLIP    = 1.0
EVAL_EVERY   = 500
EVAL_BATCH   = 32
EVAL_SAMPLES = 20
EVAL_N_SHOW  = 5
EVAL_POOL    = 200

EARLY_STOP_PATIENCE  = 5
EARLY_STOP_MIN_DELTA = 1e-4

EMBED_INIT_STD_MULTIPLIER = 2.0
SCALE_EPS = 1e-4
SEED      = 42

_CTX_COL  = "steelblue"
_PRED_COL = "crimson"
_GT_COL   = "darkorange"


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


def write_config(path: Path, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "Experiment Configuration", "=" * 40,
        f"Job ID         : {job_id}",
        f"Script         : fine_tune_sarsim0_fast_only.py",
        f"Model          : {MODEL_ID}",
        f"Dataset        : sarsim0-10k.arrow  (ALL rows — fast + reasoning)",
        f"Mode           : FAST-only (no auxiliary conditioning)",
        f"LR             : {LR}",
        f"Batch size     : {BATCH_SIZE}",
        f"Steps          : {N_STEPS:,}  |  warmup : {WARMUP_STEPS}",
        f"Ctx len        : {CONTEXT_LEN}  |  Horizon : {HORIZON}",
        f"Vocab          : extended to {EXTENDED_VOCAB}",
        f"Early stop     : patience={EARLY_STOP_PATIENCE}  delta={EARLY_STOP_MIN_DELTA}",
        "",
        "Role: 'finetuned, no conditioning' control condition for the STL ablation.",
    ]) + "\n")
    print(f"  Config written: {path.name}")


# ── data ──────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> tuple[list, list]:
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
            "context":  np.array(ctx, dtype=np.float32),
            "forecast": np.array(fct, dtype=np.float32),
        })
    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[EVAL_POOL:], rows[:EVAL_POOL]


def pretokenise(rows: list, tokenizer) -> list:
    N = len(rows)
    contexts  = torch.tensor(np.stack([r["context"]  for r in rows]), dtype=torch.float32)
    forecasts = torch.tensor(np.stack([r["forecast"] for r in rows]), dtype=torch.float32)

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

    return [{
        "item_id":       rows[i]["item_id"],
        "context":       rows[i]["context"],
        "forecast":      rows[i]["forecast"],
        "input_ids":     token_ids[i],
        "attn_mask":     attn_mask[i],
        "dec_input_ids": dec_input_ids[i],
        "labels":        labels[i],
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
    return ids, mask, dec, lbl


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_fast(model, pipeline, contexts: list, device) -> np.ndarray:
    """Standard FAST-mode generation, no auxiliary conditioning."""
    B  = len(contexts)
    mc = pipeline.model.config
    ctx = torch.tensor(np.stack(contexts), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)
    raw = model.generate(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        generation_config=GenerationConfig(
            min_new_tokens=HORIZON, max_new_tokens=HORIZON,
            do_sample=True, num_return_sequences=EVAL_SAMPLES,
            decoder_start_token_id=FAST_TOKEN_ID,
            eos_token_id=mc.eos_token_id, pad_token_id=mc.pad_token_id,
            temperature=mc.temperature, top_k=mc.top_k, top_p=mc.top_p,
        ),
    )
    raw      = raw[:, 1:]
    raw_bins = raw[:, :HORIZON].reshape(B, EVAL_SAMPLES, HORIZON)
    return pipeline.tokenizer.output_transform(raw_bins.cpu(), scale).median(dim=1).values.numpy()


def evaluate_fast(model, pipeline, eval_rows, device) -> float:
    model.eval()
    maes = []
    for i in range(0, len(eval_rows), EVAL_BATCH):
        batch = eval_rows[i:i + EVAL_BATCH]
        preds = predict_fast(model, pipeline, [r["context"] for r in batch], device)
        for r, pred in zip(batch, preds):
            maes.append(float(np.mean(np.abs(pred - r["forecast"]))))
    model.train()
    return float(np.mean(maes))


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_loss(loss_steps, losses, eval_steps, fast_maes, baseline_mae,
              path, job_id) -> None:
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
    ax.set_title("Eval MAE — FAST-only (no conditioning)", fontweight="bold")
    ax.set_xlabel("Step"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"SarSim0 FAST-only Baseline  ·  chronos-t5-small  ·  job {job_id}",
        fontweight="bold")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_eval_samples(model, pipeline, eval_rows, device, path, job_id,
                      n: int = EVAL_N_SHOW) -> None:
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(eval_rows), size=min(n, len(eval_rows)), replace=False))
    sel   = [eval_rows[i] for i in picks]

    model.eval()
    preds = predict_fast(model, pipeline, [r["context"] for r in sel], device)
    model.train()

    ctx_x = np.arange(CONTEXT_LEN)
    fct_x = np.arange(CONTEXT_LEN, CONTEXT_LEN + HORIZON)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"SarSim0 FAST-only Baseline  ·  job {job_id}\n"
        "Context (blue) → GT forecast (orange) vs FAST prediction (red)",
        fontsize=11, fontweight="bold")

    for idx, (row, pred) in enumerate(zip(sel, preds)):
        mae = float(np.mean(np.abs(pred - row["forecast"])))
        ax  = axes[idx]
        ax.fill_between(ctx_x, row["context"], alpha=0.06, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL, lw=0.6,
                label="Context" if idx == 0 else "_")
        ax.plot(fct_x, row["forecast"], color=_GT_COL, lw=1.8,
                label="GT forecast" if idx == 0 else "_")
        ax.plot(fct_x, pred, color=_PRED_COL, lw=1.4, ls="--", alpha=0.85,
                label=f"FAST pred (MAE={mae:.3f})" if idx == 0 else "_")
        ax.axvline(CONTEXT_LEN - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(f"{row['item_id']}  MAE={mae:.4f}", fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

    axes[-1].set_xlabel("Time step", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="nojobid")
    args   = parser.parse_args()
    job_id = args.job_id

    FIGURES_DIR = (FINETUNING_ROOT / "figures" / "finetunes"
                   / f"finetune_sarsim0_fastonly_{job_id}")
    CKPT_DIR    = (FINETUNING_ROOT / "checkpoints"
                   / f"finetune-sarsim0-fastonly-{job_id}")

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
        print("=" * 65)
        print("SarSim0 FAST-only Baseline  ·  chronos-t5-small")
        print(f"  Job ID  : {job_id}")
        print(f"  Device  : {device}  |  DDP={IS_DDP}  world={WORLD_SIZE}")
        print(f"  Steps   : {N_STEPS:,}  |  batch={BATCH_SIZE}  |  LR={LR}")
        if baseline_mae.get("mean_mae") is not None:
            print(f"  Baseline: mean={baseline_mae['mean_mae']:.4f}"
                  f"  median={baseline_mae['median_mae']:.4f}")
        print("=" * 65)
        write_config(FIGURES_DIR / "config.txt", job_id)
    else:
        baseline_mae = {}

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"\n[rank {LOCAL_RANK if IS_DDP else 0}] Loading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    inner    = pipeline.model.model
    if inner.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
    init_mode_embeddings(inner)

    # ── data ──────────────────────────────────────────────────────────────────
    if IS_MAIN:
        print("\nLoading dataset ...")
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
    inner = inner.to(device)
    inner.train()

    if IS_DDP:
        dp_model = DDP(inner, device_ids=[LOCAL_RANK])
        if IS_MAIN:
            print(f"  DDP across {WORLD_SIZE} GPUs")
    else:
        dp_model = inner

    optimizer = torch.optim.AdamW(inner.parameters(), lr=LR, weight_decay=1e-2)
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
                desc="Training (SarSim0 FAST-only)", file=sys.stdout,
                dynamic_ncols=True, smoothing=0.05,
                miniters=LOG_EVERY, mininterval=0.0, disable=not IS_MAIN)

    for step in pbar:
        batch = loader.next_batch()
        ids, mask, dec, lbl = assemble_batch(batch, device)

        out  = dp_model(input_ids=ids, attention_mask=mask,
                        decoder_input_ids=dec, labels=lbl)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), GRAD_CLIP)
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
                fast_mae = evaluate_fast(inner, pipeline, eval_rows, device)
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

            inner.train()
            if stopped_early:
                break

    pbar.close()
    if IS_MAIN and stopped_early:
        print(f"\nEarly stop at step {step:,}  (best FAST MAE={best_fast_mae:.4f})")

    if IS_MAIN:
        print(f"\nSaving checkpoint → {CKPT_DIR}")
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        inner.save_pretrained(str(CKPT_DIR))
        for name, arr in [
            ("loss_steps", loss_steps), ("losses",    losses),
            ("eval_steps", eval_steps), ("fast_maes", fast_maes),
        ]:
            np.save(str(CKPT_DIR / f"{name}.npy"), np.array(arr))

        print("\nGenerating plots...")
        inner.eval()
        plot_loss(loss_steps, losses, eval_steps, fast_maes, baseline_mae,
                  path=FIGURES_DIR / "loss.png", job_id=job_id)
        plot_eval_samples(inner, pipeline, eval_rows, device,
                          path=FIGURES_DIR / "eval_samples.png", job_id=job_id)
        print(f"\nDone.  Figures → {FIGURES_DIR}/")
        if fast_maes:
            print(f"Final FAST MAE : {fast_maes[-1]:.4f}")
            if baseline_mae.get("mean_mae"):
                print(f"FAST retention : {fast_maes[-1]/baseline_mae['mean_mae']:.3f}x pretrained mean MAE")

    if IS_DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
