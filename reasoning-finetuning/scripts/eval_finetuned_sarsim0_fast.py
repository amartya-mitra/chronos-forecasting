#!/usr/bin/env python3
"""
Evaluate the FAST-mode zero-shot forecasting of a SarSim0 joint-finetuned checkpoint.

Loads the finetuned model from:
  checkpoints/finetune-sarsim0-joint-{job_id}/

Runs FAST-mode inference (token 4096) on the same 100-row eval pool used
during training and compares against the pretrained baseline reference.

Outputs (saved into figures/finetune_sarsim0_joint_{job_id}/):
  eval_fast_samples.png
  eval_fast_metrics.png
  eval_fast_summary.txt

Usage:
  python eval_finetuned_sarsim0_fast.py --job-id <SLURM_JOB_ID>
"""

import argparse
import sys
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, GenerationConfig

from chronos import ChronosPipeline

# ── constants (must stay in sync with fine_tune_sarsim0_joint.py) ─────────────
DATA_PATH    = FINETUNING_ROOT / "data"    / "sarsim0-tsf.arrow"
BASELINE_REF = FINETUNING_ROOT / "figures" / "evals" / "eval_pretrained_sarsim0" / "baseline_mae.txt"
BASE_MODEL   = "amazon/chronos-t5-small"

CONTEXT_LEN    = 512
HORIZON        = 64
EVAL_POOL      = 100
SEED           = 42
SCALE_EPS      = 1e-4
NUM_SAMPLES    = 20
FAST_TOKEN_ID  = 4096

_CTX_COL  = "steelblue"
_GT_COL   = "darkorange"
_PRED_COL = "crimson"


# ── data ───────────────────────────────────────────────────────────────────────

def load_eval_rows(path: Path) -> list:
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()

    item_ids = table.column("item_id").to_pylist()
    contexts = table.column("context").to_pylist()
    targets  = table.column("target").to_pylist()
    s_thirds = table.column("s_third").to_pylist()

    rows = []
    for item_id, ctx, tgt, s3 in zip(item_ids, contexts, targets, s_thirds):
        rows.append({
            "item_id": item_id,
            "context": np.array(ctx, dtype=np.float32),
            "target":  np.array(tgt, dtype=np.float32),
            "s_third": float(s3),
        })

    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[:EVAL_POOL]


def denorm_forecast(row: dict) -> np.ndarray:
    s_ctx         = max(float(np.mean(np.abs(row["context"]))), SCALE_EPS)
    forecast_norm = row["target"][2 * HORIZON:]
    return forecast_norm * (row["s_third"] / s_ctx)


# ── model loading ──────────────────────────────────────────────────────────────

def load_finetuned_pipeline(ckpt_dir: Path, device: torch.device) -> ChronosPipeline:
    pipeline  = ChronosPipeline.from_pretrained(
        BASE_MODEL, device_map="cpu", torch_dtype=torch.bfloat16,
    )
    finetuned = AutoModelForSeq2SeqLM.from_pretrained(
        str(ckpt_dir), torch_dtype=torch.bfloat16,
    )
    pipeline.model.model = finetuned.to(device)
    pipeline.model.model.eval()
    return pipeline


# ── inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_fast_inference(pipeline: ChronosPipeline, rows: list,
                       device: torch.device) -> list[np.ndarray]:
    model_cfg = pipeline.model.config
    preds = []
    for row in tqdm(rows, desc="FAST inference", unit="sample"):
        ctx = torch.tensor(row["context"]).unsqueeze(0)
        tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)

        raw = pipeline.model.model.generate(
            input_ids=tids.to(device),
            attention_mask=amask.to(device),
            generation_config=GenerationConfig(
                min_new_tokens=HORIZON, max_new_tokens=HORIZON,
                do_sample=True, num_return_sequences=NUM_SAMPLES,
                decoder_start_token_id=FAST_TOKEN_ID,
                eos_token_id=model_cfg.eos_token_id,
                pad_token_id=model_cfg.pad_token_id,
                temperature=model_cfg.temperature,
                top_k=model_cfg.top_k, top_p=model_cfg.top_p,
            ),
        )
        raw_bins = raw[:, 1:HORIZON + 1].reshape(1, NUM_SAMPLES, HORIZON)
        out    = pipeline.tokenizer.output_transform(raw_bins.cpu(), scale)
        median = out.median(dim=1).values[0].numpy().astype(np.float32)
        preds.append(median)
    return preds


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list, preds: list) -> dict:
    maes, mases = [], []
    for row, pred in zip(rows, preds):
        gt        = denorm_forecast(row)
        mae       = float(np.mean(np.abs(pred - gt)))
        naive_mae = float(np.mean(np.abs(np.diff(row["context"]))))
        mase      = mae / max(naive_mae, 1e-6)
        maes.append(mae)
        mases.append(mase)
    return {
        "mae":  np.array(maes,  dtype=np.float32),
        "mase": np.array(mases, dtype=np.float32),
    }


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    data = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                data[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return data


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_samples(rows: list, preds: list, path: Path, job_id: str,
                 n_show: int = 16) -> None:
    rng     = np.random.default_rng(0)
    indices = sorted(rng.choice(len(rows), size=min(n_show, len(rows)), replace=False))
    n_cols  = 4
    n_rows  = (len(indices) + n_cols - 1) // n_cols
    ctx_x   = np.arange(CONTEXT_LEN)
    fct_x   = np.arange(CONTEXT_LEN, CONTEXT_LEN + HORIZON)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 3))
    axes_flat = np.asarray(axes).flatten()
    fig.suptitle(
        f"FAST mode (token 4096)  ·  finetuned job {job_id}  ·  SarSim0 eval pool\n"
        "orange = ground truth   crimson = model forecast",
        fontsize=11, fontweight="bold",
    )
    for ax_idx, si in enumerate(indices):
        ax   = axes_flat[ax_idx]
        row  = rows[si]
        pred = preds[si]
        gt   = denorm_forecast(row)
        mae  = float(np.mean(np.abs(pred - gt)))
        ax.fill_between(ctx_x, row["context"], alpha=0.06, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL,  lw=0.5, label="Context")
        ax.plot(fct_x, gt,             color=_GT_COL,   lw=1.5, label="GT")
        ax.plot(fct_x, pred,           color=_PRED_COL, lw=1.5, ls="--", label="Pred")
        ax.axvline(CONTEXT_LEN - 0.5, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.set_title(f"{row['item_id']}  MAE={mae:.4f}", fontsize=7, loc="left", pad=2)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axes_flat[0].legend(fontsize=7, loc="upper left", framealpha=0.7)
    for ax in axes_flat[len(indices):]:
        ax.set_visible(False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_metrics(metrics: dict, baseline: dict, path: Path, job_id: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pairs = [
        ("mae",  "MAE",  "steelblue", "mean_mae",  "median_mae"),
        ("mase", "MASE", "seagreen",  "mean_mase", "median_mase"),
    ]
    for ax, (key, label, colour, b_mean_k, b_med_k) in zip(axes, pairs):
        vals      = metrics[key]
        p99       = np.percentile(vals, 99)
        vals_clip = vals[vals <= p99]
        ax.hist(vals_clip, bins=40, color=colour, edgecolor="white", alpha=0.85)
        ax.axvline(np.median(vals), color="black", lw=1.4, ls="--",
                   label=f"Median {np.median(vals):.4f}")
        ax.axvline(np.mean(vals), color="red", lw=1.0, ls=":",
                   label=f"Mean {np.mean(vals):.4f}")
        if baseline.get(b_mean_k):
            ax.axvline(baseline[b_mean_k], color="grey", lw=1.2, ls="-.",
                       label=f"Pretrained mean {baseline[b_mean_k]:.4f}")
        if baseline.get(b_med_k):
            ax.axvline(baseline[b_med_k], color="dimgrey", lw=1.0, ls=":",
                       label=f"Pretrained median {baseline[b_med_k]:.4f}")
        ax.set_title(f"FAST {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"FAST mode (token 4096)  ·  finetuned job {job_id}  ·  n={len(metrics['mae'])}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args   = parser.parse_args()
    job_id = args.job_id

    CKPT_DIR    = FINETUNING_ROOT / "checkpoints" / f"finetune-sarsim0-joint-{job_id}"
    FIGURES_DIR = FINETUNING_ROOT / "figures" / "finetunes" / f"finetune_sarsim0_joint_{job_id}"
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"FAST-mode eval  ·  SarSim0  ·  finetuned job {job_id}")
    print(f"  Checkpoint : {CKPT_DIR}")
    print(f"  Eval pool  : {EVAL_POOL} series  (SEED={SEED})")
    print(f"  Horizon    : {HORIZON} steps")
    print(f"  Device     : {device}")
    print("=" * 60)

    rows     = load_eval_rows(DATA_PATH)
    baseline = load_baseline(BASELINE_REF)
    print(f"  Loaded {len(rows)} eval rows")
    if baseline:
        print(f"  Pretrained baseline  mean_mae={baseline.get('mean_mae', 'n/a'):.4f}"
              f"  median_mae={baseline.get('median_mae', 'n/a'):.4f}")

    pipeline = load_finetuned_pipeline(CKPT_DIR, device)
    print("  Finetuned model loaded.")

    preds   = run_fast_inference(pipeline, rows, device)
    metrics = compute_metrics(rows, preds)

    mean_mae    = float(np.mean(metrics["mae"]))
    median_mae  = float(np.median(metrics["mae"]))
    mean_mase   = float(np.mean(metrics["mase"]))
    median_mase = float(np.median(metrics["mase"]))

    print(f"\n{'Metric':>8}  {'Mean':>10}  {'Median':>10}")
    print("-" * 34)
    print(f"{'MAE':>8}  {mean_mae:>10.4f}  {median_mae:>10.4f}")
    print(f"{'MASE':>8}  {mean_mase:>10.4f}  {median_mase:>10.4f}")

    if baseline.get("mean_mae"):
        ratio = mean_mae / baseline["mean_mae"]
        print(f"\n  FAST retention (mean MAE ratio vs pretrained): {ratio:.3f}x")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = FIGURES_DIR / "eval_fast_summary.txt"
    summary_path.write_text(
        f"FAST-mode eval  ·  SarSim0  ·  finetuned job {job_id}\n"
        f"n_eval      : {len(rows)}\n"
        f"mean_mae    : {mean_mae:.4f}\n"
        f"median_mae  : {median_mae:.4f}\n"
        f"mean_mase   : {mean_mase:.4f}\n"
        f"median_mase : {median_mase:.4f}\n"
        + (f"retention_vs_pretrained_mean : {mean_mae / baseline['mean_mae']:.4f}x\n"
           if baseline.get("mean_mae") else "")
    )
    print(f"\n  Summary saved → {summary_path}")

    plot_samples(rows, preds, FIGURES_DIR / "eval_fast_samples.png", job_id)
    plot_metrics(metrics, baseline, FIGURES_DIR / "eval_fast_metrics.png", job_id)
    print(f"\nDone.  Figures → {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
