#!/usr/bin/env python3
"""
Evaluate pretrained amazon/chronos-t5-small on the M4 Hourly eval pool.

Uses the same SEED and EVAL_POOL split as fine_tune_m4h_joint.py so the 100
eval rows are identical to those withheld from joint training.

Outputs:
  figures/eval_pretrained_m4hourly/sample_plots.png
  figures/eval_pretrained_m4hourly/metrics.png
  figures/eval_pretrained_m4hourly/baseline_mae.txt   ← reference for joint runs
"""

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

from chronos import ChronosPipeline

# ── constants — must stay in sync with fine_tune_m4h_joint.py ──────────────────
DATA_PATH   = FINETUNING_ROOT / "data"    / "m4hourly-tsf.arrow"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "eval_pretrained_m4hourly"
MODEL_ID    = "amazon/chronos-t5-small"

CONTEXT_LEN  = 96
HORIZON      = 48
EVAL_POOL    = 100
SEED         = 42
SCALE_EPS    = 1e-4
NUM_SAMPLES  = 20

_CTX_COL  = "steelblue"
_GT_COL   = "darkorange"
_PRED_COL = "crimson"


# ── data ───────────────────────────────────────────────────────────────────────

def load_eval_rows(path: Path) -> list:
    """
    Load arrow, apply the same deterministic shuffle as training (SEED=42),
    and return the first EVAL_POOL rows — identical to the training eval split.
    """
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
            "target":  np.array(tgt, dtype=np.float32),   # (144,) context-scale-normalised
            "s_third": float(s3),
        })

    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[:EVAL_POOL]


def denorm_forecast(row: dict) -> np.ndarray:
    """
    Recover ground-truth 48-step forecast in original units.

    Stored: forecast_norm = future * (s_ctx / s_third)
    Inverse: forecast_gt  = forecast_norm * (s_third / s_ctx)
    """
    s_ctx         = max(float(np.mean(np.abs(row["context"]))), SCALE_EPS)
    forecast_norm = row["target"][2 * HORIZON:]            # last 48 of 144
    return forecast_norm * (row["s_third"] / s_ctx)


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(pipeline: ChronosPipeline, rows: list) -> list[np.ndarray]:
    """Run pretrained model on each context; return median 48-step forecasts."""
    preds = []
    for row in tqdm(rows, desc="Inference", unit="sample"):
        ctx = torch.tensor(row["context"]).unsqueeze(0)    # (1, 96)
        with torch.no_grad():
            out = pipeline.predict(
                inputs=ctx,
                prediction_length=HORIZON,
                num_samples=NUM_SAMPLES,
                limit_prediction_length=False,
            )
        median = np.median(out[0].cpu().numpy(), axis=0).astype(np.float32)
        preds.append(median)
    return preds


# ── metrics ───────────────────────────────────────────────────────────────────

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


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_samples(rows: list, preds: list, path: Path, n_show: int = 16) -> None:
    rng     = np.random.default_rng(0)
    indices = sorted(rng.choice(len(rows), size=min(n_show, len(rows)), replace=False))
    n_cols  = 4
    n_rows  = (len(indices) + n_cols - 1) // n_cols
    ctx_x   = np.arange(CONTEXT_LEN)
    fct_x   = np.arange(CONTEXT_LEN, CONTEXT_LEN + HORIZON)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    axes_flat = np.asarray(axes).flatten()
    fig.suptitle(
        f"chronos-t5-small (pretrained)  ·  M4 Hourly eval pool  ·  {len(indices)} samples\n"
        "orange = ground truth   crimson = model forecast",
        fontsize=11, fontweight="bold",
    )
    for ax_idx, si in enumerate(indices):
        ax   = axes_flat[ax_idx]
        row  = rows[si]
        pred = preds[si]
        gt   = denorm_forecast(row)
        mae  = float(np.mean(np.abs(pred - gt)))
        ax.fill_between(ctx_x, row["context"], alpha=0.08, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL,  lw=0.7, label="Context")
        ax.plot(fct_x, gt,             color=_GT_COL,   lw=1.5, label="GT")
        ax.plot(fct_x, pred,           color=_PRED_COL, lw=1.5, ls="--", label="Pred")
        ax.axvline(CONTEXT_LEN - 0.5, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.set_title(f"{row['item_id']}  MAE={mae:.1f}", fontsize=7, loc="left", pad=2)
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


def plot_metrics(metrics: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (key, label, colour) in zip(axes, [
        ("mae",  "MAE",  "steelblue"),
        ("mase", "MASE", "seagreen"),
    ]):
        vals      = metrics[key]
        p99       = np.percentile(vals, 99)
        vals_clip = vals[vals <= p99]
        ax.hist(vals_clip, bins=40, color=colour, edgecolor="white", alpha=0.85)
        ax.axvline(np.median(vals), color="black", lw=1.4, ls="--",
                   label=f"Median {np.median(vals):.2f}")
        ax.axvline(np.mean(vals),   color="red",   lw=1.0, ls=":",
                   label=f"Mean {np.mean(vals):.2f}")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"chronos-t5-small (pretrained)  ·  M4 Hourly  ·  n={len(metrics['mae'])}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Pretrained Baseline: chronos-t5-small on M4 Hourly")
    print(f"  Dataset   : {DATA_PATH}")
    print(f"  Eval pool : {EVAL_POOL} series  (SEED={SEED})")
    print(f"  Horizon   : {HORIZON} steps")
    print("=" * 60)

    rows = load_eval_rows(DATA_PATH)
    print(f"  Loaded {len(rows)} eval rows")

    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16,
    )
    print("  Model loaded.")

    preds   = run_inference(pipeline, rows)
    metrics = compute_metrics(rows, preds)

    mean_mae    = float(np.mean(metrics["mae"]))
    median_mae  = float(np.median(metrics["mae"]))
    mean_mase   = float(np.mean(metrics["mase"]))
    median_mase = float(np.median(metrics["mase"]))

    print(f"\n{'Metric':>8}  {'Mean':>10}  {'Median':>10}")
    print("-" * 34)
    print(f"{'MAE':>8}  {mean_mae:>10.4f}  {median_mae:>10.4f}")
    print(f"{'MASE':>8}  {mean_mase:>10.4f}  {median_mase:>10.4f}")

    # Save reference numbers — loaded by joint fine-tuning run for comparison
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = FIGURES_DIR / "baseline_mae.txt"
    ref_path.write_text(
        "chronos-t5-small pretrained baseline  —  M4 Hourly eval pool\n"
        f"n_eval      : {len(rows)}\n"
        f"mean_mae    : {mean_mae:.4f}\n"
        f"median_mae  : {median_mae:.4f}\n"
        f"mean_mase   : {mean_mase:.4f}\n"
        f"median_mase : {median_mase:.4f}\n"
    )
    print(f"\n  Reference saved → {ref_path}")

    plot_samples(rows, preds, FIGURES_DIR / "sample_plots.png")
    plot_metrics(metrics,     FIGURES_DIR / "metrics.png")
    print(f"\nDone.  Figures → {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
