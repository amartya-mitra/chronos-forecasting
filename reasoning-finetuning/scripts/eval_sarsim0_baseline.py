#!/usr/bin/env python3
"""
Evaluate amazon/chronos-t5-small on SarSim0 fast-mode datapoints.

Loads the context from each fast-mode sample, runs ChronosPipeline to get
the median forecast, computes MAE/MSE/MASE vs. the ground-truth forecast,
and produces two diagnostic plots:

  figures/sarsim0_baseline_sample_plots.png  — 16 randomly-selected samples
  figures/sarsim0_baseline_metrics.png       — metric distribution histograms

Usage:
  python3 scripts/eval_sarsim0_baseline.py            # uses first 200 fast samples
  python3 scripts/eval_sarsim0_baseline.py --n 500    # more samples
  python3 scripts/eval_sarsim0_baseline.py --all      # all 4000 fast samples
"""

import argparse
import sys
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from typing import Optional
import numpy as np
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from chronos import ChronosPipeline

# ── paths ──────────────────────────────────────────────────────────────────
DATA_PATH    = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
FIGURES_DIR  = FINETUNING_ROOT / "figures" / "evals" / "eval_baseline_sarsim0"
MODEL_ID     = "amazon/chronos-t5-small"

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 64

# ── plotting colours ────────────────────────────────────────────────────────
_CTX_COL  = "steelblue"
_GT_COL   = "darkorange"
_PRED_COL = "crimson"


# ── helpers ─────────────────────────────────────────────────────────────────

def load_fast_samples(path: Path, max_n: Optional[int] = None) -> list:
    """Read the Arrow IPC file and return only fast-mode rows."""
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()

    fast_rows = []
    limit = max_n if max_n is not None else 10_000_000
    modes = table.column("mode").to_pylist()
    ctxs  = table.column("context").to_pylist()
    fcts  = table.column("forecast").to_pylist()
    ids   = table.column("item_id").to_pylist()

    for item_id, mode, ctx, fct in zip(ids, modes, ctxs, fcts):
        if len(fast_rows) >= limit:
            break
        if mode == "fast":
            fast_rows.append({
                "item_id": item_id,
                "context": np.array(ctx, dtype=np.float32),
                "forecast": np.array(fct, dtype=np.float32),
            })

    return fast_rows


def run_inference(pipeline: ChronosPipeline, samples: list[dict]) -> list[np.ndarray]:
    """
    Run the pipeline on each sample's context and return the median prediction.
    Returns a list of float32 arrays of length PREDICTION_LENGTH.
    """
    preds = []
    for s in tqdm(samples, desc="Inference", unit="sample"):
        ctx = torch.tensor(s["context"]).unsqueeze(0)      # (1, T)
        with torch.no_grad():
            forecast = pipeline.predict(
                inputs=ctx,
                prediction_length=PREDICTION_LENGTH,
                num_samples=20,
                limit_prediction_length=False,
            )
        # forecast shape: (1, num_samples, pred_len) → take median
        median = np.median(forecast[0].cpu().numpy(), axis=0).astype(np.float32)
        preds.append(median)
    return preds


# ── metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(samples: list[dict], preds: list[np.ndarray]) -> dict:
    """Return per-sample MAE, MSE, and MASE arrays."""
    maes, mses, mases = [], [], []
    for s, pred in zip(samples, preds):
        gt  = s["forecast"]
        ctx = s["context"]

        err  = pred - gt
        mae  = float(np.mean(np.abs(err)))
        mse  = float(np.mean(err ** 2))

        # MASE denominator: naive 1-step MAE on context
        naive_err = np.abs(np.diff(ctx))
        denom = float(np.mean(naive_err)) if len(naive_err) > 0 else 1.0
        denom = max(denom, 1e-6)
        mase  = mae / denom

        maes.append(mae)
        mses.append(mse)
        mases.append(mase)

    return {
        "mae":  np.array(maes,  dtype=np.float32),
        "mse":  np.array(mses,  dtype=np.float32),
        "mase": np.array(mases, dtype=np.float32),
    }


# ── plots ────────────────────────────────────────────────────────────────────

def plot_samples(samples: list[dict], preds: list[np.ndarray],
                 path: Path, n_show: int = 16) -> None:
    """
    Plot n_show randomly-selected samples, each showing:
      • context (steelblue)
      • ground-truth forecast (orange)
      • predicted forecast (crimson)
    """
    rng = np.random.default_rng(0)
    indices = rng.choice(len(samples), size=min(n_show, len(samples)), replace=False)
    indices.sort()

    n_cols = 4
    n_rows = (len(indices) + n_cols - 1) // n_cols

    ctx_x = np.arange(CONTEXT_LENGTH)
    fct_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    axes_flat = axes.flatten()

    fig.suptitle(
        f"chronos-t5-small  ·  SarSim0 fast-mode  ·  {len(indices)} samples\n"
        f"orange = ground truth   crimson = model forecast",
        fontsize=11, fontweight="bold",
    )

    for ax_idx, sample_idx in enumerate(indices):
        ax  = axes_flat[ax_idx]
        s   = samples[sample_idx]
        pred = preds[sample_idx]

        ax.fill_between(ctx_x, s["context"], alpha=0.08, color=_CTX_COL)
        ax.plot(ctx_x, s["context"],  color=_CTX_COL,  lw=0.7, label="Context")
        ax.plot(fct_x, s["forecast"], color=_GT_COL,   lw=1.5, label="GT")
        ax.plot(fct_x, pred,          color=_PRED_COL, lw=1.5, ls="--", label="Pred")
        ax.axvline(CONTEXT_LENGTH - 0.5, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.set_title(s["item_id"], fontsize=7, loc="left", pad=2)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    # Legend on first subplot only
    axes_flat[0].legend(fontsize=7, loc="upper left", framealpha=0.7)

    # Hide unused axes
    for ax in axes_flat[len(indices):]:
        ax.set_visible(False)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_forecast_zoom(samples: list[dict], preds: list[np.ndarray],
                       path: Path, n_show: int = 16) -> None:
    """
    Zoomed-in view of the forecast window only (no context) for n_show samples.
    Each subplot shows ground-truth and model-predicted forecast side-by-side.
    Uses the same random indices as plot_samples (seed=0) for consistency.
    """
    rng = np.random.default_rng(0)
    indices = rng.choice(len(samples), size=min(n_show, len(samples)), replace=False)
    indices.sort()

    n_cols = 4
    n_rows = (len(indices) + n_cols - 1) // n_cols

    fct_x = np.arange(PREDICTION_LENGTH)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes_flat = np.asarray(axes).flatten()

    fig.suptitle(
        f"chronos-t5-small  ·  SarSim0 fast-mode  ·  forecast window zoom  ·  {len(indices)} samples\n"
        f"orange = ground truth   crimson = model forecast",
        fontsize=11, fontweight="bold",
    )

    for ax_idx, sample_idx in enumerate(indices):
        ax   = axes_flat[ax_idx]
        s    = samples[sample_idx]
        pred = preds[sample_idx]

        ax.fill_between(fct_x, s["forecast"], alpha=0.12, color=_GT_COL)
        ax.plot(fct_x, s["forecast"], color=_GT_COL,   lw=1.8, label="GT")
        ax.plot(fct_x, pred,          color=_PRED_COL, lw=1.8, ls="--", label="Pred")

        # Shade error region
        ax.fill_between(fct_x, s["forecast"], pred,
                        alpha=0.15, color=_PRED_COL, label="Error")

        mae = float(np.mean(np.abs(pred - s["forecast"])))
        ax.set_title(f"{s['item_id']}  MAE={mae:.3f}", fontsize=6.5, loc="left", pad=2)
        ax.set_xlabel("Forecast step", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    axes_flat[0].legend(fontsize=7, loc="best", framealpha=0.7)

    for ax in axes_flat[len(indices):]:
        ax.set_visible(False)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_metrics(metrics: dict, path: Path) -> None:
    """
    Three-panel histogram of MAE, MSE, MASE distributions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (key, label, colour) in zip(axes, [
        ("mae",  "MAE",  "steelblue"),
        ("mse",  "MSE",  "seagreen"),
        ("mase", "MASE", "darkorange"),
    ]):
        vals = metrics[key]
        # Clip extreme outliers for display
        p99 = np.percentile(vals, 99)
        vals_clip = vals[vals <= p99]

        ax.hist(vals_clip, bins=40, color=colour, edgecolor="white", alpha=0.85)
        ax.axvline(np.median(vals), color="black", lw=1.4, ls="--",
                   label=f"Median {np.median(vals):.4f}")
        ax.axvline(np.mean(vals),   color="red",   lw=1.0, ls=":",
                   label=f"Mean {np.mean(vals):.4f}")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"chronos-t5-small  ·  SarSim0 fast-mode  ·  n={len(metrics['mae'])}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int,  default=200,
                        help="Number of fast-mode samples to evaluate (default 200)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all 4 000 fast-mode samples")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda / cpu  (default cuda)")
    args = parser.parse_args()

    n_samples = None if args.all else args.n

    print("=" * 60)
    print("Baseline Evaluation: chronos-t5-small on SarSim0")
    print(f"  Dataset : {DATA_PATH}")
    print(f"  Samples : {'all' if n_samples is None else n_samples}")
    print(f"  Device  : {args.device}")
    print("=" * 60)

    # ── load data ────────────────────────────────────────────────────────
    print("\nLoading fast-mode samples...")
    samples = load_fast_samples(DATA_PATH, max_n=n_samples)
    print(f"  Loaded {len(samples)} fast-mode samples")

    # ── load model ───────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
    )
    print("  Model loaded.")

    # ── inference ────────────────────────────────────────────────────────
    print("\nRunning inference...")
    preds = run_inference(pipeline, samples)

    # ── metrics ──────────────────────────────────────────────────────────
    print("\nComputing metrics...")
    metrics = compute_metrics(samples, preds)

    print(f"\n{'Metric':>8}  {'Mean':>10}  {'Median':>10}  {'Std':>10}")
    print("-" * 44)
    for key in ["mae", "mse", "mase"]:
        v = metrics[key]
        print(f"{key.upper():>8}  {np.mean(v):>10.4f}  {np.median(v):>10.4f}  {np.std(v):>10.4f}")

    # ── plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_samples(
        samples, preds,
        path=FIGURES_DIR / "sarsim0_baseline_sample_plots.png",
    )
    plot_forecast_zoom(
        samples, preds,
        path=FIGURES_DIR / "sarsim0_baseline_forecast_zoom.png",
    )
    plot_metrics(
        metrics,
        path=FIGURES_DIR / "sarsim0_baseline_metrics.png",
    )

    print(f"\nDone.  Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
