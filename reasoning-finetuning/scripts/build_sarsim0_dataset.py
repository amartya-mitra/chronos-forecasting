#!/usr/bin/env python3
"""
Build SarSim0 reasoning fine-tuning dataset — 10,000 samples.

Split:
  40% fast-mode      (4,000): context  →  forecast
  60% reasoning-mode (6,000): context  →  trend + seasonal + forecast

The decomposed-signal generation logic is taken directly from
sarsim0_adapter.py, inlined here to avoid requiring the numba-dependent
SarSim0 library (only used in the full SARIMA-2 training pipeline).

Output:
  reasoning-finetuning/data/sarsim0-10k.arrow
  reasoning-finetuning/figures/sarsim0_fast_batch1.png
  reasoning-finetuning/figures/sarsim0_fast_batch2.png
  reasoning-finetuning/figures/sarsim0_reasoning_batch1.png
  reasoning-finetuning/figures/sarsim0_reasoning_batch2.png

Usage:
  python3 scripts/build_sarsim0_dataset.py
"""

import sys
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FINETUNING_ROOT))

import numpy as np
import pyarrow as pa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── dataset constants ────────────────────────────────────────────────────────
CONTEXT_LENGTH      = 512
PREDICTION_LENGTH   = 64
DECOMPOSITION_LENGTH = 64
BURN_IN             = 200

TOTAL_SAMPLES = 10_000
N_FAST        = 4_000   # 40 %
N_REASONING   = 6_000   # 60 %
SEED          = 42

DATA_DIR    = FINETUNING_ROOT / "data"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "datasets" / "dataset_sarsim0"
OUTPUT_PATH = DATA_DIR / "sarsim0-10k.arrow"


# ── signal generation (from sarsim0_adapter.py) ──────────────────────────────
# The three component generators are pure-numpy and do not depend on the
# numba-backed SarSim0 library.  They are reproduced verbatim from
# sarsim0_adapter._generate_{trend,seasonal,volatility}_component.

def _generate_trend(length: int, rng: np.random.Generator) -> np.ndarray:
    """AR(1) with high persistence + slow linear drift."""
    phi     = rng.uniform(0.95, 0.99)
    epsilon = rng.normal(0, 0.1, length)
    trend   = np.zeros(length)
    for t in range(1, length):
        trend[t] = phi * trend[t - 1] + epsilon[t]
    drift = np.linspace(0, rng.uniform(-1, 1), length)
    return trend + drift


def _generate_seasonal(length: int, rng: np.random.Generator) -> np.ndarray:
    """Sinusoidal with a random period, amplitude, phase, optional 2nd harmonic."""
    s         = rng.choice([7, 12, 24, 52])
    amplitude = rng.uniform(0.5, 2.0)
    phase     = rng.uniform(0, 2 * np.pi)
    t         = np.arange(length)
    seasonal  = amplitude * np.sin(2 * np.pi * t / s + phase)
    if rng.random() > 0.5:
        seasonal += (amplitude / 2) * np.sin(4 * np.pi * t / s + phase)
    return seasonal + rng.normal(0, 0.05, length)


def _generate_volatility(length: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian / heavy-tail (Student-t) / GARCH-like heteroskedastic noise."""
    noise_type = rng.choice(["gaussian", "heavy_tail", "heteroskedastic"])

    if noise_type == "gaussian":
        return rng.normal(0, rng.uniform(0.1, 0.5), length)

    elif noise_type == "heavy_tail":
        df    = rng.uniform(3, 8)
        scale = rng.uniform(0.1, 0.3)
        return rng.standard_t(df, length) * scale

    else:  # heteroskedastic (GARCH-like)
        omega = rng.uniform(0.01, 0.05)
        alpha = rng.uniform(0.05, 0.15)
        beta  = rng.uniform(0.70, 0.90)
        v     = np.zeros(length)
        h     = omega / max(1 - alpha - beta, 1e-6)
        for t in range(1, length):
            h    = omega + alpha * v[t - 1] ** 2 + beta * h
            v[t] = rng.normal(0, np.sqrt(max(h, 1e-6)))
        return v


def _normalize(series: np.ndarray) -> tuple[np.ndarray, float]:
    """Clip and scale so max |value| ≤ 10; return (normalized, scale_factor)."""
    series = np.nan_to_num(series, nan=0.0, posinf=10.0, neginf=-10.0)
    series = np.clip(series, -100, 100)
    abs_max = np.max(np.abs(series))
    if abs_max > 10.0:
        factor = 10.0 / abs_max
        return series * factor, factor
    return series, 1.0


def generate_sample(idx: int, mode: str, rng: np.random.Generator) -> dict:
    """
    Generate one time-series sample with additive decomposition.

      y = trend + seasonal + volatility   (normalised together)

    Returns a dict with keys:
      item_id, mode, context, forecast, trend, seasonal
    where trend and seasonal are [] for fast-mode samples.
    """
    total = CONTEXT_LENGTH + PREDICTION_LENGTH + BURN_IN

    trend      = _generate_trend(total, rng)
    seasonal   = _generate_seasonal(total, rng)
    volatility = _generate_volatility(total, rng)
    combined   = trend + seasonal + volatility

    # Remove burn-in
    combined   = combined[BURN_IN:]
    trend      = trend[BURN_IN:]
    seasonal   = seasonal[BURN_IN:]

    # Normalize combined; apply same scale to components to preserve additivity
    combined, scale = _normalize(combined)
    trend    = trend    * scale
    seasonal = seasonal * scale

    context  = combined[:CONTEXT_LENGTH].astype(np.float32).tolist()
    forecast = combined[CONTEXT_LENGTH:CONTEXT_LENGTH + PREDICTION_LENGTH] \
                       .astype(np.float32).tolist()

    row = {
        "item_id":  f"sarsim0_{idx:05d}",
        "mode":     mode,
        "context":  context,
        "forecast": forecast,
        "trend":    [],
        "seasonal": [],
    }

    if mode == "reasoning":
        row["trend"]    = trend[-DECOMPOSITION_LENGTH:].astype(np.float32).tolist()
        row["seasonal"] = seasonal[-DECOMPOSITION_LENGTH:].astype(np.float32).tolist()

    return row


# ── dataset generation ───────────────────────────────────────────────────────

def generate_dataset(seed: int = SEED) -> list[dict]:
    rng   = np.random.default_rng(seed)
    modes = ["fast"] * N_FAST + ["reasoning"] * N_REASONING
    rng.shuffle(modes)

    return [
        generate_sample(idx, mode, rng)
        for idx, mode in enumerate(tqdm(modes, desc="Generating", unit="sample"))
    ]


# ── persistence ──────────────────────────────────────────────────────────────

def save_arrow(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.table({
        "item_id":  pa.array([r["item_id"]  for r in rows], type=pa.string()),
        "mode":     pa.array([r["mode"]     for r in rows], type=pa.string()),
        "context":  pa.array([r["context"]  for r in rows], type=pa.list_(pa.float32())),
        "forecast": pa.array([r["forecast"] for r in rows], type=pa.list_(pa.float32())),
        "trend":    pa.array([r["trend"]    for r in rows], type=pa.list_(pa.float32())),
        "seasonal": pa.array([r["seasonal"] for r in rows], type=pa.list_(pa.float32())),
    })

    with pa.OSFile(str(path), "wb") as f:
        writer = pa.ipc.new_file(f, table.schema)
        writer.write_table(table)
        writer.close()

    n_fast = sum(1 for r in rows if r["mode"] == "fast")
    n_reas = len(rows) - n_fast
    print(f"Saved {len(rows):,} samples → {path}")
    print(f"  fast-mode:      {n_fast:,}  ({100 * n_fast / len(rows):.0f}%)")
    print(f"  reasoning-mode: {n_reas:,}  ({100 * n_reas / len(rows):.0f}%)")


# ── plotting ─────────────────────────────────────────────────────────────────

_CTX_COLOUR  = "steelblue"
_FCT_COLOUR  = "darkorange"
_TRND_COLOUR = "seagreen"
_SEAS_COLOUR = "mediumpurple"


def plot_fast_mode(samples: list[dict], batch: int, path: Path) -> None:
    """
    One figure per batch of 5 samples.
    Each row: context (blue filled) → forecast (orange), split by dashed line.
    """
    subset = samples[batch * 5 : batch * 5 + 5]

    fig, axes_raw = plt.subplots(5, 1, figsize=(14, 11))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"Fast-Mode Samples — Batch {batch + 1}   "
        f"[context  →  forecast]",
        fontsize=12, fontweight="bold", y=1.01,
    )

    ctx_len = CONTEXT_LENGTH
    fct_len = PREDICTION_LENGTH
    ctx_x   = np.arange(ctx_len)
    fct_x   = np.arange(ctx_len, ctx_len + fct_len)

    for row, (ax, s) in enumerate(zip(axes, subset)):
        ctx = np.array(s["context"],  dtype=np.float32)
        fct = np.array(s["forecast"], dtype=np.float32)

        ax.fill_between(ctx_x, ctx, alpha=0.10, color=_CTX_COLOUR)
        ax.plot(ctx_x, ctx, color=_CTX_COLOUR,  lw=0.9,
                label="Context" if row == 0 else "_nolegend_")
        ax.plot(fct_x, fct, color=_FCT_COLOUR,  lw=1.8,
                label="Forecast" if row == 0 else "_nolegend_")
        ax.axvline(ctx_len - 0.5, color="black", lw=0.9, ls="--", alpha=0.4)
        ax.set_title(s["item_id"], fontsize=8, loc="left", pad=2)
        ax.set_ylabel("Value", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.7)
    axes[-1].set_xlabel("Time step", fontsize=9)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  {path.name}")


def plot_reasoning_mode(samples: list[dict], batch: int, path: Path) -> None:
    """
    One figure per batch of 5 samples.
    Each row spans 3 columns:
      col 0 — Context (blue) + Forecast (orange), full time axis
      col 1 — Trend component over the 64-step forecast window
      col 2 — Seasonal component over the 64-step forecast window
    Forecast is shown as a faint dashed reference in cols 1 & 2.
    """
    subset = samples[batch * 5 : batch * 5 + 5]

    fig, axes = plt.subplots(5, 3, figsize=(16, 13))
    fig.suptitle(
        f"Reasoning-Mode Samples — Batch {batch + 1}   "
        f"[context  →  trend + seasonal + forecast]",
        fontsize=12, fontweight="bold", y=1.01,
    )

    col_titles = [
        "Context  +  Forecast",
        "Trend  (forecast window, 64 steps)",
        "Seasonal  (forecast window, 64 steps)",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold", pad=6)

    ctx_len  = CONTEXT_LENGTH
    fct_len  = PREDICTION_LENGTH
    ctx_x    = np.arange(ctx_len)
    fct_x    = np.arange(ctx_len, ctx_len + fct_len)
    decomp_x = np.arange(fct_len)

    for row, s in enumerate(subset):
        ctx      = np.array(s["context"],  dtype=np.float32)
        fct      = np.array(s["forecast"], dtype=np.float32)
        trend    = np.array(s["trend"],    dtype=np.float32)
        seasonal = np.array(s["seasonal"], dtype=np.float32)

        # ── col 0: full context + forecast view ──────────────────────────────
        ax = axes[row, 0]
        ax.fill_between(ctx_x, ctx, alpha=0.07, color=_CTX_COLOUR)
        ax.plot(ctx_x, ctx, color=_CTX_COLOUR, lw=0.8,
                label="Context"  if row == 0 else "_nolegend_")
        ax.plot(fct_x, fct, color=_FCT_COLOUR, lw=1.8,
                label="Forecast" if row == 0 else "_nolegend_")
        ax.axvline(ctx_len - 0.5, color="black", lw=0.9, ls="--", alpha=0.4)
        ax.set_ylabel(s["item_id"], fontsize=6, rotation=0, labelpad=50, va="center")
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

        # ── col 1: trend ─────────────────────────────────────────────────────
        ax = axes[row, 1]
        ax.plot(decomp_x, fct,   color=_FCT_COLOUR,  lw=1.0, alpha=0.30,
                ls="--", label="Forecast (ref)" if row == 0 else "_nolegend_")
        ax.plot(decomp_x, trend, color=_TRND_COLOUR, lw=1.8,
                label="Trend" if row == 0 else "_nolegend_")
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

        # ── col 2: seasonal ──────────────────────────────────────────────────
        ax = axes[row, 2]
        ax.plot(decomp_x, fct,      color=_FCT_COLOUR,  lw=1.0, alpha=0.30,
                ls="--", label="Forecast (ref)" if row == 0 else "_nolegend_")
        ax.plot(decomp_x, seasonal, color=_SEAS_COLOUR, lw=1.8,
                label="Seasonal" if row == 0 else "_nolegend_")
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

    for col in range(3):
        axes[-1, col].set_xlabel("Time step", fontsize=8)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  {path.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("SarSim0 Dataset Builder")
    print(f"  Total    : {TOTAL_SAMPLES:,}")
    print(f"  Fast     : {N_FAST:,}  (40%)")
    print(f"  Reasoning: {N_REASONING:,}  (60%)")
    print(f"  Seed     : {SEED}")
    print("=" * 60)

    rows = generate_dataset()

    print("\nSaving dataset...")
    save_arrow(rows, OUTPUT_PATH)

    fast      = [r for r in rows if r["mode"] == "fast"]
    reasoning = [r for r in rows if r["mode"] == "reasoning"]

    print("\nGenerating plots...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_fast_mode(fast,      batch=0, path=FIGURES_DIR / "sarsim0_fast_batch1.png")
    plot_fast_mode(fast,      batch=1, path=FIGURES_DIR / "sarsim0_fast_batch2.png")
    plot_reasoning_mode(reasoning, batch=0, path=FIGURES_DIR / "sarsim0_reasoning_batch1.png")
    plot_reasoning_mode(reasoning, batch=1, path=FIGURES_DIR / "sarsim0_reasoning_batch2.png")

    print(f"\nDone.\n  Dataset : {OUTPUT_PATH}\n  Figures : {FIGURES_DIR}/sarsim0_*.png")


if __name__ == "__main__":
    main()
