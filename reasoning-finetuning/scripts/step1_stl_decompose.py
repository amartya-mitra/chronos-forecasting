#!/usr/bin/env python3
"""
prefix_tuning.md — Step 1
  1. Verify Chronos forward pass
  2. Define detect_period (FFT-based, per Failure Mode 1 recommendation)
     Define stl_decompose (thin wrapper over prepare_dataset.compute_decomposition logic,
     full-length output, residual instead of volatility)
  3. Run STL on first 3 series from m4hourly-tsf.arrow, plot 3x3 grid,
     save stl_sanity_check.png

Dataset: reasoning-finetuning/data/m4hourly-tsf.arrow
  (GiftEval M4 Hourly — using 'context' column, 96-step hourly series)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "reasoning-finetuning"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# ──────────────────────────────────────────────────────────────────────────────
# Step 1a: Verify Chronos forward pass
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1a: Chronos forward pass verification")
print("=" * 60)

import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    dtype=torch.float32,
)

model = pipeline.model
device = next(model.parameters()).device
vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else "N/A"

print(f"  Model device  : {device}")
print(f"  Vocab size    : {vocab_size}")

ctx = torch.randn(1, 512)
with torch.no_grad():
    samples = pipeline.predict(ctx, prediction_length=64, num_samples=4)
print(f"  Output shape  : {samples.shape}  (batch x num_samples x horizon)")
print("  Forward pass  : OK\n")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1b: detect_periods + stl_decompose (STL for single period, MSTL for multi)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1b: detect_periods + stl_decompose definitions")
print("=" * 60)

from statsmodels.tsa.seasonal import MSTL


def detect_periods(
    series: np.ndarray,
    max_periods: int = 3,
    min_period: int = 2,
    max_period: int | None = None,
    magnitude_ratio_thresh: float = 0.15,
) -> list[int]:
    """
    FFT-based multi-period detection (Failure Mode 1 + multi-frequency mitigation).

    Iteratively picks the highest-magnitude FFT bin, converts to period,
    suppresses a window around it, then repeats up to max_periods times.
    Secondary periods are only included if their magnitude is >=
    magnitude_ratio_thresh × the dominant peak (avoids noise peaks).

    max_period defaults to n // 4 to exclude near-trend long-period components.
    A period of (n-1)//2 would cycle only twice across the context — MSTL would
    use it to absorb slow drift rather than genuine seasonality, collapsing the
    residual to near zero. Capping at n // 4 keeps only periods that cycle at
    least 4 times within the context window.

    Args:
        series:                 1-D float array
        max_periods:            maximum number of periods to return
        min_period:             lower clamp on any detected period (default 2)
        max_period:             upper clamp on any detected period;
                                defaults to n // 4 if None
        magnitude_ratio_thresh: fraction of dominant magnitude required for
                                secondary peaks (default 0.15)

    Returns:
        Sorted list of dominant periods (length 1 to max_periods)
    """
    n = len(series)
    effective_max = (n // 4) if max_period is None else max_period

    x = series.astype(np.float64) - series.mean()
    mags = np.abs(np.fft.rfft(x))
    mags[0] = 0.0  # zero out DC

    dominant_mag = mags.max()
    if dominant_mag == 0.0:
        return [min_period]

    mags_work = mags.copy()
    periods: list[int] = []

    for _ in range(max_periods):
        k = int(np.argmax(mags_work))
        if k == 0 or mags_work[k] < magnitude_ratio_thresh * dominant_mag:
            break
        period = round(n / k)
        period = max(min_period, min(period, effective_max))
        if period not in periods:
            periods.append(period)
        # Suppress a half-width window around this bin before next search
        half_w = max(1, k // 4)
        mags_work[max(0, k - half_w): k + half_w + 1] = 0.0

    return sorted(periods) if periods else [min_period]


def stl_decompose(
    series: np.ndarray,
    periods: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose series → (trend, combined_seasonal, residual), full-length output.

    Reuses the same cleaning/padding/fallback logic as
    prepare_dataset.compute_decomposition(), extended for multiple periods:
      - ffill/bfill NaN cleaning
      - padding to 2 × max(periods) if series is too short
      - robust STL for a single period; MSTL for two or more
      - combined_seasonal = sum of per-period seasonal components (MSTL case)
      - rolling-mean fallback on decomposition failure

    Unlike compute_decomposition() this function:
      - Does NOT truncate to DECOMPOSITION_LENGTH=64
      - Returns residual (not volatility / rolling std of residual)

    Args:
        series:  1-D float array (context window)
        periods: list of dominant seasonality periods (from detect_periods())

    Returns:
        (trend, seasonal, residual) — each np.ndarray of same length as series
    """
    series_clean: np.ndarray = np.asarray(
        pd.Series(series.astype(np.float64)).ffill().bfill(), dtype=np.float64
    )
    orig_len = len(series)

    min_len_needed = 2 * max(periods)
    if len(series_clean) < min_len_needed:
        pad_len = min_len_needed - len(series_clean)
        series_clean = np.concatenate(
            [np.full(pad_len, series_clean[0], dtype=np.float64), series_clean]
        )

    try:
        if len(periods) == 1:
            result = STL(series_clean, period=periods[0], robust=True).fit()
            trend:    np.ndarray = np.asarray(result.trend,    dtype=np.float64)
            seasonal: np.ndarray = np.asarray(result.seasonal, dtype=np.float64)
            residual: np.ndarray = np.asarray(result.resid,    dtype=np.float64)
        else:
            result = MSTL(series_clean, periods=tuple(periods)).fit()
            trend    = np.asarray(result.trend,               dtype=np.float64)
            seasonal = result.seasonal.sum(axis=1).astype(np.float64)  # sum components
            residual = np.asarray(result.resid,               dtype=np.float64)
    except Exception as e:
        print(f"  Warning: decomposition failed ({e}), using rolling fallback")
        p = periods[0]
        trend    = np.asarray(
            pd.Series(series_clean).rolling(window=p, min_periods=1).mean(),
            dtype=np.float64,
        )
        seasonal = (series_clean - trend).astype(np.float64)
        residual = (series_clean - trend - seasonal).astype(np.float64)

    if len(trend) > orig_len:
        trend    = trend[-orig_len:]
        seasonal = seasonal[-orig_len:]
        residual = residual[-orig_len:]

    return trend.astype(np.float32), seasonal.astype(np.float32), residual.astype(np.float32)


print("  detect_periods defined (FFT top-N, magnitude threshold, MSTL-safe upper bound)")
print("  stl_decompose defined  (STL for 1 period · MSTL for 2+, combined seasonal)\n")


# ──────────────────────────────────────────────────────────────────────────────
# Shared plotting helper
# ──────────────────────────────────────────────────────────────────────────────
FIG_DIR = REPO_ROOT / "reasoning-finetuning" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

component_names = ["Trend", "Seasonal (combined)", "Residual"]
colors = ["#2196F3", "#FF9800", "#9C27B0"]


def run_and_plot(dataset_label: str, data_path: Path, indices: list[int],
                 context_col: str, save_path: Path) -> None:
    """Load series, detect periods, run STL/MSTL, plot N×3 grid."""
    with pa.memory_map(str(data_path), "r") as src:
        table = ipc.open_file(src).read_all()

    results = []
    for idx in indices:
        series  = np.array(table[context_col][idx].as_py(), dtype=np.float32)
        periods = detect_periods(series)
        trend, seasonal, residual = stl_decompose(series, periods)
        var_ratio = float(np.var(residual)) / (float(np.var(series)) + 1e-8)
        status = "OK" if var_ratio < 0.5 else "WARN (>0.5)"
        print(f"  idx={idx:>5}  periods={str(periods):<22}  "
              f"resid/sig={var_ratio:.4f}  [{status}]")
        results.append((idx, series, periods, trend, seasonal, residual, var_ratio))

    N = len(results)
    fig, axes = plt.subplots(N, 3, figsize=(15, 3.2 * N))
    if N == 1:
        axes = axes[np.newaxis, :]

    for row, (idx, series, periods, trend, seasonal, residual, var_ratio) in enumerate(results):
        t = np.arange(len(series))
        for col, (comp, name, color) in enumerate(zip(
            [trend, seasonal, residual], component_names, colors
        )):
            ax = axes[row][col]
            ax.plot(t, series, color="gray", alpha=0.25, linewidth=0.7, label="raw")
            ax.plot(t, comp,   color=color,  linewidth=1.1, label=name)
            ax.set_title(f"[{idx}] {name}  periods={periods}", fontsize=8)
            ax.set_xlabel("Time step", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper right")
            if col == 2:
                ann_color = "red" if var_ratio >= 0.5 else "green"
                ax.text(0.02, 0.95, f"resid/sig={var_ratio:.3f}",
                        transform=ax.transAxes, fontsize=7,
                        verticalalignment="top", color=ann_color)

    fig.suptitle(
        f"STL/MSTL Sanity Check — {dataset_label}\n"
        "Periods auto-detected via FFT  |  1 period → STL · 2+ → MSTL",
        fontsize=10, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1c-i: GiftEval M4 Hourly — first 3 series
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1c-i: m4hourly-tsf.arrow  (96-step hourly context)")
print("=" * 60)
run_and_plot(
    dataset_label="GiftEval M4 Hourly (m4hourly-tsf.arrow)",
    data_path=REPO_ROOT / "reasoning-finetuning/data/m4hourly-tsf.arrow",
    indices=[0, 1, 2],
    context_col="context",
    save_path=FIG_DIR / "datasets/dataset_stl_m4hourly/stl_sanity_check_m4hourly.png",
)

# ──────────────────────────────────────────────────────────────────────────────
# Step 1c-ii: SarSim0 — 6 diverse series
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1c-ii: sarsim0-10k.arrow  (512-step synthetic context)")
print("=" * 60)
run_and_plot(
    dataset_label="SarSim0 (sarsim0-10k.arrow)",
    data_path=REPO_ROOT / "reasoning-finetuning/data/sarsim0-10k.arrow",
    indices=[0, 50, 200, 500, 1000, 3000],
    context_col="context",
    save_path=FIG_DIR / "datasets/dataset_stl_sarsim0/stl_sanity_check_sarsim0_mstl.png",
)

print("=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)
