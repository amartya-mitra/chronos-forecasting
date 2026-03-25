#!/usr/bin/env python3
"""
Build m4monthly-tsf.arrow from GiftEval M4 Monthly.

For each series (up to 10,000):
  - Context  : last 36 months before the forecast window
  - STL decomposition (period=12, robust) applied to history only (no look-ahead)
  - Target   : [trend(18) | seasonal(18) | forecast(18)]  (54 steps total)
  - Per-component normalization identical to the SarSim0 TSF approach

Stored per-row scale factors (original mean absolute values):
  s_first  — mean |trend_18|
  s_second — mean |seasonal_18|
  s_third  — mean |forecast_18|

M4 Monthly horizon: 18 months  |  seasonal period: 12
Context length    : 36 months  (3 full seasonal cycles, all M4 series have ≥42 training obs)

Output:
  data/m4monthly-tsf.arrow
  figures/dataset_m4monthly_tsf/samples.png
"""

import sys
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import datasets as hf_datasets
except ImportError:
    raise ImportError("pip install datasets huggingface_hub")

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:
    raise ImportError("pip install statsmodels")

# ── constants ─────────────────────────────────────────────────────────────────
OUTPUT_PATH = FINETUNING_ROOT / "data"    / "m4monthly-tsf.arrow"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "dataset_m4monthly_tsf"

CONTEXT_LEN     = 36   # months of context  (3 years)
HORIZON         = 18   # M4 Monthly forecast horizon
SEASONAL_PERIOD = 12   # monthly seasonality
MAX_SERIES      = 10_000
SCALE_EPS       = 1e-6

# Minimum series length: need CONTEXT_LEN + HORIZON history, plus some for STL
MIN_SERIES_LEN = CONTEXT_LEN + HORIZON + SEASONAL_PERIOD  # = 66

HF_REPO   = "Salesforce/GiftEval"
HF_COMMIT = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
HF_PATH   = "m4_monthly/data-00000-of-00001.arrow"

_CTX_COL      = "steelblue"
_TREND_COL    = "seagreen"
_SEASONAL_COL = "mediumpurple"
_FORECAST_COL = "darkorange"


# ── helpers ───────────────────────────────────────────────────────────────────

def fill_nans(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation for internal NaNs; forward/back fill for edges."""
    idx   = np.arange(len(arr))
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return np.zeros_like(arr)
    return np.interp(idx, idx[valid], arr[valid])


def process_series(full: np.ndarray, item_id: str) -> dict | None:
    """
    Decompose one M4 Monthly series.

    full : complete time series including the 18-month test window at the end.
    Returns a dataset row dict, or None if the series should be skipped.
    """
    if len(full) < MIN_SERIES_LEN:
        return None

    # Handle NaNs
    if np.isnan(full).all():
        return None
    if np.isnan(full).any():
        full = fill_nans(full)

    history = full[:-HORIZON]          # everything before the test window
    future  = full[-HORIZON:].astype(np.float32)   # the 18-month test window

    if np.isnan(future).any():
        return None

    # Context: last CONTEXT_LEN months of history (guaranteed non-empty)
    context = history[-CONTEXT_LEN:].astype(np.float32)   # (36,)
    if len(context) < CONTEXT_LEN:
        # Left-pad with zeros if history is somehow shorter than 36
        pad = np.zeros(CONTEXT_LEN - len(context), dtype=np.float32)
        context = np.concatenate([pad, context])

    # STL decomposition of the HISTORY only (no look-ahead into future).
    # trend_18 and seasonal_18 are the last HORIZON steps of the historical
    # decomposition — consistent with SarSim0 where trend/seasonal come from
    # the historical context, not the forecast window.
    try:
        stl    = STL(history.astype(np.float64), period=SEASONAL_PERIOD, robust=True).fit()
        trend_18    = stl.trend[-HORIZON:].astype(np.float32)    # (18,) — last 18 months of history
        seasonal_18 = stl.seasonal[-HORIZON:].astype(np.float32) # (18,) — last 18 months of history
    except Exception:
        return None

    # Per-component normalization: each segment rescaled to context-scale magnitude
    s_ctx      = max(float(np.mean(np.abs(context))), SCALE_EPS)
    s_trend    = max(float(np.mean(np.abs(trend_18))),    SCALE_EPS)
    s_seasonal = max(float(np.mean(np.abs(seasonal_18))), SCALE_EPS)
    s_forecast = max(float(np.mean(np.abs(future))),      SCALE_EPS)

    trend_norm    = trend_18    * (s_ctx / s_trend)
    seasonal_norm = seasonal_18 * (s_ctx / s_seasonal)
    forecast_norm = future      * (s_ctx / s_forecast)

    return {
        "item_id":  item_id,
        "mode":     "TSF",
        "context":  context,                                         # (36,)
        "target":   np.concatenate([trend_norm, seasonal_norm, forecast_norm]),  # (54,)
        "s_first":  s_trend,
        "s_second": s_seasonal,
        "s_third":  s_forecast,
    }


# ── load ──────────────────────────────────────────────────────────────────────

def load_and_build(max_series: int) -> list[dict]:
    url = f"hf://datasets/{HF_REPO}@{HF_COMMIT}/{HF_PATH}"
    print(f"  Streaming from: {url}")

    ds = hf_datasets.load_dataset(
        "arrow", data_files=url, split="train", streaming=True
    )

    dataset   = []
    n_loaded  = 0
    n_skipped = 0

    bar = tqdm(ds, desc="Processing M4 Monthly series", total=max_series, unit="series")
    for entry in bar:
        if len(dataset) >= max_series:
            break

        target = entry.get("target")
        if target is None:
            n_skipped += 1
            continue

        full = np.array(target, dtype=np.float64)
        if full.ndim > 1:
            full = full[0]   # multivariate: take first channel

        # Derive item_id from metadata if available, else enumerate
        item_id = entry.get("item_id") or entry.get("feat_static_cat") or f"M_Monthly_{n_loaded}"
        if isinstance(item_id, list):
            item_id = f"M_Monthly_{item_id[0]}" if item_id else f"M_Monthly_{n_loaded}"
        item_id = str(item_id)

        row = process_series(full, item_id)
        if row is None:
            n_skipped += 1
        else:
            dataset.append(row)
            n_loaded += 1

        bar.set_postfix({"ok": n_loaded, "skip": n_skipped})

    bar.close()
    print(f"\n  Collected {len(dataset):,} rows  |  Skipped {n_skipped:,}")
    return dataset


# ── verify ────────────────────────────────────────────────────────────────────

def verify(dataset: list[dict]) -> None:
    PRED_LEN = HORIZON * 3
    errors = [d["item_id"] for d in dataset if len(d["target"]) != PRED_LEN]
    ctx_errors = [d["item_id"] for d in dataset if len(d["context"]) != CONTEXT_LEN]
    if errors:
        print(f"  TARGET LENGTH ERRORS: {len(errors)} rows")
    if ctx_errors:
        print(f"  CONTEXT LENGTH ERRORS: {len(ctx_errors)} rows")
    if not errors and not ctx_errors:
        print(f"  All {len(dataset):,} rows: context={CONTEXT_LEN}, target={PRED_LEN} ✓")


# ── save ──────────────────────────────────────────────────────────────────────

def save_arrow(dataset: list[dict], path: Path) -> None:
    schema = pa.schema([
        pa.field("item_id",  pa.string()),
        pa.field("mode",     pa.string()),
        pa.field("context",  pa.list_(pa.float32())),
        pa.field("target",   pa.list_(pa.float32())),
        pa.field("s_first",  pa.float32()),
        pa.field("s_second", pa.float32()),
        pa.field("s_third",  pa.float32()),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        writer = ipc.new_file(sink, schema)
        writer.write_batch(pa.record_batch({
            "item_id":  [d["item_id"]  for d in dataset],
            "mode":     [d["mode"]     for d in dataset],
            "context":  [d["context"].tolist() for d in dataset],
            "target":   [d["target"].tolist()  for d in dataset],
            "s_first":  [float(d["s_first"])   for d in dataset],
            "s_second": [float(d["s_second"])  for d in dataset],
            "s_third":  [float(d["s_third"])   for d in dataset],
        }, schema=schema))
        writer.close()
    print(f"  Saved: {path}  ({len(dataset):,} rows)")


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_samples(dataset: list[dict], path: Path, n: int = 5) -> None:
    rng    = np.random.default_rng(0)
    picks  = sorted(rng.choice(len(dataset), size=min(n, len(dataset)), replace=False))
    ctx_x  = np.arange(CONTEXT_LEN)
    trd_x  = np.arange(CONTEXT_LEN,          CONTEXT_LEN + HORIZON)
    seas_x = np.arange(CONTEXT_LEN + HORIZON, CONTEXT_LEN + 2 * HORIZON)
    fct_x  = np.arange(CONTEXT_LEN + 2*HORIZON, CONTEXT_LEN + 3 * HORIZON)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        "M4 Monthly TSF Dataset  ·  Sample Plots\n"
        f"[TSF-token  |  context({CONTEXT_LEN})  →  trend_norm({HORIZON})"
        f" | seasonal_norm({HORIZON}) | forecast_norm({HORIZON})]",
        fontsize=11, fontweight="bold",
    )
    for idx, pick in enumerate(picks):
        ax   = axes[idx]
        d    = dataset[pick]

        # Inverse-transform from context-scale-normalised back to original units
        s_ctx = max(float(np.mean(np.abs(d["context"]))), SCALE_EPS)
        trd  = d["target"][:HORIZON]        * (d["s_first"]  / s_ctx)
        seas = d["target"][HORIZON:2*HORIZON] * (d["s_second"] / s_ctx)
        fct  = d["target"][2*HORIZON:]      * (d["s_third"]  / s_ctx)

        ax.fill_between(ctx_x, d["context"], alpha=0.08, color=_CTX_COL)
        ax.plot(ctx_x,  d["context"], color=_CTX_COL,      lw=0.7, label="Input context (36 mo)")
        ax.plot(trd_x,  trd,          color=_TREND_COL,    lw=1.6, label=f"Trend ({HORIZON})")
        ax.plot(seas_x, seas,         color=_SEASONAL_COL, lw=1.6, label=f"Seasonal ({HORIZON})")
        ax.plot(fct_x,  fct,          color=_FORECAST_COL, lw=1.6, label=f"Forecast ({HORIZON})")
        ax.axvline(CONTEXT_LEN - 0.5,           color="black", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(CONTEXT_LEN + HORIZON - 0.5,   color="grey",  lw=0.6, ls=":",  alpha=0.5,
                   label="T|S split")
        ax.axvline(CONTEXT_LEN + 2*HORIZON - 0.5, color="grey",  lw=0.6, ls=":",  alpha=0.5,
                   label="S|F split")
        ax.set_title(d["item_id"], fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.7)
    axes[-1].set_xlabel("Time step (months)", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Building M4 Monthly TSF Dataset")
    print(f"  context={CONTEXT_LEN} mo  |  horizon={HORIZON} mo  |  period={SEASONAL_PERIOD}")
    print(f"  Target: trend({HORIZON}) | seasonal({HORIZON}) | forecast({HORIZON})  = {3*HORIZON} steps")
    print(f"  Max series: {MAX_SERIES:,}")
    print(f"  Output : {OUTPUT_PATH}")
    print("=" * 65)

    print("\nLoading + decomposing M4 Monthly series...")
    dataset = load_and_build(MAX_SERIES)

    if len(dataset) == 0:
        print("ERROR: No valid series collected. Check HuggingFace access.")
        sys.exit(1)

    print("\nVerifying...")
    verify(dataset)

    print("\nSaving Arrow file...")
    save_arrow(dataset, OUTPUT_PATH)

    print("\nGenerating sample plots...")
    plot_samples(dataset, FIGURES_DIR / "samples.png")

    print(f"\nDone.  {len(dataset):,} rows  →  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
