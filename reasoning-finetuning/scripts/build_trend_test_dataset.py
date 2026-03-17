#!/usr/bin/env python3
"""
Build sarsim0-trend-test.arrow: a testing dataset from sarsim0-10k.arrow.

Split:
  30%  fast-mode  — input context  →  original 64-step forecast (unchanged)
  70%  trend-mode — input context  →  last 64 steps of the extracted trend

Trend extraction: centered moving average (window=52, covers longest seasonal
period used by SarSim0) via scipy.ndimage.uniform_filter1d with reflect padding.

Outputs:
  data/sarsim0-trend-test.arrow
  figures/dataset_trend_test/trend_test_fast_samples.png   (5 fast-mode)
  figures/dataset_trend_test/trend_test_trend_mode_samples.png  (5 trend-mode)
"""

import sys
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from scipy.ndimage import uniform_filter1d
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
SOURCE_PATH = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
OUTPUT_PATH = FINETUNING_ROOT / "data" / "sarsim0-trend-test.arrow"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "dataset_trend_test"

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 64
TREND_WINDOW      = 52      # MA window — covers longest SarSim0 seasonal period (52)
FAST_FRACTION     = 0.30
SEED              = 42

# ── colours ───────────────────────────────────────────────────────────────────
_CTX_COL   = "steelblue"
_FCT_COL   = "darkorange"
_TREND_COL = "seagreen"


# ── trend extraction ──────────────────────────────────────────────────────────

def extract_trend(context: np.ndarray, window: int = TREND_WINDOW) -> np.ndarray:
    """
    Centered moving average with reflect padding at edges.
    Returns float32 array of same length as context.
    """
    return uniform_filter1d(
        context.astype(np.float64), size=window, mode="reflect"
    ).astype(np.float32)


# ── data loading ──────────────────────────────────────────────────────────────

def load_source(path: Path) -> list[dict]:
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()

    rows = []
    for item_id, mode, ctx, fct in zip(
        table.column("item_id").to_pylist(),
        table.column("mode").to_pylist(),
        table.column("context").to_pylist(),
        table.column("forecast").to_pylist(),
    ):
        rows.append({
            "item_id":  item_id,
            "context":  np.array(ctx, dtype=np.float32),
            "forecast": np.array(fct, dtype=np.float32),
        })
    return rows


# ── dataset construction ──────────────────────────────────────────────────────

def build_dataset(rows: list[dict], seed: int = SEED) -> list[dict]:
    rng     = np.random.default_rng(seed)
    n_total = len(rows)
    n_fast  = round(n_total * FAST_FRACTION)

    indices = np.arange(n_total)
    rng.shuffle(indices)
    fast_set = set(indices[:n_fast].tolist())

    dataset = []
    for i, row in enumerate(rows):
        if i in fast_set:
            dataset.append({
                "item_id": row["item_id"],
                "mode":    "fast",
                "context": row["context"],
                "target":  row["forecast"],          # original forecast unchanged
            })
        else:
            trend = extract_trend(row["context"])
            dataset.append({
                "item_id": row["item_id"],
                "mode":    "trend",
                "context": row["context"],
                "target":  trend[-PREDICTION_LENGTH:],   # last 64 steps of trend
            })

    return dataset


# ── Arrow save ────────────────────────────────────────────────────────────────

def save_arrow(dataset: list[dict], path: Path) -> None:
    schema = pa.schema([
        pa.field("item_id", pa.string()),
        pa.field("mode",    pa.string()),
        pa.field("context", pa.list_(pa.float32())),
        pa.field("target",  pa.list_(pa.float32())),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        writer = ipc.new_file(sink, schema)
        batch  = pa.record_batch({
            "item_id": [d["item_id"] for d in dataset],
            "mode":    [d["mode"]    for d in dataset],
            "context": [d["context"].tolist() for d in dataset],
            "target":  [d["target"].tolist()  for d in dataset],
        }, schema=schema)
        writer.write_batch(batch)
        writer.close()
    print(f"  Saved: {path}  ({len(dataset):,} rows)")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_fast_samples(dataset: list[dict], path: Path, n: int = 5) -> None:
    """5 fast-mode samples: context → target forecast."""
    fast = [d for d in dataset if d["mode"] == "fast"]
    rng  = np.random.default_rng(0)
    picks = rng.choice(len(fast), size=min(n, len(fast)), replace=False)
    picks.sort()

    ctx_x = np.arange(CONTEXT_LENGTH)
    fct_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        "Trend-Test Dataset  ·  Fast-Mode Samples\n"
        "[input context  →  target forecast]",
        fontsize=12, fontweight="bold",
    )

    for row_idx, pick in enumerate(picks):
        ax = axes[row_idx]
        d  = fast[pick]

        ax.fill_between(ctx_x, d["context"], alpha=0.08, color=_CTX_COL)
        ax.plot(ctx_x, d["context"], color=_CTX_COL, lw=0.8, label="Input context")
        ax.plot(fct_x, d["target"],  color=_FCT_COL, lw=1.6, label="Target forecast")
        ax.axvline(CONTEXT_LENGTH - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(d["item_id"], fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.7)
    axes[-1].set_xlabel("Time step", fontsize=9)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_trend_samples(dataset: list[dict], path: Path, n: int = 5) -> None:
    """
    5 trend-mode samples, two columns each:
      Left  — input context (512 steps) followed by target trend (64 steps),
              plotted as a continuous sequence with a split line
      Right — target trend (64 steps) in isolation
    """
    trend_rows = [d for d in dataset if d["mode"] == "trend"]
    rng   = np.random.default_rng(0)
    picks = rng.choice(len(trend_rows), size=min(n, len(trend_rows)), replace=False)
    picks.sort()

    ctx_x = np.arange(CONTEXT_LENGTH)
    tgt_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)
    iso_x = np.arange(PREDICTION_LENGTH)

    fig, axes_raw = plt.subplots(n, 2, figsize=(18, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        "Trend-Test Dataset  ·  Trend-Mode Samples\n"
        "Left: input context → target trend (64 steps)   "
        "Right: target trend isolated",
        fontsize=12, fontweight="bold",
    )

    for row_idx, pick in enumerate(picks):
        d    = trend_rows[pick]
        ax_l = axes[row_idx, 0]
        ax_r = axes[row_idx, 1]

        # ── Left: context then trend target as continuation ───────────────────
        ax_l.fill_between(ctx_x, d["context"], alpha=0.08, color=_CTX_COL)
        ax_l.plot(ctx_x, d["context"], color=_CTX_COL,   lw=0.8, label="Input context")
        ax_l.fill_between(tgt_x, d["target"],  alpha=0.18, color=_TREND_COL)
        ax_l.plot(tgt_x, d["target"],  color=_TREND_COL, lw=1.8, label="Target trend")
        ax_l.axvline(CONTEXT_LENGTH - 0.5, color="black", lw=0.9, ls="--", alpha=0.4)
        ax_l.set_title(d["item_id"], fontsize=8, loc="left", pad=2)
        ax_l.tick_params(labelsize=7)
        ax_l.spines[["top", "right"]].set_visible(False)

        # ── Right: target trend alone ─────────────────────────────────────────
        ax_r.fill_between(iso_x, d["target"], alpha=0.18, color=_TREND_COL)
        ax_r.plot(iso_x, d["target"], color=_TREND_COL, lw=1.8, label="Target trend")
        ax_r.set_xlabel("Forecast step", fontsize=7)
        ax_r.tick_params(labelsize=7)
        ax_r.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.7)
    axes[0, 1].legend(fontsize=8, loc="upper left", framealpha=0.7)
    axes[-1, 0].set_xlabel("Time step", fontsize=9)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Building SarSim0 Trend-Test Dataset")
    print(f"  Source  : {SOURCE_PATH}")
    print(f"  Output  : {OUTPUT_PATH}")
    print(f"  Split   : {int(FAST_FRACTION*100)}% fast / "
          f"{int((1 - FAST_FRACTION)*100)}% trend")
    print(f"  MA win  : {TREND_WINDOW}")
    print("=" * 60)

    print("\nLoading source dataset...")
    rows = load_source(SOURCE_PATH)
    print(f"  Loaded {len(rows):,} rows")

    print("\nBuilding trend-test dataset...")
    dataset = build_dataset(rows)
    n_fast  = sum(1 for d in dataset if d["mode"] == "fast")
    n_trend = sum(1 for d in dataset if d["mode"] == "trend")
    print(f"  Fast-mode  : {n_fast:,}  ({n_fast / len(dataset) * 100:.1f}%)")
    print(f"  Trend-mode : {n_trend:,}  ({n_trend / len(dataset) * 100:.1f}%)")

    print("\nSaving Arrow file...")
    save_arrow(dataset, OUTPUT_PATH)

    print("\nGenerating plots...")
    plot_fast_samples(
        dataset,
        FIGURES_DIR / "trend_test_fast_samples.png",
    )
    plot_trend_samples(
        dataset,
        FIGURES_DIR / "trend_test_trend_mode_samples.png",
    )

    print(f"\nDone.  Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
