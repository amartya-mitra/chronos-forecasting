#!/usr/bin/env python3
"""
Build sarsim0-st.arrow from sarsim0-10k.arrow.

Uses only the 6,000 "reasoning" rows (fast rows have empty trend/seasonal).
Mode tag: "ST".
  context (512 steps) → target = [seasonal(64) | trend(64)]   (128 steps total)

Per-component scale normalisation
----------------------------------
The Chronos tokenizer divides the full 128-step target by a single context
scale (mean |context|).  If seasonal and trend have very different amplitudes,
one half will be compressed near zero in bin-space.

Fix: before concatenation each component is rescaled so that its magnitude
matches the context scale:

    seasonal_stored = seasonal * (s_ctx / max(s_seasonal, eps))
    trend_stored    = trend    * (s_ctx / max(s_trend,    eps))

After the tokenizer divides by s_ctx both halves are O(1) regardless of the
natural amplitude ratio.  The original component scales are stored as s_first
(seasonal) and s_second (trend) for downstream inverse-transform at eval time.

Outputs:
  data/sarsim0-st.arrow
  figures/dataset_st/samples.png   (5 sample plots)
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

SOURCE_PATH = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
OUTPUT_PATH = FINETUNING_ROOT / "data" / "sarsim0-st.arrow"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "datasets" / "dataset_st"

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 128   # 64 seasonal + 64 trend
SCALE_EPS         = 1e-6  # guard against near-zero component scale

_CTX_COL      = "steelblue"
_SEASONAL_COL = "mediumpurple"
_TREND_COL    = "seagreen"


def load_source(path: Path) -> list[dict]:
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()
    rows = []
    for item_id, mode, ctx, seasonal, trend in zip(
        table.column("item_id").to_pylist(),
        table.column("mode").to_pylist(),
        table.column("context").to_pylist(),
        table.column("seasonal").to_pylist(),
        table.column("trend").to_pylist(),
    ):
        if mode != "reasoning":
            continue
        rows.append({
            "item_id":  item_id,
            "context":  np.array(ctx,      dtype=np.float32),
            "seasonal": np.array(seasonal, dtype=np.float32),   # (64,)
            "trend":    np.array(trend,    dtype=np.float32),   # (64,)
        })
    return rows


def build_dataset(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        s_ctx      = float(np.mean(np.abs(r["context"])))
        s_seasonal = float(np.mean(np.abs(r["seasonal"])))
        s_trend    = float(np.mean(np.abs(r["trend"])))

        # Rescale each component to context-scale magnitude so that after the
        # tokenizer divides by s_ctx both halves occupy O(1) in bin-space.
        seasonal_norm = r["seasonal"] * (s_ctx / max(s_seasonal, SCALE_EPS))
        trend_norm    = r["trend"]    * (s_ctx / max(s_trend,    SCALE_EPS))

        out.append({
            "item_id":  r["item_id"],
            "mode":     "ST",
            "context":  r["context"],
            # ST order: [seasonal | trend], both at context-scale magnitude
            "target":   np.concatenate([seasonal_norm, trend_norm]),   # (128,)
            "s_first":  s_seasonal,   # original scale of first  half (seasonal)
            "s_second": s_trend,      # original scale of second half (trend)
        })
    return out


def save_arrow(dataset: list[dict], path: Path) -> None:
    schema = pa.schema([
        pa.field("item_id", pa.string()),
        pa.field("mode",    pa.string()),
        pa.field("context", pa.list_(pa.float32())),
        pa.field("target",  pa.list_(pa.float32())),
        pa.field("s_first",  pa.float32()),   # original scale of seasonal half
        pa.field("s_second", pa.float32()),   # original scale of trend half
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        writer = ipc.new_file(sink, schema)
        writer.write_batch(pa.record_batch({
            "item_id":  [d["item_id"]  for d in dataset],
            "mode":     [d["mode"]     for d in dataset],
            "context":  [d["context"].tolist() for d in dataset],
            "target":   [d["target"].tolist()  for d in dataset],
            "s_first":  [d["s_first"]  for d in dataset],
            "s_second": [d["s_second"] for d in dataset],
        }, schema=schema))
        writer.close()
    print(f"  Saved: {path}  ({len(dataset):,} rows)")


def plot_samples(dataset: list[dict], path: Path, n: int = 5) -> None:
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(dataset), size=min(n, len(dataset)), replace=False))
    ctx_x  = np.arange(CONTEXT_LENGTH)
    seas_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + 64)
    trd_x  = np.arange(CONTEXT_LENGTH + 64, CONTEXT_LENGTH + 128)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        "ST Dataset  ·  Sample Plots  (targets rescaled to context-scale magnitude)\n"
        "[ST-prefix  |  context(512)  →  seasonal_norm(64) | trend_norm(64)]",
        fontsize=12, fontweight="bold",
    )
    for idx, pick in enumerate(picks):
        ax = axes[idx]
        d  = dataset[pick]
        seas = d["target"][:64]
        trd  = d["target"][64:]
        ax.fill_between(ctx_x, d["context"], alpha=0.08, color=_CTX_COL)
        ax.plot(ctx_x,  d["context"], color=_CTX_COL,      lw=0.7, label="Input context")
        ax.plot(seas_x, seas,         color=_SEASONAL_COL, lw=1.6, label="Seasonal (64)")
        ax.plot(trd_x,  trd,          color=_TREND_COL,    lw=1.6, label="Trend (64)")
        ax.axvline(CONTEXT_LENGTH - 0.5,      color="black", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(CONTEXT_LENGTH + 64 - 0.5, color="grey",  lw=0.6, ls=":",  alpha=0.5,
                   label="S|T split")
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


def verify(dataset: list[dict]) -> None:
    errors = [d["item_id"] for d in dataset if len(d["target"]) != PREDICTION_LENGTH]
    if errors:
        print(f"  ERRORS: {len(errors)} rows with wrong target length")
    else:
        print(f"  All {len(dataset):,} rows: target_len={PREDICTION_LENGTH} ✓")


def main() -> None:
    print("=" * 60)
    print("Building SarSim0 ST Dataset  (seasonal | trend, 128 steps)")
    print(f"  Source : {SOURCE_PATH}")
    print(f"  Output : {OUTPUT_PATH}")
    print("=" * 60)

    print("\nLoading source (reasoning rows only)...")
    rows = load_source(SOURCE_PATH)
    print(f"  Loaded {len(rows):,} rows")

    print("\nBuilding ST dataset...")
    dataset = build_dataset(rows)

    print("\nVerifying...")
    verify(dataset)

    print("\nSaving Arrow file...")
    save_arrow(dataset, OUTPUT_PATH)

    print("\nGenerating plots...")
    plot_samples(dataset, FIGURES_DIR / "samples.png")

    print(f"\nDone.  Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
