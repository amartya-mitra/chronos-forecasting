#!/usr/bin/env python3
"""
Build sarsim0-ts.arrow from sarsim0-10k.arrow.

Uses only the 6,000 "reasoning" rows (fast rows have empty trend/seasonal).
Mode tag: "TS".
  context (512 steps) → target = [trend(64) | seasonal(64)]   (128 steps total)

Per-component scale normalisation
----------------------------------
Each component is rescaled to context-scale magnitude before concatenation:

    trend_stored    = trend    * (s_ctx / max(s_trend,    eps))
    seasonal_stored = seasonal * (s_ctx / max(s_seasonal, eps))

After the tokenizer divides by s_ctx both halves are O(1) regardless of the
natural amplitude ratio.  Original component scales are stored as s_first
(trend) and s_second (seasonal) for downstream inverse-transform at eval time.

Outputs:
  data/sarsim0-ts.arrow
  figures/dataset_ts/samples.png   (5 sample plots)
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
OUTPUT_PATH = FINETUNING_ROOT / "data" / "sarsim0-ts.arrow"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "datasets" / "dataset_ts"

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 128   # 64 trend + 64 seasonal
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
        s_trend    = float(np.mean(np.abs(r["trend"])))
        s_seasonal = float(np.mean(np.abs(r["seasonal"])))

        # Rescale each component to context-scale magnitude so that after the
        # tokenizer divides by s_ctx both halves occupy O(1) in bin-space.
        trend_norm    = r["trend"]    * (s_ctx / max(s_trend,    SCALE_EPS))
        seasonal_norm = r["seasonal"] * (s_ctx / max(s_seasonal, SCALE_EPS))

        out.append({
            "item_id":  r["item_id"],
            "mode":     "TS",
            "context":  r["context"],
            # TS order: [trend | seasonal], both at context-scale magnitude
            "target":   np.concatenate([trend_norm, seasonal_norm]),   # (128,)
            "s_first":  s_trend,      # original scale of first  half (trend)
            "s_second": s_seasonal,   # original scale of second half (seasonal)
        })
    return out


def save_arrow(dataset: list[dict], path: Path) -> None:
    schema = pa.schema([
        pa.field("item_id", pa.string()),
        pa.field("mode",    pa.string()),
        pa.field("context", pa.list_(pa.float32())),
        pa.field("target",  pa.list_(pa.float32())),
        pa.field("s_first",  pa.float32()),   # original scale of trend half
        pa.field("s_second", pa.float32()),   # original scale of seasonal half
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
    trd_x  = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + 64)
    seas_x = np.arange(CONTEXT_LENGTH + 64, CONTEXT_LENGTH + 128)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        "TS Dataset  ·  Sample Plots  (targets rescaled to context-scale magnitude)\n"
        "[TS-prefix  |  context(512)  →  trend_norm(64) | seasonal_norm(64)]",
        fontsize=12, fontweight="bold",
    )
    for idx, pick in enumerate(picks):
        ax = axes[idx]
        d  = dataset[pick]
        trd  = d["target"][:64]
        seas = d["target"][64:]
        ax.fill_between(ctx_x, d["context"], alpha=0.08, color=_CTX_COL)
        ax.plot(ctx_x,  d["context"], color=_CTX_COL,      lw=0.7, label="Input context")
        ax.plot(trd_x,  trd,          color=_TREND_COL,    lw=1.6, label="Trend (64)")
        ax.plot(seas_x, seas,         color=_SEASONAL_COL, lw=1.6, label="Seasonal (64)")
        ax.axvline(CONTEXT_LENGTH - 0.5,      color="black", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(CONTEXT_LENGTH + 64 - 0.5, color="grey",  lw=0.6, ls=":",  alpha=0.5,
                   label="T|S split")
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
    print("Building SarSim0 TS Dataset  (trend | seasonal, 128 steps)")
    print(f"  Source : {SOURCE_PATH}")
    print(f"  Output : {OUTPUT_PATH}")
    print("=" * 60)

    print("\nLoading source (reasoning rows only)...")
    rows = load_source(SOURCE_PATH)
    print(f"  Loaded {len(rows):,} rows")

    print("\nBuilding TS dataset...")
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
