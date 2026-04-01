#!/usr/bin/env python3
"""
Build sarsim0-tsf.arrow from sarsim0-10k.arrow.

Uses only the 6,000 "reasoning" rows.
Mode tag: "TSF".
  context (512) → target = [trend(64) | seasonal(64) | forecast(64)]  (192 steps total)

Per-component scale normalisation: see build_stf_dataset.py for details.

Stored per-row scale factors:
  s_first  — mean |trend|
  s_second — mean |seasonal|
  s_third  — mean |forecast|

Outputs:
  data/sarsim0-tsf.arrow
  figures/dataset_tsf/samples.png
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
OUTPUT_PATH = FINETUNING_ROOT / "data" / "sarsim0-tsf.arrow"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "datasets" / "dataset_tsf"

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 192   # 64 trend + 64 seasonal + 64 forecast
SCALE_EPS         = 1e-6

_CTX_COL      = "steelblue"
_TREND_COL    = "seagreen"
_SEASONAL_COL = "mediumpurple"
_FORECAST_COL = "darkorange"


def load_source(path: Path) -> list[dict]:
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()
    rows = []
    for item_id, mode, ctx, seasonal, trend, forecast in zip(
        table.column("item_id").to_pylist(),
        table.column("mode").to_pylist(),
        table.column("context").to_pylist(),
        table.column("seasonal").to_pylist(),
        table.column("trend").to_pylist(),
        table.column("forecast").to_pylist(),
    ):
        if mode != "reasoning":
            continue
        rows.append({
            "item_id":  item_id,
            "context":  np.array(ctx,      dtype=np.float32),
            "trend":    np.array(trend,    dtype=np.float32),   # (64,)
            "seasonal": np.array(seasonal, dtype=np.float32),   # (64,)
            "forecast": np.array(forecast, dtype=np.float32),   # (64,)
        })
    return rows


def build_dataset(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        s_ctx      = float(np.mean(np.abs(r["context"])))
        s_trend    = float(np.mean(np.abs(r["trend"])))
        s_seasonal = float(np.mean(np.abs(r["seasonal"])))
        s_forecast = float(np.mean(np.abs(r["forecast"])))

        trend_norm    = r["trend"]    * (s_ctx / max(s_trend,    SCALE_EPS))
        seasonal_norm = r["seasonal"] * (s_ctx / max(s_seasonal, SCALE_EPS))
        forecast_norm = r["forecast"] * (s_ctx / max(s_forecast, SCALE_EPS))

        out.append({
            "item_id":  r["item_id"],
            "mode":     "TSF",
            "context":  r["context"],
            # TSF order: [trend | seasonal | forecast], all at context-scale magnitude
            "target":   np.concatenate([trend_norm, seasonal_norm, forecast_norm]),   # (192,)
            "s_first":  s_trend,
            "s_second": s_seasonal,
            "s_third":  s_forecast,
        })
    return out


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
            "s_first":  [d["s_first"]  for d in dataset],
            "s_second": [d["s_second"] for d in dataset],
            "s_third":  [d["s_third"]  for d in dataset],
        }, schema=schema))
        writer.close()
    print(f"  Saved: {path}  ({len(dataset):,} rows)")


def plot_samples(dataset: list[dict], path: Path, n: int = 5) -> None:
    rng    = np.random.default_rng(0)
    picks  = sorted(rng.choice(len(dataset), size=min(n, len(dataset)), replace=False))
    ctx_x  = np.arange(CONTEXT_LENGTH)
    trd_x  = np.arange(CONTEXT_LENGTH,       CONTEXT_LENGTH + 64)
    seas_x = np.arange(CONTEXT_LENGTH + 64,  CONTEXT_LENGTH + 128)
    fct_x  = np.arange(CONTEXT_LENGTH + 128, CONTEXT_LENGTH + 192)

    fig, axes_raw = plt.subplots(n, 1, figsize=(16, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        "TSF Dataset  ·  Sample Plots  (targets rescaled to context-scale magnitude)\n"
        "[TSF-prefix  |  context(512)  →  trend_norm(64) | seasonal_norm(64) | forecast_norm(64)]",
        fontsize=12, fontweight="bold",
    )
    for idx, pick in enumerate(picks):
        ax   = axes[idx]
        d    = dataset[pick]
        trd  = d["target"][:64]
        seas = d["target"][64:128]
        fct  = d["target"][128:]
        ax.fill_between(ctx_x, d["context"], alpha=0.08, color=_CTX_COL)
        ax.plot(ctx_x,  d["context"], color=_CTX_COL,      lw=0.7, label="Input context")
        ax.plot(trd_x,  trd,          color=_TREND_COL,    lw=1.6, label="Trend (64)")
        ax.plot(seas_x, seas,         color=_SEASONAL_COL, lw=1.6, label="Seasonal (64)")
        ax.plot(fct_x,  fct,          color=_FORECAST_COL, lw=1.6, label="Forecast (64)")
        ax.axvline(CONTEXT_LENGTH - 0.5,       color="black", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(CONTEXT_LENGTH + 64  - 0.5, color="grey",  lw=0.6, ls=":",  alpha=0.5,
                   label="T|S split")
        ax.axvline(CONTEXT_LENGTH + 128 - 0.5, color="grey",  lw=0.6, ls=":",  alpha=0.5,
                   label="S|F split")
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
    print("Building SarSim0 TSF Dataset  (trend | seasonal | forecast, 192 steps)")
    print(f"  Source : {SOURCE_PATH}")
    print(f"  Output : {OUTPUT_PATH}")
    print("=" * 60)

    print("\nLoading source (reasoning rows only)...")
    rows = load_source(SOURCE_PATH)
    print(f"  Loaded {len(rows):,} rows")

    print("\nBuilding TSF dataset...")
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
