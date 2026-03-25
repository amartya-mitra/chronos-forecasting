#!/usr/bin/env python3
"""
Build sarsim0-seasonal-only.arrow from sarsim0-10k.arrow.

Uses only the 6,000 "reasoning" rows (fast rows have empty seasonal).
Every row receives the "seasonal" mode tag.
  context (512 steps) → target = seasonal component (64 future steps)

Outputs:
  data/sarsim0-seasonal-only.arrow
  figures/dataset_seasonal_only/samples.png   (5 sample plots)
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
OUTPUT_PATH = FINETUNING_ROOT / "data" / "sarsim0-seasonal-only.arrow"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "dataset_seasonal_only"

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 64

_CTX_COL      = "steelblue"
_SEASONAL_COL = "mediumpurple"


def load_source(path: Path) -> list[dict]:
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()
    rows = []
    for item_id, mode, ctx, seasonal in zip(
        table.column("item_id").to_pylist(),
        table.column("mode").to_pylist(),
        table.column("context").to_pylist(),
        table.column("seasonal").to_pylist(),
    ):
        if mode != "reasoning":
            continue   # fast rows have empty seasonal
        rows.append({
            "item_id": item_id,
            "context":  np.array(ctx,      dtype=np.float32),
            "target":   np.array(seasonal, dtype=np.float32),
        })
    return rows


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
        writer.write_batch(pa.record_batch({
            "item_id": [d["item_id"] for d in dataset],
            "mode":    ["seasonal"] * len(dataset),
            "context": [d["context"].tolist() for d in dataset],
            "target":  [d["target"].tolist()  for d in dataset],
        }, schema=schema))
        writer.close()
    print(f"  Saved: {path}  ({len(dataset):,} rows)")


def plot_samples(dataset: list[dict], path: Path, n: int = 5) -> None:
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(dataset), size=min(n, len(dataset)), replace=False))
    ctx_x = np.arange(CONTEXT_LENGTH)
    tgt_x = np.arange(CONTEXT_LENGTH, PREDICTION_LENGTH + CONTEXT_LENGTH)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        "Seasonal-Only Dataset  ·  Sample Plots\n"
        "[seasonal-mode prefix  |  input context  →  target seasonal component]",
        fontsize=12, fontweight="bold",
    )
    for idx, pick in enumerate(picks):
        ax = axes[idx]
        d  = dataset[pick]
        ax.fill_between(ctx_x, d["context"], alpha=0.08, color=_CTX_COL)
        ax.plot(ctx_x, d["context"], color=_CTX_COL,      lw=0.8, label="Input context")
        ax.plot(tgt_x, d["target"],  color=_SEASONAL_COL, lw=1.6, label="Target seasonal")
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


def verify(dataset: list[dict]) -> None:
    errors = [
        f"  {d['item_id']}: target_len={len(d['target'])}"
        for d in dataset if len(d["target"]) != PREDICTION_LENGTH
    ]
    if errors:
        for e in errors[:10]: print(e)
    else:
        print(f"  All {len(dataset):,} rows: target_len={PREDICTION_LENGTH} ✓")


def main() -> None:
    print("=" * 60)
    print("Building SarSim0 Seasonal-Only Dataset")
    print(f"  Source : {SOURCE_PATH}")
    print(f"  Output : {OUTPUT_PATH}")
    print("=" * 60)

    print("\nLoading source dataset (reasoning rows only)...")
    rows = load_source(SOURCE_PATH)
    print(f"  Loaded {len(rows):,} reasoning rows")

    print("\nVerifying...")
    verify(rows)

    print("\nSaving Arrow file...")
    save_arrow(rows, OUTPUT_PATH)

    print("\nGenerating plots...")
    plot_samples(rows, FIGURES_DIR / "samples.png")

    print(f"\nDone.  Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
