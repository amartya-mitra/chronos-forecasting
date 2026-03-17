#!/usr/bin/env python3
"""
Build sarsim0-fast-only.arrow from sarsim0-10k.arrow.

Every row receives the "fast" mode tag.  The input context and target
forecast are taken directly from the source — nothing is modified.

This dataset trains the model to associate the fast-mode prefix token
(4096) with its default forecasting behaviour.

Outputs:
  data/sarsim0-fast-only.arrow          (10,000 rows, all fast-mode)
  figures/dataset_fast_only/fast_only_samples.png   (5 sample plots)
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

# ── paths ──────────────────────────────────────────────────────────────────────
SOURCE_PATH = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
OUTPUT_PATH = FINETUNING_ROOT / "data" / "sarsim0-fast-only.arrow"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "dataset_fast_only"

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 64

_CTX_COL = "steelblue"
_FCT_COL = "darkorange"


# ── load ───────────────────────────────────────────────────────────────────────

def load_source(path: Path) -> list[dict]:
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()
    rows = []
    for item_id, ctx, fct in zip(
        table.column("item_id").to_pylist(),
        table.column("context").to_pylist(),
        table.column("forecast").to_pylist(),
    ):
        rows.append({
            "item_id": item_id,
            "context": np.array(ctx, dtype=np.float32),
            "target":  np.array(fct, dtype=np.float32),
        })
    return rows


# ── build ──────────────────────────────────────────────────────────────────────

def build_dataset(rows: list[dict]) -> list[dict]:
    return [
        {
            "item_id": r["item_id"],
            "mode":    "fast",
            "context": r["context"],
            "target":  r["target"],
        }
        for r in rows
    ]


# ── save ───────────────────────────────────────────────────────────────────────

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
            "mode":    [d["mode"]    for d in dataset],
            "context": [d["context"].tolist() for d in dataset],
            "target":  [d["target"].tolist()  for d in dataset],
        }, schema=schema))
        writer.close()
    print(f"  Saved: {path}  ({len(dataset):,} rows)")


# ── plot ───────────────────────────────────────────────────────────────────────

def plot_samples(dataset: list[dict], path: Path, n: int = 5) -> None:
    rng   = np.random.default_rng(0)
    picks = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    picks.sort()

    ctx_x = np.arange(CONTEXT_LENGTH)
    fct_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        "Fast-Only Dataset  ·  Sample Plots\n"
        "[fast-mode prefix  |  input context  →  target forecast]",
        fontsize=12, fontweight="bold",
    )

    for row_idx, pick in enumerate(picks):
        ax = axes[row_idx]
        d  = dataset[pick]
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


# ── verify ─────────────────────────────────────────────────────────────────────

def verify(dataset: list[dict]) -> None:
    errors = [
        f"  {d['item_id']}: mode={d['mode']!r}, target_len={len(d['target'])}"
        for d in dataset
        if d["mode"] != "fast" or len(d["target"]) != PREDICTION_LENGTH
    ]
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(e)
    else:
        print(f"  All {len(dataset):,} rows: mode=fast, target_len={PREDICTION_LENGTH} ✓")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Building SarSim0 Fast-Only Dataset")
    print(f"  Source : {SOURCE_PATH}")
    print(f"  Output : {OUTPUT_PATH}")
    print(f"  Mode   : fast (all rows)")
    print("=" * 60)

    print("\nLoading source dataset...")
    rows = load_source(SOURCE_PATH)
    print(f"  Loaded {len(rows):,} rows")

    print("\nBuilding fast-only dataset...")
    dataset = build_dataset(rows)

    print("\nVerifying...")
    verify(dataset)

    print("\nSaving Arrow file...")
    save_arrow(dataset, OUTPUT_PATH)

    print("\nGenerating plots...")
    plot_samples(dataset, FIGURES_DIR / "fast_only_samples.png")

    print(f"\nDone.  Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
