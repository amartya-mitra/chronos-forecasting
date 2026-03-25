#!/usr/bin/env python3
"""
Build sarsim0-contrastive-v1.arrow: contrastive fine-tuning dataset.

Split (same as trend-test-v2):
  70%  fast-mode  — correct_target = 64-step forecast,  wrong_target = trend
  30%  trend-mode — correct_target = trend,              wrong_target = 64-step forecast

Every row stores BOTH targets so no on-the-fly computation is needed during training.

Schema:
  item_id        : string
  mode           : "fast" | "trend"
  context        : float32[512]
  correct_target : float32[64]   — the target the model should produce for this mode
  wrong_target   : float32[64]   — the target for the opposite mode

Trend extraction: centered moving average (window=52) via scipy.ndimage.uniform_filter1d
with reflect padding — same method as trend-test-v2.

Outputs:
  data/sarsim0-contrastive-v1.arrow
  figures/dataset_contrastive_v1/fast_samples.png    (5 fast-mode rows)
  figures/dataset_contrastive_v1/trend_samples.png   (5 trend-mode rows)
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

# ── paths ──────────────────────────────────────────────────────────────────────
SOURCE_PATH = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
OUTPUT_PATH = FINETUNING_ROOT / "data" / "sarsim0-contrastive-v1.arrow"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "dataset_contrastive_v1"

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 64
TREND_WINDOW      = 52      # MA window — covers longest SarSim0 seasonal period
FAST_FRACTION     = 0.30    # 30% fast / 70% trend
SEED              = 42

_CTX_COL     = "steelblue"
_CORRECT_COL = "darkorange"
_WRONG_COL   = "crimson"
_TREND_COL   = "seagreen"


# ── trend extraction ───────────────────────────────────────────────────────────

def extract_trend(context: np.ndarray, window: int = TREND_WINDOW) -> np.ndarray:
    """Centered MA with reflect padding. Returns float32[PREDICTION_LENGTH]."""
    smoothed = uniform_filter1d(
        context.astype(np.float64), size=window, mode="reflect"
    ).astype(np.float32)
    return smoothed[-PREDICTION_LENGTH:]


# ── data loading ───────────────────────────────────────────────────────────────

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


# ── dataset construction ───────────────────────────────────────────────────────

def build_dataset(rows: list[dict], seed: int = SEED) -> list[dict]:
    rng     = np.random.default_rng(seed)
    n_total = len(rows)
    n_fast  = round(n_total * FAST_FRACTION)

    indices = np.arange(n_total)
    rng.shuffle(indices)
    fast_set = set(indices[:n_fast].tolist())

    dataset = []
    for i, row in enumerate(rows):
        trend = extract_trend(row["context"])   # float32[64]

        if i in fast_set:
            dataset.append({
                "item_id":        row["item_id"],
                "mode":           "fast",
                "context":        row["context"],
                "correct_target": row["forecast"],  # what fast-mode should produce
                "wrong_target":   trend,             # what trend-mode would produce
            })
        else:
            dataset.append({
                "item_id":        row["item_id"],
                "mode":           "trend",
                "context":        row["context"],
                "correct_target": trend,             # what trend-mode should produce
                "wrong_target":   row["forecast"],   # what fast-mode would produce
            })

    return dataset


# ── Arrow save ─────────────────────────────────────────────────────────────────

def save_arrow(dataset: list[dict], path: Path) -> None:
    schema = pa.schema([
        pa.field("item_id",        pa.string()),
        pa.field("mode",           pa.string()),
        pa.field("context",        pa.list_(pa.float32())),
        pa.field("correct_target", pa.list_(pa.float32())),
        pa.field("wrong_target",   pa.list_(pa.float32())),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        writer = ipc.new_file(sink, schema)
        batch  = pa.record_batch({
            "item_id":        [d["item_id"]               for d in dataset],
            "mode":           [d["mode"]                  for d in dataset],
            "context":        [d["context"].tolist()       for d in dataset],
            "correct_target": [d["correct_target"].tolist() for d in dataset],
            "wrong_target":   [d["wrong_target"].tolist()   for d in dataset],
        }, schema=schema)
        writer.write_batch(batch)
        writer.close()
    print(f"  Saved: {path}  ({len(dataset):,} rows)")


# ── verification ───────────────────────────────────────────────────────────────

def verify_dataset(dataset: list[dict]) -> None:
    errors = []
    for d in dataset:
        if d["mode"] not in ("fast", "trend"):
            errors.append(f"  {d['item_id']}: unknown mode '{d['mode']}'")
        for key in ("correct_target", "wrong_target"):
            if len(d[key]) != PREDICTION_LENGTH:
                errors.append(
                    f"  {d['item_id']}: {key} length {len(d[key])} != {PREDICTION_LENGTH}"
                )
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(e)
    else:
        print(f"  All {len(dataset):,} rows pass: valid mode, correct/wrong targets length={PREDICTION_LENGTH}")


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_samples(dataset: list[dict], mode: str, path: Path, n: int = 5) -> None:
    """
    For each sample: context + correct_target (solid) + wrong_target (dashed).
    Lets the user visually verify that correct and wrong targets are distinct.
    """
    rows = [d for d in dataset if d["mode"] == mode]
    rng  = np.random.default_rng(0)
    picks = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    picks.sort()

    ctx_x = np.arange(CONTEXT_LENGTH)
    tgt_x = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)

    correct_col = _CORRECT_COL if mode == "fast" else _TREND_COL
    correct_lbl = "Correct (forecast)" if mode == "fast" else "Correct (trend)"
    wrong_lbl   = "Wrong (trend)"      if mode == "fast" else "Wrong (forecast)"

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"Contrastive Dataset  ·  {mode}-mode samples ({int(FAST_FRACTION*100 if mode=='fast' else (1-FAST_FRACTION)*100)}%)\n"
        f"Solid = correct target   Dashed crimson = wrong target",
        fontsize=11, fontweight="bold",
    )

    for row_idx, pick in enumerate(picks):
        ax = axes[row_idx]
        d  = rows[pick]

        ax.fill_between(ctx_x, d["context"], alpha=0.07, color=_CTX_COL)
        ax.plot(ctx_x, d["context"],        color=_CTX_COL,    lw=0.7, label="Context")
        ax.plot(tgt_x, d["correct_target"], color=correct_col, lw=1.8, label=correct_lbl)
        ax.plot(tgt_x, d["wrong_target"],   color=_WRONG_COL,  lw=1.4, ls="--", label=wrong_lbl)
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


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Building SarSim0 Contrastive-V1 Dataset")
    print(f"  Source  : {SOURCE_PATH}")
    print(f"  Output  : {OUTPUT_PATH}")
    print(f"  Split   : {int(FAST_FRACTION*100)}% fast / {int((1-FAST_FRACTION)*100)}% trend")
    print(f"  MA win  : {TREND_WINDOW}")
    print("=" * 60)

    print("\nLoading source dataset...")
    rows = load_source(SOURCE_PATH)
    print(f"  Loaded {len(rows):,} rows")

    print("\nBuilding contrastive dataset...")
    dataset = build_dataset(rows)
    n_fast  = sum(1 for d in dataset if d["mode"] == "fast")
    n_trend = sum(1 for d in dataset if d["mode"] == "trend")
    print(f"  Fast-mode  : {n_fast:,}  ({n_fast / len(dataset) * 100:.1f}%)")
    print(f"  Trend-mode : {n_trend:,}  ({n_trend / len(dataset) * 100:.1f}%)")

    print("\nVerifying dataset...")
    verify_dataset(dataset)

    print("\nSaving Arrow file...")
    save_arrow(dataset, OUTPUT_PATH)

    print("\nGenerating plots...")
    plot_samples(dataset, "fast",  FIGURES_DIR / "fast_samples.png")
    plot_samples(dataset, "trend", FIGURES_DIR / "trend_samples.png")

    print(f"\nDone.  Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
