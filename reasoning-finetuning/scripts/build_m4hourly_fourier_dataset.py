#!/usr/bin/env python3
"""
Build m4hourly-fourier.arrow from m4hourly-tsf.arrow.

Uses all 414 rows — Fourier computation depends only on the context column,
which is already present in the TSF dataset.

Mode tag: "FOURIER"

For each row:
  1. De-mean the 96-step context.
  2. Compute rfft; zero out DC (index 0).
  3. Identify top-3 frequency indices by magnitude.
  4. For each frequency k, reconstruct the sinusoid at the FORWARD-LOOKING
     horizon window (time steps 96 … 143):
       A  = 2 * |X[k]| / N
       φ  = angle(X[k])
       f  = k / N   (cycles per sample)
       fourier_i[j] = A · cos(2π · f · (N + j) + φ),  j = 0 … 47

  5. Per-component scale normalisation (same as ST/TSF datasets):
       s_ctx        = mean|context|
       fourier_norm = fourier * (s_ctx / max(mean|fourier|, ε))

Output target: [fourier_1_norm(48) | fourier_2_norm(48) | fourier_3_norm(48)]
               = 144 steps total, ordered highest→lowest dominant frequency.

Stored scale factors (original, pre-normalisation):
  s_first  — mean|fourier_1|
  s_second — mean|fourier_2|
  s_third  — mean|fourier_3|

Outputs:
  data/m4hourly-fourier.arrow
  figures/datasets/dataset_m4hourly_fourier/samples.png   (5 sample plots)
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

SOURCE_PATH = FINETUNING_ROOT / "data" / "m4hourly-tsf.arrow"
OUTPUT_PATH = FINETUNING_ROOT / "data" / "m4hourly-fourier.arrow"
FIGURES_DIR = FINETUNING_ROOT / "figures" / "datasets" / "dataset_m4hourly_fourier"

CONTEXT_LEN = 96
HORIZON     = 48
N_FOURIER   = 3
SCALE_EPS   = 1e-6

_CTX_COL  = "steelblue"
_F1_COL   = "crimson"
_F2_COL   = "darkorange"
_F3_COL   = "seagreen"
_F_COLS   = [_F1_COL, _F2_COL, _F3_COL]
_F_LABELS = ["Fourier-1 (dominant)", "Fourier-2", "Fourier-3"]


# ── Fourier helpers ───────────────────────────────────────────────────────────

def top3_fourier_forward(context: np.ndarray, horizon: int = HORIZON) -> list[np.ndarray]:
    """
    Return the top-3 Fourier component sinusoids projected into the forward
    horizon window (time steps N … N+horizon-1), ordered by descending magnitude.

    Args:
        context : (CONTEXT_LEN,) float32 observed signal.
        horizon : number of forward steps to project.

    Returns:
        List of 3 float32 arrays each of shape (horizon,).
    """
    n = len(context)
    x = context.astype(np.float64) - context.mean()

    X    = np.fft.rfft(x)
    mags = np.abs(X).copy()
    mags[0] = 0.0                               # exclude DC
    top3 = np.argsort(mags)[::-1][:N_FOURIER]  # indices, descending magnitude

    t = np.arange(n, n + horizon, dtype=np.float64)   # forward time indices

    components = []
    for k in top3:
        A   = 2.0 * np.abs(X[k]) / n
        phi = np.angle(X[k])
        f   = k / n                             # cycles per sample
        comp = A * np.cos(2.0 * np.pi * f * t + phi)
        components.append(comp.astype(np.float32))

    return components


# ── load ──────────────────────────────────────────────────────────────────────

def load_source(path: Path) -> list[dict]:
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()
    rows = []
    for item_id, context in zip(
        table.column("item_id").to_pylist(),
        table.column("context").to_pylist(),
    ):
        rows.append({
            "item_id": item_id,
            "context": np.array(context, dtype=np.float32),
        })
    return rows


# ── build ─────────────────────────────────────────────────────────────────────

def build_dataset(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        ctx   = r["context"]
        comps = top3_fourier_forward(ctx)              # 3 × (48,)

        s_ctx = max(float(np.mean(np.abs(ctx))), SCALE_EPS)

        norms  = []
        scales = []
        for c in comps:
            s_c = max(float(np.mean(np.abs(c))), SCALE_EPS)
            norms.append(c * (s_ctx / s_c))
            scales.append(s_c)

        out.append({
            "item_id":  r["item_id"],
            "mode":     "FOURIER",
            "context":  ctx,
            "target":   np.concatenate(norms),   # (144,)
            "s_first":  scales[0],
            "s_second": scales[1],
            "s_third":  scales[2],
        })
    return out


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
            "s_first":  [d["s_first"]  for d in dataset],
            "s_second": [d["s_second"] for d in dataset],
            "s_third":  [d["s_third"]  for d in dataset],
        }, schema=schema))
        writer.close()
    print(f"  Saved: {path}  ({len(dataset):,} rows)")


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_samples(dataset: list[dict], path: Path, n: int = 5) -> None:
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(dataset), size=min(n, len(dataset)), replace=False))

    ctx_x = np.arange(CONTEXT_LEN)
    f_x   = [np.arange(CONTEXT_LEN + i * HORIZON, CONTEXT_LEN + (i + 1) * HORIZON)
             for i in range(N_FOURIER)]

    fig, axes_raw = plt.subplots(n, 1, figsize=(14, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        "M4 Hourly Fourier Dataset  ·  Sample Plots  (targets rescaled to context-scale magnitude)\n"
        "[FOURIER-prefix  |  context(96)  →  fourier_1(48) | fourier_2(48) | fourier_3(48)]",
        fontsize=11, fontweight="bold",
    )

    for idx, pick in enumerate(picks):
        ax = axes[idx]
        d  = dataset[pick]
        segs = [d["target"][i * HORIZON:(i + 1) * HORIZON] for i in range(N_FOURIER)]

        ax.fill_between(ctx_x, d["context"], alpha=0.08, color=_CTX_COL)
        ax.plot(ctx_x, d["context"], color=_CTX_COL, lw=0.7, label="Input context")

        for i, (seg, fx, col, lbl) in enumerate(zip(segs, f_x, _F_COLS, _F_LABELS)):
            ax.plot(fx, seg, color=col, lw=1.6, label=lbl)
            if i < N_FOURIER - 1:
                ax.axvline(CONTEXT_LEN + (i + 1) * HORIZON - 0.5,
                           color="grey", lw=0.6, ls=":", alpha=0.5)

        ax.axvline(CONTEXT_LEN - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
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


# ── verify ────────────────────────────────────────────────────────────────────

def verify(dataset: list[dict]) -> None:
    expected = N_FOURIER * HORIZON
    errors = [d["item_id"] for d in dataset if len(d["target"]) != expected]
    if errors:
        print(f"  ERRORS: {len(errors)} rows with wrong target length (expected {expected})")
    else:
        print(f"  All {len(dataset):,} rows: target_len={expected} ✓")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Building M4 Hourly Fourier Dataset")
    print(f"  context={CONTEXT_LEN}  horizon={HORIZON}  components={N_FOURIER}")
    print(f"  Source : {SOURCE_PATH}")
    print(f"  Output : {OUTPUT_PATH}")
    print("=" * 60)

    print("\nLoading source...")
    rows = load_source(SOURCE_PATH)
    print(f"  Loaded {len(rows):,} rows")

    print("\nComputing Fourier components and building dataset...")
    dataset = build_dataset(rows)

    print("\nVerifying...")
    verify(dataset)

    print("\nSaving Arrow file...")
    save_arrow(dataset, OUTPUT_PATH)

    print("\nGenerating sample plots...")
    plot_samples(dataset, FIGURES_DIR / "samples.png")

    print(f"\nDone.  Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
