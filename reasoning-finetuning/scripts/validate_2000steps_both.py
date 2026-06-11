#!/usr/bin/env python3
"""
validate_2000steps_both.py — Validate checkpoints from the 2000-step Fix-C run.

Checkpoints:
  - checkpoints/run-2000steps-1000.pt   (2k-ckpt-1k)
  - checkpoints/run-2000steps-2000.pt   (2k-ckpt-2k)

Protocol (identical to prior Step A runs):
  - 2000 held-out series (seed=42, 20% split of sarsim0-10k.arrow)
  - 20 samples, median point forecast
  - MSE, MAE, MASE (period=24)
  - Per-series improvement rate on MAE vs zero-shot

Preprocessing uses Fix 1a (min_period=6) and Fix 1b (resid_sig>0.4 → zeros),
matching what these checkpoints were trained with.

Baseline (no-prefix) is computed once and shared across both checkpoints.

Extra stats per checkpoint:
  - MSE distribution: mean, p95, max
  - Fix 1b subset improvement: fraction of resid_sig>0.4 series that beat zero-shot
"""

import sys
from pathlib import Path

REPO_ROOT       = Path(__file__).parent.parent.parent
FINETUNING_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(FINETUNING_ROOT))

import numpy as np
import pyarrow.ipc as ipc
import torch

from chronos import ChronosPipeline
from prefix_generator import PrefixGenerator
from prefix_injection import inject_prefix, remove_prefix_hooks
from scripts.step1_stl_decompose import detect_periods, stl_decompose


# ── Config ─────────────────────────────────────────────────────────────────────

ARROW_PATH   = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
CKPT_1K      = FINETUNING_ROOT / "checkpoints" / "run-2000steps-1000.pt"
CKPT_2K      = FINETUNING_ROOT / "checkpoints" / "run-2000steps-2000.pt"

TRAIN_SEED   = 42
NUM_SAMPLES  = 20
PRED_LEN     = 64
MASE_PERIOD  = 24
BATCH_SIZE   = 16
D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
RANK         = 64

# Fix 1a/1b constants (must match training)
MIN_PERIOD   = 6
RS_THRESHOLD = 0.4

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prior validated results (fixed reference columns)
REF = {
    "Zero-shot": {"mse": 3.3456,  "mae": 0.6714, "mase": 0.6545, "improv": None},
    "Step-500":  {"mse": 18.6146, "mae": 0.8182, "mase": 0.7037, "improv": 40.0},
    "Step-1000": {"mse": 12.8805, "mae": 0.7768, "mase": 0.6856, "improv": 45.6},
}


# ── Dataset split ──────────────────────────────────────────────────────────────

print("Loading sarsim0-10k.arrow …")
with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_total = len(table)
rng     = np.random.default_rng(TRAIN_SEED)
indices = rng.permutation(n_total)
val_idx = indices[int(n_total * 0.8):]
n_val   = len(val_idx)
print(f"  {n_total} rows  →  {n_val} validation series")


# ── Load Chronos ───────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
print("Loading Chronos T5-Small …")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=str(DEVICE),
    dtype=torch.float32,
)
chronos_model = pipeline.model
for p in chronos_model.parameters():
    p.requires_grad_(False)
chronos_model.eval()
print(f"  {sum(p.numel() for p in chronos_model.parameters()):,} params frozen\n")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _mse(p, t): return float(np.mean((p - t) ** 2))
def _mae(p, t): return float(np.mean(np.abs(p - t)))

def _naive_mae(ctx, period=MASE_PERIOD):
    if len(ctx) <= period:
        return float("nan")
    return float(np.mean(np.abs(ctx[period:] - ctx[:-period])))

def nanmean(lst):
    a = np.array([x for x in lst if not np.isnan(x)])
    return float(a.mean()) if len(a) else float("nan")

def dpct(base, new):
    if base == 0 or np.isnan(base) or np.isnan(new):
        return float("nan")
    return (new - base) / base * 100.0


# ── Precompute decompositions for all val series ───────────────────────────────

print("=" * 70)
print("Precomputing STL decompositions (Fix 1a: min_period=6, Fix 1b: rs>0.4→0) …")

all_ctx      = []   # np.ndarray per series
all_ftr      = []
all_trend    = []   # after Fix 1b zeroing
all_seasonal = []
all_residual = []
all_naive_mae = []
all_resid_sig = []  # raw resid_sig before zeroing (for Fix 1b subset analysis)
n_filtered   = 0

for idx in val_idx:
    ctx = np.array(table.column("context")[int(idx)].as_py(), dtype=np.float32)
    ftr = np.array(table.column("forecast")[int(idx)].as_py(), dtype=np.float32)
    pds = detect_periods(ctx, min_period=MIN_PERIOD)
    tr, se, re = stl_decompose(ctx, pds)
    rs  = float(np.var(re) / (np.var(ctx) + 1e-9))

    all_resid_sig.append(rs)
    if rs > RS_THRESHOLD:          # Fix 1b
        tr = np.zeros_like(ctx, dtype=np.float32)
        se = np.zeros_like(ctx, dtype=np.float32)
        re = np.zeros_like(ctx, dtype=np.float32)
        n_filtered += 1

    all_ctx.append(ctx)
    all_ftr.append(ftr)
    all_trend.append(tr.astype(np.float32))
    all_seasonal.append(se.astype(np.float32))
    all_residual.append(re.astype(np.float32))
    all_naive_mae.append(_naive_mae(ctx))

all_resid_sig = np.array(all_resid_sig)
fix1b_mask    = all_resid_sig > RS_THRESHOLD     # bool array, before zeroing
print(f"  Decompositions ready.  Fix 1b triggered: {n_filtered}/{n_val} "
      f"({n_filtered/n_val*100:.1f}%)\n")

n_batches = (n_val + BATCH_SIZE - 1) // BATCH_SIZE


# ── Baseline (no-prefix) pass — computed once ─────────────────────────────────

print("=" * 70)
print(f"BASELINE PASS (no prefix)  —  {n_val} series, {NUM_SAMPLES} samples")
print("=" * 70)

mse_base = np.zeros(n_val, dtype=np.float64)
mae_base = np.zeros(n_val, dtype=np.float64)

for b in range(n_batches):
    b0   = b * BATCH_SIZE
    idxs = range(b0, min(b0 + BATCH_SIZE, n_val))
    bs   = len(list(idxs))

    ctx_t = torch.tensor(np.stack([all_ctx[i] for i in idxs]), dtype=torch.float32)
    ftr_a = np.stack([all_ftr[i] for i in idxs])

    with torch.no_grad():
        samp = pipeline.predict(
            ctx_t, prediction_length=PRED_LEN,
            num_samples=NUM_SAMPLES, limit_prediction_length=False,
        )
        med = samp.median(dim=1).values.numpy()

    for k, i in enumerate(idxs):
        mse_base[i] = _mse(med[k], ftr_a[k])
        mae_base[i] = _mae(med[k], ftr_a[k])

    if (b + 1) % 20 == 0 or (b + 1) == n_batches:
        done = min((b + 1) * BATCH_SIZE, n_val)
        print(f"  [{done:>4}/{n_val}]  running MAE={np.mean(mae_base[:done]):.4f}")

mase_base = np.array([
    mae_base[i] / all_naive_mae[i] if all_naive_mae[i] > 0 else float("nan")
    for i in range(n_val)
])
print(f"\n  Baseline MSE={nanmean(mse_base):.4f}  "
      f"MAE={nanmean(mae_base):.4f}  MASE={nanmean(mase_base):.4f}\n")


# ── Prefix validation function ─────────────────────────────────────────────────

def validate_checkpoint(ckpt_path: Path, label: str):
    """Load checkpoint, run prefix inference, return per-series metric arrays."""

    print("=" * 70)
    print(f"PREFIX PASS — {label}  ({ckpt_path.name})")
    print("=" * 70)

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    pg = PrefixGenerator(
        d_model=D_MODEL, num_layers=NUM_LAYERS,
        prefix_len_per_component=PREFIX_LEN, rank=RANK,
    ).to(DEVICE)
    pg.load_state_dict(ckpt["pg_state"])
    pg.eval()
    print(f"  Loaded step={ckpt['step']}  "
          f"params={sum(p.numel() for p in pg.parameters()):,}\n")

    mse_pfx = np.zeros(n_val, dtype=np.float64)
    mae_pfx = np.zeros(n_val, dtype=np.float64)

    for b in range(n_batches):
        b0   = b * BATCH_SIZE
        idxs = range(b0, min(b0 + BATCH_SIZE, n_val))
        bs   = len(list(idxs))

        ctx_t = torch.tensor(np.stack([all_ctx[i]      for i in idxs]), dtype=torch.float32)
        ftr_a = np.stack(            [all_ftr[i]       for i in idxs])
        tr_t  = torch.tensor(np.stack([all_trend[i]    for i in idxs]), dtype=torch.float32).to(DEVICE)
        se_t  = torch.tensor(np.stack([all_seasonal[i] for i in idxs]), dtype=torch.float32).to(DEVICE)
        re_t  = torch.tensor(np.stack([all_residual[i] for i in idxs]), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            try:
                inject_prefix(chronos_model, pg, trend=tr_t, seasonal=se_t, residual=re_t)
                samp = pipeline.predict(
                    ctx_t, prediction_length=PRED_LEN,
                    num_samples=NUM_SAMPLES, limit_prediction_length=False,
                )
                med = samp.median(dim=1).values.numpy()
            finally:
                remove_prefix_hooks(chronos_model)

        for k, i in enumerate(idxs):
            mse_pfx[i] = _mse(med[k], ftr_a[k])
            mae_pfx[i] = _mae(med[k], ftr_a[k])

        if (b + 1) % 20 == 0 or (b + 1) == n_batches:
            done = min((b + 1) * BATCH_SIZE, n_val)
            print(f"  [{done:>4}/{n_val}]  running MAE  "
                  f"prefix={np.mean(mae_pfx[:done]):.4f}  "
                  f"base={np.mean(mae_base[:done]):.4f}")

    del pg

    mase_pfx = np.array([
        mae_pfx[i] / all_naive_mae[i] if all_naive_mae[i] > 0 else float("nan")
        for i in range(n_val)
    ])

    improved_mae = int(np.sum(mae_pfx < mae_base))

    return {
        "label":       label,
        "ckpt":        ckpt_path.name,
        "step":        int(ckpt["step"]),
        "mse_arr":     mse_pfx,
        "mae_arr":     mae_pfx,
        "mase_arr":    mase_pfx,
        "agg_mse":     nanmean(mse_pfx),
        "agg_mae":     nanmean(mae_pfx),
        "agg_mase":    nanmean(mase_pfx),
        "improv_mae":  improved_mae / n_val * 100,
    }


# ── Run both checkpoints ───────────────────────────────────────────────────────

res_1k = validate_checkpoint(CKPT_1K, "2k-ckpt-1k")
res_2k = validate_checkpoint(CKPT_2K, "2k-ckpt-2k")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*90}")
print("COMPARISON TABLE")
print(f"  Val series: {n_val}  |  num_samples: {NUM_SAMPLES}  |  MASE period: {MASE_PERIOD}")
print(f"  Preprocessing: min_period=6 (Fix 1a), resid_sig>0.4→zeros (Fix 1b)")
print(f"{'='*90}")

hdr = f"{'Metric':<8}  {'Zero-shot':>10}  {'Step-500':>10}  {'Step-1000':>10}  {'2k-ckpt-1k':>10}  {'2k-ckpt-2k':>10}"
sep = f"{'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}"
print(f"\n{hdr}")
print(sep)
print(f"{'MSE':<8}  {REF['Zero-shot']['mse']:>10.4f}  {REF['Step-500']['mse']:>10.4f}  "
      f"{REF['Step-1000']['mse']:>10.4f}  {res_1k['agg_mse']:>10.4f}  {res_2k['agg_mse']:>10.4f}")
print(f"{'MAE':<8}  {REF['Zero-shot']['mae']:>10.4f}  {REF['Step-500']['mae']:>10.4f}  "
      f"{REF['Step-1000']['mae']:>10.4f}  {res_1k['agg_mae']:>10.4f}  {res_2k['agg_mae']:>10.4f}")
print(f"{'MASE':<8}  {REF['Zero-shot']['mase']:>10.4f}  {REF['Step-500']['mase']:>10.4f}  "
      f"{REF['Step-1000']['mase']:>10.4f}  {res_1k['agg_mase']:>10.4f}  {res_2k['agg_mase']:>10.4f}")

def improv_str(v):
    return f"{v:>9.1f}%" if v is not None else f"{'—':>10}"

print(f"{'Improv%':<8}  {improv_str(REF['Zero-shot']['improv'])}  "
      f"{improv_str(REF['Step-500']['improv'])}  "
      f"{improv_str(REF['Step-1000']['improv'])}  "
      f"{res_1k['improv_mae']:>9.1f}%  {res_2k['improv_mae']:>9.1f}%")


# ── MSE distribution ──────────────────────────────────────────────────────────

print(f"\n{'─'*90}")
print("PER-SERIES MSE DISTRIBUTION")
print(f"\n  {'Stat':<12}  {'2k-ckpt-1k':>12}  {'2k-ckpt-2k':>12}  {'Zero-shot ref':>14}")
print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*14}")

for stat, fn in [
    ("mean",  lambda a: np.mean(a)),
    ("p95",   lambda a: np.percentile(a, 95)),
    ("max",   lambda a: np.max(a)),
]:
    v1  = fn(res_1k["mse_arr"])
    v2  = fn(res_2k["mse_arr"])
    vb  = fn(mse_base)
    print(f"  {stat:<12}  {v1:>12.4f}  {v2:>12.4f}  {vb:>14.4f}")


# ── Fix 1b subset effectiveness ───────────────────────────────────────────────

print(f"\n{'─'*90}")
print("FIX 1B SUBSET EFFECTIVENESS  (series with resid_sig > 0.4)")

n_hino = int(np.sum(fix1b_mask))
print(f"\n  High-noise series in val set: {n_hino}/{n_val} ({n_hino/n_val*100:.1f}%)")

if n_hino > 0:
    for res in [res_1k, res_2k]:
        mae_pfx_hi  = res["mae_arr"][fix1b_mask]
        mae_base_hi = mae_base[fix1b_mask]
        n_improved  = int(np.sum(mae_pfx_hi < mae_base_hi))
        pct         = n_improved / n_hino * 100
        agg_pfx_hi  = float(np.mean(mae_pfx_hi))
        agg_bas_hi  = float(np.mean(mae_base_hi))
        print(f"\n  {res['label']} ({res['ckpt']}):")
        print(f"    MAE on high-noise:  prefix={agg_pfx_hi:.4f}  base={agg_bas_hi:.4f}  "
              f"Δ={dpct(agg_bas_hi, agg_pfx_hi):+.2f}%")
        print(f"    Series improved vs zero-shot: {n_improved}/{n_hino} ({pct:.1f}%)")
else:
    print("  (No series with resid_sig > 0.4 found in validation set.)")


print(f"\n{'='*90}")
print("DONE.")
print(f"{'='*90}")
