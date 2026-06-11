#!/usr/bin/env python3
"""
validate_prefix_500.py — Validation of checkpoints/run-1000steps-500.pt
                          on the held-out 20% split of sarsim0-10k.arrow.

For each validation series:
  - prefix forecast:    inject_prefix active, pipeline.predict(num_samples=20)
  - baseline forecast:  no inject_prefix, pipeline.predict(num_samples=20)

Point forecast = median across 20 samples.

Metrics (vs ground-truth forecast window):
  MSE   — mean squared error
  MAE   — mean absolute error
  MASE  — MAE / naive seasonal MAE  (period=24, naive baseline on context)

Reports:
  - Aggregate comparison table (No-prefix vs With-prefix)
  - Fraction of series improved with prefix
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

ARROW_PATH  = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
CKPT_PATH   = FINETUNING_ROOT / "checkpoints" / "run-1000steps-500.pt"

TRAIN_SEED  = 42
NUM_SAMPLES = 20
PRED_LEN    = 64
MASE_PERIOD = 24
BATCH_SIZE  = 16    # series per inference call

D_MODEL     = 512
NUM_LAYERS  = 6
PREFIX_LEN  = 16
RANK        = 64

# Device selection:
#   pipeline.predict() always receives a CPU context tensor — it handles
#   tokenization (bucketize) on CPU internally, then moves input_ids to the
#   model device.  trend/seasonal/residual are moved to DEVICE for the
#   PrefixGenerator forward pass.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ── Dataset split ──────────────────────────────────────────────────────────────

print("Loading sarsim0-10k.arrow …")
with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_total  = len(table)
rng      = np.random.default_rng(TRAIN_SEED)
indices  = rng.permutation(n_total)
val_idx  = indices[int(n_total * 0.8):]   # 20% held-out
print(f"  {n_total} rows  →  {len(val_idx)} validation")


# ── Load Chronos (CPU) ─────────────────────────────────────────────────────────

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
print(f"  {sum(p.numel() for p in chronos_model.parameters()):,} params frozen")


# ── Load PrefixGenerator from checkpoint ──────────────────────────────────────

print(f"Loading checkpoint: {CKPT_PATH} …")
ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)
pg = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(DEVICE)
pg.load_state_dict(ckpt["pg_state"])
pg.eval()
print(f"  Loaded step {ckpt['step']} weights")
print(f"  PrefixGenerator params: {sum(p.numel() for p in pg.parameters()):,}\n")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))

def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))

def naive_seasonal_mae(context: np.ndarray, period: int = MASE_PERIOD) -> float:
    """MAE of the seasonal naive forecast on the context window (in-sample)."""
    if len(context) <= period:
        return float("nan")
    return float(np.mean(np.abs(context[period:] - context[:-period])))


# ── Validation loop ────────────────────────────────────────────────────────────

print("=" * 60)
print(f"VALIDATION  ({len(val_idx)} series, num_samples={NUM_SAMPLES})")
print("=" * 60)

mse_prefix,  mse_base  = [], []
mae_prefix,  mae_base  = [], []
mase_prefix, mase_base = [], []
improved_mse = 0
improved_mae = 0

n_batches = (len(val_idx) + BATCH_SIZE - 1) // BATCH_SIZE

for b_idx in range(n_batches):
    batch_indices = val_idx[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]

    contexts, futures, trends, seasonals, residuals = [], [], [], [], []
    naive_maes = []

    for idx in batch_indices:
        ctx = np.array(table.column("context")[int(idx)].as_py(), dtype=np.float32)
        ftr = np.array(table.column("forecast")[int(idx)].as_py(), dtype=np.float32)
        periods    = detect_periods(ctx)
        tr, se, re = stl_decompose(ctx, periods)
        contexts.append(ctx)
        futures.append(ftr)
        trends.append(tr.astype(np.float32))
        seasonals.append(se.astype(np.float32))
        residuals.append(re.astype(np.float32))
        naive_maes.append(naive_seasonal_mae(ctx))

    ctx_tensor  = torch.tensor(np.stack(contexts),  dtype=torch.float32)
    ftr_arr     = np.stack(futures)
    trend_t     = torch.tensor(np.stack(trends),    dtype=torch.float32).to(DEVICE)
    seasonal_t  = torch.tensor(np.stack(seasonals), dtype=torch.float32).to(DEVICE)
    residual_t  = torch.tensor(np.stack(residuals), dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        # ── Prefix forecast ────────────────────────────────────────────────
        try:
            inject_prefix(
                chronos_model, pg,
                trend=trend_t, seasonal=seasonal_t, residual=residual_t,
            )
            pfx_samples = pipeline.predict(
                ctx_tensor,
                prediction_length=PRED_LEN,
                num_samples=NUM_SAMPLES,
                limit_prediction_length=False,
            )  # (batch, num_samples, pred_len)
            pfx_median = pfx_samples.median(dim=1).values.numpy()  # (batch, pred_len)
        finally:
            remove_prefix_hooks(chronos_model)

        # ── Baseline forecast (no prefix) ──────────────────────────────────
        base_samples = pipeline.predict(
            ctx_tensor,
            prediction_length=PRED_LEN,
            num_samples=NUM_SAMPLES,
            limit_prediction_length=False,
        )
        base_median = base_samples.median(dim=1).values.numpy()

    # ── Per-series metrics ─────────────────────────────────────────────────
    for i in range(len(batch_indices)):
        gt     = ftr_arr[i]
        p_fcst = pfx_median[i]
        b_fcst = base_median[i]
        n_mae  = naive_maes[i]

        m_p = mse(p_fcst, gt);  m_b = mse(b_fcst, gt)
        a_p = mae(p_fcst, gt);  a_b = mae(b_fcst, gt)
        s_p = a_p / n_mae if n_mae > 0 else float("nan")
        s_b = a_b / n_mae if n_mae > 0 else float("nan")

        mse_prefix.append(m_p);  mse_base.append(m_b)
        mae_prefix.append(a_p);  mae_base.append(a_b)
        mase_prefix.append(s_p); mase_base.append(s_b)

        if m_p < m_b: improved_mse += 1
        if a_p < a_b: improved_mae += 1

    if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
        done = min((b_idx + 1) * BATCH_SIZE, len(val_idx))
        print(f"  [{done:>4}/{len(val_idx)}]  running MAE — "
              f"prefix={np.mean(mae_prefix):.4f}  "
              f"base={np.mean(mae_base):.4f}")


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def nanmean(lst):
    arr = np.array([x for x in lst if not np.isnan(x)])
    return float(arr.mean()) if len(arr) > 0 else float("nan")

agg_mse_b  = nanmean(mse_base);   agg_mse_p  = nanmean(mse_prefix)
agg_mae_b  = nanmean(mae_base);   agg_mae_p  = nanmean(mae_prefix)
agg_mase_b = nanmean(mase_base);  agg_mase_p = nanmean(mase_prefix)

def delta_pct(base, new):
    if base == 0 or np.isnan(base) or np.isnan(new):
        return float("nan")
    return (new - base) / base * 100.0

d_mse  = delta_pct(agg_mse_b,  agg_mse_p)
d_mae  = delta_pct(agg_mae_b,  agg_mae_p)
d_mase = delta_pct(agg_mase_b, agg_mase_p)

n_val = len(val_idx)
frac_mse = improved_mse / n_val * 100
frac_mae = improved_mae / n_val * 100

# ── Report ─────────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("VALIDATION RESULTS")
print(f"  Checkpoint: {CKPT_PATH.name}  (step {ckpt['step']})")
print(f"  Val series: {n_val}  |  num_samples: {NUM_SAMPLES}  |  MASE period: {MASE_PERIOD}")
print(f"{'='*65}")

print(f"\n{'Metric':<8} {'No-prefix (zero-shot)':>22}  {'With prefix':>12}  {'Δ (%)':>8}")
print(f"{'─'*8} {'─'*22}  {'─'*12}  {'─'*8}")
print(f"{'MSE':<8} {agg_mse_b:>22.4f}  {agg_mse_p:>12.4f}  {d_mse:>+8.2f}%")
print(f"{'MAE':<8} {agg_mae_b:>22.4f}  {agg_mae_p:>12.4f}  {d_mae:>+8.2f}%")
print(f"{'MASE':<8} {agg_mase_b:>22.4f}  {agg_mase_p:>12.4f}  {d_mase:>+8.2f}%")

print(f"\nPer-series improvement rate:")
print(f"  MSE improved with prefix: {improved_mse}/{n_val}  ({frac_mse:.1f}%)")
print(f"  MAE improved with prefix: {improved_mae}/{n_val}  ({frac_mae:.1f}%)")
print(f"{'='*65}")
