#!/usr/bin/env python3
"""
eval_cross_dataset.py — Phase 2 Task 1: cross-dataset evaluation.

Tests whether the SarSim0-trained prefix generalises to GiftEval M4 Hourly
without retraining.

Dataset: m4hourly-tsf.arrow
  Schema: item_id, mode, context (96 pts), target (144 pts)
  target layout: target[:48]=trend_fc  target[48:96]=seasonal_fc  target[96:]=raw_future(48pts)
  Ground truth forecast: target[96:] (48 pts)
  Context: context column (96 pts)

Decomposition:
  - Compute STL on-the-fly on the context window (Fix 1a: min_period=6)
  - Fix 1b: resid_sig > 0.4 → zero trend/seasonal/residual
  - Ceiling at inference: training_state fixed at current_step=2000, total_steps=2000
    → ceiling = 0.80 throughout inference

Outputs:
  results/phase2_gifteval_eval.json    — per-series metrics
  results/phase2_gifteval_summary.txt  — comparison table
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT       = Path(__file__).parent.parent
FINETUNING_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(FINETUNING_ROOT))

import numpy as np
import pyarrow.ipc as ipc
import torch

from chronos import ChronosPipeline
from prefix_generator import PrefixGenerator
from prefix_injection import inject_prefix, remove_prefix_hooks
from scripts.step1_stl_decompose import detect_periods, stl_decompose


# ── Args ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="reasoning-finetuning/checkpoints/run-2000steps-2000.pt")
parser.add_argument("--dataset",    default="reasoning-finetuning/data/m4hourly-tsf.arrow")
parser.add_argument("--output",     default="results/phase2_gifteval_eval.json")
args = parser.parse_args()

CKPT_PATH    = Path(args.checkpoint)
ARROW_PATH   = Path(args.dataset)
OUTPUT_PATH  = Path(args.output)
SUMMARY_PATH = OUTPUT_PATH.parent / "phase2_gifteval_summary.txt"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────

NUM_SAMPLES  = 20
PRED_LEN     = 48        # M4 Hourly forecast horizon
MASE_PERIOD  = 24        # daily seasonality in hourly data
BATCH_SIZE   = 16
D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN   # 48
RANK         = 64
MIN_PERIOD   = 6
RS_THRESHOLD = 0.4

# Inference ceiling: fixed at end-of-training value (step=2000/2000 → 0.80)
INFER_CEIL_STEP  = 2000
INFER_CEIL_TOTAL = 2000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reference values from prior SarSim0 reports
REF_SARSIM0 = {
    "zero_shot": {"mse": 3.3456,  "mae": 0.6714, "mase": 0.6545},
    "with_pfx":  {"mse": 3.3283,  "mae": 0.6724, "mase": 0.6511, "improv_pct": 50.4},
}


# ── Load dataset ───────────────────────────────────────────────────────────────

print(f"Loading {ARROW_PATH} …")
with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_series = len(table)
print(f"  {n_series} series")
print(f"  Schema: {table.column_names}")

# Verify expected layout: context=96, target=144 (trend48|seasonal48|raw48)
ctx_len = len(table.column("context")[0].as_py())
tgt_len = len(table.column("target")[0].as_py())
assert tgt_len == ctx_len + PRED_LEN, (
    f"Unexpected target length {tgt_len}; expected {ctx_len + PRED_LEN} "
    f"(ctx_len={ctx_len} + pred_len={PRED_LEN})"
)
print(f"  context={ctx_len} pts, raw_future=target[{ctx_len}:] ({PRED_LEN} pts)\n")


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


# ── Precompute decompositions ──────────────────────────────────────────────────

print("Precomputing STL decompositions (Fix 1a: min_period=6, Fix 1b: rs>0.4→0) …")

all_ctx      = []
all_ftr      = []
all_trend    = []
all_seasonal = []
all_residual = []
all_naive_mae = []
all_resid_sig = []
all_periods   = []
n_filtered    = 0

for i in range(n_series):
    ctx = np.array(table.column("context")[i].as_py(), dtype=np.float32)
    tgt = np.array(table.column("target")[i].as_py(),  dtype=np.float32)
    ftr = tgt[ctx_len:]   # raw forecast ground truth

    pds = detect_periods(ctx, min_period=MIN_PERIOD)
    tr, se, re = stl_decompose(ctx, pds)
    rs  = float(np.var(re) / (np.var(ctx) + 1e-9))

    all_resid_sig.append(rs)
    all_periods.append(pds)

    if rs > RS_THRESHOLD:
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
fix1b_mask    = all_resid_sig > RS_THRESHOLD
print(f"  Decompositions ready.  Fix 1b triggered: {n_filtered}/{n_series} "
      f"({n_filtered/n_series*100:.1f}%)\n")

n_batches = (n_series + BATCH_SIZE - 1) // BATCH_SIZE


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
print(f"  {sum(p.numel() for p in chronos_model.parameters()):,} params frozen")


# ── Load PrefixGenerator ───────────────────────────────────────────────────────

print(f"Loading checkpoint: {CKPT_PATH} …")
ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)
pg = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(DEVICE)
pg.load_state_dict(ckpt["pg_state"])
pg.eval()
print(f"  Loaded step={ckpt['step']}  "
      f"params={sum(p.numel() for p in pg.parameters()):,}\n")

# Inference training_state: fixed ceiling at 0.80 (end-of-training value)
infer_state = {
    'current_step':                   INFER_CEIL_STEP,
    'total_steps':                    INFER_CEIL_TOTAL,
    'ceiling_fired_accumulator':      [],
    'pfx_total_pre_ceil_accumulator': [],
}


# ── Baseline (no-prefix) pass ─────────────────────────────────────────────────

print("=" * 70)
print(f"BASELINE PASS (no prefix)  —  {n_series} series")
print("=" * 70)

mse_base = np.zeros(n_series, dtype=np.float64)
mae_base = np.zeros(n_series, dtype=np.float64)

for b in range(n_batches):
    b0   = b * BATCH_SIZE
    idxs = range(b0, min(b0 + BATCH_SIZE, n_series))

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

    if (b + 1) % 5 == 0 or (b + 1) == n_batches:
        done = min((b + 1) * BATCH_SIZE, n_series)
        print(f"  [{done:>3}/{n_series}]  MAE={np.mean(mae_base[:done]):.4f}")

mase_base = np.array([
    mae_base[i] / all_naive_mae[i] if all_naive_mae[i] > 0 else float("nan")
    for i in range(n_series)
])
print(f"\n  Baseline  MSE={nanmean(mse_base):.4f}  "
      f"MAE={nanmean(mae_base):.4f}  MASE={nanmean(mase_base):.4f}\n")


# ── Prefix pass ───────────────────────────────────────────────────────────────

print("=" * 70)
print(f"PREFIX PASS (with ceiling=0.80)  —  {n_series} series")
print("=" * 70)

mse_pfx = np.zeros(n_series, dtype=np.float64)
mae_pfx = np.zeros(n_series, dtype=np.float64)

for b in range(n_batches):
    b0   = b * BATCH_SIZE
    idxs = range(b0, min(b0 + BATCH_SIZE, n_series))

    ctx_t = torch.tensor(np.stack([all_ctx[i]      for i in idxs]), dtype=torch.float32)
    ftr_a = np.stack(            [all_ftr[i]       for i in idxs])
    tr_t  = torch.tensor(np.stack([all_trend[i]    for i in idxs]), dtype=torch.float32).to(DEVICE)
    se_t  = torch.tensor(np.stack([all_seasonal[i] for i in idxs]), dtype=torch.float32).to(DEVICE)
    re_t  = torch.tensor(np.stack([all_residual[i] for i in idxs]), dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        try:
            inject_prefix(
                chronos_model, pg,
                trend=tr_t, seasonal=se_t, residual=re_t,
                training_state=infer_state,
            )
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

    if (b + 1) % 5 == 0 or (b + 1) == n_batches:
        done = min((b + 1) * BATCH_SIZE, n_series)
        print(f"  [{done:>3}/{n_series}]  prefix={np.mean(mae_pfx[:done]):.4f}  "
              f"base={np.mean(mae_base[:done]):.4f}")

mase_pfx = np.array([
    mae_pfx[i] / all_naive_mae[i] if all_naive_mae[i] > 0 else float("nan")
    for i in range(n_series)
])

# Ceiling stats
ceil_acc  = infer_state['ceiling_fired_accumulator']
pre_acc   = infer_state['pfx_total_pre_ceil_accumulator']
ceil_fire_rate = float(np.mean(ceil_acc)) if ceil_acc else float("nan")
pre_ceil_mean  = float(np.mean(pre_acc))  if pre_acc  else float("nan")
print(f"\n  Ceiling fire rate: {ceil_fire_rate:.4f} ({ceil_fire_rate*100:.1f}% of computations)")
print(f"  Pre-ceiling mean pfx_total: {pre_ceil_mean:.4f}\n")


# ── Aggregate metrics ──────────────────────────────────────────────────────────

agg_mse_b  = nanmean(mse_base);  agg_mse_p  = nanmean(mse_pfx)
agg_mae_b  = nanmean(mae_base);  agg_mae_p  = nanmean(mae_pfx)
agg_mase_b = nanmean(mase_base); agg_mase_p = nanmean(mase_pfx)
improved_mae = int(np.sum(mae_pfx < mae_base))


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

lines = []

def P(s=""):
    lines.append(s)
    print(s)

P("=" * 90)
P("PHASE 2 — CROSS-DATASET EVALUATION  (SarSim0-trained prefix → GiftEval M4 Hourly)")
P(f"  Checkpoint: {CKPT_PATH.name}  (step {ckpt['step']})")
P(f"  Dataset:    {ARROW_PATH.name}  ({n_series} series)")
P(f"  Preprocessing: min_period={MIN_PERIOD} (Fix 1a), resid_sig>{RS_THRESHOLD}→zeros (Fix 1b)")
P(f"  Inference ceiling: {min(0.5 + 0.3*(INFER_CEIL_STEP/INFER_CEIL_TOTAL), 0.8):.2f} (fixed, end-of-training)")
P("=" * 90)

# ── 1. Main comparison table ───────────────────────────────────────────────────
P()
P("1. MAIN COMPARISON TABLE")
P()
hdr = (f"{'Metric':<8}  {'SarSim0 zero-shot':>18}  {'SarSim0 w/prefix':>17}  "
       f"{'GiftEval zero-shot':>19}  {'GiftEval w/prefix':>18}")
sep = f"{'─'*8}  {'─'*18}  {'─'*17}  {'─'*19}  {'─'*18}"
P(hdr)
P(sep)
P(f"{'MSE':<8}  {REF_SARSIM0['zero_shot']['mse']:>18.4f}  "
  f"{REF_SARSIM0['with_pfx']['mse']:>17.4f}  "
  f"{agg_mse_b:>19.4f}  {agg_mse_p:>18.4f}")
P(f"{'MAE':<8}  {REF_SARSIM0['zero_shot']['mae']:>18.4f}  "
  f"{REF_SARSIM0['with_pfx']['mae']:>17.4f}  "
  f"{agg_mae_b:>19.4f}  {agg_mae_p:>18.4f}")
P(f"{'MASE':<8}  {REF_SARSIM0['zero_shot']['mase']:>18.4f}  "
  f"{REF_SARSIM0['with_pfx']['mase']:>17.4f}  "
  f"{agg_mase_b:>19.4f}  {agg_mase_p:>18.4f}")
P(f"{'Improv%':<8}  {'—':>18}  {REF_SARSIM0['with_pfx']['improv_pct']:>16.1f}%  "
  f"{'—':>19}  {improved_mae/n_series*100:>17.1f}%")

# ── 2. Per-series MSE distribution ────────────────────────────────────────────
P()
P("─" * 90)
P("2. PER-SERIES MSE DISTRIBUTION")
P()
P(f"  {'Stat':<10}  {'GiftEval zero-shot':>20}  {'GiftEval w/prefix':>19}")
P(f"  {'─'*10}  {'─'*20}  {'─'*19}")
for stat, fn in [("mean", np.mean), ("p95", lambda a: np.percentile(a, 95)), ("max", np.max)]:
    P(f"  {stat:<10}  {fn(mse_base):>20.4f}  {fn(mse_pfx):>19.4f}")

# ── 3. Fix 1b filter rate ──────────────────────────────────────────────────────
P()
P("─" * 90)
P("3. FIX 1B FILTER RATE ON GIFTEVAL")
n_hino = int(np.sum(fix1b_mask))
P(f"  Series with resid_sig > {RS_THRESHOLD}: {n_hino}/{n_series} ({n_hino/n_series*100:.1f}%)")
P(f"  (SarSim0 reference: 2.9%)")

# ── 4. Ceiling activation rate ────────────────────────────────────────────────
P()
P("─" * 90)
P("4. FIX 2 CEILING ACTIVATION RATE (inference, ceiling=0.80 fixed)")
P(f"  Fired: {ceil_fire_rate*100:.2f}% of attention computations")
P(f"  Pre-ceiling mean pfx_total: {pre_ceil_mean:.4f}")

# ── 5. Per-series improvement rate ────────────────────────────────────────────
P()
P("─" * 90)
P("5. PER-SERIES IMPROVEMENT RATE (MAE, prefix vs zero-shot)")
P(f"  {improved_mae}/{n_series} series improved  ({improved_mae/n_series*100:.1f}%)")

# ── Generalisation verdict ────────────────────────────────────────────────────
P()
P("─" * 90)
P("GENERALISATION VERDICT")
prefix_helps = agg_mase_p < agg_mase_b
prefix_helps_sarsim0 = REF_SARSIM0['with_pfx']['mase'] < REF_SARSIM0['zero_shot']['mase']
delta_mase_gifteval = dpct(agg_mase_b, agg_mase_p)
delta_mase_sarsim0  = dpct(REF_SARSIM0['zero_shot']['mase'], REF_SARSIM0['with_pfx']['mase'])
P(f"  SarSim0:  prefix {'HELPS' if prefix_helps_sarsim0 else 'HURTS'}  "
  f"(ΔMASE={delta_mase_sarsim0:+.2f}%)")
P(f"  GiftEval: prefix {'HELPS' if prefix_helps else 'HURTS'}  "
  f"(ΔMASE={delta_mase_gifteval:+.2f}%)")
if prefix_helps:
    P("  → Cross-domain transfer: POSITIVE")
else:
    P("  → Cross-domain transfer: NEGATIVE (decomposition mismatch)")


# ── 6. Outlier diagnosis (only if prefix hurts on GiftEval) ───────────────────

if not prefix_helps:
    P()
    P("═" * 90)
    P("6. OUTLIER DIAGNOSIS  (prefix hurts overall — diagnosing worst 5%)")
    P("═" * 90)

    mse_pfx_arr  = mse_pfx
    mae_pfx_arr  = mae_pfx
    mae_base_arr = mae_base
    threshold    = np.percentile(mse_pfx_arr, 95)
    outlier_pos  = np.where(mse_pfx_arr > threshold)[0]
    n_out        = len(outlier_pos)
    P(f"\n  95th-pct MSE threshold: {threshold:.4f}")
    P(f"  Outlier count: {n_out}")

    outlier_periods   = [all_periods[i]   for i in outlier_pos]
    outlier_resid_sig = [all_resid_sig[i] for i in outlier_pos]
    outlier_mae_delta = (mae_pfx_arr - mae_base_arr)[outlier_pos]

    # Period distribution
    flat_out = [p for ps in outlier_periods for p in ps]
    flat_all = [p for ps in all_periods     for p in ps]
    cnt_out  = Counter(flat_out)
    cnt_all  = Counter(flat_all)
    P()
    P("  (a) PERIOD DISTRIBUTION")
    P(f"  {'Period':>8}  {'#out':>6}  {'#all':>8}  {'f_out':>7}  {'f_all':>7}  {'ratio':>6}")
    P(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}")
    flagged = []
    for period, co in sorted(cnt_out.items(), key=lambda x: -x[1])[:15]:
        ca   = cnt_all.get(period, 0)
        fo   = co / max(len(flat_out), 1)
        fa   = ca / max(len(flat_all), 1)
        r    = fo / fa if fa > 0 else float("inf")
        flag = " ←" if r > 1.5 else ""
        P(f"  {period:>8}  {co:>6}  {ca:>8}  {fo:>7.3f}  {fa:>7.3f}  {r:>6.2f}{flag}")
        if r > 1.5:
            flagged.append((period, r))
    if flagged:
        P("  Over-represented (>1.5×): " + ", ".join(f"p={p}({r:.2f}×)" for p, r in flagged))
    else:
        P("  No period >1.5× over-represented.")

    # Residual ratio
    out_rs = np.array(outlier_resid_sig)
    all_rs = all_resid_sig
    rs_ratio = np.mean(out_rs) / np.mean(all_rs) if np.mean(all_rs) > 0 else float("nan")
    P()
    P("  (b) RESIDUAL RATIO")
    P(f"  Outliers: mean={np.mean(out_rs):.4f}  std={np.std(out_rs):.4f}  frac>0.5={np.mean(out_rs>0.5):.3f}")
    P(f"  All:      mean={np.mean(all_rs):.4f}  std={np.std(all_rs):.4f}  ratio={rs_ratio:.2f}×")

    # L5 attention
    P()
    P("  (c) LAYER-5 PREFIX ATTENTION  (encoder pass for outlier series)")
    l5_outlier = []
    l5_all_arr = []
    for b in range(n_batches):
        b0   = b * BATCH_SIZE
        idxs = list(range(b0, min(b0 + BATCH_SIZE, n_series)))
        bs   = len(idxs)

        ctx_t_b = torch.tensor(np.stack([all_ctx[i] for i in idxs]), dtype=torch.float32)
        inp_ids, att_msk, _ = pipeline.tokenizer.context_input_transform(
            pipeline._prepare_and_validate_context(ctx_t_b)
        )
        inp_ids = inp_ids.to(DEVICE)
        att_msk = att_msk.to(DEVICE)
        tr_tb = torch.tensor(np.stack([all_trend[i]    for i in idxs]), dtype=torch.float32).to(DEVICE)
        se_tb = torch.tensor(np.stack([all_seasonal[i] for i in idxs]), dtype=torch.float32).to(DEVICE)
        re_tb = torch.tensor(np.stack([all_residual[i] for i in idxs]), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            try:
                inject_prefix(chronos_model, pg, trend=tr_tb, seasonal=se_tb, residual=re_tb,
                               training_state=None)
                enc = chronos_model.model.encoder(
                    input_ids=inp_ids, attention_mask=att_msk, output_attentions=True,
                )
                l5_w = enc.attentions[5]
                for k, i in enumerate(idxs):
                    pfx_w = l5_w[k, :, :, :PREFIX_TOTAL].mean().item() * PREFIX_TOTAL
                    l5_all_arr.append((i, pfx_w))
            finally:
                remove_prefix_hooks(chronos_model)

    l5_all_arr_np = np.array([v for _, v in sorted(l5_all_arr)])
    l5_out_np     = l5_all_arr_np[outlier_pos]
    P(f"  Outliers: mean={np.mean(l5_out_np):.4f}  std={np.std(l5_out_np):.4f}")
    P(f"  All:      mean={np.mean(l5_all_arr_np):.4f}  std={np.std(l5_all_arr_np):.4f}")
else:
    P()
    P("6. OUTLIER DIAGNOSIS  — skipped (prefix helps on GiftEval)")

P()
P("=" * 90)
P("DONE.")
P("=" * 90)


# ── Save outputs ───────────────────────────────────────────────────────────────

per_series = []
for i in range(n_series):
    per_series.append({
        "item_id":   table.column("item_id")[i].as_py(),
        "mse_base":  float(mse_base[i]),
        "mae_base":  float(mae_base[i]),
        "mse_pfx":   float(mse_pfx[i]),
        "mae_pfx":   float(mae_pfx[i]),
        "mase_base": float(mase_base[i]),
        "mase_pfx":  float(mase_pfx[i]),
        "resid_sig": float(all_resid_sig[i]),
        "fix1b":     bool(fix1b_mask[i]),
        "improved":  bool(mae_pfx[i] < mae_base[i]),
    })

result_json = {
    "checkpoint":    str(CKPT_PATH),
    "dataset":       str(ARROW_PATH),
    "n_series":      n_series,
    "agg_zero_shot": {"mse": agg_mse_b,  "mae": agg_mae_b,  "mase": agg_mase_b},
    "agg_with_pfx":  {"mse": agg_mse_p,  "mae": agg_mae_p,  "mase": agg_mase_p},
    "improved_pct":  improved_mae / n_series * 100,
    "mse_dist": {
        "base": {"mean": float(np.mean(mse_base)), "p95": float(np.percentile(mse_base, 95)), "max": float(np.max(mse_base))},
        "pfx":  {"mean": float(np.mean(mse_pfx)),  "p95": float(np.percentile(mse_pfx, 95)),  "max": float(np.max(mse_pfx))},
    },
    "fix1b_rate":       n_filtered / n_series,
    "ceil_fire_rate":   ceil_fire_rate,
    "pre_ceil_mean":    pre_ceil_mean,
    "per_series":       per_series,
}

with open(str(OUTPUT_PATH), "w") as f:
    json.dump(result_json, f, indent=2)
print(f"\nSaved JSON:    {OUTPUT_PATH}")

summary_text = "\n".join(lines)
with open(str(SUMMARY_PATH), "w") as f:
    f.write(summary_text + "\n")
print(f"Saved summary: {SUMMARY_PATH}")
