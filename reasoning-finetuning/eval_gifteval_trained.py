#!/usr/bin/env python3
"""
eval_gifteval_trained.py — Phase 2 Task 2: GiftEval evaluation with trained prefix.

3-column comparison on the 83 held-out val series (80/20 split, seed=42):
  Column 1: zero-shot         (no prefix)
  Column 2: w/SarSim0-pfx     (checkpoint: run-2000steps-2000.pt, ceiling=0.80)
  Column 3: w/GiftEval-pfx    (checkpoint: gifteval-1000steps.pt,  ceiling=0.80)

Val split is identical to training (seed=42, 80/20 on 414 series → 83 val series).

Dataset: m4hourly-tsf.arrow
  context: 96 pts (raw hourly scale)
  future:  target[96:] → 48 pts raw forecast ground truth

Outputs:
  results/phase2_gifteval_trained_eval.json
  results/phase2_gifteval_trained_summary.txt
"""

import argparse
import json
import sys
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
parser.add_argument("--ckpt_sarsim0", default="reasoning-finetuning/checkpoints/run-2000steps-2000.pt",
                    help="SarSim0-trained checkpoint (Phase 1 best)")
parser.add_argument("--ckpt_gifteval", default="reasoning-finetuning/checkpoints/gifteval-1000steps.pt",
                    help="GiftEval-trained checkpoint (Phase 2 Task 2)")
parser.add_argument("--dataset",  default="reasoning-finetuning/data/m4hourly-tsf.arrow")
parser.add_argument("--output",   default="results/phase2_gifteval_trained_eval.json")
args = parser.parse_args()

CKPT_SARSIM0   = Path(args.ckpt_sarsim0)
CKPT_GIFTEVAL  = Path(args.ckpt_gifteval)
ARROW_PATH     = Path(args.dataset)
OUTPUT_PATH    = Path(args.output)
SUMMARY_PATH   = OUTPUT_PATH.parent / "phase2_gifteval_trained_summary.txt"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Config ─────────────────────────────────────────────────────────────────────

TRAIN_SEED   = 42
NUM_SAMPLES  = 20
PRED_LEN     = 48
CONTEXT_LEN  = 96
TARGET_OFFSET = 96
MASE_PERIOD  = 24
BATCH_SIZE   = 16
D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN   # 48
RANK         = 64
MIN_PERIOD   = 6
RS_THRESHOLD = 0.4

# Inference ceiling: fixed at 0.80 (end-of-training) for both checkpoints
SARSIM0_CEIL_STEP  = 2000;  SARSIM0_CEIL_TOTAL = 2000
GIFTEVAL_CEIL_STEP = 1000;  GIFTEVAL_CEIL_TOTAL = 1000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Known aggregate results from Phase 1 (SarSim0 training, full 414-series eval)
REF_PHASE1 = {
    "sarsim0_zero_shot": {"mse": 3.3456,      "mae": 0.6714, "mase": 0.6545},
    "sarsim0_with_pfx":  {"mse": 3.3283,      "mae": 0.6724, "mase": 0.6511},
    "gifteval_crossdomain_zero_shot": {"mse": 2389428.0,   "mae": 321.99, "mase": 1.61},
    "gifteval_crossdomain_sarsim0":   {"mse": 659470080.0, "mae": 3570.91, "mase": 27.62},
}


# ── Load dataset — val split only ─────────────────────────────────────────────

print(f"Loading {ARROW_PATH} …")
with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_total   = len(table)
rng_split = np.random.default_rng(TRAIN_SEED)
all_idx   = rng_split.permutation(n_total)
n_train   = int(n_total * 0.8)
val_idx   = all_idx[n_train:]   # 83 held-out series

n_val = len(val_idx)
print(f"  {n_total} total  →  {n_train} train / {n_val} val")
print(f"  Evaluating {n_val} held-out val series")

# Verify layout
ctx_0 = np.array(table.column("context")[int(val_idx[0])].as_py(), dtype=np.float32)
tgt_0 = np.array(table.column("target")[int(val_idx[0])].as_py(),  dtype=np.float32)
assert len(ctx_0) == CONTEXT_LEN
assert len(tgt_0[TARGET_OFFSET:]) == PRED_LEN
print(f"  context={CONTEXT_LEN} pts, future=target[{TARGET_OFFSET}:] ({PRED_LEN} pts)  ✓\n")


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


# ── Precompute decompositions for val series ──────────────────────────────────

print("Precomputing STL decompositions (Fix 1a: min_period=6, Fix 1b: rs>0.4→0) …")

all_ctx       = []
all_ftr       = []
all_trend     = []
all_seasonal  = []
all_residual  = []
all_naive_mae_ = []
all_resid_sig = []
n_filtered    = 0

for raw_i in val_idx:
    i   = int(raw_i)
    ctx = np.array(table.column("context")[i].as_py(), dtype=np.float32)
    tgt = np.array(table.column("target")[i].as_py(),  dtype=np.float32)
    ftr = tgt[TARGET_OFFSET:]

    pds = detect_periods(ctx, min_period=MIN_PERIOD)
    tr, se, re = stl_decompose(ctx, pds)
    rs  = float(np.var(re) / (np.var(ctx) + 1e-9))
    all_resid_sig.append(rs)

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
    all_naive_mae_.append(_naive_mae(ctx))

all_resid_sig = np.array(all_resid_sig)
fix1b_mask    = all_resid_sig > RS_THRESHOLD
print(f"  Done.  Fix 1b triggered: {n_filtered}/{n_val} ({n_filtered/n_val*100:.1f}%)\n")

n_batches = (n_val + BATCH_SIZE - 1) // BATCH_SIZE


# ── Load Chronos ───────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
print("Loading Chronos T5-Small …")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=str(DEVICE),
    dtype=torch.float32,
)
chronos_model = pipeline.model
tokenizer     = pipeline.tokenizer
for p in chronos_model.parameters():
    p.requires_grad_(False)
chronos_model.eval()
print(f"  {sum(p.numel() for p in chronos_model.parameters()):,} params frozen\n")


# ── Helper: run one eval pass (prefix or baseline) ────────────────────────────

def run_baseline_pass(label: str):
    """Zero-shot pass: no prefix."""
    print(f"{'='*70}")
    print(f"BASELINE PASS — {label}  ({n_val} series)")
    print(f"{'='*70}")
    mse_arr = np.zeros(n_val, dtype=np.float64)
    mae_arr = np.zeros(n_val, dtype=np.float64)

    for b in range(n_batches):
        b0   = b * BATCH_SIZE
        idxs = range(b0, min(b0 + BATCH_SIZE, n_val))
        ctx_t = torch.tensor(np.stack([all_ctx[i] for i in idxs]), dtype=torch.float32)
        ftr_a = np.stack([all_ftr[i] for i in idxs])

        with torch.no_grad():
            samp = pipeline.predict(
                ctx_t, prediction_length=PRED_LEN,
                num_samples=NUM_SAMPLES, limit_prediction_length=False,
            )
            med = samp.median(dim=1).values.numpy()

        for k, i in enumerate(idxs):
            mse_arr[i] = _mse(med[k], ftr_a[k])
            mae_arr[i] = _mae(med[k], ftr_a[k])

        if (b + 1) % 3 == 0 or (b + 1) == n_batches:
            done = min((b + 1) * BATCH_SIZE, n_val)
            print(f"  [{done:>3}/{n_val}]  MAE={np.mean(mae_arr[:done]):.4f}")

    mase_arr = np.array([
        mae_arr[i] / all_naive_mae_[i] if all_naive_mae_[i] > 0 else float("nan")
        for i in range(n_val)
    ])
    print(f"\n  {label}: MSE={nanmean(mse_arr):.4f}  "
          f"MAE={nanmean(mae_arr):.4f}  MASE={nanmean(mase_arr):.4f}\n")
    return mse_arr, mae_arr, mase_arr


def run_prefix_pass(label: str, pg: PrefixGenerator,
                    ceil_step: int, ceil_total: int):
    """Prefix inference pass with ceiling fixed at end-of-training value."""
    infer_state = {
        'current_step':                   ceil_step,
        'total_steps':                    ceil_total,
        'ceiling_fired_accumulator':      [],
        'pfx_total_pre_ceil_accumulator': [],
    }
    ceil_val = min(0.5 + 0.3 * (ceil_step / ceil_total), 0.8)

    print(f"{'='*70}")
    print(f"PREFIX PASS — {label}  (ceiling={ceil_val:.2f} fixed)  ({n_val} series)")
    print(f"{'='*70}")

    mse_arr = np.zeros(n_val, dtype=np.float64)
    mae_arr = np.zeros(n_val, dtype=np.float64)

    for b in range(n_batches):
        b0   = b * BATCH_SIZE
        idxs = range(b0, min(b0 + BATCH_SIZE, n_val))

        ctx_t = torch.tensor(np.stack([all_ctx[i]      for i in idxs]), dtype=torch.float32)
        ftr_a = np.stack(            [all_ftr[i]       for i in idxs])
        # Normalise STL components by ctx_scale — must match train_ddp.py _tokenize_batch.
        # PrefixGenerator was trained on ctx_scale-normalised inputs; raw-scale inputs
        # cause attention collapse at inference (pre_ceil_mean → 0.99+).
        _, _, ctx_scale = tokenizer.context_input_transform(
            pipeline._prepare_and_validate_context(ctx_t)
        )
        scale = ctx_scale.to(DEVICE).unsqueeze(-1)   # (B, 1)
        tr_t  = torch.tensor(np.stack([all_trend[i]    for i in idxs]), dtype=torch.float32).to(DEVICE) / scale
        se_t  = torch.tensor(np.stack([all_seasonal[i] for i in idxs]), dtype=torch.float32).to(DEVICE) / scale
        re_t  = torch.tensor(np.stack([all_residual[i] for i in idxs]), dtype=torch.float32).to(DEVICE) / scale

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
            mse_arr[i] = _mse(med[k], ftr_a[k])
            mae_arr[i] = _mae(med[k], ftr_a[k])

        if (b + 1) % 3 == 0 or (b + 1) == n_batches:
            done = min((b + 1) * BATCH_SIZE, n_val)
            print(f"  [{done:>3}/{n_val}]  prefix={np.mean(mae_arr[:done]):.4f}")

    mase_arr = np.array([
        mae_arr[i] / all_naive_mae_[i] if all_naive_mae_[i] > 0 else float("nan")
        for i in range(n_val)
    ])

    ceil_acc  = infer_state['ceiling_fired_accumulator']
    pre_acc   = infer_state['pfx_total_pre_ceil_accumulator']
    ceil_fire = float(np.mean(ceil_acc)) if ceil_acc else float("nan")
    pre_mean  = float(np.mean(pre_acc))  if pre_acc  else float("nan")

    print(f"\n  {label}: MSE={nanmean(mse_arr):.4f}  "
          f"MAE={nanmean(mae_arr):.4f}  MASE={nanmean(mase_arr):.4f}")
    print(f"  Ceiling fire rate: {ceil_fire*100:.2f}%   "
          f"Pre-ceiling mean pfx_total: {pre_mean:.4f}\n")

    return mse_arr, mae_arr, mase_arr, ceil_fire, pre_mean


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION PASSES
# ══════════════════════════════════════════════════════════════════════════════

# Pass 1: zero-shot
mse_base, mae_base, mase_base = run_baseline_pass("zero-shot")

# Pass 2: SarSim0-trained prefix
print(f"Loading SarSim0 checkpoint: {CKPT_SARSIM0} …")
ckpt_ss = torch.load(str(CKPT_SARSIM0), map_location=DEVICE)
pg_ss   = PrefixGenerator(d_model=D_MODEL, num_layers=NUM_LAYERS,
                           prefix_len_per_component=PREFIX_LEN, rank=RANK).to(DEVICE)
pg_ss.load_state_dict(ckpt_ss["pg_state"])
pg_ss.eval()
print(f"  Loaded step={ckpt_ss['step']}\n")

mse_ss, mae_ss, mase_ss, ceil_fire_ss, pre_mean_ss = run_prefix_pass(
    "w/SarSim0-pfx", pg_ss,
    ceil_step=SARSIM0_CEIL_STEP, ceil_total=SARSIM0_CEIL_TOTAL,
)
del pg_ss   # free memory before loading next checkpoint

# Pass 3: GiftEval-trained prefix
print(f"Loading GiftEval checkpoint: {CKPT_GIFTEVAL} …")
ckpt_ge = torch.load(str(CKPT_GIFTEVAL), map_location=DEVICE)
pg_ge   = PrefixGenerator(d_model=D_MODEL, num_layers=NUM_LAYERS,
                           prefix_len_per_component=PREFIX_LEN, rank=RANK).to(DEVICE)
pg_ge.load_state_dict(ckpt_ge["pg_state"])
pg_ge.eval()
print(f"  Loaded step={ckpt_ge['step']}\n")

mse_ge, mae_ge, mase_ge, ceil_fire_ge, pre_mean_ge = run_prefix_pass(
    "w/GiftEval-pfx", pg_ge,
    ceil_step=GIFTEVAL_CEIL_STEP, ceil_total=GIFTEVAL_CEIL_TOTAL,
)
del pg_ge


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE METRICS
# ══════════════════════════════════════════════════════════════════════════════

agg_mse_b  = nanmean(mse_base);   agg_mae_b  = nanmean(mae_base);   agg_mase_b  = nanmean(mase_base)
agg_mse_ss = nanmean(mse_ss);     agg_mae_ss = nanmean(mae_ss);     agg_mase_ss = nanmean(mase_ss)
agg_mse_ge = nanmean(mse_ge);     agg_mae_ge = nanmean(mae_ge);     agg_mase_ge = nanmean(mase_ge)

improv_ss = int(np.sum(mae_ss < mae_base))
improv_ge = int(np.sum(mae_ge < mae_base))


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

lines = []

def P(s=""):
    lines.append(s)
    print(s)

P("=" * 90)
P("PHASE 2 TASK 2 — GiftEval evaluation with GiftEval-trained prefix")
P(f"  SarSim0 ckpt:  {CKPT_SARSIM0.name}  (step {ckpt_ss['step']})")
P(f"  GiftEval ckpt: {CKPT_GIFTEVAL.name}  (step {ckpt_ge['step']})")
P(f"  Dataset:       {ARROW_PATH.name}  ({n_val} held-out val series, 80/20 split seed=42)")
P(f"  Preprocessing: min_period={MIN_PERIOD} (Fix 1a), resid_sig>{RS_THRESHOLD}→zeros (Fix 1b)")
P(f"  Inference ceiling: 0.80 (fixed at end-of-training) for both prefix checkpoints")
P("=" * 90)

P()
P("1. MAIN COMPARISON TABLE  (on 83 held-out val series)")
P()
hdr = (f"{'Metric':<9}  {'GiftEval':>14}  {'GiftEval':>16}  {'GiftEval':>16}")
sub = (f"{'':9}  {'zero-shot':>14}  {'w/SarSim0-pfx':>16}  {'w/GiftEval-pfx':>16}")
sep = f"{'─'*9}  {'─'*14}  {'─'*16}  {'─'*16}"
P(hdr)
P(sub)
P(sep)
P(f"{'MSE':<9}  {agg_mse_b:>14.2f}  {agg_mse_ss:>16.2f}  {agg_mse_ge:>16.2f}")
P(f"{'MAE':<9}  {agg_mae_b:>14.4f}  {agg_mae_ss:>16.4f}  {agg_mae_ge:>16.4f}")
P(f"{'MASE':<9}  {agg_mase_b:>14.4f}  {agg_mase_ss:>16.4f}  {agg_mase_ge:>16.4f}")
P(f"{'Improv%':<9}  {'—':>14}  {improv_ss/n_val*100:>15.1f}%  {improv_ge/n_val*100:>15.1f}%")

P()
P(f"  ΔMASE (SarSim0-pfx vs zero-shot): {dpct(agg_mase_b, agg_mase_ss):+.2f}%")
P(f"  ΔMASE (GiftEval-pfx vs zero-shot): {dpct(agg_mase_b, agg_mase_ge):+.2f}%")
P(f"  ΔMASE (GiftEval-pfx vs SarSim0-pfx): {dpct(agg_mase_ss, agg_mase_ge):+.2f}%")

# Context from Phase 2 Task 1 (all 414 series, for reference)
P()
P("  [Reference from Phase 2 Task 1 — full 414-series eval, not val split]")
P(f"  {'':9}  {'zero-shot':>14}  {'w/SarSim0-pfx':>16}")
P(f"  {'MSE':9}  {REF_PHASE1['gifteval_crossdomain_zero_shot']['mse']:>14.0f}  "
  f"{REF_PHASE1['gifteval_crossdomain_sarsim0']['mse']:>16.0f}")
P(f"  {'MASE':9}  {REF_PHASE1['gifteval_crossdomain_zero_shot']['mase']:>14.2f}  "
  f"{REF_PHASE1['gifteval_crossdomain_sarsim0']['mase']:>16.2f}")

# ── 2. Per-series MSE distribution ────────────────────────────────────────────
P()
P("─" * 90)
P("2. PER-SERIES MSE DISTRIBUTION")
P()
P(f"  {'Stat':<8}  {'zero-shot':>14}  {'w/SarSim0-pfx':>16}  {'w/GiftEval-pfx':>16}")
P(f"  {'─'*8}  {'─'*14}  {'─'*16}  {'─'*16}")
for stat, fn in [("mean", np.mean),
                 ("p95",  lambda a: np.percentile(a, 95)),
                 ("max",  np.max)]:
    P(f"  {stat:<8}  {fn(mse_base):>14.2f}  {fn(mse_ss):>16.2f}  {fn(mse_ge):>16.2f}")

# ── 3. Fix 1b filter rate ──────────────────────────────────────────────────────
P()
P("─" * 90)
P("3. FIX 1B FILTER RATE (val series)")
n_hino = int(np.sum(fix1b_mask))
P(f"  Series with resid_sig > {RS_THRESHOLD}: {n_hino}/{n_val} ({n_hino/n_val*100:.1f}%)")
P(f"  (SarSim0 reference: 2.9%;  M4H expected: < 2.9%, clean real-world data)")

# ── 4. Fix 2 ceiling activation ───────────────────────────────────────────────
P()
P("─" * 90)
P("4. FIX 2 CEILING ACTIVATION (inference, ceiling=0.80 fixed)")
P(f"  SarSim0-pfx:   fired={ceil_fire_ss*100:.2f}%  pre_ceil_mean={pre_mean_ss:.4f}")
P(f"  GiftEval-pfx:  fired={ceil_fire_ge*100:.2f}%  pre_ceil_mean={pre_mean_ge:.4f}")
P(f"  (Cross-domain SarSim0 reference: fired=98.6%, pre_ceil=0.9912)")
P(f"  Key diagnostic: GiftEval-pfx pre_ceil_mean should be ~0.40–0.50 (matching training) if no collapse")

# ── 5. Per-series improvement ──────────────────────────────────────────────────
P()
P("─" * 90)
P("5. PER-SERIES IMPROVEMENT RATE (MAE, prefix vs zero-shot)")
P(f"  SarSim0-pfx:  {improv_ss}/{n_val} improved  ({improv_ss/n_val*100:.1f}%)")
P(f"  GiftEval-pfx: {improv_ge}/{n_val} improved  ({improv_ge/n_val*100:.1f}%)")

# ── Verdict ────────────────────────────────────────────────────────────────────
P()
P("─" * 90)
P("VERDICT")
gifteval_helps   = agg_mase_ge < agg_mase_b
sarsim0_helps    = agg_mase_ss < agg_mase_b
gifteval_vs_ss   = agg_mase_ge < agg_mase_ss
P(f"  SarSim0-pfx vs zero-shot:   {'HELPS' if sarsim0_helps else 'HURTS'}  "
  f"(ΔMASE={dpct(agg_mase_b, agg_mase_ss):+.2f}%)")
P(f"  GiftEval-pfx vs zero-shot:  {'HELPS' if gifteval_helps else 'HURTS'}  "
  f"(ΔMASE={dpct(agg_mase_b, agg_mase_ge):+.2f}%)")
P(f"  GiftEval-pfx vs SarSim0-pfx: {'BETTER' if gifteval_vs_ss else 'WORSE'}  "
  f"(ΔMASE={dpct(agg_mase_ss, agg_mase_ge):+.2f}%)")
if gifteval_helps:
    P("  → In-domain training SUCCEEDED: GiftEval-pfx improves over zero-shot")
else:
    P("  → In-domain training did not overcome zero-shot; investigate prefix collapse")

P()
P("=" * 90)
P("DONE.")
P("=" * 90)


# ── Save outputs ───────────────────────────────────────────────────────────────

per_series = []
for k, raw_i in enumerate(val_idx):
    per_series.append({
        "item_id":    table.column("item_id")[int(raw_i)].as_py(),
        "mse_base":   float(mse_base[k]),
        "mae_base":   float(mae_base[k]),
        "mse_ss":     float(mse_ss[k]),
        "mae_ss":     float(mae_ss[k]),
        "mse_ge":     float(mse_ge[k]),
        "mae_ge":     float(mae_ge[k]),
        "mase_base":  float(mase_base[k]),
        "mase_ss":    float(mase_ss[k]),
        "mase_ge":    float(mase_ge[k]),
        "resid_sig":  float(all_resid_sig[k]),
        "fix1b":      bool(fix1b_mask[k]),
        "improv_ss":  bool(mae_ss[k] < mae_base[k]),
        "improv_ge":  bool(mae_ge[k] < mae_base[k]),
    })

result_json = {
    "ckpt_sarsim0":     str(CKPT_SARSIM0),
    "ckpt_gifteval":    str(CKPT_GIFTEVAL),
    "dataset":          str(ARROW_PATH),
    "n_val":            n_val,
    "agg_zero_shot":    {"mse": agg_mse_b,  "mae": agg_mae_b,  "mase": agg_mase_b},
    "agg_sarsim0_pfx":  {"mse": agg_mse_ss, "mae": agg_mae_ss, "mase": agg_mase_ss},
    "agg_gifteval_pfx": {"mse": agg_mse_ge, "mae": agg_mae_ge, "mase": agg_mase_ge},
    "improv_pct_ss":    improv_ss / n_val * 100,
    "improv_pct_ge":    improv_ge / n_val * 100,
    "mse_dist": {
        "base": {"mean": float(np.mean(mse_base)), "p95": float(np.percentile(mse_base, 95)), "max": float(np.max(mse_base))},
        "ss":   {"mean": float(np.mean(mse_ss)),   "p95": float(np.percentile(mse_ss, 95)),   "max": float(np.max(mse_ss))},
        "ge":   {"mean": float(np.mean(mse_ge)),   "p95": float(np.percentile(mse_ge, 95)),   "max": float(np.max(mse_ge))},
    },
    "fix1b_rate":       n_filtered / n_val,
    "ceil_fire_ss":     ceil_fire_ss,
    "pre_mean_ss":      pre_mean_ss,
    "ceil_fire_ge":     ceil_fire_ge,
    "pre_mean_ge":      pre_mean_ge,
    "per_series":       per_series,
}

with open(str(OUTPUT_PATH), "w") as f:
    json.dump(result_json, f, indent=2)
print(f"\nSaved JSON:    {OUTPUT_PATH}")

summary_text = "\n".join(lines)
with open(str(SUMMARY_PATH), "w") as f:
    f.write(summary_text + "\n")
print(f"Saved summary: {SUMMARY_PATH}")
