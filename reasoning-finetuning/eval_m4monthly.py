#!/usr/bin/env python3
"""
eval_m4monthly.py — Sweep evaluation: M4 Monthly prefix vs zero-shot.

2-column comparison on 2,000 held-out val series (80/20 split, seed=42):
  Column 1: zero-shot         (no prefix)
  Column 2: w/m4monthly-pfx  (checkpoint: sweep-m4monthly-2000steps.pt, ceiling=0.80)

Outputs:
  results/sweep_m4monthly_eval.json
  results/sweep_m4monthly_summary.txt
  results/sweep_summary.md   (updated ablation table)
"""

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


# ── Config ─────────────────────────────────────────────────────────────────────

ARROW_PATH    = FINETUNING_ROOT / "data" / "m4monthly-tsf.arrow"
CKPT_PATH     = FINETUNING_ROOT / "checkpoints" / "sweep-m4monthly-2000steps.pt"
OUTPUT_JSON   = Path("results/sweep_m4monthly_eval.json")
OUTPUT_TXT    = Path("results/sweep_m4monthly_summary.txt")
SWEEP_MD      = Path("results/sweep_summary.md")

TRAIN_SEED    = 42
NUM_SAMPLES   = 20
PRED_LEN      = 18
CONTEXT_LEN   = 36
TARGET_OFFSET = 36
MASE_PERIOD   = 12
BATCH_SIZE    = 16
MIN_PERIOD    = 4
MAX_PERIOD    = 12
RS_THRESHOLD  = 0.4

CEIL_STEP     = 2000
CEIL_TOTAL    = 2000

D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN
RANK         = 64

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


# ── Load dataset — val split only ─────────────────────────────────────────────

print(f"Loading {ARROW_PATH.name} …")
with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_total   = len(table)
rng_split = np.random.default_rng(TRAIN_SEED)
all_idx   = rng_split.permutation(n_total)
n_train   = int(n_total * 0.8)
val_idx   = all_idx[n_train:]
n_val     = len(val_idx)

print(f"  {n_total} total  →  {n_train} train / {n_val} val")
print(f"  Evaluating {n_val} held-out val series")
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


# ── Precompute decompositions ──────────────────────────────────────────────────

print("Precomputing STL decompositions …")
all_ctx, all_ftr     = [], []
all_trend, all_seas, all_resid = [], [], []
all_naive_mae_       = []
all_resid_sig        = []
n_filtered           = 0

for raw_i in val_idx:
    i   = int(raw_i)
    ctx = np.array(table.column("context")[i].as_py(), dtype=np.float32)
    tgt = np.array(table.column("target")[i].as_py(),  dtype=np.float32)
    ftr = tgt[TARGET_OFFSET:]

    pds = detect_periods(ctx, min_period=MIN_PERIOD, max_period=MAX_PERIOD)
    tr, se, re = stl_decompose(ctx, pds)
    rs  = float(np.var(re) / (np.var(ctx) + 1e-9))
    all_resid_sig.append(rs)

    if rs > RS_THRESHOLD:
        tr = np.zeros_like(ctx, dtype=np.float32)
        se = np.zeros_like(ctx, dtype=np.float32)
        re = np.zeros_like(ctx, dtype=np.float32)
        n_filtered += 1

    all_ctx.append(ctx);  all_ftr.append(ftr)
    all_trend.append(tr.astype(np.float32))
    all_seas.append(se.astype(np.float32))
    all_resid.append(re.astype(np.float32))
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


# ── Eval passes ────────────────────────────────────────────────────────────────

def run_baseline_pass():
    print("=" * 70)
    print(f"BASELINE PASS — zero-shot  ({n_val} series)")
    print("=" * 70)
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

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            done = min((b + 1) * BATCH_SIZE, n_val)
            print(f"  [{done:>4}/{n_val}]  MAE={np.mean(mae_arr[:done]):.4f}")

    mase_arr = np.array([
        mae_arr[i] / all_naive_mae_[i] if all_naive_mae_[i] > 0 else float("nan")
        for i in range(n_val)
    ])
    print(f"\n  zero-shot: MSE={nanmean(mse_arr):.4f}  "
          f"MAE={nanmean(mae_arr):.4f}  MASE={nanmean(mase_arr):.4f}\n")
    return mse_arr, mae_arr, mase_arr


def run_prefix_pass(pg: PrefixGenerator):
    ceil_val = min(0.5 + 0.3 * (CEIL_STEP / CEIL_TOTAL), 0.8)
    infer_state = {
        'current_step':                   CEIL_STEP,
        'total_steps':                    CEIL_TOTAL,
        'ceiling_fired_accumulator':      [],
        'pfx_total_pre_ceil_accumulator': [],
    }
    print("=" * 70)
    print(f"PREFIX PASS — w/m4monthly-pfx  (ceiling={ceil_val:.2f} fixed)  ({n_val} series)")
    print("=" * 70)

    mse_arr = np.zeros(n_val, dtype=np.float64)
    mae_arr = np.zeros(n_val, dtype=np.float64)

    for b in range(n_batches):
        b0   = b * BATCH_SIZE
        idxs = range(b0, min(b0 + BATCH_SIZE, n_val))

        ctx_t = torch.tensor(np.stack([all_ctx[i]     for i in idxs]), dtype=torch.float32)
        ftr_a = np.stack(            [all_ftr[i]       for i in idxs])

        # ctx_scale normalisation — must match train_ddp.py _tokenize_batch
        _, _, ctx_scale = tokenizer.context_input_transform(
            pipeline._prepare_and_validate_context(ctx_t)
        )
        scale = ctx_scale.to(DEVICE).unsqueeze(-1)   # (B, 1)
        tr_t  = torch.tensor(np.stack([all_trend[i] for i in idxs]), dtype=torch.float32).to(DEVICE) / scale
        se_t  = torch.tensor(np.stack([all_seas[i]  for i in idxs]), dtype=torch.float32).to(DEVICE) / scale
        re_t  = torch.tensor(np.stack([all_resid[i] for i in idxs]), dtype=torch.float32).to(DEVICE) / scale

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

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            done = min((b + 1) * BATCH_SIZE, n_val)
            print(f"  [{done:>4}/{n_val}]  prefix={np.mean(mae_arr[:done]):.4f}")

    mase_arr = np.array([
        mae_arr[i] / all_naive_mae_[i] if all_naive_mae_[i] > 0 else float("nan")
        for i in range(n_val)
    ])
    ceil_acc  = infer_state['ceiling_fired_accumulator']
    pre_acc   = infer_state['pfx_total_pre_ceil_accumulator']
    ceil_fire = float(np.mean(ceil_acc)) if ceil_acc else float("nan")
    pre_mean  = float(np.mean(pre_acc))  if pre_acc  else float("nan")
    print(f"\n  w/m4monthly-pfx: MSE={nanmean(mse_arr):.4f}  "
          f"MAE={nanmean(mae_arr):.4f}  MASE={nanmean(mase_arr):.4f}")
    print(f"  Ceiling fire rate: {ceil_fire*100:.2f}%   "
          f"Pre-ceiling mean pfx_total: {pre_mean:.4f}\n")
    return mse_arr, mae_arr, mase_arr, ceil_fire, pre_mean


# ── Run passes ─────────────────────────────────────────────────────────────────

mse_base, mae_base, mase_base = run_baseline_pass()

print(f"Loading checkpoint: {CKPT_PATH} …")
ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)
pg   = PrefixGenerator(d_model=D_MODEL, num_layers=NUM_LAYERS,
                       prefix_len_per_component=PREFIX_LEN, rank=RANK).to(DEVICE)
pg.load_state_dict(ckpt["pg_state"])
pg.eval()
print(f"  Loaded step={ckpt['step']}\n")

mse_pfx, mae_pfx, mase_pfx, ceil_fire, pre_mean = run_prefix_pass(pg)
del pg


# ── Aggregate ──────────────────────────────────────────────────────────────────

agg_mse_b = nanmean(mse_base);  agg_mae_b = nanmean(mae_base);  agg_mase_b = nanmean(mase_base)
agg_mse_p = nanmean(mse_pfx);   agg_mae_p = nanmean(mae_pfx);   agg_mase_p = nanmean(mase_pfx)
improv    = int(np.sum(mae_pfx < mae_base))


# ── Report ─────────────────────────────────────────────────────────────────────

lines = []
def P(s=""):
    lines.append(s)
    print(s)

P("=" * 90)
P("SWEEP — M4 Monthly  (Phase 2 dataset sweep)")
P(f"  Checkpoint:  {CKPT_PATH.name}  (step {ckpt['step']})")
P(f"  Dataset:     {ARROW_PATH.name}  ({n_val} held-out val series, 80/20 split seed=42)")
P(f"  Context:     {CONTEXT_LEN} pts  |  Pred: {PRED_LEN} pts  |  MASE period: {MASE_PERIOD}")
P(f"  STL params:  min_period={MIN_PERIOD}, max_period={MAX_PERIOD}")
P(f"  Inference ceiling: 0.80 (fixed at end-of-training, step={CEIL_STEP}/{CEIL_TOTAL})")
P("=" * 90)

P()
P("1. MAIN COMPARISON TABLE  (on 2,000 held-out val series)")
P()
hdr = f"{'Metric':<9}  {'Zero-shot':>14}  {'w/prefix':>14}  {'Δ(%)':>10}"
sep = f"{'─'*9}  {'─'*14}  {'─'*14}  {'─'*10}"
P(hdr); P(sep)
P(f"{'MSE':<9}  {agg_mse_b:>14.2f}  {agg_mse_p:>14.2f}  {dpct(agg_mse_b, agg_mse_p):>+9.1f}%")
P(f"{'MAE':<9}  {agg_mae_b:>14.4f}  {agg_mae_p:>14.4f}  {dpct(agg_mae_b, agg_mae_p):>+9.2f}%")
P(f"{'MASE':<9}  {agg_mase_b:>14.4f}  {agg_mase_p:>14.4f}  {dpct(agg_mase_b, agg_mase_p):>+9.2f}%")
P(f"{'Improv%':<9}  {'—':>14}  {improv/n_val*100:>13.1f}%  {'—':>10}")

P()
P(f"  ΔMASE (w/prefix vs zero-shot): {dpct(agg_mase_b, agg_mase_p):+.2f}%")

P()
P("─" * 90)
P("2. PER-SERIES MSE DISTRIBUTION")
P()
P(f"  {'Stat':<8}  {'Zero-shot':>14}  {'w/prefix':>14}")
P(f"  {'─'*8}  {'─'*14}  {'─'*14}")
for stat, fn in [("mean", np.mean), ("p95", lambda a: np.percentile(a, 95)), ("max", np.max)]:
    P(f"  {stat:<8}  {fn(mse_base):>14.2f}  {fn(mse_pfx):>14.2f}")

P()
P("─" * 90)
P("3. FIX 1B FILTER RATE (val series)")
n_hino = int(np.sum(fix1b_mask))
P(f"  Series with resid_sig > {RS_THRESHOLD}: {n_hino}/{n_val} ({n_hino/n_val*100:.1f}%)")
P(f"  (M4H reference: 0%;  monthly economic series may be higher)")

P()
P("─" * 90)
P("4. FIX 2 CEILING ACTIVATION (inference, ceiling=0.80 fixed)")
P(f"  w/m4monthly-pfx:  fired={ceil_fire*100:.2f}%  pre_ceil_mean={pre_mean:.4f}")
P(f"  (Healthy range: pre_ceil_mean ~0.40–0.50, fire rate ~15–30%)")

P()
P("─" * 90)
P("5. PER-SERIES IMPROVEMENT RATE (MAE)")
P(f"  w/m4monthly-pfx: {improv}/{n_val} improved  ({improv/n_val*100:.1f}%)")

P()
P("─" * 90)
P("VERDICT")
helps = agg_mase_p < agg_mase_b
P(f"  w/m4monthly-pfx vs zero-shot: {'HELPS' if helps else 'HURTS'}  "
  f"(ΔMASE={dpct(agg_mase_b, agg_mase_p):+.2f}%)")
if helps:
    P("  → In-domain training SUCCEEDED on M4 Monthly")
else:
    P("  → Prefix did not improve over zero-shot — investigate")

P()
P("=" * 90)
P("DONE.")
P("=" * 90)


# ── Save JSON + TXT ────────────────────────────────────────────────────────────

result_json = {
    "dataset":          "m4monthly",
    "ckpt":             str(CKPT_PATH),
    "n_val":            n_val,
    "agg_zero_shot":    {"mse": agg_mse_b, "mae": agg_mae_b, "mase": agg_mase_b},
    "agg_prefix":       {"mse": agg_mse_p, "mae": agg_mae_p, "mase": agg_mase_p},
    "delta_mase_pct":   dpct(agg_mase_b, agg_mase_p),
    "improv_pct":       improv / n_val * 100,
    "mse_dist": {
        "base": {"mean": float(np.mean(mse_base)), "p95": float(np.percentile(mse_base, 95)), "max": float(np.max(mse_base))},
        "pfx":  {"mean": float(np.mean(mse_pfx)),  "p95": float(np.percentile(mse_pfx,  95)), "max": float(np.max(mse_pfx))},
    },
    "fix1b_rate":      n_hino / n_val,
    "ceil_fire":       ceil_fire,
    "pre_mean":        pre_mean,
}

with open(str(OUTPUT_JSON), "w") as f:
    json.dump(result_json, f, indent=2)
print(f"\nSaved JSON:    {OUTPUT_JSON}")

with open(str(OUTPUT_TXT), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary: {OUTPUT_TXT}")


# ── Update sweep_summary.md ────────────────────────────────────────────────────

dmase_str = f"{dpct(agg_mase_b, agg_mase_p):+.1f}%"
improv_str = f"{improv/n_val*100:.1f}%"

sweep_table = f"""\
# Sweep Summary — Prefix Tuning Ablation

| Group | Context | Period | N_train | MASE_base | MASE_pfx | ΔMASE | Improv% |
|---|---|---|---|---|---|---|---|
| SarSim0 | 512 | mixed | 8,000 | 0.6545 | 0.6511 | −0.52% | 50.4% |
| M4 Hourly | 96 | 24 | 331 | 1.6565 | 1.2181 | −26.5% | 77.1% |
| M4 Monthly | 36 | 12 | 8,000 | {agg_mase_b:.4f} | {agg_mase_p:.4f} | {dmase_str} | {improv_str} |
| M4 Daily | — | 7 | — | TBD | TBD | TBD | TBD |
| M4 Weekly | — | 52 | — | TBD | TBD | TBD | TBD |
| ETT Hourly | — | 24 | — | TBD | TBD | TBD | TBD |

_Updated after M4 Monthly run._
"""

with open(str(SWEEP_MD), "w") as f:
    f.write(sweep_table)
print(f"Saved sweep:   {SWEEP_MD}")
