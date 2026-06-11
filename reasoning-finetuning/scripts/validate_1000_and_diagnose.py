#!/usr/bin/env python3
"""
validate_1000_and_diagnose.py — Step A + Step B

Step A: Validate checkpoints/run-1000steps-1000.pt on the 20% held-out split
        of sarsim0-10k.arrow (same seed=42 split as prior runs).
        Report 4-column table: Zero-shot / Step-500 / Step-1000 / Δ(500→1000)

Step B: Diagnose worst 5% of series by per-series MSE (with prefix).
        Per outlier: detected periods, MSTL residual ratio, Layer-5 prefix
        attention weight, MAE delta. Aggregated and compared to full set.
"""

import sys
from pathlib import Path
from collections import Counter

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
CKPT_PATH    = FINETUNING_ROOT / "checkpoints" / "run-1000steps-1000.pt"
TRAIN_SEED   = 42
NUM_SAMPLES  = 20
PRED_LEN     = 64
MASE_PERIOD  = 24
BATCH_SIZE   = 16
D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN   # 48
RANK         = 64

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Step-500 reference values (from prior validation run)
REF_MSE_B,  REF_MSE_P  = 3.3456, 18.6146
REF_MAE_B,  REF_MAE_P  = 0.6714,  0.8182
REF_MASE_B, REF_MASE_P = 0.6545,  0.7037
REF_FRAC_MAE           = 40.0


# ── Dataset ────────────────────────────────────────────────────────────────────

print("Loading sarsim0-10k.arrow …")
with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_total = len(table)
rng     = np.random.default_rng(TRAIN_SEED)
indices = rng.permutation(n_total)
val_idx = indices[int(n_total * 0.8):]   # 20% held-out
print(f"  {n_total} rows  →  {len(val_idx)} validation")


# ── Load Chronos & PrefixGenerator ────────────────────────────────────────────

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

print(f"Loading checkpoint: {CKPT_PATH} …")
ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)
pg = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(DEVICE)
pg.load_state_dict(ckpt["pg_state"])
pg.eval()
print(f"  Chronos:    {sum(p.numel() for p in chronos_model.parameters()):,} params (frozen)")
print(f"  PrefixGen:  step={ckpt['step']}  "
      f"{sum(p.numel() for p in pg.parameters()):,} params\n")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _mse(p, t): return float(np.mean((p - t) ** 2))
def _mae(p, t): return float(np.mean(np.abs(p - t)))

def _naive_mae(ctx, period=MASE_PERIOD):
    return float(np.mean(np.abs(ctx[period:] - ctx[:-period]))) if len(ctx) > period else float("nan")

def nanmean(lst):
    a = np.array([x for x in lst if not np.isnan(x)])
    return float(a.mean()) if len(a) else float("nan")

def dpct(base, new):
    if base == 0 or np.isnan(base) or np.isnan(new): return float("nan")
    return (new - base) / base * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# STEP A — Validation loop
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print(f"STEP A — VALIDATION  ({len(val_idx)} series, num_samples={NUM_SAMPLES})")
print("=" * 65)

mse_pfx, mse_bas  = [], []
mae_pfx, mae_bas  = [], []
mase_pfx, mase_bas = [], []
periods_all    = []   # list[list[int]], one per val series
resid_sig_all  = []   # float per val series
improved_mse   = 0
improved_mae   = 0

n_batches = (len(val_idx) + BATCH_SIZE - 1) // BATCH_SIZE

for b in range(n_batches):
    b_start = b * BATCH_SIZE
    idxs    = val_idx[b_start : b_start + BATCH_SIZE]
    bs      = len(idxs)

    ctxs, ftrs, trens, seas, ress = [], [], [], [], []
    nmaes, bpds, brsig = [], [], []

    for idx in idxs:
        ctx = np.array(table.column("context")[int(idx)].as_py(), dtype=np.float32)
        ftr = np.array(table.column("forecast")[int(idx)].as_py(), dtype=np.float32)
        pds = detect_periods(ctx)
        tr, se, re = stl_decompose(ctx, pds)
        rs  = float(np.var(re) / (np.var(ctx) + 1e-8))

        ctxs.append(ctx);  ftrs.append(ftr)
        trens.append(tr);  seas.append(se);  ress.append(re)
        nmaes.append(_naive_mae(ctx))
        bpds.append(pds);  brsig.append(rs)

    # ctx_t stays on CPU — pipeline tokenizer's boundaries tensor is always CPU
    ctx_t = torch.tensor(np.stack(ctxs),  dtype=torch.float32)
    ftr_a = np.stack(ftrs)
    tr_t  = torch.tensor(np.stack(trens), dtype=torch.float32).to(DEVICE)
    se_t  = torch.tensor(np.stack(seas),  dtype=torch.float32).to(DEVICE)
    re_t  = torch.tensor(np.stack(ress),  dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        # ── Prefix forecast ────────────────────────────────────────────────
        try:
            inject_prefix(chronos_model, pg, trend=tr_t, seasonal=se_t, residual=re_t)
            pfx_samples = pipeline.predict(
                ctx_t, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES,
                limit_prediction_length=False,
            )
            pfx_med = pfx_samples.median(dim=1).values.numpy()
        finally:
            remove_prefix_hooks(chronos_model)

        # ── Baseline forecast ──────────────────────────────────────────────
        bas_samples = pipeline.predict(
            ctx_t, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES,
            limit_prediction_length=False,
        )
        bas_med = bas_samples.median(dim=1).values.numpy()

    for i in range(bs):
        gt = ftr_a[i]; p = pfx_med[i]; q = bas_med[i]; nm = nmaes[i]
        mp, mb = _mse(p, gt), _mse(q, gt)
        ap, ab = _mae(p, gt), _mae(q, gt)

        mse_pfx.append(mp);  mse_bas.append(mb)
        mae_pfx.append(ap);  mae_bas.append(ab)
        mase_pfx.append(ap / nm if nm > 0 else float("nan"))
        mase_bas.append(ab / nm if nm > 0 else float("nan"))
        periods_all.append(bpds[i])
        resid_sig_all.append(brsig[i])
        if mp < mb: improved_mse += 1
        if ap < ab: improved_mae += 1

    if (b + 1) % 10 == 0 or (b + 1) == n_batches:
        done = min((b + 1) * BATCH_SIZE, len(val_idx))
        print(f"  [{done:>4}/{len(val_idx)}]  MAE  prefix={np.mean(mae_pfx):.4f}  "
              f"base={np.mean(mae_bas):.4f}")


# ── Step A: report ─────────────────────────────────────────────────────────────

agg_mse_p  = nanmean(mse_pfx);  agg_mse_b  = nanmean(mse_bas)
agg_mae_p  = nanmean(mae_pfx);  agg_mae_b  = nanmean(mae_bas)
agg_mase_p = nanmean(mase_pfx); agg_mase_b = nanmean(mase_bas)
n_val      = len(val_idx)

print(f"\n{'='*80}")
print("STEP A — COMPARISON TABLE")
print(f"  Checkpoint: {CKPT_PATH.name}  (step {ckpt['step']})")
print(f"  Val series: {n_val}  |  num_samples: {NUM_SAMPLES}  |  MASE period: {MASE_PERIOD}")
print(f"{'='*80}")
print(f"\n{'Metric':<8} {'Zero-shot':>12}  {'Step-500':>12}  {'Step-1000':>12}  {'Δ(500→1000)':>14}")
print(f"{'─'*8} {'─'*12}  {'─'*12}  {'─'*12}  {'─'*14}")
print(f"{'MSE':<8} {REF_MSE_B:>12.4f}  {REF_MSE_P:>12.4f}  {agg_mse_p:>12.4f}  "
      f"{dpct(REF_MSE_P, agg_mse_p):>+13.2f}%")
print(f"{'MAE':<8} {REF_MAE_B:>12.4f}  {REF_MAE_P:>12.4f}  {agg_mae_p:>12.4f}  "
      f"{dpct(REF_MAE_P, agg_mae_p):>+13.2f}%")
print(f"{'MASE':<8} {REF_MASE_B:>12.4f}  {REF_MASE_P:>12.4f}  {agg_mase_p:>12.4f}  "
      f"{dpct(REF_MASE_P, agg_mase_p):>+13.2f}%")
print(f"\nPer-series improvement (MAE):  {REF_FRAC_MAE:.1f}%  →  "
      f"{improved_mae/n_val*100:.1f}%")
print(f"  MSE improved: {improved_mse}/{n_val} ({improved_mse/n_val*100:.1f}%)")
print(f"  MAE improved: {improved_mae}/{n_val} ({improved_mae/n_val*100:.1f}%)")
print(f"{'='*80}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP B — Outlier diagnosis
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("STEP B — OUTLIER DIAGNOSIS")
print(f"{'='*65}")

mse_pfx_arr = np.array(mse_pfx)
mae_pfx_arr = np.array(mae_pfx)
mae_bas_arr = np.array(mae_bas)

threshold   = np.percentile(mse_pfx_arr, 95)
outlier_pos = np.where(mse_pfx_arr > threshold)[0]
n_out       = len(outlier_pos)
print(f"\n  95th-pct MSE threshold: {threshold:.4f}")
print(f"  Outlier count:          {n_out}  (~5% of {n_val})\n")


# ── Step B Part 1: Compute Layer-5 prefix attention for ALL series ────────────
# Uses encoder-only pass (output_attentions=True) — avoids full pipeline.predict()
# This is a proxy for attention during actual inference (decoder does not affect
# encoder attention weights).

print("Computing Layer-5 prefix attention for all series (encoder pass) …")
l5_all = np.zeros(n_val, dtype=np.float32)

for b in range(n_batches):
    b_start = b * BATCH_SIZE
    idxs    = val_idx[b_start : b_start + BATCH_SIZE]
    bs      = len(idxs)

    ctxs_b, trens_b, seas_b, ress_b = [], [], [], []
    for j, idx in enumerate(idxs):
        ctx_arr = np.array(table.column("context")[int(idx)].as_py(), dtype=np.float32)
        pds_j   = periods_all[b_start + j]   # reuse periods from Step A
        tr_j, se_j, re_j = stl_decompose(ctx_arr, pds_j)
        ctxs_b.append(ctx_arr)
        trens_b.append(tr_j); seas_b.append(se_j); ress_b.append(re_j)

    # Tokenize on CPU (bucketize boundary constraint)
    ctx_t_b = torch.tensor(np.stack(ctxs_b), dtype=torch.float32)
    inp_ids, att_msk, _ = pipeline.tokenizer.context_input_transform(
        pipeline._prepare_and_validate_context(ctx_t_b)
    )
    inp_ids = inp_ids.to(DEVICE)
    att_msk = att_msk.to(DEVICE)

    tr_tb = torch.tensor(np.stack(trens_b), dtype=torch.float32).to(DEVICE)
    se_tb = torch.tensor(np.stack(seas_b),  dtype=torch.float32).to(DEVICE)
    re_tb = torch.tensor(np.stack(ress_b),  dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        try:
            inject_prefix(chronos_model, pg, trend=tr_tb, seasonal=se_tb, residual=re_tb)
            enc = chronos_model.model.encoder(
                input_ids=inp_ids, attention_mask=att_msk, output_attentions=True,
            )
            # enc.attentions[5]: (bs, n_heads=8, q_len, PREFIX_TOTAL + seq_len)
            # Softmax weights sum to 1 over last dim for each (head, query pos).
            # pfx_w × PREFIX_TOTAL = total fraction of attention mass on prefix.
            l5_w = enc.attentions[5]
            for s_i in range(bs):
                pfx_w = l5_w[s_i, :, :, :PREFIX_TOTAL].mean().item()
                l5_all[b_start + s_i] = pfx_w * PREFIX_TOTAL
        finally:
            remove_prefix_hooks(chronos_model)

    if (b + 1) % 20 == 0 or (b + 1) == n_batches:
        done = min((b + 1) * BATCH_SIZE, n_val)
        print(f"  [{done:>4}/{n_val}]")

# Extract outlier-subset values
outlier_l5        = l5_all[outlier_pos]
outlier_periods   = [periods_all[i]   for i in outlier_pos]
outlier_resid_sig = [resid_sig_all[i] for i in outlier_pos]
outlier_mae_delta = (mae_pfx_arr - mae_bas_arr)[outlier_pos]


# ── Step B (a): Period distribution ───────────────────────────────────────────

print(f"\n{'─'*65}")
print("(a) PERIOD DISTRIBUTION")

flat_out = [p for ps in outlier_periods for p in ps]
flat_all = [p for ps in periods_all    for p in ps]
cnt_out  = Counter(flat_out)
cnt_all  = Counter(flat_all)

print(f"\n  {'Period':>8}  {'#out':>6}  {'#all':>8}  "
      f"{'f_out':>7}  {'f_all':>7}  {'ratio':>6}")
print(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}")

flagged = []
for period, co in sorted(cnt_out.items(), key=lambda x: -x[1])[:15]:
    ca   = cnt_all.get(period, 0)
    fo   = co / max(len(flat_out), 1)
    fa   = ca / max(len(flat_all), 1)
    r    = fo / fa if fa > 0 else float("inf")
    flag = " ←" if r > 1.5 else ""
    print(f"  {period:>8}  {co:>6}  {ca:>8}  {fo:>7.3f}  {fa:>7.3f}  {r:>6.2f}{flag}")
    if r > 1.5:
        flagged.append((period, r))

if flagged:
    print(f"\n  Over-represented (>1.5×): " +
          ", ".join(f"period={p} ({r:.2f}×)" for p, r in flagged))
else:
    print(f"\n  No period is >1.5× over-represented in the outlier set.")


# ── Step B (b): Residual ratio ─────────────────────────────────────────────────

print(f"\n{'─'*65}")
print("(b) RESIDUAL RATIO  (resid_sig = var(resid) / var(ctx))")

out_rs = np.array(outlier_resid_sig)
all_rs = np.array(resid_sig_all)

print(f"\n  Outliers ({n_out}):  mean={np.mean(out_rs):.4f}  std={np.std(out_rs):.4f}  "
      f"frac>0.5={np.mean(out_rs > 0.5):.3f}")
print(f"  All {n_val}:    mean={np.mean(all_rs):.4f}  std={np.std(all_rs):.4f}  "
      f"frac>0.5={np.mean(all_rs > 0.5):.3f}")
rs_ratio = np.mean(out_rs) / np.mean(all_rs) if np.mean(all_rs) > 0 else float("nan")
print(f"  Ratio outlier/full:  {rs_ratio:.2f}×")


# ── Step B (c): Layer-5 prefix attention ──────────────────────────────────────

print(f"\n{'─'*65}")
print("(c) LAYER-5 PREFIX ATTENTION  (fraction of attention mass on prefix)")

mean_out_l5 = float(np.mean(outlier_l5))
std_out_l5  = float(np.std(outlier_l5))
mean_all_l5 = float(np.mean(l5_all))
std_all_l5  = float(np.std(l5_all))

print(f"\n  Outliers ({n_out}):  mean={mean_out_l5:.4f}  std={std_out_l5:.4f}")
print(f"  All {n_val}:    mean={mean_all_l5:.4f}  std={std_all_l5:.4f}")

if mean_out_l5 > mean_all_l5 + 0.01:
    direction_l5 = "HIGHER"
elif mean_out_l5 < mean_all_l5 - 0.01:
    direction_l5 = "LOWER"
else:
    direction_l5 = "SIMILAR"
print(f"  Outliers show {direction_l5} L5 prefix attention than average.")


# ── Step B (d): Top 10 worst series ───────────────────────────────────────────

print(f"\n{'─'*65}")
print("(d) TOP 10 WORST SERIES BY MSE (with prefix)")
print(f"\n  {'Val-Idx':>8}  {'Period(s)':>16}  {'resid_sig':>10}  "
      f"{'L5_pfx_attn':>12}  {'mae_delta':>10}  {'MSE_pfx':>10}")
print(f"  {'─'*8}  {'─'*16}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*10}")

top10_local = np.argsort(mse_pfx_arr[outlier_pos])[::-1][:10]
for oi in top10_local:
    vi  = outlier_pos[oi]
    pds = str(outlier_periods[oi])
    rs  = outlier_resid_sig[oi]
    l5a = outlier_l5[oi]
    md  = outlier_mae_delta[oi]
    mv  = mse_pfx_arr[vi]
    print(f"  {vi:>8}  {pds:>16}  {rs:>10.4f}  {l5a:>12.4f}  {md:>+10.4f}  {mv:>10.2f}")


# ── Step B: Summary verdict ────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("STEP B — SUMMARY VERDICT")
print(f"{'='*65}")

print(f"\nQ1: Do outliers cluster around specific period values?")
if flagged:
    print(f"    YES — over-represented: " +
          ", ".join(f"period={p} ({r:.2f}×)" for p, r in flagged))
else:
    print(f"    NO — no period >1.5× over-represented in outlier set")

print(f"\nQ2: Do outliers have significantly higher resid_sig than the full set?")
if rs_ratio > 1.25:
    print(f"    YES — outlier mean={np.mean(out_rs):.4f}  vs  "
          f"full mean={np.mean(all_rs):.4f}  ({rs_ratio:.2f}×)")
else:
    print(f"    NO  — outlier mean={np.mean(out_rs):.4f}  vs  "
          f"full mean={np.mean(all_rs):.4f}  ({rs_ratio:.2f}×, not significantly higher)")

print(f"\nQ3: Do outliers show unusually high or low L5 prefix attention?")
print(f"    {direction_l5} — outlier mean={mean_out_l5:.4f}  vs  "
      f"full mean={mean_all_l5:.4f}")

print(f"\n{'='*65}")
print("DONE — Step A and Step B complete.")
print(f"{'='*65}")
