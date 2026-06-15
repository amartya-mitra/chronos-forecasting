#!/usr/bin/env python3
"""
eval_joint_pilot.py — Evaluate joint checkpoint on M4 Hourly and M4 Monthly val sets.

Loads:
  checkpoints/pilot-joint-m4h-m4m-3000.pt  (joint prefix-tuning checkpoint)
  data/m4hourly-tsf.arrow                   (CONTEXT=96, TARGET_OFFSET=96, PRED_LEN=48)
  data/m4monthly-tsf.arrow                  (CONTEXT=36, TARGET_OFFSET=36, PRED_LEN=18)

Produces:
  results/pilot_joint_m4h_m4m_eval.json     (full metrics, D6 trajectory)
  results/pilot_joint_m4h_m4m_summary.txt   (comparison table, per-dataset delta)
  results/joint_training_pilot.md           (pilot verdict: PASS / MARGINAL / FAIL)

Solo and baseline MASE values are hardcoded from previous single-dataset sweeps.
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT       = Path(__file__).parent.parent
FINETUNING_ROOT = Path(__file__).parent
RESULTS_DIR     = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(FINETUNING_ROOT))

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F

from chronos import ChronosPipeline
from prefix_generator import PrefixGenerator
from prefix_injection import inject_prefix, remove_prefix_hooks
from scripts.step1_stl_decompose import detect_periods, stl_decompose


# ── Hardcoded references from single-dataset sweep ─────────────────────────────

MASE_BASE_HOURLY   = 1.6565
MASE_SOLO_HOURLY   = 1.2181

MASE_BASE_MONTHLY  = 1.2105
MASE_SOLO_MONTHLY  = 0.7274

# ── Dataset configs ────────────────────────────────────────────────────────────

HOURLY_EVAL_CFG = {
    'data_path':     'data/m4hourly-tsf.arrow',
    'context_col':   'context',
    'forecast_col':  'target',
    'target_offset': 96,
    'context_len':   96,
    'pred_len':      48,
    'mase_period':   24,
    'min_period':    6,
    'max_period':    24,
    'label':         'M4 Hourly',
}
MONTHLY_EVAL_CFG = {
    'data_path':     'data/m4monthly-tsf.arrow',
    'context_col':   'context',
    'forecast_col':  'target',
    'target_offset': 36,
    'context_len':   36,
    'pred_len':      18,
    'mase_period':   12,
    'min_period':    4,
    'max_period':    12,
    'label':         'M4 Monthly',
}

# ── Arch constants (must match joint_train.py) ─────────────────────────────────

D_MODEL     = 512
NUM_LAYERS  = 6
PREFIX_LEN  = 16
RANK        = 64
TRAIN_SEED  = 42
RS_THRESH   = 0.4
CEIL_STEP   = 3000    # training ended at step 3000
CEIL_TOTAL  = 3000
ATTN_THRESH = 0.05

# PASS/MARGINAL verdict thresholds
PASS_DELTA_MASE  = 0.03   # MASE_joint ≤ MASE_solo + 0.03 → PASS
D6_PASS_THRESH   = 0.90   # D6 < 0.90 → differentiated (required for PASS)
D6_FAIL_THRESH   = 0.97   # D6 > 0.97 → collapsed (triggers FAIL)

CKPT_PATH = FINETUNING_ROOT / "checkpoints" / "pilot-joint-m4h-m4m-3000.pt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Load Chronos ───────────────────────────────────────────────────────────────

print("Loading Chronos T5-Small …")
pipeline      = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small", device_map=str(device), dtype=torch.float32,
)
chronos_model = pipeline.model
tokenizer     = pipeline.tokenizer
for p in chronos_model.parameters():
    p.requires_grad_(False)

# ── Load joint checkpoint ─────────────────────────────────────────────────────

print(f"Loading joint checkpoint: {CKPT_PATH.name} …")
ckpt = torch.load(str(CKPT_PATH), map_location=device)
pg_cfg = ckpt.get('config', {})
print(f"  Trained for {ckpt['step']} steps  |  final loss: {ckpt['loss']:.4f}")

prefix_gen = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(device)
prefix_gen.load_state_dict(ckpt['pg_state'])
prefix_gen.eval()

# Extract diagnostics saved during training
diagnostics  = ckpt.get('diagnostics', {})
d6_traj      = diagnostics.get('d6_traj', [])
ema_fc_traj  = diagnostics.get('ema_fc_traj', {'hourly': [], 'monthly': []})
print(f"  D6 trajectory from training: {d6_traj}")


# ── Ceiling formula (matches train_ddp pattern) ────────────────────────────────

def _ceil_cap(step_pct):
    return min(0.5 + 0.3 * step_pct, 0.8)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _naive_mae(ctx, period):
    if len(ctx) <= period:
        return float("nan")
    return float(np.mean(np.abs(ctx[period:] - ctx[:-period])))


def _mase(mae, ctx, period):
    denom = _naive_mae(ctx, period)
    if np.isnan(denom) or denom < 1e-9:
        return float("nan")
    return mae / denom


def _load_val(cfg):
    path = FINETUNING_ROOT / cfg['data_path']
    with open(str(path), "rb") as f:
        table = ipc.open_file(f).read_all()
    n = len(table)
    rng = np.random.default_rng(TRAIN_SEED)
    idx = rng.permutation(n)
    val_idx = idx[int(n * 0.8):]
    print(f"  {path.name}: {n} total → {len(val_idx)} val series")
    return table, val_idx


# ── Per-dataset evaluation ─────────────────────────────────────────────────────

def evaluate_dataset(cfg, table, val_idx, prefix_gen_or_none, tag="baseline"):
    label       = cfg['label']
    context_len = cfg['context_len']
    pred_len    = cfg['pred_len']
    offset      = cfg['target_offset']
    mase_period = cfg['mase_period']
    min_p       = cfg['min_period']
    max_p       = cfg.get('max_period')
    ctx_col     = cfg['context_col']
    fct_col     = cfg['forecast_col']

    use_prefix = prefix_gen_or_none is not None

    mse_list      = []
    mae_list      = []
    mase_list     = []
    ceil_fires    = []
    pre_ceil_list = []
    period_match  = []

    for i in val_idx:
        i = int(i)
        ctx = np.array(table.column(ctx_col)[i].as_py(), dtype=np.float32)
        raw = np.array(table.column(fct_col)[i].as_py(), dtype=np.float32)
        ftr = raw[offset:] if offset > 0 else raw

        periods = detect_periods(ctx, min_period=min_p, max_period=max_p)
        period_match.append(1 if mase_period in periods else 0)
        tr, se, re = stl_decompose(ctx, periods)
        rs         = float(np.var(re) / (np.var(ctx) + 1e-9))
        if rs > RS_THRESH:
            tr = np.zeros_like(ctx, dtype=np.float32)
            se = np.zeros_like(ctx, dtype=np.float32)
            re = np.zeros_like(ctx, dtype=np.float32)

        ctx_t    = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        input_ids, attn_mask, ctx_scale = tokenizer.context_input_transform(
            pipeline._prepare_and_validate_context(ctx_t)
        )
        input_ids  = input_ids.to(device)
        attn_mask  = attn_mask.to(device)
        ctx_scale_d = ctx_scale.to(device).unsqueeze(-1)

        with torch.no_grad():
            if use_prefix:
                tr_t = torch.tensor(tr, dtype=torch.float32).unsqueeze(0).to(device) / ctx_scale_d
                se_t = torch.tensor(se, dtype=torch.float32).unsqueeze(0).to(device) / ctx_scale_d
                re_t = torch.tensor(re, dtype=torch.float32).unsqueeze(0).to(device) / ctx_scale_d

                training_state = {
                    'current_step':                   CEIL_STEP,
                    'total_steps':                    CEIL_TOTAL,
                    'ceiling_fired_accumulator':      ceil_fires,
                    'pfx_total_pre_ceil_accumulator': pre_ceil_list,
                }
                prefix_kvs = prefix_gen_or_none(tr_t, se_t, re_t)
                inject_prefix(chronos_model, prefix_generator=prefix_gen_or_none,
                              prefix_kvs=prefix_kvs, training_state=training_state)
                try:
                    samples = pipeline.predict(
                        ctx_t, prediction_length=pred_len,
                        num_samples=20, limit_prediction_length=False,
                    )
                finally:
                    remove_prefix_hooks(chronos_model)
            else:
                samples = pipeline.predict(
                    ctx_t, prediction_length=pred_len,
                    num_samples=20, limit_prediction_length=False,
                )

        pred = samples.median(dim=1).values.squeeze(0).cpu().numpy()
        ftr_ = ftr[:pred_len]
        mse_ = float(np.mean((pred - ftr_) ** 2))
        mae_ = float(np.mean(np.abs(pred - ftr_)))
        mse_list.append(mse_)
        mae_list.append(mae_)
        mase_list.append(_mase(mae_, ctx, mase_period))

    mase_clean = [v for v in mase_list if not np.isnan(v)]
    pm_pct     = float(np.mean(period_match)) * 100 if period_match else float("nan")
    cf_mean    = float(np.mean(ceil_fires))   if ceil_fires    else float("nan")
    pc_mean    = float(np.mean(pre_ceil_list)) if pre_ceil_list else float("nan")

    return {
        'label':          label,
        'tag':            tag,
        'n_val':          len(val_idx),
        'MSE':            float(np.mean(mse_list)),
        'MAE':            float(np.mean(mae_list)),
        'MASE':           float(np.mean(mase_clean)),
        'period_match_pct': pm_pct,
        'ceil_fire_mean': cf_mean,
        'pre_ceil_mean':  pc_mean,
    }


# ── Run evaluations ────────────────────────────────────────────────────────────

print("\nLoading datasets …")
hourly_table,  hourly_val_idx  = _load_val(HOURLY_EVAL_CFG)
monthly_table, monthly_val_idx = _load_val(MONTHLY_EVAL_CFG)

for cfg, table, val_idx, label in [
    (HOURLY_EVAL_CFG,  hourly_table,  hourly_val_idx,  "M4 Hourly"),
    (MONTHLY_EVAL_CFG, monthly_table, monthly_val_idx, "M4 Monthly"),
]:
    t0 = time.time()
    print(f"\n--- {label} ---")
    print(f"  Baseline (zero-shot) …")
    res_base = evaluate_dataset(cfg, table, val_idx, None, tag="baseline")
    print(f"    MASE={res_base['MASE']:.4f}  MSE={res_base['MSE']:.4f}  MAE={res_base['MAE']:.4f}")

    print(f"  Joint prefix pass …")
    res_joint = evaluate_dataset(cfg, table, val_idx, prefix_gen, tag="joint")
    print(f"    MASE={res_joint['MASE']:.4f}  MSE={res_joint['MSE']:.4f}  MAE={res_joint['MAE']:.4f}")
    print(f"    ceil_fire={res_joint['ceil_fire_mean']:.4f}  pre_ceil_mean={res_joint['pre_ceil_mean']:.4f}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    if label == "M4 Hourly":
        r_base_h, r_joint_h = res_base, res_joint
    else:
        r_base_m, r_joint_m = res_base, res_joint


# ── Compute deltas and build comparison table ──────────────────────────────────

def _delta_pct(new, old):
    if old == 0 or np.isnan(old):
        return float("nan")
    return (new - old) / abs(old) * 100

mase_joint_h = r_joint_h['MASE']
mase_joint_m = r_joint_m['MASE']

delta_h_base = _delta_pct(mase_joint_h, MASE_BASE_HOURLY)
delta_m_base = _delta_pct(mase_joint_m, MASE_BASE_MONTHLY)
delta_h_solo = mase_joint_h - MASE_SOLO_HOURLY
delta_m_solo = mase_joint_m - MASE_SOLO_MONTHLY

# Improv%: fraction of val samples where prefix beats baseline
# Approximate: ΔMASE as fraction of base-solo gap retained
retain_h = 1 - (mase_joint_h - MASE_SOLO_HOURLY) / max(MASE_BASE_HOURLY - MASE_SOLO_HOURLY, 1e-9)
retain_m = 1 - (mase_joint_m - MASE_SOLO_MONTHLY) / max(MASE_BASE_MONTHLY - MASE_SOLO_MONTHLY, 1e-9)

d6_final = d6_traj[-1][1] if d6_traj else None
d6_init  = d6_traj[0][1]  if d6_traj else None


# ── Pilot verdict ───────────────────────────────────────────────────────────────

h_within_tol = delta_h_solo <= PASS_DELTA_MASE
m_within_tol = delta_m_solo <= PASS_DELTA_MASE
both_ok      = h_within_tol and m_within_tol
d6_ok        = (d6_final is None) or (d6_final < D6_PASS_THRESH)
d6_collapsed = (d6_final is not None) and (d6_final > D6_FAIL_THRESH)

h_regression = (mase_joint_h > MASE_BASE_HOURLY)
m_regression = (mase_joint_m > MASE_BASE_MONTHLY)

if (h_regression or m_regression) or d6_collapsed:
    verdict     = "FAIL"
    if h_regression or m_regression:
        reason = ("MASE_joint exceeds baseline (regression beyond zero-shot)"
                  + (" [hourly]" if h_regression else "")
                  + (" [monthly]" if m_regression else ""))
    else:
        reason = f"D6={d6_final:.4f} > {D6_FAIL_THRESH} — prefix collapsed to a single representation"
elif both_ok and d6_ok:
    verdict = "PASS"
    reason  = (f"MASE_joint within {PASS_DELTA_MASE:.2f} of solo on both datasets"
               + (f" and D6={d6_final:.4f} < {D6_PASS_THRESH} (differentiated)" if d6_final else ""))
elif both_ok and not d6_ok:
    verdict = "MARGINAL"
    reason  = (f"MASE within tolerance on both datasets but D6={d6_final:.4f} in ({D6_PASS_THRESH:.2f}, {D6_FAIL_THRESH:.2f})"
               " — prefix partially collapsed; may limit multi-dataset scaling")
elif (h_within_tol or m_within_tol) and not d6_collapsed:
    verdict = "MARGINAL"
    reason  = ("Within tolerance on "
               + ("hourly only" if h_within_tol else "monthly only")
               + f"; {'' if d6_ok else f'D6={d6_final:.4f} marginal; '}"
               + "interference on the weaker dataset needs investigation")
else:
    verdict = "FAIL"
    reason  = f"Neither dataset within ±{PASS_DELTA_MASE:.2f} tolerance of solo MASE"


# ── Format comparison table ────────────────────────────────────────────────────

comparison_lines = [
    "=" * 75,
    "PILOT: Joint Training  M4 Hourly + M4 Monthly  — COMPARISON TABLE",
    "=" * 75,
    "",
    f"{'Dataset':<14}  {'MASE_base':>10}  {'MASE_solo':>10}  {'MASE_joint':>11}  {'Δ(solo→joint)':>14}  {'Retain%':>8}",
    "-" * 75,
    f"{'M4 Hourly':<14}  {MASE_BASE_HOURLY:>10.4f}  {MASE_SOLO_HOURLY:>10.4f}  "
    f"{mase_joint_h:>11.4f}  {delta_h_solo:>+14.4f}  {retain_h*100:>7.1f}%",
    f"{'M4 Monthly':<14}  {MASE_BASE_MONTHLY:>10.4f}  {MASE_SOLO_MONTHLY:>10.4f}  "
    f"{mase_joint_m:>11.4f}  {delta_m_solo:>+14.4f}  {retain_m*100:>7.1f}%",
    "-" * 75,
    "",
    f"  M4 Hourly  ΔMASE(joint vs base): {delta_h_base:+.2f}%",
    f"  M4 Monthly ΔMASE(joint vs base): {delta_m_base:+.2f}%",
    "",
    "D6 TRAJECTORY (cosine similarity: hourly vs monthly prefix KVs):",
]
for s, v in d6_traj:
    d6_lbl = ('differentiated ✓' if v < 0.90
              else '⚠ partial collapse' if v < 0.97
              else '⚠⚠ COLLAPSED')
    comparison_lines.append(f"  step {s:>4}: {v:.4f}  {d6_lbl}")

if not d6_traj:
    comparison_lines.append("  (no D6 data in checkpoint)")

comparison_lines += [
    "",
    f"EMA fc_loss trajectory (hourly): "
    + str([(s, round(v, 4) if v is not None else None) for s, v in ema_fc_traj.get('hourly', [])]),
    f"EMA fc_loss trajectory (monthly): "
    + str([(s, round(v, 4) if v is not None else None) for s, v in ema_fc_traj.get('monthly', [])]),
    "",
    "=" * 75,
    f"PILOT VERDICT:  {verdict}",
    "=" * 75,
    f"Reason: {reason}",
    "",
    "PASS criteria:",
    f"  • MASE_joint ≤ MASE_solo + {PASS_DELTA_MASE:.2f} on BOTH datasets",
    f"  • D6_final < {D6_PASS_THRESH:.2f}  (differentiated prefix representations)",
    "",
    f"Hourly  within tol:   {h_within_tol}  (delta={delta_h_solo:+.4f}, tol={PASS_DELTA_MASE:.2f})",
    f"Monthly within tol:   {m_within_tol}  (delta={delta_m_solo:+.4f}, tol={PASS_DELTA_MASE:.2f})",
    f"D6 condition:         {d6_ok}  (D6={d6_final:.4f} < {D6_PASS_THRESH:.2f})"
    if d6_final else "  D6 condition: (no D6 data)",
    "=" * 75,
]

table_str = "\n".join(comparison_lines)
print("\n" + table_str)


# ── Save outputs ───────────────────────────────────────────────────────────────

json_path = RESULTS_DIR / "pilot_joint_m4h_m4m_eval.json"
txt_path  = RESULTS_DIR / "pilot_joint_m4h_m4m_summary.txt"
md_path   = RESULTS_DIR / "joint_training_pilot.md"

eval_blob = {
    'checkpoint': str(CKPT_PATH.name),
    'ckpt_step':  ckpt['step'],
    'hourly': {
        'MASE_base':  MASE_BASE_HOURLY,
        'MASE_solo':  MASE_SOLO_HOURLY,
        'MASE_joint': mase_joint_h,
        'MSE_joint':  r_joint_h['MSE'],
        'MAE_joint':  r_joint_h['MAE'],
        'delta_vs_base_pct': delta_h_base,
        'delta_vs_solo_abs': delta_h_solo,
        'retain_pct':        retain_h * 100,
        'ceil_fire_mean':    r_joint_h['ceil_fire_mean'],
        'pre_ceil_mean':     r_joint_h['pre_ceil_mean'],
        'period_match_pct':  r_joint_h['period_match_pct'],
    },
    'monthly': {
        'MASE_base':  MASE_BASE_MONTHLY,
        'MASE_solo':  MASE_SOLO_MONTHLY,
        'MASE_joint': mase_joint_m,
        'MSE_joint':  r_joint_m['MSE'],
        'MAE_joint':  r_joint_m['MAE'],
        'delta_vs_base_pct': delta_m_base,
        'delta_vs_solo_abs': delta_m_solo,
        'retain_pct':        retain_m * 100,
        'ceil_fire_mean':    r_joint_m['ceil_fire_mean'],
        'pre_ceil_mean':     r_joint_m['pre_ceil_mean'],
        'period_match_pct':  r_joint_m['period_match_pct'],
    },
    'd6_traj':       d6_traj,
    'ema_fc_traj':   ema_fc_traj,
    'verdict':       verdict,
    'verdict_reason': reason,
    'thresholds': {
        'pass_delta_mase': PASS_DELTA_MASE,
        'd6_pass_thresh':  D6_PASS_THRESH,
        'd6_fail_thresh':  D6_FAIL_THRESH,
    },
}

with open(str(json_path), "w") as f:
    json.dump(eval_blob, f, indent=2)
print(f"\nSaved → {json_path}")

with open(str(txt_path), "w") as f:
    f.write(table_str + "\n")
print(f"Saved → {txt_path}")

md_content = f"""# Joint Training Pilot — M4 Hourly + M4 Monthly

**Checkpoint:** `{CKPT_PATH.name}` (step {ckpt['step']})
**Date:** 2026-06-12

## Verdict: {verdict}

{reason}

## Comparison Table

| Dataset     | MASE_base | MASE_solo | MASE_joint | Δ(solo→joint) | Retain% |
|-------------|-----------|-----------|------------|---------------|---------|
| M4 Hourly   | {MASE_BASE_HOURLY:.4f}    | {MASE_SOLO_HOURLY:.4f}    | {mase_joint_h:.4f}     | {delta_h_solo:+.4f}       | {retain_h*100:.1f}%    |
| M4 Monthly  | {MASE_BASE_MONTHLY:.4f}    | {MASE_SOLO_MONTHLY:.4f}    | {mase_joint_m:.4f}     | {delta_m_solo:+.4f}       | {retain_m*100:.1f}%    |

## D6 Trajectory (prefix differentiation)

| Step | Cosine Similarity | Assessment |
|------|-------------------|------------|
""" + "\n".join(
    f"| {s:>4} | {v:.4f} | "
    + ("differentiated ✓" if v < 0.90 else "⚠ partial collapse" if v < 0.97 else "⚠⚠ COLLAPSED")
    + " |"
    for s, v in d6_traj
) + (
    "\n| (no D6 data) | — | — |" if not d6_traj else ""
) + f"""

## EMA fc_loss Trajectories

**M4 Hourly:**  {[(s, round(v, 4) if v is not None else None) for s, v in ema_fc_traj.get('hourly',  [])]}

**M4 Monthly:** {[(s, round(v, 4) if v is not None else None) for s, v in ema_fc_traj.get('monthly', [])]}

## PASS Criteria Applied

- MASE_joint ≤ MASE_solo + {PASS_DELTA_MASE:.2f} on **both** datasets: `{"✓" if both_ok else "✗"}`
- D6_final < {D6_PASS_THRESH:.2f} (prefix differentiation): `{"✓" if d6_ok else "✗"}`
- No regression beyond zero-shot baseline: `{"✓" if not (h_regression or m_regression) else "✗"}`

## Next Steps

{"**Proceed to Stage 2:** Full 6-dataset joint run with Electricity held out as validation." if verdict == "PASS" else "**Do not proceed to Stage 2.** Review results above and address failure mode before scaling to full joint run." if verdict == "FAIL" else "**Review before Stage 2.** Address the marginal condition noted above; consider reducing TOTAL_STEPS or adjusting per-dataset weighting."}
"""

with open(str(md_path), "w") as f:
    f.write(md_content)
print(f"Saved → {md_path}")

print(f"\nPILOT VERDICT:  {verdict}")
print(f"See {md_path.name} for full report.")
