#!/usr/bin/env python3
"""
eval_joint_stage2.py — Evaluate Stage 2 joint checkpoint.

PART A — In-distribution: 5 training datasets (val split, 80/20, seed=42)
  Compare against MASE_base (zero-shot), MASE_solo (Phase 2), and Stage 1 pilot
  where available.

PART B — Out-of-distribution: ALL Electricity series (domain-shift test)
  The Stage 2 model was never trained on Electricity.
  Compare against zero-shot Chronos and solo Electricity checkpoint.

Outputs:
  results/joint_stage2_eval.json
  results/joint_stage2_summary.txt
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

from chronos import ChronosPipeline
from prefix_generator import PrefixGenerator
from prefix_injection import inject_prefix, remove_prefix_hooks
from scripts.step1_stl_decompose import detect_periods, stl_decompose


# ── Arch constants ─────────────────────────────────────────────────────────────

D_MODEL     = 512
NUM_LAYERS  = 6
PREFIX_LEN  = 16
RANK        = 64
TRAIN_SEED  = 42
RS_THRESH   = 0.4
ATTN_THRESH = 0.05

CKPT_PATH   = FINETUNING_ROOT / "checkpoints" / "joint-stage2-5ds-5000.pt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CEIL_STEP   = 5000
CEIL_TOTAL  = 5000


# ── Phase 2 and Stage 1 reference values ──────────────────────────────────────

REF = {
    'sarsim0':   {'label': 'SarSim0',     'MASE_base': 0.6545, 'MASE_solo': 0.6511, 'MASE_stage1': None},
    'm4hourly':  {'label': 'M4 Hourly',   'MASE_base': 1.6565, 'MASE_solo': 1.2181, 'MASE_stage1': 1.1585},
    'm4monthly': {'label': 'M4 Monthly',  'MASE_base': 1.2105, 'MASE_solo': 0.7274, 'MASE_stage1': 0.7453},
    'm4daily':   {'label': 'M4 Daily',    'MASE_base': 1.2123, 'MASE_solo': 1.0304, 'MASE_stage1': None},
    'm4weekly':  {'label': 'M4 Weekly',   'MASE_base': 0.6308, 'MASE_solo': 0.5629, 'MASE_stage1': None},
}

ELEC_REF = {
    'MASE_base': 0.9002,
    'MASE_solo': 0.8004,
    'MSE_base':  3182265.0,
    'MSE_solo':  1263494.0,
    'MAE_base':  326.18,
    'MAE_solo':  235.21,
}

# ── Part A: per-dataset eval configs ──────────────────────────────────────────

EVAL_CONFIGS = {
    'sarsim0': {
        'data_path':     'data/sarsim0-10k.arrow',
        'context_col':   'context',
        'forecast_col':  'forecast',
        'target_offset': 0,
        'pred_len':      64,
        'mase_period':   24,
        'min_period':    6,
        'max_period':    None,
    },
    'm4hourly': {
        'data_path':     'data/m4hourly-tsf.arrow',
        'context_col':   'context',
        'forecast_col':  'target',
        'target_offset': 96,
        'pred_len':      48,
        'mase_period':   24,
        'min_period':    6,
        'max_period':    24,
    },
    'm4monthly': {
        'data_path':     'data/m4monthly-tsf.arrow',
        'context_col':   'context',
        'forecast_col':  'target',
        'target_offset': 36,
        'pred_len':      18,
        'mase_period':   12,
        'min_period':    4,
        'max_period':    12,
    },
    'm4daily': {
        'data_path':     'data/m4daily-tsf.arrow',
        'context_col':   'context',
        'forecast_col':  'target',
        'target_offset': 0,
        'pred_len':      14,
        'mase_period':   7,
        'min_period':    4,
        'max_period':    31,
    },
    'm4weekly': {
        'data_path':     'data/m4weekly-tsf.arrow',
        'context_col':   'context',
        'forecast_col':  'target',
        'target_offset': 0,
        'pred_len':      13,
        'mase_period':   52,
        'min_period':    4,
        'max_period':    80,
    },
}

ELEC_CFG = {
    'data_path':     'data/electricity-tsf.arrow',
    'context_col':   'context',
    'forecast_col':  'target',
    'target_offset': 0,
    'pred_len':      24,
    'mase_period':   24,
    'min_period':    6,
    'max_period':    84,
}

DATASET_ORDER = ['sarsim0', 'm4hourly', 'm4monthly', 'm4daily', 'm4weekly']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Load Chronos + joint checkpoint ───────────────────────────────────────────

print("Loading Chronos T5-Small …")
pipeline      = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small", device_map=str(device), dtype=torch.float32,
)
chronos_model = pipeline.model
tokenizer     = pipeline.tokenizer
for p in chronos_model.parameters():
    p.requires_grad_(False)

print(f"Loading joint checkpoint: {CKPT_PATH.name} …")
ckpt = torch.load(str(CKPT_PATH), map_location=device)
print(f"  Trained for {ckpt['step']} steps  |  datasets: {ckpt['config'].get('datasets', '?')}")

prefix_gen = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(device)
prefix_gen.load_state_dict(ckpt['pg_state'])
prefix_gen.eval()


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


def evaluate_dataset(cfg, val_idx_arr, table, tag="joint"):
    pred_len    = cfg['pred_len']
    offset      = cfg['target_offset']
    mase_period = cfg['mase_period']
    min_p       = cfg['min_period']
    max_p       = cfg.get('max_period')
    ctx_col     = cfg['context_col']
    fct_col     = cfg['forecast_col']

    mse_list   = []
    mae_list   = []
    mase_list  = []
    ceil_fires = []
    pre_ceil   = []

    for i in val_idx_arr:
        i = int(i)
        ctx = np.array(table.column(ctx_col)[i].as_py(), dtype=np.float32)
        raw = np.array(table.column(fct_col)[i].as_py(), dtype=np.float32)
        ftr = raw[offset:] if offset > 0 else raw

        periods = detect_periods(ctx, min_period=min_p, max_period=max_p)
        tr, se, re = stl_decompose(ctx, periods)
        rs = float(np.var(re) / (np.var(ctx) + 1e-9))
        if rs > RS_THRESH:
            tr = np.zeros_like(ctx, dtype=np.float32)
            se = np.zeros_like(ctx, dtype=np.float32)
            re = np.zeros_like(ctx, dtype=np.float32)

        ctx_t    = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        input_ids, attn_mask, ctx_scale = tokenizer.context_input_transform(
            pipeline._prepare_and_validate_context(ctx_t)
        )
        ctx_scale_d = ctx_scale.to(device).unsqueeze(-1)

        with torch.no_grad():
            tr_t = torch.tensor(tr, dtype=torch.float32).unsqueeze(0).to(device) / ctx_scale_d
            se_t = torch.tensor(se, dtype=torch.float32).unsqueeze(0).to(device) / ctx_scale_d
            re_t = torch.tensor(re, dtype=torch.float32).unsqueeze(0).to(device) / ctx_scale_d

            training_state = {
                'current_step':                   CEIL_STEP,
                'total_steps':                    CEIL_TOTAL,
                'ceiling_fired_accumulator':      ceil_fires,
                'pfx_total_pre_ceil_accumulator': pre_ceil,
            }
            prefix_kvs = prefix_gen(tr_t, se_t, re_t)
            inject_prefix(chronos_model, prefix_generator=prefix_gen,
                          prefix_kvs=prefix_kvs, training_state=training_state)
            try:
                samples = pipeline.predict(
                    ctx_t, prediction_length=pred_len,
                    num_samples=20, limit_prediction_length=False,
                )
            finally:
                remove_prefix_hooks(chronos_model)

        pred  = samples.median(dim=1).values.squeeze(0).cpu().numpy()
        ftr_  = ftr[:pred_len]
        mse_  = float(np.mean((pred - ftr_) ** 2))
        mae_  = float(np.mean(np.abs(pred - ftr_)))
        mse_list.append(mse_)
        mae_list.append(mae_)
        mase_list.append(_mase(mae_, ctx, mase_period))

    mase_clean = [v for v in mase_list if not np.isnan(v)]
    p95_mse    = float(np.percentile(mse_list, 95)) if mse_list else float("nan")
    return {
        'n_val':          len(val_idx_arr),
        'MSE':            float(np.mean(mse_list)),
        'MAE':            float(np.mean(mae_list)),
        'MASE':           float(np.mean(mase_clean)),
        'MSE_p95':        p95_mse,
        'ceil_fire_mean': float(np.mean(ceil_fires)) if ceil_fires else float("nan"),
        'pre_ceil_mean':  float(np.mean(pre_ceil))   if pre_ceil   else float("nan"),
    }


def _load_val_split(cfg):
    path = FINETUNING_ROOT / cfg['data_path']
    with open(str(path), "rb") as f:
        table = ipc.open_file(f).read_all()
    n     = len(table)
    rng   = np.random.default_rng(TRAIN_SEED)
    idx   = rng.permutation(n)
    val_i = idx[int(n * 0.8):]
    return table, val_i, n


# ── PART A — In-distribution evaluation ───────────────────────────────────────

print("\n" + "=" * 70)
print("PART A — In-distribution evaluation (5 training datasets)")
print("=" * 70)

part_a = {}
for ds in DATASET_ORDER:
    lbl = REF[ds]['label']
    cfg = EVAL_CONFIGS[ds]
    t0  = time.time()
    print(f"\n--- {lbl} ---")
    table, v_idx, n_total = _load_val_split(cfg)
    print(f"  {n_total} total → {len(v_idx)} val series")
    res = evaluate_dataset(cfg, v_idx, table)
    print(f"  MASE={res['MASE']:.4f}  MSE={res['MSE']:.1f}  MAE={res['MAE']:.4f}"
          f"  ceil_fire={res['ceil_fire_mean']:.4f}  pre_ceil={res['pre_ceil_mean']:.4f}"
          f"  ({time.time()-t0:.1f}s)")
    part_a[ds] = res

# ── PART B — Electricity (out-of-distribution) ────────────────────────────────

print("\n" + "=" * 70)
print("PART B — Electricity (domain-shift test, ALL series)")
print("=" * 70)

elec_path = FINETUNING_ROOT / ELEC_CFG['data_path']
with open(str(elec_path), "rb") as f:
    elec_table = ipc.open_file(f).read_all()
n_elec = len(elec_table)
all_elec_idx = np.arange(n_elec)
print(f"  {n_elec} total Electricity series (ALL evaluated — none seen during Stage 2 training)")

t0  = time.time()
res_elec = evaluate_dataset(ELEC_CFG, all_elec_idx, elec_table)
print(f"  MASE={res_elec['MASE']:.4f}  MSE={res_elec['MSE']:.1f}  MAE={res_elec['MAE']:.4f}"
      f"  ceil_fire={res_elec['ceil_fire_mean']:.4f}  ({time.time()-t0:.1f}s)")


# ── Compute deltas and verdicts ────────────────────────────────────────────────

def _retain(mase_joint, mase_base, mase_solo):
    gap = mase_base - mase_solo
    if gap <= 0:
        return float("nan")
    return 1.0 - (mase_joint - mase_solo) / gap

def _delta_pct(new, old):
    if old == 0 or np.isnan(old):
        return float("nan")
    return (new - old) / abs(old) * 100


print("\n\n" + "=" * 80)
print("STAGE 2 RESULTS — 5-DATASET JOINT TRAINING")
print("=" * 80)

print("\nPART A: In-distribution retention")
print()
print(f"{'Dataset':<12}  {'MASE_base':>10}  {'MASE_solo':>10}  {'MASE_s1':>9}"
      f"  {'MASE_joint':>11}  {'Δ(solo→jt)':>11}  {'Retain%':>8}")
print("-" * 80)

part_a_summary = {}
for ds in DATASET_ORDER:
    lbl    = REF[ds]['label']
    r      = REF[ds]
    res    = part_a[ds]
    mase_j = res['MASE']
    delta  = mase_j - r['MASE_solo']
    retain = _retain(mase_j, r['MASE_base'], r['MASE_solo'])
    s1_str = f"{r['MASE_stage1']:.4f}" if r['MASE_stage1'] is not None else "   —   "
    print(f"{lbl:<12}  {r['MASE_base']:>10.4f}  {r['MASE_solo']:>10.4f}  {s1_str:>9}"
          f"  {mase_j:>11.4f}  {delta:>+11.4f}  {retain*100:>7.1f}%")
    part_a_summary[ds] = {
        'MASE_base':    r['MASE_base'],
        'MASE_solo':    r['MASE_solo'],
        'MASE_stage1':  r['MASE_stage1'],
        'MASE_joint':   mase_j,
        'MSE_joint':    res['MSE'],
        'MAE_joint':    res['MAE'],
        'MSE_p95_joint':res['MSE_p95'],
        'delta_solo':   delta,
        'retain_pct':   retain * 100,
        'ceil_fire':    res['ceil_fire_mean'],
        'pre_ceil':     res['pre_ceil_mean'],
    }

print()

# Part B
mase_j_elec  = res_elec['MASE']
delta_base   = mase_j_elec - ELEC_REF['MASE_base']
delta_solo   = mase_j_elec - ELEC_REF['MASE_solo']
improv_over_base_pct = _delta_pct(mase_j_elec, ELEC_REF['MASE_base'])
mse_delta_pct = _delta_pct(res_elec['MSE'], ELEC_REF['MSE_base'])

print("PART B: Electricity (out-of-distribution)")
print()
print(f"{'Metric':<10}  {'zero-shot':>12}  {'solo-elec':>12}  {'joint-5ds':>12}  {'Δ(0-shot→jt)':>14}")
print("-" * 65)
print(f"{'MASE':<10}  {ELEC_REF['MASE_base']:>12.4f}  {ELEC_REF['MASE_solo']:>12.4f}"
      f"  {mase_j_elec:>12.4f}  {improv_over_base_pct:>+14.2f}%")
print(f"{'MSE':<10}  {ELEC_REF['MSE_base']:>12.0f}  {ELEC_REF['MSE_solo']:>12.0f}"
      f"  {res_elec['MSE']:>12.0f}  {mse_delta_pct:>+14.2f}%")
print(f"{'MAE':<10}  {ELEC_REF['MAE_base']:>12.2f}  {ELEC_REF['MAE_solo']:>12.2f}"
      f"  {res_elec['MAE']:>12.2f}  {'—':>14}")
print()

# Interpret outcome
if mase_j_elec < ELEC_REF['MASE_solo']:
    transfer_verdict = "STRONG TRANSFER"
    transfer_note = ("Joint training beat in-domain solo Electricity training — "
                     "exceptional positive transfer.")
elif mase_j_elec < ELEC_REF['MASE_base']:
    transfer_verdict = "PARTIAL TRANSFER"
    transfer_note = ("Joint training improved over zero-shot without ever seeing Electricity. "
                     "In-domain solo is still better. Primary paper target achieved.")
elif mase_j_elec < ELEC_REF['MASE_base'] + 0.01:
    transfer_verdict = "NEUTRAL"
    transfer_note = ("Joint training is approximately zero-shot performance. "
                     "No benefit over baseline; no harm either.")
else:
    transfer_verdict = "NEGATIVE TRANSFER"
    transfer_note = ("Joint training hurt Electricity vs zero-shot. "
                     "Investigate fix1b rate and period mismatch on this domain.")

stage3_cleared = mase_j_elec < ELEC_REF['MASE_base']

print(f"VERDICT: {transfer_verdict}")
print(f"  {transfer_note}")
print(f"STAGE 3 CLEARED: {stage3_cleared}"
      f"  (MASE_joint={mase_j_elec:.4f} {'<' if stage3_cleared else '>='} MASE_base={ELEC_REF['MASE_base']:.4f})")


# ── Save JSON ─────────────────────────────────────────────────────────────────

json_blob = {
    'checkpoint':      str(CKPT_PATH.name),
    'ckpt_step':       int(ckpt['step']),
    'datasets_trained': DATASET_ORDER,
    'part_a':          part_a_summary,
    'part_b': {
        'n_series':         n_elec,
        'MASE_base':        ELEC_REF['MASE_base'],
        'MASE_solo':        ELEC_REF['MASE_solo'],
        'MSE_base':         ELEC_REF['MSE_base'],
        'MSE_solo':         ELEC_REF['MSE_solo'],
        'MAE_base':         ELEC_REF['MAE_base'],
        'MAE_solo':         ELEC_REF['MAE_solo'],
        'MASE_joint':       mase_j_elec,
        'MSE_joint':        res_elec['MSE'],
        'MAE_joint':        res_elec['MAE'],
        'MSE_p95_joint':    res_elec['MSE_p95'],
        'delta_vs_base':    delta_base,
        'delta_vs_solo':    delta_solo,
        'delta_base_pct':   improv_over_base_pct,
        'mse_delta_base_pct': mse_delta_pct,
        'ceil_fire_mean':   res_elec['ceil_fire_mean'],
        'pre_ceil_mean':    res_elec['pre_ceil_mean'],
        'transfer_verdict': transfer_verdict,
        'transfer_note':    transfer_note,
    },
    'stage3_cleared': stage3_cleared,
}

json_out = RESULTS_DIR / "joint_stage2_eval.json"
with open(str(json_out), "w") as f:
    json.dump(json_blob, f, indent=2)
print(f"\nSaved → {json_out}")


# ── Save summary text ─────────────────────────────────────────────────────────

summary_lines = [
    "STAGE 2 RESULTS — 5-DATASET JOINT TRAINING",
    "═" * 60,
    "",
    "Checkpoint: " + str(CKPT_PATH.name),
    "",
    "PART A: In-distribution retention",
    "",
    f"{'Dataset':<12}  {'MASE_base':>10}  {'MASE_solo':>10}  {'MASE_s1':>9}"
    f"  {'MASE_joint':>11}  {'Δ(solo→jt)':>11}  {'Retain%':>8}",
    "-" * 80,
]
for ds in DATASET_ORDER:
    s = part_a_summary[ds]
    lbl = REF[ds]['label']
    s1s = f"{s['MASE_stage1']:.4f}" if s['MASE_stage1'] is not None else "   —   "
    summary_lines.append(
        f"{lbl:<12}  {s['MASE_base']:>10.4f}  {s['MASE_solo']:>10.4f}  {s1s:>9}"
        f"  {s['MASE_joint']:>11.4f}  {s['delta_solo']:>+11.4f}  {s['retain_pct']:>7.1f}%"
    )
summary_lines += [
    "",
    "PART B: Electricity (out-of-distribution)",
    "",
    f"{'Metric':<10}  {'zero-shot':>12}  {'solo-elec':>12}  {'joint-5ds':>12}  {'Δ(0-shot→jt)':>14}",
    "-" * 65,
    f"{'MASE':<10}  {ELEC_REF['MASE_base']:>12.4f}  {ELEC_REF['MASE_solo']:>12.4f}"
    f"  {mase_j_elec:>12.4f}  {improv_over_base_pct:>+14.2f}%",
    f"{'MSE':<10}  {ELEC_REF['MSE_base']:>12.0f}  {ELEC_REF['MSE_solo']:>12.0f}"
    f"  {res_elec['MSE']:>12.0f}  {mse_delta_pct:>+14.2f}%",
    f"{'MAE':<10}  {ELEC_REF['MAE_base']:>12.2f}  {ELEC_REF['MAE_solo']:>12.2f}"
    f"  {res_elec['MAE']:>12.2f}",
    "",
    f"VERDICT: {transfer_verdict}",
    f"  {transfer_note}",
    f"STAGE 3 CLEARED: {stage3_cleared}",
]
summary_str = "\n".join(summary_lines)

txt_out = RESULTS_DIR / "joint_stage2_summary.txt"
with open(str(txt_out), "w") as f:
    f.write(summary_str + "\n")
print(f"Saved → {txt_out}")

print(f"\nVERDICT: {transfer_verdict}")
print(f"STAGE 3 CLEARED: {stage3_cleared}")
