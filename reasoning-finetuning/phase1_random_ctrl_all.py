#!/usr/bin/env python3
"""
phase1_random_ctrl_all.py — Phase 1 Random Control on ALL 5 Datasets

Extends Step 3 from phase1_recheck.py (which ran random backbone only on
M4 Hourly) to all 5 datasets.

Pretrained R² loaded directly from results/phase1_recheck.json — no need
to re-run the pretrained backbone.

Random backbone uses SAME normalized labels (FIX A: /ctx_scale) and
FFT-detected periods (no forced benchmark override) as phase1_recheck.py.

Verdict: OUTCOME A holds for a dataset only where
  pretrained R² > random R² + 0.15
on the SAME dataset (not borrowing M4 Hourly's control for other datasets).

Saves: results/phase1_random_ctrl_all.json
       results/phase1_random_ctrl_all.md
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

REPO_ROOT       = Path(__file__).parent.parent
FINETUNING_ROOT = Path(__file__).parent
DATA_DIR        = FINETUNING_ROOT / "data"
RESULTS_DIR     = REPO_ROOT / "results"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(FINETUNING_ROOT))

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn as nn

from chronos import ChronosPipeline
from scripts.step1_stl_decompose import detect_periods, stl_decompose
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (identical to phase1_recheck.py) ────────────────────────────────

SMOKE_TEST  = os.environ.get("SMOKE_TEST", "0") == "1"
TRAIN_SEED  = 42
TRAIN_FRAC  = 0.8
D_MODEL     = 512
NUM_LAYERS  = 6
RS_THRESH   = 0.4
RIDGE_ALPHA = 1.0
BATCH_SIZE  = 16 if not SMOKE_TEST else 4

_MAX = {
    'm4hourly':    None,
    'm4monthly':   1000,
    'm4daily':     2000,
    'm4weekly':    None,
    'electricity': None,
}
if SMOKE_TEST:
    _MAX = {k: 20 for k in _MAX}

DS_CONFIGS = {
    'm4hourly': {
        'data_path': 'm4hourly-tsf.arrow',
        'context_col': 'context', 'ctx_len': 96,
        'min_period': 6, 'max_period': 24, 'label': 'M4 Hourly',
    },
    'm4monthly': {
        'data_path': 'm4monthly-tsf.arrow',
        'context_col': 'context', 'ctx_len': 36,
        'min_period': 4, 'max_period': 12, 'label': 'M4 Monthly',
    },
    'm4daily': {
        'data_path': 'm4daily-tsf.arrow',
        'context_col': 'context', 'ctx_len': 93,
        'min_period': 4, 'max_period': 31, 'label': 'M4 Daily',
    },
    'm4weekly': {
        'data_path': 'm4weekly-tsf.arrow',
        'context_col': 'context', 'ctx_len': 80,
        'min_period': 4, 'max_period': 80, 'label': 'M4 Weekly',
    },
    'electricity': {
        'data_path': 'electricity-tsf.arrow',
        'context_col': 'context', 'ctx_len': 336,
        'min_period': 6, 'max_period': 84, 'label': 'Electricity',
    },
}

COMPONENTS = ['trend', 'seasonal', 'residual']
DS_ORDER   = ['m4hourly', 'm4monthly', 'm4daily', 'm4weekly', 'electricity']
GAP_THRESH = 0.15   # pretrained must exceed random control by this much per dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  {'SMOKE' if SMOKE_TEST else 'FULL'} run")
print()


# ── Load pretrained results from phase1_recheck.json ─────────────────────────

print("=" * 60)
print("Loading pretrained R² from phase1_recheck.json")
print("=" * 60)
print()

recheck_path = RESULTS_DIR / "phase1_recheck.json"
assert recheck_path.exists(), f"Missing {recheck_path} — run phase1_recheck.py first"
with open(str(recheck_path)) as f:
    recheck = json.load(f)

pre_r2 = {}   # ds_name → np.ndarray (NUM_LAYERS, 3)  — val R², FFT periods, normalized
for ds_name in DS_ORDER:
    pre_r2[ds_name] = np.array(recheck['datasets'][ds_name]['r2_val_fft'])

print("  Pretrained best R² per dataset (normalized, FFT periods):")
print(f"  {'Dataset':>14}  {'BestTrend':>10}  {'Layer':>5}  {'BestSeas':>10}  {'Layer':>5}")
for ds_name in DS_ORDER:
    r2  = pre_r2[ds_name]
    bt  = float(r2[:, 0].max());  lt = int(r2[:, 0].argmax())
    bs  = float(r2[:, 1].max());  ls = int(r2[:, 1].argmax())
    print(f"  {DS_CONFIGS[ds_name]['label']:>14}  {bt:>10.4f}  {lt:>5}  {bs:>10.4f}  {ls:>5}")
print()


# ── Backbone loader ───────────────────────────────────────────────────────────

def load_backbone(random_init: bool = False):
    pl = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map=str(device), dtype=torch.float32,
    )
    if random_init:
        for p in pl.model.parameters():
            nn.init.normal_(p, mean=0.0, std=0.02)
    for p in pl.model.parameters():
        p.requires_grad_(False)
    pl.model.eval()
    assert sum(1 for p in pl.model.parameters() if p.requires_grad) == 0
    return pl


# ── Collection + probes (normalized labels, FFT periods) ─────────────────────

def collect_and_probe(ds_name, backbone_pl):
    """
    Run frozen backbone on ds_name, fit Ridge probes with normalized labels.
    Uses SAME split/scaler/alpha as phase1_recheck.py.
    Returns np.ndarray (NUM_LAYERS, 3) of val R² values.
    """
    cfg  = DS_CONFIGS[ds_name]
    path = DATA_DIR / cfg['data_path']
    with open(str(path), "rb") as f:
        table = ipc.open_file(f).read_all()

    n_total = len(table)
    rng     = np.random.default_rng(TRAIN_SEED)
    perm    = rng.permutation(n_total)
    max_n   = _MAX[ds_name]
    if max_n is not None and n_total > max_n:
        perm  = perm[:max_n]
        n_use = max_n
    else:
        n_use = n_total

    n_train = int(n_use * TRAIN_FRAC)
    idx_tr  = perm[:n_train]
    idx_va  = perm[n_train:]
    ctx_len = cfg['ctx_len']

    def gather(series_idx):
        N   = len(series_idx)
        tok = backbone_pl.tokenizer
        bpl = backbone_pl
        lh  = [np.empty((N * ctx_len, D_MODEL), dtype=np.float32) for _ in range(NUM_LAYERS)]
        lab = {c: np.empty(N * ctx_len, dtype=np.float32) for c in COMPONENTS}
        t0  = time.time()

        for b_start in range(0, N, BATCH_SIZE):
            b_idx = series_idx[b_start : b_start + BATCH_SIZE]
            B     = len(b_idx)
            ctxs  = []
            for si in b_idx:
                raw = np.array(table.column(cfg['context_col'])[int(si)].as_py(),
                               dtype=np.float32)
                ctxs.append(raw[:ctx_len])

            ctx_batch = torch.tensor(np.stack(ctxs), dtype=torch.float32)

            with torch.no_grad():
                inp_ids, attn, scale_t = tok.context_input_transform(
                    bpl._prepare_and_validate_context(ctx_batch)
                )
                enc_out = bpl.model.model.encoder(
                    input_ids      = inp_ids.to(device),
                    attention_mask = attn.to(device),
                    output_hidden_states = True,
                )
            ctx_scales = scale_t.cpu().numpy()  # (B,)

            for ℓ in range(NUM_LAYERS):
                h    = enc_out.hidden_states[ℓ + 1][:, :ctx_len, :].detach().cpu().numpy()
                slot = slice(b_start * ctx_len, (b_start + B) * ctx_len)
                lh[ℓ][slot] = h.reshape(B * ctx_len, D_MODEL)

            for i, ctx_np in enumerate(ctxs):
                ctx_f64 = ctx_np.astype(np.float64)
                cscale  = float(ctx_scales[i])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    periods = detect_periods(ctx_f64,
                                             min_period=cfg['min_period'],
                                             max_period=cfg['max_period'])
                    tr, se, re = stl_decompose(ctx_f64, periods)
                pos = (b_start + i) * ctx_len
                lab['trend']   [pos:pos+ctx_len] = (tr / cscale).astype(np.float32)
                lab['seasonal'][pos:pos+ctx_len] = (se / cscale).astype(np.float32)
                lab['residual'][pos:pos+ctx_len] = (re / cscale).astype(np.float32)

            done = b_start + B
            if done % (20 * BATCH_SIZE) == 0 or done >= N:
                print(f"    {done:4d}/{N}  ({time.time()-t0:.0f}s)")

        return lh, lab

    print(f"  N={n_use}  train={len(idx_tr)}  val={len(idx_va)}")
    print("  Collecting train …")
    h_tr, lab_tr = gather(idx_tr)
    print("  Collecting val …")
    h_va, lab_va = gather(idx_va)

    r2_val = np.full((NUM_LAYERS, 3), np.nan)
    for ci, comp in enumerate(COMPONENTS):
        y_tr = lab_tr[comp];  y_va = lab_va[comp]
        for ℓ in range(NUM_LAYERS):
            sc     = StandardScaler()
            Xtr_s  = sc.fit_transform(h_tr[ℓ])
            Xva_s  = sc.transform(h_va[ℓ])
            clf    = Ridge(alpha=RIDGE_ALPHA)
            clf.fit(Xtr_s, y_tr)
            r2_val[ℓ, ci] = float(r2_score(y_va, clf.predict(Xva_s)))

    del h_tr, h_va, lab_tr, lab_va
    gc.collect()
    return r2_val


# ── Run random backbone on all 5 datasets ────────────────────────────────────

print("=" * 60)
print("Random backbone control — all 5 datasets")
print("=" * 60)
print()

print("Loading random-initialized Chronos T5-Small …")
pl_rand = load_backbone(random_init=True)
print("  Weights: N(0, 0.02) — no pretrained knowledge")
print()

rand_r2 = {}
for ds_name in DS_ORDER:
    label = DS_CONFIGS[ds_name]['label']
    print(f"── {label} ──────────────────────")
    t0 = time.time()
    rand_r2[ds_name] = collect_and_probe(ds_name, pl_rand)
    print(f"  Wall time: {time.time()-t0:.0f}s")
    print()

del pl_rand
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()


# ── Report: per-dataset best R² comparison ───────────────────────────────────

print("=" * 60)
print("Results: pretrained vs random control (normalized labels)")
print("=" * 60)
print()

# Seasonal
print("  SEASONAL:")
print(f"  {'Dataset':>14}  {'Pre':>8}  {'Rand':>8}  {'Gap':>8}  {'Above ctrl?':>14}")
seas_verdict = {}
for ds_name in DS_ORDER:
    pr  = float(pre_r2[ds_name][:, 1].max())
    rnd = float(rand_r2[ds_name][:, 1].max())
    gap = pr - rnd
    above = gap > GAP_THRESH
    seas_verdict[ds_name] = above
    flag = f"YES (+{gap:.3f})" if above else f"NO  ({gap:+.3f})"
    print(f"  {DS_CONFIGS[ds_name]['label']:>14}  {pr:>8.4f}  {rnd:>8.4f}  {gap:>+8.4f}  {flag}")

print()

# Trend
print("  TREND:")
print(f"  {'Dataset':>14}  {'Pre':>8}  {'Rand':>8}  {'Gap':>8}  {'Above ctrl?':>14}")
trend_verdict = {}
for ds_name in DS_ORDER:
    pr  = float(pre_r2[ds_name][:, 0].max())
    rnd = float(rand_r2[ds_name][:, 0].max())
    gap = pr - rnd
    above = gap > GAP_THRESH
    trend_verdict[ds_name] = above
    flag = f"YES (+{gap:.3f})" if above else f"NO  ({gap:+.3f})"
    print(f"  {DS_CONFIGS[ds_name]['label']:>14}  {pr:>8.4f}  {rnd:>8.4f}  {gap:>+8.4f}  {flag}")

print()

# Layer-wise tables for random backbone
print("  Layer-wise R² for RANDOM backbone (all datasets):")
for ds_name in DS_ORDER:
    r2  = rand_r2[ds_name]
    print(f"  {DS_CONFIGS[ds_name]['label']}:")
    print(f"  {'Layer':>5}  {'Trend':>8}  {'Seasonal':>10}  {'Residual':>10}")
    for ℓ in range(NUM_LAYERS):
        print(f"  {ℓ:>5}  {r2[ℓ,0]:>8.4f}  {r2[ℓ,1]:>10.4f}  {r2[ℓ,2]:>10.4f}")
    print()


# ── Verdict ───────────────────────────────────────────────────────────────────

print("=" * 60)
print("VERDICT (per dataset, gap > 0.15 = above own random control)")
print("=" * 60)
print()

outcome_a_ds = []
outcome_b_ds = []
mixed_ds     = []

for ds_name in DS_ORDER:
    t_above = trend_verdict[ds_name]
    s_above = seas_verdict[ds_name]
    label   = DS_CONFIGS[ds_name]['label']
    pr_t    = float(pre_r2[ds_name][:, 0].max())
    pr_s    = float(pre_r2[ds_name][:, 1].max())
    rnd_t   = float(rand_r2[ds_name][:, 0].max())
    rnd_s   = float(rand_r2[ds_name][:, 1].max())

    if t_above and s_above:
        outcome_a_ds.append(ds_name)
        tag = "OUTCOME_A (both trend+seasonal decodable above ctrl)"
    elif not t_above and not s_above:
        outcome_b_ds.append(ds_name)
        tag = "OUTCOME_B (neither above ctrl)"
    else:
        mixed_ds.append(ds_name)
        comp_t = f"trend {'✓' if t_above else '✗'}"
        comp_s = f"seasonal {'✓' if s_above else '✗'}"
        tag = f"MIXED ({comp_t}, {comp_s})"

    print(f"  {label}:")
    print(f"    trend:    pre={pr_t:.4f}  rand={rnd_t:.4f}  gap={pr_t-rnd_t:+.4f}  {'✓' if t_above else '✗'}")
    print(f"    seasonal: pre={pr_s:.4f}  rand={rnd_s:.4f}  gap={pr_s-rnd_s:+.4f}  {'✓' if s_above else '✗'}")
    print(f"    → {tag}")
    print()

overall = ("OUTCOME_A" if outcome_a_ds and not outcome_b_ds and not mixed_ds
           else "OUTCOME_B" if outcome_b_ds and not outcome_a_ds and not mixed_ds
           else "MIXED")
print(f"  OVERALL: {overall}")
print(f"    OUTCOME_A datasets: {[DS_CONFIGS[d]['label'] for d in outcome_a_ds]}")
print(f"    MIXED datasets:     {[DS_CONFIGS[d]['label'] for d in mixed_ds]}")
print(f"    OUTCOME_B datasets: {[DS_CONFIGS[d]['label'] for d in outcome_b_ds]}")
print()
print("  STOP — awaiting review before Phase 2.")


# ── Save ─────────────────────────────────────────────────────────────────────

rand_r2_serializable = {ds: rand_r2[ds].tolist() for ds in DS_ORDER}

results = {
    'run_config': {
        'smoke_test':   SMOKE_TEST,
        'ridge_alpha':  RIDGE_ALPHA,
        'train_seed':   TRAIN_SEED,
        'train_frac':   TRAIN_FRAC,
        'gap_threshold':GAP_THRESH,
        'normalization':'labels / ctx_scale (FIX A)',
        'periods':      'FFT-detected per series (same as phase1_recheck)',
    },
    'pretrained_source': 'results/phase1_recheck.json (datasets.r2_val_fft)',
    'random_r2': rand_r2_serializable,
    'per_dataset': {},
    'overall_verdict': overall,
    'outcome_a_datasets': outcome_a_ds,
    'mixed_datasets':     mixed_ds,
    'outcome_b_datasets': outcome_b_ds,
}

for ds_name in DS_ORDER:
    pr_t  = float(pre_r2[ds_name][:, 0].max())
    pr_s  = float(pre_r2[ds_name][:, 1].max())
    rnd_t = float(rand_r2[ds_name][:, 0].max())
    rnd_s = float(rand_r2[ds_name][:, 1].max())
    results['per_dataset'][ds_name] = {
        'pretrained_trend':    pr_t,
        'pretrained_seasonal': pr_s,
        'random_trend':        rnd_t,
        'random_seasonal':     rnd_s,
        'gap_trend':           pr_t - rnd_t,
        'gap_seasonal':        pr_s - rnd_s,
        'trend_above_ctrl':    bool(trend_verdict[ds_name]),
        'seasonal_above_ctrl': bool(seas_verdict[ds_name]),
    }

json_path = RESULTS_DIR / "phase1_random_ctrl_all.json"
with open(str(json_path), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {json_path}")

# Markdown
md = [
    "# Phase 1 — Random Control on All 5 Datasets",
    "",
    f"**Gap threshold**: {GAP_THRESH} (same as M4 Hourly precedent from job 7408).",
    "**Labels**: normalized by `ctx_scale` (FIX A). **Periods**: FFT-detected.",
    "",
    "## Seasonal",
    "",
    "| Dataset | Pretrained | Random ctrl | Gap | Above ctrl? |",
    "|---------|-----------|-------------|-----|-------------|",
]
for ds_name in DS_ORDER:
    d = results['per_dataset'][ds_name]
    gap = d['gap_seasonal']
    md.append(f"| {DS_CONFIGS[ds_name]['label']} | {d['pretrained_seasonal']:.4f} | "
              f"{d['random_seasonal']:.4f} | {gap:+.4f} | "
              f"{'✓' if d['seasonal_above_ctrl'] else '✗'} |")

md += [
    "",
    "## Trend",
    "",
    "| Dataset | Pretrained | Random ctrl | Gap | Above ctrl? |",
    "|---------|-----------|-------------|-----|-------------|",
]
for ds_name in DS_ORDER:
    d = results['per_dataset'][ds_name]
    gap = d['gap_trend']
    md.append(f"| {DS_CONFIGS[ds_name]['label']} | {d['pretrained_trend']:.4f} | "
              f"{d['random_trend']:.4f} | {gap:+.4f} | "
              f"{'✓' if d['trend_above_ctrl'] else '✗'} |")

md += [
    "",
    f"## Overall Verdict: {overall}",
    "",
    f"- OUTCOME A datasets: {[DS_CONFIGS[d]['label'] for d in outcome_a_ds]}",
    f"- MIXED datasets: {[DS_CONFIGS[d]['label'] for d in mixed_ds]}",
    f"- OUTCOME B datasets: {[DS_CONFIGS[d]['label'] for d in outcome_b_ds]}",
    "",
    "## Random Backbone Layer Tables",
    "",
]
for ds_name in DS_ORDER:
    r2 = rand_r2[ds_name]
    md += [
        f"### {DS_CONFIGS[ds_name]['label']}",
        "| Layer | Trend | Seasonal | Residual |",
        "|-------|-------|----------|----------|",
    ]
    for ℓ in range(NUM_LAYERS):
        md.append(f"| {ℓ} | {r2[ℓ,0]:.4f} | {r2[ℓ,1]:.4f} | {r2[ℓ,2]:.4f} |")
    md.append("")

md_path = RESULTS_DIR / "phase1_random_ctrl_all.md"
with open(str(md_path), "w") as f:
    f.write("\n".join(md) + "\n")
print(f"Saved → {md_path}")
print()
print(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
