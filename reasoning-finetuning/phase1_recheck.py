#!/usr/bin/env python3
"""
phase1_recheck.py — Phase 1 Re-check

FIX A: Re-run trend probe with target = trend / ctx_scale
       ctx_scale = mean(|context|), clamped >=1.0 (same as Chronos tokenizer).
       Raw trend is correlated with ctx_scale (absolute level), which inflates
       apparent R² and masks whether the backbone encodes trend SHAPE.

FIX B: STL period audit.
       For each dataset, report EXACTLY which period detect_periods() returned
       (dominant FFT period), vs the M4/Electricity benchmark period.
       For M4 Daily and Electricity, run DUAL STL: (a) FFT-detected, (b) benchmark.
       Compare seasonal R² under each; if they differ, report both.

Saves: results/phase1_recheck.json
       results/phase1_recheck.md
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

# ── Constants (identical to phase1_internal_decomposition_probe.py) ───────────

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

# Industry-standard benchmark (M4 competition / standard TSF) periods
BENCHMARK_PERIODS = {
    'm4hourly':    24,   # daily cycle in hourly data
    'm4monthly':   12,   # annual cycle in monthly data
    'm4daily':      7,   # weekly cycle in daily data
    'm4weekly':    52,   # annual cycle in weekly data
    'electricity': 24,   # daily cycle in hourly data
}

# Datasets for which we run the dual-STL comparison (FIX B suspects)
DUAL_STL_DATASETS = {'m4daily', 'electricity'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  {'SMOKE' if SMOKE_TEST else 'FULL'} run")
print(f"RIDGE_ALPHA={RIDGE_ALPHA}  BATCH_SIZE={BATCH_SIZE}  TRAIN_SEED={TRAIN_SEED}")
print()


# ── STEP 0: Period audit (CPU-only, no encoder) ───────────────────────────────

print("=" * 60)
print("STEP 0: STL period audit per dataset")
print("=" * 60)
print()

AUDIT_N = 50 if not SMOKE_TEST else 10

audit_results = {}

for ds_name in DS_ORDER:
    cfg      = DS_CONFIGS[ds_name]
    path     = DATA_DIR / cfg['data_path']
    with open(str(path), "rb") as f:
        table = ipc.open_file(f).read_all()
    n_total = len(table)

    rng   = np.random.default_rng(TRAIN_SEED)
    perm  = rng.permutation(n_total)
    sample_idx = perm[:min(AUDIT_N, n_total)]

    dominant_periods = []
    all_periods      = []

    for si in sample_idx:
        raw = np.array(table.column(cfg['context_col'])[int(si)].as_py(), dtype=np.float64)
        raw = raw[:cfg['ctx_len']]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detected = detect_periods(
                raw,
                min_period=cfg['min_period'],
                max_period=cfg['max_period'],
            )
        dominant_periods.append(detected[0])
        all_periods.append(detected)

    cnt      = Counter(dominant_periods)
    benchmark = BENCHMARK_PERIODS[ds_name]
    mode_p   = cnt.most_common(1)[0][0]
    median_p = int(np.median(dominant_periods))
    matches_benchmark = (mode_p == benchmark)

    print(f"  {cfg['label']:>14}  "
          f"benchmark={benchmark:>3}  "
          f"mode_detected={mode_p:>3}  "
          f"median={median_p:>3}  "
          f"{'MATCH' if matches_benchmark else 'MISMATCH ← flag'}")
    if not matches_benchmark:
        top5 = cnt.most_common(5)
        print(f"    top-5 detected: {top5}")

    audit_results[ds_name] = {
        'benchmark_period':  benchmark,
        'mode_period':       mode_p,
        'median_period':     median_p,
        'matches_benchmark': matches_benchmark,
        'distribution':      dict(cnt.most_common(10)),
    }

print()
mismatch_sets = {k for k, v in audit_results.items() if not v['matches_benchmark']}
if mismatch_sets:
    print(f"  FLAGGED DATASETS (period mismatch): {mismatch_sets}")
else:
    print("  All datasets: mode FFT period matches benchmark ✓")
print()


# ── Load frozen backbone ─────────────────────────────────────────────────────

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
    n_grad = sum(1 for p in pl.model.parameters() if p.requires_grad)
    assert n_grad == 0, f"Model NOT fully frozen: {n_grad} trainable params"
    return pl


print("Loading frozen pretrained Chronos T5-Small …")
pl_pre    = load_backbone(random_init=False)
model     = pl_pre.model
tokenizer = pl_pre.tokenizer
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  FROZEN")
print()


# ── Core collection ─────────────────────────────────────────────────────────

def collect_hidden_and_labels(cfg, table, series_idx, backbone_pl,
                               also_benchmark: bool = False,
                               benchmark_period: int = 7):
    """
    Collect encoder hidden states + STL labels (normalized by ctx_scale).

    FIX A: labels are divided by ctx_scale = mean(|context|) ≥ 1.0,
           matching the Chronos tokenizer normalization.
    FIX B: if also_benchmark=True, also compute labels using BENCHMARK period,
           returned as a separate dict.

    Returns
    -------
    layer_hiddens  : list[6] np.ndarray (N*L, 512)
    labels_fft     : dict component → np.ndarray (N*L,)  — FFT-detected period, normalized
    labels_bmk     : dict or None                          — benchmark period, normalized
    n_rs           : int   — series where residual/signal > RS_THRESH
    period_log     : list  — detected periods per series (for FIX B audit)
    """
    ctx_len = cfg['ctx_len']
    N       = len(series_idx)
    tok     = backbone_pl.tokenizer
    bpl     = backbone_pl

    layer_hiddens = [
        np.empty((N * ctx_len, D_MODEL), dtype=np.float32)
        for _ in range(NUM_LAYERS)
    ]
    labels_fft = {c: np.empty(N * ctx_len, dtype=np.float32) for c in COMPONENTS}
    labels_bmk: dict | None = \
        {c: np.empty(N * ctx_len, dtype=np.float32) for c in COMPONENTS} \
        if also_benchmark else None

    n_rs       = 0
    period_log = []
    t0         = time.time()

    for b_start in range(0, N, BATCH_SIZE):
        b_idx = series_idx[b_start : b_start + BATCH_SIZE]
        B     = len(b_idx)

        ctxs_np = []
        for si in b_idx:
            raw = np.array(table.column(cfg['context_col'])[int(si)].as_py(),
                           dtype=np.float32)
            ctxs_np.append(raw[:ctx_len])

        ctx_batch = torch.tensor(np.stack(ctxs_np), dtype=torch.float32)

        # Encoder forward
        with torch.no_grad():
            inp_ids, attn, scale_t = tok.context_input_transform(
                bpl._prepare_and_validate_context(ctx_batch)
            )
            enc_out = backbone_pl.model.model.encoder(
                input_ids       = inp_ids.to(device),
                attention_mask  = attn.to(device),
                output_hidden_states = True,
            )
        # scale_t shape: (B,) — this IS ctx_scale from Chronos tokenizer
        ctx_scales = scale_t.cpu().numpy()  # (B,)

        for ℓ in range(NUM_LAYERS):
            h    = enc_out.hidden_states[ℓ + 1]
            h    = h[:, :ctx_len, :].detach().cpu().numpy()
            slot = slice(b_start * ctx_len, (b_start + B) * ctx_len)
            layer_hiddens[ℓ][slot] = h.reshape(B * ctx_len, D_MODEL)

        # STL decomposition (FIX A + FIX B)
        for i, ctx_np in enumerate(ctxs_np):
            ctx_f64    = ctx_np.astype(np.float64)
            ctx_scale  = float(ctx_scales[i])  # Chronos mean-abs scale

            # FFT-detected periods
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fft_periods = detect_periods(ctx_f64,
                                             min_period=cfg['min_period'],
                                             max_period=cfg.get('max_period'))
            period_log.append(fft_periods[0])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tr, se, re = stl_decompose(ctx_f64, fft_periods)

            rs = float(np.var(re) / (np.var(ctx_f64) + 1e-9))
            if rs > RS_THRESH:
                n_rs += 1

            pos = (b_start + i) * ctx_len
            # FIX A: divide by ctx_scale
            labels_fft['trend']   [pos:pos+ctx_len] = (tr / ctx_scale).astype(np.float32)
            labels_fft['seasonal'][pos:pos+ctx_len] = (se / ctx_scale).astype(np.float32)
            labels_fft['residual'][pos:pos+ctx_len] = (re / ctx_scale).astype(np.float32)

            if labels_bmk is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tr_b, se_b, re_b = stl_decompose(ctx_f64, [benchmark_period])
                labels_bmk['trend']   [pos:pos+ctx_len] = (tr_b / ctx_scale).astype(np.float32)
                labels_bmk['seasonal'][pos:pos+ctx_len] = (se_b / ctx_scale).astype(np.float32)
                labels_bmk['residual'][pos:pos+ctx_len] = (re_b / ctx_scale).astype(np.float32)

        done = b_start + B
        if done % (20 * BATCH_SIZE) == 0 or done >= N:
            print(f"    {done:4d}/{N}  ({time.time()-t0:.0f}s)")

    return layer_hiddens, labels_fft, labels_bmk, n_rs, period_log


# ── Ridge probe helper ────────────────────────────────────────────────────────

def run_probes(layer_h_tr, labels_tr, layer_h_va, labels_va):
    """Ridge probes per (layer, component). Returns (r2_val, r2_train, null_r2)."""
    r2_val   = np.full((NUM_LAYERS, 3), np.nan)
    r2_train = np.full((NUM_LAYERS, 3), np.nan)
    null_r2  = np.full(3, np.nan)

    for ci, comp in enumerate(COMPONENTS):
        y_tr = labels_tr[comp]
        y_va = labels_va[comp]
        y_mean = float(y_tr.mean())
        null_r2[ci] = float(r2_score(y_va, np.full_like(y_va, y_mean)))

        for ℓ in range(NUM_LAYERS):
            sc       = StandardScaler()
            X_tr_s   = sc.fit_transform(layer_h_tr[ℓ])
            X_va_s   = sc.transform(layer_h_va[ℓ])
            clf      = Ridge(alpha=RIDGE_ALPHA)
            clf.fit(X_tr_s, y_tr)
            r2_train[ℓ, ci] = float(r2_score(y_tr, clf.predict(X_tr_s)))
            r2_val  [ℓ, ci] = float(r2_score(y_va, clf.predict(X_va_s)))

    return r2_val, r2_train, null_r2


# ── Process one dataset ───────────────────────────────────────────────────────

def process_dataset(ds_name, backbone_pl):
    cfg  = DS_CONFIGS[ds_name]
    path = DATA_DIR / cfg['data_path']
    with open(str(path), "rb") as f:
        table = ipc.open_file(f).read_all()

    n_total = len(table)
    rng     = np.random.default_rng(TRAIN_SEED)
    perm    = rng.permutation(n_total)

    max_n   = _MAX[ds_name]
    if max_n is not None and n_total > max_n:
        perm   = perm[:max_n]
        n_use  = max_n
    else:
        n_use  = n_total

    n_train = int(n_use * TRAIN_FRAC)
    idx_tr  = perm[:n_train]
    idx_va  = perm[n_train:]
    ctx_len = cfg['ctx_len']

    also_bmk = ds_name in DUAL_STL_DATASETS
    print(f"  N={n_use}  train={len(idx_tr)}  val={len(idx_va)}  "
          f"dual_stl={'YES' if also_bmk else 'no'}")

    print("  Collecting train hidden states + labels …")
    bmk_period = BENCHMARK_PERIODS[ds_name]
    h_tr, lab_tr_fft, lab_tr_bmk, rs_tr, pl_tr = collect_hidden_and_labels(
        cfg, table, idx_tr, backbone_pl,
        also_benchmark=also_bmk, benchmark_period=bmk_period,
    )
    print(f"    RS_THRESH exceeded: {rs_tr}/{len(idx_tr)}")
    print(f"    Dominant periods (sample): mode={Counter(pl_tr).most_common(1)[0][0]}")

    print("  Collecting val hidden states + labels …")
    h_va, lab_va_fft, lab_va_bmk, rs_va, pl_va = collect_hidden_and_labels(
        cfg, table, idx_va, backbone_pl,
        also_benchmark=also_bmk, benchmark_period=bmk_period,
    )

    print("  Fitting Ridge probes (FFT-detected periods, normalized) …")
    r2_val_fft, r2_tr_fft, null_r2 = run_probes(
        h_tr, lab_tr_fft, h_va, lab_va_fft,
    )

    r2_val_bmk, r2_tr_bmk = None, None
    if also_bmk:
        print(f"  Fitting Ridge probes (benchmark period={BENCHMARK_PERIODS[ds_name]}, normalized) …")
        r2_val_bmk, r2_tr_bmk, _ = run_probes(
            h_tr, lab_tr_bmk, h_va, lab_va_bmk,
        )

    # Print table
    print()
    hdr = f"  {'Layer':>5}  {'Trend':>8}  {'Seas(FFT)':>10}  {'Seas(BMK)':>10}  {'Resid':>8}  (val R²)"
    print(hdr)
    for ℓ in range(NUM_LAYERS):
        bmk_str = f"{r2_val_bmk[ℓ,1]:>10.4f}" if r2_val_bmk is not None else f"{'—':>10}"
        print(f"  {ℓ:>5}  {r2_val_fft[ℓ,0]:>8.4f}  "
              f"{r2_val_fft[ℓ,1]:>10.4f}  {bmk_str}  {r2_val_fft[ℓ,2]:>8.4f}")
    print()

    del h_tr, h_va, lab_tr_fft, lab_va_fft, lab_tr_bmk, lab_va_bmk
    gc.collect()

    return {
        'r2_val_fft':  r2_val_fft.tolist(),
        'r2_val_bmk':  r2_val_bmk.tolist() if r2_val_bmk is not None else None,
        'r2_train_fft':r2_tr_fft.tolist(),
        'null_r2':     null_r2.tolist(),
        'n_total': n_total, 'n_use': n_use,
        'n_train': int(len(idx_tr)), 'n_val': int(len(idx_va)),
        'rs_thresh_train': int(rs_tr), 'rs_thresh_val': int(rs_va),
        'ctx_len': ctx_len,
        'period_mode_train': int(Counter(pl_tr).most_common(1)[0][0]),
        'benchmark_period': BENCHMARK_PERIODS[ds_name],
    }


# ── STEP 1+2: Probes on all datasets ─────────────────────────────────────────

print("=" * 60)
print("STEP 1+2: Hidden states + Ridge probes (normalized labels)")
print("=" * 60)
print()

ds_results = {}
for ds_name in DS_ORDER:
    cfg = DS_CONFIGS[ds_name]
    print(f"── {cfg['label']} ──────────────────────")
    t_ds = time.time()
    ds_results[ds_name] = process_dataset(ds_name, pl_pre)
    print(f"  Wall time: {time.time()-t_ds:.0f}s")
    print()


# ── STEP 3: Control — random backbone on M4 Hourly ───────────────────────────

print("=" * 60)
print("STEP 3: Control — random backbone on M4 Hourly")
print("=" * 60)
print()

pl_rand = load_backbone(random_init=True)
print("  Random init: N(0, 0.02)")
print()

print("── M4 Hourly (random backbone) ──────────")
ctrl_results = process_dataset('m4hourly', pl_rand)
ctrl_results['backbone'] = 'random'
del pl_rand
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()


# ── STEP 4: Aggregate result grid ────────────────────────────────────────────

print("=" * 60)
print("STEP 4: Aggregate result grid")
print("=" * 60)
print()

# Aggregate over 5 datasets using FFT-period probes
agg_r2_fft = np.zeros((NUM_LAYERS, 3))
for ds_name in DS_ORDER:
    agg_r2_fft += np.array(ds_results[ds_name]['r2_val_fft'])
agg_r2_fft /= len(DS_ORDER)

ctrl_r2 = np.array(ctrl_results['r2_val_fft'])

print("  Pretrained — mean R²_val (5 datasets, normalized labels):")
print(f"  {'Layer':>5}  {'Trend':>8}  {'Seasonal':>10}  {'Residual':>10}")
for ℓ in range(NUM_LAYERS):
    print(f"  {ℓ:>5}  {agg_r2_fft[ℓ,0]:>8.4f}  {agg_r2_fft[ℓ,1]:>10.4f}  {agg_r2_fft[ℓ,2]:>10.4f}")

print()
print("  Random backbone — M4H R²_val (normalized labels):")
print(f"  {'Layer':>5}  {'Trend':>8}  {'Seasonal':>10}  {'Residual':>10}")
for ℓ in range(NUM_LAYERS):
    print(f"  {ℓ:>5}  {ctrl_r2[ℓ,0]:>8.4f}  {ctrl_r2[ℓ,1]:>10.4f}  {ctrl_r2[ℓ,2]:>10.4f}")

# Comparison: Old (raw) vs New (normalized) aggregate trend
OLD_AGG_TREND = [
    -0.0725, -0.0502, -0.0317, -0.0396, -0.0176, 0.0174
]  # from job 7405

print()
print("  FIX A comparison — Trend R² (aggregate): raw (job 7405) → normalized (this run)")
for ℓ in range(NUM_LAYERS):
    old = OLD_AGG_TREND[ℓ]
    new = agg_r2_fft[ℓ, 0]
    delta_str = f"{'▲' if new>old else '▼'}{abs(new-old):.4f}"
    print(f"    Layer {ℓ}:  {old:>8.4f}  →  {new:>8.4f}  ({delta_str})")

# FIX B comparison for M4 Daily and Electricity
print()
print("  FIX B comparison — Seasonal R² (M4 Daily, Electricity): FFT vs Benchmark period")
for ds_name in DUAL_STL_DATASETS:
    r2_fft = np.array(ds_results[ds_name]['r2_val_fft'])
    r2_bmk = np.array(ds_results[ds_name]['r2_val_bmk'])
    bmk_p  = ds_results[ds_name]['benchmark_period']
    mode_p = ds_results[ds_name]['period_mode_train']
    label  = DS_CONFIGS[ds_name]['label']
    print(f"  {label}  (fft_mode={mode_p}, benchmark={bmk_p})")
    print(f"  {'Layer':>5}  {'FFT-seasonal':>14}  {'BMK-seasonal':>14}  {'Δ':>8}")
    for ℓ in range(NUM_LAYERS):
        d = r2_bmk[ℓ, 1] - r2_fft[ℓ, 1]
        print(f"  {ℓ:>5}  {r2_fft[ℓ,1]:>14.4f}  {r2_bmk[ℓ,1]:>14.4f}  {d:>+8.4f}")
    print()

# Per-dataset tables
print("  Per-dataset tables (normalized, FFT periods):")
for ds_name in DS_ORDER:
    r2  = np.array(ds_results[ds_name]['r2_val_fft'])
    nul = np.array(ds_results[ds_name]['null_r2'])
    label = DS_CONFIGS[ds_name]['label']
    print(f"  {label}:")
    print(f"  {'Layer':>5}  {'Trend':>8}  {'Seasonal':>10}  {'Residual':>10}")
    for ℓ in range(NUM_LAYERS):
        print(f"  {ℓ:>5}  {r2[ℓ,0]:>8.4f}  {r2[ℓ,1]:>10.4f}  {r2[ℓ,2]:>10.4f}")
    print(f"  {'NULL':>5}  {nul[0]:>8.4f}  {nul[1]:>10.4f}  {nul[2]:>10.4f}")
    print()


# ── STEP 5: Verdict (corrected) ───────────────────────────────────────────────

print("=" * 60)
print("STEP 5: Verdict (corrected)")
print("=" * 60)
print()

THRESHOLD_GOOD = 0.4
THRESHOLD_POOR = 0.2

best_layer_trend    = int(np.argmax(agg_r2_fft[:, 0]))
best_layer_seasonal = int(np.argmax(agg_r2_fft[:, 1]))
best_layer_residual = int(np.argmax(agg_r2_fft[:, 2]))
best_r2_trend    = float(agg_r2_fft[best_layer_trend,    0])
best_r2_seasonal = float(agg_r2_fft[best_layer_seasonal, 1])
best_r2_residual = float(agg_r2_fft[best_layer_residual, 2])

ctrl_best_trend    = float(ctrl_r2[:, 0].max())
ctrl_best_seasonal = float(ctrl_r2[:, 1].max())
ctrl_best_residual = float(ctrl_r2[:, 2].max())

trend_above_ctrl    = best_r2_trend    > ctrl_best_trend    + 0.05
seasonal_above_ctrl = best_r2_seasonal > ctrl_best_seasonal + 0.05
trend_decodable    = best_r2_trend    > THRESHOLD_GOOD and trend_above_ctrl
seasonal_decodable = best_r2_seasonal > THRESHOLD_GOOD and seasonal_above_ctrl
trend_poor         = best_r2_trend    < THRESHOLD_POOR
seasonal_poor      = best_r2_seasonal < THRESHOLD_POOR

if trend_decodable and seasonal_decodable:
    verdict = "OUTCOME_A"
    verdict_text = (
        f"BACKBONE ALREADY DECOMPOSES — surfacing path.\n"
        f"  Trend R²={best_r2_trend:.4f} (layer {best_layer_trend}), "
        f"Seasonal R²={best_r2_seasonal:.4f} (layer {best_layer_seasonal}), "
        f"both >0.4 and clearly above random control."
    )
elif trend_poor and seasonal_poor:
    verdict = "OUTCOME_B"
    verdict_text = (
        f"BACKBONE DOES NOT DECOMPOSE — teaching path.\n"
        f"  Trend R²={best_r2_trend:.4f}, Seasonal R²={best_r2_seasonal:.4f}, "
        f"both <0.2 or not above random control."
    )
else:
    mix_trend    = "DECODABLE" if trend_decodable    else ("POOR" if trend_poor    else "PARTIAL")
    mix_seasonal = "DECODABLE" if seasonal_decodable else ("POOR" if seasonal_poor else "PARTIAL")
    verdict = "MIXED"
    verdict_text = (
        f"MIXED — trend={mix_trend}  seasonal={mix_seasonal}.\n"
        f"  Trend R²={best_r2_trend:.4f} (layer {best_layer_trend}), "
        f"Seasonal R²={best_r2_seasonal:.4f} (layer {best_layer_seasonal}).\n"
        f"  Random ctrl: trend={ctrl_best_trend:.4f}, seasonal={ctrl_best_seasonal:.4f}."
    )

print(f"  VERDICT: {verdict}")
print()
for line in verdict_text.split("\n"):
    print(f"  {line}")
print()

print(f"  Control: trend={ctrl_best_trend:.4f}  seasonal={ctrl_best_seasonal:.4f}  residual={ctrl_best_residual:.4f}")
print(f"  trend_above_ctrl={trend_above_ctrl}  seasonal_above_ctrl={seasonal_above_ctrl}")
print()
print("  STOP — awaiting review before Phase 2.")


# ── Save ─────────────────────────────────────────────────────────────────────

results = {
    'run_config': {
        'smoke_test':   SMOKE_TEST,
        'ridge_alpha':  RIDGE_ALPHA,
        'train_seed':   TRAIN_SEED,
        'train_frac':   TRAIN_FRAC,
        'rs_thresh':    RS_THRESH,
        'num_layers':   NUM_LAYERS,
        'd_model':      D_MODEL,
        'components':   COMPONENTS,
        'ds_order':     DS_ORDER,
        'fix_a':        'labels normalized by ctx_scale = mean(|context|) >= 1.0',
        'fix_b':        'dual STL: FFT-detected + benchmark period for m4daily, electricity',
        'benchmark_periods': BENCHMARK_PERIODS,
    },
    'step0_period_audit': audit_results,
    'datasets':            ds_results,
    'control_random':      ctrl_results,
    'aggregate': {
        'mean_r2_val':       agg_r2_fft.tolist(),
        'best_layer_trend':  best_layer_trend,
        'best_r2_trend':     best_r2_trend,
        'best_layer_seasonal': best_layer_seasonal,
        'best_r2_seasonal':  best_r2_seasonal,
        'best_layer_residual': best_layer_residual,
        'best_r2_residual':  best_r2_residual,
        'ctrl_best_trend':   ctrl_best_trend,
        'ctrl_best_seasonal':ctrl_best_seasonal,
        'ctrl_best_residual':ctrl_best_residual,
        'trend_above_ctrl':  trend_above_ctrl,
        'seasonal_above_ctrl': seasonal_above_ctrl,
    },
    'verdict': {
        'outcome':      verdict,
        'trend_r2':     best_r2_trend,
        'seasonal_r2':  best_r2_seasonal,
        'residual_r2':  best_r2_residual,
        'text':         verdict_text,
    },
    'fix_a_comparison_trend_aggregate': {
        'old_raw_job7405': OLD_AGG_TREND,
        'new_normalized':  [float(agg_r2_fft[ℓ, 0]) for ℓ in range(NUM_LAYERS)],
    },
}

json_path = RESULTS_DIR / "phase1_recheck.json"
with open(str(json_path), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {json_path}")

# Markdown
md = [
    "# Phase 1 Re-check — Normalized Labels + Period Audit",
    "",
    "**FIX A**: Labels normalized by `ctx_scale = mean(|context|) ≥ 1.0`.",
    "**FIX B**: Dual STL (FFT-detected + benchmark period) for M4 Daily and Electricity.",
    "",
    "## STEP 0: Period Audit",
    "",
    "| Dataset | Benchmark | FFT mode | Median | Match? |",
    "|---------|-----------|----------|--------|--------|",
]
for ds_name in DS_ORDER:
    a = audit_results[ds_name]
    md.append(f"| {DS_CONFIGS[ds_name]['label']} | {a['benchmark_period']} | "
              f"{a['mode_period']} | {a['median_period']} | "
              f"{'✓' if a['matches_benchmark'] else '✗ MISMATCH'} |")

md += [
    "",
    "## Aggregate R²_val (normalized, FFT periods)",
    "",
    "| Layer | Trend (new) | Trend (old raw) | Seasonal | Residual |",
    "|-------|-------------|-----------------|----------|----------|",
]
for ℓ in range(NUM_LAYERS):
    md.append(
        f"| {ℓ} | {agg_r2_fft[ℓ,0]:.4f} | {OLD_AGG_TREND[ℓ]:.4f} | "
        f"{agg_r2_fft[ℓ,1]:.4f} | {agg_r2_fft[ℓ,2]:.4f} |"
    )

md += ["", "## FIX B — M4 Daily and Electricity Seasonal (FFT vs Benchmark period)", ""]
for ds_name in sorted(DUAL_STL_DATASETS):
    r2f = np.array(ds_results[ds_name]['r2_val_fft'])
    r2b = np.array(ds_results[ds_name]['r2_val_bmk'])
    mp  = ds_results[ds_name]['period_mode_train']
    bp  = ds_results[ds_name]['benchmark_period']
    md += [
        f"### {DS_CONFIGS[ds_name]['label']}  (FFT mode={mp}, benchmark={bp})",
        "",
        "| Layer | Seas(FFT) | Seas(benchmark) | Δ |",
        "|-------|-----------|-----------------|---|",
    ]
    for ℓ in range(NUM_LAYERS):
        d = r2b[ℓ, 1] - r2f[ℓ, 1]
        md.append(f"| {ℓ} | {r2f[ℓ,1]:.4f} | {r2b[ℓ,1]:.4f} | {d:+.4f} |")
    md.append("")

md += [
    "## Per-dataset Tables (normalized, FFT periods)",
    "",
]
for ds_name in DS_ORDER:
    r2  = np.array(ds_results[ds_name]['r2_val_fft'])
    nul = np.array(ds_results[ds_name]['null_r2'])
    md += [
        f"### {DS_CONFIGS[ds_name]['label']}",
        "",
        "| Layer | Trend | Seasonal | Residual |",
        "|-------|-------|----------|----------|",
    ]
    for ℓ in range(NUM_LAYERS):
        md.append(f"| {ℓ} | {r2[ℓ,0]:.4f} | {r2[ℓ,1]:.4f} | {r2[ℓ,2]:.4f} |")
    md.append(f"| NULL | {nul[0]:.4f} | {nul[1]:.4f} | {nul[2]:.4f} |")
    md.append("")

md += [
    f"## Verdict: {verdict}",
    "",
]
for line in verdict_text.split("\n"):
    md.append(line.strip())

md_path = RESULTS_DIR / "phase1_recheck.md"
with open(str(md_path), "w") as f:
    f.write("\n".join(md) + "\n")
print(f"Saved → {md_path}")
print()
print(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
