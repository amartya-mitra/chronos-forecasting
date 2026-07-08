#!/usr/bin/env python3
"""
phase1_internal_decomposition_probe.py — Phase 1: Internal Decomposition Probe.

Does frozen Chronos-T5-Small already encode trend/seasonal/residual in its
encoder hidden states? No backbone training. Linear probes only.

STEP 0: Tokenization alignment verification.
  Chronos T5-Small: use_eos_token=True, model_type=seq2seq.
  Context length L → token seq_len = L+1 (L context tokens + 1 EOS).
  hidden_states[:, t, :] aligns 1:1 with STL[t] for t=0..L-1.
  Position L (EOS) excluded from probe.

STEP 1: Collect hidden states from ALL 6 encoder blocks + STL ground-truth.
  Datasets: M4 Hourly, M4 Monthly, M4 Daily, M4 Weekly, Electricity.
  Pure pretrained backbone — NO prefix, NO fine-tuned checkpoint.
  Uses encoder(output_hidden_states=True): hidden_states[ℓ+1] = block ℓ output.

STEP 2: Position-wise Ridge probes per (layer, component, dataset).
  X: (N_series × ctx_len, 512),  y: (N_series × ctx_len,) STL component.
  Series-level train/val split: TRAIN_SEED=42, TRAIN_FRAC=0.8.
  StandardScaler on X, Ridge(alpha=1.0), R²_val reported.
  Null baseline: predict training-set mean → R²≈0 is the expected floor.

STEP 3: Control — random (untrained) backbone on M4 Hourly.
  Rules out "large random projection trivially decodes STL" critique.

STEP 4: Result grid.
  Table: 6 layers × 3 components × 5 datasets (val R²).
  Aggregate: mean across datasets per (layer, component).

STEP 5: Verdict.
  OUTCOME A — BACKBONE DECOMPOSES:
    ≥1 layer with R²(trend) > 0.4 AND R²(seasonal) > 0.4, clearly above control.
    → surface path: Phase 2a readout heads, remove STL from inference.
  OUTCOME B — DOES NOT DECOMPOSE:
    All layers: R²(trend) < 0.2 AND R²(seasonal) < 0.2, not above control.
    → teach path: Phase 2b STL distillation.
  MIXED: some components decompose, others don't.
    → per-component; Phase 2 scoped to teach only missing component(s).

Saves: results/phase1_internal_decomposition_probe.json
       results/phase1_internal_decomposition_probe.md
STOP after verdict.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
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

from chronos import ChronosPipeline
from scripts.step1_stl_decompose import detect_periods, stl_decompose
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

SMOKE_TEST  = os.environ.get("SMOKE_TEST", "0") == "1"
TRAIN_SEED  = 42
TRAIN_FRAC  = 0.8
D_MODEL     = 512
NUM_LAYERS  = 6
RS_THRESH   = 0.4
RIDGE_ALPHA = 1.0
BATCH_SIZE  = 16 if not SMOKE_TEST else 4

# Per-dataset series cap (bounds RAM + STL wall time)
_MAX = {
    'm4hourly':   None,  # 414 total — use all
    'm4monthly':  1000,  # 10000 total — subsample
    'm4daily':    2000,  # 4227 total — subsample
    'm4weekly':   None,  # 359 total — use all
    'electricity': None, # 370 total — use all
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  {'SMOKE' if SMOKE_TEST else 'FULL'} run")
print(f"RIDGE_ALPHA={RIDGE_ALPHA}  BATCH_SIZE={BATCH_SIZE}  TRAIN_SEED={TRAIN_SEED}")
print()


# ── STEP 0: Tokenization alignment verification ───────────────────────────────

print("=" * 60)
print("STEP 0: Tokenization alignment verification")
print("=" * 60)
print()

_pl = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small", device_map="cpu", dtype=torch.float32,
)
_tok = _pl.tokenizer
_cfg = _tok.config

print(f"  model_type    : {_cfg.model_type}")
print(f"  use_eos_token : {_cfg.use_eos_token}")
print(f"  eos_token_id  : {_cfg.eos_token_id}")
print()

for _L in [36, 80, 93, 96, 336]:
    _ids, _attn, _ = _tok.context_input_transform(
        _pl._prepare_and_validate_context(torch.randn(1, _L))
    )
    _slen = _ids.shape[1]
    assert _slen == _L + 1, f"seq_len={_slen} ≠ {_L+1} for ctx_len={_L}"
    assert _ids[0, -1].item() == _cfg.eos_token_id, "Last token is not EOS"
    print(f"  ctx_len={_L:3d} → seq_len={_slen}  "
          f"(pos 0..{_L-1} = context, pos {_L} = EOS excluded) ✓")

print()
print("  ALIGNMENT: hidden_states[ℓ+1][:, t, :] ↔ STL[t] for t=0..ctx_len-1 ✓")
print()

del _pl, _tok, _cfg
if device.type == "cuda":
    torch.cuda.empty_cache()
gc.collect()


# ── Load frozen pretrained backbone ───────────────────────────────────────────

def load_backbone(random_init: bool = False):
    """Load Chronos T5-Small. If random_init, randomize all weights after loading arch."""
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


import torch.nn as nn  # needed for random_init path

print("Loading frozen pretrained Chronos T5-Small …")
pl_pre  = load_backbone(random_init=False)
model   = pl_pre.model
tokenizer = pl_pre.tokenizer
n_total_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {n_total_params:,}  FROZEN  (no prefix, no fine-tuned ckpt)")
print()


# ── Core collection: encoder hidden states + STL labels ──────────────────────

def collect_hidden_states_and_labels(cfg, table, series_idx, backbone_pl):
    """
    For each series in series_idx, run frozen encoder and compute STL.

    Returns
    -------
    layer_hiddens : list[6] of np.ndarray (N * ctx_len, D_MODEL)
    labels        : dict  component → np.ndarray (N * ctx_len,)
    n_rs          : int — series where residual/signal > RS_THRESH
    """
    ctx_len = cfg['ctx_len']
    N       = len(series_idx)
    tok     = backbone_pl.tokenizer
    bpl     = backbone_pl

    # Pre-allocate flat arrays (series flattened along axis 0)
    layer_hiddens = [
        np.empty((N * ctx_len, D_MODEL), dtype=np.float32)
        for _ in range(NUM_LAYERS)
    ]
    labels  = {c: np.empty(N * ctx_len, dtype=np.float32) for c in COMPONENTS}
    n_rs    = 0
    t0      = time.time()

    for b_start in range(0, N, BATCH_SIZE):
        b_idx = series_idx[b_start : b_start + BATCH_SIZE]
        B     = len(b_idx)

        # Load raw context values
        ctxs_np = []
        for si in b_idx:
            raw = np.array(table.column(cfg['context_col'])[int(si)].as_py(),
                           dtype=np.float32)
            ctxs_np.append(raw[:ctx_len])

        ctx_batch = torch.tensor(np.stack(ctxs_np), dtype=torch.float32)  # (B, ctx_len)

        # ── Encoder forward: all 6 layers ──────────────────────────────────
        with torch.no_grad():
            inp_ids, attn, _ = tok.context_input_transform(
                bpl._prepare_and_validate_context(ctx_batch)
            )
            enc_out = backbone_pl.model.model.encoder(
                input_ids  = inp_ids.to(device),
                attention_mask = attn.to(device),
                output_hidden_states = True,
            )
        # enc_out.hidden_states: tuple of NUM_LAYERS+1 tensors (B, seq_len, D)
        # hidden_states[0]   = embedding output
        # hidden_states[ℓ+1] = T5Block ℓ output   (ℓ = 0..5)
        # positions 0..ctx_len-1 = context tokens (aligned to STL time steps)
        # position ctx_len = EOS (excluded)
        assert enc_out.hidden_states is not None, "output_hidden_states not returned"
        assert len(enc_out.hidden_states) == NUM_LAYERS + 1, (
            f"Expected {NUM_LAYERS+1} hidden states, got {len(enc_out.hidden_states)}"
        )

        for ℓ in range(NUM_LAYERS):
            h = enc_out.hidden_states[ℓ + 1]  # (B, seq_len, D)
            h = h[:, :ctx_len, :].detach().cpu().numpy()  # (B, ctx_len, D) — EOS excluded
            slot = slice(b_start * ctx_len, (b_start + B) * ctx_len)
            layer_hiddens[ℓ][slot] = h.reshape(B * ctx_len, D_MODEL)

        # ── STL decomposition ───────────────────────────────────────────────
        for i, ctx_np in enumerate(ctxs_np):
            ctx_f64 = ctx_np.astype(np.float64)
            periods  = detect_periods(ctx_f64,
                                      min_period=cfg['min_period'],
                                      max_period=cfg.get('max_period'))
            tr, se, re = stl_decompose(ctx_f64, periods)
            rs = float(np.var(re) / (np.var(ctx_f64) + 1e-9))
            if rs > RS_THRESH:
                n_rs += 1
            pos = (b_start + i) * ctx_len
            labels['trend']   [pos : pos + ctx_len] = tr.astype(np.float32)
            labels['seasonal'][pos : pos + ctx_len] = se.astype(np.float32)
            labels['residual'][pos : pos + ctx_len] = re.astype(np.float32)

        done = b_start + B
        if done % (20 * BATCH_SIZE) == 0 or done >= N:
            print(f"    {done:4d}/{N}  ({time.time()-t0:.0f}s)")

    return layer_hiddens, labels, n_rs


# ── Ridge probes for one dataset ──────────────────────────────────────────────

def run_probes(layer_h_tr, labels_tr, layer_h_va, labels_va, N_tr, N_va, ctx_len):
    """
    Ridge probes per (layer, component).

    Returns
    -------
    r2_val   : (NUM_LAYERS, 3) float  — val R² per (layer, component)
    r2_train : (NUM_LAYERS, 3) float
    null_r2  : (3,) float             — R² of predicting train-mean on val
    """
    r2_val   = np.full((NUM_LAYERS, 3), np.nan)
    r2_train = np.full((NUM_LAYERS, 3), np.nan)
    null_r2  = np.full(3, np.nan)

    for ci, comp in enumerate(COMPONENTS):
        y_tr = labels_tr[comp]   # (N_tr * ctx_len,)
        y_va = labels_va[comp]   # (N_va * ctx_len,)

        # Null baseline: predict train mean on val
        y_mean = float(y_tr.mean())
        null_r2[ci] = float(r2_score(y_va, np.full_like(y_va, y_mean)))

        for ℓ in range(NUM_LAYERS):
            X_tr = layer_h_tr[ℓ]   # (N_tr * ctx_len, 512)
            X_va = layer_h_va[ℓ]   # (N_va * ctx_len, 512)

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_va_s = sc.transform(X_va)

            clf = Ridge(alpha=RIDGE_ALPHA)
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
        perm = perm[:max_n]
        n_use = max_n
    else:
        n_use = n_total

    n_train  = int(n_use * TRAIN_FRAC)
    idx_tr   = perm[:n_train]
    idx_va   = perm[n_train:]
    ctx_len  = cfg['ctx_len']

    print(f"  N_total={n_total}  N_use={n_use}  n_train={len(idx_tr)}  n_val={len(idx_va)}")

    print("  Collecting train hidden states + STL …")
    h_tr, lab_tr, rs_tr = collect_hidden_states_and_labels(cfg, table, idx_tr, backbone_pl)
    print(f"    RS_THRESH exceeded: {rs_tr}/{len(idx_tr)}")

    print("  Collecting val hidden states + STL …")
    h_va, lab_va, rs_va = collect_hidden_states_and_labels(cfg, table, idx_va, backbone_pl)
    print(f"    RS_THRESH exceeded: {rs_va}/{len(idx_va)}")

    print("  Fitting Ridge probes …")
    r2_val, r2_train, null_r2 = run_probes(
        h_tr, lab_tr, h_va, lab_va, len(idx_tr), len(idx_va), ctx_len
    )

    # Print result table
    print()
    print(f"  {'Layer':>5}  {'Trend':>8}  {'Seasonal':>10}  {'Residual':>10}  (val R²)")
    for ℓ in range(NUM_LAYERS):
        print(f"  {ℓ:>5}  {r2_val[ℓ,0]:>8.4f}  {r2_val[ℓ,1]:>10.4f}  {r2_val[ℓ,2]:>10.4f}")
    print(f"  {'NULL':>5}  {null_r2[0]:>8.4f}  {null_r2[1]:>10.4f}  {null_r2[2]:>10.4f}")
    print()

    # Free large arrays immediately
    del h_tr, h_va, lab_tr, lab_va
    gc.collect()

    return {
        'r2_val':   r2_val.tolist(),
        'r2_train': r2_train.tolist(),
        'null_r2':  null_r2.tolist(),
        'n_total':  n_total, 'n_use': n_use,
        'n_train':  int(len(idx_tr)), 'n_val': int(len(idx_va)),
        'rs_thresh_train': int(rs_tr), 'rs_thresh_val': int(rs_va),
        'ctx_len':  ctx_len,
    }


# ── STEP 1+2: Run probes on all datasets ─────────────────────────────────────

print("=" * 60)
print("STEP 1+2: Hidden state collection + Ridge probes (pretrained backbone)")
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
print("STEP 3: Control — random (untrained) backbone on M4 Hourly")
print("=" * 60)
print()

print("Loading random-initialized Chronos T5-Small …")
# Re-use pretrained architecture but scramble weights
pl_rand = load_backbone(random_init=True)
print("  Random init: weights replaced with N(0, 0.02)")
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

# Per-layer, per-component: mean R²_val across 5 datasets
agg_r2 = np.zeros((NUM_LAYERS, 3))
for ds_name in DS_ORDER:
    r2 = np.array(ds_results[ds_name]['r2_val'])
    agg_r2 += r2
agg_r2 /= len(DS_ORDER)

ctrl_r2 = np.array(ctrl_results['r2_val'])  # M4 Hourly, random backbone

print(f"  Pretrained backbone — mean R²_val (across 5 datasets):")
print(f"  {'Layer':>5}  {'Trend':>8}  {'Seasonal':>10}  {'Residual':>10}")
for ℓ in range(NUM_LAYERS):
    print(f"  {ℓ:>5}  {agg_r2[ℓ,0]:>8.4f}  {agg_r2[ℓ,1]:>10.4f}  {agg_r2[ℓ,2]:>10.4f}")

print()
print(f"  Random backbone — R²_val (M4 Hourly control):")
print(f"  {'Layer':>5}  {'Trend':>8}  {'Seasonal':>10}  {'Residual':>10}")
for ℓ in range(NUM_LAYERS):
    print(f"  {ℓ:>5}  {ctrl_r2[ℓ,0]:>8.4f}  {ctrl_r2[ℓ,1]:>10.4f}  {ctrl_r2[ℓ,2]:>10.4f}")

print()

# Per-dataset tables
for ds_name in DS_ORDER:
    r2 = np.array(ds_results[ds_name]['r2_val'])
    null = np.array(ds_results[ds_name]['null_r2'])
    label = DS_CONFIGS[ds_name]['label']
    print(f"  {label}:")
    print(f"  {'Layer':>5}  {'Trend':>8}  {'Seasonal':>10}  {'Residual':>10}")
    for ℓ in range(NUM_LAYERS):
        print(f"  {ℓ:>5}  {r2[ℓ,0]:>8.4f}  {r2[ℓ,1]:>10.4f}  {r2[ℓ,2]:>10.4f}")
    print(f"  {'NULL':>5}  {null[0]:>8.4f}  {null[1]:>10.4f}  {null[2]:>10.4f}")
    print()

# Best layer per component (aggregate)
best_layer_trend    = int(np.argmax(agg_r2[:, 0]))
best_layer_seasonal = int(np.argmax(agg_r2[:, 1]))
best_layer_residual = int(np.argmax(agg_r2[:, 2]))
best_r2_trend    = float(agg_r2[best_layer_trend,    0])
best_r2_seasonal = float(agg_r2[best_layer_seasonal, 1])
best_r2_residual = float(agg_r2[best_layer_residual, 2])

print(f"  Best layer (aggregate):")
print(f"    Trend    → layer {best_layer_trend}   R²={best_r2_trend:.4f}")
print(f"    Seasonal → layer {best_layer_seasonal}   R²={best_r2_seasonal:.4f}")
print(f"    Residual → layer {best_layer_residual}   R²={best_r2_residual:.4f}")

# Control ceiling (M4 Hourly random)
ctrl_best_trend    = float(ctrl_r2[:, 0].max())
ctrl_best_seasonal = float(ctrl_r2[:, 1].max())
ctrl_best_residual = float(ctrl_r2[:, 2].max())
print()
print(f"  Random backbone M4H best R²: "
      f"trend={ctrl_best_trend:.4f}  seasonal={ctrl_best_seasonal:.4f}  residual={ctrl_best_residual:.4f}")
print()

# Patterns check
print("  Layer-wise trends:")
for comp_i, comp in enumerate(COMPONENTS):
    vals = [f"{agg_r2[ℓ, comp_i]:.3f}" for ℓ in range(NUM_LAYERS)]
    print(f"    {comp:>8}: " + "  ".join(vals))
print()


# ── STEP 5: Verdict ───────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 5: Verdict")
print("=" * 60)
print()

THRESHOLD_GOOD  = 0.4   # R² > this → component clearly decodable
THRESHOLD_POOR  = 0.2   # R² < this → component not decodable

trend_above_ctrl    = best_r2_trend    > ctrl_best_trend    + 0.05
seasonal_above_ctrl = best_r2_seasonal > ctrl_best_seasonal + 0.05

trend_decodable    = best_r2_trend    > THRESHOLD_GOOD and trend_above_ctrl
seasonal_decodable = best_r2_seasonal > THRESHOLD_GOOD and seasonal_above_ctrl
trend_poor         = best_r2_trend    < THRESHOLD_POOR
seasonal_poor      = best_r2_seasonal < THRESHOLD_POOR

if trend_decodable and seasonal_decodable:
    verdict = "OUTCOME_A"
    verdict_text = (
        "BACKBONE ALREADY DECOMPOSES — surfacing path.\n"
        f"  Trend R²={best_r2_trend:.4f} (layer {best_layer_trend}) and "
        f"Seasonal R²={best_r2_seasonal:.4f} (layer {best_layer_seasonal}), "
        f"both > {THRESHOLD_GOOD} and clearly above random control.\n"
        "  Phase 2a: build lightweight READOUT heads on best layer(s),\n"
        "  remove STL from inference path, verify readout-based prefix\n"
        "  matches or beats STL-injected prefix on forecast quality."
    )
elif trend_poor and seasonal_poor:
    verdict = "OUTCOME_B"
    verdict_text = (
        "BACKBONE DOES NOT DECOMPOSE — teaching path.\n"
        f"  Trend R²={best_r2_trend:.4f} and Seasonal R²={best_r2_seasonal:.4f}, "
        f"both < {THRESHOLD_POOR} or not above random control.\n"
        "  Phase 2b: STL-distilled internal decomposition — backbone genuinely\n"
        "  lacks this structure and needs to be taught."
    )
else:
    verdict = "MIXED"
    mix_trend    = "DECODABLE" if trend_decodable    else ("POOR" if trend_poor    else "PARTIAL")
    mix_seasonal = "DECODABLE" if seasonal_decodable else ("POOR" if seasonal_poor else "PARTIAL")
    verdict_text = (
        f"MIXED — trend={mix_trend}  seasonal={mix_seasonal}.\n"
        f"  Trend R²={best_r2_trend:.4f} (layer {best_layer_trend}), "
        f"Seasonal R²={best_r2_seasonal:.4f} (layer {best_layer_seasonal}).\n"
        "  Phase 2 scoped per-component: teach only the missing component(s),\n"
        "  reuse frozen backbone representations for those already present."
    )

print(f"  VERDICT: {verdict}")
print()
for line in verdict_text.split("\n"):
    print(f"  {line}")
print()
print("  STOP — awaiting review before proceeding to Phase 2.")


# ── Save results ──────────────────────────────────────────────────────────────

results = {
    'run_config': {
        'smoke_test': SMOKE_TEST,
        'ridge_alpha': RIDGE_ALPHA,
        'batch_size': BATCH_SIZE,
        'train_seed': TRAIN_SEED,
        'train_frac': TRAIN_FRAC,
        'rs_thresh': RS_THRESH,
        'num_layers': NUM_LAYERS,
        'd_model': D_MODEL,
        'components': COMPONENTS,
        'ds_order': DS_ORDER,
        'max_series_per_ds': {k: _MAX[k] for k in DS_ORDER},
        'backbone': 'amazon/chronos-t5-small (pretrained, frozen, no prefix)',
        'alignment': 'hidden_states[ℓ+1][:, t, :] ↔ STL[t] for t=0..ctx_len-1; EOS excluded',
    },
    'step0_alignment': {
        'model_type': 'seq2seq',
        'use_eos_token': True,
        'mapping': {str(L): f'ctx_len={L} → seq_len={L+1}, pos 0..{L-1}=context, pos {L}=EOS'
                    for L in [36, 80, 93, 96, 336]},
    },
    'datasets': ds_results,
    'control_random_backbone': ctrl_results,
    'aggregate': {
        'mean_r2_val_across_datasets': agg_r2.tolist(),
        'best_layer_trend':    best_layer_trend,
        'best_layer_seasonal': best_layer_seasonal,
        'best_layer_residual': best_layer_residual,
        'best_r2_trend':    best_r2_trend,
        'best_r2_seasonal': best_r2_seasonal,
        'best_r2_residual': best_r2_residual,
        'ctrl_best_trend':    ctrl_best_trend,
        'ctrl_best_seasonal': ctrl_best_seasonal,
        'ctrl_best_residual': ctrl_best_residual,
        'trend_above_ctrl':    trend_above_ctrl,
        'seasonal_above_ctrl': seasonal_above_ctrl,
    },
    'verdict': {
        'outcome': verdict,
        'trend_r2': best_r2_trend,
        'seasonal_r2': best_r2_seasonal,
        'residual_r2': best_r2_residual,
        'text': verdict_text,
    },
}

json_path = RESULTS_DIR / "phase1_internal_decomposition_probe.json"
with open(str(json_path), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {json_path}")

# ── Markdown report ───────────────────────────────────────────────────────────

md_lines = [
    "# Phase 1 — Internal Decomposition Probe",
    "",
    "**Question**: Does frozen Chronos-T5-Small already encode trend/seasonal/residual "
    "in its encoder hidden states? (No training, linear probes only.)",
    "",
    "**Backbone**: `amazon/chronos-t5-small` pretrained, fully frozen, no prefix injection.",
    "",
    f"**Alignment**: context length L → seq_len L+1 (EOS appended). "
    "Position t=0..L-1 in hidden states aligns 1:1 with STL time step t. EOS excluded.",
    "",
    f"**Ridge probe**: `Ridge(alpha={RIDGE_ALPHA})`, features StandardScaled, "
    "series-level 80/20 split (`TRAIN_SEED={TRAIN_SEED}`).",
    "",
    "## Aggregate R²_val (mean across 5 datasets)",
    "",
    "| Layer | Trend | Seasonal | Residual |",
    "|-------|-------|----------|----------|",
]
for ℓ in range(NUM_LAYERS):
    md_lines.append(
        f"| {ℓ} | {agg_r2[ℓ,0]:.4f} | {agg_r2[ℓ,1]:.4f} | {agg_r2[ℓ,2]:.4f} |"
    )
md_lines += [
    "",
    f"**Best aggregate**: trend=**{best_r2_trend:.4f}** (layer {best_layer_trend}), "
    f"seasonal=**{best_r2_seasonal:.4f}** (layer {best_layer_seasonal}), "
    f"residual=**{best_r2_residual:.4f}** (layer {best_layer_residual}).",
    "",
    f"**Random backbone control** (M4 Hourly): "
    f"trend={ctrl_best_trend:.4f}, seasonal={ctrl_best_seasonal:.4f}, "
    f"residual={ctrl_best_residual:.4f}.",
    "",
    "## Per-dataset tables",
    "",
]

for ds_name in DS_ORDER:
    r2  = np.array(ds_results[ds_name]['r2_val'])
    nul = np.array(ds_results[ds_name]['null_r2'])
    info = ds_results[ds_name]
    label = DS_CONFIGS[ds_name]['label']
    md_lines += [
        f"### {label}",
        f"N_use={info['n_use']} (train={info['n_train']}, val={info['n_val']}), "
        f"ctx_len={info['ctx_len']}, RS_THRESH exceeded: {info['rs_thresh_train']+info['rs_thresh_val']}",
        "",
        "| Layer | Trend | Seasonal | Residual |",
        "|-------|-------|----------|----------|",
    ]
    for ℓ in range(NUM_LAYERS):
        md_lines.append(
            f"| {ℓ} | {r2[ℓ,0]:.4f} | {r2[ℓ,1]:.4f} | {r2[ℓ,2]:.4f} |"
        )
    md_lines += [
        f"| NULL | {nul[0]:.4f} | {nul[1]:.4f} | {nul[2]:.4f} |",
        "",
    ]

md_lines += [
    "## Control: Random Backbone (M4 Hourly)",
    "",
    "| Layer | Trend | Seasonal | Residual |",
    "|-------|-------|----------|----------|",
]
for ℓ in range(NUM_LAYERS):
    md_lines.append(
        f"| {ℓ} | {ctrl_r2[ℓ,0]:.4f} | {ctrl_r2[ℓ,1]:.4f} | {ctrl_r2[ℓ,2]:.4f} |"
    )
md_lines += [
    "",
    f"## Verdict: {verdict}",
    "",
]
for line in verdict_text.split("\n"):
    md_lines.append(line.strip())

md_path = RESULTS_DIR / "phase1_internal_decomposition_probe.md"
with open(str(md_path), "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Saved → {md_path}")
print()
print("STOP — awaiting review before any Phase 2 work.")
print(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
