#!/usr/bin/env python3
"""
fetch_m4weekly.py — Fetch M4 Weekly from GiftEval HuggingFace repo, audit, preprocess.

Downloads:
  hf://datasets/Salesforce/GiftEval/m4_weekly/data-00000-of-00001.arrow

Raw GluonTS format: columns ['item_id', 'start', 'freq', 'target']
  'target' = full time series (variable length)

Preprocessing:
  pred_len = 13  (M4 weekly standard forecast horizon)
  context_len = auto-chosen: 104 if >= 90% of series survive, else 80, else 52
  Each row → context = series[-context_len-pred_len:-pred_len]
           → target  = series[-pred_len:]
  Series shorter than context_len + pred_len are filtered out.

Saves to:
  reasoning-finetuning/data/m4weekly-tsf.arrow

Audit report:
  - Number of series (raw / after filter)
  - Series length distribution
  - CRITICAL: context_length / 52 (cycles available for period=52 STL)
  - Chosen context_len and pred_len
  - Scale check
"""

import sys
from pathlib import Path

REPO_ROOT       = Path(__file__).parent.parent.parent
FINETUNING_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

SAVE_PATH = FINETUNING_ROOT / "data" / "m4weekly-tsf.arrow"
HF_URL    = "hf://datasets/Salesforce/GiftEval/m4_weekly/data-00000-of-00001.arrow"
PRED_LEN  = 13   # M4 Weekly standard forecast horizon

print("=" * 70)
print("FETCH + AUDIT — M4 Weekly (GiftEval, raw GluonTS format)")
print("=" * 70)

# ── Download ───────────────────────────────────────────────────────────────────

print(f"\nDownloading from {HF_URL} …")
ds = datasets.load_dataset(
    "arrow",
    data_files=HF_URL,
    split="train",
)
print(f"  Downloaded {len(ds)} rows")
print(f"  Columns: {ds.column_names}")

raw_table   = ds.data.table
n_raw       = len(raw_table)
series_lens = []

for i in range(n_raw):
    tgt = raw_table.column("target")[i].as_py()
    series_lens.append(len(tgt) if tgt is not None else 0)

series_lens = np.array(series_lens)

print("\n" + "=" * 70)
print("RAW AUDIT — SERIES LENGTHS")
print("=" * 70)
print(f"\n  Total raw series:  {n_raw}")
print(f"  Min length:        {series_lens.min()}")
print(f"  Max length:        {series_lens.max()}")
print(f"  Mean length:       {series_lens.mean():.1f}")
print(f"  Median length:     {np.median(series_lens):.1f}")
print(f"  p10 length:        {np.percentile(series_lens, 10):.1f}")
print(f"  p25 length:        {np.percentile(series_lens, 25):.1f}")
print(f"  p75 length:        {np.percentile(series_lens, 75):.1f}")
print(f"  p90 length:        {np.percentile(series_lens, 90):.1f}")

# ── Context length selection ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("CONTEXT LENGTH SELECTION  (n_cycles=1: max_period = context_len)")
print("=" * 70)

for candidate in [104, 80, 65, 52]:
    need = candidate + PRED_LEN
    survive = int(np.sum(series_lens >= need))
    pct = survive / n_raw * 100
    cycles = candidate / 52.0
    flag = "✓" if pct >= 90.0 else "⚠"
    print(f"  context_len={candidate:>3}  (need>={need})  "
          f"survive={survive}/{n_raw} ({pct:.1f}%)  "
          f"cycles={cycles:.2f}  {flag}")

# Auto-choose
if np.sum(series_lens >= 104 + PRED_LEN) >= n_raw * 0.9:
    CONTEXT_LEN = 104
elif np.sum(series_lens >= 80 + PRED_LEN) >= n_raw * 0.9:
    CONTEXT_LEN = 80
elif np.sum(series_lens >= 65 + PRED_LEN) >= n_raw * 0.9:
    CONTEXT_LEN = 65
else:
    CONTEXT_LEN = 52

cycles_avail = CONTEXT_LEN / 52.0

print(f"\n  AUTO-CHOSEN context_len = {CONTEXT_LEN}")
print(f"\n  ╔══════════════════════════════════════════════════════╗")
print(f"  ║  CYCLES AVAILABLE = {cycles_avail:.2f}  (context={CONTEXT_LEN} / period=52)  ║")
if cycles_avail < 2.0:
    print(f"  ║  ⚠  < 2 complete cycles — STL unreliable for period=52  ║")
    print(f"  ║     Using n_cycles=1: max_period = {CONTEXT_LEN}                 ║")
else:
    print(f"  ║  ✓  >= 2 complete cycles — STL reliable for period=52   ║")
print(f"  ╚══════════════════════════════════════════════════════╝")

# ── Preprocess: extract context / target windows ───────────────────────────────

print("\n" + "=" * 70)
print(f"PREPROCESSING  (context={CONTEXT_LEN}, pred_len={PRED_LEN}, target_offset=0)")
print("=" * 70)

MIN_TOTAL = CONTEXT_LEN + PRED_LEN
contexts = []
targets  = []
n_skip   = 0

for i in range(n_raw):
    series = np.array(raw_table.column("target")[i].as_py(), dtype=np.float32)
    if len(series) < MIN_TOTAL:
        n_skip += 1
        continue
    ctx = series[-(CONTEXT_LEN + PRED_LEN):-PRED_LEN]
    tgt = series[-PRED_LEN:]
    contexts.append(ctx.tolist())
    targets.append(tgt.tolist())

n_kept = len(contexts)
print(f"\n  Raw series:    {n_raw}")
print(f"  Too short (<{MIN_TOTAL}): {n_skip}  filtered out")
print(f"  Kept:          {n_kept}  ({n_kept/n_raw*100:.1f}%)")

# Audit the preprocessed data
ctx_0 = np.array(contexts[0], dtype=np.float32)
tgt_0 = np.array(targets[0],  dtype=np.float32)

assert len(ctx_0) == CONTEXT_LEN, f"BUG: context length {len(ctx_0)} != {CONTEXT_LEN}"
assert len(tgt_0) == PRED_LEN,    f"BUG: target length {len(tgt_0)} != {PRED_LEN}"

print(f"\n  Series 0 context length:  {len(ctx_0)} pts  ✓")
print(f"  Series 0 target  length:  {len(tgt_0)} pts  ✓")
print(f"  Series 0 context[:3]:     {ctx_0[:3].tolist()}")
print(f"  Series 0 context[-3:]:    {ctx_0[-3:].tolist()}")
print(f"  Series 0 target[ 0: 3]:   {tgt_0[0:3].tolist()}")
print(f"  Series 0 target[-3:  ]:   {tgt_0[-3:].tolist()}")

ctx_absmax = float(np.abs(ctx_0).max())
print(f"\n  Raw scale check:  |ctx_0|_max = {ctx_absmax:.4f}")
if ctx_absmax > 1.0:
    print(f"  ✓ Raw scale confirmed (not unit-normalised)")
else:
    print(f"  ⚠ Scale looks normalised — verify this is raw")

# ── Save as Arrow IPC file ─────────────────────────────────────────────────────

ctx_arr = pa.array(contexts, type=pa.list_(pa.float32()))
tgt_arr = pa.array(targets,  type=pa.list_(pa.float32()))
out_table = pa.table({"context": ctx_arr, "target": tgt_arr})

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
with ipc.new_file(str(SAVE_PATH), out_table.schema) as writer:
    writer.write_table(out_table)

print(f"\n  Saved → {SAVE_PATH}")
print(f"  File size: {SAVE_PATH.stat().st_size / 1024 / 1024:.2f} MB")

# ── Final summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("FETCH + AUDIT SUMMARY")
print("=" * 70)
print(f"\n  Series:          {n_kept} (of {n_raw} raw; {n_skip} filtered)")
print(f"  Context length:  {CONTEXT_LEN} pts")
print(f"  Target length:   {PRED_LEN} pts")
print(f"  Target offset:   0  (future = target[:], no aux prefix)")
print(f"  Cycles (52):     {cycles_avail:.2f}  {'✓ >= 2' if cycles_avail >= 2.0 else '⚠ < 2 (flagged)'}")
print(f"  max_period:      {CONTEXT_LEN}  (n_cycles=1, covers period=52)")
print(f"  Scale range:     raw (O(100–100000) typical for economic weekly)")
print()
print(f"  DATASET_CONFIG entry (add to train_ddp.py):")
print(f"    'm4weekly': {{")
print(f"        'data_path':    'data/m4weekly-tsf.arrow',")
print(f"        'context_col':  'context',")
print(f"        'forecast_col': 'target',")
print(f"        'target_offset': 0,")
print(f"        'min_period':   4,")
print(f"        'max_period':   {CONTEXT_LEN},  # n_cycles=1")
print(f"        'batch_size':   4,")
print(f"        'default_steps': 500,")
print(f"        'ckpt_steps':   {{500}},")
print(f"        'ckpt_names':   {{500: 'sweep-m4weekly-500steps.pt'}},")
print(f"        'diag_milestones': [0, 100, 250, 500],")
print(f"        'weight_decay': 0.05,")
print(f"        'warmup_steps': 30,")
print(f"    }}")
print()
print("=" * 70)
print("DONE")
print("=" * 70)
