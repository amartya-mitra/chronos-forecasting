#!/usr/bin/env python3
"""
fetch_electricity.py — Fetch Electricity (hourly) from GiftEval HuggingFace repo, audit, preprocess.

Step 1: discover the correct file path by listing Salesforce/GiftEval repo files.
        Filters for any path containing 'electr' (handles electricity, electricity_hourly, elec).
        If multiple found, prefer the one containing 'hourly' (period=24).

Raw GluonTS format: columns ['item_id', 'start', 'freq', 'target']
  'target' = full time series (variable length, typically ~26k pts for UCI electricity)

Preprocessing:
  pred_len = 24  (1 day ahead — standard for hourly electricity)
  context_len = auto-chosen: 336 if >= 90% survive, else 168, else 96, else 72
  Each row → context = series[-context_len-pred_len:-pred_len]
           → target  = series[-pred_len:]
  Series shorter than context_len + pred_len are filtered.

Saves to:
  reasoning-finetuning/data/electricity-tsf.arrow

Audit report:
  - Number of series (raw / after filter)
  - Series length distribution
  - CRITICAL: context_length / 24 (cycles of dominant daily period)
  - Chosen context_len and pred_len
  - Scale check (electricity: O(100–5000) kWh typical)
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
from huggingface_hub import list_repo_files

SAVE_PATH = FINETUNING_ROOT / "data" / "electricity-tsf.arrow"
PRED_LEN  = 24   # 1 day ahead — standard for hourly electricity

print("=" * 70)
print("FETCH + AUDIT — Electricity hourly (GiftEval, raw GluonTS format)")
print("=" * 70)

# ── Discover the correct path in GiftEval ──────────────────────────────────────

print("\nListing Salesforce/GiftEval repo files …")
all_files   = list(list_repo_files("Salesforce/GiftEval", repo_type="dataset"))
elec_files  = [f for f in all_files if 'electr' in f.lower() and f.endswith('.arrow')]
print(f"\n  All electricity-related .arrow files found:")
if not elec_files:
    print("  ⚠  NO files containing 'electr' found!")
    print("  Available .arrow files:")
    for f in sorted(all_files):
        if f.endswith('.arrow'):
            print(f"    {f}")
    raise FileNotFoundError("No electricity dataset found in Salesforce/GiftEval")

for f in elec_files:
    print(f"    {f}")

# GiftEval uses pandas freq codes: H=hourly, 15T=15-min, D=daily, W=weekly.
# Prefer /H/ (hourly); also accept paths containing 'hourly'. Prefer data- over cache-.
hourly_files = [f for f in elec_files if '/H/' in f or 'hourly' in f.lower()]
if not hourly_files:
    # Fallback: exclude 15-min and sub-hourly frequencies
    hourly_files = [f for f in elec_files if '/15T/' not in f and '/30T/' not in f
                    and '/5T/' not in f and '/1T/' not in f]
# Among candidates, prefer data files over cache files
data_files  = [f for f in hourly_files if '/data-' in f]
chosen_file = data_files[0] if data_files else (hourly_files[0] if hourly_files else elec_files[0])
HF_URL = f"hf://datasets/Salesforce/GiftEval/{chosen_file}"
print(f"\n  Using: {HF_URL}")

# ── Download ────────────────────────────────────────────────────────────────────

print(f"\nDownloading …")
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

# ── Minimum series check ───────────────────────────────────────────────────────

n_train_80pct = int(n_raw * 0.8)
if n_train_80pct < 100:
    print(f"\n  ⚠ INSUFFICIENT SERIES: {n_train_80pct} train (< 100 threshold)")
    print(f"     Consider switching to Traffic (~862 series) — report back before training.")

# ── Context length selection ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("CONTEXT LENGTH SELECTION  (n_cycles=4: max_period = context_len // 4)")
print("=" * 70)
print(f"  Target: >= 4 complete cycles of period=24 (daily).")
print(f"  Minimum: 96 pts = 4 days. Flag if cycles < 2.")
print()

for candidate in [336, 168, 96, 72]:
    need       = candidate + PRED_LEN
    survive    = int(np.sum(series_lens >= need))
    pct        = survive / n_raw * 100
    cycles_24  = candidate / 24.0
    max_period = candidate // 4
    flag       = "✓" if pct >= 90.0 else "⚠"
    flag_c     = "✓" if cycles_24 >= 4.0 else ("⚠" if cycles_24 >= 2.0 else "✗")
    print(f"  context_len={candidate:>4}  (need>={need:>4})  "
          f"survive={survive:>4}/{n_raw} ({pct:.1f}%)  "
          f"cycles={cycles_24:.1f} {flag_c}  max_period={max_period}  {flag}")

# Auto-choose: first candidate with >= 90% survival
if np.sum(series_lens >= 336 + PRED_LEN) >= n_raw * 0.9:
    CONTEXT_LEN = 336
elif np.sum(series_lens >= 168 + PRED_LEN) >= n_raw * 0.9:
    CONTEXT_LEN = 168
elif np.sum(series_lens >= 96 + PRED_LEN) >= n_raw * 0.9:
    CONTEXT_LEN = 96
else:
    CONTEXT_LEN = 72

cycles_avail = CONTEXT_LEN / 24.0
max_period   = CONTEXT_LEN // 4

print(f"\n  AUTO-CHOSEN context_len = {CONTEXT_LEN}")
print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
print(f"  ║  CYCLES AVAILABLE = {cycles_avail:.1f}  (context={CONTEXT_LEN} / period=24)       ║")
if cycles_avail < 2.0:
    print(f"  ║  ✗  < 2 complete cycles — STL unreliable for period=24      ║")
    print(f"  ║     Consider reducing pred_len or switching dataset          ║")
elif cycles_avail < 4.0:
    print(f"  ║  ⚠  < 4 cycles — usable but sub-optimal                    ║")
    print(f"  ║     n_cycles=4: max_period = {max_period:>3}                          ║")
else:
    print(f"  ║  ✓  >= 4 complete cycles — STL reliable for period=24       ║")
    print(f"  ║     n_cycles=4: max_period = {max_period:>3}                          ║")
print(f"  ╚══════════════════════════════════════════════════════════════╝")

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

# Audit preprocessed data
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

ctx_mean  = float(np.abs(ctx_0).mean())
ctx_max   = float(np.abs(ctx_0).max())
print(f"\n  Scale check:  mean|ctx_0|={ctx_mean:.2f}  max|ctx_0|={ctx_max:.2f}")
if ctx_max > 1.0:
    print(f"  ✓ Raw scale confirmed (electricity: O(100–5000) kWh typical)")
else:
    print(f"  ⚠ Scale looks normalised — verify this is raw")

n_train = int(n_kept * 0.8)
n_val   = n_kept - n_train
if n_train < 100:
    print(f"\n  ⚠ WARNING: only {n_train} training series after 80/20 split")
    print(f"     Consider switching to Traffic (~862 series) if this is too few")
else:
    print(f"\n  80/20 split: {n_train} train / {n_val} val  "
          f"{'✓' if n_train >= 100 else '⚠'}")

# ── Save as Arrow IPC file ─────────────────────────────────────────────────────

ctx_arr   = pa.array(contexts, type=pa.list_(pa.float32()))
tgt_arr   = pa.array(targets,  type=pa.list_(pa.float32()))
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
print(f"\n  Source:          {chosen_file}")
print(f"  Series:          {n_kept} (of {n_raw} raw; {n_skip} filtered)")
print(f"  Context length:  {CONTEXT_LEN} pts")
print(f"  Target length:   {PRED_LEN} pts")
print(f"  Target offset:   0  (future = target[:], no aux prefix)")
print(f"  Cycles (24):     {cycles_avail:.1f}  {'✓ >= 4' if cycles_avail >= 4.0 else ('⚠ < 4' if cycles_avail >= 2.0 else '✗ < 2 (critical!)')}")
print(f"  max_period:      {max_period}  (n_cycles=4: {CONTEXT_LEN} // 4)")
print(f"  Scale range:     raw (O(100–5000) typical for kWh electricity)")
print(f"  Train / Val:     {n_train} / {n_val}")
print()

wd   = 0.05 if n_train < 500 else 0.01
gpus = 1    if n_train < 500 else 2
steps = 500 if n_train < 500 else 1000
bs   = 4    if n_train < 500 else 8
ws   = 30   if steps == 500 else 50

print(f"  Training params:  N_train={n_train} {'< 500' if n_train < 500 else '>= 500'}")
print(f"    TOTAL_STEPS={steps}, batch_size={bs}/GPU, {gpus} GPU(s), weight_decay={wd}")
print()
print(f"  DATASET_CONFIG entry (add to train_ddp.py):")
print(f"    'electricity': {{")
print(f"        'data_path':       'data/electricity-tsf.arrow',")
print(f"        'context_col':     'context',")
print(f"        'forecast_col':    'target',")
print(f"        'target_offset':   0,")
print(f"        'min_period':      6,")
print(f"        'max_period':      {max_period},  # n_cycles=4: {CONTEXT_LEN} // 4")
print(f"        'batch_size':      {bs},")
print(f"        'default_steps':   {steps},")
print(f"        'ckpt_steps':      {{{steps}}},")
print(f"        'ckpt_names':      {{{steps}: 'sweep-electricity-{steps}steps.pt'}},")
print(f"        'diag_milestones': [0, 100, 250, {steps}],")
print(f"        'weight_decay':    {wd},")
print(f"        'warmup_steps':    {ws},")
print(f"    }}")
print()
print("=" * 70)
print("DONE")
print("=" * 70)
