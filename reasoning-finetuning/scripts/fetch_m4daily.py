#!/usr/bin/env python3
"""
fetch_m4daily.py — Fetch M4 Daily from GiftEval HuggingFace repo and audit.

Downloads:
  hf://datasets/Salesforce/GiftEval/m4_daily/data-00000-of-00001.arrow

Saves to:
  reasoning-finetuning/data/m4daily-tsf.arrow

Then prints a full audit report:
  - Number of series
  - Column names
  - context length (all series)
  - target length (all series)
  - target encoding: target[0:3] and target[-3:]
  - First 3 context values of series 0 (raw scale check)
  - Minimum context length
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

SAVE_PATH = FINETUNING_ROOT / "data" / "m4daily-tsf.arrow"
HF_URL    = "hf://datasets/Salesforce/GiftEval/m4_daily/data-00000-of-00001.arrow"

print("=" * 70)
print("FETCH + AUDIT — M4 Daily (GiftEval)")
print("=" * 70)

# ── Download ───────────────────────────────────────────────────────────────────

print(f"\nDownloading from {HF_URL} …")
ds = datasets.load_dataset(
    "arrow",
    data_files=HF_URL,
    split="train",
)
print(f"  Downloaded {len(ds)} rows")

# ── Save as Arrow IPC file ─────────────────────────────────────────────────────

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
table = ds.data.table
with ipc.new_file(str(SAVE_PATH), table.schema) as writer:
    writer.write_table(table)
print(f"  Saved → {SAVE_PATH}")

# ── Audit ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("AUDIT REPORT")
print("=" * 70)

n_series = len(table)
cols     = table.schema.names
print(f"\n  Number of series:  {n_series}")
print(f"  Column names:      {cols}")

# Inspect context and target lengths
ctx_col = "context"
tgt_col = "target"

ctx_lens = [len(table.column(ctx_col)[i].as_py()) for i in range(n_series)]
tgt_lens = [len(table.column(tgt_col)[i].as_py()) for i in range(n_series)]

ctx_unique = sorted(set(ctx_lens))
tgt_unique = sorted(set(tgt_lens))

print(f"\n  Context lengths:   min={min(ctx_lens)}  max={max(ctx_lens)}  "
      f"unique={ctx_unique if len(ctx_unique) <= 5 else str(ctx_unique[:5]) + '...'}")
print(f"  Target lengths:    min={min(tgt_lens)}  max={max(tgt_lens)}  "
      f"unique={tgt_unique if len(tgt_unique) <= 5 else str(tgt_unique[:5]) + '...'}")

# Series 0 inspection
ctx_0 = np.array(table.column(ctx_col)[0].as_py(), dtype=np.float32)
tgt_0 = np.array(table.column(tgt_col)[0].as_py(), dtype=np.float32)

print(f"\n  Series 0 context length:  {len(ctx_0)}")
print(f"  Series 0 target  length:  {len(tgt_0)}")
print(f"  Series 0 context[:3]:     {ctx_0[:3].tolist()}")
print(f"  Series 0 context[-3:]:    {ctx_0[-3:].tolist()}")
print(f"  Series 0 target[ 0: 3]:   {tgt_0[0:3].tolist()}")
print(f"  Series 0 target[-3:  ]:   {tgt_0[-3:].tolist()}")

# Check all unique lengths more carefully
if len(ctx_unique) == 1 and len(tgt_unique) == 1:
    ctx_len = ctx_unique[0]
    tgt_len = tgt_unique[0]
    print(f"\n  ✓ All series have uniform lengths: context={ctx_len}, target={tgt_len}")
else:
    ctx_len = min(ctx_lens)
    tgt_len = min(tgt_lens)
    print(f"\n  ⚠ Non-uniform lengths! Min context={ctx_len}, min target={tgt_len}")

# Scale check
ctx_absmax = float(np.abs(ctx_0).max())
print(f"\n  Raw scale check:  |ctx_0|_max = {ctx_absmax:.4f}")
if ctx_absmax > 1.0:
    print(f"  ✓ Raw scale confirmed (not unit-normalised)")
else:
    print(f"  ⚠ Scale looks normalised — verify this is raw")

# Determine target_offset
# For GiftEval: target = [auxiliary_forecast | raw_future]
# For M4H: context_len=96, target has 144 pts, offset=96, pred_len=48
# For M4Monthly: context_len=36, target has 54 pts, offset=36, pred_len=18
# Pattern: offset = context_len, pred_len = total_target - context_len
# But let's check if it's just raw target or decomposed format
# Check if target[:ctx_len] looks like trend/seasonal forecast or raw data
if tgt_len > ctx_len:
    pred_len_guess = tgt_len - ctx_len
    print(f"\n  Inferred target structure:")
    print(f"    If target_offset = ctx_len ({ctx_len}):")
    print(f"      → pred_len = {pred_len_guess}")
    print(f"      → target[:{ctx_len}] = auxiliary_forecast")
    print(f"      → target[{ctx_len}:] = raw_future ({pred_len_guess} pts)")
    fut_guess = tgt_0[ctx_len:]
    print(f"      → target[{ctx_len}:{ctx_len+3}] = {fut_guess[:3].tolist()}")
    print(f"      → target[-3:]          = {tgt_0[-3:].tolist()}")
elif tgt_len == ctx_len:
    print(f"\n  ⚠ target_len == context_len — possibly raw future only, check manually")
else:
    print(f"\n  ⚠ target_len ({tgt_len}) < context_len ({ctx_len}) — unexpected!")

print(f"\n  Minimum context length: {min(ctx_lens)} (need >= CONTEXT_LENGTH + some buffer)")

print("\n" + "=" * 70)
print("FETCH + AUDIT COMPLETE")
print("=" * 70)
print(f"\nFile saved to: {SAVE_PATH}")
print(f"File size:     {SAVE_PATH.stat().st_size / 1024 / 1024:.2f} MB")
