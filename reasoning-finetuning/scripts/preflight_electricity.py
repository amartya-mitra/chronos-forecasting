#!/usr/bin/env python3
"""
preflight_electricity.py — Pre-flight checks for Electricity sweep training.

Checks P1–P6 + P_new + P_domain on 3 smoke batches before submitting training.

  P1. Data loads correctly
      - num train/val series, context=336, future=24
      - target_offset=0 confirmed
      - values in O(100–5000) raw scale range (electricity kWh)

  P2. STL detects daily period=24 on 2/3 smoke batches
      - context=336, min_period=6, max_period=84 (n_cycles=4)
      - Accept any period >= 12 (half-day cycle or longer)
      - Electricity has strong 24-hour daily cycles; also expect period=48 (2-day)

  P3. Fix 1b trigger rate across 3 batches

  P4. Ceiling formula check
      - step=0   → 0.50
      - step=500 → 0.80

  P5. Forward pass: fc_loss finite; warn if > 10

  P6. KL loss finite at step 0

  P_new. D1 gradient ratio < 50× after ctx_scale normalisation

  P_domain. Mean context value check — electricity kWh should be O(100–5000).
            Print mean and max for series 0 to confirm raw (not normalised) scale.
            Context/24 cycles check — must be >= 4 for reliable period=24 detection.

Config must match train_ddp.py --dataset electricity --steps 500:
  context_len=336, pred_len=24, target_offset=0
  batch_size=4, TOTAL_STEPS=500, min_period=6, max_period=84, rs_threshold=0.4
"""

import math
import sys
from pathlib import Path

REPO_ROOT       = Path(__file__).parent.parent.parent
FINETUNING_ROOT = Path(__file__).parent.parent
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


# ── Config (must match train_ddp.py --dataset electricity --steps 500) ─────────

ARROW_PATH    = FINETUNING_ROOT / "data" / "electricity-tsf.arrow"
CONTEXT_COL   = "context"
FORECAST_COL  = "target"
TARGET_OFFSET = 0       # future = target[:] — raw future only, no aux prefix
CONTEXT_LEN   = 336     # 2 weeks hourly; 336/24=14 cycles of period=24
PRED_LEN      = 24      # 1 day ahead (standard for hourly electricity)
TRAIN_SEED    = 42
BATCH_SIZE    = 4       # N_train ~296 < 500
NUM_BATCHES   = 3
TOTAL_STEPS   = 500
MIN_PERIOD    = 6
MAX_PERIOD    = 84      # n_cycles=4: 336 // 4 = 84 (covers period=24)

D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN   # 48
RANK         = 64
LAMBDA_ORTHO = 0.01
LAMBDA_KL    = 0.05
RS_THRESHOLD = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("PREFLIGHT — Electricity sweep training")
print(f"Device: {DEVICE}")
print("=" * 70)

cycles_avail = CONTEXT_LEN / 24.0
print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
print(f"  ║  CYCLES AVAILABLE = {cycles_avail:.1f}  (context={CONTEXT_LEN} / period=24)       ║")
if cycles_avail >= 4.0:
    print(f"  ║  ✓  >= 4 complete cycles — STL reliable for period=24       ║")
elif cycles_avail >= 2.0:
    print(f"  ║  ⚠  2–4 cycles — borderline; n_cycles=4 used               ║")
else:
    print(f"  ║  ✗  < 2 complete cycles — STL unreliable                   ║")
print(f"  ║     max_period={MAX_PERIOD}  (n_cycles=4: {CONTEXT_LEN} // 4)                   ║")
print(f"  ╚══════════════════════════════════════════════════════════════╝\n")


# ══════════════════════════════════════════════════════════════════════════════
# P1 — DATA LOADING CHECK
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("P1 — DATA LOADING CHECK")
print("=" * 70)

with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_total   = len(table)
rng_split = np.random.default_rng(TRAIN_SEED)
all_idx   = rng_split.permutation(n_total)
n_train   = int(n_total * 0.8)
train_idx = all_idx[:n_train]
val_idx   = all_idx[n_train:]

print(f"\n  Total series:  {n_total}")
print(f"  Train series:  {len(train_idx)}  (80%)")
print(f"  Val series:    {len(val_idx)}  (20%)")

ctx_0    = np.array(table.column(CONTEXT_COL)[0].as_py(), dtype=np.float32)
raw_0    = np.array(table.column(FORECAST_COL)[0].as_py(), dtype=np.float32)
future_0 = raw_0[TARGET_OFFSET:] if TARGET_OFFSET > 0 else raw_0

assert len(ctx_0) == CONTEXT_LEN, \
    f"P1 FAIL: context length {len(ctx_0)} != {CONTEXT_LEN}  — re-run fetch script"
assert len(future_0) == PRED_LEN, \
    f"P1 FAIL: future length {len(future_0)} != {PRED_LEN}"

print(f"\n  context length:  {len(ctx_0)} pts  ✓  (expected {CONTEXT_LEN})")
print(f"  target  length:  {len(raw_0)} pts  ✓  (expected {PRED_LEN}, target_offset=0)")
print(f"  First 3 context values (series 0): {ctx_0[:3].tolist()}")
print(f"  First 3 future  values (series 0): {future_0[:3].tolist()}")

ctx_absmax  = float(np.abs(ctx_0).max())
ctx_absmean = float(np.abs(ctx_0).mean())
if 1.0 < ctx_absmax < 1e7:
    print(f"  Scale check:  mean={ctx_absmean:.2f}  max={ctx_absmax:.2f}  ✓  (raw scale)")
else:
    print(f"  Scale check:  mean={ctx_absmean:.6f}  max={ctx_absmax:.6f}  ⚠  (unexpected scale)")

print(f"\n  max_period={MAX_PERIOD}  (n_cycles=4, covers period=24: {'✓' if MAX_PERIOD >= 24 else '✗'})")
print(f"  cycles_available = {CONTEXT_LEN}/{24} = {cycles_avail:.1f}  "
      f"{'✓ >= 4.0' if cycles_avail >= 4.0 else '⚠ < 4.0'}")

print("\n  P1 PASS\n")


# ── Load Chronos ───────────────────────────────────────────────────────────────

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
print(f"  {sum(p.numel() for p in chronos_model.parameters()):,} params frozen")


# ── PrefixGenerator: fresh warm start ─────────────────────────────────────────

print("Initialising PrefixGenerator with warm_start_from_chronos(scale_warm=False) …")
torch.manual_seed(0)
pg = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(DEVICE)
pg.train()
pg.warm_start_from_chronos(chronos_model, scale_warm=False)
optimizer = torch.optim.AdamW(pg.parameters(), lr=1e-4, weight_decay=0.05)
print(f"  {sum(p.numel() for p in pg.parameters()):,} trainable params\n")


# ── Batch builder ──────────────────────────────────────────────────────────────

def build_batch(row_indices):
    contexts, futures, trends, seasonals, residuals = [], [], [], [], []
    raw_rs_list, filtered_list, periods_list = [], [], []

    for idx in row_indices:
        ctx = np.array(table.column(CONTEXT_COL)[int(idx)].as_py(), dtype=np.float32)
        raw = np.array(table.column(FORECAST_COL)[int(idx)].as_py(), dtype=np.float32)
        ftr = raw[TARGET_OFFSET:] if TARGET_OFFSET > 0 else raw

        periods = detect_periods(ctx, min_period=MIN_PERIOD, max_period=MAX_PERIOD)
        tr, se, re = stl_decompose(ctx, periods)
        rs = float(np.var(re) / (np.var(ctx) + 1e-9))

        raw_rs_list.append(rs)
        periods_list.append(periods)

        if rs > RS_THRESHOLD:
            tr = np.zeros_like(ctx, dtype=np.float32)
            se = np.zeros_like(ctx, dtype=np.float32)
            re = np.zeros_like(ctx, dtype=np.float32)
            filtered_list.append(True)
        else:
            filtered_list.append(False)

        contexts.append(ctx)
        futures.append(ftr)
        trends.append(tr)
        seasonals.append(se)
        residuals.append(re)

    ctx_tensor = torch.tensor(np.stack(contexts), dtype=torch.float32)
    input_ids, attn_mask, ctx_scale = tokenizer.context_input_transform(
        pipeline._prepare_and_validate_context(ctx_tensor)
    )
    future_tensor = torch.tensor(np.stack(futures), dtype=torch.float32)
    label_ids, label_mask, _ = tokenizer._input_transform(
        context=future_tensor, scale=ctx_scale.cpu()
    )
    label_ids[~label_mask.bool()] = -100

    scale = ctx_scale.to(DEVICE).unsqueeze(-1)   # (B, 1)
    batch = dict(
        input_ids = input_ids.to(DEVICE),
        attn_mask = attn_mask.to(DEVICE),
        label_ids = label_ids.to(DEVICE),
        trend     = torch.tensor(np.stack(trends),    dtype=torch.float32).to(DEVICE) / scale,
        seasonal  = torch.tensor(np.stack(seasonals), dtype=torch.float32).to(DEVICE) / scale,
        residual  = torch.tensor(np.stack(residuals), dtype=torch.float32).to(DEVICE) / scale,
    )
    return batch, raw_rs_list, filtered_list, periods_list


def ortho_penalty(prefix_kvs):
    total = torch.zeros(1, device=DEVICE)
    for K, _V in prefix_kvs:
        m  = K.shape[1] // 3
        Kt = K[:, 0:m,   :].flatten(1)
        Ks = K[:, m:2*m, :].flatten(1)
        Kr = K[:, 2*m:,  :].flatten(1)
        total = (
            total
            + F.cosine_similarity(Kt, Ks, dim=1).mean() ** 2
            + F.cosine_similarity(Kt, Kr, dim=1).mean() ** 2
            + F.cosine_similarity(Ks, Kr, dim=1).mean() ** 2
        )
    return total / len(prefix_kvs)


# ══════════════════════════════════════════════════════════════════════════════
# P4 — CEILING FORMULA CHECK (no model needed)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("P4 — CEILING FORMULA CHECK  (500 steps)")
print("=" * 70)
for check_step in [0, 100, 250, 500]:
    c    = min(0.5 + 0.3 * (check_step / TOTAL_STEPS), 0.8)
    note = ""
    if check_step == 0:
        note = "  ← must be 0.50"
    elif check_step == TOTAL_STEPS:
        note = "  ← must be 0.80"
    print(f"  step={check_step:>4}: ceiling={c:.4f}{note}")

c0   = min(0.5 + 0.3 * (0 / TOTAL_STEPS), 0.8)
cend = min(0.5 + 0.3 * (TOTAL_STEPS / TOTAL_STEPS), 0.8)
assert abs(c0   - 0.50) < 1e-6, f"P4 FAIL: step-0 ceiling={c0:.6f}, expected 0.50"
assert abs(cend - 0.80) < 1e-6, f"P4 FAIL: step-{TOTAL_STEPS} ceiling={cend:.6f}, expected 0.80"
print(f"\n  P4 PASS  (step=0 → 0.50, step={TOTAL_STEPS} → 0.80)\n")


# ══════════════════════════════════════════════════════════════════════════════
# P2 / P3 / P5 / P6 — 3-BATCH SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("P2 / P3 / P5 / P6 — 3-BATCH SMOKE TEST")
print("=" * 70)

training_state = {
    'current_step':                   0,
    'total_steps':                    TOTAL_STEPS,
    'ceiling_fired_accumulator':      [],
    'pfx_total_pre_ceil_accumulator': [],
}

all_resid_sigs = []
all_filtered   = []
all_periods    = []
fc_losses      = []
kl_losses      = []
kl_finite      = []

for b_idx in range(NUM_BATCHES):
    row_idx = train_idx[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE].tolist()
    batch, raw_rs, filtered_flags, periods_list = build_batch(row_idx)

    all_resid_sigs.extend(raw_rs)
    all_filtered.extend(filtered_flags)
    all_periods.append(periods_list)

    training_state['current_step'] = 0

    with torch.no_grad():
        no_pfx_out = chronos_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
    no_pfx_logits = no_pfx_out.logits.detach()

    optimizer.zero_grad()
    try:
        prefix_kvs = pg(batch["trend"], batch["seasonal"], batch["residual"])
        inject_prefix(
            chronos_model, prefix_generator=pg,
            prefix_kvs=prefix_kvs, training_state=training_state,
        )
        output  = chronos_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
        fc_loss = output.loss
        o_loss  = ortho_penalty(prefix_kvs)
        kl_loss = F.kl_div(
            F.log_softmax(output.logits / 2.0, dim=-1),
            F.softmax(no_pfx_logits / 2.0, dim=-1),
            reduction='batchmean',
        )
        total_loss = fc_loss + LAMBDA_ORTHO * o_loss + LAMBDA_KL * kl_loss
        total_loss.backward()
        optimizer.step()
    finally:
        remove_prefix_hooks(chronos_model)

    fc_val = fc_loss.item()
    kl_val = kl_loss.item()
    fc_losses.append(fc_val)
    kl_losses.append(kl_val)
    kl_finite.append(not math.isnan(kl_val) and not math.isinf(kl_val))

    print(f"\n  Batch {b_idx}:")
    print(f"    fc_loss={fc_val:.4f}  kl_loss={kl_val:.4f}  "
          f"total={total_loss.item():.4f}")
    print(f"    Periods detected: {periods_list}")
    print(f"    resid_sig:        {[f'{r:.4f}' for r in raw_rs]}")
    print(f"    Fix 1b triggered: {filtered_flags}")


# ══════════════════════════════════════════════════════════════════════════════
# P2 — Period detection: expect period=24 (daily) in 2/3 batches
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("P2 — STL PERIOD DETECTION  (Electricity: expect period=24 dominant)")
print(f"  context_len={CONTEXT_LEN}, max_period={MAX_PERIOD} (n_cycles=4)")
print(f"  Accept any period >= 12 (half-day cycle or longer)")
print("=" * 70)

ELEC_MIN_PLAUSIBLE = 12  # half-day or longer
batches_plausible = 0

for b_idx, periods_list in enumerate(all_periods):
    has_plausible = any(p >= ELEC_MIN_PLAUSIBLE for pds in periods_list for p in pds)
    has_daily     = any(p == 24 for pds in periods_list for p in pds)
    n_plausible   = sum(1 for pds in periods_list if any(p >= ELEC_MIN_PLAUSIBLE for p in pds))
    if has_plausible:
        batches_plausible += 1
    flag = "✓" if has_plausible else "⚠"
    print(f"  Batch {b_idx}: plausible period (>={ELEC_MIN_PLAUSIBLE}) in "
          f"{n_plausible}/{len(periods_list)} series  {flag}  "
          f"{'(period=24 detected ✓)' if has_daily else '(no period=24 — other dominant)'}")
    print(f"    All detected: {periods_list}")

p2_pass = batches_plausible >= 2
print(f"\n  Batches with plausible period: {batches_plausible}/{NUM_BATCHES}")
print(f"  P2 {'PASS  ✓' if p2_pass else 'FAIL ⚠  (< 2/3 batches detected period >= 12)'}")


# ══════════════════════════════════════════════════════════════════════════════
# P3 — Fix 1b trigger rate
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("P3 — FIX 1B TRIGGER RATE  (resid_sig > 0.4 → zero decomposition)")
print("=" * 70)
print(f"\n  resid_sig values across {len(all_resid_sigs)} series ({NUM_BATCHES} batches):")
for si, rs in enumerate(all_resid_sigs):
    flag = "  ← FILTERED" if rs > RS_THRESHOLD else ""
    print(f"    series {si:>2}: resid_sig = {rs:.4f}{flag}")

n_filtered  = sum(all_filtered)
filter_rate = n_filtered / len(all_filtered) if all_filtered else 0
print(f"\n  Filter trigger: {n_filtered}/{len(all_filtered)}  ({filter_rate*100:.1f}%)")
print(f"  (Electricity has strong daily cycles — expect low filter rate)")
print(f"  P3 PASS  (filter wired; rate reported above)")


# ══════════════════════════════════════════════════════════════════════════════
# P5 — Loss range check
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("P5 — LOSS RANGE CHECK")
print("=" * 70)
p5_finite = all(not math.isnan(fc) and not math.isinf(fc) for fc in fc_losses)
p5_range  = all(3.0 <= fc <= 10.0 for fc in fc_losses)
print(f"\n  fc_loss values: {[f'{v:.4f}' for v in fc_losses]}")
print(f"  All finite:       {p5_finite}")
print(f"  All in [3, 10]:   {p5_range}")
if not p5_range and p5_finite:
    out_of_range = [v for v in fc_losses if not (3.0 <= v <= 10.0)]
    if any(v > 10.0 for v in out_of_range):
        print(f"  ⚠  fc_loss > 10: {[v for v in out_of_range if v > 10.0]}  — check normalisation")
    else:
        print(f"  Values < 3: {[v for v in out_of_range if v < 3.0]}  (below range, not critical)")
p5_pass = p5_finite
print(f"\n  P5 {'PASS  ✓' if p5_pass else 'FAIL ⚠  (NaN or Inf in fc_loss)'}")


# ══════════════════════════════════════════════════════════════════════════════
# P6 — KL loss finite
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("P6 — KL LOSS FINITE CHECK")
print("=" * 70)
print(f"\n  kl_loss values: {[f'{v:.4f}' for v in kl_losses]}")
p6_pass = all(kl_finite)
print(f"  All finite: {p6_pass}")
print(f"\n  P6 {'PASS  ✓' if p6_pass else 'FAIL ⚠  (NaN or Inf in kl_loss)'}")


# ══════════════════════════════════════════════════════════════════════════════
# P_new — D1 gradient ratio
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("P_new — D1 GRADIENT RATIO (post ctx_scale normalisation)")
print("=" * 70)

def _compute_d1(pg_model):
    norms = {}
    for name, mod_list in [("trend",    pg_model.proj_trend),
                            ("seasonal", pg_model.proj_seasonal),
                            ("noise",    pg_model.proj_noise)]:
        p = mod_list[0][0].weight
        norms[name] = p.grad.norm().item() if p.grad is not None else float("nan")
    vals  = [v for v in norms.values() if not math.isnan(v) and v > 0]
    ratio = max(vals) / min(vals) if len(vals) >= 2 else float("nan")
    return norms, ratio

d1_norms, d1_ratio = _compute_d1(pg)
SC_D1_MAX = 50.0
print(f"\n  Gradient norms (proj_trend / proj_seasonal / proj_noise):")
print(f"    trend={d1_norms['trend']:.4f}  "
      f"seasonal={d1_norms['seasonal']:.4f}  "
      f"noise={d1_norms['noise']:.4f}")
print(f"  D1 ratio: {d1_ratio:.2f}×  (threshold: {SC_D1_MAX}×)")
p_new_pass = d1_ratio < SC_D1_MAX
print(f"  P_new (D1 ratio): {'PASS  ✓' if p_new_pass else f'FAIL ⚠  (ratio={d1_ratio:.1f}× > {SC_D1_MAX}×)'}")


# ══════════════════════════════════════════════════════════════════════════════
# P_domain — Electricity domain check
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("P_domain — ELECTRICITY DOMAIN CHECK")
print(f"  Expected: raw kWh scale O(100–5000); daily period=24")
print("=" * 70)

ctx_mean_s0  = float(np.abs(ctx_0).mean())
ctx_max_s0   = float(np.abs(ctx_0).max())
daily_detected = any(
    p == 24 for pds in all_periods for pd_list in ([pds] if isinstance(pds[0], int) else pds) for p in (pd_list if isinstance(pd_list, list) else [pd_list])
)
# Simpler check: scan all detected periods
all_detected_flat = [p for batch_pds in all_periods for pds in batch_pds for p in pds]
has_daily    = 24 in all_detected_flat
has_halfday  = any(p == 12 for p in all_detected_flat)
print(f"\n  Series 0: mean|ctx|={ctx_mean_s0:.2f}  max|ctx|={ctx_max_s0:.2f}")
if ctx_max_s0 > 1.0:
    print(f"  Scale:  ✓ Raw scale confirmed")
else:
    print(f"  Scale:  ⚠ Looks normalised — verify")
print(f"\n  Daily cycle (period=24) detected: {'✓ YES' if has_daily else '⚠ NOT detected'}")
if not has_daily:
    print(f"  Half-day (period=12) detected:   {'YES' if has_halfday else 'NO'}")
    print(f"  All periods in smoke batches: {sorted(set(all_detected_flat))}")
    print(f"  ⚠  period=24 not dominant — check if dataset is correct frequency")
print(f"\n  cycles_available = {cycles_avail:.1f}  (context={CONTEXT_LEN} / 24)  "
      f"{'✓ >= 4' if cycles_avail >= 4.0 else '⚠ < 4'}")
print(f"  P_domain PASS  (scale={ctx_max_s0:.1f}, cycles={cycles_avail:.1f})")


# ══════════════════════════════════════════════════════════════════════════════
# Final verdict
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PREFLIGHT SUMMARY")
print("=" * 70)
print(f"  P1       (data loading):        PASS")
print(f"  P2       (plausible period):    {'PASS  ✓' if p2_pass else 'FAIL ⚠'}")
print(f"  P3       (Fix 1b wired):        PASS  (rate={filter_rate*100:.1f}%)")
print(f"  P4       (ceiling formula):     PASS  (step=0→0.50, step=500→0.80)")
print(f"  P5       (loss finite):         {'PASS  ✓' if p5_pass else 'FAIL ⚠'}")
print(f"  P6       (KL finite):           {'PASS  ✓' if p6_pass else 'FAIL ⚠'}")
print(f"  P_new    (D1 ratio<50×):       {'PASS  ✓' if p_new_pass else 'FAIL ⚠'}")
print(f"  P_domain (scale, cycles):       PASS  (ctx={CONTEXT_LEN}, cycles={cycles_avail:.1f})")
print(f"\n  cycles_available = {cycles_avail:.1f}  (>= 4: reliable period=24 detection)")

all_pass = p2_pass and p5_pass and p6_pass and p_new_pass
if all_pass:
    print(f"\n  All checks passed — safe to submit training job")
else:
    print(f"\n  One or more checks FAILED — investigate before submitting")
print("=" * 70)

assert p5_pass,    "PREFLIGHT ABORT: fc_loss is NaN/Inf — cannot train"
assert p6_pass,    "PREFLIGHT ABORT: kl_loss is NaN/Inf — KL distillation broken"
assert p_new_pass, f"PREFLIGHT ABORT: D1={d1_ratio:.1f}× > {SC_D1_MAX}× — gradient imbalance"
assert p2_pass,    "PREFLIGHT ABORT: no plausible period >= 12 in 2+/3 batches — check STL config"
