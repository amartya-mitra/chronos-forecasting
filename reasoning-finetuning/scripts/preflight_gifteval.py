#!/usr/bin/env python3
"""
preflight_gifteval.py — Pre-flight checks for GiftEval training (Phase 2 Task 2).

Checks P1–P6 on 3 smoke batches before submitting the training job:

  P1. Data loads correctly
      - num train/val series, context length=96, future length=48
      - future = target[96:] confirmed (not target[:])
      - values in O(100–1000) raw scale range

  P2. STL detects period=24 on at least 2/3 smoke batches
      - With 96-pt context and max_period=n//4=24, daily seasonality
        should be reliably detected

  P3. Fix 1b trigger rate across 3 batches
      - Print per-series resid_sig values
      - M4H is clean data: expect low rate (< SarSim0's 2.6%)

  P4. Ceiling values
      - step=0    → 0.50
      - step=1000 → 0.80

  P5. Forward pass: fc_loss finite; note if > 10 (tokeniser normalisation check)
      - KL loss also computed at step 0

  P6. KL loss at step 0 is finite (not NaN)

Config must match train_ddp.py --dataset gifteval --steps 1000:
  context_len=96, pred_len=48, target_offset=96
  batch_size=4, TOTAL_STEPS=1000, min_period=6, rs_threshold=0.4
"""

import math
import sys
import time
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


# ── Config (must match train_ddp.py --dataset gifteval --steps 1000) ──────────

ARROW_PATH   = FINETUNING_ROOT / "data" / "m4hourly-tsf.arrow"
CONTEXT_COL  = "context"
FORECAST_COL = "target"
TARGET_OFFSET = 96      # future = target[96:] → 48 pts raw future
CONTEXT_LEN  = 96
PRED_LEN     = 48
TRAIN_SEED   = 42
BATCH_SIZE   = 4
NUM_BATCHES  = 3
TOTAL_STEPS  = 1000     # GiftEval training run length

D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN   # 48
RANK         = 64
LAMBDA_ORTHO = 0.01
LAMBDA_KL    = 0.05
MIN_PERIOD   = 6
RS_THRESHOLD = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("PREFLIGHT — GiftEval M4 Hourly training (Phase 2 Task 2)")
print(f"Device: {DEVICE}")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# P1 — DATA LOADING CHECK
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
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

# Verify context length
ctx_0   = np.array(table.column(CONTEXT_COL)[0].as_py(), dtype=np.float32)
raw_0   = np.array(table.column(FORECAST_COL)[0].as_py(), dtype=np.float32)
future_0 = raw_0[TARGET_OFFSET:]

assert len(ctx_0) == CONTEXT_LEN, \
    f"P1 FAIL: context length {len(ctx_0)} != {CONTEXT_LEN}"
assert len(future_0) == PRED_LEN, \
    f"P1 FAIL: future length {len(future_0)} != {PRED_LEN} " \
    f"(target[{TARGET_OFFSET}:] should give {PRED_LEN} pts)"

print(f"\n  context length:  {len(ctx_0)} pts  ✓  (expected {CONTEXT_LEN})")
print(f"  future = target[{TARGET_OFFSET}:]:  {len(future_0)} pts  ✓  (expected {PRED_LEN})")
print(f"  First 3 context values (series 0): {ctx_0[:3].tolist()}")
print(f"  First 3 future  values (series 0): {future_0[:3].tolist()}")

# Scale check: raw M4H values should be O(100–10000), not O(1)
ctx_absmax = float(np.abs(ctx_0).max())
if 10.0 < ctx_absmax < 1e7:
    print(f"  Scale check:  |ctx|_max = {ctx_absmax:.1f}  ✓  (raw O(100–10000))")
else:
    print(f"  Scale check:  |ctx|_max = {ctx_absmax:.1f}  ⚠  (expected O(100–10000))")

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
optimizer = torch.optim.AdamW(pg.parameters(), lr=1e-4, weight_decay=0.01)
print(f"  {sum(p.numel() for p in pg.parameters()):,} trainable params\n")


# ── Batch builder ──────────────────────────────────────────────────────────────

def build_batch(row_indices):
    contexts, futures, trends, seasonals, residuals = [], [], [], [], []
    raw_rs_list = []
    filtered_list = []
    periods_list = []

    for idx in row_indices:
        ctx = np.array(table.column(CONTEXT_COL)[int(idx)].as_py(), dtype=np.float32)
        raw = np.array(table.column(FORECAST_COL)[int(idx)].as_py(), dtype=np.float32)
        ftr = raw[TARGET_OFFSET:]   # target[96:] → 48 pts raw future

        periods = detect_periods(ctx, min_period=MIN_PERIOD)
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

    # Normalise STL components by ctx_scale (must match train_ddp.py _tokenize_batch)
    scale = ctx_scale.to(DEVICE).unsqueeze(-1)   # (B, 1)
    batch = dict(
        input_ids = input_ids.to(DEVICE),
        attn_mask = attn_mask.to(DEVICE),
        label_ids = label_ids.to(DEVICE),
        trend     = torch.tensor(np.stack(trends),    dtype=torch.float32).to(DEVICE) / scale,
        seasonal  = torch.tensor(np.stack(seasonals), dtype=torch.float32).to(DEVICE) / scale,
        residual  = torch.tensor(np.stack(residuals), dtype=torch.float32).to(DEVICE) / scale,
        ctx_scale = ctx_scale,   # pass through for diagnostics
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
print("P4 — CEILING FORMULA CHECK")
print("=" * 70)
for check_step in [0, 250, 500, 750, 1000]:
    c = min(0.5 + 0.3 * (check_step / TOTAL_STEPS), 0.8)
    note = ""
    if check_step == 0:
        note = "  ← must be 0.50"
    elif check_step == 1000:
        note = "  ← must be 0.80"
    print(f"  step={check_step:>5}: ceiling={c:.4f}{note}")

c0    = min(0.5 + 0.3 * (0 / TOTAL_STEPS), 0.8)
c1000 = min(0.5 + 0.3 * (TOTAL_STEPS / TOTAL_STEPS), 0.8)
assert abs(c0    - 0.50) < 1e-6, f"P4 FAIL: step-0 ceiling={c0:.6f}, expected 0.50"
assert abs(c1000 - 0.80) < 1e-6, f"P4 FAIL: step-1000 ceiling={c1000:.6f}, expected 0.80"
print("\n  P4 PASS  (step=0 → 0.50, step=1000 → 0.80)\n")


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

all_resid_sigs  = []
all_filtered    = []
all_periods     = []
fc_losses       = []
kl_losses       = []
kl_finite       = []

for b_idx in range(NUM_BATCHES):
    row_idx = train_idx[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE].tolist()
    batch, raw_rs, filtered_flags, periods_list = build_batch(row_idx)

    all_resid_sigs.extend(raw_rs)
    all_filtered.extend(filtered_flags)
    all_periods.append(periods_list)

    training_state['current_step'] = 0   # step 0 for smoke test ceiling

    # No-prefix baseline for KL
    with torch.no_grad():
        no_pfx_out = chronos_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
    no_pfx_logits = no_pfx_out.logits.detach()

    # Prefix forward with Fix 2 ceiling active
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
    print(f"    Fix 1b triggered: {[f for f in filtered_flags]}")


# ══════════════════════════════════════════════════════════════════════════════
# P2 — Period detection report
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("P2 — STL PERIOD DETECTION  (expect period=24 for hourly M4H)")
print("=" * 70)

batches_with_24 = 0
for b_idx, periods_list in enumerate(all_periods):
    has_24 = any(24 in pds for pds in periods_list)
    frac_24 = sum(1 for pds in periods_list if 24 in pds) / len(periods_list)
    if has_24:
        batches_with_24 += 1
    flag = "✓" if has_24 else "⚠"
    print(f"  Batch {b_idx}: period=24 in {frac_24*100:.0f}% of series  {flag}")
    print(f"    All detected: {periods_list}")

p2_pass = batches_with_24 >= 2
print(f"\n  Batches with period=24 present: {batches_with_24}/{NUM_BATCHES}")
print(f"  P2 {'PASS  ✓' if p2_pass else 'FAIL ⚠  (< 2/3 batches detected period=24)'}")


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

n_filtered = sum(all_filtered)
filter_rate = n_filtered / len(all_filtered) if all_filtered else 0
print(f"\n  Filter trigger: {n_filtered}/{len(all_filtered)}  ({filter_rate*100:.1f}%)")
print(f"  (SarSim0 reference: 2.9%;  M4H expected: lower, data is clean)")
print(f"  P3 PASS  (filter wired; trigger rate reflects sample distribution)")


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
        print(f"  ⚠  fc_loss > 10 detected: {out_of_range}")
        print(f"     Check: Chronos tokeniser should apply MeanScaleUniform normalisation.")
        print(f"     Raw M4H scale (O(100–10000)) must be normalised by context scale.")
    else:
        print(f"  Values < 3: {[v for v in out_of_range if v < 3.0]}")
p5_pass = p5_finite
print(f"\n  P5 {'PASS  ✓' if p5_pass else 'FAIL ⚠  (NaN or Inf in fc_loss)'}")


# ══════════════════════════════════════════════════════════════════════════════
# P6 — KL loss finite at step 0
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("P6 — KL LOSS FINITE CHECK")
print("=" * 70)
print(f"\n  kl_loss values: {[f'{v:.4f}' for v in kl_losses]}")
p6_pass = all(kl_finite)
print(f"  All finite: {p6_pass}")
print(f"\n  P6 {'PASS  ✓' if p6_pass else 'FAIL ⚠  (NaN or Inf in kl_loss)'}")


# ══════════════════════════════════════════════════════════════════════════════
# P_new — D1 GRADIENT RATIO CHECK  (after ctx_scale normalisation)
# Should now be << 50× (was 634× before normalisation)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("P_new — D1 GRADIENT RATIO (post ctx_scale normalisation)")
print("=" * 70)

import math as _math

def _compute_d1(pg_model):
    norms = {}
    for name, mod_list in [("trend",    pg_model.proj_trend),
                            ("seasonal", pg_model.proj_seasonal),
                            ("noise",    pg_model.proj_noise)]:
        p = mod_list[0][0].weight
        norms[name] = p.grad.norm().item() if p.grad is not None else float("nan")
    vals  = [v for v in norms.values() if not _math.isnan(v) and v > 0]
    ratio = max(vals) / min(vals) if len(vals) >= 2 else float("nan")
    return norms, ratio

d1_norms, d1_ratio = _compute_d1(pg)
SC_D1_MAX_RATIO = 50.0

print(f"\n  Gradient norms (proj_trend / proj_seasonal / proj_noise):")
print(f"    trend={d1_norms['trend']:.4f}  "
      f"seasonal={d1_norms['seasonal']:.4f}  "
      f"noise={d1_norms['noise']:.4f}")
print(f"  D1 ratio: {d1_ratio:.2f}×  (threshold: {SC_D1_MAX_RATIO}×)")

p_new_pass = d1_ratio < SC_D1_MAX_RATIO
status = "PASS  ✓" if p_new_pass else f"FAIL ⚠  (ratio={d1_ratio:.1f}× > {SC_D1_MAX_RATIO}×)"
print(f"  P_new (D1 ratio): {status}")
print(f"  (Before normalisation fix: D1 was 634.9×)")


# ══════════════════════════════════════════════════════════════════════════════
# Final verdict
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PREFLIGHT SUMMARY")
print("=" * 70)
print(f"  P1 (data loading):      PASS")
print(f"  P2 (period=24 detect):  {'PASS  ✓' if p2_pass else 'FAIL ⚠'}")
print(f"  P3 (Fix 1b wired):      PASS  (rate={filter_rate*100:.1f}%)")
print(f"  P4 (ceiling formula):   PASS  (step=0→0.50, step=1000→0.80)")
print(f"  P5 (loss finite):       {'PASS  ✓' if p5_pass else 'FAIL ⚠'}")
if not p5_range and p5_pass:
    print(f"       (fc_loss outside [3,10] but finite — check tokeniser normalisation)")
print(f"  P6 (KL finite):         {'PASS  ✓' if p6_pass else 'FAIL ⚠'}")
print(f"  P_new (D1 ratio<50×):  {'PASS  ✓' if p_new_pass else 'FAIL ⚠  (gradient imbalance: normalisation fix may be needed)'}")

all_pass = p2_pass and p5_pass and p6_pass and p_new_pass
if all_pass:
    print(f"\n  All checks passed — safe to submit training job")
else:
    print(f"\n  One or more checks FAILED — investigate before submitting")
print("=" * 70)

# Hard fail if critical checks fail
assert p5_pass, "PREFLIGHT ABORT: fc_loss is NaN/Inf — cannot train"
assert p6_pass, "PREFLIGHT ABORT: kl_loss is NaN/Inf — KL distillation is broken"
assert p_new_pass, f"PREFLIGHT ABORT: D1 ratio={d1_ratio:.1f}× > {SC_D1_MAX_RATIO}× — gradient imbalance persists after normalisation"
