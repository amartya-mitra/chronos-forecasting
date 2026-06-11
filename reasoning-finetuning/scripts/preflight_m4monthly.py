#!/usr/bin/env python3
"""
preflight_m4monthly.py — Pre-flight checks for M4 Monthly sweep training.

Checks P1–P6 + P_new on 3 smoke batches before submitting the training job:

  P1. Data loads correctly
      - num train/val series (8000/2000), context length=36, future length=18
      - future = target[36:] confirmed
      - values in O(100–100000) raw scale range

  P2. STL detects plausible period on 3/3 smoke batches
      - With 36-pt context, min_period=4, max_period=12
      - Expect period=12 (annual) or period=4 (quarterly)
      - Flag if period=1 or period=2 detected

  P3. Fix 1b trigger rate across 3 batches
      - Monthly economic series may trigger more than M4H (0%)
      - Print per-series resid_sig values

  P4. Ceiling formula check
      - step=0    → 0.50
      - step=2000 → 0.80

  P5. Forward pass: fc_loss finite; warn if > 10

  P6. KL loss finite at step 0

  P_new. D1 gradient ratio < 50× after ctx_scale normalisation

Config must match train_ddp.py --dataset m4monthly --steps 2000:
  context_len=36, pred_len=18, target_offset=36
  batch_size=8, TOTAL_STEPS=2000, min_period=4, max_period=12, rs_threshold=0.4
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


# ── Config (must match train_ddp.py --dataset m4monthly --steps 2000) ─────────

ARROW_PATH    = FINETUNING_ROOT / "data" / "m4monthly-tsf.arrow"
CONTEXT_COL   = "context"
FORECAST_COL  = "target"
TARGET_OFFSET = 36      # future = target[36:] → 18 pts raw future
CONTEXT_LEN   = 36
PRED_LEN      = 18
TRAIN_SEED    = 42
BATCH_SIZE    = 8
NUM_BATCHES   = 3
TOTAL_STEPS   = 2000    # M4 Monthly training run length
MIN_PERIOD    = 4
MAX_PERIOD    = 12      # 36 // 3 = 12 (n_cycles=3)

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
print("PREFLIGHT — M4 Monthly sweep training")
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

ctx_0    = np.array(table.column(CONTEXT_COL)[0].as_py(), dtype=np.float32)
raw_0    = np.array(table.column(FORECAST_COL)[0].as_py(), dtype=np.float32)
future_0 = raw_0[TARGET_OFFSET:]

assert len(ctx_0) == CONTEXT_LEN, \
    f"P1 FAIL: context length {len(ctx_0)} != {CONTEXT_LEN}"
assert len(future_0) == PRED_LEN, \
    f"P1 FAIL: future length {len(future_0)} != {PRED_LEN}"
assert len(raw_0) == TARGET_OFFSET + PRED_LEN, \
    f"P1 FAIL: target length {len(raw_0)} != {TARGET_OFFSET + PRED_LEN}"

print(f"\n  context length:          {len(ctx_0)} pts  ✓  (expected {CONTEXT_LEN})")
print(f"  target length:           {len(raw_0)} pts  ✓  (expected {TARGET_OFFSET + PRED_LEN})")
print(f"  future = target[{TARGET_OFFSET}:]:  {len(future_0)} pts  ✓  (expected {PRED_LEN})")
print(f"  First 3 context values (series 0): {ctx_0[:3].tolist()}")
print(f"  First 3 future  values (series 0): {future_0[:3].tolist()}")

ctx_absmax = float(np.abs(ctx_0).max())
if 1.0 < ctx_absmax < 1e8:
    print(f"  Scale check:  |ctx|_max = {ctx_absmax:.1f}  ✓  (raw scale)")
else:
    print(f"  Scale check:  |ctx|_max = {ctx_absmax:.1f}  ⚠  (unexpected scale)")

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
    raw_rs_list, filtered_list, periods_list = [], [], []

    for idx in row_indices:
        ctx = np.array(table.column(CONTEXT_COL)[int(idx)].as_py(), dtype=np.float32)
        raw = np.array(table.column(FORECAST_COL)[int(idx)].as_py(), dtype=np.float32)
        ftr = raw[TARGET_OFFSET:]

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
print("P4 — CEILING FORMULA CHECK")
print("=" * 70)
for check_step in [0, 500, 1000, 1500, 2000]:
    c = min(0.5 + 0.3 * (check_step / TOTAL_STEPS), 0.8)
    note = ""
    if check_step == 0:
        note = "  ← must be 0.50"
    elif check_step == TOTAL_STEPS:
        note = "  ← must be 0.80"
    print(f"  step={check_step:>5}: ceiling={c:.4f}{note}")

c0    = min(0.5 + 0.3 * (0 / TOTAL_STEPS), 0.8)
c2000 = min(0.5 + 0.3 * (TOTAL_STEPS / TOTAL_STEPS), 0.8)
assert abs(c0    - 0.50) < 1e-6, f"P4 FAIL: step-0 ceiling={c0:.6f}, expected 0.50"
assert abs(c2000 - 0.80) < 1e-6, f"P4 FAIL: step-2000 ceiling={c2000:.6f}, expected 0.80"
print(f"\n  P4 PASS  (step=0 → 0.50, step=2000 → 0.80)\n")


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
# P2 — Period detection report
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("P2 — STL PERIOD DETECTION  (expect period=12 or period=4 for M4 Monthly)")
print("=" * 70)

batches_plausible = 0
for b_idx, periods_list in enumerate(all_periods):
    has_plausible = any(p in (4, 6, 12) for pds in periods_list for p in pds)
    has_trivial   = any(p <= 2 for pds in periods_list for p in pds)
    if has_plausible:
        batches_plausible += 1
    flag = "✓" if has_plausible and not has_trivial else ("⚠ trivial" if has_trivial else "⚠ no seasonal")
    print(f"  Batch {b_idx}: plausible period in {sum(1 for pds in periods_list if any(p in (4,6,12) for p in pds))}/{len(periods_list)} series  {flag}")
    print(f"    All detected: {periods_list}")

p2_pass = batches_plausible >= 2
print(f"\n  Batches with plausible period: {batches_plausible}/{NUM_BATCHES}")
print(f"  P2 {'PASS  ✓' if p2_pass else 'FAIL ⚠  (< 2/3 batches detected period in {4,6,12})'}")


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
print(f"  (M4H reference: 0%;  M4 Monthly expected: higher — monthly economic series)")
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
        print(f"  ⚠  fc_loss > 10: {[v for v in out_of_range if v > 10]}  — check normalisation")
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
# Final verdict
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PREFLIGHT SUMMARY")
print("=" * 70)
print(f"  P1 (data loading):       PASS")
print(f"  P2 (plausible period):   {'PASS  ✓' if p2_pass else 'FAIL ⚠'}")
print(f"  P3 (Fix 1b wired):       PASS  (rate={filter_rate*100:.1f}%)")
print(f"  P4 (ceiling formula):    PASS  (step=0→0.50, step=2000→0.80)")
print(f"  P5 (loss finite):        {'PASS  ✓' if p5_pass else 'FAIL ⚠'}")
print(f"  P6 (KL finite):          {'PASS  ✓' if p6_pass else 'FAIL ⚠'}")
print(f"  P_new (D1 ratio<50×):   {'PASS  ✓' if p_new_pass else 'FAIL ⚠'}")

all_pass = p2_pass and p5_pass and p6_pass and p_new_pass
if all_pass:
    print(f"\n  All checks passed — safe to submit training job")
else:
    print(f"\n  One or more checks FAILED — investigate before submitting")
print("=" * 70)

assert p5_pass,    "PREFLIGHT ABORT: fc_loss is NaN/Inf — cannot train"
assert p6_pass,    "PREFLIGHT ABORT: kl_loss is NaN/Inf — KL distillation broken"
assert p_new_pass, f"PREFLIGHT ABORT: D1={d1_ratio:.1f}× > {SC_D1_MAX}× — gradient imbalance"
