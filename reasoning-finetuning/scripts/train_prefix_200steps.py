#!/usr/bin/env python3
"""
train_prefix_200steps.py — Phase 1 prefix tuning: 200-step training run.

Configuration (locked):
  - warm_start_from_chronos(scale_warm=False)
  - AdamW, lr=1e-4, weight_decay=0.01, single param group
  - lambda_ortho=0.01, flat LR
  - batch_size=4, sarsim0-10k.arrow, 80/20 split, seed=42

Logging:
  - Every 10 steps: step, fc_loss, ortho_loss, total_loss
  - At steps 0, 50, 100, 200: D1 (grad ratio), D4 (max cos sim), D5 (per-layer attn)

Stop conditions (halt immediately):
  - fc_loss > 10.0 at any step after step 20
  - fc_loss is NaN
  - D1 ratio > 50× at any checkpoint
  - D4 max cos sim > 0.5 at any checkpoint

Saves checkpoint to: reasoning-finetuning/checkpoints/run-200steps.pt
"""

import sys
from pathlib import Path

REPO_ROOT       = Path(__file__).parent.parent.parent
FINETUNING_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(FINETUNING_ROOT))

import math
import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F

from chronos import ChronosPipeline
from prefix_generator import PrefixGenerator
from prefix_injection import inject_prefix, remove_prefix_hooks
from scripts.step1_stl_decompose import detect_periods, stl_decompose


# ── Config ─────────────────────────────────────────────────────────────────────

ARROW_PATH      = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
CKPT_PATH       = FINETUNING_ROOT / "checkpoints" / "run-200steps.pt"

TOTAL_STEPS     = 200
BATCH_SIZE      = 4
TRAIN_SEED      = 42
LOG_INTERVAL    = 10
DIAG_STEPS      = {0, 50, 100, 200}

D_MODEL         = 512
NUM_LAYERS      = 6
PREFIX_LEN      = 16
PREFIX_TOTAL    = 3 * PREFIX_LEN   # 48
RANK            = 64
LAMBDA_ORTHO    = 0.01

# Stop-condition thresholds
SC_FC_MAX       = 10.0   # after step 20
SC_D1_MAX_RATIO = 50.0
SC_D4_MAX_COS   = 0.5
ATTN_THRESH     = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ────────────────────────────────────────────────────────────────────

print("Loading sarsim0-10k.arrow …")
with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_total   = len(table)
rng_split = np.random.default_rng(TRAIN_SEED)
indices   = rng_split.permutation(n_total)
train_idx = indices[: int(n_total * 0.8)]
print(f"  {n_total} rows  →  {len(train_idx)} train")


# ── Chronos ────────────────────────────────────────────────────────────────────

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
print(f"  {sum(p.numel() for p in chronos_model.parameters()):,} params frozen\n")


# ── PrefixGenerator ────────────────────────────────────────────────────────────

torch.manual_seed(0)
pg = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(DEVICE)
pg.train()
pg.warm_start_from_chronos(chronos_model, scale_warm=False)
print(f"  PrefixGenerator params: {sum(p.numel() for p in pg.parameters()):,}")

optimizer = torch.optim.AdamW(pg.parameters(), lr=1e-4, weight_decay=0.01)


# ── Helpers ────────────────────────────────────────────────────────────────────

def ortho_penalty(prefix_kvs):
    total = torch.zeros(1, device=prefix_kvs[0][0].device)
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


def compute_d1(pg):
    """Bottleneck grad norm ratio max/min across three heads (layer 0 inner linear)."""
    norms = {}
    for name, mod_list in [("trend", pg.proj_trend), ("seasonal", pg.proj_seasonal), ("noise", pg.proj_noise)]:
        p = mod_list[0][0].weight
        norms[name] = p.grad.norm().item() if p.grad is not None else float("nan")
    vals = [v for v in norms.values() if not math.isnan(v) and v > 0]
    ratio = max(vals) / min(vals) if len(vals) >= 2 else float("nan")
    return norms, ratio


def compute_d4(prefix_kvs):
    """Max pairwise |cos sim| across trend/seasonal/noise heads (layer 0)."""
    m   = PREFIX_LEN
    P_K = prefix_kvs[0][0].detach().cpu()
    Kt  = P_K[:, 0:m,   :].flatten(1)
    Ks  = P_K[:, m:2*m, :].flatten(1)
    Kr  = P_K[:, 2*m:,  :].flatten(1)
    ts  = F.cosine_similarity(Kt, Ks, dim=1).mean().item()
    tr  = F.cosine_similarity(Kt, Kr, dim=1).mean().item()
    sr  = F.cosine_similarity(Ks, Kr, dim=1).mean().item()
    return {"ts": ts, "tr": tr, "sr": sr}, max(abs(ts), abs(tr), abs(sr))


def compute_d5(chronos_model, batch):
    """Per-layer total prefix attention (no_grad; hooks must be active)."""
    with torch.no_grad():
        enc_out = chronos_model.model.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            output_attentions=True,
        )
    layer_attn = []
    for w in enc_out.attentions:
        pfx_w = w[:, :, :, :PREFIX_TOTAL].mean().item()
        layer_attn.append(pfx_w * PREFIX_TOTAL)
    layers_above = sum(1 for v in layer_attn if v >= ATTN_THRESH)
    return layer_attn, layers_above


def build_batch(row_indices):
    contexts, futures, trends, seasonals, residuals = [], [], [], [], []
    for idx in row_indices:
        ctx = np.array(table.column("context")[idx].as_py(), dtype=np.float32)
        ftr = np.array(table.column("forecast")[idx].as_py(), dtype=np.float32)
        periods = detect_periods(ctx)
        tr, se, re = stl_decompose(ctx, periods)
        contexts.append(ctx); futures.append(ftr)
        trends.append(tr);    seasonals.append(se);  residuals.append(re)

    ctx_tensor = torch.tensor(np.stack(contexts), dtype=torch.float32).to(DEVICE)
    input_ids, attn_mask, ctx_scale = tokenizer.context_input_transform(
        pipeline._prepare_and_validate_context(ctx_tensor)
    )
    future_tensor = torch.tensor(np.stack(futures), dtype=torch.float32).to(DEVICE)
    label_ids, label_mask, _ = tokenizer._input_transform(
        context=future_tensor, scale=ctx_scale
    )
    label_ids[~label_mask.bool()] = -100
    return dict(
        input_ids = input_ids.to(DEVICE),
        attn_mask = attn_mask.to(DEVICE),
        label_ids = label_ids.to(DEVICE),
        trend     = torch.tensor(np.stack(trends),    dtype=torch.float32).to(DEVICE),
        seasonal  = torch.tensor(np.stack(seasonals), dtype=torch.float32).to(DEVICE),
        residual  = torch.tensor(np.stack(residuals), dtype=torch.float32).to(DEVICE),
    )


# ── Data cursor (cycles through train_idx) ─────────────────────────────────────

data_rng    = np.random.default_rng(TRAIN_SEED + 1)
shuffled    = data_rng.permutation(train_idx).tolist()
cursor      = 0

def next_batch():
    global shuffled, cursor
    if cursor + BATCH_SIZE > len(shuffled):
        shuffled = data_rng.permutation(train_idx).tolist()
        cursor   = 0
    row_idx = shuffled[cursor : cursor + BATCH_SIZE]
    cursor += BATCH_SIZE
    return build_batch(row_idx)


# ── Diagnostic records ─────────────────────────────────────────────────────────

loss_log     = []   # (step, fc, ortho, total)
diag_records = {}   # step → {d1_norms, d1_ratio, d4_cos, d4_max, d5_attn, d5_above}
stop_reason  = None


# ── Training loop ──────────────────────────────────────────────────────────────

print("=" * 65)
print(f"PHASE 1 TRAINING  ({TOTAL_STEPS} steps, batch={BATCH_SIZE})")
print(f"Device: {DEVICE}")
print("=" * 65)
print(f"{'Step':>5}  {'fc_loss':>9}  {'ortho':>8}  {'total':>9}")
print("-" * 40)

for step in range(TOTAL_STEPS):
    batch = next_batch()
    is_diag = step in DIAG_STEPS

    optimizer.zero_grad()
    try:
        prefix_kvs = pg(batch["trend"], batch["seasonal"], batch["residual"])
        inject_prefix(chronos_model, prefix_generator=pg, prefix_kvs=prefix_kvs)

        # D5: capture while hooks are active (no_grad, separate call)
        if is_diag:
            d5_attn, d5_above = compute_d5(chronos_model, batch)

        # Main forward + loss
        output = chronos_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
        fc_loss    = output.loss
        o_loss     = ortho_penalty(prefix_kvs)
        total_loss = fc_loss + LAMBDA_ORTHO * o_loss
        total_loss.backward()

        # D1, D4: capture after backward (grads populated)
        if is_diag:
            d1_norms, d1_ratio = compute_d1(pg)
            d4_cos,   d4_max   = compute_d4(prefix_kvs)
            diag_records[step] = dict(
                d1_norms=d1_norms, d1_ratio=d1_ratio,
                d4_cos=d4_cos,     d4_max=d4_max,
                d5_attn=d5_attn,   d5_above=d5_above,
            )

        optimizer.step()

    finally:
        remove_prefix_hooks(chronos_model)

    fc_val    = fc_loss.item()
    o_val     = o_loss.item()
    tot_val   = total_loss.item()
    loss_log.append((step, fc_val, o_val, tot_val))

    # ── Logging ────────────────────────────────────────────────────────────
    if step % LOG_INTERVAL == 0:
        print(f"{step:>5}  {fc_val:>9.4f}  {o_val:>8.5f}  {tot_val:>9.4f}")

    # ── Diagnostic print at checkpoint steps ───────────────────────────────
    if is_diag:
        rec = diag_records[step]
        print(f"       D1 grad ratio: {rec['d1_ratio']:.1f}×  "
              f"(t={rec['d1_norms']['trend']:.4f} "
              f"s={rec['d1_norms']['seasonal']:.4f} "
              f"n={rec['d1_norms']['noise']:.4f})")
        d4c = rec['d4_cos']
        print(f"       D4 max|cos|:   {rec['d4_max']:.4f}  "
              f"(ts={d4c['ts']:+.4f} tr={d4c['tr']:+.4f} sr={d4c['sr']:+.4f})")
        attn_str = "  ".join(f"L{i}:{v:.3f}" for i, v in enumerate(rec['d5_attn']))
        print(f"       D5 attn:        {attn_str}  ({rec['d5_above']}/6 ≥ 5%)")

    # ── Stop conditions ────────────────────────────────────────────────────
    if math.isnan(fc_val):
        stop_reason = f"STOP: fc_loss is NaN at step {step}"
        break
    if step > 20 and fc_val > SC_FC_MAX:
        stop_reason = f"STOP: fc_loss={fc_val:.4f} > {SC_FC_MAX} at step {step}"
        break
    if is_diag:
        rec = diag_records[step]
        if rec['d1_ratio'] > SC_D1_MAX_RATIO:
            stop_reason = (f"STOP: D1 ratio={rec['d1_ratio']:.1f}× > {SC_D1_MAX_RATIO}× "
                           f"at step {step}")
            break
        if rec['d4_max'] > SC_D4_MAX_COS:
            stop_reason = (f"STOP: D4 max|cos|={rec['d4_max']:.4f} > {SC_D4_MAX_COS} "
                           f"at step {step}")
            break

if stop_reason:
    print(f"\n{'!'*65}")
    print(stop_reason)
    print(f"{'!'*65}")
else:
    # Log the final step if not already logged
    if TOTAL_STEPS - 1 not in {s for s, *_ in loss_log[-1:]}:
        pass   # loop ended normally, last step already printed if multiple of 10
    print("-" * 40)
    print("200 steps complete.")


# ── Save checkpoint ────────────────────────────────────────────────────────────

CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "step":       TOTAL_STEPS,
        "pg_state":   pg.state_dict(),
        "opt_state":  optimizer.state_dict(),
        "loss_log":   loss_log,
        "diag_records": diag_records,
        "config": dict(
            d_model=D_MODEL, num_layers=NUM_LAYERS,
            prefix_len_per_component=PREFIX_LEN, rank=RANK,
            lambda_ortho=LAMBDA_ORTHO, lr=1e-4, weight_decay=0.01,
        ),
    },
    str(CKPT_PATH),
)
print(f"\nCheckpoint saved → {CKPT_PATH}")


# ── End-of-run report ──────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("END-OF-RUN REPORT")
print(f"{'='*65}")

# 1. Loss curve summary
print("\n1. Loss curve summary")
print(f"   {'Step':>5}  {'fc_loss':>9}")
print(f"   {'-'*5}  {'-'*9}")
report_steps = {0, 50, 100, 150, 200}
for s, fc, o, tot in loss_log:
    if s in report_steps:
        print(f"   {s:>5}  {fc:>9.4f}")

fc_series = [fc for s, fc, o, tot in loss_log]
if len(fc_series) >= 2:
    first_half = fc_series[:len(fc_series)//2]
    second_half = fc_series[len(fc_series)//2:]
    trend_str  = ("decreasing" if np.mean(second_half) < np.mean(first_half) - 0.1
                  else "flat" if abs(np.mean(second_half) - np.mean(first_half)) <= 0.1
                  else "increasing")
    print(f"\n   Overall trend:  {trend_str}")

# Spike detection: steps where fc_loss jumps > 1.0 from previous
spikes = []
for i in range(1, len(loss_log)):
    jump = abs(loss_log[i][1] - loss_log[i-1][1])
    if jump > 1.0:
        spikes.append((loss_log[i][0], jump))
if spikes:
    print(f"   Spikes (|Δ| > 1.0):  " +
          "  ".join(f"step {s}: Δ={j:.3f}" for s, j in spikes))
else:
    print("   Spikes: none")

# 2. D1/D4/D5 at step 0 vs 200
print("\n2. D1 / D4 / D5 at step 0 vs step 200")
for ref_step in [0, 200]:
    if ref_step not in diag_records:
        # Find nearest
        candidates = [s for s in diag_records if s <= ref_step]
        ref_step   = max(candidates) if candidates else min(diag_records)
    if ref_step not in diag_records:
        continue
    rec = diag_records[ref_step]
    print(f"\n   Step {ref_step}:")
    print(f"     D1 ratio:   {rec['d1_ratio']:.1f}×  "
          f"(t={rec['d1_norms']['trend']:.4f} "
          f"s={rec['d1_norms']['seasonal']:.4f} "
          f"n={rec['d1_norms']['noise']:.4f})")
    d4c = rec['d4_cos']
    print(f"     D4 max|cos|:{rec['d4_max']:.4f}  "
          f"(ts={d4c['ts']:+.4f} tr={d4c['tr']:+.4f} sr={d4c['sr']:+.4f})")
    attn_str = "  ".join(f"L{i}:{v:.3f}" for i, v in enumerate(rec['d5_attn']))
    print(f"     D5 attn:    {attn_str}  ({rec['d5_above']}/6 ≥ 5%)")

# 3. Stop conditions
print(f"\n3. Stop conditions:  {'FIRED — ' + stop_reason if stop_reason else 'none fired'}")

# 4. Checkpoint
print(f"\n4. Checkpoint:  {CKPT_PATH}")
print(f"{'='*65}")
