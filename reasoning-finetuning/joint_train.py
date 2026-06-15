#!/usr/bin/env python3
"""
joint_train.py — Stage 1 Pilot: Joint prefix-tuning on M4 Hourly + M4 Monthly.

Alternates between M4 Hourly (even steps) and M4 Monthly (odd steps).
A single shared PrefixGenerator serves both periodicities via ctx_scale-normalised
STL decompositions and AdaptiveAvgPool1d(32) in the encoder heads.

Training:
  TOTAL_STEPS = 3000  (1,500 effective steps per dataset)
  2 × H100 via torchrun, batch = 8/GPU (effective 16)
  lr = 1e-4, weight_decay = 0.01, warmup = 50 steps, cosine thereafter

New diagnostics beyond single-dataset sweep:
  Per-dataset fc_loss EMA (α=0.05) — primary interference signal
  D6: cosine similarity between mean prefix KV vectors for M4 Hourly vs M4 Monthly
      val-set probe batches, computed before loop, at step 1500, and at step 3000.
      D6→1: prefix collapsed to a dataset-agnostic average (BAD)
      D6<0.9: meaningful differentiation between periodicities (GOOD)

Checkpoint: checkpoints/pilot-joint-m4h-m4m-3000.pt
  Includes d6_traj and ema_fc_traj in 'diagnostics' key for eval_joint_pilot.py.

Smoke test: SMOKE_TEST=1 runs 3 steps, skips KL profiling.
"""

import itertools
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT       = Path(__file__).parent.parent
FINETUNING_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(FINETUNING_ROOT))

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from chronos import ChronosPipeline
from prefix_generator import PrefixGenerator
from prefix_injection import inject_prefix, remove_prefix_hooks
from scripts.step1_stl_decompose import detect_periods, stl_decompose


# ── DDP utilities ──────────────────────────────────────────────────────────────

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def rank0():
    return (not is_distributed()) or dist.get_rank() == 0


# ── Dataset-specific configs ───────────────────────────────────────────────────

HOURLY_CFG = {
    'data_path':     'data/m4hourly-tsf.arrow',
    'context_col':   'context',
    'forecast_col':  'target',
    'target_offset': 96,    # future = target[96:] — last 48 pts (144-pt target column)
    'min_period':    6,
    'max_period':    24,    # 96 // 4 = 24, covers daily period=24
}
MONTHLY_CFG = {
    'data_path':     'data/m4monthly-tsf.arrow',
    'context_col':   'context',
    'forecast_col':  'target',
    'target_offset': 36,    # future = target[36:] — last 18 pts (54-pt target column)
    'min_period':    4,
    'max_period':    12,    # 36 // 3 = 12, covers annual period=12
}

# ── Smoke test + training hyperparameters ─────────────────────────────────────

SMOKE_TEST         = os.environ.get("SMOKE_TEST", "0") == "1"

TOTAL_STEPS        = 3    if SMOKE_TEST else 3000
BATCH_SIZE_PER_GPU = 8
WARMUP_STEPS       = 1    if SMOKE_TEST else 50
WEIGHT_DECAY       = 0.01
TRAIN_SEED         = 42
LOG_INTERVAL       = 1    if SMOKE_TEST else 20
EMA_ALPHA          = 0.05   # per-dataset fc_loss exponential moving average

CKPT_NAME          = 'pilot-joint-m4h-m4m-3000.pt'
CKPT_STEPS         = {3}    if SMOKE_TEST else {3000}
# Diag at every 500 steps; for readability, indexed 0-based (0, 499, 999, ...)
# except final step is 2999 for TOTAL_STEPS=3000
DIAG_STEPS         = {0, 1, 2}            if SMOKE_TEST else {0, 500, 1000, 1499, 1999, 2499, 2999}
# D6 computed before loop (step 0), after step 1499 (=1500), after step 2999 (=3000)
D6_AFTER_STEPS     = {1, 2}               if SMOKE_TEST else {1500, 3000}
D6_PROBE_N         = 4                    if SMOKE_TEST else 32

D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN    # 48
RANK         = 64
LAMBDA_ORTHO = 0.01
LAMBDA_KL    = 0.05
RS_THRESHOLD = 0.4

SC_FC_MAX       = 10.0
SC_D1_MAX_RATIO = 50.0
SC_D4_MAX_COS   = 0.5
ATTN_THRESH     = 0.05
D1_WARMUP_GRACE = 10


# ── DDP / device setup ─────────────────────────────────────────────────────────

distributed = ("LOCAL_RANK" in os.environ) and not SMOKE_TEST
if distributed:
    local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")
else:
    local_rank = 0
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if rank0():
    mode = "SMOKE TEST (3 steps)" if SMOKE_TEST else f"DDP={distributed}"
    print(f"Device: {device}  |  {mode}"
          + (f"  |  World size: {dist.get_world_size()}" if distributed else ""))
    print("Pilot: M4 Hourly (even steps) + M4 Monthly (odd steps)")


# ── Dataset class (per-dataset min/max period) ─────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, table, indices, cfg):
        self.table        = table
        self.indices      = indices
        self.context_col  = cfg['context_col']
        self.forecast_col = cfg['forecast_col']
        self.offset       = cfg['target_offset']
        self.min_period   = cfg['min_period']
        self.max_period   = cfg.get('max_period', None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        ctx = np.array(self.table.column(self.context_col)[idx].as_py(),  dtype=np.float32)
        raw = np.array(self.table.column(self.forecast_col)[idx].as_py(), dtype=np.float32)
        ftr = raw[self.offset:] if self.offset > 0 else raw

        periods    = detect_periods(ctx, min_period=self.min_period, max_period=self.max_period)
        tr, se, re = stl_decompose(ctx, periods)
        rs         = float(np.var(re) / (np.var(ctx) + 1e-9))
        filtered   = rs > RS_THRESHOLD
        if filtered:
            tr = np.zeros_like(ctx, dtype=np.float32)
            se = np.zeros_like(ctx, dtype=np.float32)
            re = np.zeros_like(ctx, dtype=np.float32)

        return dict(
            context  = torch.from_numpy(ctx),
            forecast = torch.from_numpy(ftr),
            trend    = torch.from_numpy(tr.astype(np.float32)),
            seasonal = torch.from_numpy(se.astype(np.float32)),
            residual = torch.from_numpy(re.astype(np.float32)),
            filtered = torch.tensor(1.0 if filtered else 0.0),
        )


# ── Load both datasets ─────────────────────────────────────────────────────────

def _load_table(cfg_key, cfg):
    path = FINETUNING_ROOT / cfg['data_path']
    if rank0():
        print(f"  Loading {path.name} …")
    with open(str(path), "rb") as f:
        table = ipc.open_file(f).read_all()
    n     = len(table)
    rng   = np.random.default_rng(TRAIN_SEED)
    idx   = rng.permutation(n)
    n_tr  = int(n * 0.8)
    tr_i  = idx[:n_tr]
    val_i = idx[n_tr:]
    if rank0():
        print(f"    {n} rows  →  {len(tr_i)} train / {len(val_i)} val")
    return table, tr_i, val_i

if rank0():
    print("Loading datasets …")
hourly_table,  hourly_train_idx,  hourly_val_idx  = _load_table('m4hourly',  HOURLY_CFG)
monthly_table, monthly_train_idx, monthly_val_idx = _load_table('m4monthly', MONTHLY_CFG)


# ── DataLoaders ────────────────────────────────────────────────────────────────

def _build_loader(table, train_idx, cfg):
    ds = TimeSeriesDataset(table, train_idx, cfg)
    if distributed:
        sampler = DistributedSampler(ds, shuffle=True)
        sampler.set_epoch(0)
        return DataLoader(ds, batch_size=BATCH_SIZE_PER_GPU, sampler=sampler,
                          num_workers=2, pin_memory=True)
    return DataLoader(ds, batch_size=BATCH_SIZE_PER_GPU, shuffle=True, num_workers=2)

loader_hourly  = _build_loader(hourly_table,  hourly_train_idx,  HOURLY_CFG)
loader_monthly = _build_loader(monthly_table, monthly_train_idx, MONTHLY_CFG)
iter_hourly    = itertools.cycle(loader_hourly)
iter_monthly   = itertools.cycle(loader_monthly)


# ── Chronos (frozen) ───────────────────────────────────────────────────────────

if rank0():
    print("Loading Chronos T5-Small …")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small", device_map=str(device), dtype=torch.float32,
)
chronos_model = pipeline.model
tokenizer     = pipeline.tokenizer
for p in chronos_model.parameters():
    p.requires_grad_(False)
if rank0():
    print(f"  {sum(p.numel() for p in chronos_model.parameters()):,} params frozen")


# ── PrefixGenerator: fresh warm start ─────────────────────────────────────────

torch.manual_seed(0)
prefix_gen = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(device)
prefix_gen.warm_start_from_chronos(chronos_model, scale_warm=False)
prefix_gen.train()

if distributed:
    prefix_gen_ddp = DDP(prefix_gen, device_ids=[local_rank])
else:
    prefix_gen_ddp = prefix_gen

if rank0():
    print(f"  PrefixGenerator: {sum(p.numel() for p in prefix_gen.parameters()):,} trainable params")


# ── Optimizer + scheduler ──────────────────────────────────────────────────────

optimizer    = torch.optim.AdamW(prefix_gen_ddp.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_STEPS)
cosine_sched = CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS - WARMUP_STEPS, eta_min=1e-5)
scheduler    = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                            milestones=[WARMUP_STEPS])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize_batch(raw_batch):
    ctx_tensor    = raw_batch["context"].cpu()
    input_ids, attn_mask, ctx_scale = tokenizer.context_input_transform(
        pipeline._prepare_and_validate_context(ctx_tensor)
    )
    future_tensor = raw_batch["forecast"].cpu()
    label_ids, label_mask, _ = tokenizer._input_transform(
        context=future_tensor, scale=ctx_scale.cpu()
    )
    label_ids[~label_mask.bool()] = -100
    scale = ctx_scale.to(device).unsqueeze(-1)
    return dict(
        input_ids = input_ids.to(device),
        attn_mask = attn_mask.to(device),
        label_ids = label_ids.to(device),
        trend     = raw_batch["trend"].to(device)    / scale,
        seasonal  = raw_batch["seasonal"].to(device) / scale,
        residual  = raw_batch["residual"].to(device) / scale,
        filtered  = raw_batch["filtered"],
    )


def _ortho_penalty(prefix_kvs):
    total = torch.zeros(1, device=device)
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
    norms = {}
    for name, mod_list in [("trend", pg.proj_trend), ("seasonal", pg.proj_seasonal),
                           ("noise", pg.proj_noise)]:
        p = mod_list[0][0].weight
        norms[name] = p.grad.norm().item() if p.grad is not None else float("nan")
    vals  = [v for v in norms.values() if not math.isnan(v) and v > 0]
    ratio = max(vals) / min(vals) if len(vals) >= 2 else float("nan")
    return norms, ratio


def compute_d4(prefix_kvs):
    m   = PREFIX_LEN
    P_K = prefix_kvs[0][0].detach()
    Kt  = P_K[:, 0:m,   :].flatten(1)
    Ks  = P_K[:, m:2*m, :].flatten(1)
    Kr  = P_K[:, 2*m:,  :].flatten(1)
    ts  = F.cosine_similarity(Kt, Ks, dim=1).mean().item()
    tr_ = F.cosine_similarity(Kt, Kr, dim=1).mean().item()
    sr  = F.cosine_similarity(Ks, Kr, dim=1).mean().item()
    return {"ts": ts, "tr": tr_, "sr": sr}, max(abs(ts), abs(tr_), abs(sr))


def compute_d5(batch):
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
    return layer_attn, sum(1 for v in layer_attn if v >= ATTN_THRESH)


# ── D6: cross-dataset prefix KV cosine similarity ─────────────────────────────

def _build_d6_probe(table, val_idx, cfg, n=D6_PROBE_N):
    """Build a fixed batch of val samples for D6, normalised by ctx_scale."""
    probe_idx = val_idx[:min(n, len(val_idx))]
    ctx_list, tr_list, se_list, re_list = [], [], [], []
    for i in probe_idx:
        ctx = np.array(table.column(cfg['context_col'])[int(i)].as_py(), dtype=np.float32)
        pds = detect_periods(ctx, min_period=cfg['min_period'],
                             max_period=cfg.get('max_period'))
        tr, se, re = stl_decompose(ctx, pds)
        rs = float(np.var(re) / (np.var(ctx) + 1e-9))
        if rs > RS_THRESHOLD:
            tr = np.zeros_like(ctx, dtype=np.float32)
            se = np.zeros_like(ctx, dtype=np.float32)
            re = np.zeros_like(ctx, dtype=np.float32)
        ctx_list.append(ctx)
        tr_list.append(tr.astype(np.float32))
        se_list.append(se.astype(np.float32))
        re_list.append(re.astype(np.float32))

    ctx_t = torch.tensor(np.stack(ctx_list), dtype=torch.float32)
    _, _, ctx_scale = tokenizer.context_input_transform(
        pipeline._prepare_and_validate_context(ctx_t)
    )
    scale = ctx_scale.to(device).unsqueeze(-1)
    return (
        torch.tensor(np.stack(tr_list), dtype=torch.float32).to(device) / scale,
        torch.tensor(np.stack(se_list), dtype=torch.float32).to(device) / scale,
        torch.tensor(np.stack(re_list), dtype=torch.float32).to(device) / scale,
    )


def compute_d6(pg, probe_h, probe_m):
    """D6: cosine similarity between mean prefix KV vectors for hourly vs monthly."""
    pg.eval()
    with torch.no_grad():
        kvs_h = pg(*probe_h)
        kvs_m = pg(*probe_m)

        def _mean_flat(kvs):
            parts = []
            for K, V in kvs:
                parts.append(K.mean(dim=0).flatten())
                parts.append(V.mean(dim=0).flatten())
            return torch.cat(parts)

        vec_h = _mean_flat(kvs_h)
        vec_m = _mean_flat(kvs_m)
        d6    = F.cosine_similarity(vec_h.unsqueeze(0), vec_m.unsqueeze(0)).item()
    pg.train()
    return d6


# Build D6 probe batches on rank 0 (used only for D6 reporting, no gradient flow)
if rank0():
    print("Building D6 probe batches …")
    probe_h = _build_d6_probe(hourly_table,  hourly_val_idx,  HOURLY_CFG)
    probe_m = _build_d6_probe(monthly_table, monthly_val_idx, MONTHLY_CFG)

    # Initial D6 (before any training)
    d6_init = compute_d6(prefix_gen, probe_h, probe_m)
    d6_traj = [(0, d6_init)]
    print(f"  D6 INIT (step 0):  {d6_init:.4f}  "
          f"({'differentiated' if d6_init < 0.9 else '⚠ collapsing'})")
else:
    d6_traj = []


# ── KL: always enabled (H100 KL step time confirmed < 0.1s) ──────────────────

include_kl = True
if rank0():
    print(f"  KL distillation: enabled (forced)")


# ── Per-dataset training state ─────────────────────────────────────────────────

h_state = {
    'current_step':                   0,
    'total_steps':                    TOTAL_STEPS,
    'ceiling_fired_accumulator':      [],
    'pfx_total_pre_ceil_accumulator': [],
}
m_state = {
    'current_step':                   0,
    'total_steps':                    TOTAL_STEPS,
    'ceiling_fired_accumulator':      [],
    'pfx_total_pre_ceil_accumulator': [],
}

# Per-dataset EMA and filter trackers
ema_h = None;  filt_h = 0;  n_h = 0
ema_m = None;  filt_m = 0;  n_m = 0

# EMA trajectory snapshots (at each DIAG_STEP)
ema_fc_traj_h = []
ema_fc_traj_m = []


# ── Checkpoint save ────────────────────────────────────────────────────────────

CKPT_DIR = FINETUNING_ROOT / "checkpoints"

def save_checkpoint(step, fc_val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / CKPT_NAME
    torch.save({
        'step':            step,
        'pg_state':        prefix_gen.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss':            fc_val,
        'config': dict(
            d_model=D_MODEL, num_layers=NUM_LAYERS,
            prefix_len_per_component=PREFIX_LEN, rank=RANK,
            lambda_ortho=LAMBDA_ORTHO, lambda_kl=LAMBDA_KL,
            lr=1e-4, weight_decay=WEIGHT_DECAY,
            datasets=['m4hourly', 'm4monthly'],
            total_steps=TOTAL_STEPS,
            include_kl=include_kl,
        ),
        'diagnostics': dict(
            d6_traj=d6_traj,
            ema_fc_traj={'hourly': ema_fc_traj_h, 'monthly': ema_fc_traj_m},
        ),
    }, str(path))
    print(f"  Checkpoint saved → {path}")


# ── Training loop ──────────────────────────────────────────────────────────────

if rank0():
    world        = dist.get_world_size() if distributed else 1
    effective_bs = BATCH_SIZE_PER_GPU * world
    print(f"\n{'='*75}")
    print(f"JOINT TRAINING  ({TOTAL_STEPS} steps, ~{TOTAL_STEPS//2} per dataset)")
    print(f"  GPUs: {world}  |  batch/GPU: {BATCH_SIZE_PER_GPU}  |  effective batch: {effective_bs}")
    print(f"  M4 Hourly  (even steps): period=24, 96-pt context, min_p=6, max_p=24")
    print(f"  M4 Monthly (odd  steps): period=12, 36-pt context, min_p=4, max_p=12")
    print(f"  Warmup: {WARMUP_STEPS}  |  weight_decay: {WEIGHT_DECAY}  |  KL: {include_kl}")
    print(f"  D6 probe: {D6_PROBE_N} hourly + {D6_PROBE_N} monthly val samples")
    print(f"{'='*75}")
    kl_label = "   kl_loss"
    print(f"{'Step':>6}  {'dataset':>7}  {'fc_loss':>9}  {'ortho':>8}{kl_label}  {'total':>9}  {'lr':>8}")
    print("-" * 70)

stop_reason = None
t_start     = time.time()

for step in range(TOTAL_STEPS):
    is_hourly    = (step % 2 == 0)
    raw_batch    = next(iter_hourly if is_hourly else iter_monthly)
    batch        = _tokenize_batch(raw_batch)
    active_state = h_state if is_hourly else m_state
    active_state['current_step'] = step
    is_diag      = step in DIAG_STEPS
    ds_label     = 'hourly ' if is_hourly else 'monthly'

    # ── Forward + backward ────────────────────────────────────────────────────
    optimizer.zero_grad()

    with torch.no_grad():
        no_pfx_out    = chronos_model.model(
            input_ids=batch["input_ids"], attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
    no_pfx_logits = no_pfx_out.logits.detach()

    prefix_kvs = prefix_gen_ddp(batch["trend"], batch["seasonal"], batch["residual"])
    inject_prefix(chronos_model, prefix_generator=prefix_gen,
                  prefix_kvs=prefix_kvs, training_state=active_state)
    try:
        if is_diag and rank0():
            d5_attn, d5_above = compute_d5(batch)

        output     = chronos_model.model(
            input_ids=batch["input_ids"], attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
        fc_loss    = output.loss
        o_loss     = _ortho_penalty(prefix_kvs)
        kl_loss    = F.kl_div(
            F.log_softmax(output.logits / 2.0, dim=-1),
            F.softmax(no_pfx_logits / 2.0, dim=-1),
            reduction='batchmean',
        )
        total_loss = fc_loss + LAMBDA_ORTHO * o_loss + LAMBDA_KL * kl_loss
        total_loss.backward()

        if is_diag and rank0():
            d1_norms, d1_ratio = compute_d1(prefix_gen)
            d4_cos,   d4_max   = compute_d4(prefix_kvs)

        optimizer.step()
        scheduler.step()
    finally:
        remove_prefix_hooks(chronos_model)

    fc_val  = fc_loss.item()
    o_val   = o_loss.item()
    kl_val  = kl_loss.item()
    tot_val = total_loss.item()
    cur_lr  = optimizer.param_groups[0]["lr"]

    # ── Per-dataset EMA + filter tracking ─────────────────────────────────────
    if is_hourly:
        ema_h = EMA_ALPHA * fc_val + (1 - EMA_ALPHA) * ema_h if ema_h is not None else fc_val
        filt_h += float(batch["filtered"].sum().item())
        n_h += 1
    else:
        ema_m = EMA_ALPHA * fc_val + (1 - EMA_ALPHA) * ema_m if ema_m is not None else fc_val
        filt_m += float(batch["filtered"].sum().item())
        n_m += 1

    # ── Logging ────────────────────────────────────────────────────────────────
    if rank0() and step % LOG_INTERVAL == 0:
        print(f"{step:>6}  {ds_label}  {fc_val:>9.4f}  {o_val:>8.5f}"
              f"  {kl_val:>9.5f}  {tot_val:>9.4f}  {cur_lr:>8.2e}")

    if is_diag and rank0():
        print(f"         D1 ratio: {d1_ratio:.1f}×  "
              f"(t={d1_norms['trend']:.4f} s={d1_norms['seasonal']:.4f} n={d1_norms['noise']:.4f})")
        d4c = d4_cos
        print(f"         D4 max|cos|: {d4_max:.4f}  "
              f"(ts={d4c['ts']:+.4f} tr={d4c['tr']:+.4f} sr={d4c['sr']:+.4f})")
        attn_str = "  ".join(f"L{i}:{v:.3f}" for i, v in enumerate(d5_attn))
        print(f"         D5 attn [{ds_label}]:  {attn_str}  ({d5_above}/6 ≥ 5%)")
        # EMA snapshot
        print(f"         EMA fc_loss:  hourly={ema_h:.4f}  monthly={ema_m:.4f}"
              if ema_h is not None and ema_m is not None
              else f"         EMA fc_loss:  h={ema_h}  m={ema_m}")
        ema_fc_traj_h.append((step, ema_h))
        ema_fc_traj_m.append((step, ema_m))

    # ── Ceiling activation per-dataset (every 100 steps) ──────────────────────
    if rank0() and (step + 1) % 100 == 0:
        cur_ceil = min(0.5 + 0.3 * (step / TOTAL_STEPS), 0.8)
        for label, state, fc_e, n_e, filt_e in [
            ('hourly ', h_state, ema_h, n_h, filt_h),
            ('monthly', m_state, ema_m, n_m, filt_m),
        ]:
            ceil_acc = state['ceiling_fired_accumulator']
            pre_acc  = state['pfx_total_pre_ceil_accumulator']
            fr       = filt_e / (n_e * BATCH_SIZE_PER_GPU) if n_e > 0 else 0
            if ceil_acc:
                c_r  = float(np.mean(ceil_acc))
                p_m  = float(np.mean(pre_acc)) if pre_acc else float("nan")
                print(f"  [step {step+1:>4}] {label}  ceiling={cur_ceil:.3f}  "
                      f"fired={c_r:.4f}  pre_ceil_mean={p_m:.4f}  fix1b_rate={fr:.3f}")
            state['ceiling_fired_accumulator'].clear()
            state['pfx_total_pre_ceil_accumulator'].clear()

    # ── D6 at milestones ────────────────────────────────────────────────────────
    if rank0() and (step + 1) in D6_AFTER_STEPS:
        d6_val = compute_d6(prefix_gen, probe_h, probe_m)
        d6_traj.append((step + 1, d6_val))
        d6_label = ('differentiated ✓' if d6_val < 0.9
                    else '⚠ partial collapse' if d6_val < 0.97
                    else '⚠⚠ COLLAPSED')
        print(f"\n  [D6 @ step {step+1:>4}]  cosine(hourly, monthly) = {d6_val:.4f}  {d6_label}")

    # ── Checkpoint ─────────────────────────────────────────────────────────────
    if rank0() and step + 1 in CKPT_STEPS:
        save_checkpoint(step + 1, fc_val)

    # ── Stop conditions ────────────────────────────────────────────────────────
    stop = False
    if math.isnan(fc_val):
        stop_reason = f"STOP: fc_loss is NaN at step {step} ({ds_label})"
        stop = True
    elif step >= 50 and fc_val > SC_FC_MAX:
        stop_reason = f"STOP: fc_loss={fc_val:.4f} > {SC_FC_MAX} at step {step} ({ds_label})"
        stop = True
    elif is_diag and rank0():
        if step >= D1_WARMUP_GRACE and d1_ratio > SC_D1_MAX_RATIO:
            stop_reason = f"STOP: D1={d1_ratio:.1f}× > {SC_D1_MAX_RATIO}× at step {step}"
            stop = True
        elif d4_max > SC_D4_MAX_COS:
            stop_reason = f"STOP: D4={d4_max:.4f} > {SC_D4_MAX_COS} at step {step}"
            stop = True

    if distributed:
        stop_t = torch.tensor(int(stop), device=device)
        dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
        stop = bool(stop_t.item())

    if stop:
        if rank0():
            print(f"\n{'!'*75}")
            print(stop_reason or "STOP condition fired")
            print(f"{'!'*75}")
        break


# ── End-of-run report ─────────────────────────────────────────────────────────

t_elapsed  = time.time() - t_start
final_step = step + 1

# Save final checkpoint if not already saved
if not stop_reason and rank0() and final_step not in CKPT_STEPS:
    save_checkpoint(final_step, fc_val)

if rank0():
    world     = dist.get_world_size() if distributed else 1
    fr_h      = filt_h / (n_h * BATCH_SIZE_PER_GPU) if n_h > 0 else 0
    fr_m      = filt_m / (n_m * BATCH_SIZE_PER_GPU) if n_m > 0 else 0

    print(f"\n{'='*75}")
    print("END-OF-RUN REPORT  — Joint Pilot: M4 Hourly + M4 Monthly")
    print(f"{'='*75}")
    print(f"\nCompleted {final_step} steps ({n_h} hourly + {n_m} monthly) in {t_elapsed:.1f}s")
    print(f"Final total_loss: {tot_val:.4f}  (fc={fc_val:.4f}  ortho={o_val:.5f}  kl={kl_val:.5f})")
    print(f"\nPer-dataset EMA fc_loss (final):")
    print(f"  M4 Hourly:  {ema_h:.4f}  (fix1b={fr_h*100:.1f}%,  {n_h} steps)")
    print(f"  M4 Monthly: {ema_m:.4f}  (fix1b={fr_m*100:.1f}%,  {n_m} steps)")
    print(f"\nD6 trajectory (cosine similarity, hourly vs monthly prefix KV):")
    for s, v in d6_traj:
        label = ('differentiated ✓' if v < 0.9
                 else '⚠ partial' if v < 0.97 else '⚠⚠ collapsed')
        print(f"  step {s:>4}: {v:.4f}  {label}")
    print(f"\nStop conditions: {'FIRED — ' + stop_reason if stop_reason else 'none fired'}")
    print(f"{'='*75}")

    if SMOKE_TEST:
        assert not math.isnan(tot_val), "SMOKE TEST FAIL: total_loss is NaN"
        print(f"\nSMOKE TEST PASSED  ✓  (total_loss={tot_val:.4f})")

if distributed:
    cleanup_ddp()
