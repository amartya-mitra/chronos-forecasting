#!/usr/bin/env python3
"""
joint_stage2_train.py — Stage 2: 5-dataset round-robin joint prefix-tuning.

Datasets (round-robin, DATASET_ORDER[step % 5]):
  SarSim0     512-pt  mixed    10k series
  M4 Hourly    96-pt  24         414 series
  M4 Monthly   36-pt  12      10000 series
  M4 Daily     93-pt  7        4227 series
  M4 Weekly    80-pt  52         359 series

Electricity (336-pt, period=24, 370 series) is held out as domain-shift test.

Training:
  TOTAL_STEPS = 5000  (1,000 effective steps per dataset)
  4 × H100 via torchrun, batch = 8/GPU (effective 32)
  lr = 1e-4, weight_decay = 0.01, warmup = 100 steps, cosine thereafter

Checkpoint: checkpoints/joint-stage2-5ds-5000.pt

Smoke test: SMOKE_TEST=1 runs 10 steps.
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


# ── Dataset configs ────────────────────────────────────────────────────────────

DATASET_ORDER = ['sarsim0', 'm4hourly', 'm4monthly', 'm4daily', 'm4weekly']

DATASET_CONFIG = {
    'sarsim0': {
        'data_path':     'data/sarsim0-10k.arrow',
        'context_col':   'context',
        'forecast_col':  'forecast',
        'target_offset': 0,
        'min_period':    6,
        'max_period':    None,
        'label':         'SarSim0',
    },
    'm4hourly': {
        'data_path':     'data/m4hourly-tsf.arrow',
        'context_col':   'context',
        'forecast_col':  'target',
        'target_offset': 96,
        'min_period':    6,
        'max_period':    24,
        'label':         'M4 Hourly',
    },
    'm4monthly': {
        'data_path':     'data/m4monthly-tsf.arrow',
        'context_col':   'context',
        'forecast_col':  'target',
        'target_offset': 36,
        'min_period':    4,
        'max_period':    12,
        'label':         'M4 Monthly',
    },
    'm4daily': {
        'data_path':     'data/m4daily-tsf.arrow',
        'context_col':   'context',
        'forecast_col':  'target',
        'target_offset': 0,
        'min_period':    4,
        'max_period':    31,
        'label':         'M4 Daily',
    },
    'm4weekly': {
        'data_path':     'data/m4weekly-tsf.arrow',
        'context_col':   'context',
        'forecast_col':  'target',
        'target_offset': 0,
        'min_period':    4,
        'max_period':    80,
        'label':         'M4 Weekly',
    },
}

# ── Training hyperparameters ───────────────────────────────────────────────────

SMOKE_TEST         = os.environ.get("SMOKE_TEST", "0") == "1"

TOTAL_STEPS        = 10   if SMOKE_TEST else 5000
BATCH_SIZE_PER_GPU = 8
WARMUP_STEPS       = 2    if SMOKE_TEST else 100
WEIGHT_DECAY       = 0.01
TRAIN_SEED         = 42
LOG_INTERVAL       = 1    if SMOKE_TEST else 20
EMA_ALPHA          = 0.05

CKPT_NAME          = 'joint-stage2-5ds-5000.pt'
CKPT_STEPS         = {10}   if SMOKE_TEST else {5000}
DIAG_STEPS         = {0, 1, 2, 3, 4, 9}  if SMOKE_TEST else {0, 999, 2499, 4999}
# EMA trajectory snapshots at these steps (1-indexed milestones)
EMA_TRAJ_STEPS     = {0, 1, 9}            if SMOKE_TEST else {0, 999, 1999, 2999, 3999, 4999}
# Final D5 probe computed at end of run (all 5 datasets)
D5_PROBE_N         = 4                    if SMOKE_TEST else 16

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
    mode = "SMOKE TEST" if SMOKE_TEST else f"DDP={distributed}"
    print(f"Device: {device}  |  {mode}"
          + (f"  |  World size: {dist.get_world_size()}" if distributed else ""))
    print("Stage 2: 5-dataset round-robin (SarSim0 / M4H / M4M / M4D / M4W)")
    print("Electricity held out as domain-shift test.")


# ── Dataset class ──────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, table, indices, cfg):
        self.table        = table
        self.indices      = indices
        self.context_col  = cfg['context_col']
        self.forecast_col = cfg['forecast_col']
        self.offset       = cfg['target_offset']
        self.min_period   = cfg['min_period']
        self.max_period   = cfg.get('max_period')

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


# ── Load all 5 datasets ────────────────────────────────────────────────────────

def _load_table(ds_name, cfg):
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

tables    = {}
train_idx = {}
val_idx   = {}
for ds in DATASET_ORDER:
    tables[ds], train_idx[ds], val_idx[ds] = _load_table(ds, DATASET_CONFIG[ds])


# ── DataLoaders ────────────────────────────────────────────────────────────────

def _build_loader(ds_name):
    cfg = DATASET_CONFIG[ds_name]
    ds  = TimeSeriesDataset(tables[ds_name], train_idx[ds_name], cfg)
    if distributed:
        sampler = DistributedSampler(ds, shuffle=True)
        sampler.set_epoch(0)
        return DataLoader(ds, batch_size=BATCH_SIZE_PER_GPU, sampler=sampler,
                          num_workers=2, pin_memory=True)
    return DataLoader(ds, batch_size=BATCH_SIZE_PER_GPU, shuffle=True, num_workers=2)


loaders   = {ds: _build_loader(ds) for ds in DATASET_ORDER}
iterators = {ds: itertools.cycle(loaders[ds]) for ds in DATASET_ORDER}


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


def _build_d5_probe(ds_name, n=D5_PROBE_N):
    cfg     = DATASET_CONFIG[ds_name]
    table   = tables[ds_name]
    v_idx   = val_idx[ds_name][:min(n, len(val_idx[ds_name]))]
    ctx_list, tr_list, se_list, re_list = [], [], [], []
    for i in v_idx:
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
    input_ids_, attn_mask_, ctx_scale = tokenizer.context_input_transform(
        pipeline._prepare_and_validate_context(ctx_t)
    )
    scale = ctx_scale.to(device).unsqueeze(-1)
    tr_t  = torch.tensor(np.stack(tr_list), dtype=torch.float32).to(device) / scale
    se_t  = torch.tensor(np.stack(se_list), dtype=torch.float32).to(device) / scale
    re_t  = torch.tensor(np.stack(re_list), dtype=torch.float32).to(device) / scale

    training_state = {
        'current_step':                   TOTAL_STEPS,
        'total_steps':                    TOTAL_STEPS,
        'ceiling_fired_accumulator':      [],
        'pfx_total_pre_ceil_accumulator': [],
    }
    with torch.no_grad():
        prefix_kvs = prefix_gen(tr_t, se_t, re_t)
        inject_prefix(chronos_model, prefix_generator=prefix_gen,
                      prefix_kvs=prefix_kvs, training_state=training_state)
        try:
            enc_out = chronos_model.model.encoder(
                input_ids=input_ids_.to(device),
                attention_mask=attn_mask_.to(device),
                output_attentions=True,
            )
        finally:
            remove_prefix_hooks(chronos_model)
    layer_attn = [w[:, :, :, :PREFIX_TOTAL].mean().item() * PREFIX_TOTAL
                  for w in enc_out.attentions]
    return layer_attn, sum(1 for v in layer_attn if v >= ATTN_THRESH)


# ── Per-dataset training state, EMA, and filter trackers ──────────────────────

states = {
    ds: {
        'current_step':                   0,
        'total_steps':                    TOTAL_STEPS,
        'ceiling_fired_accumulator':      [],
        'pfx_total_pre_ceil_accumulator': [],
    }
    for ds in DATASET_ORDER
}

ema   = {ds: None for ds in DATASET_ORDER}
filt  = {ds: 0    for ds in DATASET_ORDER}
n_ds  = {ds: 0    for ds in DATASET_ORDER}

ema_fc_traj = {ds: [] for ds in DATASET_ORDER}


# ── Checkpoint ─────────────────────────────────────────────────────────────────

CKPT_DIR = FINETUNING_ROOT / "checkpoints"

def save_checkpoint(step, fc_val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / CKPT_NAME
    fr = {ds: filt[ds] / (n_ds[ds] * BATCH_SIZE_PER_GPU) if n_ds[ds] > 0 else 0.0
          for ds in DATASET_ORDER}
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
            datasets=DATASET_ORDER,
            total_steps=TOTAL_STEPS,
        ),
        'diagnostics': dict(
            ema_fc_traj=ema_fc_traj,
            fix1b_rates=fr,
        ),
    }, str(path))
    print(f"  Checkpoint saved → {path}")


# ── Training loop ──────────────────────────────────────────────────────────────

if rank0():
    world        = dist.get_world_size() if distributed else 1
    effective_bs = BATCH_SIZE_PER_GPU * world
    print(f"\n{'='*75}")
    print(f"STAGE 2 JOINT TRAINING  ({TOTAL_STEPS} steps, ~{TOTAL_STEPS//5} per dataset)")
    print(f"  GPUs: {world}  |  batch/GPU: {BATCH_SIZE_PER_GPU}  |  effective batch: {effective_bs}")
    print(f"  Datasets: {' → '.join(DATASET_ORDER)} (cycle)")
    print(f"  Warmup: {WARMUP_STEPS}  |  weight_decay: {WEIGHT_DECAY}")
    print(f"{'='*75}")
    print(f"{'Step':>6}  {'dataset':>9}  {'fc_loss':>9}  {'ortho':>8}  {'kl_loss':>9}  {'total':>9}  {'lr':>8}")
    print("-" * 75)

stop_reason = None
t_start     = time.time()

for step in range(TOTAL_STEPS):
    ds_name      = DATASET_ORDER[step % 5]
    raw_batch    = next(iterators[ds_name])
    batch        = _tokenize_batch(raw_batch)
    active_state = states[ds_name]
    active_state['current_step'] = step
    is_diag      = step in DIAG_STEPS
    ds_label     = DATASET_CONFIG[ds_name]['label']

    optimizer.zero_grad()

    with torch.no_grad():
        no_pfx_out = chronos_model.model(
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

    ema[ds_name]  = (EMA_ALPHA * fc_val + (1 - EMA_ALPHA) * ema[ds_name]
                     if ema[ds_name] is not None else fc_val)
    filt[ds_name] += float(batch["filtered"].sum().item())
    n_ds[ds_name] += 1

    if rank0() and step % LOG_INTERVAL == 0:
        print(f"{step:>6}  {ds_label:>9}  {fc_val:>9.4f}  {o_val:>8.5f}"
              f"  {kl_val:>9.5f}  {tot_val:>9.4f}  {cur_lr:>8.2e}")

    if is_diag and rank0():
        print(f"         D1 ratio: {d1_ratio:.1f}×  "
              f"(t={d1_norms['trend']:.4f} s={d1_norms['seasonal']:.4f} n={d1_norms['noise']:.4f})")
        d4c = d4_cos
        print(f"         D4 max|cos|: {d4_max:.4f}  "
              f"(ts={d4c['ts']:+.4f} tr={d4c['tr']:+.4f} sr={d4c['sr']:+.4f})")
        attn_str = "  ".join(f"L{i}:{v:.3f}" for i, v in enumerate(d5_attn))
        print(f"         D5 attn [{ds_label}]:  {attn_str}  ({d5_above}/6 ≥ 5%)")
        ema_str = "  ".join(
            f"{ds}:{ema[ds]:.4f}" if ema[ds] is not None else f"{ds}:—"
            for ds in DATASET_ORDER
        )
        print(f"         EMA fc_loss: {ema_str}")

    if step in EMA_TRAJ_STEPS and rank0():
        for ds in DATASET_ORDER:
            ema_fc_traj[ds].append((step, ema[ds]))

    if rank0() and (step + 1) % 100 == 0:
        cur_ceil = min(0.5 + 0.3 * (step / TOTAL_STEPS), 0.8)
        for ds in DATASET_ORDER:
            state   = states[ds]
            ceil_acc = state['ceiling_fired_accumulator']
            pre_acc  = state['pfx_total_pre_ceil_accumulator']
            fr_ds    = filt[ds] / (n_ds[ds] * BATCH_SIZE_PER_GPU) if n_ds[ds] > 0 else 0
            if ceil_acc:
                c_r  = float(np.mean(ceil_acc))
                p_m  = float(np.mean(pre_acc)) if pre_acc else float("nan")
                lbl  = DATASET_CONFIG[ds]['label']
                print(f"  [step {step+1:>4}] {lbl:>9}  ceiling={cur_ceil:.3f}  "
                      f"fired={c_r:.4f}  pre_ceil_mean={p_m:.4f}  fix1b={fr_ds:.3f}")
            state['ceiling_fired_accumulator'].clear()
            state['pfx_total_pre_ceil_accumulator'].clear()

    if rank0() and step + 1 in CKPT_STEPS:
        save_checkpoint(step + 1, fc_val)

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


# ── End-of-run: per-dataset D5 comparison ─────────────────────────────────────

if rank0() and not stop_reason:
    print("\nComputing final D5 for all 5 datasets …")
    d5_final = {}
    for ds in DATASET_ORDER:
        lbl = DATASET_CONFIG[ds]['label']
        try:
            la, ab = _build_d5_probe(ds)
            d5_final[ds] = {'layer_attn': la, 'above_thresh': ab}
            attn_str = "  ".join(f"L{i}:{v:.3f}" for i, v in enumerate(la))
            print(f"  D5 [{lbl}]:  {attn_str}  ({ab}/6 ≥ 5%)")
        except Exception as exc:
            print(f"  D5 [{lbl}] failed: {exc}")
            d5_final[ds] = None

    # Save d5_final into checkpoint diagnostics
    ckpt_path = CKPT_DIR / CKPT_NAME
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        ckpt['diagnostics']['d5_final'] = d5_final
        torch.save(ckpt, str(ckpt_path))
        print(f"  D5 final appended to checkpoint.")


# ── End-of-run report ─────────────────────────────────────────────────────────

t_elapsed  = time.time() - t_start
final_step = step + 1

if rank0():
    world = dist.get_world_size() if distributed else 1
    print(f"\n{'='*75}")
    print("END-OF-RUN REPORT  — Stage 2: 5-dataset joint training")
    print(f"{'='*75}")
    print(f"\nCompleted {final_step} steps in {t_elapsed:.1f}s")

    # Per-dataset EMA summary
    print(f"\nPer-dataset EMA fc_loss (final):")
    for ds in DATASET_ORDER:
        lbl  = DATASET_CONFIG[ds]['label']
        fr   = filt[ds] / (n_ds[ds] * BATCH_SIZE_PER_GPU) if n_ds[ds] > 0 else 0
        ema_v = f"{ema[ds]:.4f}" if ema[ds] is not None else "N/A"
        print(f"  {lbl:>12}:  ema={ema_v}  fix1b={fr*100:.1f}%  n_steps={n_ds[ds]}")

    # EMA trajectories
    print(f"\nEMA fc_loss trajectories (snapshots):")
    for ds in DATASET_ORDER:
        lbl = DATASET_CONFIG[ds]['label']
        traj_str = "  ".join(f"@{s}:{v:.4f}" if v is not None else f"@{s}:—"
                             for s, v in ema_fc_traj[ds])
        print(f"  {lbl:>12}: {traj_str}")

    if not stop_reason and SMOKE_TEST:
        assert not math.isnan(tot_val), "SMOKE TEST FAIL: total_loss is NaN"
        print(f"\nSMOKE TEST PASSED  ✓  (total_loss={tot_val:.4f})")

    print(f"\nStop conditions: {'FIRED — ' + stop_reason if stop_reason else 'none fired'}")
    print(f"{'='*75}")

if distributed:
    cleanup_ddp()
