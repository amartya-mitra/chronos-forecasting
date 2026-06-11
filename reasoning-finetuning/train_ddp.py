#!/usr/bin/env python3
"""
train_ddp.py — Canonical prefix-tuning training script (Phase 1 clean-up).

Supports multiple datasets via --dataset flag:
    python train_ddp.py --dataset sarsim0 --steps 2000   (default)
    python train_ddp.py --dataset gifteval --steps 1000

All Step C fixes:
  Fix 1a: min_period=6 in detect_periods()
  Fix 1b: resid_sig > 0.4 → zero decomposition
  Fix 2:  annealing prefix-attention ceiling (0.50 → 0.80 over TOTAL_STEPS)
  Fix 3b: cosine LR decay after 50-step warmup
  Fix 3c: configurable steps (default per dataset)
  Fix 3d: KL distillation — single combined backward (fc + ortho + KL)

KL consolidation (Phase 1):
  Single forward pass with prefix → computes fc_loss, ortho, AND kl_loss together.
  Single total_loss.backward().  No second inject_prefix call.
  total_loss = fc_loss + LAMBDA_ORTHO * ortho + LAMBDA_KL * kl_loss
  Logged 'total' column now reflects all optimised terms.

Smoke test: set SMOKE_TEST=1 to run 3 steps (single GPU, no DDP).
"""

import argparse
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

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def setup_ddp() -> int:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp() -> None:
    dist.destroy_process_group()

def rank0() -> bool:
    return (not is_distributed()) or dist.get_rank() == 0


# ── Dataset configuration ──────────────────────────────────────────────────────

DATASET_CONFIG = {
    'sarsim0': {
        'data_path':       'data/sarsim0-10k.arrow',
        'context_col':     'context',
        'forecast_col':    'forecast',
        'target_offset':   0,           # future = forecast[:] (full column)
        'min_period':      6,
        'max_period':      None,        # default: n//4 = 128 for 512-pt context
        'batch_size':      8,
        'default_steps':   2000,
        'ckpt_steps':      {1000, 2000},
        'ckpt_names':      {1000: 'run-ddp-1000.pt', 2000: 'run-ddp-2000.pt'},
        'diag_milestones': [0, 500, 1000, 2000],
    },
    'gifteval': {
        'data_path':       'data/m4hourly-tsf.arrow',
        'context_col':     'context',
        'forecast_col':    'target',
        'target_offset':   96,          # future = target[96:] — last 48 pts raw future
        'min_period':      6,
        'max_period':      None,        # default: n//4 = 24 for 96-pt context
        'batch_size':      4,
        'default_steps':   1000,
        'ckpt_steps':      {1000},
        'ckpt_names':      {1000: 'gifteval-1000steps.pt'},
        'diag_milestones': [0, 500, 1000],
    },
    'm4monthly': {
        'data_path':       'data/m4monthly-tsf.arrow',
        'context_col':     'context',
        'forecast_col':    'target',
        'target_offset':   36,          # future = target[36:] — last 18 pts raw future
        'min_period':      4,           # quarterly cycles meaningful for monthly data
        'max_period':      12,          # 36 // 3 = 12  (n_cycles=3 → captures annual period)
        'batch_size':      8,
        'default_steps':   2000,
        'ckpt_steps':      {2000},
        'ckpt_names':      {2000: 'sweep-m4monthly-2000steps.pt'},
        'diag_milestones': [0, 500, 1000, 2000],
    },
    'm4daily': {
        'data_path':       'data/m4daily-tsf.arrow',
        'context_col':     'context',
        'forecast_col':    'target',
        'target_offset':   0,           # future = target[:] — 14 pts raw future (no aux prefix)
        'min_period':      4,           # capture sub-weekly and weekly cycles
        'max_period':      31,          # 93 // 3 = 31 (n_cycles=3); dominant periods are 12-23 (monthly/biweekly)
        'batch_size':      8,
        'default_steps':   2000,
        'ckpt_steps':      {2000},
        'ckpt_names':      {2000: 'sweep-m4daily-2000steps.pt'},
        'diag_milestones': [0, 500, 1000, 2000],
    },
    'm4weekly': {
        'data_path':       'data/m4weekly-tsf.arrow',
        'context_col':     'context',
        'forecast_col':    'target',
        'target_offset':   0,           # future = target[:] — 13 pts raw future (no aux prefix)
        'min_period':      4,
        'max_period':      80,          # n_cycles=1: 80 // 1 = 80 (allows period=52; cycles=1.54)
        'batch_size':      4,           # small dataset (~288 train series)
        'default_steps':   500,
        'ckpt_steps':      {500},
        'ckpt_names':      {500: 'sweep-m4weekly-500steps.pt'},
        'diag_milestones': [0, 100, 250, 500],
        'weight_decay':    0.05,        # stronger regularisation for small dataset
        'warmup_steps':    30,          # proportionally shortened warmup (500 steps total)
    },
    'electricity': {
        'data_path':       'data/electricity-tsf.arrow',
        'context_col':     'context',
        'forecast_col':    'target',
        'target_offset':   0,           # future = target[:] — 24 pts raw future (no aux prefix)
        'min_period':      6,
        'max_period':      84,          # n_cycles=4: 336 // 4 = 84 (covers period=24)
        'batch_size':      4,           # N_train ~296 < 500 → small-dataset regime
        'default_steps':   500,
        'ckpt_steps':      {500},
        'ckpt_names':      {500: 'sweep-electricity-500steps.pt'},
        'diag_milestones': [0, 100, 250, 500],
        'weight_decay':    0.05,        # stronger regularisation for small dataset
        'warmup_steps':    30,          # proportionally shortened warmup (500 steps total)
    },
}


# ── Argument parsing ───────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="Prefix-tuning training for Chronos T5-Small")
_parser.add_argument("--dataset", default="sarsim0", choices=list(DATASET_CONFIG.keys()),
                     help="Dataset to train on")
_parser.add_argument("--steps", type=int, default=None,
                     help="Override total training steps (default: dataset config)")
_args, _ = _parser.parse_known_args()   # parse_known_args skips torchrun/DDP injected args

ds_cfg = DATASET_CONFIG[_args.dataset]


# ── Config ─────────────────────────────────────────────────────────────────────

ARROW_PATH  = FINETUNING_ROOT / ds_cfg['data_path']
LOG_DIR     = FINETUNING_ROOT / "logs"

SMOKE_TEST  = os.environ.get("SMOKE_TEST", "0") == "1"

_cfg_steps          = _args.steps if _args.steps is not None else ds_cfg['default_steps']
TOTAL_STEPS         = 3 if SMOKE_TEST else _cfg_steps
BATCH_SIZE_PER_GPU  = ds_cfg['batch_size']
TRAIN_SEED          = 42
LOG_INTERVAL        = 1 if SMOKE_TEST else 20
_diag_m             = ds_cfg['diag_milestones']
DIAG_STEPS          = {0, 1, 2} if SMOKE_TEST else {s for s in _diag_m if s <= TOTAL_STEPS}
CKPT_STEPS          = set() if SMOKE_TEST else {s for s in ds_cfg['ckpt_steps'] if s <= TOTAL_STEPS}
CKPT_NAMES          = ds_cfg['ckpt_names']   # step → filename

D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN   # 48
RANK         = 64
LAMBDA_ORTHO = 0.01
LAMBDA_KL    = 0.05

MIN_PERIOD   = ds_cfg.get('min_period', 6)
MAX_PERIOD   = ds_cfg.get('max_period', None)  # None → detect_periods default (n//4)
RS_THRESHOLD = 0.4
WEIGHT_DECAY: float = float(ds_cfg.get('weight_decay', 0.01))  # type: ignore[arg-type]
WARMUP_STEPS: int   = 1 if SMOKE_TEST else int(ds_cfg.get('warmup_steps', 50))  # type: ignore[arg-type]

_KL_OVERRIDE = os.environ.get("INCLUDE_KL")

# Stop-condition thresholds
SC_FC_MAX       = 10.0
SC_D1_MAX_RATIO = 50.0
SC_D4_MAX_COS   = 0.5
ATTN_THRESH     = 0.05
D1_WARMUP_GRACE = 10   # steps before D1 ratio stop guard fires


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
          + (f"  |  World size: {dist.get_world_size()}" if distributed else "")
          + f"  |  dataset={_args.dataset}")


# ── Dataset ────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Generic time series dataset from an Arrow file.

    Supports any (context_col, forecast_col) pair with an optional
    target_offset to slice the forecast column:
        future = table[forecast_col][idx][target_offset:]

    For SarSim0:  forecast_col='forecast', target_offset=0  → full column
    For GiftEval: forecast_col='target',   target_offset=96 → target[96:] (48 pts raw future)
    """

    def __init__(self, table, indices: np.ndarray, cfg: dict):
        self.table        = table
        self.indices      = indices
        self.context_col  = cfg['context_col']
        self.forecast_col = cfg['forecast_col']
        self.offset       = cfg['target_offset']

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = int(self.indices[i])
        ctx = np.array(self.table.column(self.context_col)[idx].as_py(),  dtype=np.float32)
        raw = np.array(self.table.column(self.forecast_col)[idx].as_py(), dtype=np.float32)
        ftr = raw[self.offset:] if self.offset > 0 else raw

        periods    = detect_periods(ctx, min_period=MIN_PERIOD, max_period=MAX_PERIOD)
        tr, se, re = stl_decompose(ctx, periods)

        rs       = float(np.var(re) / (np.var(ctx) + 1e-9))
        filtered = rs > RS_THRESHOLD
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


if rank0():
    print(f"Loading {ARROW_PATH.name} …")
with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_total   = len(table)
rng_split = np.random.default_rng(TRAIN_SEED)
all_idx   = rng_split.permutation(n_total)
train_idx = all_idx[: int(n_total * 0.8)]

if rank0():
    print(f"  {n_total} rows  →  {len(train_idx)} train")

train_dataset = TimeSeriesDataset(table, train_idx, ds_cfg)

if distributed:
    sampler    = DistributedSampler(train_dataset, shuffle=True)
    sampler.set_epoch(0)
    dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_PER_GPU,
        sampler=sampler, num_workers=2, pin_memory=True,
    )
else:
    dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_PER_GPU,
        shuffle=True, num_workers=2,
    )


# ── Chronos (frozen, not DDP-wrapped) ─────────────────────────────────────────

if rank0():
    print("Loading Chronos T5-Small …")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=str(device),
    dtype=torch.float32,
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


# ── Optimizer + scheduler ──────────────────────────────────────────────────────

optimizer = torch.optim.AdamW(
    prefix_gen_ddp.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY,
)
warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                        total_iters=WARMUP_STEPS)
cosine_sched = CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS - WARMUP_STEPS,
                                 eta_min=1e-5)
scheduler    = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                            milestones=[WARMUP_STEPS])

if rank0():
    print(f"  PrefixGenerator params: {sum(p.numel() for p in prefix_gen.parameters()):,}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize_batch(raw_batch: dict) -> dict:
    ctx_tensor    = raw_batch["context"].cpu()
    input_ids, attn_mask, ctx_scale = tokenizer.context_input_transform(
        pipeline._prepare_and_validate_context(ctx_tensor)
    )
    future_tensor = raw_batch["forecast"].cpu()
    label_ids, label_mask, _ = tokenizer._input_transform(
        context=future_tensor, scale=ctx_scale.cpu()
    )
    label_ids[~label_mask.bool()] = -100
    # Normalise STL components by ctx_scale so PrefixGenerator always receives
    # unit-scale inputs (same normalised space as the tokenised context).
    # ctx_scale shape: (B,) → unsqueeze to (B, 1) for broadcasting over time.
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


def _ortho_penalty(prefix_kvs: list) -> torch.Tensor:
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


def compute_d1(pg: PrefixGenerator) -> tuple:
    norms = {}
    for name, mod_list in [("trend", pg.proj_trend), ("seasonal", pg.proj_seasonal),
                           ("noise", pg.proj_noise)]:
        p = mod_list[0][0].weight
        norms[name] = p.grad.norm().item() if p.grad is not None else float("nan")
    vals  = [v for v in norms.values() if not math.isnan(v) and v > 0]
    ratio = max(vals) / min(vals) if len(vals) >= 2 else float("nan")
    return norms, ratio


def compute_d4(prefix_kvs: list) -> tuple:
    m   = PREFIX_LEN
    P_K = prefix_kvs[0][0].detach()
    Kt  = P_K[:, 0:m,   :].flatten(1)
    Ks  = P_K[:, m:2*m, :].flatten(1)
    Kr  = P_K[:, 2*m:,  :].flatten(1)
    ts  = F.cosine_similarity(Kt, Ks, dim=1).mean().item()
    tr_ = F.cosine_similarity(Kt, Kr, dim=1).mean().item()
    sr  = F.cosine_similarity(Ks, Kr, dim=1).mean().item()
    return {"ts": ts, "tr": tr_, "sr": sr}, max(abs(ts), abs(tr_), abs(sr))


def compute_d5(batch: dict) -> tuple:
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


def save_checkpoint(step: int, path: Path, fc_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step":            step,
            "pg_state":        prefix_gen.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss":            fc_loss,
            "config": dict(
                d_model=D_MODEL, num_layers=NUM_LAYERS,
                prefix_len_per_component=PREFIX_LEN, rank=RANK,
                lambda_ortho=LAMBDA_ORTHO, lambda_kl=LAMBDA_KL if include_kl else 0.0,
                lr=1e-4, weight_decay=WEIGHT_DECAY,
                min_period=MIN_PERIOD, rs_threshold=RS_THRESHOLD,
                include_kl=include_kl,
                dataset=_args.dataset,
            ),
        },
        str(path),
    )
    print(f"  Checkpoint saved → {path}")


# ── KL gate ────────────────────────────────────────────────────────────────────

def _profile_kl_step(batch: dict, state: dict) -> float:
    """Time one complete step with consolidated KL. Returns seconds."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    optimizer.zero_grad()
    with torch.no_grad():
        no_pfx_out = chronos_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
    no_pfx_logits = no_pfx_out.logits.detach()

    kvs = prefix_gen_ddp(batch["trend"], batch["seasonal"], batch["residual"])
    inject_prefix(chronos_model, prefix_generator=prefix_gen,
                  prefix_kvs=kvs, training_state=state)
    try:
        out = chronos_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
        kl = F.kl_div(
            F.log_softmax(out.logits / 2.0, dim=-1),
            F.softmax(no_pfx_logits / 2.0, dim=-1),
            reduction='batchmean',
        )
        total = out.loss + LAMBDA_ORTHO * _ortho_penalty(kvs) + LAMBDA_KL * kl
        total.backward()
        optimizer.step()
    finally:
        remove_prefix_hooks(chronos_model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter() - t0


if SMOKE_TEST:
    include_kl = True
    if rank0():
        print("  SMOKE TEST: include_kl=True (forced)")
elif _KL_OVERRIDE is not None:
    include_kl = (_KL_OVERRIDE == "1")
    if rank0():
        print(f"  KL override via INCLUDE_KL={_KL_OVERRIDE}: include_kl={include_kl}")
else:
    if rank0():
        print("  Profiling KL wall time …")
    probe_iter  = iter(dataloader)
    probe_batch = _tokenize_batch(next(probe_iter))
    probe_state = {'current_step': 0, 'total_steps': TOTAL_STEPS}

    # Warmup pass
    optimizer.zero_grad()
    kvs_w = prefix_gen_ddp(probe_batch["trend"], probe_batch["seasonal"],
                            probe_batch["residual"])
    inject_prefix(chronos_model, prefix_generator=prefix_gen,
                  prefix_kvs=kvs_w, training_state=probe_state)
    try:
        out_w = chronos_model.model(input_ids=probe_batch["input_ids"],
                                     attention_mask=probe_batch["attn_mask"],
                                     labels=probe_batch["label_ids"])
        (out_w.loss + LAMBDA_ORTHO * _ortho_penalty(kvs_w)).backward()
        optimizer.step()
    finally:
        remove_prefix_hooks(chronos_model)
    optimizer.zero_grad()

    kl_time   = _profile_kl_step(probe_batch, probe_state)
    optimizer.zero_grad()
    prefix_gen.warm_start_from_chronos(chronos_model, scale_warm=False)
    torch.manual_seed(0)

    include_kl = kl_time < 0.5
    if rank0():
        print(f"  KL step time: {kl_time:.3f}s  →  include_kl={include_kl}")

if rank0():
    print()


# ── Training state ─────────────────────────────────────────────────────────────

training_state = {
    'current_step':                   0,
    'total_steps':                    TOTAL_STEPS,
    'ceiling_fired_accumulator':      [],
    'pfx_total_pre_ceil_accumulator': [],
}

filter_count  = 0
total_batches = 0


# ── Infinite data iterator ─────────────────────────────────────────────────────

def infinite_loader(loader):
    while True:
        yield from loader

data_iter = infinite_loader(dataloader)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Training loop ──────────────────────────────────────────────────────────────

if rank0():
    print("=" * 75)
    world        = dist.get_world_size() if distributed else 1
    effective_bs = BATCH_SIZE_PER_GPU * world
    print(f"TRAINING  ({TOTAL_STEPS} steps)  dataset={_args.dataset}")
    print(f"  GPUs: {world}  |  batch/GPU: {BATCH_SIZE_PER_GPU}  "
          f"|  effective batch: {effective_bs}")
    print(f"  Fix 1a: min_period={MIN_PERIOD}  max_period={MAX_PERIOD if MAX_PERIOD is not None else 'n//4'}")
    print(f"  Fix 1b: rs_threshold={RS_THRESHOLD}")
    print(f"  Fix 2:  annealing ceiling 0.50→0.80")
    print(f"  Fix 3b: cosine LR decay (warmup={WARMUP_STEPS} steps, eta_min=1e-5)")
    print(f"  Fix 3d: KL distillation = {include_kl}")
    print(f"  KL consolidated into single backward: YES")
    print("=" * 75)
    kl_label = "   kl_loss" if include_kl else ""
    print(f"{'Step':>6}  {'fc_loss':>9}  {'ortho':>8}{kl_label}  {'total':>9}  {'lr':>8}")
    print("-" * (55 + (10 if include_kl else 0)))

stop_reason = None
t_start     = time.time()

for step in range(TOTAL_STEPS):
    raw_batch = next(data_iter)
    batch     = _tokenize_batch(raw_batch)
    is_diag   = step in DIAG_STEPS

    filter_count  += float(batch["filtered"].sum().item())
    total_batches += 1

    training_state['current_step'] = step

    # ── Zero gradients at start of step ───────────────────────────────────────
    optimizer.zero_grad()

    # ── No-prefix baseline (no_grad) — before inject, for KL ─────────────────
    if include_kl:
        with torch.no_grad():
            no_pfx_out = chronos_model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attn_mask"],
                labels=batch["label_ids"],
            )
        no_pfx_logits = no_pfx_out.logits.detach()

    # ── Prefix forward ────────────────────────────────────────────────────────
    prefix_kvs = prefix_gen_ddp(
        batch["trend"], batch["seasonal"], batch["residual"]
    )
    inject_prefix(
        chronos_model,
        prefix_generator=prefix_gen,
        prefix_kvs=prefix_kvs,
        training_state=training_state,
    )

    try:
        if is_diag and rank0():
            d5_attn, d5_above = compute_d5(batch)

        output  = chronos_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
        fc_loss = output.loss
        o_loss  = _ortho_penalty(prefix_kvs)

        if include_kl:
            kl_loss = F.kl_div(
                F.log_softmax(output.logits / 2.0, dim=-1),
                F.softmax(no_pfx_logits / 2.0, dim=-1),
                reduction='batchmean',
            )
            total_loss = fc_loss + LAMBDA_ORTHO * o_loss + LAMBDA_KL * kl_loss
        else:
            kl_loss    = torch.tensor(0.0)
            total_loss = fc_loss + LAMBDA_ORTHO * o_loss

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
    kl_val  = kl_loss.item() if include_kl else 0.0
    tot_val = total_loss.item()
    cur_lr  = optimizer.param_groups[0]["lr"]

    # ── Logging ────────────────────────────────────────────────────────────
    if rank0() and step % LOG_INTERVAL == 0:
        kl_str = f"  {kl_val:>9.5f}" if include_kl else ""
        print(f"{step:>6}  {fc_val:>9.4f}  {o_val:>8.5f}{kl_str}  "
              f"{tot_val:>9.4f}  {cur_lr:>8.2e}")

    if is_diag and rank0():
        print(f"         D1 ratio: {d1_ratio:.1f}×  "
              f"(t={d1_norms['trend']:.4f} "
              f"s={d1_norms['seasonal']:.4f} "
              f"n={d1_norms['noise']:.4f})")
        d4c = d4_cos
        print(f"         D4 max|cos|: {d4_max:.4f}  "
              f"(ts={d4c['ts']:+.4f} tr={d4c['tr']:+.4f} sr={d4c['sr']:+.4f})")
        attn_str = "  ".join(f"L{i}:{v:.3f}" for i, v in enumerate(d5_attn))
        print(f"         D5 attn:  {attn_str}  ({d5_above}/6 ≥ 5%)")

    # ── Ceiling activation logging (every 100 steps) ──────────────────────
    if rank0() and (step + 1) % 100 == 0:
        ceil_acc  = training_state['ceiling_fired_accumulator']
        pre_acc   = training_state['pfx_total_pre_ceil_accumulator']
        cur_ceil  = min(0.5 + 0.3 * (step / TOTAL_STEPS), 0.8)
        filt_rate = filter_count / (total_batches * BATCH_SIZE_PER_GPU) if total_batches > 0 else 0
        if ceil_acc:
            c_rate = np.mean(ceil_acc)
            p_mean = np.mean(pre_acc) if pre_acc else float("nan")
            print(f"  [step {step+1:>4}] ceiling={cur_ceil:.3f}  "
                  f"fired={c_rate:.4f}  pre_ceil_mean={p_mean:.4f}  "
                  f"fix1b_rate={filt_rate:.3f}")
        training_state['ceiling_fired_accumulator'].clear()
        training_state['pfx_total_pre_ceil_accumulator'].clear()

    # ── Checkpoint saves ───────────────────────────────────────────────────
    if rank0() and step + 1 in CKPT_STEPS:
        ckpt_name = CKPT_NAMES.get(step + 1, f"{_args.dataset}-{step+1}steps.pt")
        ckpt_path = FINETUNING_ROOT / "checkpoints" / ckpt_name
        save_checkpoint(step + 1, ckpt_path, fc_val)

    # ── Stop conditions ────────────────────────────────────────────────────
    stop = False
    if math.isnan(fc_val):
        stop_reason = f"STOP: fc_loss is NaN at step {step}"
        stop = True
    elif step >= 50 and fc_val > SC_FC_MAX:
        stop_reason = f"STOP: fc_loss={fc_val:.4f} > {SC_FC_MAX} at step {step}"
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

# Save final checkpoint if not already saved at an intermediate step
if not stop_reason and rank0() and final_step not in CKPT_STEPS:
    last_configured = max(CKPT_NAMES.keys()) if CKPT_NAMES else final_step
    final_ckpt_name = CKPT_NAMES.get(last_configured,
                                      f"{_args.dataset}-{final_step}steps.pt")
    save_checkpoint(final_step, FINETUNING_ROOT / "checkpoints" / final_ckpt_name, fc_val)

if rank0():
    world      = dist.get_world_size() if distributed else 1
    filt_rate  = filter_count / (total_batches * BATCH_SIZE_PER_GPU) if total_batches > 0 else 0

    print(f"\n{'='*75}")
    print("END-OF-RUN REPORT")
    print(f"{'='*75}")
    print(f"\nCompleted {final_step} steps in {t_elapsed:.1f}s "
          f"({final_step / t_elapsed:.2f} steps/s)")
    samples_ps = final_step * BATCH_SIZE_PER_GPU * world / t_elapsed
    print(f"Effective throughput: {samples_ps:.1f} samples/s across {world} GPU(s)")
    print(f"\nFinal total_loss: {tot_val:.4f}  "
          f"(fc={fc_val:.4f}  ortho={o_val:.5f}  kl={kl_val:.5f})")
    print(f"Fix 1b filter rate:  {filt_rate*100:.1f}%")
    print(f"KL distillation:     {include_kl}  (single backward)")
    print(f"Stop conditions:     {'FIRED — ' + stop_reason if stop_reason else 'none fired'}")

    if SMOKE_TEST:
        assert not math.isnan(tot_val), "SMOKE TEST FAIL: total_loss is NaN"
        assert 3.0 <= tot_val <= 15.0, f"SMOKE TEST FAIL: total_loss={tot_val:.4f} outside [3, 15]"
        print(f"\nSMOKE TEST PASSED  ✓  (total_loss={tot_val:.4f} in [3, 15])")

if distributed:
    cleanup_ddp()
