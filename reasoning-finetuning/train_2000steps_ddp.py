#!/usr/bin/env python3
"""
train_2000steps_ddp.py — Step C prefix tuning: 2000-step DDP training.

All three Step C fixes applied:
  Fix 1a: min_period=6 in detect_periods() (eliminates spurious 2,3,4,5 periods)
  Fix 1b: resid_sig > 0.4 → zero decomposition (neutral prefix for noisy series)
  Fix 2:  annealing prefix-attention ceiling in inject_prefix()
  Fix 3b: cosine LR decay after 50-step warmup
  Fix 3c: 2000 steps
  Fix 3d: KL distillation (included if wall time < 0.5s/step, else dropped)

Note on Fix 3a (pre-computed decompositions):
  sarsim0-10k.arrow stores only 64-step forecast-window trend/seasonal, not the
  512-step context-window decompositions needed by PrefixGenerator. Fix 3a is
  therefore inapplicable to this dataset; on-the-fly STL is kept, but the
  min_period=6 fix (Fix 1a) still eliminates the dominant failure-mode artefacts.

DDP topology: same as train_1000steps_ddp.py
  Only PrefixGenerator is DDP-wrapped; Chronos stays frozen and unwrapped.

Fresh warm-start (no checkpoint resume): warm_start_from_chronos(scale_warm=False)

Checkpoints:
  checkpoints/run-2000steps-1000.pt
  checkpoints/run-2000steps-2000.pt
"""

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


# ── Config ─────────────────────────────────────────────────────────────────────

ARROW_PATH   = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
CKPT_1000    = FINETUNING_ROOT / "checkpoints" / "run-2000steps-1000.pt"
CKPT_2000    = FINETUNING_ROOT / "checkpoints" / "run-2000steps-2000.pt"
LOG_DIR      = FINETUNING_ROOT / "logs"

TOTAL_STEPS         = 2000
BATCH_SIZE_PER_GPU  = 8
TRAIN_SEED          = 42
LOG_INTERVAL        = 20
DIAG_STEPS          = {0, 500, 1000, 2000}
CKPT_STEPS          = {1000, 2000}
WARMUP_STEPS        = 50

D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN   # 48
RANK         = 64
LAMBDA_ORTHO = 0.01
LAMBDA_KL    = 0.05

# Fix 1a: raised from 2 → 6 (eliminates periods 2, 3, 4, 5 — artefacts from
# FFT noise peaks in high-resid series; shortest true SarSim0 period = 7)
MIN_PERIOD   = 6
# Fix 1b: residual ratio threshold above which decomposition is zeroed
RS_THRESHOLD = 0.4

# KL gate: measured by preflight_fixC.py; set True/False before submission
# (auto-detected by profiling one batch at startup if env var not set)
_KL_OVERRIDE = os.environ.get("INCLUDE_KL")  # "1" or "0"

# Stop-condition thresholds
SC_FC_MAX       = 10.0
SC_D1_MAX_RATIO = 50.0
SC_D4_MAX_COS   = 0.5
ATTN_THRESH     = 0.05


# ── DDP / device setup ─────────────────────────────────────────────────────────

distributed = "LOCAL_RANK" in os.environ
if distributed:
    local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")
else:
    local_rank = 0
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if rank0():
    print(f"Device: {device}  |  Distributed: {distributed}"
          + (f"  |  World size: {dist.get_world_size()}" if distributed else ""))


# ── Dataset ────────────────────────────────────────────────────────────────────

class SarSim0Dataset(Dataset):
    """
    SarSim0 dataset with Fix 1a (min_period=6) and Fix 1b (resid_sig filter).

    Fix 1a: detect_periods(ctx, min_period=6) eliminates FFT noise peaks at
            periods 2–5 that were the dominant artefact in Step B outliers.

    Fix 1b: if resid_sig = var(resid)/var(ctx) > 0.4, replace the computed
            trend/seasonal/residual with zeros. PrefixGenerator then learns
            to produce a neutral prefix for noisy/chaotic series rather than
            confidently encoding spurious short-period signatures. The series
            is still trained on (forecast loss computed normally).
    """
    def __init__(self, table, indices: np.ndarray):
        self.table   = table
        self.indices = indices
        # Mutable filter counter — shared across workers via Manager or
        # simply accumulated by the rank-0 training loop via the 'filtered'
        # field in each returned item.
        self.filter_count = 0
        self.total_count  = 0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = int(self.indices[i])
        ctx = np.array(self.table.column("context")[idx].as_py(),  dtype=np.float32)
        ftr = np.array(self.table.column("forecast")[idx].as_py(), dtype=np.float32)

        # Fix 1a: min_period=6
        periods    = detect_periods(ctx, min_period=MIN_PERIOD)
        tr, se, re = stl_decompose(ctx, periods)

        # Fix 1b: zero decomposition for high-noise series
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
    print("Loading sarsim0-10k.arrow …")
with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_total   = len(table)
rng_split = np.random.default_rng(TRAIN_SEED)
all_idx   = rng_split.permutation(n_total)
train_idx = all_idx[: int(n_total * 0.8)]

if rank0():
    print(f"  {n_total} rows  →  {len(train_idx)} train")

train_dataset = SarSim0Dataset(table, train_idx)

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


# ── Optimizer + scheduler (Fix 3b: cosine decay after warmup) ─────────────────

optimizer = torch.optim.AdamW(
    prefix_gen_ddp.parameters(),
    lr=1e-4,
    weight_decay=0.01,
)

warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=WARMUP_STEPS)
cosine_sched  = CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS - WARMUP_STEPS,
                                   eta_min=1e-5)
scheduler     = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                              milestones=[WARMUP_STEPS])

if rank0():
    print(f"  PrefixGenerator params: {sum(p.numel() for p in prefix_gen.parameters()):,}")
    print(f"  Scheduler: {WARMUP_STEPS}-step linear warmup → cosine to 1e-5")


# ── KL gate: auto-detect or use env override ───────────────────────────────────

def _profile_kl_step(batch: dict, state: dict) -> float:
    """Time one complete training step WITH KL. Returns seconds."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    optimizer.zero_grad()
    try:
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
        loss = out.loss + LAMBDA_ORTHO * _ortho_penalty(kvs) + LAMBDA_KL * kl
        loss.backward()
        optimizer.step()
    finally:
        remove_prefix_hooks(chronos_model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter() - t0


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
    return dict(
        input_ids = input_ids.to(device),
        attn_mask = attn_mask.to(device),
        label_ids = label_ids.to(device),
        trend     = raw_batch["trend"].to(device),
        seasonal  = raw_batch["seasonal"].to(device),
        residual  = raw_batch["residual"].to(device),
        filtered  = raw_batch["filtered"],
    )


# Helpers needed before KL profiling ──────────────────────────────────────────

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


# Determine KL inclusion
if _KL_OVERRIDE is not None:
    include_kl = (_KL_OVERRIDE == "1")
    if rank0():
        print(f"  KL override via INCLUDE_KL={_KL_OVERRIDE}: include_kl={include_kl}")
else:
    if rank0():
        print("  Profiling KL wall time …")
    # Build one warmup batch for profiling
    probe_iter  = iter(dataloader)
    probe_batch = _tokenize_batch(next(probe_iter))
    probe_state = {'current_step': 0, 'total_steps': TOTAL_STEPS}

    # Warmup pass (not timed)
    optimizer.zero_grad()
    try:
        kvs_w = prefix_gen_ddp(probe_batch["trend"], probe_batch["seasonal"],
                                probe_batch["residual"])
        inject_prefix(chronos_model, prefix_generator=prefix_gen,
                      prefix_kvs=kvs_w, training_state=probe_state)
        out_w = chronos_model.model(input_ids=probe_batch["input_ids"],
                                     attention_mask=probe_batch["attn_mask"],
                                     labels=probe_batch["label_ids"])
        (out_w.loss + LAMBDA_ORTHO * _ortho_penalty(kvs_w)).backward()
        optimizer.step()
    finally:
        remove_prefix_hooks(chronos_model)
        optimizer.zero_grad()

    kl_time  = _profile_kl_step(probe_batch, probe_state)
    optimizer.zero_grad()   # reset after profiling
    prefix_gen.warm_start_from_chronos(chronos_model, scale_warm=False)  # re-init weights
    torch.manual_seed(0)

    include_kl = kl_time < 0.5
    if rank0():
        print(f"  KL step time: {kl_time:.3f}s  →  "
              f"include_kl={include_kl} ({'< 0.5s' if include_kl else '≥ 0.5s — dropped'})")

if rank0():
    print()


# ── Diagnostics helpers ────────────────────────────────────────────────────────

def compute_d1(pg: PrefixGenerator) -> tuple:
    norms = {}
    for name, mod_list in [
        ("trend",    pg.proj_trend),
        ("seasonal", pg.proj_seasonal),
        ("noise",    pg.proj_noise),
    ]:
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
                lr=1e-4, weight_decay=0.01,
                min_period=MIN_PERIOD, rs_threshold=RS_THRESHOLD,
                include_kl=include_kl,
            ),
        },
        str(path),
    )
    print(f"  Checkpoint saved → {path}")


# ── Training state (Fix 2 ceiling + filter tracking) ──────────────────────────

training_state = {
    'current_step':                   0,
    'total_steps':                    TOTAL_STEPS,
    'ceiling_fired_accumulator':      [],   # per-layer per-step firing rates
    'pfx_total_pre_ceil_accumulator': [],   # pre-ceiling pfx_total values
}

filter_count  = 0   # Fix 1b trigger count across all steps
total_batches = 0


# ── Infinite data iterator ─────────────────────────────────────────────────────

def infinite_loader(loader):
    while True:
        yield from loader

data_iter   = infinite_loader(dataloader)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Training loop ──────────────────────────────────────────────────────────────

if rank0():
    print("=" * 75)
    world = dist.get_world_size() if distributed else 1
    effective_bs = BATCH_SIZE_PER_GPU * world
    print(f"STEP C DDP TRAINING  ({TOTAL_STEPS} steps)")
    print(f"  GPUs: {world}  |  batch/GPU: {BATCH_SIZE_PER_GPU}  "
          f"|  effective batch: {effective_bs}")
    print(f"  Fix 1a: min_period={MIN_PERIOD}")
    print(f"  Fix 1b: rs_threshold={RS_THRESHOLD}")
    print(f"  Fix 2:  annealing ceiling 0.50→0.80")
    print(f"  Fix 3b: cosine LR decay (warmup={WARMUP_STEPS} steps, eta_min=1e-5)")
    print(f"  Fix 3d: KL distillation = {include_kl}")
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

    # Track Fix 1b filter rate
    filter_count  += float(batch["filtered"].sum().item())
    total_batches += 1

    # Update training state for ceiling annealing
    training_state['current_step'] = step

    try:
        # Forward through DDP wrapper (AllReduce on backward)
        prefix_kvs = prefix_gen_ddp(
            batch["trend"], batch["seasonal"], batch["residual"]
        )

        # inject_prefix: ceiling active via training_state
        inject_prefix(
            chronos_model,
            prefix_generator=prefix_gen,
            prefix_kvs=prefix_kvs,
            training_state=training_state,
        )

        # D5 while hooks are active (rank 0, no_grad)
        if is_diag and rank0():
            d5_attn, d5_above = compute_d5(batch)

        # Main forward (with prefix)
        output  = chronos_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
        fc_loss = output.loss
        o_loss  = _ortho_penalty(prefix_kvs)

        if include_kl:
            # Baseline forward (no prefix) for KL — hooks already removed by
            # the finally block below after the main forward completes
            pass   # KL computed after finally (see below)

        total_loss = fc_loss + LAMBDA_ORTHO * o_loss
        total_loss.backward()

        # D1, D4 after backward
        if is_diag and rank0():
            d1_norms, d1_ratio = compute_d1(prefix_gen)
            d4_cos,   d4_max   = compute_d4(prefix_kvs)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    finally:
        remove_prefix_hooks(chronos_model)

    # ── Fix 3d: KL loss (separate forward, no prefix) ─────────────────────────
    if include_kl:
        with torch.no_grad():
            no_pfx_out = chronos_model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attn_mask"],
                labels=batch["label_ids"],
            )
        no_pfx_logits = no_pfx_out.logits.detach()

        # Re-run prefix forward to get logits with gradients
        try:
            prefix_kvs_kl = prefix_gen_ddp(
                batch["trend"], batch["seasonal"], batch["residual"]
            )
            inject_prefix(
                chronos_model,
                prefix_generator=prefix_gen,
                prefix_kvs=prefix_kvs_kl,
                training_state=training_state,
            )
            out_kl = chronos_model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attn_mask"],
                labels=batch["label_ids"],
            )
            kl_loss = F.kl_div(
                F.log_softmax(out_kl.logits / 2.0, dim=-1),
                F.softmax(no_pfx_logits / 2.0, dim=-1),
                reduction='batchmean',
            )
            kl_total = LAMBDA_KL * kl_loss
            kl_total.backward()
        finally:
            remove_prefix_hooks(chronos_model)

        optimizer.step()
        optimizer.zero_grad()
    else:
        kl_loss = torch.tensor(0.0)

    fc_val  = fc_loss.item()
    o_val   = o_loss.item()
    kl_val  = kl_loss.item() if include_kl else 0.0
    tot_val = total_loss.item()
    cur_lr  = optimizer.param_groups[0]["lr"]

    # ── Logging (rank 0) ───────────────────────────────────────────────────
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

    # ── Checkpoint saves (rank 0) ──────────────────────────────────────────
    if rank0() and step + 1 in CKPT_STEPS:
        ckpt_path = CKPT_1000 if step + 1 == 1000 else CKPT_2000
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
        if d1_ratio > SC_D1_MAX_RATIO:
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


# ── Final checkpoint if loop completed without stop ────────────────────────────

t_elapsed  = time.time() - t_start
final_step = step + 1

if not stop_reason and rank0() and final_step not in CKPT_STEPS:
    save_checkpoint(final_step, CKPT_2000, fc_val)


# ── End-of-run report ─────────────────────────────────────────────────────────

if rank0():
    world       = dist.get_world_size() if distributed else 1
    filt_rate   = filter_count / (total_batches * BATCH_SIZE_PER_GPU) if total_batches > 0 else 0

    print(f"\n{'='*75}")
    print("STEP C END-OF-RUN REPORT")
    print(f"{'='*75}")
    print(f"\nCompleted {final_step} steps in {t_elapsed:.1f}s "
          f"({final_step / t_elapsed:.2f} steps/s)")
    samples_ps = final_step * BATCH_SIZE_PER_GPU * world / t_elapsed
    print(f"Effective throughput: {samples_ps:.1f} samples/s across {world} GPU(s)")

    print(f"\nFix 1b filter rate:  {filt_rate*100:.1f}%  "
          f"({int(filter_count)} series zeroed across {total_batches} batches)")
    print(f"KL distillation:     {include_kl}")
    print(f"Stop conditions:     {'FIRED — ' + stop_reason if stop_reason else 'none fired'}")
    print(f"Checkpoints:         {CKPT_1000}  {CKPT_2000}")


if distributed:
    cleanup_ddp()
