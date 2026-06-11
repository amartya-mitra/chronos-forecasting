#!/usr/bin/env python3
"""
train_1000steps_ddp.py — Phase 1 prefix tuning: 1000-step DDP training run.

Resumes from: checkpoints/run-200steps.pt  (key: pg_state)

DDP topology
------------
Only PrefixGenerator is DDP-wrapped; Chronos stays frozen and unwrapped.
Each GPU holds an identical Chronos copy; inject_prefix patches the LOCAL
copy using prefix_kvs from the LOCAL PrefixGenerator replica.
DDP AllReduce syncs PrefixGenerator gradients after loss.backward().
The full gradient chain is closed on each device independently.

Single-GPU fallback
-------------------
If LOCAL_RANK is absent (non-distributed launch), runs without DDP
using a standard DataLoader.

Checkpoints saved at steps 500 and 1000:
  checkpoints/run-1000steps-500.pt
  checkpoints/run-1000steps-1000.pt
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
from torch.optim.lr_scheduler import LinearLR

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

ARROW_PATH      = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
CKPT_RESUME     = FINETUNING_ROOT / "checkpoints" / "run-200steps.pt"
CKPT_500        = FINETUNING_ROOT / "checkpoints" / "run-1000steps-500.pt"
CKPT_1000       = FINETUNING_ROOT / "checkpoints" / "run-1000steps-1000.pt"
LOG_DIR         = FINETUNING_ROOT / "logs"

TOTAL_STEPS         = 1000
BATCH_SIZE_PER_GPU  = 8
TRAIN_SEED          = 42
LOG_INTERVAL        = 20
DIAG_STEPS          = {0, 200, 500, 1000}
CKPT_STEPS          = {500, 1000}
WARMUP_STEPS        = 50

D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN   # 48
RANK         = 64
LAMBDA_ORTHO = 0.01

# Stop-condition thresholds
SC_FC_MAX       = 10.0   # after step 50
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
    Wraps sarsim0-10k.arrow rows into a torch Dataset.
    STL decomposition is computed on-the-fly per item.
    """
    def __init__(self, table, indices: np.ndarray):
        self.table   = table
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = int(self.indices[i])
        ctx = np.array(self.table.column("context")[idx].as_py(),  dtype=np.float32)
        ftr = np.array(self.table.column("forecast")[idx].as_py(), dtype=np.float32)
        periods    = detect_periods(ctx)
        tr, se, re = stl_decompose(ctx, periods)
        return dict(
            context  = torch.from_numpy(ctx),
            forecast = torch.from_numpy(ftr),
            trend    = torch.from_numpy(tr.astype(np.float32)),
            seasonal = torch.from_numpy(se.astype(np.float32)),
            residual = torch.from_numpy(re.astype(np.float32)),
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
    sampler.set_epoch(0)   # single-pass 1000 steps, epoch=0 throughout
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
else:
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=2,
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


# ── PrefixGenerator ────────────────────────────────────────────────────────────

prefix_gen = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(device)

# Load 200-step checkpoint into the UNDERLYING module (not DDP wrapper)
if rank0():
    print(f"Resuming from {CKPT_RESUME} …")
ckpt = torch.load(str(CKPT_RESUME), map_location=device)
prefix_gen.load_state_dict(ckpt["pg_state"])   # key from run-200steps.pt
if rank0():
    print(f"  Loaded step {ckpt['step']} weights")

prefix_gen.train()

# Wrap only PrefixGenerator with DDP
if distributed:
    prefix_gen_ddp = DDP(prefix_gen, device_ids=[local_rank])
else:
    prefix_gen_ddp = prefix_gen   # single-GPU fallback: no wrap


# ── Optimizer and scheduler ────────────────────────────────────────────────────

# Optimizer built from prefix_gen_ddp.parameters() so DDP manages views
optimizer = torch.optim.AdamW(
    prefix_gen_ddp.parameters(),
    lr=1e-4,
    weight_decay=0.01,
)

# 50-step linear warmup: lr goes 1e-5 → 1e-4
scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=WARMUP_STEPS,
)

if rank0():
    print(f"  PrefixGenerator params: {sum(p.numel() for p in prefix_gen.parameters()):,}")
    print()


# ── Helpers ────────────────────────────────────────────────────────────────────

def ortho_penalty(prefix_kvs: list) -> torch.Tensor:
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
    """Bottleneck grad norm ratio (layer 0 inner linear, max/min across heads)."""
    norms = {}
    for name, mod_list in [
        ("trend",    pg.proj_trend),
        ("seasonal", pg.proj_seasonal),
        ("noise",    pg.proj_noise),
    ]:
        p = mod_list[0][0].weight
        norms[name] = p.grad.norm().item() if p.grad is not None else float("nan")
    vals   = [v for v in norms.values() if not math.isnan(v) and v > 0]
    ratio  = max(vals) / min(vals) if len(vals) >= 2 else float("nan")
    return norms, ratio


def compute_d4(prefix_kvs: list) -> tuple:
    """Max pairwise |cos sim| across heads (layer 0)."""
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
    """Per-layer total prefix attention (hooks must be active, no_grad)."""
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


def tokenize_batch(raw_batch: dict) -> dict:
    """Tokenize a raw collated batch from the DataLoader.

    Tokenization runs on CPU — the Chronos tokenizer's internal boundaries
    tensor is always on CPU, so ctx/forecast tensors must stay on CPU during
    the bucketize call.  Tokenized outputs are moved to device afterward.
    """
    ctx_tensor = raw_batch["context"].cpu()
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
    )


def save_checkpoint(step: int, path: Path, fc_loss: float) -> None:
    """Rank-0 only checkpoint save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step":               step,
            "pg_state":           prefix_gen.state_dict(),   # underlying module
            "optimizer_state":    optimizer.state_dict(),
            "loss":               fc_loss,
            "config": dict(
                d_model=D_MODEL, num_layers=NUM_LAYERS,
                prefix_len_per_component=PREFIX_LEN, rank=RANK,
                lambda_ortho=LAMBDA_ORTHO, lr=1e-4, weight_decay=0.01,
            ),
        },
        str(path),
    )
    print(f"  Checkpoint saved → {path}")


# ── Infinite data iterator ─────────────────────────────────────────────────────

def infinite_loader(loader):
    while True:
        yield from loader


data_iter   = infinite_loader(dataloader)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Training loop ──────────────────────────────────────────────────────────────

if rank0():
    print("=" * 70)
    world = dist.get_world_size() if distributed else 1
    effective_bs = BATCH_SIZE_PER_GPU * world
    print(f"PHASE 1 DDP TRAINING  ({TOTAL_STEPS} steps)")
    print(f"  GPUs: {world}  |  batch/GPU: {BATCH_SIZE_PER_GPU}  "
          f"|  effective batch: {effective_bs}")
    print(f"  Warmup: {WARMUP_STEPS} steps  |  Flat LR thereafter")
    print("=" * 70)
    print(f"{'Step':>6}  {'fc_loss':>9}  {'ortho':>8}  {'total':>9}  {'lr':>8}")
    print("-" * 48)

stop_reason = None
t_start     = time.time()

for step in range(TOTAL_STEPS):
    raw_batch = next(data_iter)
    batch     = tokenize_batch(raw_batch)
    is_diag   = step in DIAG_STEPS

    try:
        # Forward through DDP wrapper so AllReduce fires on backward
        prefix_kvs = prefix_gen_ddp(
            batch["trend"], batch["seasonal"], batch["residual"]
        )

        # inject_prefix uses the UNDERLYING module (no DDP hook mid-step)
        inject_prefix(chronos_model, prefix_generator=prefix_gen, prefix_kvs=prefix_kvs)

        # D5: capture while hooks are active (rank 0 only, no_grad)
        if is_diag and rank0():
            d5_attn, d5_above = compute_d5(batch)

        # Main forward
        output = chronos_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
        fc_loss    = output.loss
        o_loss     = ortho_penalty(prefix_kvs)
        total_loss = fc_loss + LAMBDA_ORTHO * o_loss

        total_loss.backward()

        # D1, D4: collect after backward (rank 0 only)
        if is_diag and rank0():
            d1_norms, d1_ratio = compute_d1(prefix_gen)
            d4_cos,   d4_max   = compute_d4(prefix_kvs)

        optimizer.step()
        if step < WARMUP_STEPS:
            scheduler.step()
        optimizer.zero_grad()

    finally:
        remove_prefix_hooks(chronos_model)

    fc_val  = fc_loss.item()
    o_val   = o_loss.item()
    tot_val = total_loss.item()
    cur_lr  = optimizer.param_groups[0]["lr"]

    # ── Logging (rank 0) ───────────────────────────────────────────────────
    if rank0() and step % LOG_INTERVAL == 0:
        print(f"{step:>6}  {fc_val:>9.4f}  {o_val:>8.5f}  {tot_val:>9.4f}  {cur_lr:>8.2e}")

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

    # ── Checkpoint saves (rank 0) ──────────────────────────────────────────
    if rank0() and step + 1 in CKPT_STEPS:
        ckpt_path = CKPT_500 if step + 1 == 500 else CKPT_1000
        save_checkpoint(step + 1, ckpt_path, fc_val)

    # ── Stop conditions (check all ranks; only rank 0 reports) ────────────
    stop = False
    if math.isnan(fc_val):
        stop_reason = f"STOP: fc_loss is NaN at step {step}"
        stop = True
    elif step >= 50 and fc_val > SC_FC_MAX:
        stop_reason = f"STOP: fc_loss={fc_val:.4f} > {SC_FC_MAX} at step {step}"
        stop = True
    elif is_diag and rank0():
        if d1_ratio > SC_D1_MAX_RATIO:
            stop_reason = f"STOP: D1 ratio={d1_ratio:.1f}× > {SC_D1_MAX_RATIO}× at step {step}"
            stop = True
        elif d4_max > SC_D4_MAX_COS:
            stop_reason = f"STOP: D4 max|cos|={d4_max:.4f} > {SC_D4_MAX_COS} at step {step}"
            stop = True

    # Broadcast stop flag across ranks so all break together
    if distributed:
        stop_t = torch.tensor(int(stop), device=device)
        dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
        stop = bool(stop_t.item())

    if stop:
        if rank0():
            print(f"\n{'!'*70}")
            print(stop_reason or "STOP condition fired")
            print(f"{'!'*70}")
        break


# ── Final checkpoint if loop completed without stop ────────────────────────────

t_elapsed = time.time() - t_start
final_step = step + 1   # last completed step (0-indexed)

if not stop_reason and rank0() and final_step not in CKPT_STEPS:
    save_checkpoint(final_step, CKPT_1000, fc_val)


# ── End-of-run report (rank 0) ─────────────────────────────────────────────────

if rank0():
    print(f"\n{'='*70}")
    print("END-OF-RUN REPORT")
    print(f"{'='*70}")
    print(f"\nCompleted {final_step} steps in {t_elapsed:.1f}s "
          f"({final_step / t_elapsed:.2f} steps/s)")
    world = dist.get_world_size() if distributed else 1
    samples_per_sec = final_step * BATCH_SIZE_PER_GPU * world / t_elapsed
    print(f"Effective throughput: {samples_per_sec:.1f} samples/s "
          f"across {world} GPU(s)")

    print(f"\nStop conditions: {'FIRED — ' + stop_reason if stop_reason else 'none fired'}")
    print(f"Checkpoints: {CKPT_500}  {CKPT_1000}")


if distributed:
    cleanup_ddp()
