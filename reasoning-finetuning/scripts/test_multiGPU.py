#!/usr/bin/env python3
"""
Smoke test for multi-GPU training with DDP (or single-GPU fallback).

Tests the exact failure pattern from job 34881:
  5 training steps → eval (calls inner.generate()) → 5 training steps → assert no crash.

Usage:
  Single GPU:  python test_multiGPU.py
  Multi-GPU:   torchrun --nproc_per_node=4 test_multiGPU.py
"""

import os
import sys
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist
import pyarrow.ipc as ipc
from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from chronos import ChronosPipeline

# ── constants (TSF_M4M config) ─────────────────────────────────────────────────
MODEL_ID       = "amazon/chronos-t5-small"
DATA_PATH      = FINETUNING_ROOT / "data" / "m4monthly-tsf.arrow"
EXTENDED_VOCAB = 4105
SEP_1_TOKEN_ID = 4103
SEP_2_TOKEN_ID = 4104
MODE_TOKEN_ID  = 4102   # TSF
SEGMENT_LEN    = 18
CTX_LEN        = 36
PRED_LEN       = 54     # 3 × 18
GEN_LEN        = 56     # 54 + 2 SEPs
FORCED         = [(19, SEP_1_TOKEN_ID), (38, SEP_2_TOKEN_ID)]
N_PARTS        = 3
BATCH_SIZE     = 8      # small for quick test
EVAL_SAMPLES   = 5
N_TRAIN_STEPS  = 5      # per phase

# ── DDP init ──────────────────────────────────────────────────────────────────

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
IS_DDP     = LOCAL_RANK != -1

if IS_DDP:
    dist.init_process_group(backend="nccl")
    device     = torch.device(f"cuda:{LOCAL_RANK}")
    WORLD_SIZE = dist.get_world_size()
    IS_MAIN    = (LOCAL_RANK == 0)
    torch.cuda.set_device(LOCAL_RANK)
else:
    device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    WORLD_SIZE = 1
    IS_MAIN    = True

if IS_MAIN:
    print(f"{'='*60}")
    print(f"Smoke test: DDP={IS_DDP}  world_size={WORLD_SIZE}  device={device}")
    print(f"Data: {DATA_PATH}")
    print(f"{'='*60}")


# ── helpers ───────────────────────────────────────────────────────────────────

class SEPForcer(LogitsProcessor):
    def __init__(self, forced):
        self._forced = {pos - 1: tok for pos, tok in forced}

    def __call__(self, input_ids, scores):
        cur = input_ids.shape[-1] - 1
        if cur in self._forced:
            scores[:] = -float("inf")
            scores[:, self._forced[cur]] = 0.0
        return scores


def load_rows(n: int = 50):
    with ipc.open_file(str(DATA_PATH)) as reader:
        tbl = reader.read_all()
    rows = []
    for i in range(min(n, len(tbl))):
        ctx = np.array(tbl["context"][i].as_py(), dtype=np.float32)
        tgt = np.array(tbl["target"][i].as_py(),  dtype=np.float32)
        rows.append({"context": ctx, "target": tgt})
    return rows


def pretokenise_rows(rows, tokenizer):
    contexts = torch.tensor(np.stack([r["context"] for r in rows]), dtype=torch.float32)
    targets  = torch.tensor(np.stack([r["target"]  for r in rows]), dtype=torch.float32)

    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    label_ids, label_mask, _   = tokenizer._input_transform(context=targets, scale=scale)

    sep1 = torch.full((len(rows), 1), SEP_1_TOKEN_ID, dtype=label_ids.dtype)
    sep2 = torch.full((len(rows), 1), SEP_2_TOKEN_ID, dtype=label_ids.dtype)
    ones = torch.ones((len(rows), 1), dtype=label_mask.dtype)

    p0, p1, p2 = [label_ids[:, i*SEGMENT_LEN:(i+1)*SEGMENT_LEN] for i in range(N_PARTS)]
    m0, m1, m2 = [label_mask[:, i*SEGMENT_LEN:(i+1)*SEGMENT_LEN] for i in range(N_PARTS)]
    mode_tok = torch.full((len(rows), 1), MODE_TOKEN_ID, dtype=label_ids.dtype)
    eos_tok  = torch.full((len(rows), 1), 1, dtype=label_ids.dtype)  # EOS=1

    labels        = torch.cat([p0, sep1, p1, sep2, p2, eos_tok], dim=1)
    dec_input_ids = torch.cat([mode_tok, p0, sep1, p1, sep2, p2], dim=1)
    label_att     = torch.cat([m0, ones, m1, ones, m2, ones], dim=1)
    # Mask EOS in labels: keep as-is (loss will include EOS prediction)
    labels[labels == -1] = -100  # ignore padding tokens in loss

    out = []
    for i in range(len(rows)):
        out.append({
            "input_ids":     token_ids[i],
            "attn_mask":     attn_mask[i],
            "dec_input_ids": dec_input_ids[i],
            "labels":        labels[i],
        })
    return out


def make_batch(tokenised, indices, device):
    batch = [tokenised[i % len(tokenised)] for i in indices]
    ids  = torch.stack([b["input_ids"]     for b in batch]).to(device)
    mask = torch.stack([b["attn_mask"]     for b in batch]).to(device)
    dec  = torch.stack([b["dec_input_ids"] for b in batch]).to(device)
    lbl  = torch.stack([b["labels"]        for b in batch]).to(device)
    return ids, mask, dec, lbl


@torch.no_grad()
def run_eval(inner, tokenizer, eval_rows, device):
    """Run a single eval batch using inner.generate() — the exact operation that triggered the crash."""
    inner.eval()
    ctx_t = torch.tensor(np.stack([r["context"] for r in eval_rows]), dtype=torch.float32)
    tids, amask, scale = tokenizer.context_input_transform(ctx_t)
    lp = LogitsProcessorList([SEPForcer(FORCED)])
    out = inner.generate(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        logits_processor=lp,
        generation_config=GenerationConfig(
            min_new_tokens=GEN_LEN, max_new_tokens=GEN_LEN,
            do_sample=True, num_return_sequences=EVAL_SAMPLES,
            decoder_start_token_id=MODE_TOKEN_ID,
            eos_token_id=1, pad_token_id=0,
            temperature=1.0, top_k=50, top_p=1.0,
        ),
    )
    inner.train()
    return out.shape  # Just verify no crash; return shape for logging


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Load data
    if IS_MAIN:
        print("\nLoading data...")
    rows = load_rows(n=60)
    eval_rows  = rows[:10]
    train_rows = rows[10:]

    # Load model
    if IS_MAIN:
        print(f"Loading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    inner    = pipeline.model.model
    inner.resize_token_embeddings(EXTENDED_VOCAB)
    inner = inner.to(device)
    inner.train()

    # Wrap with DDP
    if IS_DDP:
        from torch.nn.parallel import DistributedDataParallel as DDP
        dp_model = DDP(inner, device_ids=[LOCAL_RANK])
        if IS_MAIN:
            print(f"  DDP: {WORLD_SIZE} ranks")
    else:
        dp_model = inner
        if IS_MAIN:
            print(f"  Single GPU: {device}")

    # Pre-tokenise
    if IS_MAIN:
        print("Pre-tokenising...")
    tokenised = pretokenise_rows(train_rows, pipeline.tokenizer)
    per_rank  = max(1, BATCH_SIZE // WORLD_SIZE)

    # Shard by rank
    shard_tok = [tokenised[i] for i in range(len(tokenised)) if i % WORLD_SIZE == (LOCAL_RANK if IS_DDP else 0)]
    if len(shard_tok) == 0:
        shard_tok = tokenised  # fallback if too few rows

    optimizer = torch.optim.AdamW(inner.parameters(), lr=1e-4)

    # ── Phase 1: train 5 steps ─────────────────────────────────────────────────
    if IS_MAIN:
        print(f"\nPhase 1: {N_TRAIN_STEPS} training steps...")
    for step in range(N_TRAIN_STEPS):
        indices = list(range(step * per_rank, (step + 1) * per_rank))
        ids, mask, dec, lbl = make_batch(shard_tok, indices, device)
        out  = dp_model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec, labels=lbl)
        loss = out.loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), 1.0)
        optimizer.step()
        if IS_MAIN:
            print(f"  train step {step+1}/{N_TRAIN_STEPS}  loss={loss.item():.4f}")

    # ── Eval (only rank 0, others wait) ────────────────────────────────────────
    if IS_DDP:
        dist.barrier()  # Sync before eval

    if IS_MAIN:
        print(f"\nRunning eval (inner.generate() on rank 0)...")
        out_shape = run_eval(inner, pipeline.tokenizer, eval_rows, device)
        print(f"  generate() output shape: {out_shape}  ✓")

    if IS_DDP:
        dist.barrier()  # Sync after eval

    # ── Phase 2: train 5 more steps (the failing step in job 34881) ────────────
    if IS_MAIN:
        print(f"\nPhase 2: {N_TRAIN_STEPS} training steps after eval...")
    inner.train()
    for step in range(N_TRAIN_STEPS):
        indices = list(range((step + N_TRAIN_STEPS) * per_rank, (step + N_TRAIN_STEPS + 1) * per_rank))
        ids, mask, dec, lbl = make_batch(shard_tok, indices, device)
        out  = dp_model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec, labels=lbl)
        loss = out.loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), 1.0)
        optimizer.step()
        if IS_MAIN:
            print(f"  train step {N_TRAIN_STEPS+step+1}/{2*N_TRAIN_STEPS}  loss={loss.item():.4f}")

    if IS_DDP:
        dist.barrier()
        dist.destroy_process_group()

    if IS_MAIN:
        print(f"\n{'='*60}")
        print(f"PASS  —  train→eval→train cycle completed without error.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
