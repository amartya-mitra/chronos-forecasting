#!/usr/bin/env python3
"""
Fine-tune chronos-t5-small on decomposition targets (128-step or 192-step).

Variants (--variant):
  ST      — [seasonal(64)  | SEP_1 | trend(64)]                          → 129 generated tokens
  TS      — [trend(64)     | SEP_1 | seasonal(64)]                       → 129 generated tokens
  STF     — [seasonal(64)  | SEP_1 | trend(64) | SEP_2 | forecast(64)]   → 194 generated tokens
  TSF     — [trend(64)     | SEP_1 | seasonal(64) | SEP_2 | forecast(64)] → 194 generated tokens
  TSF_M4M — [trend(18)     | SEP_1 | seasonal(18) | SEP_2 | forecast(18)] →  56 generated tokens
            (M4 Monthly: 36-month context, 18-month horizon, period=12)

Separator tokens
----------------
SEP_1 (token 4103) is inserted between component 0 and component 1.
SEP_2 (token 4104) is inserted between component 1 and component 2 (3-part variants only).
At inference SEP tokens are force-decoded at their fixed positions so the
model always sees the same trigger it was trained on.

Architecture (decoder-side conditioning, v5-style):
  encoder           = [ctx_tokens | EOS]
  decoder_input_ids = [MODE | part_0 | SEP_1 | part_1 [| SEP_2 | part_2]]
  labels            = [part_0 | SEP_1 | part_1 [| SEP_2 | part_2] | EOS]

Vocab is extended to 4105:
  4096 fast | 4097 trend | 4098 seasonal | 4099 ST | 4100 TS
  4101 STF  | 4102 TSF   | 4103 SEP_1    | 4104 SEP_2

ST/TS datasets (sarsim0-st.arrow, sarsim0-ts.arrow) do NOT need to be
rebuilt — SEP tokens are inserted on-the-fly during pretokenisation.
STF/TSF datasets (sarsim0-stf.arrow, sarsim0-tsf.arrow) carry a third
scale column s_third for the forecast component.

Usage:
  python fine_tune_st_ts.py --variant ST  --job-id $SLURM_JOB_ID
  python fine_tune_st_ts.py --variant TSF --job-id $SLURM_JOB_ID
"""

import os
import sys
import argparse
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from chronos import ChronosPipeline

# ── variant registry ──────────────────────────────────────────────────────────
# pred_len : number of float-valued tokens in the output (excl. SEPs and EOS)
# gen_len  : total tokens generated at inference (pred_len + n_SEPs)
# forced   : (step, token_id) pairs for force-decoding SEP tokens (1-indexed)
_VARIANT_CFG = {
    "ST": {
        "token_id":    4099,
        "data_path":   "sarsim0-st.arrow",
        "n_parts":     2,
        "segment_len": 64,
        "ctx_len":     512,
        "pred_len":    128,
        "gen_len":     129,   # 128 + 1 SEP
        "forced":      [(65, 4103)],
        "part_names":  ["Seasonal", "Trend"],
        "part_cols":   ["mediumpurple", "seagreen"],
    },
    "TS": {
        "token_id":    4100,
        "data_path":   "sarsim0-ts.arrow",
        "n_parts":     2,
        "segment_len": 64,
        "ctx_len":     512,
        "pred_len":    128,
        "gen_len":     129,
        "forced":      [(65, 4103)],
        "part_names":  ["Trend", "Seasonal"],
        "part_cols":   ["seagreen", "mediumpurple"],
    },
    "STF": {
        "token_id":    4101,
        "data_path":   "sarsim0-stf.arrow",
        "n_parts":     3,
        "segment_len": 64,
        "ctx_len":     512,
        "pred_len":    192,
        "gen_len":     194,   # 192 + 2 SEPs
        "forced":      [(65, 4103), (130, 4104)],
        "part_names":  ["Seasonal", "Trend", "Forecast"],
        "part_cols":   ["mediumpurple", "seagreen", "darkorange"],
    },
    "TSF": {
        "token_id":    4102,
        "data_path":   "sarsim0-tsf.arrow",
        "n_parts":     3,
        "segment_len": 64,
        "ctx_len":     512,
        "pred_len":    192,
        "gen_len":     194,
        "forced":      [(65, 4103), (130, 4104)],
        "part_names":  ["Trend", "Seasonal", "Forecast"],
        "part_cols":   ["seagreen", "mediumpurple", "darkorange"],
    },
    # M4 Monthly variant: 36-month context, 18-month horizon per segment
    "TSF_M4M": {
        "token_id":    4102,          # reuse TSF mode token
        "data_path":   "m4monthly-tsf.arrow",
        "n_parts":     3,
        "segment_len": 18,            # M4 Monthly forecast horizon
        "ctx_len":     36,            # 3 years of monthly context
        "pred_len":    54,            # 3 × 18
        "gen_len":     56,            # 54 + 2 SEPs
        "forced":      [(19, 4103), (38, 4104)],
        "part_names":  ["Trend", "Seasonal", "Forecast"],
        "part_cols":   ["seagreen", "mediumpurple", "darkorange"],
    },
    # M4 Hourly variant: 96-hour context, 48-hour horizon per segment
    "TSF_M4H": {
        "token_id":    4102,          # reuse TSF mode token
        "data_path":   "m4hourly-tsf.arrow",
        "n_parts":     3,
        "segment_len": 48,            # M4 Hourly forecast horizon
        "ctx_len":     96,            # 4 days of hourly context
        "pred_len":    144,           # 3 × 48
        "gen_len":     146,           # 144 + 2 SEPs
        "forced":      [(49, 4103), (98, 4104)],
        "part_names":  ["Trend", "Seasonal", "Forecast"],
        "part_cols":   ["seagreen", "mediumpurple", "darkorange"],
    },
}

# ── special token IDs ─────────────────────────────────────────────────────────
SEP_1_TOKEN_ID = 4103
SEP_2_TOKEN_ID = 4104
EXTENDED_VOCAB = 4105   # 4096 original + 9 new tokens (4096..4104)

# ── hyperparameters ───────────────────────────────────────────────────────────
CONTEXT_LENGTH = 512
BATCH_SIZE     = 128
N_STEPS        = 10_000
WARMUP_STEPS   = 500
LR             = 1e-4
GRAD_CLIP      = 1.0
EVAL_EVERY     = 500
EVAL_BATCH     = 32
EVAL_SAMPLES   = 20
EVAL_N_SHOW    = 5
EVAL_POOL      = 100

EARLY_STOP_PATIENCE  = 5
EARLY_STOP_MIN_DELTA = 1e-4

EMBED_INIT_STD_MULTIPLIER = 2.0
SEED     = 42
MODEL_ID = "amazon/chronos-t5-small"

_CTX_COL  = "steelblue"
_PRED_COL = "crimson"


class SEPForcer(LogitsProcessor):
    """
    Forces SEP tokens at fixed positions during autoregressive generation.

    `forced` is a list of (1-indexed-position, token_id) pairs matching the
    _VARIANT_CFG['forced'] format.  Internally converts to 0-indexed from the
    first generated token (i.e. position in input_ids after decoder_start).

    forced_decoder_ids inside GenerationConfig is silently ignored in
    transformers >=4.45, so this LogitsProcessor is the reliable replacement.
    """
    def __init__(self, forced: list):
        # Convert 1-indexed generated-token position → 0-indexed
        # input_ids.shape[-1] - 1 == 0-indexed position of token being decided
        self._forced = {pos - 1: tok for pos, tok in forced}

    def __call__(self, input_ids, scores):
        cur = input_ids.shape[-1] - 1   # 0-indexed from first generated token
        if cur in self._forced:
            scores[:] = -float("inf")
            scores[:, self._forced[cur]] = 0.0
        return scores


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def init_mode_embeddings(inner) -> None:
    """Initialise all new tokens (4096-4104) as EOS + noise."""
    w = inner.get_input_embeddings().weight
    existing_std = w[:4096].std().item()
    noise_std    = EMBED_INIT_STD_MULTIPLIER * existing_std
    with torch.no_grad():
        base = w[1].clone()   # EOS token (ID=1)
        gen  = torch.Generator(); gen.manual_seed(SEED)
        for tok_id in range(4096, EXTENDED_VOCAB):
            w[tok_id] = base + noise_std * torch.randn(w.shape[1], generator=gen, dtype=w.dtype)
    print(f"  Embed init : eos_plus_noise,  noise_std={noise_std:.5f}")
    for name, tid in [("ST", 4099), ("TS", 4100), ("STF", 4101), ("TSF", 4102),
                      ("SEP_1", 4103), ("SEP_2", 4104)]:
        cos = F.cosine_similarity(w[tid].float().unsqueeze(0), w[1].float().unsqueeze(0)).item()
        print(f"  Cosine sim : {name}↔EOS={cos:.3f}")


def write_config(path: Path, job_id: str, variant: str, cfg: dict) -> None:
    n_parts      = cfg["n_parts"]
    seg          = cfg.get("segment_len", 64)
    ctx_len      = cfg.get("ctx_len", CONTEXT_LENGTH)
    parts_desc   = " | ".join(f"{cfg['part_names'][i]}({seg})" for i in range(n_parts))
    sep_desc     = f"SEP_1={SEP_1_TOKEN_ID}" + (f"  SEP_2={SEP_2_TOKEN_ID}" if n_parts == 3 else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "Experiment Configuration", "=" * 40,
        f"Job ID    : {job_id}",
        f"Script    : fine_tune_st_ts.py  --variant {variant}",
        f"Model     : {MODEL_ID}",
        f"Dataset   : {cfg['data_path']}",
        f"Token ID  : {variant}_TOKEN_ID={cfg['token_id']}",
        f"SEP tokens: {sep_desc}",
        f"Vocab     : extended to {EXTENDED_VOCAB}",
        f"Steps     : {N_STEPS:,}  |  batch={BATCH_SIZE}  |  LR={LR}",
        f"Ctx len   : {ctx_len}  |  Seg len: {seg}",
        f"Pred len  : {cfg['pred_len']}  ({parts_desc})",
        f"Gen len   : {cfg['gen_len']}  (pred + {n_parts - 1} SEP token(s))",
        f"Early stop: patience={EARLY_STOP_PATIENCE} evals  |  min_delta={EARLY_STOP_MIN_DELTA}",
        "",
        "Architecture: decoder-side mode conditioning with SEP boundary tokens",
        f"  encoder           = [ctx_tokens | EOS]               ({ctx_len + 1} tokens)",
        f"  decoder_input_ids = [{variant}_TOKEN | {parts_desc}]  ({cfg['gen_len']} tokens, SEPs inserted)",
        f"  labels            = [{parts_desc} | EOS]              ({cfg['gen_len']} tokens, SEPs inserted)",
        f"Embed init: eos_plus_noise  (multiplier={EMBED_INIT_STD_MULTIPLIER})",
    ]) + "\n")
    print(f"  Config written: {path.name}")


# ── data ──────────────────────────────────────────────────────────────────────

def load_dataset(data_path: Path, n_parts: int):
    with pa.memory_map(str(data_path), "r") as src:
        table = ipc.open_file(src).read_all()

    scale_cols = ["s_first", "s_second"] + (["s_third"] if n_parts == 3 else [])
    scale_data = {c: table.column(c).to_pylist() for c in scale_cols}

    rows = []
    for i, (item_id, ctx, tgt) in enumerate(zip(
        table.column("item_id").to_pylist(),
        table.column("context").to_pylist(),
        table.column("target").to_pylist(),
    )):
        rows.append({
            "item_id": item_id,
            "context": np.array(ctx, dtype=np.float32),
            "target":  np.array(tgt, dtype=np.float32),   # (pred_len,) context-scale-normalised
            "s_parts": [float(scale_data[c][i]) for c in scale_cols],
        })
    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[EVAL_POOL:], rows[:EVAL_POOL]


def pretokenise(rows, tokenizer, mode_token_id: int, n_parts: int, segment_len: int = 64):
    """
    Pre-tokenise all rows, inserting SEP boundary tokens between components.

    For 2-part variants (ST/TS, segment_len=64):
      labels        = [part_0(seg) | SEP_1 | part_1(seg) | EOS]
      dec_input_ids = [MODE | part_0(seg) | SEP_1 | part_1(seg)]

    For 3-part variants (STF/TSF, segment_len=64):
      labels        = [part_0(seg) | SEP_1 | part_1(seg) | SEP_2 | part_2(seg) | EOS]
      dec_input_ids = [MODE | part_0 | SEP_1 | part_1 | SEP_2 | part_2]

    For TSF_M4M (segment_len=18):
      Same structure with seg=18 → labels length = 18+1+18+1+18+1 = 57 tokens.

    label_input_transform asserts len==64; bypassed by calling _input_transform directly.
    """
    N        = len(rows)
    pred_len = segment_len * n_parts
    print(f"  Pre-tokenising {N:,} rows  (n_parts={n_parts}, segment_len={segment_len}, pred_len={pred_len})...")

    contexts = torch.tensor(np.stack([d["context"] for d in rows]), dtype=torch.float32)
    targets  = torch.tensor(np.stack([d["target"]  for d in rows]), dtype=torch.float32)  # (N, pred_len)

    # Encoder
    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)

    # Tokenise full target (bypasses the assert-len==64 in label_input_transform)
    label_ids, label_mask, _ = tokenizer._input_transform(context=targets, scale=scale)  # (N, pred_len)

    # Split into segment_len-step parts
    parts      = [label_ids[:,  i*segment_len:(i+1)*segment_len] for i in range(n_parts)]
    parts_mask = [label_mask[:, i*segment_len:(i+1)*segment_len] for i in range(n_parts)]

    # Build sequence with SEP tokens interleaved
    sep1     = torch.full((N, 1), SEP_1_TOKEN_ID, dtype=label_ids.dtype)
    ones     = torch.ones((N, 1), dtype=label_mask.dtype)

    seq_parts = [parts[0], sep1, parts[1]]
    msk_parts = [parts_mask[0], ones, parts_mask[1]]
    if n_parts == 3:
        sep2 = torch.full((N, 1), SEP_2_TOKEN_ID, dtype=label_ids.dtype)
        seq_parts += [sep2, parts[2]]
        msk_parts += [ones, parts_mask[2]]

    # Append EOS
    if tokenizer.config.use_eos_token:
        eos_col = torch.full((N, 1), tokenizer.config.eos_token_id, dtype=label_ids.dtype)
        seq_parts.append(eos_col)
        msk_parts.append(ones)

    labels_seq = torch.cat(seq_parts, dim=1)   # (N, 130) or (N, 195)
    mask_seq   = torch.cat(msk_parts, dim=1)
    labels     = labels_seq.masked_fill(~mask_seq.bool(), -100)

    # Decoder input: [MODE | labels_seq[:-1]]  (right-shift by 1, replace pos-0 with mode)
    mode_col      = torch.full((N, 1), mode_token_id, dtype=token_ids.dtype)
    dec_input_ids = torch.cat([mode_col, labels_seq[:, :-1]], dim=1)  # (N, 130) or (N, 195)

    out = []
    for i, d in enumerate(rows):
        out.append({
            "item_id":       d["item_id"],
            "context":       d["context"],
            "target":        d["target"],
            "s_parts":       d["s_parts"],
            "input_ids":     token_ids[i],
            "attn_mask":     attn_mask[i],
            "dec_input_ids": dec_input_ids[i],
            "labels":        labels[i],
        })
    print("  Pre-tokenisation complete.")
    return out


class CyclicLoader:
    def __init__(self, rows, batch_size, seed=0):
        self.rows, self.batch_size = rows, batch_size
        self.rng = np.random.default_rng(seed)
        self._idx = np.arange(len(rows)); self._pos = len(rows)

    def next_batch(self):
        if self._pos + self.batch_size > len(self._idx):
            self.rng.shuffle(self._idx); self._pos = 0
        b = [self.rows[i] for i in self._idx[self._pos:self._pos+self.batch_size]]
        self._pos += self.batch_size
        return b


def assemble_batch(batch, device):
    return (
        torch.stack([d["input_ids"]     for d in batch]).to(device),
        torch.stack([d["attn_mask"]     for d in batch]).to(device),
        torch.stack([d["dec_input_ids"] for d in batch]).to(device),
        torch.stack([d["labels"]        for d in batch]).to(device),
    )


# ── eval & plots ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(pipeline, contexts, mode_token_id: int, device, cfg: dict):
    """
    Generate and return predictions as numpy array of shape (B, pred_len).

    SEP tokens are force-decoded at their fixed positions and stripped from
    the output before calling output_transform (which expects only bin tokens).
    """
    n_parts  = cfg["n_parts"]
    pred_len = cfg["pred_len"]
    gen_len  = cfg["gen_len"]

    ctx = torch.tensor(np.stack(contexts), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)
    model_cfg = pipeline.model.config

    # forced_decoder_ids inside GenerationConfig is silently ignored in
    # transformers >=4.45; use SEPForcer LogitsProcessor instead.
    lp = LogitsProcessorList([SEPForcer(cfg["forced"])]) if cfg["forced"] else LogitsProcessorList()

    raw = pipeline.model.model.generate(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        logits_processor=lp,
        generation_config=GenerationConfig(
            min_new_tokens=gen_len,
            max_new_tokens=gen_len,
            do_sample=True,
            num_return_sequences=EVAL_SAMPLES,
            decoder_start_token_id=mode_token_id,
            eos_token_id=model_cfg.eos_token_id,
            pad_token_id=model_cfg.pad_token_id,
            temperature=model_cfg.temperature,
            top_k=model_cfg.top_k,
            top_p=model_cfg.top_p,
        ),
    )
    B = len(contexts)
    # Drop the mode start token; raw[:, 1:] has shape (B*EVAL_SAMPLES, gen_len)
    raw = raw[:, 1:]

    # Strip SEP tokens, keep only the bin-token segments
    seg = cfg.get("segment_len", 64)
    if n_parts == 2:
        # layout: [part_0(seg) | SEP_1(1) | part_1(seg)]
        raw_bins = torch.cat([raw[:, :seg], raw[:, seg+1:]], dim=1)
    else:
        # layout: [part_0(seg) | SEP_1(1) | part_1(seg) | SEP_2(1) | part_2(seg)]
        raw_bins = torch.cat([raw[:, :seg], raw[:, seg+1:2*seg+1], raw[:, 2*seg+2:]], dim=1)

    raw_bins = raw_bins.reshape(B, EVAL_SAMPLES, pred_len)
    return pipeline.tokenizer.output_transform(raw_bins.cpu(), scale).median(dim=1).values.numpy()


def evaluate(pipeline, eval_rows, mode_token_id, device, step, cfg):
    pipeline.model.model.eval()
    maes = []
    bar  = tqdm(range(0, len(eval_rows), EVAL_BATCH),
                desc=f"    eval step={step:,}", file=sys.stdout, leave=False, dynamic_ncols=True)
    for i in bar:
        batch = eval_rows[i:i+EVAL_BATCH]
        preds = predict_batch(pipeline, [r["context"] for r in batch], mode_token_id, device, cfg)
        for r, pred in zip(batch, preds):
            maes.append(float(np.mean(np.abs(pred - r["target"]))))
    pipeline.model.model.train()
    return float(np.mean(maes))


def plot_loss(loss_steps, losses, eval_steps, maes, path, job_id, variant):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(loss_steps, losses, color="steelblue", lw=0.8, alpha=0.6)
    if len(losses) >= 50:
        sm = np.convolve(losses, np.ones(50)/50, mode="valid")
        axes[0].plot(loss_steps[49:], sm, color="steelblue", lw=1.8, label="Loss (MA-50)")
    axes[0].set_title("Training Loss (CE)", fontweight="bold")
    axes[0].set_xlabel("Step"); axes[0].legend(fontsize=8)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[1].plot(eval_steps, maes, color="darkorange", lw=1.8, marker="o", ms=3,
                 label=f"{variant} MAE (normalised)")
    axes[1].set_title(f"Eval MAE — {variant} Mode", fontweight="bold")
    axes[1].set_xlabel("Step"); axes[1].legend(fontsize=8)
    axes[1].spines[["top","right"]].set_visible(False)
    fig.suptitle(f"Fine-tuning {variant}: chronos-t5-small  ·  job {job_id}", fontweight="bold")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_eval_samples(pipeline, eval_rows, mode_token_id, device, path, job_id, variant,
                      cfg: dict, n=EVAL_N_SHOW):
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(eval_rows), size=min(n, len(eval_rows)), replace=False))
    sel   = [eval_rows[i] for i in picks]
    pipeline.model.model.eval()
    preds = predict_batch(pipeline, [r["context"] for r in sel], mode_token_id, device, cfg)
    pipeline.model.model.train()

    n_parts = cfg["n_parts"]
    seg     = cfg.get("segment_len", 64)
    ctx_len = cfg.get("ctx_len", CONTEXT_LENGTH)
    ctx_x   = np.arange(ctx_len)
    parts_x = [np.arange(ctx_len + i*seg, ctx_len + (i+1)*seg)
               for i in range(n_parts)]

    legend_str = "  ".join(
        f"{cfg['part_cols'][i]} = {cfg['part_names'][i]}" for i in range(n_parts)
    )
    fig, axes_raw = plt.subplots(n, 1, figsize=(16, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"fine_tune_{variant.lower()}  ·  job {job_id}\n"
        f"{legend_str}   crimson = model prediction",
        fontsize=11, fontweight="bold",
    )
    for idx, (row, pred) in enumerate(zip(sel, preds)):
        ax    = axes[idx]
        s_ctx = float(np.mean(np.abs(row["context"])))
        # Inverse transform: stored target and model output are both in context-scale
        # units; multiply by (s_component / s_ctx) to recover original values.
        inv = [s / s_ctx for s in row["s_parts"]]

        ax.fill_between(ctx_x, row["context"], alpha=0.07, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL, lw=0.7, label="Input context")

        for i in range(n_parts):
            tgt_i  = row["target"][i*seg:(i+1)*seg] * inv[i]
            pred_i = pred[i*seg:(i+1)*seg]          * inv[i]
            ax.plot(parts_x[i], tgt_i,  color=cfg["part_cols"][i], lw=1.8,
                    label=f"Target {cfg['part_names'][i]}")
            lbl = "Prediction" if i == n_parts - 1 else None
            ax.plot(parts_x[i], pred_i, color=_PRED_COL, lw=1.4, ls="--",
                    alpha=0.85, label=lbl)

        # Context / segment boundary lines
        ax.axvline(ctx_len - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        for i in range(1, n_parts):
            ax.axvline(ctx_len + i*seg - 0.5, color="grey", lw=0.6, ls=":", alpha=0.5)

        mae_norm = float(np.mean(np.abs(pred - row["target"])))
        ax.set_title(f"{row['item_id']}  MAE(norm)={mae_norm:.4f}", fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7); ax.spines[["top","right"]].set_visible(False)

    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.7)
    axes[-1].set_xlabel("Time step", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(_VARIANT_CFG.keys()), required=True)
    parser.add_argument("--job-id",  default="nojobid")
    args    = parser.parse_args()
    variant = args.variant
    job_id  = args.job_id
    cfg     = _VARIANT_CFG[variant]

    MODE_TOKEN_ID = cfg["token_id"]
    DATA_PATH     = FINETUNING_ROOT / "data" / cfg["data_path"]

    FIGURES_DIR = FINETUNING_ROOT / "figures"     / f"finetune_{variant.lower()}_{job_id}"
    CKPT_DIR    = FINETUNING_ROOT / "checkpoints" / f"finetune-{variant.lower()}-{job_id}"

    # ── Multi-GPU setup (DDP via torchrun, or single-GPU fallback) ──────────────
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
    IS_DDP     = LOCAL_RANK != -1
    if IS_DDP:
        dist.init_process_group(backend="nccl")
        WORLD_SIZE = dist.get_world_size()
        IS_MAIN    = (LOCAL_RANK == 0)
        device     = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(LOCAL_RANK)
    else:
        WORLD_SIZE = 1
        IS_MAIN    = True
        device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(SEED + (LOCAL_RANK if IS_DDP else 0))

    if IS_MAIN:
        print("=" * 60)
        print(f"Fine-tuning {variant}: chronos-t5-small")
        print(f"  Job ID    : {job_id}")
        print(f"  Device    : {device}  |  DDP={IS_DDP}  world={WORLD_SIZE}")
        print(f"  Variant   : {variant}  →  {' | '.join(cfg['part_names'])}")
        print(f"  Token ID  : {MODE_TOKEN_ID}  |  SEP_1={SEP_1_TOKEN_ID}"
              + (f"  SEP_2={SEP_2_TOKEN_ID}" if cfg["n_parts"] == 3 else ""))
        print(f"  Pred len  : {cfg['pred_len']}  |  Gen len: {cfg['gen_len']}")
        print(f"  Steps     : {N_STEPS:,}  |  batch={BATCH_SIZE} (per-rank: {BATCH_SIZE // WORLD_SIZE})  |  LR={LR}")
        print(f"  Figures   : {FIGURES_DIR}")
        print("=" * 60)
        write_config(FIGURES_DIR / "config.txt", job_id, variant, cfg)

    print(f"\n[rank {LOCAL_RANK if IS_DDP else 0}] Loading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    inner    = pipeline.model.model
    if inner.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
    if IS_MAIN:
        print(f"  Vocab extended → {EXTENDED_VOCAB}")
        init_mode_embeddings(inner)
    else:
        init_mode_embeddings(inner)

    if IS_MAIN:
        print("\nLoading dataset...")
    train_rows, eval_rows = load_dataset(DATA_PATH, cfg["n_parts"])
    if IS_MAIN:
        print(f"  Train : {len(train_rows):,}  |  Eval : {len(eval_rows):,}")
    tokenised = pretokenise(train_rows, pipeline.tokenizer, MODE_TOKEN_ID,
                            cfg["n_parts"], cfg.get("segment_len", 64))

    # Each DDP rank processes its own shard; effective batch = BATCH_SIZE across all ranks
    per_rank_batch = max(1, BATCH_SIZE // WORLD_SIZE)
    if IS_DDP:
        shard = [tokenised[i] for i in range(len(tokenised)) if i % WORLD_SIZE == LOCAL_RANK]
    else:
        shard = tokenised
    loader = CyclicLoader(shard, per_rank_batch, seed=SEED + (LOCAL_RANK if IS_DDP else 0))

    print(f"\n[rank {LOCAL_RANK if IS_DDP else 0}] Moving model to {device}...")
    inner = inner.to(device); inner.train()

    if IS_DDP:
        dp_model = DDP(inner, device_ids=[LOCAL_RANK])
        if IS_MAIN:
            print(f"  DDP across {WORLD_SIZE} GPUs")
    else:
        dp_model = inner
        if IS_MAIN:
            print(f"  Single GPU / CPU mode")

    optimizer = torch.optim.AdamW(inner.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, N_STEPS)

    loss_steps, losses = [], []
    eval_steps, maes   = [], []
    running = 0.0; LOG_EVERY = 50

    best_mae       = float("inf")
    no_improve_cnt = 0
    stopped_early  = False

    if IS_MAIN:
        print("\nTraining...")
    pbar = tqdm(range(1, N_STEPS+1), total=N_STEPS, desc=f"Training ({variant})",
                file=sys.stdout, dynamic_ncols=True, smoothing=0.05,
                miniters=LOG_EVERY, mininterval=0.0,
                disable=not IS_MAIN)

    for step in pbar:
        ids, mask, dec, lbl = assemble_batch(loader.next_batch(), device)
        out  = dp_model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec, labels=lbl)
        loss = out.loss   # DDP all-reduces gradients automatically; no .mean() needed
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), GRAD_CLIP)
        optimizer.step(); scheduler.step()
        running += loss.item()

        if step % LOG_EVERY == 0 and IS_MAIN:
            avg    = running / LOG_EVERY; running = 0.0
            lr_now = scheduler.get_last_lr()[0]
            loss_steps.append(step); losses.append(avg)
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")
            tqdm.write(f"  step {step:>6,}/{N_STEPS:,}  loss={avg:.4f}  lr={lr_now:.2e}")

        if step % EVAL_EVERY == 0:
            # All ranks sync before eval; only rank 0 runs generate()
            if IS_DDP:
                dist.barrier()
            if IS_MAIN:
                pipeline.model.model = inner
                mae = evaluate(pipeline, eval_rows, MODE_TOKEN_ID, device, step, cfg)
                eval_steps.append(step); maes.append(mae)
                tqdm.write(f"  [eval] step={step:,}  {variant} MAE={mae:.4f}")

                if mae < best_mae - EARLY_STOP_MIN_DELTA:
                    best_mae       = mae
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
                    tqdm.write(
                        f"  [early stop] No improvement  {no_improve_cnt}/{EARLY_STOP_PATIENCE}"
                        f"  (best={best_mae:.4f})"
                    )
                    if no_improve_cnt >= EARLY_STOP_PATIENCE:
                        tqdm.write(
                            f"  [early stop] Triggered at step {step:,}."
                            f"  Best MAE={best_mae:.4f}"
                        )
                        stopped_early = True

            # Broadcast early-stop decision from rank 0 to all ranks
            if IS_DDP:
                stop_tensor = torch.tensor([1 if stopped_early else 0],
                                           dtype=torch.long, device=device)
                dist.broadcast(stop_tensor, src=0)
                if not IS_MAIN:
                    stopped_early = bool(stop_tensor.item())
                dist.barrier()  # All ranks sync after eval

            inner.train()
            if stopped_early:
                break

    pbar.close()
    if IS_MAIN and stopped_early:
        print(f"\nTraining stopped early at step {step:,}  "
              f"(patience={EARLY_STOP_PATIENCE}, best MAE={best_mae:.4f})")

    if IS_MAIN:
        print(f"\nSaving checkpoint to {CKPT_DIR} ...")
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        inner.save_pretrained(str(CKPT_DIR))
        for name, arr in [("loss_steps", loss_steps), ("losses", losses),
                          ("eval_steps", eval_steps), ("maes", maes)]:
            np.save(str(CKPT_DIR / f"{name}.npy"), np.array(arr))

        print("\nGenerating plots...")
        inner.eval(); pipeline.model.model = inner
        plot_loss(loss_steps, losses, eval_steps, maes,
                  path=FIGURES_DIR / "loss.png", job_id=job_id, variant=variant)
        plot_eval_samples(pipeline, eval_rows, MODE_TOKEN_ID, device,
                          path=FIGURES_DIR / "eval_samples.png",
                          job_id=job_id, variant=variant, cfg=cfg)

        print(f"\nDone.  Figures in {FIGURES_DIR}/")
        if maes:
            print(f"Final {variant} MAE : {maes[-1]:.4f}")

    if IS_DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
