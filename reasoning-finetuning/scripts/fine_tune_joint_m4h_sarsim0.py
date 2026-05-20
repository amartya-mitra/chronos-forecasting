#!/usr/bin/env python3
"""
Joint TSF + FAST fine-tuning across M4 Hourly and SarSim0.

Design principles
-----------------
1. Dataset-specific batches: each training step draws a full batch from exactly
   one dataset (M4H or SarSim0).  M4H and SarSim0 data are never mixed within
   the same batch, so the decoder always sees a single, consistent output length.

2. Metadata conditioning (Mechanism 1): a MetadataEmbedder maps
   (H / H_MAX, L_ctx / L_MAX) → d_model-dimensional token prepended to the T5
   encoder input.  This tells the model which dataset it is processing without
   any structural changes to T5 or the vocabulary.

3. Dataset-specific configurations:
     M4H     — H=48, L_ctx=96,  SEPs @ positions (49, 98),  pred=144 tokens
     SarSim0 — H=64, L_ctx=512, SEPs @ positions (65, 130), pred=192 tokens

4. Normalized early-stop metric: each dataset's TSF MAE is divided by its
   first-eval value so both contribute equally regardless of absolute scale.
   Early stop monitors the average of these two dimensionless MAEs.

5. Immediate best-checkpoint save: the best checkpoint is written to disk as
   soon as norm_avg_tsf improves, so it is never lost even if training crashes.

Training schedule (per step):
  Dataset : SarSim0 with probability  --dataset-ratio  (default 0.50)
  Mode    : FAST    with probability  --fast-ratio     (default 0.20)

Checkpoint:
  checkpoints/finetune-joint-m4h-sarsim0-{job_id}/
    config.json + model.safetensors  (T5 backbone, via save_pretrained)
    meta_embedder.pt                 (MetadataEmbedder state_dict)

Figures:
  figures/finetunes/finetune_joint_m4h_sarsim0_{job_id}/
    config.txt
    loss.png
    eval_tsf_samples_m4h.png
    eval_tsf_samples_sarsim0.png

Usage
-----
  python fine_tune_joint_m4h_sarsim0.py --job-id $SLURM_JOB_ID
  python fine_tune_joint_m4h_sarsim0.py --job-id $SLURM_JOB_ID \\
      --fast-ratio 0.20 --dataset-ratio 0.70
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
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    GenerationConfig,
    AutoModelForSeq2SeqLM,
)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from chronos import ChronosPipeline

# ── paths ─────────────────────────────────────────────────────────────────────
M4H_DATA_PATH        = FINETUNING_ROOT / "data"    / "m4hourly-tsf.arrow"
SARSIM0_DATA_PATH    = FINETUNING_ROOT / "data"    / "sarsim0-tsf.arrow"
M4H_BASELINE_REF     = FINETUNING_ROOT / "figures" / "evals" / "eval_pretrained_m4hourly" / "baseline_mae.txt"
SARSIM0_BASELINE_REF = FINETUNING_ROOT / "figures" / "evals" / "eval_pretrained_sarsim0"  / "baseline_mae.txt"
MODEL_ID             = "amazon/chronos-t5-small"

# ── metadata conditioning constants ──────────────────────────────────────────
# H_MAX and L_MAX define the normalisation range for (horizon, context_length).
# Keep these fixed across all experiments so the metadata embedding space is
# consistent.
H_MAX   = 256    # max forecast horizon across current + anticipated datasets
L_MAX   = 512    # max context length (Chronos native clip limit)
D_MODEL = 512    # T5-small hidden dimension
D_SIN   = 64     # sinusoidal encoding dimension per scalar  (2×D_SIN → D_MODEL)

# ── dataset configs ───────────────────────────────────────────────────────────
#   H      : forecast horizon (segment length in TSF; output length in FAST)
#   L_ctx  : native context length of the dataset
#   tsf    : TSF-mode specific parameters
#   fast   : FAST-mode specific parameters
M4H_CFG = {
    "name":  "M4Hourly",
    "H":     48,
    "L_ctx": 96,
    "tsf": {
        "token_id":   4102,
        "pred_len":   144,                       # 3 × 48
        "gen_len":    146,                       # 144 + 2 SEPs
        "forced":     [(49, 4103), (98, 4104)],  # SEP_1 @ pos 49, SEP_2 @ pos 98
        "part_cols":  ["seagreen", "mediumpurple", "darkorange"],
        "part_names": ["Trend", "Seasonal", "Forecast"],
    },
    "fast": {
        "token_id": 4096,
        "gen_len":  48,
    },
}

SARSIM0_CFG = {
    "name":  "SarSim0",
    "H":     64,
    "L_ctx": 512,
    "tsf": {
        "token_id":   4102,
        "pred_len":   192,                         # 3 × 64
        "gen_len":    194,                         # 192 + 2 SEPs
        "forced":     [(65, 4103), (130, 4104)],   # SEP_1 @ pos 65, SEP_2 @ pos 130
        "part_cols":  ["seagreen", "mediumpurple", "darkorange"],
        "part_names": ["Trend", "Seasonal", "Forecast"],
    },
    "fast": {
        "token_id": 4096,
        "gen_len":  64,
    },
}

# ── special tokens ────────────────────────────────────────────────────────────
SEP_1_TOKEN_ID = 4103
SEP_2_TOKEN_ID = 4104
EXTENDED_VOCAB = 4105

# ── hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 128
N_STEPS      = 10_000
WARMUP_STEPS = 500
LR           = 1e-4     # backbone (T5) learning rate
LR_META      = 1e-3     # MetadataEmbedder learning rate (10× — new parameters)
GRAD_CLIP    = 1.0
EVAL_EVERY   = 500
EVAL_BATCH   = 32
EVAL_SAMPLES = 20
EVAL_N_SHOW  = 5
EVAL_POOL    = 100

EARLY_STOP_PATIENCE  = 5
EARLY_STOP_MIN_DELTA = 1e-4

EMBED_INIT_STD_MULTIPLIER = 2.0
SCALE_EPS = 1e-4
SEED      = 42

_CTX_COL  = "steelblue"
_PRED_COL = "crimson"


# ── Mechanism 1: sinusoidal scalar metadata conditioning ──────────────────────

def normalise_metadata(H: int, L_ctx: int) -> torch.Tensor:
    """Return [2] float32 tensor: (H / H_MAX, L_ctx / L_MAX)."""
    return torch.tensor([H / H_MAX, L_ctx / L_MAX], dtype=torch.float32)


class MetadataEmbedder(nn.Module):
    """
    Encodes a (H_norm, L_ctx_norm) pair into a single d_model embedding token
    for prepending to the T5 encoder.

    Architecture:
        sinusoidal_encode(H_norm)      → R^{D_SIN}
        sinusoidal_encode(L_ctx_norm)  → R^{D_SIN}
        concat                         → R^{2 * D_SIN}
        Linear(2*D_SIN, D_MODEL) + LayerNorm
    """

    def __init__(self, d_model: int = D_MODEL, d_sin: int = D_SIN):
        super().__init__()
        self.d_sin = d_sin
        self.proj  = nn.Linear(2 * d_sin, d_model)
        self.norm  = nn.LayerNorm(d_model)
        freqs = 1.0 / (10000 ** (
            torch.arange(0, d_sin // 2, dtype=torch.float32) / (d_sin // 2)
        ))
        self.register_buffer("freqs", freqs)   # [d_sin // 2]

    def _sin_encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B] → [B, d_sin]."""
        angles = x.unsqueeze(-1) * self.freqs.unsqueeze(0)   # [B, d_sin//2]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metadata: [B, 2] — (H_norm, L_ctx_norm) per sample
        Returns:
            [B, 1, d_model] — ready to prepend to encoder inputs_embeds
        """
        h_enc = self._sin_encode(metadata[:, 0])   # [B, d_sin]
        l_enc = self._sin_encode(metadata[:, 1])   # [B, d_sin]
        emb   = self.norm(self.proj(torch.cat([h_enc, l_enc], dim=-1)))  # [B, d_model]
        return emb.unsqueeze(1)                    # [B, 1, d_model]


class ChronosWithMetadata(nn.Module):
    """
    Wraps T5ForConditionalGeneration to prepend a MetadataEmbedder token at
    every forward / generate call.  The T5 architecture is unchanged; the
    metadata token is at encoder position 0 and visible to the decoder via
    cross-attention.
    """

    def __init__(self, inner, d_model: int = D_MODEL, d_sin: int = D_SIN):
        super().__init__()
        self.inner         = inner
        self.meta_embedder = MetadataEmbedder(d_model=d_model, d_sin=d_sin)

    def _prepend_meta(self, input_ids, attention_mask, metadata):
        meta_emb      = self.meta_embedder(metadata)                       # [B, 1, d]
        ctx_emb       = self.inner.get_input_embeddings()(input_ids)       # [B, L, d]
        inputs_embeds = torch.cat([meta_emb, ctx_emb], dim=1)             # [B, L+1, d]
        meta_mask     = torch.ones(attention_mask.size(0), 1,
                                   dtype=attention_mask.dtype,
                                   device=attention_mask.device)
        attention_mask_ext = torch.cat([meta_mask, attention_mask], dim=1) # [B, L+1]
        return inputs_embeds, attention_mask_ext

    def forward(self, input_ids, attention_mask, decoder_input_ids,
                labels=None, metadata=None):
        inputs_embeds, attention_mask_ext = self._prepend_meta(
            input_ids, attention_mask, metadata
        )
        return self.inner(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_ext,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    @torch.no_grad()
    def generate_with_meta(self, input_ids, attention_mask, metadata, **gen_kwargs):
        inputs_embeds, attention_mask_ext = self._prepend_meta(
            input_ids, attention_mask, metadata
        )
        return self.inner.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_ext,
            **gen_kwargs,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

class SEPForcer(LogitsProcessor):
    """Force SEP tokens at fixed decoder positions during autoregressive generation."""
    def __init__(self, forced: list):
        # forced = [(pos, token_id), ...]  where pos is 1-indexed in the output
        # The `pos - 1` converts to 0-indexed length of input_ids at that step.
        self._forced = {pos - 1: tok for pos, tok in forced}

    def __call__(self, input_ids, scores):
        cur = input_ids.shape[-1] - 1
        if cur in self._forced:
            scores[:] = -float("inf")
            scores[:, self._forced[cur]] = 0.0
        return scores



def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def init_mode_embeddings(inner) -> None:
    """Initialise all new tokens (4096–4104) as EOS + noise."""
    w         = inner.get_input_embeddings().weight
    noise_std = EMBED_INIT_STD_MULTIPLIER * w[:4096].std().item()
    with torch.no_grad():
        base = w[1].clone()   # EOS token
        gen  = torch.Generator(); gen.manual_seed(SEED)
        for tok_id in range(4096, EXTENDED_VOCAB):
            w[tok_id] = base + noise_std * torch.randn(
                w.shape[1], generator=gen, dtype=w.dtype
            )
    print(f"  Embed init : eos_plus_noise  noise_std={noise_std:.5f}")
    for name, tid in [("FAST", 4096), ("TSF", 4102), ("SEP_1", 4103), ("SEP_2", 4104)]:
        cos = F.cosine_similarity(
            w[tid].float().unsqueeze(0), w[1].float().unsqueeze(0)
        ).item()
        print(f"  Cosine sim : {name}↔EOS = {cos:.3f}")


def load_baseline_mae(path: Path) -> dict:
    ref = {"mean_mae": None, "median_mae": None}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("mean_mae"):
                ref["mean_mae"]   = float(line.split(":")[-1].strip())
            elif line.startswith("median_mae"):
                ref["median_mae"] = float(line.split(":")[-1].strip())
    return ref


def write_config(path: Path, job_id: str, fast_ratio: float,
                 dataset_ratio: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "Experiment Configuration", "=" * 40,
        f"Job ID         : {job_id}",
        f"Script         : fine_tune_joint_m4h_sarsim0.py",
        f"Model          : {MODEL_ID}",
        f"Datasets       : m4hourly-tsf.arrow  +  sarsim0-tsf.arrow",
        f"Dataset ratio  : {dataset_ratio:.0%} SarSim0 / {1-dataset_ratio:.0%} M4H",
        f"Fast ratio     : {fast_ratio:.0%} FAST  /  {1-fast_ratio:.0%} TSF",
        f"Conditioning   : Mechanism 1 — sinusoidal scalar prepend",
        f"                 H_MAX={H_MAX}  L_MAX={L_MAX}  D_MODEL={D_MODEL}  D_SIN={D_SIN}",
        f"MetadataEmbed  : ~66K new params (Linear + LayerNorm)  LR_META={LR_META}",
        f"Vocab          : extended to {EXTENDED_VOCAB}",
        f"Steps          : {N_STEPS:,}  batch={BATCH_SIZE}  LR_backbone={LR}",
        f"M4H            : H=48  L_ctx=96   SEPs@(49,98)   TSF 3×48  FAST 48",
        f"SarSim0        : H=64  L_ctx=512  SEPs@(65,130)  TSF 3×64  FAST 64",
        f"Early stop     : patience={EARLY_STOP_PATIENCE}  min_delta={EARLY_STOP_MIN_DELTA}"
        f"  metric=avg(norm_M4H_TSF, norm_SS_TSF)",
        "",
        "Note: batches are dataset-homogeneous — M4H and SarSim0 are never mixed.",
        f"Embed init     : eos_plus_noise  (multiplier={EMBED_INIT_STD_MULTIPLIER})",
    ]) + "\n")
    print(f"  Config written: {path.name}")


# ── data ──────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> tuple[list, list]:
    """
    Load an Arrow file, shuffle with SEED=42, split into
    (train_rows, eval_rows) where eval_rows is the first EVAL_POOL samples.
    """
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()

    item_ids  = table.column("item_id").to_pylist()
    contexts  = table.column("context").to_pylist()
    targets   = table.column("target").to_pylist()
    s_firsts  = table.column("s_first").to_pylist()
    s_seconds = table.column("s_second").to_pylist()
    s_thirds  = table.column("s_third").to_pylist()

    rows = [
        {
            "item_id": item_id,
            "context": np.array(ctx, dtype=np.float32),
            "target":  np.array(tgt, dtype=np.float32),
            "s_parts": [float(s1), float(s2), float(s3)],
        }
        for item_id, ctx, tgt, s1, s2, s3 in zip(
            item_ids, contexts, targets, s_firsts, s_seconds, s_thirds
        )
    ]

    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[EVAL_POOL:], rows[:EVAL_POOL]   # (train, eval)


def pretokenise_tsf(rows: list, tokenizer, ds_cfg: dict) -> list:
    """
    Tokenise rows for TSF mode.

    Decoder labels  = [trend(H) | SEP_1 | seasonal(H) | SEP_2 | forecast(H) | EOS]
    decoder_input   = [TSF_TOKEN | trend(H) | SEP_1 | seasonal(H) | SEP_2 | forecast(H)]
    """
    N   = len(rows)
    H   = ds_cfg["H"]

    contexts = torch.tensor(np.stack([r["context"] for r in rows]), dtype=torch.float32)
    targets  = torch.tensor(np.stack([r["target"]  for r in rows]), dtype=torch.float32)

    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    label_ids, _, _             = tokenizer._input_transform(context=targets, scale=scale)

    parts    = [label_ids[:, i*H:(i+1)*H] for i in range(3)]
    sep1_col = torch.full((N, 1), SEP_1_TOKEN_ID, dtype=label_ids.dtype)
    sep2_col = torch.full((N, 1), SEP_2_TOKEN_ID, dtype=label_ids.dtype)
    eos_col  = torch.full((N, 1), tokenizer.config.eos_token_id, dtype=label_ids.dtype)

    labels_seq    = torch.cat(
        [parts[0], sep1_col, parts[1], sep2_col, parts[2], eos_col], dim=1
    )
    labels        = torch.clone(labels_seq)   # all positions supervised
    mode_col      = torch.full((N, 1), ds_cfg["tsf"]["token_id"], dtype=token_ids.dtype)
    dec_input_ids = torch.cat([mode_col, labels_seq[:, :-1]], dim=1)

    return [{
        "item_id":       rows[i]["item_id"],
        "context":       rows[i]["context"],
        "target":        rows[i]["target"],
        "s_parts":       rows[i]["s_parts"],
        "input_ids":     token_ids[i],
        "attn_mask":     attn_mask[i],
        "dec_input_ids": dec_input_ids[i],
        "labels":        labels[i],
    } for i in range(N)]


def pretokenise_fast(rows: list, tokenizer, ds_cfg: dict) -> list:
    """
    Tokenise rows for FAST mode.

    target          = forecast portion only (last H of the 3H-token target)
    Decoder labels  = [forecast(H) | EOS]
    decoder_input   = [FAST_TOKEN | forecast(H)]
    """
    N = len(rows)
    H = ds_cfg["H"]

    contexts     = torch.tensor(
        np.stack([r["context"]       for r in rows]), dtype=torch.float32
    )
    fast_targets = torch.tensor(
        np.stack([r["target"][2*H:]  for r in rows]), dtype=torch.float32
    )

    token_ids, attn_mask, scale = tokenizer.context_input_transform(contexts)
    label_ids, label_mask, _   = tokenizer._input_transform(
        context=fast_targets, scale=scale
    )

    eos_col       = torch.full((N, 1), tokenizer.config.eos_token_id, dtype=label_ids.dtype)
    labels_seq    = torch.cat([label_ids, eos_col], dim=1)
    ones          = torch.ones((N, 1), dtype=label_ids.dtype)
    mask_seq      = torch.cat([label_mask, ones], dim=1)
    labels        = labels_seq.masked_fill(~mask_seq.bool(), -100)
    mode_col      = torch.full((N, 1), ds_cfg["fast"]["token_id"], dtype=token_ids.dtype)
    dec_input_ids = torch.cat([mode_col, labels_seq[:, :-1]], dim=1)

    return [{
        "item_id":       rows[i]["item_id"],
        "context":       rows[i]["context"],
        "target":        rows[i]["target"],
        "s_parts":       rows[i]["s_parts"],
        "input_ids":     token_ids[i],
        "attn_mask":     attn_mask[i],
        "dec_input_ids": dec_input_ids[i],
        "labels":        labels[i],
    } for i in range(N)]


class CyclicLoader:
    """Infinite cyclic dataloader with per-epoch shuffle."""
    def __init__(self, rows: list, batch_size: int, seed: int = 0):
        self.rows       = rows
        self.batch_size = batch_size
        self.rng        = np.random.default_rng(seed)
        self._idx       = np.arange(len(rows))
        self._pos       = len(rows)   # trigger shuffle on first call

    def next_batch(self) -> list:
        if self._pos + self.batch_size > len(self._idx):
            self.rng.shuffle(self._idx)
            self._pos = 0
        b = [self.rows[i] for i in self._idx[self._pos:self._pos + self.batch_size]]
        self._pos += self.batch_size
        return b


def assemble_batch(batch: list, ds_cfg: dict, device) -> tuple:
    """Stack tokenised tensors and build a per-batch metadata tensor."""
    ids  = torch.stack([d["input_ids"]     for d in batch]).to(device)
    mask = torch.stack([d["attn_mask"]     for d in batch]).to(device)
    dec  = torch.stack([d["dec_input_ids"] for d in batch]).to(device)
    lbl  = torch.stack([d["labels"]        for d in batch]).to(device)
    B    = len(batch)
    meta = (
        normalise_metadata(ds_cfg["H"], ds_cfg["L_ctx"])
        .unsqueeze(0).expand(B, -1).to(device)
    )
    return ids, mask, dec, lbl, meta


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_tsf(model: ChronosWithMetadata, pipeline,
                contexts: list, ds_cfg: dict, device) -> np.ndarray:
    """
    Generate TSF predictions.
    Returns (B, pred_len) array of normalised values (tokenizer-scale).
    """
    H       = ds_cfg["H"]
    gen_len = ds_cfg["tsf"]["gen_len"]
    mc      = pipeline.model.config
    B       = len(contexts)

    ctx  = torch.tensor(np.stack(contexts), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)
    meta = normalise_metadata(H, ds_cfg["L_ctx"]).unsqueeze(0).expand(B, -1).to(device)

    lp  = LogitsProcessorList([SEPForcer(ds_cfg["tsf"]["forced"])])
    raw = model.generate_with_meta(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        metadata=meta,
        logits_processor=lp,
        generation_config=GenerationConfig(
            min_new_tokens=gen_len, max_new_tokens=gen_len,
            do_sample=True, num_return_sequences=EVAL_SAMPLES,
            decoder_start_token_id=ds_cfg["tsf"]["token_id"],
            eos_token_id=mc.eos_token_id,
            pad_token_id=mc.pad_token_id,
            temperature=mc.temperature,
            top_k=mc.top_k, top_p=mc.top_p,
        ),
    )
    raw = raw[:, 1:]   # strip decoder start token
    # Strip SEPs: [part0(H) | SEP_1 | part1(H) | SEP_2 | part2(H)]
    raw_bins = torch.cat(
        [raw[:, :H], raw[:, H+1:2*H+1], raw[:, 2*H+2:]], dim=1
    )
    raw_bins = raw_bins.reshape(B, EVAL_SAMPLES, ds_cfg["tsf"]["pred_len"])
    return pipeline.tokenizer.output_transform(
        raw_bins.cpu(), scale
    ).median(dim=1).values.numpy()


@torch.no_grad()
def predict_fast(model: ChronosWithMetadata, pipeline,
                 contexts: list, ds_cfg: dict, device) -> np.ndarray:
    """
    Generate FAST predictions.
    Returns (B, H) array in original (denormalised) units.
    """
    H  = ds_cfg["H"]
    mc = pipeline.model.config
    B  = len(contexts)

    ctx  = torch.tensor(np.stack(contexts), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)
    meta = normalise_metadata(H, ds_cfg["L_ctx"]).unsqueeze(0).expand(B, -1).to(device)

    raw = model.generate_with_meta(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        metadata=meta,
        generation_config=GenerationConfig(
            min_new_tokens=H, max_new_tokens=H,
            do_sample=True, num_return_sequences=EVAL_SAMPLES,
            decoder_start_token_id=ds_cfg["fast"]["token_id"],
            eos_token_id=mc.eos_token_id,
            pad_token_id=mc.pad_token_id,
            temperature=mc.temperature,
            top_k=mc.top_k, top_p=mc.top_p,
        ),
    )
    raw      = raw[:, 1:]   # strip decoder start token
    raw_bins = raw[:, :H].reshape(B, EVAL_SAMPLES, H)
    return pipeline.tokenizer.output_transform(
        raw_bins.cpu(), scale
    ).median(dim=1).values.numpy()


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_tsf(model: ChronosWithMetadata, pipeline,
                 eval_rows: list, ds_cfg: dict, device) -> float:
    """Normalised MAE of TSF predictions vs stored 3-part target."""
    model.eval()
    maes = []
    for i in range(0, len(eval_rows), EVAL_BATCH):
        batch = eval_rows[i:i + EVAL_BATCH]
        preds = predict_tsf(model, pipeline, [r["context"] for r in batch],
                            ds_cfg, device)
        for r, pred in zip(batch, preds):
            maes.append(float(np.mean(np.abs(pred - r["target"]))))
    model.train()
    return float(np.mean(maes))


def evaluate_fast(model: ChronosWithMetadata, pipeline,
                  eval_rows: list, ds_cfg: dict, device) -> float:
    """Original-unit MAE of FAST predictions vs denormalised ground-truth forecast."""
    H = ds_cfg["H"]
    model.eval()
    maes = []
    for i in range(0, len(eval_rows), EVAL_BATCH):
        batch = eval_rows[i:i + EVAL_BATCH]
        preds = predict_fast(model, pipeline, [r["context"] for r in batch],
                             ds_cfg, device)
        for r, pred in zip(batch, preds):
            s_ctx = max(float(np.mean(np.abs(r["context"]))), SCALE_EPS)
            gt    = r["target"][2*H:] * (r["s_parts"][2] / s_ctx)
            maes.append(float(np.mean(np.abs(pred - gt))))
    model.train()
    return float(np.mean(maes))


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_loss(
    loss_steps, losses, eval_steps,
    m4h_tsf_maes, m4h_fast_maes,
    sarsim0_tsf_maes, sarsim0_fast_maes,
    avg_tsf_maes,
    baselines: dict, path: Path,
    job_id: str, fast_ratio: float, dataset_ratio: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(21, 8))

    # (0,0) Training loss
    ax = axes[0, 0]
    ax.plot(loss_steps, losses, color="steelblue", lw=0.8, alpha=0.6)
    if len(losses) >= 50:
        sm = np.convolve(losses, np.ones(50) / 50, mode="valid")
        ax.plot(loss_steps[49:], sm, color="steelblue", lw=1.8, label="Loss (MA-50)")
    ax.set_title("Training Loss (CE)", fontweight="bold")
    ax.set_xlabel("Step"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (1,0) Normalised avg TSF MAE — early-stop metric
    ax = axes[1, 0]
    ax.plot(eval_steps, avg_tsf_maes, color="indigo", lw=1.8, marker="o", ms=3,
            label="Norm avg TSF MAE")
    ax.set_title("Avg TSF MAE — Early-stop Metric (normalised)", fontweight="bold")
    ax.set_xlabel("Step"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (0,1) M4H TSF MAE
    ax = axes[0, 1]
    ax.plot(eval_steps, m4h_tsf_maes, color="darkorange", lw=1.8, marker="o", ms=3,
            label="M4H TSF MAE (normalised)")
    ax.set_title("M4H — Eval MAE (TSF Mode)", fontweight="bold")
    ax.set_xlabel("Step"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (0,2) SarSim0 TSF MAE
    ax = axes[0, 2]
    ax.plot(eval_steps, sarsim0_tsf_maes, color="darkorange", lw=1.8, marker="o", ms=3,
            label="SarSim0 TSF MAE (normalised)")
    ax.set_title("SarSim0 — Eval MAE (TSF Mode)", fontweight="bold")
    ax.set_xlabel("Step"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (1,1) M4H FAST MAE vs pretrained baseline
    ax = axes[1, 1]
    ax.plot(eval_steps, m4h_fast_maes, color="seagreen", lw=1.8, marker="o", ms=3,
            label="M4H FAST MAE")
    bm = baselines.get("m4h", {})
    if bm.get("mean_mae") is not None:
        ax.axhline(bm["mean_mae"],   color="grey",  lw=1.2, ls="--",
                   label=f"Pretrained mean  {bm['mean_mae']:.4f}")
    if bm.get("median_mae") is not None:
        ax.axhline(bm["median_mae"], color="black", lw=1.0, ls=":",
                   label=f"Pretrained median {bm['median_mae']:.4f}")
    ax.set_title("M4H — FAST MAE (retention check)", fontweight="bold")
    ax.set_xlabel("Step"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (1,2) SarSim0 FAST MAE vs pretrained baseline
    ax = axes[1, 2]
    ax.plot(eval_steps, sarsim0_fast_maes, color="seagreen", lw=1.8, marker="o", ms=3,
            label="SarSim0 FAST MAE")
    bs = baselines.get("sarsim0", {})
    if bs.get("mean_mae") is not None:
        ax.axhline(bs["mean_mae"],   color="grey",  lw=1.2, ls="--",
                   label=f"Pretrained mean  {bs['mean_mae']:.4f}")
    if bs.get("median_mae") is not None:
        ax.axhline(bs["median_mae"], color="black", lw=1.0, ls=":",
                   label=f"Pretrained median {bs['median_mae']:.4f}")
    ax.set_title("SarSim0 — FAST MAE (retention check)", fontweight="bold")
    ax.set_xlabel("Step"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Joint M4H+SarSim0  ·  chronos-t5-small  ·  Mechanism 1 scalar conditioning"
        f"  ·  job {job_id}"
        f"  ·  fast_ratio={fast_ratio:.0%}"
        f"  ·  ds_ratio={dataset_ratio:.0%} SarSim0",
        fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_eval_samples_tsf(
    model: ChronosWithMetadata, pipeline, eval_rows: list,
    ds_cfg: dict, device, path: Path,
    job_id: str, fast_ratio: float,
    n: int = EVAL_N_SHOW,
) -> None:
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(eval_rows), size=min(n, len(eval_rows)), replace=False))
    sel   = [eval_rows[i] for i in picks]
    model.eval()
    preds = predict_tsf(model, pipeline, [r["context"] for r in sel], ds_cfg, device)
    model.train()

    H       = ds_cfg["H"]
    L_ctx   = ds_cfg["L_ctx"]
    ctx_x   = np.arange(L_ctx)
    parts_x = [np.arange(L_ctx + i*H, L_ctx + (i+1)*H) for i in range(3)]

    fig, axes_raw = plt.subplots(n, 1, figsize=(18, 3 * n))
    axes = np.asarray(axes_raw)
    fig.suptitle(
        f"Joint TSF · {ds_cfg['name']}  ·  job {job_id}  ·  fast_ratio={fast_ratio:.0%}\n"
        "seagreen=Trend  mediumpurple=Seasonal  darkorange=Forecast  crimson=Prediction",
        fontsize=11, fontweight="bold",
    )
    for idx, (row, pred) in enumerate(zip(sel, preds)):
        ax    = axes[idx]
        s_ctx = float(np.mean(np.abs(row["context"])))
        inv   = [s / s_ctx for s in row["s_parts"]]

        ax.fill_between(ctx_x, row["context"], alpha=0.05, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL, lw=0.5, label="Context")
        for i in range(3):
            tgt_i  = row["target"][i*H:(i+1)*H] * inv[i]
            pred_i = pred[i*H:(i+1)*H]           * inv[i]
            ax.plot(parts_x[i], tgt_i,  color=ds_cfg["tsf"]["part_cols"][i],
                    lw=1.8, label=f"Target {ds_cfg['tsf']['part_names'][i]}")
            ax.plot(parts_x[i], pred_i, color=_PRED_COL, lw=1.4, ls="--", alpha=0.85)
        ax.axvline(L_ctx - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        for i in range(1, 3):
            ax.axvline(L_ctx + i*H - 0.5, color="grey", lw=0.6, ls=":", alpha=0.5)
        mae_norm = float(np.mean(np.abs(pred - row["target"])))
        ax.set_title(
            f"{row['item_id']}  MAE(norm)={mae_norm:.4f}", fontsize=8, loc="left", pad=2
        )
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.7)
    axes[-1].set_xlabel("Time step", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id",        default="nojobid",
                        help="SLURM job ID — used in directory names")
    parser.add_argument("--fast-ratio",    type=float, default=0.20,
                        help="Fraction of steps using FAST mode (default 0.20)")
    parser.add_argument("--dataset-ratio", type=float, default=0.50,
                        help="Fraction of steps drawn from SarSim0 (default 0.50)")
    args          = parser.parse_args()
    job_id        = args.job_id
    fast_ratio    = args.fast_ratio
    dataset_ratio = args.dataset_ratio

    FIGURES_DIR = (
        FINETUNING_ROOT / "figures" / "finetunes"
        / f"finetune_joint_m4h_sarsim0_{job_id}"
    )
    CKPT_DIR = (
        FINETUNING_ROOT / "checkpoints"
        / f"finetune-joint-m4h-sarsim0-{job_id}"
    )

    # ── DDP setup ─────────────────────────────────────────────────────────────
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
        baselines = {
            "m4h":     load_baseline_mae(M4H_BASELINE_REF),
            "sarsim0": load_baseline_mae(SARSIM0_BASELINE_REF),
        }
        print("=" * 65)
        print("Joint TSF+FAST  ·  M4H + SarSim0  ·  Mechanism 1 conditioning")
        print(f"  Job ID         : {job_id}")
        print(f"  Device         : {device}  DDP={IS_DDP}  world={WORLD_SIZE}")
        print(f"  Dataset ratio  : {dataset_ratio:.0%} SarSim0 / {1-dataset_ratio:.0%} M4H")
        print(f"  Fast ratio     : {fast_ratio:.0%} FAST  /  {1-fast_ratio:.0%} TSF")
        print(f"  Steps          : {N_STEPS:,}  batch={BATCH_SIZE}  LR={LR}  LR_meta={LR_META}")
        for ds, ref in baselines.items():
            if ref["mean_mae"] is not None:
                print(f"  Baseline {ds:<8}: mean={ref['mean_mae']:.4f}"
                      f"  median={ref['median_mae']:.4f}")
        print("=" * 65)
        write_config(FIGURES_DIR / "config.txt", job_id, fast_ratio, dataset_ratio)
    else:
        baselines = {}

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"\n[rank {LOCAL_RANK if IS_DDP else 0}] Loading {MODEL_ID} ...")
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.bfloat16
    )
    inner = pipeline.model.model
    if inner.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
    init_mode_embeddings(inner)

    model = ChronosWithMetadata(inner, d_model=D_MODEL, d_sin=D_SIN)

    # ── data ──────────────────────────────────────────────────────────────────
    if IS_MAIN:
        print("\nLoading datasets...")
    m4h_train,     m4h_eval     = load_rows(M4H_DATA_PATH)
    sarsim0_train, sarsim0_eval = load_rows(SARSIM0_DATA_PATH)
    if IS_MAIN:
        print(f"  M4H     : train={len(m4h_train):,}  eval={len(m4h_eval):,}")
        print(f"  SarSim0 : train={len(sarsim0_train):,}  eval={len(sarsim0_eval):,}")
        print("  Pre-tokenising all four (dataset × mode) splits ...")

    m4h_tsf_tok      = pretokenise_tsf( m4h_train,     pipeline.tokenizer, M4H_CFG)
    m4h_fast_tok     = pretokenise_fast(m4h_train,     pipeline.tokenizer, M4H_CFG)
    sarsim0_tsf_tok  = pretokenise_tsf( sarsim0_train, pipeline.tokenizer, SARSIM0_CFG)
    sarsim0_fast_tok = pretokenise_fast(sarsim0_train, pipeline.tokenizer, SARSIM0_CFG)

    per_rank_batch = max(1, BATCH_SIZE // WORLD_SIZE)

    def shard(rows):
        if IS_DDP:
            return [rows[i] for i in range(len(rows)) if i % WORLD_SIZE == LOCAL_RANK]
        return rows

    loaders = {
        ("m4h",     "tsf"):  CyclicLoader(shard(m4h_tsf_tok),      per_rank_batch, seed=SEED),
        ("m4h",     "fast"): CyclicLoader(shard(m4h_fast_tok),     per_rank_batch, seed=SEED+1000),
        ("sarsim0", "tsf"):  CyclicLoader(shard(sarsim0_tsf_tok),  per_rank_batch, seed=SEED+2000),
        ("sarsim0", "fast"): CyclicLoader(shard(sarsim0_fast_tok), per_rank_batch, seed=SEED+3000),
    }

    # Pre-generate per-step schedules — deterministic and identical across ranks.
    _rng          = np.random.default_rng(SEED)
    ds_schedule   = _rng.random(N_STEPS) < dataset_ratio   # True → draw from SarSim0
    mode_schedule = _rng.random(N_STEPS) < fast_ratio      # True → FAST mode

    # ── training ──────────────────────────────────────────────────────────────
    print(f"\n[rank {LOCAL_RANK if IS_DDP else 0}] Moving model to {device} ...")
    model = model.to(device)
    model.train()

    if IS_DDP:
        dp_model = DDP(model, device_ids=[LOCAL_RANK])
        if IS_MAIN:
            print(f"  DDP across {WORLD_SIZE} GPUs")
    else:
        dp_model = model

    optimizer = torch.optim.AdamW(
        [
            {"params": model.meta_embedder.parameters(), "lr": LR_META},
            {"params": model.inner.parameters(),          "lr": LR},
        ],
        weight_decay=1e-2,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, N_STEPS)

    loss_steps              = []; losses = []
    eval_steps              = []
    m4h_tsf_maes            = []; m4h_fast_maes     = []
    sarsim0_tsf_maes        = []; sarsim0_fast_maes = []
    avg_tsf_maes            = []
    running                 = 0.0
    LOG_EVERY               = 50

    best_avg_tsf_mae        = float("inf")
    no_improve_cnt          = 0
    stopped_early           = False
    best_step               = None
    m4h_tsf_init            = None   # set at first eval for normalisation
    ss_tsf_init             = None

    if IS_MAIN:
        print("\nTraining...")
    pbar = tqdm(
        range(1, N_STEPS + 1), total=N_STEPS,
        desc="Training (joint M4H+SarSim0)", file=sys.stdout,
        dynamic_ncols=True, smoothing=0.05,
        miniters=LOG_EVERY, mininterval=0.0, disable=not IS_MAIN,
    )

    for step in pbar:
        use_sarsim0 = bool(ds_schedule[step - 1])
        use_fast    = bool(mode_schedule[step - 1])
        ds_name     = "sarsim0" if use_sarsim0 else "m4h"
        mode_name   = "fast"    if use_fast    else "tsf"
        ds_cfg      = SARSIM0_CFG if use_sarsim0 else M4H_CFG

        batch                     = loaders[(ds_name, mode_name)].next_batch()
        ids, mask, dec, lbl, meta = assemble_batch(batch, ds_cfg, device)

        out  = dp_model(input_ids=ids, attention_mask=mask,
                        decoder_input_ids=dec, labels=lbl, metadata=meta)
        loss = out.loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step(); scheduler.step()
        running += loss.item()

        if step % LOG_EVERY == 0 and IS_MAIN:
            avg    = running / LOG_EVERY; running = 0.0
            lr_now = scheduler.get_last_lr()[0]
            loss_steps.append(step); losses.append(avg)
            tag = f"{ds_name.upper()}/{mode_name.upper()}"
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}", batch=tag)
            tqdm.write(
                f"  step {step:>6,}/{N_STEPS:,}  loss={avg:.4f}"
                f"  lr={lr_now:.2e}  {tag}"
            )

        if step % EVAL_EVERY == 0:
            if IS_DDP:
                dist.barrier()
            if IS_MAIN:
                m4h_tsf  = evaluate_tsf( model, pipeline, m4h_eval,     M4H_CFG,     device)
                m4h_fast = evaluate_fast(model, pipeline, m4h_eval,     M4H_CFG,     device)
                ss_tsf   = evaluate_tsf( model, pipeline, sarsim0_eval, SARSIM0_CFG, device)
                ss_fast  = evaluate_fast(model, pipeline, sarsim0_eval, SARSIM0_CFG, device)

                # Normalise by first-eval values so M4H (real-unit MAE ~700) and
                # SarSim0 (normalised MAE ~1.0) contribute equally to early stop.
                if m4h_tsf_init is None:
                    m4h_tsf_init = max(m4h_tsf, 1e-6)
                    ss_tsf_init  = max(ss_tsf,  1e-6)
                avg_tsf = (m4h_tsf / m4h_tsf_init + ss_tsf / ss_tsf_init) / 2.0

                eval_steps.append(step)
                m4h_tsf_maes.append(m4h_tsf);     m4h_fast_maes.append(m4h_fast)
                sarsim0_tsf_maes.append(ss_tsf);  sarsim0_fast_maes.append(ss_fast)
                avg_tsf_maes.append(avg_tsf)

                tqdm.write(
                    f"  [eval] step={step:,}"
                    f"  M4H TSF={m4h_tsf:.4f}  M4H FAST={m4h_fast:.4f}"
                    f"  SS TSF={ss_tsf:.4f}  SS FAST={ss_fast:.4f}"
                    f"  norm_avg={avg_tsf:.4f}"
                )

                if avg_tsf < best_avg_tsf_mae - EARLY_STOP_MIN_DELTA:
                    best_avg_tsf_mae = avg_tsf
                    no_improve_cnt   = 0
                    best_step        = step
                    # Save best checkpoint immediately so it is never lost.
                    CKPT_DIR.mkdir(parents=True, exist_ok=True)
                    model.inner.save_pretrained(str(CKPT_DIR))
                    torch.save(
                        model.meta_embedder.state_dict(),
                        str(CKPT_DIR / "meta_embedder.pt"),
                    )
                    tqdm.write(
                        f"  [checkpoint] Best saved → step {step:,}"
                        f"  norm_avg_tsf={avg_tsf:.4f}"
                    )
                else:
                    no_improve_cnt += 1
                    tqdm.write(
                        f"  [early stop] No improvement"
                        f"  {no_improve_cnt}/{EARLY_STOP_PATIENCE}"
                        f"  (best={best_avg_tsf_mae:.4f})"
                    )
                    if no_improve_cnt >= EARLY_STOP_PATIENCE:
                        tqdm.write(
                            f"  [early stop] Triggered at step {step:,}."
                            f"  Best norm_avg_tsf={best_avg_tsf_mae:.4f}"
                        )
                        stopped_early = True

            if IS_DDP:
                stop_tensor = torch.tensor(
                    [1 if stopped_early else 0], dtype=torch.long, device=device
                )
                dist.broadcast(stop_tensor, src=0)
                if not IS_MAIN:
                    stopped_early = bool(stop_tensor.item())
                dist.barrier()

            model.train()
            if stopped_early:
                break

    pbar.close()
    if IS_MAIN and stopped_early:
        print(f"\nEarly stop at step {step:,}  (best norm_avg_tsf={best_avg_tsf_mae:.4f})")

    if IS_MAIN:
        if best_step is not None:
            print(
                f"\nBest checkpoint already saved at step {best_step:,}"
                f"  (norm_avg_tsf={best_avg_tsf_mae:.4f})  →  {CKPT_DIR}"
            )
            # Reload best weights so final sample plots use the best model.
            print("Reloading best checkpoint for final plots...")
            finetuned_best = AutoModelForSeq2SeqLM.from_pretrained(
                str(CKPT_DIR), torch_dtype=torch.bfloat16
            )
            model.inner.load_state_dict(finetuned_best.state_dict())
            model.meta_embedder.load_state_dict(
                torch.load(str(CKPT_DIR / "meta_embedder.pt"), map_location=device)
            )
            del finetuned_best
        else:
            print(f"\nNo eval ran — saving final checkpoint → {CKPT_DIR}")
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            model.inner.save_pretrained(str(CKPT_DIR))
            torch.save(
                model.meta_embedder.state_dict(),
                str(CKPT_DIR / "meta_embedder.pt"),
            )

        for name, arr in [
            ("loss_steps",        loss_steps),
            ("losses",            losses),
            ("eval_steps",        eval_steps),
            ("m4h_tsf_maes",      m4h_tsf_maes),
            ("m4h_fast_maes",     m4h_fast_maes),
            ("sarsim0_tsf_maes",  sarsim0_tsf_maes),
            ("sarsim0_fast_maes", sarsim0_fast_maes),
            ("avg_tsf_maes",      avg_tsf_maes),
        ]:
            np.save(str(CKPT_DIR / f"{name}.npy"), np.array(arr))

        print("\nGenerating plots...")
        model.eval()
        plot_loss(
            loss_steps, losses, eval_steps,
            m4h_tsf_maes, m4h_fast_maes,
            sarsim0_tsf_maes, sarsim0_fast_maes,
            avg_tsf_maes, baselines,
            path=FIGURES_DIR / "loss.png",
            job_id=job_id, fast_ratio=fast_ratio, dataset_ratio=dataset_ratio,
        )
        plot_eval_samples_tsf(
            model, pipeline, m4h_eval, M4H_CFG, device,
            path=FIGURES_DIR / "eval_tsf_samples_m4h.png",
            job_id=job_id, fast_ratio=fast_ratio,
        )
        plot_eval_samples_tsf(
            model, pipeline, sarsim0_eval, SARSIM0_CFG, device,
            path=FIGURES_DIR / "eval_tsf_samples_sarsim0.png",
            job_id=job_id, fast_ratio=fast_ratio,
        )

        print(f"\nDone.  Figures → {FIGURES_DIR}/")
        if eval_steps:
            print(f"Final M4H    TSF MAE  : {m4h_tsf_maes[-1]:.4f}")
            print(f"Final M4H    FAST MAE : {m4h_fast_maes[-1]:.4f}")
            print(f"Final SarSim0 TSF MAE : {sarsim0_tsf_maes[-1]:.4f}")
            print(f"Final SarSim0 FAST MAE: {sarsim0_fast_maes[-1]:.4f}")
            for ds, ref in baselines.items():
                if ref.get("mean_mae"):
                    fast_final = (m4h_fast_maes if ds == "m4h" else sarsim0_fast_maes)[-1]
                    ratio = fast_final / ref["mean_mae"]
                    print(f"FAST retention {ds:<8}: {ratio:.3f}x pretrained mean MAE")

    if IS_DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
