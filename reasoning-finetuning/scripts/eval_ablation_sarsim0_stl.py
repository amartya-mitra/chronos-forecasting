#!/usr/bin/env python3
"""
Ablation evaluation: three-way comparison on SarSim0 held-out set.

Conditions:
  1. Pretrained  — amazon/chronos-t5-small, zero finetuning
  2. FAST-only   — finetuned on SarSim0 (FAST mode, no conditioning)
  3. STL-cond    — finetuned on SarSim0 (FAST mode + STL aux conditioning)

All three conditions use the same 200-row held-out eval set from
sarsim0-10k.arrow (same SEED and EVAL_POOL as the training scripts).

Outputs (written to figures/evals/ablation_sarsim0_stl_{tag}/):
  metrics.txt        — mean/median MAE per condition
  mae_comparison.png — bar chart + box plot of per-sample MAEs
  samples.png        — 5 sample plots, all 3 predictions overlaid

Usage:
  /opt/conda/bin/python3.12 reasoning-finetuning/scripts/eval_ablation_sarsim0_stl.py \\
      --fast-job-id  <JOB_ID_of_fastonly_run>  \\
      --stl-job-id   <JOB_ID_of_stlcond_run>   \\
      [--tag         <output_subfolder_suffix>]
"""

import sys
import argparse
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.seasonal import STL
from transformers import GenerationConfig
from transformers.modeling_outputs import BaseModelOutput
from safetensors.torch import load_file

from chronos import ChronosPipeline

# ── constants (must match training scripts) ───────────────────────────────────
DATA_PATH      = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
MODEL_ID       = "amazon/chronos-t5-small"
CONTEXT_LEN    = 512
HORIZON        = 64
DECOMP_LEN     = 64
D_MODEL        = 512
FAST_TOKEN_ID  = 4096
EXTENDED_VOCAB = 4105
EVAL_SAMPLES   = 20
EVAL_BATCH     = 16
EVAL_POOL      = 200
EVAL_N_SHOW    = 5
SCALE_EPS      = 1e-4
SEED           = 42
EMBED_INIT_STD_MULTIPLIER = 2.0

_CTX_COL      = "steelblue"
_GT_COL       = "darkorange"
_PRETRAIN_COL = "grey"
_FAST_COL     = "royalblue"
_STL_COL      = "crimson"


# ── STL decomposition (identical to training script) ─────────────────────────

def stl_decompose(context: np.ndarray, decomp_len: int = DECOMP_LEN):
    n = len(context)
    fft_mag = np.abs(np.fft.rfft(context - context.mean()))
    fft_mag[0] = 0.0
    freqs = np.fft.rfftfreq(n)
    peak_idx = int(np.argmax(fft_mag[1:])) + 1
    dominant_freq = freqs[peak_idx]
    period = int(round(1.0 / dominant_freq)) if dominant_freq > 1e-6 else 12
    period = int(np.clip(period, 2, n // 2 - 1))
    res = STL(context, period=period, robust=True).fit()
    return res.trend[-decomp_len:].astype(np.float32), res.seasonal[-decomp_len:].astype(np.float32)


# ── AuxiliaryConditioningModule + wrapper (identical to training script) ──────

class AuxiliaryConditioningModule(nn.Module):
    def __init__(self, decomp_len=DECOMP_LEN, d_model=D_MODEL):
        super().__init__()
        self.trend_proj    = nn.Linear(decomp_len, d_model)
        self.seasonal_proj = nn.Linear(decomp_len, d_model)
        self.act = nn.Tanh()

    def forward(self, trend, seasonal):
        return torch.stack([self.act(self.trend_proj(trend)),
                            self.act(self.seasonal_proj(seasonal))], dim=1)


class ChronosWithAuxCond(nn.Module):
    def __init__(self, inner, decomp_len=DECOMP_LEN, d_model=D_MODEL):
        super().__init__()
        self.inner      = inner
        self.aux_module = AuxiliaryConditioningModule(decomp_len, d_model)

    def _augment_encoder(self, input_ids, attention_mask, aux_trend, aux_seasonal):
        enc_out   = self.inner.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc_dtype = enc_out.last_hidden_state.dtype
        aux_tokens = self.aux_module(aux_trend.to(enc_dtype), aux_seasonal.to(enc_dtype))
        aug_hidden = torch.cat([enc_out.last_hidden_state, aux_tokens], dim=1)
        aux_mask   = torch.ones(attention_mask.size(0), 2,
                                dtype=attention_mask.dtype, device=attention_mask.device)
        aug_mask   = torch.cat([attention_mask, aux_mask], dim=1)
        return BaseModelOutput(last_hidden_state=aug_hidden), aug_mask

    @torch.no_grad()
    def generate_with_aux(self, input_ids, attention_mask, aux_trend, aux_seasonal,
                          **gen_kwargs):
        aug_enc_out, aug_mask = self._augment_encoder(
            input_ids, attention_mask, aux_trend, aux_seasonal)
        return self.inner.generate(
            encoder_outputs=aug_enc_out, attention_mask=aug_mask, **gen_kwargs)


# ── data ──────────────────────────────────────────────────────────────────────

def load_eval_rows(path: Path) -> list:
    """Load the same 200-row held-out set used during training."""
    with pa.memory_map(str(path), "r") as src:
        table = ipc.open_file(src).read_all()
    rows = []
    for item_id, ctx, fct in zip(
        table.column("item_id").to_pylist(),
        table.column("context").to_pylist(),
        table.column("forecast").to_pylist(),
    ):
        rows.append({
            "item_id":  item_id,
            "context":  np.array(ctx, dtype=np.float32),
            "forecast": np.array(fct, dtype=np.float32),
        })
    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    eval_rows = rows[:EVAL_POOL]

    print(f"  Computing STL for {len(eval_rows)} eval rows ...")
    for r in tqdm(eval_rows, unit="row", file=sys.stdout):
        r["stl_trend"], r["stl_seasonal"] = stl_decompose(r["context"])
    return eval_rows


# ── inference helpers ─────────────────────────────────────────────────────────

def _gen_config(mc):
    return GenerationConfig(
        min_new_tokens=HORIZON, max_new_tokens=HORIZON,
        do_sample=True, num_return_sequences=EVAL_SAMPLES,
        decoder_start_token_id=FAST_TOKEN_ID,
        eos_token_id=mc.eos_token_id, pad_token_id=mc.pad_token_id,
        temperature=mc.temperature, top_k=mc.top_k, top_p=mc.top_p,
    )


@torch.no_grad()
def predict_pretrained(pipeline, contexts: list, device) -> np.ndarray:
    """Standard ChronosPipeline prediction (uses EOS start, not FAST token)."""
    ctx_tensors = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    pred = pipeline.predict(ctx_tensors, prediction_length=HORIZON,
                            num_samples=EVAL_SAMPLES, limit_prediction_length=False)
    return pred.median(dim=1).values.numpy()


@torch.no_grad()
def predict_fast(inner, pipeline, contexts: list, device) -> np.ndarray:
    B  = len(contexts)
    mc = pipeline.model.config
    ctx = torch.tensor(np.stack(contexts), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)
    raw = inner.generate(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        generation_config=_gen_config(mc),
    )
    raw      = raw[:, 1:]
    raw_bins = raw[:, :HORIZON].reshape(B, EVAL_SAMPLES, HORIZON)
    return pipeline.tokenizer.output_transform(raw_bins.cpu(), scale).median(dim=1).values.numpy()


@torch.no_grad()
def predict_stl_cond(stl_model, pipeline, rows: list, device) -> np.ndarray:
    B  = len(rows)
    mc = pipeline.model.config
    ctx = torch.tensor(np.stack([r["context"] for r in rows]), dtype=torch.float32)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)
    scale_np     = scale.numpy().reshape(-1, 1)
    aux_trend    = torch.tensor(
        np.stack([r["stl_trend"]    for r in rows]) / np.maximum(scale_np, SCALE_EPS),
        dtype=torch.float32).to(device)
    aux_seasonal = torch.tensor(
        np.stack([r["stl_seasonal"] for r in rows]) / np.maximum(scale_np, SCALE_EPS),
        dtype=torch.float32).to(device)
    raw = stl_model.generate_with_aux(
        input_ids=tids.to(device), attention_mask=amask.to(device),
        aux_trend=aux_trend, aux_seasonal=aux_seasonal,
        generation_config=_gen_config(mc),
    )
    raw      = raw[:, 1:]
    raw_bins = raw[:, :HORIZON].reshape(B, EVAL_SAMPLES, HORIZON)
    return pipeline.tokenizer.output_transform(raw_bins.cpu(), scale).median(dim=1).values.numpy()


def run_eval(predict_fn, eval_rows, label) -> np.ndarray:
    """Run prediction in batches, return per-sample MAE array."""
    all_maes = []
    for i in tqdm(range(0, len(eval_rows), EVAL_BATCH),
                  desc=f"  Evaluating {label}", unit="batch", file=sys.stdout):
        batch = eval_rows[i:i + EVAL_BATCH]
        preds = predict_fn(batch)
        for r, pred in zip(batch, preds):
            all_maes.append(float(np.mean(np.abs(pred - r["forecast"]))))
    return np.array(all_maes)


# ── model loading ─────────────────────────────────────────────────────────────

def init_mode_embeddings(inner) -> None:
    w = inner.get_input_embeddings().weight
    noise_std = EMBED_INIT_STD_MULTIPLIER * w[:4096].std().item()
    with torch.no_grad():
        base = w[1].clone()
        gen  = torch.Generator(); gen.manual_seed(SEED)
        for tok_id in range(4096, EXTENDED_VOCAB):
            w[tok_id] = base + noise_std * torch.randn(
                w.shape[1], generator=gen, dtype=w.dtype)


def load_fast_model(job_id: str, pipeline, device):
    """Load FAST-only finetuned backbone."""
    ckpt_dir = FINETUNING_ROOT / "checkpoints" / f"finetune-sarsim0-fastonly-{job_id}"
    print(f"  Loading FAST-only checkpoint: {ckpt_dir}")
    inner = pipeline.model.model
    state = load_file(str(ckpt_dir / "model.safetensors"))
    inner.load_state_dict(state, strict=False)
    inner.to(dtype=torch.bfloat16).to(device).eval()
    return inner


def load_stl_model(job_id: str, pipeline, device):
    """Load STL-cond finetuned backbone + aux module."""
    ckpt_dir = FINETUNING_ROOT / "checkpoints" / f"finetune-sarsim0-stlcond-{job_id}"
    print(f"  Loading STL-cond checkpoint: {ckpt_dir}")
    inner = pipeline.model.model
    state = load_file(str(ckpt_dir / "model.safetensors"))
    inner.load_state_dict(state, strict=False)
    model = ChronosWithAuxCond(inner, decomp_len=DECOMP_LEN, d_model=D_MODEL)
    aux_state = torch.load(str(ckpt_dir / "aux_module.pt"), map_location="cpu")
    model.aux_module.load_state_dict(aux_state)
    model.to(dtype=torch.bfloat16).to(device).eval()
    return model


# ── output ────────────────────────────────────────────────────────────────────

def write_metrics(maes_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["SarSim0 STL Conditioning Ablation — MAE Summary", "=" * 50, ""]
    for label, maes in maes_dict.items():
        lines += [
            f"  {label}",
            f"    mean   MAE : {np.mean(maes):.4f}",
            f"    median MAE : {np.median(maes):.4f}",
            f"    std    MAE : {np.std(maes):.4f}",
            "",
        ]
    # Pairwise improvement
    conds = list(maes_dict.keys())
    if "STL-cond" in maes_dict and "FAST-only" in maes_dict:
        delta_mean = np.mean(maes_dict["FAST-only"]) - np.mean(maes_dict["STL-cond"])
        delta_pct  = 100 * delta_mean / np.mean(maes_dict["FAST-only"])
        lines += [
            f"  STL-cond vs FAST-only: {delta_mean:+.4f} mean MAE ({delta_pct:+.1f}%)",
            f"    positive = STL-cond is better",
        ]
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {path.name}")
    # Echo to stdout
    for l in lines:
        print(l)


def plot_mae_comparison(maes_dict: dict, path: Path) -> None:
    labels = list(maes_dict.keys())
    colors = [_PRETRAIN_COL, _FAST_COL, _STL_COL][:len(labels)]
    means  = [np.mean(maes_dict[l]) for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart: mean MAE per condition
    ax = axes[0]
    bars = ax.bar(labels, means, color=colors, width=0.5, alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_title("Mean MAE by Condition", fontweight="bold")
    ax.set_ylabel("Mean MAE (original units)")
    ax.spines[["top", "right"]].set_visible(False)

    # Box plot: per-sample MAE distribution
    ax = axes[1]
    bp = ax.boxplot(
        [maes_dict[l] for l in labels],
        labels=labels, patch_artist=True,
        medianprops=dict(color="black", lw=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title("Per-Sample MAE Distribution", fontweight="bold")
    ax.set_ylabel("MAE (original units)")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("SarSim0 STL Conditioning Ablation  ·  chronos-t5-small",
                 fontweight="bold")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


def plot_samples(predict_fns: dict, eval_rows: list, path: Path) -> None:
    """
    5 sample plots.  Each row: context | GT forecast vs all 3 predictions.
    """
    rng   = np.random.default_rng(0)
    picks = sorted(rng.choice(len(eval_rows), size=min(EVAL_N_SHOW, len(eval_rows)),
                               replace=False))
    sel   = [eval_rows[i] for i in picks]

    # Run inference for all conditions on selected samples
    preds_dict = {}
    for label, fn in predict_fns.items():
        preds_dict[label] = fn(sel)

    ctx_x = np.arange(CONTEXT_LEN)
    fct_x = np.arange(CONTEXT_LEN, CONTEXT_LEN + HORIZON)

    fig, axes_raw = plt.subplots(EVAL_N_SHOW, 1, figsize=(16, 3 * EVAL_N_SHOW))
    axes = np.asarray(axes_raw)

    colors = {"Pretrained": _PRETRAIN_COL, "FAST-only": _FAST_COL, "STL-cond": _STL_COL}
    ls_map = {"Pretrained": "-", "FAST-only": "--", "STL-cond": ":"}

    fig.suptitle(
        "SarSim0 STL Conditioning Ablation  ·  Sample Predictions\n"
        "Grey=Pretrained  Blue=FAST-only  Red=STL-cond",
        fontsize=11, fontweight="bold")

    for idx, row in enumerate(sel):
        ax = axes[idx]
        ax.fill_between(ctx_x, row["context"], alpha=0.06, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL, lw=0.6,
                label="Context" if idx == 0 else "_")
        ax.plot(fct_x, row["forecast"], color=_GT_COL, lw=1.8,
                label="GT forecast" if idx == 0 else "_")

        for label, preds in preds_dict.items():
            pred = preds[idx]
            mae  = float(np.mean(np.abs(pred - row["forecast"])))
            ax.plot(fct_x, pred, color=colors[label], lw=1.3,
                    ls=ls_map[label], alpha=0.85,
                    label=f"{label} (MAE={mae:.3f})" if idx == 0 else "_")

        ax.axvline(CONTEXT_LEN - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(row["item_id"], fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

    axes[-1].set_xlabel("Time step", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-job-id", required=True,
                        help="Job ID of the FAST-only run (fine_tune_sarsim0_fast_only.py)")
    parser.add_argument("--stl-job-id",  required=True,
                        help="Job ID of the STL-cond run (fine_tune_sarsim0_stl_cond.py)")
    parser.add_argument("--tag", default=None,
                        help="Optional suffix for the output directory")
    args = parser.parse_args()

    tag     = args.tag or f"{args.fast_job_id}_{args.stl_job_id}"
    OUT_DIR = FINETUNING_ROOT / "figures" / "evals" / f"ablation_sarsim0_stl_{tag}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load eval rows ────────────────────────────────────────────────────────
    print("\nLoading eval rows ...")
    eval_rows = load_eval_rows(DATA_PATH)
    print(f"  {len(eval_rows)} eval rows ready.")

    # ── load models ───────────────────────────────────────────────────────────
    print("\nLoading pretrained pipeline ...")
    pipeline_pretrained = ChronosPipeline.from_pretrained(
        MODEL_ID, device_map=str(device), dtype=torch.bfloat16)
    pipeline_pretrained.model.eval()

    # Shared pipeline for finetuned models (we reload backbone weights each time)
    print("\nLoading pipeline for finetuned models ...")
    pipeline_ft = ChronosPipeline.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    if pipeline_ft.model.model.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        pipeline_ft.model.model.resize_token_embeddings(EXTENDED_VOCAB)
    init_mode_embeddings(pipeline_ft.model.model)

    print("\nLoading FAST-only model ...")
    fast_inner = load_fast_model(args.fast_job_id, pipeline_ft, device)

    # Must reload the pipeline inner for STL-cond (load_fast_model modified it in-place)
    print("\nReloading pipeline for STL-cond ...")
    pipeline_stl = ChronosPipeline.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    if pipeline_stl.model.model.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        pipeline_stl.model.model.resize_token_embeddings(EXTENDED_VOCAB)
    init_mode_embeddings(pipeline_stl.model.model)

    print("\nLoading STL-cond model ...")
    stl_model = load_stl_model(args.stl_job_id, pipeline_stl, device)

    # ── evaluation ────────────────────────────────────────────────────────────
    print("\nRunning evaluations ...")

    def fn_pretrained(batch):
        return predict_pretrained(pipeline_pretrained, [r["context"] for r in batch], device)

    def fn_fast(batch):
        return predict_fast(fast_inner, pipeline_ft, [r["context"] for r in batch], device)

    def fn_stl(batch):
        return predict_stl_cond(stl_model, pipeline_stl, batch, device)

    maes = {
        "Pretrained": run_eval(fn_pretrained, eval_rows, "Pretrained"),
        "FAST-only":  run_eval(fn_fast,       eval_rows, "FAST-only"),
        "STL-cond":   run_eval(fn_stl,        eval_rows, "STL-cond"),
    }

    # ── outputs ───────────────────────────────────────────────────────────────
    print(f"\nSaving results to {OUT_DIR}/")
    write_metrics(maes, OUT_DIR / "metrics.txt")
    plot_mae_comparison(maes, OUT_DIR / "mae_comparison.png")

    predict_fns = {
        "Pretrained": fn_pretrained,
        "FAST-only":  fn_fast,
        "STL-cond":   fn_stl,
    }
    plot_samples(predict_fns, eval_rows, OUT_DIR / "samples.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
