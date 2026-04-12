#!/usr/bin/env python3
"""
FAST-mode evaluation for a joint M4H+SarSim0 finetuned checkpoint.

Loads:
  checkpoints/finetune-joint-m4h-sarsim0-{job_id}/
    model.safetensors   (T5 backbone)
    meta_embedder.pt    (MetadataEmbedder state_dict)

Runs FAST-mode inference (token 4096) on the 100-row eval pool of each
dataset (same SEED shuffle as training) using the metadata-conditioned wrapper.

Outputs (appended into figures/finetunes/finetune_joint_m4h_sarsim0_{job_id}/):
  eval_fast_samples_m4h.png
  eval_fast_metrics_m4h.png
  eval_fast_samples_sarsim0.png
  eval_fast_metrics_sarsim0.png
  eval_fast_summary.txt

Usage:
  python eval_fast_joint_m4h_sarsim0.py --job-id 88272
"""

import argparse
import sys
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from chronos import ChronosPipeline

# ── constants (must stay in sync with fine_tune_joint_m4h_sarsim0.py) ───────────
BASE_MODEL     = "amazon/chronos-t5-small"
EXTENDED_VOCAB = 4105
FAST_TOKEN_ID  = 4096
EVAL_POOL      = 100
SEED           = 42
SCALE_EPS      = 1e-4
NUM_SAMPLES    = 20

H_MAX   = 256
L_MAX   = 512
D_MODEL = 512
D_SIN   = 64

M4H_BASELINE_REF     = FINETUNING_ROOT / "figures" / "evals" / "eval_pretrained_m4hourly" / "baseline_mae.txt"
SARSIM0_BASELINE_REF = FINETUNING_ROOT / "figures" / "evals" / "eval_pretrained_sarsim0"  / "baseline_mae.txt"

DATASET_CFGS = {
    "m4h": {
        "name":      "M4Hourly",
        "path":      FINETUNING_ROOT / "data" / "m4hourly-tsf.arrow",
        "baseline":  M4H_BASELINE_REF,
        "H":         48,
        "L_ctx":     96,
    },
    "sarsim0": {
        "name":      "SarSim0",
        "path":      FINETUNING_ROOT / "data" / "sarsim0-tsf.arrow",
        "baseline":  SARSIM0_BASELINE_REF,
        "H":         64,
        "L_ctx":     512,
    },
}

_CTX_COL  = "steelblue"
_GT_COL   = "darkorange"
_PRED_COL = "crimson"


# ── metadata conditioning (inline from training script) ──────────────────────────

def normalise_metadata(H: int, L_ctx: int) -> torch.Tensor:
    return torch.tensor([H / H_MAX, L_ctx / L_MAX], dtype=torch.float32)


class MetadataEmbedder(nn.Module):
    def __init__(self, d_model: int = D_MODEL, d_sin: int = D_SIN):
        super().__init__()
        self.d_sin = d_sin
        self.proj  = nn.Linear(2 * d_sin, d_model)
        self.norm  = nn.LayerNorm(d_model)
        freqs = 1.0 / (10000 ** (
            torch.arange(0, d_sin // 2, dtype=torch.float32) / (d_sin // 2)
        ))
        self.register_buffer("freqs", freqs)

    def _sin_encode(self, x: torch.Tensor) -> torch.Tensor:
        angles = x.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        h_enc = self._sin_encode(metadata[:, 0])
        l_enc = self._sin_encode(metadata[:, 1])
        emb   = self.norm(self.proj(torch.cat([h_enc, l_enc], dim=-1)))
        return emb.unsqueeze(1)   # [B, 1, d_model]


class ChronosWithMetadata(nn.Module):
    def __init__(self, inner, d_model: int = D_MODEL, d_sin: int = D_SIN):
        super().__init__()
        self.inner         = inner
        self.meta_embedder = MetadataEmbedder(d_model=d_model, d_sin=d_sin)

    def _prepend_meta(self, input_ids, attention_mask, metadata):
        meta_emb = self.meta_embedder(metadata)
        ctx_emb  = self.inner.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([meta_emb, ctx_emb], dim=1)
        meta_mask = torch.ones(
            attention_mask.size(0), 1,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        return inputs_embeds, torch.cat([meta_mask, attention_mask], dim=1)

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


# ── model loading ────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: Path, device: torch.device) -> tuple:
    """
    Returns (model: ChronosWithMetadata, pipeline: ChronosPipeline).
    pipeline is used only for its tokenizer and model.config.
    """
    pipeline  = ChronosPipeline.from_pretrained(
        BASE_MODEL, device_map="cpu", torch_dtype=torch.bfloat16,
    )
    finetuned = AutoModelForSeq2SeqLM.from_pretrained(
        str(ckpt_dir), torch_dtype=torch.bfloat16,
    )
    # Ensure vocab is extended (should already be from checkpoint)
    if finetuned.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        finetuned.resize_token_embeddings(EXTENDED_VOCAB)

    model = ChronosWithMetadata(finetuned.to(device))
    model.meta_embedder.load_state_dict(
        torch.load(str(ckpt_dir / "meta_embedder.pt"), map_location=device)
    )
    model = model.to(device)
    model.eval()
    return model, pipeline


# ── data ─────────────────────────────────────────────────────────────────────────

def load_eval_rows(cfg: dict) -> list:
    """Load the same 100-row eval pool used during training (same SEED shuffle)."""
    with pa.memory_map(str(cfg["path"]), "r") as src:
        table = ipc.open_file(src).read_all()

    item_ids  = table.column("item_id").to_pylist()
    contexts  = table.column("context").to_pylist()
    targets   = table.column("target").to_pylist()
    s_thirds  = table.column("s_third").to_pylist()

    rows = []
    for item_id, ctx, tgt, s3 in zip(item_ids, contexts, targets, s_thirds):
        rows.append({
            "item_id": item_id,
            "context": np.array(ctx, dtype=np.float32),
            "target":  np.array(tgt, dtype=np.float32),
            "s_third": float(s3),
        })

    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[:EVAL_POOL]   # first EVAL_POOL after shuffle = eval split


def denorm_forecast(row: dict, H: int) -> np.ndarray:
    s_ctx = max(float(np.mean(np.abs(row["context"]))), SCALE_EPS)
    return row["target"][2 * H:] * (row["s_third"] / s_ctx)


# ── inference ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_fast_inference(model: ChronosWithMetadata, pipeline: ChronosPipeline,
                       rows: list, cfg: dict, device: torch.device) -> list:
    """Return list of (H,) arrays in original units (median over NUM_SAMPLES)."""
    H     = cfg["H"]
    L_ctx = cfg["L_ctx"]
    mc    = pipeline.model.config
    preds = []

    for row in tqdm(rows, desc=f"FAST {cfg['name']}", unit="sample"):
        ctx = torch.tensor(row["context"]).float().unsqueeze(0)
        tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)
        meta = normalise_metadata(H, L_ctx).unsqueeze(0).to(device)

        raw = model.generate_with_meta(
            input_ids=tids.to(device),
            attention_mask=amask.to(device),
            metadata=meta,
            generation_config=GenerationConfig(
                min_new_tokens=H, max_new_tokens=H,
                do_sample=True, num_return_sequences=NUM_SAMPLES,
                decoder_start_token_id=FAST_TOKEN_ID,
                eos_token_id=mc.eos_token_id,
                pad_token_id=mc.pad_token_id,
                temperature=mc.temperature,
                top_k=mc.top_k, top_p=mc.top_p,
            ),
        )
        raw_bins = raw[:, 1:H + 1].reshape(1, NUM_SAMPLES, H)
        out      = pipeline.tokenizer.output_transform(raw_bins.cpu(), scale)
        preds.append(out.median(dim=1).values[0].numpy().astype(np.float32))

    return preds


# ── metrics ───────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list, preds: list, H: int) -> dict:
    maes, mases = [], []
    for row, pred in zip(rows, preds):
        gt        = denorm_forecast(row, H)
        mae       = float(np.mean(np.abs(pred - gt)))
        naive_mae = float(np.mean(np.abs(np.diff(row["context"]))))
        mases.append(mae / max(naive_mae, 1e-6))
        maes.append(mae)
    return {"mae": np.array(maes, dtype=np.float32),
            "mase": np.array(mases, dtype=np.float32)}


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    data = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                data[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return data


# ── plots ─────────────────────────────────────────────────────────────────────────

def plot_samples(rows: list, preds: list, cfg: dict, path: Path,
                 job_id: str, n_show: int = 16) -> None:
    H     = cfg["H"]
    L_ctx = cfg["L_ctx"]
    rng     = np.random.default_rng(0)
    indices = sorted(rng.choice(len(rows), size=min(n_show, len(rows)), replace=False))
    n_cols  = 4
    n_rows  = (len(indices) + n_cols - 1) // n_cols
    ctx_x   = np.arange(L_ctx)
    fct_x   = np.arange(L_ctx, L_ctx + H)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 3))
    axes_flat = np.asarray(axes).flatten()
    fig.suptitle(
        f"FAST mode (token 4096)  ·  joint finetuned job {job_id}  ·  {cfg['name']} eval pool\n"
        "orange = ground truth   crimson = model forecast",
        fontsize=11, fontweight="bold",
    )
    for ax_idx, si in enumerate(indices):
        ax   = axes_flat[ax_idx]
        row  = rows[si]
        pred = preds[si]
        gt   = denorm_forecast(row, H)
        mae  = float(np.mean(np.abs(pred - gt)))
        ax.fill_between(ctx_x, row["context"], alpha=0.06, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL,  lw=0.5, label="Context")
        ax.plot(fct_x, gt,             color=_GT_COL,   lw=1.5, label="GT")
        ax.plot(fct_x, pred,           color=_PRED_COL, lw=1.5, ls="--", label="Pred")
        ax.axvline(L_ctx - 0.5, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.set_title(f"{row['item_id']}  MAE={mae:.4f}", fontsize=7, loc="left", pad=2)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axes_flat[0].legend(fontsize=7, loc="upper left", framealpha=0.7)
    for ax in axes_flat[len(indices):]:
        ax.set_visible(False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_metrics(metrics: dict, baseline: dict, cfg: dict, path: Path,
                 job_id: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pairs = [
        ("mae",  "MAE",  "steelblue", "mean_mae",  "median_mae"),
        ("mase", "MASE", "seagreen",  "mean_mase", "median_mase"),
    ]
    for ax, (key, label, colour, b_mean_k, b_med_k) in zip(axes, pairs):
        vals      = metrics[key]
        p99       = np.percentile(vals, 99)
        vals_clip = vals[vals <= p99]
        ax.hist(vals_clip, bins=40, color=colour, edgecolor="white", alpha=0.85)
        ax.axvline(np.median(vals), color="black", lw=1.4, ls="--",
                   label=f"Median {np.median(vals):.4f}")
        ax.axvline(np.mean(vals),   color="red",   lw=1.0, ls=":",
                   label=f"Mean {np.mean(vals):.4f}")
        if baseline.get(b_mean_k):
            ax.axvline(baseline[b_mean_k], color="grey",    lw=1.2, ls="-.",
                       label=f"Pretrained mean {baseline[b_mean_k]:.4f}")
        if baseline.get(b_med_k):
            ax.axvline(baseline[b_med_k],  color="dimgrey", lw=1.0, ls=":",
                       label=f"Pretrained median {baseline[b_med_k]:.4f}")
        ax.set_title(f"FAST {label}  ·  {cfg['name']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"FAST mode (token 4096)  ·  joint finetuned job {job_id}"
        f"  ·  {cfg['name']}  ·  n={len(metrics['mae'])}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args   = parser.parse_args()
    job_id = args.job_id

    CKPT_DIR    = FINETUNING_ROOT / "checkpoints" / f"finetune-joint-m4h-sarsim0-{job_id}"
    FIGURES_DIR = FINETUNING_ROOT / "figures" / "finetunes" / f"finetune_joint_m4h_sarsim0_{job_id}"
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(f"FAST-mode eval  ·  Joint M4H+SarSim0  ·  finetuned job {job_id}")
    print(f"  Checkpoint : {CKPT_DIR}")
    print(f"  Figures    : {FIGURES_DIR}")
    print(f"  Eval pool  : {EVAL_POOL} series per dataset  (SEED={SEED})")
    print(f"  Device     : {device}")
    print("=" * 65)

    print("\nLoading checkpoint ...")
    model, pipeline = load_model(CKPT_DIR, device)
    print("  Model loaded (T5 + MetadataEmbedder).")

    summary_lines = [
        f"FAST-mode eval  ·  Joint M4H+SarSim0  ·  finetuned job {job_id}",
        f"n_eval_per_dataset : {EVAL_POOL}",
        "",
    ]

    for ds_key, cfg in DATASET_CFGS.items():
        print(f"\n{'─'*50}")
        print(f"  Dataset : {cfg['name']}  (H={cfg['H']}  L_ctx={cfg['L_ctx']})")

        rows     = load_eval_rows(cfg)
        baseline = load_baseline(cfg["baseline"])
        if baseline.get("mean_mae"):
            print(f"  Pretrained baseline  mean_mae={baseline['mean_mae']:.4f}"
                  f"  median_mae={baseline['median_mae']:.4f}")

        preds   = run_fast_inference(model, pipeline, rows, cfg, device)
        metrics = compute_metrics(rows, preds, cfg["H"])

        mean_mae    = float(np.mean(metrics["mae"]))
        median_mae  = float(np.median(metrics["mae"]))
        mean_mase   = float(np.mean(metrics["mase"]))
        median_mase = float(np.median(metrics["mase"]))

        print(f"\n  {'Metric':>8}  {'Mean':>10}  {'Median':>10}")
        print(f"  {'-'*32}")
        print(f"  {'MAE':>8}  {mean_mae:>10.4f}  {median_mae:>10.4f}")
        print(f"  {'MASE':>8}  {mean_mase:>10.4f}  {median_mase:>10.4f}")
        if baseline.get("mean_mae"):
            ratio = mean_mae / baseline["mean_mae"]
            print(f"  FAST retention (mean MAE vs pretrained): {ratio:.3f}x")

        plot_samples(
            rows, preds, cfg,
            path=FIGURES_DIR / f"eval_fast_samples_{ds_key}.png",
            job_id=job_id,
        )
        plot_metrics(
            metrics, baseline, cfg,
            path=FIGURES_DIR / f"eval_fast_metrics_{ds_key}.png",
            job_id=job_id,
        )

        summary_lines += [
            f"[{cfg['name']}]",
            f"mean_mae    : {mean_mae:.4f}",
            f"median_mae  : {median_mae:.4f}",
            f"mean_mase   : {mean_mase:.4f}",
            f"median_mase : {median_mase:.4f}",
            *(
                [f"retention_vs_pretrained_mean : {mean_mae / baseline['mean_mae']:.4f}x"]
                if baseline.get("mean_mae") else []
            ),
            "",
        ]

    summary_path = FIGURES_DIR / "eval_fast_summary.txt"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines))
    print(f"\n  Summary saved → {summary_path.name}")
    print(f"\nDone.  Figures → {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
