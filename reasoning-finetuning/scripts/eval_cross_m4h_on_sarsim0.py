#!/usr/bin/env python3
"""
Cross-dataset generalization: M4H-finetuned model infers on SarSim0 data.

Loads the M4H-finetuned checkpoint (finetune-m4h-joint-{job_id}) and runs
both FAST (token 4096) and TSF (token 4102) inference on N_SHOW=5 samples
drawn from sarsim0-tsf.arrow.

The M4H model's native HORIZON is 48 (used only for FAST mode output length).
For TSF mode, SEP tokens are forced at SarSim0's native positions (65, 130),
so each generated segment is 64 steps — matching SarSim0 GT exactly.
FAST comparison is still over min(48, 64) = 48 steps (model output length).

Outputs saved to figures/evals/cross_eval/m4h_{job_id}_on_sarsim0/:
  cross_eval_samples.png  — 5-row × 2-col combined FAST | TSF panel

Usage:
  python eval_cross_m4h_on_sarsim0.py --job-id 46008
"""

import argparse
import sys
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from chronos import ChronosPipeline

# ── constants ──────────────────────────────────────────────────────────────────
BASE_MODEL = "amazon/chronos-t5-small"
DATA_PATH  = FINETUNING_ROOT / "data" / "sarsim0-tsf.arrow"

# M4H model's trained HORIZON — used only for FAST mode output length
MODEL_HORIZON = 48

# SarSim0 data dimensions
DATA_CONTEXT_LEN = 512
DATA_HORIZON     = 64                         # GT parts are 3×64

# TSF generation uses DATA-native SEP positions so output aligns with GT exactly
GEN_SEP_POS  = [(65, 4103), (130, 4104)]      # SarSim0-native
GEN_GEN_LEN  = 194                            # 3×64 + 2 SEPs
GEN_PRED_LEN = 192                            # 3×64
GEN_SEG      = DATA_HORIZON                   # 64 steps per segment

# FAST comparison: model generates MODEL_HORIZON steps, GT has DATA_HORIZON steps
FAST_CMP_LEN = MODEL_HORIZON                  # = 48

N_SHOW      = 5
SEED        = 42
NUM_SAMPLES = 20
SCALE_EPS   = 1e-4

PART_COLS  = ["seagreen", "mediumpurple", "darkorange"]
PART_NAMES = ["Trend", "Seasonal", "Forecast"]
_CTX_COL   = "steelblue"
_GT_COL    = "darkorange"
_PRED_COL  = "crimson"


# ── SEP forcer ─────────────────────────────────────────────────────────────────

class SEPForcer(LogitsProcessor):
    def __init__(self, forced: list):
        self._forced = {pos - 1: tok for pos, tok in forced}

    def __call__(self, input_ids, scores):
        cur = input_ids.shape[-1] - 1
        if cur in self._forced:
            scores[:] = -float("inf")
            scores[:, self._forced[cur]] = 0.0
        return scores


# ── data ───────────────────────────────────────────────────────────────────────

def load_samples(n: int) -> list:
    with pa.memory_map(str(DATA_PATH), "r") as src:
        table = ipc.open_file(src).read_all()

    rows = []
    for item_id, ctx, tgt, s1, s2, s3 in zip(
        table.column("item_id").to_pylist(),
        table.column("context").to_pylist(),
        table.column("target").to_pylist(),
        table.column("s_first").to_pylist(),
        table.column("s_second").to_pylist(),
        table.column("s_third").to_pylist(),
    ):
        rows.append({
            "item_id": item_id,
            "context": np.array(ctx,  dtype=np.float32),
            "target":  np.array(tgt,  dtype=np.float32),  # (192,) context-scale-normalised
            "s_parts": [float(s1), float(s2), float(s3)],
        })

    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows[:n]


# ── model ──────────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: Path, device: torch.device) -> ChronosPipeline:
    pipeline  = ChronosPipeline.from_pretrained(BASE_MODEL, device_map="cpu",
                                                torch_dtype=torch.bfloat16)
    finetuned = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_dir),
                                                      torch_dtype=torch.bfloat16)
    pipeline.model.model = finetuned.to(device)
    pipeline.model.model.eval()
    return pipeline


# ── inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_fast(pipeline: ChronosPipeline, row: dict,
                 device: torch.device) -> np.ndarray:
    """FAST mode → returns (MODEL_HORIZON=48,) array in original units."""
    model_cfg = pipeline.model.config
    ctx = torch.tensor(row["context"]).unsqueeze(0)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)

    raw = pipeline.model.model.generate(
        input_ids=tids.to(device),
        attention_mask=amask.to(device),
        generation_config=GenerationConfig(
            min_new_tokens=MODEL_HORIZON, max_new_tokens=MODEL_HORIZON,
            do_sample=True, num_return_sequences=NUM_SAMPLES,
            decoder_start_token_id=4096,
            eos_token_id=model_cfg.eos_token_id,
            pad_token_id=model_cfg.pad_token_id,
            temperature=model_cfg.temperature,
            top_k=model_cfg.top_k, top_p=model_cfg.top_p,
        ),
    )
    raw_bins = raw[:, 1:MODEL_HORIZON + 1].reshape(1, NUM_SAMPLES, MODEL_HORIZON)
    return pipeline.tokenizer.output_transform(raw_bins.cpu(), scale).median(dim=1).values[0].numpy()


@torch.no_grad()
def predict_tsf(pipeline: ChronosPipeline, row: dict,
                device: torch.device) -> np.ndarray:
    """TSF mode → returns (GEN_PRED_LEN=192,) context-normalised array (3×64).
    SEPs forced at SarSim0-native positions (65, 130) so each segment is 64 steps,
    matching SarSim0 GT boundaries exactly."""
    model_cfg = pipeline.model.config
    ctx = torch.tensor(row["context"]).unsqueeze(0)
    tids, amask, scale = pipeline.tokenizer.context_input_transform(ctx)

    lp  = LogitsProcessorList([SEPForcer(GEN_SEP_POS)])
    raw = pipeline.model.model.generate(
        input_ids=tids.to(device),
        attention_mask=amask.to(device),
        logits_processor=lp,
        generation_config=GenerationConfig(
            min_new_tokens=GEN_GEN_LEN, max_new_tokens=GEN_GEN_LEN,
            do_sample=True, num_return_sequences=NUM_SAMPLES,
            decoder_start_token_id=4102,
            eos_token_id=model_cfg.eos_token_id,
            pad_token_id=model_cfg.pad_token_id,
            temperature=model_cfg.temperature,
            top_k=model_cfg.top_k, top_p=model_cfg.top_p,
        ),
    )
    seg = GEN_SEG  # 64
    raw = raw[:, 1:]   # strip mode token
    # strip SEPs: [part0(64) | SEP(1) | part1(64) | SEP(1) | part2(64)]
    raw_bins = torch.cat([raw[:, :seg], raw[:, seg+1:2*seg+1], raw[:, 2*seg+2:3*seg+2]], dim=1)
    raw_bins = raw_bins.reshape(1, NUM_SAMPLES, GEN_PRED_LEN)
    return pipeline.tokenizer.output_transform(raw_bins.cpu(), scale).median(dim=1).values[0].numpy()


# ── plot ───────────────────────────────────────────────────────────────────────

def make_plot(rows: list, fast_preds: list, tsf_preds: list,
              path: Path, job_id: str) -> None:
    n      = len(rows)
    ctx_x  = np.arange(DATA_CONTEXT_LEN)

    fig, axes = plt.subplots(n, 2, figsize=(22, 3.5 * n))
    fig.suptitle(
        f"Cross-generalization: M4H-finetuned model (job {job_id}) → SarSim0 data\n"
        f"Left: FAST mode (token 4096, predicts {MODEL_HORIZON} steps, GT={DATA_HORIZON} steps, compare={FAST_CMP_LEN})\n"
        f"Right: TSF mode (token 4102, SEPs at SarSim0-native positions, predicts 3×{GEN_SEG} = GT)",
        fontsize=11, fontweight="bold",
    )

    for i, (row, fast_pred, tsf_pred) in enumerate(zip(rows, fast_preds, tsf_preds)):
        s_ctx = max(float(np.mean(np.abs(row["context"]))), SCALE_EPS)

        # ── FAST panel ────────────────────────────────────────────────────────
        ax = axes[i, 0]
        gt_fast = row["target"][2 * DATA_HORIZON:] * (row["s_parts"][2] / s_ctx)
        pred_x  = np.arange(DATA_CONTEXT_LEN, DATA_CONTEXT_LEN + MODEL_HORIZON)
        gt_x    = np.arange(DATA_CONTEXT_LEN, DATA_CONTEXT_LEN + DATA_HORIZON)

        ax.fill_between(ctx_x, row["context"], alpha=0.05, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL, lw=0.5, label="Context")
        ax.plot(gt_x,   gt_fast,   color=_GT_COL,   lw=1.8, label=f"GT ({DATA_HORIZON} steps)")
        ax.plot(pred_x, fast_pred, color=_PRED_COL, lw=1.5, ls="--",
                label=f"Pred ({MODEL_HORIZON} steps)")

        cmp_mae = float(np.mean(np.abs(fast_pred[:FAST_CMP_LEN] - gt_fast[:FAST_CMP_LEN])))
        ax.axvline(DATA_CONTEXT_LEN - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(DATA_CONTEXT_LEN + FAST_CMP_LEN - 0.5, color="grey", lw=0.7, ls=":", alpha=0.5)
        ax.set_title(f"{row['item_id']}  FAST  MAE[:{FAST_CMP_LEN}]={cmp_mae:.4f}",
                     fontsize=8, loc="left", pad=2)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

        # ── TSF panel ─────────────────────────────────────────────────────────
        ax = axes[i, 1]
        inv = [s / s_ctx for s in row["s_parts"]]

        ax.fill_between(ctx_x, row["context"], alpha=0.05, color=_CTX_COL)
        ax.plot(ctx_x, row["context"], color=_CTX_COL, lw=0.5, label="Context")

        # Full comparison: GEN_SEG == DATA_HORIZON == 64, no truncation needed
        total_mae_norm = float(np.mean(np.abs(tsf_pred - row["target"])))

        for j in range(3):
            gt_j   = row["target"][j * DATA_HORIZON:(j + 1) * DATA_HORIZON] * inv[j]
            pred_j = tsf_pred[j * GEN_SEG:(j + 1) * GEN_SEG] * inv[j]
            # pred and GT x-ranges are identical since GEN_SEG == DATA_HORIZON
            seg_x  = np.arange(DATA_CONTEXT_LEN + j * DATA_HORIZON,
                                DATA_CONTEXT_LEN + (j + 1) * DATA_HORIZON)
            ax.plot(seg_x, gt_j,   color=PART_COLS[j], lw=1.8, label=f"GT {PART_NAMES[j]}")
            ax.plot(seg_x, pred_j, color=_PRED_COL, lw=1.4, ls="--", alpha=0.85)

        ax.axvline(DATA_CONTEXT_LEN - 0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(f"{row['item_id']}  TSF  MAE(norm)={total_mae_norm:.4f}",
                     fontsize=8, loc="left", pad=2)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1, 0].set_xlabel("Time step", fontsize=9)
    axes[-1, 1].set_xlabel("Time step", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, help="M4H joint finetune job ID")
    args   = parser.parse_args()
    job_id = args.job_id

    CKPT_DIR    = FINETUNING_ROOT / "checkpoints" / f"finetune-m4h-joint-{job_id}"
    FIGURES_DIR = FINETUNING_ROOT / "figures" / "evals" / "cross_eval" / f"m4h_{job_id}_on_sarsim0"
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(f"Cross-eval: M4H-finetuned (job {job_id}) → SarSim0 data")
    print(f"  Checkpoint     : {CKPT_DIR}")
    print(f"  Data           : sarsim0-tsf.arrow  (CONTEXT={DATA_CONTEXT_LEN}, H={DATA_HORIZON})")
    print(f"  Model native   : HORIZON={MODEL_HORIZON}  (FAST only)")
    print(f"  TSF SEPs       : {GEN_SEP_POS}  (SarSim0-native, gen_len={GEN_GEN_LEN})")
    print(f"  FAST compare   : {FAST_CMP_LEN} steps  |  TSF compare: full {GEN_SEG} steps/segment")
    print(f"  Device         : {device}")
    print("=" * 65)

    rows = load_samples(N_SHOW)
    print(f"  Loaded {len(rows)} samples")

    pipeline = load_model(CKPT_DIR, device)
    print("  Model loaded.")

    fast_preds, tsf_preds = [], []
    for row in tqdm(rows, desc="Inference", unit="sample"):
        fast_preds.append(predict_fast(pipeline, row, device))
        tsf_preds.append(predict_tsf(pipeline,  row, device))

    make_plot(rows, fast_preds, tsf_preds,
              FIGURES_DIR / "cross_eval_samples.png", job_id)
    print(f"\nDone. Figures → {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
