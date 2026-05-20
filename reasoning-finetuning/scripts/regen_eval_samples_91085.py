#!/usr/bin/env python3
"""
Standalone script to regenerate eval_samples.png for Job 91085
using the updated plot_eval_samples (fixed reference lines).
"""
import sys
from pathlib import Path

FINETUNING_ROOT = Path(__file__).parent.parent
PROJECT_ROOT    = FINETUNING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

# Import everything from the training script
sys.path.insert(0, str(Path(__file__).parent))
from fine_tune_sarsim0_aux_cond import (
    ChronosWithAuxCond, load_rows, plot_eval_samples,
    DATA_PATH, D_MODEL, DECOMP_LEN, EXTENDED_VOCAB, FINETUNING_ROOT,
    init_mode_embeddings,
)
from chronos import ChronosPipeline

JOB_ID   = "91085"
CKPT_DIR = FINETUNING_ROOT / "checkpoints" / f"finetune-sarsim0-auxcond-{JOB_ID}"
FIG_DIR  = FINETUNING_ROOT / "figures" / "finetunes" / f"finetune_sarsim0_auxcond_{JOB_ID}"
MODEL_ID = "amazon/chronos-t5-small"

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading pipeline ...")
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    inner = pipeline.model.model
    if inner.get_input_embeddings().weight.shape[0] < EXTENDED_VOCAB:
        inner.resize_token_embeddings(EXTENDED_VOCAB)
    init_mode_embeddings(inner)

    model = ChronosWithAuxCond(inner, decomp_len=DECOMP_LEN, d_model=D_MODEL)

    print(f"Loading checkpoint from {CKPT_DIR} ...")
    # Load backbone weights
    from safetensors.torch import load_file
    state = load_file(str(CKPT_DIR / "model.safetensors"))
    missing, unexpected = inner.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}")

    # Load aux module
    aux_state = torch.load(str(CKPT_DIR / "aux_module.pt"), map_location="cpu")
    model.aux_module.load_state_dict(aux_state)
    print("  Checkpoint loaded.")

    model.to(dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    print("Loading eval rows ...")
    _, eval_rows = load_rows(DATA_PATH)
    print(f"  {len(eval_rows):,} eval rows.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating eval_samples.png ...")
    plot_eval_samples(
        model, pipeline, eval_rows, device,
        path=FIG_DIR / "eval_samples.png",
        job_id=JOB_ID,
        freeze_backbone=True,   # Job 91085 was Phase 1
    )
    print(f"Done. Saved to {FIG_DIR / 'eval_samples.png'}")

if __name__ == "__main__":
    main()
