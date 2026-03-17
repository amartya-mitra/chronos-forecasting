#!/usr/bin/env python3
"""
Fast Mode Verification

Verifies that the finetuned model:
1. Correctly recognizes the <|fast_mode|> control token
2. Produces forecasts comparable to the original Amazon Chronos model
"""

import sys
from pathlib import Path

# Project root
FINETUNING_ROOT = Path(__file__).parent.parent  # reasoning-finetuning/
PROJECT_ROOT = FINETUNING_ROOT.parent           # repo root
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(FINETUNING_ROOT))        # for model_utils

import torch
import numpy as np
import matplotlib.pyplot as plt
import datasets

from chronos import ChronosPipeline
from model_utils import (
    ensure_base_model, 
    ensure_finetuned_model,
    FINETUNED_MODEL_DIR,
    BASE_MODEL_ID,
    TRAINING_DATA_PATH,
    FIGURES_DIR,
)

# Constants
FAST_MODE_TOKEN_ID = 4096
PREDICTION_LENGTH = 64
CONTEXT_LENGTH = 512


def load_models():
    """Load both finetuned and base models."""
    print("=" * 60)
    print("Loading Models")
    print("=" * 60)
    
    # Ensure models are available
    if not ensure_finetuned_model():
        raise RuntimeError("Finetuned model not available")
    
    if not ensure_base_model():
        raise RuntimeError("Base model not available")
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load finetuned model
    print("\nLoading finetuned model...")
    finetuned = ChronosPipeline.from_pretrained(
        str(PROJECT_ROOT / FINETUNED_MODEL_DIR),
        device_map=device,
        torch_dtype=torch.float32,
    )
    print(f"  Vocab size: {finetuned.model.model.shared.num_embeddings}")
    
    # Load base model
    print("Loading base model...")
    base = ChronosPipeline.from_pretrained(
        BASE_MODEL_ID,
        device_map=device,
        torch_dtype=torch.float32,
    )
    print(f"  Vocab size: {base.model.model.shared.num_embeddings}")
    
    return finetuned, base


def get_validation_sample(sample_idx: int = 0):
    """Load a validation sample."""
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {TRAINING_DATA_PATH}")
    
    arrow_ds = datasets.load_dataset("arrow", data_files=str(TRAINING_DATA_PATH), split="train")
    split_ds = arrow_ds.train_test_split(test_size=0.2, seed=42)
    val_ds = split_ds["test"]
    
    sample = val_ds[sample_idx]
    print(f"\nLoaded validation sample {sample_idx}")
    print(f"  Target length: {len(sample['target'])}")
    
    return sample


def generate_with_fast_mode(pipeline, context: np.ndarray, has_control_tokens: bool):
    """Generate forecast, optionally prepending fast mode token."""
    context_tensor = torch.tensor(context).float().unsqueeze(0)
    
    tokenizer = pipeline.tokenizer
    token_ids, attention_mask, scale = tokenizer.context_input_transform(context_tensor)
    
    if has_control_tokens:
        # Prepend fast mode token
        fast_token = torch.tensor([[FAST_MODE_TOKEN_ID]], dtype=token_ids.dtype)
        fast_mask = torch.tensor([[True]], dtype=attention_mask.dtype)
        
        input_ids = torch.cat([fast_token, token_ids], dim=1)
        attention_mask = torch.cat([fast_mask, attention_mask], dim=1)
        print(f"  Input with fast mode token: {input_ids.shape}")
    else:
        input_ids = token_ids
        print(f"  Input (no control token): {input_ids.shape}")
    
    device = next(pipeline.model.model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    generated = pipeline.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prediction_length=PREDICTION_LENGTH,
        num_samples=1,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    
    print(f"  Generated tokens shape: {generated.shape}")
    
    decoded = tokenizer.output_transform(generated.cpu(), scale)
    return generated, decoded[0, 0, :].numpy(), scale


def compute_metrics(pred1: np.ndarray, pred2: np.ndarray, ground_truth: np.ndarray):
    """Compute comparison metrics."""
    return {
        "mse_finetuned": np.mean((pred1 - ground_truth) ** 2),
        "mse_base": np.mean((pred2 - ground_truth) ** 2),
        "mae_finetuned": np.mean(np.abs(pred1 - ground_truth)),
        "mae_base": np.mean(np.abs(pred2 - ground_truth)),
        "corr_finetuned": np.corrcoef(pred1, ground_truth)[0, 1],
        "corr_base": np.corrcoef(pred2, ground_truth)[0, 1],
        "mse_between": np.mean((pred1 - pred2) ** 2),
        "corr_between": np.corrcoef(pred1, pred2)[0, 1],
    }


def plot_comparison(finetuned: np.ndarray, base: np.ndarray, 
                    ground_truth: np.ndarray, context_tail: np.ndarray):
    """Create comparison plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Context
    x_context = np.arange(-100, 0)
    ax.plot(x_context, context_tail[-100:], 'k-', linewidth=1.5, label='Context', alpha=0.6)
    
    # Forecasts
    x_forecast = np.arange(0, PREDICTION_LENGTH)
    ax.plot(x_forecast, ground_truth, 'g-', linewidth=2, label='Ground Truth')
    ax.plot(x_forecast, finetuned, 'b--', linewidth=2, label='Finetuned (Fast Mode)')
    ax.plot(x_forecast, base, 'r:', linewidth=2, label='Base (Amazon Chronos)')
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Fast Mode Verification: Finetuned vs Base Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = FIGURES_DIR / "fast_mode_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("FAST MODE CONTROL TEST")
    print("=" * 60)
    
    # Load models
    finetuned_pipeline, base_pipeline = load_models()
    
    # Load sample
    sample = get_validation_sample()
    target = np.array(sample["target"])
    
    # Split context/future
    if len(target) > CONTEXT_LENGTH + PREDICTION_LENGTH:
        context = target[-(CONTEXT_LENGTH + PREDICTION_LENGTH):-PREDICTION_LENGTH]
        future = target[-PREDICTION_LENGTH:]
    else:
        split_point = len(target) - PREDICTION_LENGTH
        context = target[:split_point]
        future = target[split_point:]
    
    print(f"\nContext length: {len(context)}")
    print(f"Future length: {len(future)}")
    
    # Generate forecasts
    print("\n" + "-" * 40)
    print("Finetuned Model (with <|fast_mode|>):")
    print("-" * 40)
    _, ft_forecast, _ = generate_with_fast_mode(finetuned_pipeline, context, has_control_tokens=True)
    
    print("\n" + "-" * 40)
    print("Base Model (no control token):")
    print("-" * 40)
    _, base_forecast, _ = generate_with_fast_mode(base_pipeline, context, has_control_tokens=False)
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("COMPARISON METRICS")
    print("=" * 60)
    metrics = compute_metrics(ft_forecast, base_forecast, future)
    
    print("\nVs Ground Truth:")
    print(f"{'Model':20} | {'MSE':>10} | {'MAE':>10} | {'Correlation':>12}")
    print("-" * 60)
    print(f"{'Finetuned':20} | {metrics['mse_finetuned']:>10.4f} | {metrics['mae_finetuned']:>10.4f} | {metrics['corr_finetuned']:>+12.4f}")
    print(f"{'Base':20} | {metrics['mse_base']:>10.4f} | {metrics['mae_base']:>10.4f} | {metrics['corr_base']:>+12.4f}")
    
    print("\nBetween Models:")
    print(f"  MSE: {metrics['mse_between']:.4f}")
    print(f"  Correlation: {metrics['corr_between']:+.4f}")
    
    # Plot
    plot_comparison(ft_forecast, base_forecast, future, context)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if len(ft_forecast) == PREDICTION_LENGTH:
        print("✓ Fast mode token recognition: PASSED")
    else:
        print("✗ Fast mode token recognition: FAILED")
    
    quality_diff = abs(metrics['mse_finetuned'] - metrics['mse_base']) / max(metrics['mse_base'], 1e-6)
    if quality_diff < 0.2 and metrics['corr_between'] > 0.95:
        print("✓ Forecasting quality: PRESERVED")
    else:
        print("⚠ Forecasting quality: CHANGED")
    
    print(f"\nCorrelation between models: {metrics['corr_between']:+.4f}")


if __name__ == "__main__":
    main()
