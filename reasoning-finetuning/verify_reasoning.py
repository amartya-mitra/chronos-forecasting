#!/usr/bin/env python3
"""
Reasoning Mode Verification

Verifies that the finetuned model produces decomposition components
(trend, seasonality, volatility) that correlate with STL ground truth.
"""

import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import datasets

from chronos import ChronosPipeline
from model_utils import (
    ensure_finetuned_model, 
    FINETUNED_MODEL_DIR,
    TRAINING_DATA_PATH,
    FIGURES_DIR,
)

# Constants
REASONING_MODE_TOKEN_ID = 4097
PREDICTION_LENGTH = 64
CONTEXT_LENGTH = 512
DECOMPOSITION_LENGTH = 64
SEASONAL_AMP = 10.0
VOLATILITY_AMP = 50.0


def load_model():
    """Load the finetuned model."""
    if not ensure_finetuned_model():
        raise RuntimeError("Finetuned model not available")
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Loading model from: {PROJECT_ROOT / FINETUNED_MODEL_DIR}")
    pipeline = ChronosPipeline.from_pretrained(
        str(PROJECT_ROOT / FINETUNED_MODEL_DIR),
        device_map=device,
        torch_dtype=torch.float32,
    )
    print(f"  Vocab size: {pipeline.model.model.shared.num_embeddings}")
    
    return pipeline


def get_validation_sample(sample_idx: int = 0):
    """Load a validation sample."""
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {TRAINING_DATA_PATH}")
    
    arrow_ds = datasets.load_dataset("arrow", data_files=str(TRAINING_DATA_PATH), split="train")
    split_ds = arrow_ds.train_test_split(test_size=0.2, seed=42)
    val_ds = split_ds["test"]
    
    sample = val_ds[sample_idx]
    print(f"Loaded validation sample {sample_idx}")
    print(f"  Target length: {len(sample['target'])}")
    
    return sample


def generate_with_reasoning_mode(pipeline, context: np.ndarray):
    """Generate decomposition + forecast with reasoning mode token."""
    context_tensor = torch.tensor(context).float().unsqueeze(0)
    
    tokenizer = pipeline.tokenizer
    token_ids, attention_mask, scale = tokenizer.context_input_transform(context_tensor)
    
    # Prepend reasoning mode token
    reasoning_token = torch.tensor([[REASONING_MODE_TOKEN_ID]], dtype=token_ids.dtype)
    reasoning_mask = torch.tensor([[True]], dtype=attention_mask.dtype)
    
    input_ids = torch.cat([reasoning_token, token_ids], dim=1)
    attention_mask = torch.cat([reasoning_mask, attention_mask], dim=1)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"First token (should be {REASONING_MODE_TOKEN_ID}): {input_ids[0, 0].item()}")
    
    device = next(pipeline.model.model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Generate more tokens for decomposition + forecast
    max_new_tokens = 3 * DECOMPOSITION_LENGTH + PREDICTION_LENGTH  # 256 tokens
    
    generated = pipeline.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prediction_length=max_new_tokens,
        num_samples=1,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    
    print(f"Generated tokens shape: {generated.shape}")
    
    # Decode to values
    decoded = tokenizer.output_transform(generated.cpu(), scale)
    decoded_values = decoded[0, 0, :].numpy()
    
    # Split into components and de-amplify to recover original scale
    return {
        "trend": decoded_values[:DECOMPOSITION_LENGTH],
        "seasonal": decoded_values[DECOMPOSITION_LENGTH:2*DECOMPOSITION_LENGTH] / SEASONAL_AMP,
        "volatility": decoded_values[2*DECOMPOSITION_LENGTH:3*DECOMPOSITION_LENGTH] / VOLATILITY_AMP,
        "forecast": decoded_values[3*DECOMPOSITION_LENGTH:3*DECOMPOSITION_LENGTH+PREDICTION_LENGTH] if len(decoded_values) > 3*DECOMPOSITION_LENGTH else None,
        "raw_tokens": generated,
        "raw_values": decoded_values,
    }


def compute_ground_truth_decomposition(series: np.ndarray, period: int = 7) -> dict:
    """Compute STL decomposition for ground truth."""
    series_clean = pd.Series(series).fillna(method='ffill').fillna(method='bfill').values
    
    try:
        stl = STL(series_clean, period=period, robust=True)
        result = stl.fit()
        
        trend = result.trend
        seasonal = result.seasonal
        resid = result.resid
        
        # Volatility
        vol_window = max(3, period // 2)
        volatility = pd.Series(resid).rolling(window=vol_window, min_periods=1).std().values
        
        return {
            "trend": trend[-DECOMPOSITION_LENGTH:],
            "seasonal": seasonal[-DECOMPOSITION_LENGTH:],
            "volatility": volatility[-DECOMPOSITION_LENGTH:],
        }
    except Exception as e:
        print(f"Warning: STL failed: {e}")
        return None


def compute_correlation(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute correlation, handling edge cases."""
    if len(pred) != len(gt):
        min_len = min(len(pred), len(gt))
        pred = pred[:min_len]
        gt = gt[:min_len]
    
    if np.std(pred) < 1e-6 or np.std(gt) < 1e-6:
        return 0.0
    
    return np.corrcoef(pred, gt)[0, 1]


def plot_comparison(model_output: dict, ground_truth: dict, future: np.ndarray):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    components = ["trend", "seasonal", "volatility"]
    
    for i, comp in enumerate(components):
        ax = axes[i // 2, i % 2]
        
        gt = ground_truth[comp]
        pred = model_output[comp]
        
        # Truncate to same length
        min_len = min(len(gt), len(pred))
        
        ax.plot(gt[:min_len], 'g-', linewidth=2, label='Ground Truth (STL)')
        ax.plot(pred[:min_len], 'b--', linewidth=2, label='Model Prediction')
        
        corr = compute_correlation(pred[:min_len], gt[:min_len])
        ax.set_title(f'{comp.capitalize()} (Correlation: {corr:+.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Forecast comparison
    ax = axes[1, 1]
    if model_output["forecast"] is not None:
        ax.plot(future, 'g-', linewidth=2, label='Ground Truth')
        ax.plot(model_output["forecast"][:len(future)], 'b--', linewidth=2, label='Model Forecast')
        ax.set_title('Forecast Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Forecast not available', ha='center', va='center')
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "reasoning_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("REASONING MODE DECOMPOSITION VERIFICATION")
    print("=" * 60)
    
    # Load model
    pipeline = load_model()
    
    # Load sample
    sample = get_validation_sample()
    target = np.array(sample["target"])
    
    # Split
    if len(target) > CONTEXT_LENGTH + PREDICTION_LENGTH:
        context = target[-(CONTEXT_LENGTH + PREDICTION_LENGTH):-PREDICTION_LENGTH]
        future = target[-PREDICTION_LENGTH:]
    else:
        split_point = len(target) - PREDICTION_LENGTH
        context = target[:split_point]
        future = target[split_point:]
    
    print(f"Context length: {len(context)}")
    print(f"Future length: {len(future)}")
    
    # Generate with reasoning mode
    print("\n" + "-" * 40)
    print("Generating with <|reasoning_mode|> token...")
    print("-" * 40)
    model_output = generate_with_reasoning_mode(pipeline, context)
    
    # Compute ground truth
    print("\n" + "-" * 40)
    print("Computing STL ground truth...")
    print("-" * 40)
    ground_truth = compute_ground_truth_decomposition(target, period=7)
    
    if ground_truth is None:
        print("Error: Could not compute ground truth")
        return
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON STATISTICS")
    print("=" * 60)
    
    for comp in ["trend", "seasonal", "volatility"]:
        gt = ground_truth[comp]
        pred = model_output[comp]
        
        min_len = min(len(gt), len(pred))
        corr = compute_correlation(pred[:min_len], gt[:min_len])
        mse = np.mean((pred[:min_len] - gt[:min_len]) ** 2)
        
        print(f"{comp.capitalize():12} | Correlation: {corr:+.4f} | MSE: {mse:.4f}")
    
    # Plot
    plot_comparison(model_output, ground_truth, future)
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
