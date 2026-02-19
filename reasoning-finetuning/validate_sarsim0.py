#!/usr/bin/env python3
"""
Validate SarSim0 Data with Base Chronos Model

Tests that synthetic data from SarSim0 can be forecasted by the 
base amazon/chronos-t5-small model.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
import datasets

from chronos import ChronosPipeline

# Constants
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 64
NUM_SAMPLES_TO_PLOT = 6

def main():
    print("=" * 60)
    print("VALIDATING SARSIM0 DATA WITH BASE MODEL")
    print("=" * 60)
    
    # Load SarSim0 dataset
    data_path = Path(__file__).parent / "data" / "sarsim0-reasoning.arrow"
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Please run: python prepare_dataset.py --dataset-source sarsim0")
        return
    
    print(f"Loading dataset from: {data_path}")
    ds = datasets.load_dataset("arrow", data_files=str(data_path), split="train")
    print(f"  Loaded {len(ds)} samples")
    
    # Load base model
    print("\nLoading base model (amazon/chronos-t5-small)...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        torch_dtype=torch.float32,
    )
    print(f"  Model loaded on: {device}")
    
    # Select samples for validation
    indices = list(range(min(NUM_SAMPLES_TO_PLOT, len(ds))))
    
    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    all_mse = []
    all_corr = []
    
    print("\nGenerating predictions...")
    for i, idx in enumerate(indices):
        sample = ds[idx]
        target = np.array(sample["target"])
        
        # Split into context and future
        if len(target) >= CONTEXT_LENGTH + PREDICTION_LENGTH:
            context = target[-(CONTEXT_LENGTH + PREDICTION_LENGTH):-PREDICTION_LENGTH]
            future = target[-PREDICTION_LENGTH:]
        else:
            split = max(1, len(target) - PREDICTION_LENGTH)
            context = target[:split]
            future = target[split:split + PREDICTION_LENGTH]
        
        # Make prediction
        context_tensor = torch.tensor(context).float().unsqueeze(0)
        with torch.no_grad():
            forecast = pipeline.predict(
                context_tensor,
                prediction_length=len(future),
                num_samples=20,
            )
        
        # Get median prediction
        pred = forecast.median(dim=1).values[0].numpy()
        pred = pred[:len(future)]  # Ensure same length
        
        # Compute metrics
        mse = np.mean((pred - future[:len(pred)]) ** 2)
        if np.std(pred) > 1e-6 and np.std(future[:len(pred)]) > 1e-6:
            corr = np.corrcoef(pred, future[:len(pred)])[0, 1]
        else:
            corr = 0.0
        
        all_mse.append(mse)
        all_corr.append(corr)
        
        # Plot
        ax = axes[i]
        
        # Context (last 100 points)
        ctx_len = min(100, len(context))
        x_context = np.arange(-ctx_len, 0)
        ax.plot(x_context, context[-ctx_len:], 'gray', linewidth=1, alpha=0.6, label='Context')
        
        # Future/Target
        x_future = np.arange(0, len(future))
        ax.plot(x_future, future, 'g-', linewidth=2, label='Target (SarSim0)')
        
        # Prediction
        x_pred = np.arange(0, len(pred))
        ax.plot(x_pred, pred, 'b--', linewidth=2, label='Prediction (Base Model)')
        
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f"Sample {idx} | Mode: {sample['mode']}\nMSE: {mse:.4f} | Corr: {corr:+.4f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("SarSim0 Synthetic Data vs Base Chronos Predictions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sarsim0_validation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_path}")
    plt.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Samples tested: {len(indices)}")
    print(f"Average MSE: {np.mean(all_mse):.4f}")
    print(f"Average Correlation: {np.mean(all_corr):+.4f}")
    
    if np.mean(all_corr) > 0.5:
        print("\n✓ SarSim0 data is suitable for forecasting!")
    elif np.mean(all_corr) > 0.0:
        print("\n⚠ SarSim0 data shows some predictability")
    else:
        print("\n✗ SarSim0 data may need adjustment")


if __name__ == "__main__":
    main()
