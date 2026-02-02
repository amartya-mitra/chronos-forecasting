#!/usr/bin/env python3
"""
Diagnostic Script: Investigate Poor Forecasting Performance

This script tests multiple hypotheses for why forecasting is poor:
1. Are we using the correct base model?
2. Is there an issue with how we call the pipeline?
3. Is the validation sample problematic?
4. Is there a decoding issue?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datasets

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chronos import ChronosPipeline

PREDICTION_LENGTH = 64
CONTEXT_LENGTH = 512


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Investigating Poor Forecasting Performance")
    print("=" * 70)
    
    # =========================================================================
    # 1. Check which models we're loading
    # =========================================================================
    print("\n[1] MODEL INFORMATION")
    print("-" * 70)
    
    # Our "base" model - actually a finetuned checkpoint
    local_base_path = "output/gifteval-subset-finetune/run-0/checkpoint-final"
    print(f"Local 'base' model: {local_base_path}")
    print("  WARNING: This is NOT the original Amazon Chronos model!")
    print("  This was finetuned on gifteval-subset, which may have degraded performance.")
    
    # =========================================================================
    # 2. Load the ORIGINAL Amazon Chronos model for comparison
    # =========================================================================
    print("\n[2] LOADING ORIGINAL AMAZON CHRONOS MODEL")
    print("-" * 70)
    
    try:
        original_pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="mps",
            torch_dtype=torch.float32,
        )
        print("✓ Successfully loaded amazon/chronos-t5-small")
        print(f"  Vocab size: {original_pipeline.model.model.shared.num_embeddings}")
        has_original = True
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        has_original = False
    
    # Load local base model
    print("\n[3] LOADING LOCAL BASE MODEL")
    print("-" * 70)
    local_pipeline = ChronosPipeline.from_pretrained(
        local_base_path,
        device_map="mps",
        torch_dtype=torch.float32,
    )
    print(f"✓ Loaded local model")
    print(f"  Vocab size: {local_pipeline.model.model.shared.num_embeddings}")
    
    # =========================================================================
    # 3. Load validation sample
    # =========================================================================
    print("\n[4] VALIDATION SAMPLE ANALYSIS")
    print("-" * 70)
    
    arrow_ds = datasets.load_dataset("arrow", data_files="gifteval-reasoning.arrow", split="train")
    split_ds = arrow_ds.train_test_split(test_size=0.2, seed=42)
    val_ds = split_ds["test"]
    
    sample = val_ds[0]
    target = np.array(sample["target"])
    
    print(f"Sample 0:")
    print(f"  Total length: {len(target)}")
    print(f"  Min: {np.nanmin(target):.4f}, Max: {np.nanmax(target):.4f}")
    print(f"  Mean: {np.nanmean(target):.4f}, Std: {np.nanstd(target):.4f}")
    print(f"  NaN count: {np.isnan(target).sum()}")
    
    # Split into context and future
    if len(target) > CONTEXT_LENGTH + PREDICTION_LENGTH:
        context = target[-(CONTEXT_LENGTH + PREDICTION_LENGTH):-PREDICTION_LENGTH]
        future = target[-PREDICTION_LENGTH:]
    else:
        split_point = len(target) - PREDICTION_LENGTH
        context = target[:split_point]
        future = target[split_point:]
    
    print(f"\n  Context: {len(context)} points, range [{np.nanmin(context):.2f}, {np.nanmax(context):.2f}]")
    print(f"  Future:  {len(future)} points, range [{np.nanmin(future):.2f}, {np.nanmax(future):.2f}]")
    
    # =========================================================================
    # 4. Use the STANDARD ChronosPipeline.predict() method
    # =========================================================================
    print("\n[5] FORECASTING WITH STANDARD PIPELINE.PREDICT()")
    print("-" * 70)
    print("Using the proper ChronosPipeline.predict() method instead of manual generation...")
    
    context_tensor = torch.tensor(context).float()
    
    # Original Amazon model
    if has_original:
        print("\n  Amazon Chronos-T5-Small:")
        with torch.no_grad():
            original_forecast = original_pipeline.predict(
                inputs=context_tensor,
                prediction_length=PREDICTION_LENGTH,
                num_samples=20,  # Multiple samples for probabilistic forecast
            )
        original_median = np.median(original_forecast[0].numpy(), axis=0)  # [samples, time] -> [time]
        print(f"    Forecast shape: {original_forecast.shape}")
        print(f"    Median forecast range: [{original_median.min():.2f}, {original_median.max():.2f}]")
        
        # Compute metrics
        mse_original = np.mean((original_median - future) ** 2)
        mae_original = np.mean(np.abs(original_median - future))
        corr_original = np.corrcoef(original_median, future)[0, 1]
        print(f"    MSE: {mse_original:.4f}, MAE: {mae_original:.4f}, Corr: {corr_original:+.4f}")
    
    # Local base model
    print("\n  Local Base Model (gifteval-finetuned):")
    with torch.no_grad():
        local_forecast = local_pipeline.predict(
            inputs=context_tensor,
            prediction_length=PREDICTION_LENGTH,
            num_samples=20,
        )
    local_median = np.median(local_forecast[0].numpy(), axis=0)  # [samples, time] -> [time]
    print(f"    Forecast shape: {local_forecast.shape}")
    print(f"    Median forecast range: [{local_median.min():.2f}, {local_median.max():.2f}]")
    
    mse_local = np.mean((local_median - future) ** 2)
    mae_local = np.mean(np.abs(local_median - future))
    corr_local = np.corrcoef(local_median, future)[0, 1]
    print(f"    MSE: {mse_local:.4f}, MAE: {mae_local:.4f}, Corr: {corr_local:+.4f}")
    
    # =========================================================================
    # 5. Compare with our manual generation approach
    # =========================================================================
    print("\n[6] COMPARING STANDARD VS MANUAL GENERATION")
    print("-" * 70)
    
    # Manual approach (what we did in verify_fast_mode.py)
    tokenizer = local_pipeline.tokenizer
    context_tensor_2d = context_tensor.unsqueeze(0)
    token_ids, attention_mask, scale = tokenizer.context_input_transform(context_tensor_2d)
    
    device = next(local_pipeline.model.model.parameters()).device
    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    manual_tokens = local_pipeline.model(
        input_ids=token_ids,
        attention_mask=attention_mask,
        prediction_length=PREDICTION_LENGTH,
        num_samples=1,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    
    manual_decoded = tokenizer.output_transform(manual_tokens.cpu(), scale)
    manual_forecast = manual_decoded[0, 0, :].numpy()
    
    print(f"Manual generation: range [{manual_forecast.min():.2f}, {manual_forecast.max():.2f}]")
    
    mse_manual = np.mean((manual_forecast - future) ** 2)
    mae_manual = np.mean(np.abs(manual_forecast - future))
    corr_manual = np.corrcoef(manual_forecast, future)[0, 1]
    print(f"  MSE: {mse_manual:.4f}, MAE: {mae_manual:.4f}, Corr: {corr_manual:+.4f}")
    
    # =========================================================================
    # 6. Visualization
    # =========================================================================
    print("\n[7] SAVING DIAGNOSTIC PLOT")
    print("-" * 70)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Forecasts comparison
    ax1 = axes[0]
    x_context = np.arange(-100, 0)
    x_future = np.arange(0, PREDICTION_LENGTH)
    
    ax1.plot(x_context, context[-100:], 'k-', linewidth=1.5, label='Context', alpha=0.6)
    ax1.plot(x_future, future, 'g-', linewidth=2, label='Ground Truth')
    
    if has_original:
        # Squeeze batch dimension: (1, 64) -> (64,)
        original_median_plot = original_median.squeeze()
        ax1.plot(x_future, original_median_plot, 'b--', linewidth=2, label=f'Amazon Chronos (MSE={mse_original:.1f})')
    ax1.plot(x_future, local_median, 'r:', linewidth=2, label=f'Local Base (MSE={mse_local:.1f})')
    ax1.plot(x_future, manual_forecast, 'm-.', linewidth=2, label=f'Manual Gen (MSE={mse_manual:.1f})')
    
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Forecast Comparison: Standard Pipeline vs Manual Generation')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Full time series
    ax2 = axes[1]
    ax2.plot(target, 'k-', linewidth=1, alpha=0.7)
    ax2.axvline(x=len(target)-PREDICTION_LENGTH, color='red', linestyle='--', label='Forecast Start')
    ax2.set_title('Full Time Series (Validation Sample 0)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diagnostic_forecasting.png', dpi=150)
    print("Saved: diagnostic_forecasting.png")
    
    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    print("\nKey Findings:")
    if has_original:
        if mse_original < mse_local:
            print(f"  1. Amazon Chronos performs BETTER than local finetuned model")
            print(f"     (MSE: {mse_original:.2f} vs {mse_local:.2f})")
        else:
            print(f"  1. Local model performs similar/better than Amazon Chronos")
    
    if abs(mse_local - mse_manual) / mse_local > 0.1:
        print(f"  2. Standard predict() differs from manual generation")
        print(f"     This suggests our manual approach has issues")
    else:
        print(f"  2. Standard predict() matches manual generation")
    
    if corr_local < 0:
        print(f"  3. NEGATIVE correlation with ground truth ({corr_local:.2f})")
        print(f"     This is very unusual and suggests a fundamental issue")


if __name__ == "__main__":
    main()
