#!/usr/bin/env python3
"""
Prepare Reasoning Dataset

This script generates the training dataset for reasoning mode finetuning.
It creates samples with pre-computed decomposition (trend, seasonality, volatility)
derived from STL decomposition.
"""

import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import datasets
from typing import Optional

# Output paths (relative to reasoning-finetuning/)
REASONING_DIR = Path(__file__).parent
DATA_DIR = REASONING_DIR / "data"
OUTPUT_PATH = DATA_DIR / "gifteval-reasoning.arrow"
SOURCE_PATH = DATA_DIR / "gifteval-subset.arrow"

# Constants
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 64
DECOMPOSITION_LENGTH = 64  # Each component: trend, seasonal, volatility


def compute_decomposition(series: np.ndarray, period: int = 7) -> dict:
    """
    Compute STL decomposition and volatility for a time series.
    
    Args:
        series: Input time series
        period: Seasonality period
    
    Returns:
        Dictionary with trend, seasonal, and volatility arrays
    """
    # Clean the series
    series_clean = pd.Series(series).fillna(method='ffill').fillna(method='bfill').values
    
    # Ensure minimum length for STL
    if len(series_clean) < 2 * period:
        # Pad if necessary
        pad_length = 2 * period - len(series_clean)
        series_clean = np.concatenate([np.full(pad_length, series_clean[0]), series_clean])
    
    try:
        # STL decomposition
        stl = STL(series_clean, period=period, robust=True)
        result = stl.fit()
        
        trend = result.trend
        seasonal = result.seasonal
        resid = result.resid
        
        # Volatility: rolling standard deviation of residuals
        vol_window = max(3, period // 2)
        volatility = pd.Series(resid).rolling(window=vol_window, min_periods=1).std().values
        
        return {
            "trend": trend[-DECOMPOSITION_LENGTH:],
            "seasonal": seasonal[-DECOMPOSITION_LENGTH:],
            "volatility": volatility[-DECOMPOSITION_LENGTH:],
        }
    except Exception as e:
        print(f"Warning: STL failed: {e}, using fallback")
        # Fallback: simple rolling statistics
        trend = pd.Series(series_clean).rolling(window=period, min_periods=1).mean().values
        seasonal = series_clean - trend
        volatility = pd.Series(series_clean).rolling(window=period, min_periods=1).std().values
        
        return {
            "trend": trend[-DECOMPOSITION_LENGTH:],
            "seasonal": seasonal[-DECOMPOSITION_LENGTH:],
            "volatility": volatility[-DECOMPOSITION_LENGTH:],
        }


def create_reasoning_sample(sample: dict, tokenizer, mode: str = "reasoning") -> dict:
    """
    Create a training sample with reasoning tokens.
    
    Args:
        sample: Original sample with 'target' field
        tokenizer: Chronos tokenizer for value→token conversion
        mode: "fast" or "reasoning"
    
    Returns:
        Sample with reasoning_tokens and mode fields
    """
    import torch
    
    target = np.array(sample["target"])
    
    # Use the last (context + future) portion
    total_needed = CONTEXT_LENGTH + PREDICTION_LENGTH
    if len(target) >= total_needed:
        context = target[-(total_needed):-PREDICTION_LENGTH]
        future = target[-PREDICTION_LENGTH:]
    else:
        split = len(target) - PREDICTION_LENGTH
        context = target[:split]
        future = target[split:]
    
    if mode == "fast":
        # Fast mode: just the forecast tokens
        future_tensor = torch.tensor(future).float().unsqueeze(0)
        _, _, scale = tokenizer.context_input_transform(future_tensor)
        tokens = tokenizer.context_input_transform(future_tensor)[0][0].numpy()
        reasoning_tokens = tokens[:PREDICTION_LENGTH].tolist()
    else:
        # Reasoning mode: decomposition + forecast
        decomp = compute_decomposition(context, period=7)
        
        # Combine: trend + seasonal + volatility + forecast
        combined = np.concatenate([
            decomp["trend"],
            decomp["seasonal"],
            decomp["volatility"],
            future
        ])
        
        # Tokenize
        combined_tensor = torch.tensor(combined).float().unsqueeze(0)
        tokens, _, scale = tokenizer.context_input_transform(combined_tensor)
        reasoning_tokens = tokens[0].numpy().tolist()
    
    return {
        **sample,
        "reasoning_tokens": reasoning_tokens,
        "mode": mode,
    }


def main():
    """Generate the reasoning training dataset."""
    print("=" * 60)
    print("PREPARING REASONING DATASET")
    print("=" * 60)
    
    # Load source data
    if not SOURCE_PATH.exists():
        print(f"Error: Source data not found at {SOURCE_PATH}")
        print("Please run prepare_gifteval_subset.py first")
        return
    
    print(f"Loading source data from: {SOURCE_PATH}")
    source_ds = datasets.load_dataset("arrow", data_files=str(SOURCE_PATH), split="train")
    print(f"  Loaded {len(source_ds)} samples")
    
    # Initialize tokenizer
    from chronos.config import ChronosConfig
    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    tokenizer = config.create_tokenizer()
    
    # Create reasoning samples
    print("Generating reasoning samples...")
    reasoning_samples = []
    
    for i, sample in enumerate(source_ds):
        if i % 50 == 0:
            print(f"  Processing sample {i}/{len(source_ds)}")
        
        try:
            # Alternate between fast and reasoning modes
            mode = "fast" if i % 2 == 0 else "reasoning"
            reasoning_sample = create_reasoning_sample(sample, tokenizer, mode=mode)
            reasoning_samples.append(reasoning_sample)
        except Exception as e:
            print(f"  Warning: Skipped sample {i}: {e}")
    
    print(f"  Generated {len(reasoning_samples)} samples")
    
    # Save as Arrow dataset
    print(f"Saving to: {OUTPUT_PATH}")
    reasoning_ds = datasets.Dataset.from_list(reasoning_samples)
    reasoning_ds.to_parquet(str(OUTPUT_PATH).replace(".arrow", ".parquet"))
    
    # Convert to Arrow format
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    table = pq.read_table(str(OUTPUT_PATH).replace(".arrow", ".parquet"))
    with pa.OSFile(str(OUTPUT_PATH), 'wb') as f:
        writer = pa.ipc.new_file(f, table.schema)
        writer.write_table(table)
        writer.close()
    
    # Clean up parquet
    Path(str(OUTPUT_PATH).replace(".arrow", ".parquet")).unlink()
    
    print(f"\n✓ Dataset saved: {OUTPUT_PATH}")
    print(f"  Total samples: {len(reasoning_samples)}")
    print(f"  Fast mode: {sum(1 for s in reasoning_samples if s['mode'] == 'fast')}")
    print(f"  Reasoning mode: {sum(1 for s in reasoning_samples if s['mode'] == 'reasoning')}")


if __name__ == "__main__":
    main()
