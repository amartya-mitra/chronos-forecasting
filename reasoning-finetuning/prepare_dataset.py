#!/usr/bin/env python3
"""
Prepare Reasoning Dataset

This script generates the training dataset for reasoning mode finetuning.
It creates samples with pre-computed decomposition (trend, seasonality, volatility)
derived from STL decomposition.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import tqdm
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
    series_clean = pd.Series(series).ffill().bfill().values
    
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


# Fixed amplification factors for small-scale decomposition components.
# These bring seasonal/volatility into a token-bin range comparable to trend/forecast,
# ensuring the tokenizer allocates adequate resolution to each component.
# At inference, de-amplify by dividing by these same constants.
SEASONAL_AMP = 10.0
VOLATILITY_AMP = 50.0


def create_reasoning_sample(sample: dict, tokenizer, mode: str = "reasoning") -> dict:
    """
    Create a training sample with reasoning tokens.
    
    Tokenization strategy:
    - Compute scale from the CONTEXT (same scale used at inference)
    - Apply fixed amplification to seasonal/volatility for bin resolution
    - Tokenize all components using context scale
    
    Args:
        sample: Original sample with 'target' field
        tokenizer: Chronos tokenizer for value→token conversion
        mode: "fast" or "reasoning"
    
    Returns:
        Sample with reasoning_tokens, mode, and context_scale fields
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
    
    # Step 1: Compute scale from CONTEXT (consistent with inference)
    context_tensor = torch.tensor(context).float().unsqueeze(0)
    _, _, context_scale = tokenizer.context_input_transform(context_tensor)
    
    if mode == "fast":
        # Fast mode: tokenize forecast using context scale
        future_tensor = torch.tensor(future).float().unsqueeze(0)
        tokens, _, _ = tokenizer._input_transform(future_tensor, scale=context_scale)
        reasoning_tokens = tokens[0, :PREDICTION_LENGTH].numpy().tolist()
    else:
        # Reasoning mode: decomposition + forecast
        decomp = compute_decomposition(context, period=7)
        
        # Step 2: Apply amplification to small-scale components
        trend = decomp["trend"]
        seasonal_amp = decomp["seasonal"] * SEASONAL_AMP
        volatility_amp = decomp["volatility"] * VOLATILITY_AMP
        
        # Step 3: Concatenate in canonical order: trend → seasonal → volatility → forecast
        combined = np.concatenate([trend, seasonal_amp, volatility_amp, future])
        
        # Step 4: Tokenize using context scale (NOT the combined scale)
        combined_tensor = torch.tensor(combined).float().unsqueeze(0)
        tokens, _, _ = tokenizer._input_transform(combined_tensor, scale=context_scale)
        reasoning_tokens = tokens[0].numpy().tolist()
    
    return {
        **sample,
        "reasoning_tokens": reasoning_tokens,
        "mode": mode,
        "context_scale": float(context_scale.item()),
    }


def main(dataset_source: str = "gifteval", num_samples: int = 500, source_path: str = None, output_path: str = None):
    """
    Generate the reasoning training dataset.
    
    Args:
        dataset_source: "gifteval" or "sarsim0"
        num_samples: Number of samples (used for sarsim0)
    """
    import fire
    
    print("=" * 60)
    print("PREPARING REASONING DATASET")
    print("=" * 60)
    print(f"Dataset source: {dataset_source}")
    
    # Initialize tokenizer
    from chronos import ChronosConfig
    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    tokenizer = config.create_tokenizer()
    
    if dataset_source == "sarsim0":
        # Use SarSim0 synthetic data
        reasoning_samples = _prepare_sarsim0_dataset(tokenizer, num_samples)
        current_output_path = DATA_DIR / "sarsim0-reasoning.arrow"
    else:
        # Use GiftEval dataset (default)
        current_source_path = Path(source_path) if source_path else SOURCE_PATH
        reasoning_samples = _prepare_gifteval_dataset(tokenizer, source_path=current_source_path)
        current_output_path = Path(output_path) if output_path else OUTPUT_PATH
    
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as Arrow dataset
    print(f"Saving to: {current_output_path}")
    reasoning_ds = datasets.Dataset.from_list(reasoning_samples)
    reasoning_ds.to_parquet(str(current_output_path).replace(".arrow", ".parquet"))
    
    # Convert to Arrow format
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    table = pq.read_table(str(current_output_path).replace(".arrow", ".parquet"))
    with pa.OSFile(str(current_output_path), 'wb') as f:
        writer = pa.ipc.new_file(f, table.schema)
        writer.write_table(table)
        writer.close()
    
    # Clean up parquet
    Path(str(current_output_path).replace(".arrow", ".parquet")).unlink()
    
    print(f"\n✓ Dataset saved: {current_output_path}")
    print(f"  Total samples: {len(reasoning_samples)}")
    print(f"  Fast mode: {sum(1 for s in reasoning_samples if s['mode'] == 'fast')}")
    print(f"  Reasoning mode: {sum(1 for s in reasoning_samples if s['mode'] == 'reasoning')}")


def _prepare_gifteval_dataset(tokenizer, source_path: Path = SOURCE_PATH) -> list:
    """Prepare dataset from GiftEval source (single-threaded, reliable)."""
    # Load source data
    if not source_path.exists():
        print(f"Error: Source data not found at {source_path}")
        return []
    
    print(f"Loading source data from: {source_path}")
    source_ds = datasets.load_dataset("arrow", data_files=str(source_path), split="train")
    print(f"  Loaded {len(source_ds)} samples")
    
    # Create reasoning samples sequentially with progress bar
    reasoning_samples = []
    for i in tqdm.tqdm(range(len(source_ds)), desc="Generating Reasoning Tokens"):
        try:
            mode = "fast" if i % 2 == 0 else "reasoning"
            sample = source_ds[i]
            result = create_reasoning_sample(sample, tokenizer, mode=mode)
            if result:
                reasoning_samples.append(result)
        except Exception as e:
            if i % 500 == 0:
                print(f"  Warning: Skipped sample {i}: {e}")
    
    print(f"  Generated {len(reasoning_samples)} samples")
    return reasoning_samples


def _prepare_sarsim0_dataset(tokenizer, num_samples: int) -> list:
    """Prepare dataset from SarSim0 synthetic generator."""
    from sarsim0_adapter import generate_sarsim0_dataset, create_reasoning_tokens
    
    print(f"Generating {num_samples} samples from SarSim0...")
    samples = generate_sarsim0_dataset(num_samples=num_samples, seed=42)
    
    # Add reasoning tokens
    print("Adding reasoning tokens...")
    reasoning_samples = []
    
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"  Processing sample {i}/{len(samples)}")
        
        try:
            reasoning_tokens, context_scale = create_reasoning_tokens(sample, tokenizer)
            sample["reasoning_tokens"] = reasoning_tokens
            sample["context_scale"] = context_scale
            reasoning_samples.append(sample)
        except Exception as e:
            print(f"  Warning: Skipped sample {i}: {e}")
    
    print(f"  Generated {len(reasoning_samples)} samples")
    return reasoning_samples


if __name__ == "__main__":
    import fire
    fire.Fire(main)

