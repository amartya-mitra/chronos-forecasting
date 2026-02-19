#!/usr/bin/env python3
"""
SarSim0 Adapter for Reasoning Finetuning

Bridges SarSim0 synthetic data generator with the reasoning-finetuning pipeline.
Generates time series with known decomposition components for training.

Output Modes:
- fast_mode: Raw time series only (for zero-shot forecasting)
- reasoning_mode: Pre-computed decomposition (trend, seasonal, volatility) + forecast
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import torch

# Add SarSim0 to path
SARSIM0_PATH = Path(__file__).parent.parent / "synth-data" / "SarSim0-main"
sys.path.insert(0, str(SARSIM0_PATH))

from sarsim0 import SarSimConfig, SarSim0Generator
from sarsim0.sarima import (
    generate_sarima_batch,
    apply_arma_lfilter_batch,
    build_sarima_polynomials,
    sample_poles,
    poles_to_coeffs,
)

# Constants matching reasoning-finetuning
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 64
DECOMPOSITION_LENGTH = 64


def generate_decomposed_series(
    batch_size: int,
    config: SarSimConfig,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate time series with known decomposition components.
    
    Unlike STL decomposition which approximates, this returns the exact
    components used to construct the series:
    - trend: Low-frequency component (large d, no seasonality)
    - seasonal: Seasonal ARIMA component
    - volatility: Noise component from noisers
    
    Args:
        batch_size: Number of series to generate
        config: SarSim0 configuration
        seed: Random seed
    
    Returns:
        List of dicts with keys: target, trend, seasonal, volatility, mode
    """
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    
    samples = []
    
    for i in range(batch_size):
        # Determine mode (50/50 split)
        mode = "fast" if i % 2 == 0 else "reasoning"
        
        # Generate length for full series
        total_length = CONTEXT_LENGTH + PREDICTION_LENGTH + config.burn_in
        
        # Generate using known components
        sample = _generate_single_decomposed(
            total_length=total_length,
            config=config,
            rng=rng,
            mode=mode,
        )
        
        samples.append(sample)
    
    return samples


def _generate_single_decomposed(
    total_length: int,
    config: SarSimConfig,
    rng: np.random.Generator,
    mode: str,
) -> Dict:
    """Generate a single series with known decomposition."""
    
    # 1. Generate TREND component (low-frequency, high d)
    trend = _generate_trend_component(total_length, config, rng)
    
    # 2. Generate SEASONAL component
    seasonal = _generate_seasonal_component(total_length, config, rng)
    
    # 3. Generate VOLATILITY/NOISE component
    volatility = _generate_volatility_component(total_length, config, rng)
    
    # 4. Combine: y = trend + seasonal + volatility
    combined = trend + seasonal + volatility
    
    # Remove burn-in
    burn_in = config.burn_in
    combined = combined[burn_in:]
    trend = trend[burn_in:]
    seasonal = seasonal[burn_in:]
    volatility = volatility[burn_in:]
    
    # Normalize the COMBINED series only, then rescale each component by the
    # same factor so that the additive relationship
    #   combined = trend + seasonal + volatility
    # is preserved after normalization.
    combined, scale_factor = _normalize_series(combined, return_scale=True)
    trend = trend * scale_factor
    seasonal = seasonal * scale_factor
    volatility = volatility * scale_factor
    
    return {
        "target": combined.tolist(),
        "trend": trend[-DECOMPOSITION_LENGTH:].tolist(),
        "seasonal": seasonal[-DECOMPOSITION_LENGTH:].tolist(),
        "volatility": volatility[-DECOMPOSITION_LENGTH:].tolist(),
        "mode": mode,
        "item_id": f"sarsim0_{rng.integers(0, 1000000)}",
        "freq": "H",  # Hourly (arbitrary)
    }


def _generate_trend_component(
    length: int,
    config: SarSimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate smooth trend using high fractional integration."""
    # Use random walk with drift (d close to 1)
    d = rng.uniform(0.7, 1.0)
    
    # Simple AR(1) with high persistence
    phi = rng.uniform(0.95, 0.99)
    
    # Generate innovations
    epsilon = rng.normal(0, 0.1, length)
    
    # Apply AR(1) filter
    trend = np.zeros(length)
    for t in range(1, length):
        trend[t] = phi * trend[t-1] + epsilon[t]
    
    # Add slow drift
    drift = np.linspace(0, rng.uniform(-1, 1), length)
    trend = trend + drift
    
    return trend


def _generate_seasonal_component(
    length: int,
    config: SarSimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate seasonal pattern using SARIMA."""
    # Pick a seasonal period
    periods = [7, 12, 24, 52]  # Weekly, monthly, daily, yearly
    s = rng.choice(periods)
    
    # Generate pure seasonal pattern
    amplitude = rng.uniform(0.5, 2.0)
    phase = rng.uniform(0, 2 * np.pi)
    
    t = np.arange(length)
    seasonal = amplitude * np.sin(2 * np.pi * t / s + phase)
    
    # Add harmonics for complexity
    if rng.random() > 0.5:
        seasonal += (amplitude / 2) * np.sin(4 * np.pi * t / s + phase)
    
    # Add slight variation
    noise = rng.normal(0, 0.05, length)
    seasonal = seasonal + noise
    
    return seasonal


def _generate_volatility_component(
    length: int,
    config: SarSimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate volatility/noise component."""
    # Choose noise distribution
    noise_type = rng.choice(["gaussian", "heavy_tail", "heteroskedastic"])
    
    if noise_type == "gaussian":
        sigma = rng.uniform(0.1, 0.5)
        volatility = rng.normal(0, sigma, length)
        
    elif noise_type == "heavy_tail":
        # Student-t with low df
        df = rng.uniform(3, 8)
        scale = rng.uniform(0.1, 0.3)
        volatility = rng.standard_t(df, length) * scale
        
    else:  # heteroskedastic
        # Time-varying variance (GARCH-like)
        omega = rng.uniform(0.01, 0.05)
        alpha = rng.uniform(0.05, 0.15)
        beta = rng.uniform(0.7, 0.9)
        
        volatility = np.zeros(length)
        h = omega / (1 - alpha - beta + 1e-6)  # Unconditional variance
        
        for t in range(1, length):
            h = omega + alpha * volatility[t-1]**2 + beta * h
            volatility[t] = rng.normal(0, np.sqrt(max(h, 1e-6)))
    
    return volatility


def _normalize_series(
    series: np.ndarray, return_scale: bool = False
) -> np.ndarray:
    """
    Normalize series to reasonable range.

    When return_scale=True, returns (normalized_series, scale_factor) so the
    caller can apply the same factor to sibling components and preserve
    additive relationships (e.g. combined = trend + seasonal + volatility).
    """
    series = np.nan_to_num(series, nan=0.0, posinf=10.0, neginf=-10.0)

    # Clip extreme values before computing the scale
    series = np.clip(series, -100, 100)

    # Scale so the max absolute value is at most 10
    abs_max = np.max(np.abs(series))
    if abs_max > 10.0:
        scale_factor = 10.0 / abs_max
        series = series * scale_factor
    else:
        scale_factor = 1.0

    if return_scale:
        return series, scale_factor
    return series


# Import amplification factors from prepare_dataset (single source of truth)
from prepare_dataset import SEASONAL_AMP, VOLATILITY_AMP


def create_reasoning_tokens(
    sample: Dict,
    tokenizer,
) -> List[int]:
    """
    Convert decomposition + forecast to token sequence.
    
    Uses context scale for tokenization consistency with inference.
    Applies amplification to seasonal/volatility for bin resolution.
    
    For reasoning_mode:
        [trend_tokens] + [seasonal_amp_tokens] + [volatility_amp_tokens] + [forecast_tokens]
    
    For fast_mode:
        [forecast_tokens] only
    
    Returns:
        Tuple of (token list, context_scale float)
    """
    target = np.array(sample["target"])
    
    # Split into context and future
    if len(target) >= CONTEXT_LENGTH + PREDICTION_LENGTH:
        context = target[-(CONTEXT_LENGTH + PREDICTION_LENGTH):-PREDICTION_LENGTH]
        future = target[-PREDICTION_LENGTH:]
    else:
        split = max(1, len(target) - PREDICTION_LENGTH)
        context = target[:split]
        future = target[split:split + PREDICTION_LENGTH]
    
    # Compute context scale (consistent with inference)
    context_tensor = torch.tensor(context).float().unsqueeze(0)
    _, _, context_scale = tokenizer.context_input_transform(context_tensor)
    
    if sample["mode"] == "fast":
        # Fast mode: tokenize forecast using context scale
        future_tensor = torch.tensor(future).float().unsqueeze(0)
        tokens, _, _ = tokenizer._input_transform(future_tensor, scale=context_scale)
        return tokens[0, :PREDICTION_LENGTH].numpy().tolist(), float(context_scale.item())
    
    else:
        # Reasoning mode: decomposition + forecast
        trend = np.array(sample["trend"])
        seasonal_amp = np.array(sample["seasonal"]) * SEASONAL_AMP
        volatility_amp = np.array(sample["volatility"]) * VOLATILITY_AMP
        
        # Canonical order: trend → seasonal → volatility → forecast
        combined = np.concatenate([trend, seasonal_amp, volatility_amp, future])
        
        # Tokenize using context scale
        combined_tensor = torch.tensor(combined).float().unsqueeze(0)
        tokens, _, _ = tokenizer._input_transform(combined_tensor, scale=context_scale)
        
        return tokens[0].numpy().tolist(), float(context_scale.item())


def generate_sarsim0_dataset(
    num_samples: int = 500,
    config: Optional[SarSimConfig] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate reasoning-finetuning dataset from SarSim0.
    
    Args:
        num_samples: Number of samples to generate
        config: SarSim0 configuration
        seed: Random seed
    
    Returns:
        List of training samples ready for tokenization
    """
    if config is None:
        config = SarSimConfig(
            context_window=CONTEXT_LENGTH,
            prediction_window=PREDICTION_LENGTH,
            burn_in=200,
        )
    
    print(f"Generating {num_samples} SarSim0 samples...")
    samples = generate_decomposed_series(
        batch_size=num_samples,
        config=config,
        seed=seed,
    )
    
    print(f"  Generated {len(samples)} samples")
    print(f"  Fast mode: {sum(1 for s in samples if s['mode'] == 'fast')}")
    print(f"  Reasoning mode: {sum(1 for s in samples if s['mode'] == 'reasoning')}")
    
    return samples


if __name__ == "__main__":
    # Test the adapter
    print("Testing SarSim0 Adapter...")
    
    samples = generate_sarsim0_dataset(num_samples=10, seed=42)
    
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i}:")
        print(f"  Mode: {sample['mode']}")
        print(f"  Target length: {len(sample['target'])}")
        print(f"  Trend length: {len(sample['trend'])}")
        print(f"  Seasonal length: {len(sample['seasonal'])}")
        print(f"  Volatility length: {len(sample['volatility'])}")
