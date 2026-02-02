"""
SarSim0 Covariate Generation - Phase 2 of the extension plan.

Generates correlated covariates for Chronos-2 training.
Uses the multivariate generation infrastructure to create correlated time series,
then designates some as targets and others as covariates.

Covariate types:
- past_covariates: Only available historically (truncated at context length)
- future_covariates: Known into the forecast horizon (full length)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from .config import SarSimConfig
from .multivariate import generate_multivariate_sarsim0_batch


def generate_with_covariates(
    batch_size: int,
    context_length: int,
    prediction_length: int,
    n_targets: int,
    n_past_covariates: int,
    n_future_covariates: int,
    config: SarSimConfig,
    generator: torch.Generator,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, np.ndarray]]:
    """
    Generate correlated target and covariate time series.

    All series are generated together with shared correlation structure,
    then designated as targets or covariates.

    Args:
        batch_size: Number of samples to generate
        context_length: Length of context window
        prediction_length: Length of prediction horizon
        n_targets: Number of target series (1 for univariate, >1 for multivariate)
        n_past_covariates: Number of past-only covariates
        n_future_covariates: Number of known-future covariates
        config: SarSim0 configuration
        generator: PyTorch random generator
        device: Device for tensor generation

    Returns:
        List of dicts in Chronos-2 format:
        {
            "target": np.ndarray,  # (length,) or (n_targets, length)
            "past_covariates": {"cov_0": np.ndarray, ...},  # (context_length,)
            "future_covariates": {"cov_0": np.ndarray, ...},  # (context_length + prediction_length,)
        }
    """
    total_length = context_length + prediction_length
    full_length = total_length + config.burn_in
    n_variates = n_targets + n_past_covariates + n_future_covariates

    if n_variates < 1:
        raise ValueError("Must have at least 1 target")

    # Generate all correlated series together
    all_series = generate_multivariate_sarsim0_batch(
        batch_size=batch_size,
        n_variates=n_variates,
        length=full_length,
        config=config,
        generator=generator,
        device=device,
    )  # Shape: (batch_size, n_variates, series_length)

    series_length = all_series.shape[-1]

    inputs = []
    for b in range(batch_size):
        # Random window selection
        max_start = series_length - total_length
        start = (
            int(torch.randint(0, max(1, max_start + 1), (1,), generator=generator).item())
            if max_start > 0
            else 0
        )

        # Extract window for this sample
        sample = all_series[b, :, start : start + total_length].numpy()

        # Split into targets and covariates
        target_series = sample[:n_targets]  # (n_targets, total_length)
        past_cov_series = sample[n_targets : n_targets + n_past_covariates]
        future_cov_series = sample[n_targets + n_past_covariates :]

        # Format target
        if n_targets == 1:
            target = target_series[0]  # (total_length,) for univariate
        else:
            target = target_series  # (n_targets, total_length) for multivariate

        # Build output dict
        output: Dict[str, np.ndarray | Dict[str, np.ndarray]] = {"target": target}

        # Add past covariates (full length = context + prediction)
        # Chronos-2 expects past_covariates to have same length as target
        if n_past_covariates > 0:
            past_covariates = {}
            for i in range(n_past_covariates):
                # Full length - Chronos-2 will handle masking during inference
                past_covariates[f"past_cov_{i}"] = past_cov_series[i, :total_length]
            output["past_covariates"] = past_covariates

        # Add future covariates
        # - past_covariates dict: full length (context + prediction)
        # - future_covariates dict: prediction_length only
        if n_future_covariates > 0:
            for i in range(n_future_covariates):
                cov_name = f"future_cov_{i}"
                # Add full history to past_covariates
                if "past_covariates" not in output:
                    output["past_covariates"] = {}
                output["past_covariates"][cov_name] = future_cov_series[i, :total_length]
                # Add future part to future_covariates
                if "future_covariates" not in output:
                    output["future_covariates"] = {}
                output["future_covariates"][cov_name] = future_cov_series[i, context_length:total_length]

        inputs.append(output)

    return inputs


def generate_mixed_with_covariates(
    batch_size: int,
    context_length: int,
    prediction_length: int,
    config: SarSimConfig,
    generator: torch.Generator,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, np.ndarray]]:
    """
    Generate a mixed batch with varying:
    - Univariate vs multivariate targets
    - With/without covariates
    - Different numbers of past/future covariates

    Args:
        batch_size: Number of samples to generate
        context_length: Length of context window
        prediction_length: Length of prediction horizon
        config: SarSim0 configuration
        generator: PyTorch random generator
        device: Device for tensor generation

    Returns:
        List of dicts in Chronos-2 format with mixed configurations
    """
    inputs = []
    seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
    rng = np.random.default_rng(seed)

    for _ in range(batch_size):
        # Decide if this sample has covariates
        has_covariates = rng.random() < config.with_covariates_prob

        if has_covariates:
            # Decide multivariate vs univariate
            is_multivariate = rng.random() < config.multivariate_prob
            n_targets = (
                rng.integers(config.n_variates_range[0], config.n_variates_range[1] + 1)
                if is_multivariate
                else 1
            )

            # Sample number of covariates
            n_past_cov = rng.integers(
                config.n_past_covariates_range[0], config.n_past_covariates_range[1] + 1
            )
            n_future_cov = rng.integers(
                config.n_future_covariates_range[0], config.n_future_covariates_range[1] + 1
            )

            # Generate single sample with covariates
            sample = generate_with_covariates(
                batch_size=1,
                context_length=context_length,
                prediction_length=prediction_length,
                n_targets=n_targets,
                n_past_covariates=n_past_cov,
                n_future_covariates=n_future_cov,
                config=config,
                generator=generator,
                device=device,
            )[0]
            inputs.append(sample)
        else:
            # Generate without covariates (use existing mixed generation)
            is_multivariate = rng.random() < config.multivariate_prob

            total_length = context_length + prediction_length
            full_length = total_length + config.burn_in

            if is_multivariate:
                n_variates = rng.integers(
                    config.n_variates_range[0], config.n_variates_range[1] + 1
                )
                y = generate_multivariate_sarsim0_batch(
                    batch_size=1,
                    n_variates=n_variates,
                    length=full_length,
                    config=config,
                    generator=generator,
                    device=device,
                )
                series_length = y.shape[-1]
                max_start = series_length - total_length
                start = (
                    int(
                        torch.randint(
                            0, max(1, max_start + 1), (1,), generator=generator
                        ).item()
                    )
                    if max_start > 0
                    else 0
                )
                target = y[0, :, start : start + total_length].numpy()
            else:
                from .pipeline import generate_sarsim0_batch

                y = generate_sarsim0_batch(
                    batch_size=1,
                    length=full_length,
                    config=config,
                    generator=generator,
                    device=device,
                )
                series_length = y.shape[-1]
                max_start = series_length - total_length
                start = (
                    int(
                        torch.randint(
                            0, max(1, max_start + 1), (1,), generator=generator
                        ).item()
                    )
                    if max_start > 0
                    else 0
                )
                target = y[0, start : start + total_length].numpy()

            inputs.append({"target": target})

    return inputs


def generate_sarsim0_chronos2_with_covariates(
    num_series: int,
    context_length: int,
    prediction_length: int,
    config: Optional[SarSimConfig] = None,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """
    Convenience function to generate mixed data with covariates for Chronos-2.

    Args:
        num_series: Number of series to generate
        context_length: Context window length
        prediction_length: Prediction horizon
        config: SarSim0 configuration (uses defaults if None)
        seed: Random seed

    Returns:
        List of dicts with "target", "past_covariates", "future_covariates"
    """
    if config is None:
        config = SarSimConfig()

    # Override window sizes
    config.context_window = context_length
    config.prediction_window = prediction_length

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    return generate_mixed_with_covariates(
        batch_size=num_series,
        context_length=context_length,
        prediction_length=prediction_length,
        config=config,
        generator=generator,
    )
