"""
Noiser Modules - Section 4.3 of the paper.

Implements heavy-tailed noise distributions that condition variance
on the local series level using vectorized PyTorch operations:
- Poisson: count spikes for intermittent demand
- Generalized Gamma: controllable burstiness with power transform
- Lognormal: multiplicative, heavy-tailed volatility shocks
- Passthrough: no noise (identity)
"""

import torch
import torch.distributions as dist
from typing import Optional, Tuple
from enum import Enum

from .config import SarSimConfig


class NoiserType(Enum):
    """Types of noisers available."""

    POISSON = 0
    GAMMA = 1
    LOGNORMAL = 2
    PASSTHROUGH = 3


def log_uniform_sample(
    low: float,
    high: float,
    size: Tuple[int, ...],
    generator: torch.Generator,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample from LogUniform distribution (vectorized)."""
    log_low = torch.log(torch.tensor(low, device=device))
    log_high = torch.log(torch.tensor(high, device=device))
    log_samples = torch.empty(size, device=device).uniform_(
        log_low.item(), log_high.item(), generator=generator
    )
    return torch.exp(log_samples)


def compute_rate(
    y: torch.Tensor,
    lambda_0: torch.Tensor,
) -> torch.Tensor:
    """
    Compute time-varying rate from signal level (Equation 9).

    λ_t = λ₀ · (y_t - min(y)) / (max(y) - min(y))

    Args:
        y: Input signal of shape (B, T)
        lambda_0: Base rate parameter of shape (B,) or (B, 1)

    Returns:
        Rate tensor of shape (B, T)
    """
    # Handle NaN/Inf in input
    y_clean = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

    # Per-series min and max
    y_min = y_clean.min(dim=-1, keepdim=True).values
    y_max = y_clean.max(dim=-1, keepdim=True).values

    # Avoid division by zero
    y_range = y_max - y_min
    y_range = torch.where(y_range < 1e-10, torch.ones_like(y_range), y_range)

    # Normalized signal in [0, 1]
    y_normalized = (y_clean - y_min) / y_range

    # Clamp to valid range
    y_normalized = torch.clamp(y_normalized, min=0.0, max=1.0)

    # Ensure lambda_0 is broadcastable
    if lambda_0.dim() == 1:
        lambda_0 = lambda_0.unsqueeze(-1)

    # Scale by base rate, add small offset to avoid zero rates
    lambda_t = lambda_0 * y_normalized + 0.01

    # Ensure positive rates
    lambda_t = torch.clamp(lambda_t, min=1e-4)

    return lambda_t


def poisson_noiser(
    y: torch.Tensor,
    config: SarSimConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Apply Poisson noiser (vectorized): η_t ~ Poisson(λ_t).

    Args:
        y: Input signal of shape (B, T)
        config: Configuration with lambda range
        generator: PyTorch random generator

    Returns:
        Noised signal of shape (B, T)
    """
    B, T = y.shape
    device = y.device

    # Sample base rate from LogUniform for each series
    lambda_0 = log_uniform_sample(
        config.poisson_lambda_range[0],
        config.poisson_lambda_range[1],
        (B,),
        generator,
        device,
    )

    # Compute time-varying rate
    lambda_t = compute_rate(y, lambda_0)

    # Sample from Poisson (vectorized)
    eta = torch.poisson(lambda_t, generator=generator)

    return eta


def gamma_noiser(
    y: torch.Tensor,
    config: SarSimConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Apply Generalized Gamma noiser with power transform (vectorized).

    η'_t ~ Gamma(κ, λ_t/κ)
    η_t = (η'_t)^ζ

    Args:
        y: Input signal of shape (B, T)
        config: Configuration
        generator: PyTorch random generator

    Returns:
        Noised signal of shape (B, T)
    """
    B, T = y.shape
    device = y.device

    # Sample base rate from LogUniform
    lambda_0 = log_uniform_sample(
        config.gamma_lambda_range[0],
        config.gamma_lambda_range[1],
        (B,),
        generator,
        device,
    )

    # Sample shape parameter from LogUniform
    kappa = log_uniform_sample(
        config.gamma_kappa_range[0],
        config.gamma_kappa_range[1],
        (B,),
        generator,
        device,
    )

    # Sample power parameter from Uniform
    zeta = torch.empty(B, device=device).uniform_(
        config.gamma_zeta_range[0],
        config.gamma_zeta_range[1],
        generator=generator,
    )

    # Compute time-varying rate
    lambda_t = compute_rate(y, lambda_0)

    # Gamma parameterization: shape=kappa, rate=kappa/lambda_t
    # PyTorch uses concentration (shape) and rate
    # scale = lambda_t / kappa, rate = kappa / lambda_t
    concentration = kappa.unsqueeze(-1).expand(B, T)
    rate = kappa.unsqueeze(-1) / lambda_t

    # Sample from Gamma (vectorized)
    gamma_dist = dist.Gamma(concentration, rate)
    eta_prime = gamma_dist.sample()

    # Apply power transform
    zeta_expanded = zeta.unsqueeze(-1)
    eta = torch.pow(eta_prime, zeta_expanded)

    return eta


def lognormal_noiser(
    y: torch.Tensor,
    config: SarSimConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Apply Lognormal noiser (vectorized): η'_t ~ LogNormal(μ, κ).

    Mean-preserving parameterization: μ = log(λ_t) - σ²/2

    Args:
        y: Input signal of shape (B, T)
        config: Configuration
        generator: PyTorch random generator

    Returns:
        Noised signal of shape (B, T)
    """
    B, T = y.shape
    device = y.device

    # Sample base rate from LogUniform
    lambda_0 = log_uniform_sample(
        config.lognormal_lambda_range[0],
        config.lognormal_lambda_range[1],
        (B,),
        generator,
        device,
    )

    # Sample shape (sigma) parameter from LogUniform
    kappa = log_uniform_sample(
        config.lognormal_kappa_range[0],
        config.lognormal_kappa_range[1],
        (B,),
        generator,
        device,
    )

    # Compute time-varying rate
    lambda_t = compute_rate(y, lambda_0)

    # Mean-preserving parameterization
    sigma = kappa.unsqueeze(-1).expand(B, T)
    mu = torch.log(lambda_t + 1e-10) - (sigma**2) / 2

    # Sample from LogNormal (vectorized)
    lognormal_dist = dist.LogNormal(mu, sigma)
    eta = lognormal_dist.sample()

    return eta


def apply_noiser_vectorized(
    y: torch.Tensor,
    config: SarSimConfig,
    generator: Optional[torch.Generator] = None,
    noiser_types: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply noisers to a batch of signals (fully vectorized).

    Each series can have a different noiser type applied.

    Args:
        y: Input signals of shape (B, T)
        config: Configuration
        generator: PyTorch random generator
        noiser_types: Optional tensor of noiser types (B,), random if None

    Returns:
        Tuple of (noised signals, noiser type indices)
    """
    if generator is None:
        generator = torch.Generator()

    B, T = y.shape
    device = y.device

    # Randomly select noiser types if not provided
    if noiser_types is None:
        noiser_types = torch.randint(0, 4, (B,), generator=generator, device=device)

    # Initialize output
    result = torch.zeros_like(y)

    # Create masks for each noiser type
    poisson_mask = noiser_types == NoiserType.POISSON.value
    gamma_mask = noiser_types == NoiserType.GAMMA.value
    lognormal_mask = noiser_types == NoiserType.LOGNORMAL.value
    passthrough_mask = noiser_types == NoiserType.PASSTHROUGH.value

    # Apply each noiser to its subset (vectorized per-type)
    if poisson_mask.any():
        result[poisson_mask] = poisson_noiser(y[poisson_mask], config, generator)

    if gamma_mask.any():
        result[gamma_mask] = gamma_noiser(y[gamma_mask], config, generator)

    if lognormal_mask.any():
        result[lognormal_mask] = lognormal_noiser(y[lognormal_mask], config, generator)

    if passthrough_mask.any():
        result[passthrough_mask] = y[passthrough_mask].clone()

    return result, noiser_types
