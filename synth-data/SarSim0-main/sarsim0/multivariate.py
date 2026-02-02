"""
Multivariate SARIMA Generation - Phase 1 Extension.

Generates correlated multivariate time series by:
1. Sampling a valid correlation matrix
2. Generating correlated innovations via Cholesky decomposition
3. Running SARIMA filter on each variate with shared parameters

Output format compatible with Chronos-2: (batch, n_variates, length)
"""

import torch
import numpy as np
from scipy import signal
from typing import Tuple, Optional

from .config import SarSimConfig
from .sarima import (
    sample_poles,
    poles_to_coeffs,
    build_sarima_polynomials,
    fractional_diff_coeffs,
    apply_seasonal_integration_batch,
)
from .noisers import apply_noiser_vectorized


def sample_correlation_matrix(
    n_variates: int,
    strength_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a random valid correlation matrix.

    Uses the "onion" method to generate a random positive definite correlation matrix.

    Args:
        n_variates: Number of variates
        strength_range: (min, max) for off-diagonal correlation magnitudes
        rng: NumPy random generator

    Returns:
        Correlation matrix of shape (n_variates, n_variates)
    """
    if n_variates == 1:
        return np.array([[1.0]])

    # Simple approach: generate random correlations and make PSD
    # Start with identity
    corr = np.eye(n_variates, dtype=np.float64)

    # Fill off-diagonal with random correlations
    min_strength, max_strength = strength_range
    for i in range(n_variates):
        for j in range(i + 1, n_variates):
            # Random sign and magnitude
            sign = rng.choice([-1, 1])
            magnitude = rng.uniform(min_strength, max_strength)
            corr[i, j] = sign * magnitude
            corr[j, i] = corr[i, j]

    # Make positive definite via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0.01)  # Ensure positive eigenvalues
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Normalize to correlation matrix (diag = 1)
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    return corr


def generate_correlated_innovations(
    batch_size: int,
    n_variates: int,
    length: int,
    corr_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate correlated Gaussian innovations.

    Args:
        batch_size: Number of samples
        n_variates: Number of variates
        length: Time series length
        corr_matrix: Correlation matrix (n_variates, n_variates)
        rng: NumPy random generator

    Returns:
        Correlated innovations of shape (batch_size, n_variates, length)
    """
    # Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)

    # Generate independent innovations: (batch, n_variates, length)
    eps_indep = rng.standard_normal((batch_size, n_variates, length))

    # Apply correlation: eps_corr[b, :, t] = L @ eps_indep[b, :, t]
    # Reshape for batch matrix multiply
    eps_corr = np.einsum("ij,bjt->bit", L, eps_indep)

    return eps_corr.astype(np.float64)


def apply_arma_filter_multivariate(
    epsilon: np.ndarray,
    ar_poly: np.ndarray,
    ma_poly: np.ndarray,
) -> np.ndarray:
    """
    Apply ARMA filter to multivariate innovations.

    Args:
        epsilon: Innovations of shape (batch, n_variates, length)
        ar_poly: AR polynomial coefficients
        ma_poly: MA polynomial coefficients

    Returns:
        Filtered output of shape (batch, n_variates, length)
    """
    batch_size, n_variates, length = epsilon.shape

    # Apply lfilter along time axis for each variate
    # Reshape to (batch * n_variates, length) for efficiency
    eps_flat = epsilon.reshape(-1, length)
    y_flat = signal.lfilter(ma_poly, ar_poly, eps_flat, axis=-1)
    y = y_flat.reshape(batch_size, n_variates, length)

    return y


def generate_multivariate_sarima(
    batch_size: int,
    n_variates: int,
    length: int,
    ar_coeffs: np.ndarray,
    ma_coeffs: np.ndarray,
    sar_coeffs: np.ndarray,
    sma_coeffs: np.ndarray,
    s: int,
    d: float,
    D: int,
    corr_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate multivariate SARIMA series with correlated innovations.

    All variates share the same ARMA parameters but have correlated noise.

    Args:
        batch_size: Number of samples
        n_variates: Number of variates
        length: Time series length
        ar_coeffs, ma_coeffs, sar_coeffs, sma_coeffs: SARIMA coefficients
        s: Seasonal period
        d: Fractional differencing order
        D: Seasonal integration order
        corr_matrix: Correlation matrix for innovations
        rng: NumPy random generator

    Returns:
        Array of shape (batch_size, n_variates, length)
    """
    # Generate correlated innovations
    epsilon = generate_correlated_innovations(
        batch_size, n_variates, length, corr_matrix, rng
    )

    # Build SARIMA polynomials
    ar_poly, ma_poly = build_sarima_polynomials(
        ar_coeffs, ma_coeffs, sar_coeffs, sma_coeffs, s
    )

    # Apply ARMA filter
    y = apply_arma_filter_multivariate(epsilon, ar_poly, ma_poly)

    # Fractional integration (apply to each variate)
    if d > 0:
        frac_coeffs = fractional_diff_coeffs(-d)
        # Reshape for convolution
        y_flat = y.reshape(-1, length)
        y_conv = signal.oaconvolve(
            y_flat, frac_coeffs.reshape(1, -1), mode="full", axes=-1
        )
        y = y_conv[:, :length].reshape(batch_size, n_variates, length)

    # Seasonal integration (apply to each variate)
    if D > 0 and s > 0:
        y_flat = y.reshape(-1, length)
        y_int = apply_seasonal_integration_batch(y_flat, D, s)
        y = y_int.reshape(batch_size, n_variates, length)

    # Handle numerical issues
    y = np.clip(y, -1e6, 1e6)
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

    return y


def generate_multivariate_sarima_batch(
    batch_size: int,
    n_variates: int,
    length: int,
    config: SarSimConfig,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
    corr_matrix: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Generate batch of multivariate SARIMA series.

    Args:
        batch_size: Number of samples
        n_variates: Number of variates per sample
        length: Time series length
        config: SarSim0 configuration
        generator: PyTorch generator for seeding
        device: Output device
        corr_matrix: Optional pre-specified correlation matrix

    Returns:
        Tuple of (tensor of shape (batch, n_variates, length), correlation matrix)
    """
    if generator is not None:
        seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
    else:
        seed = int(torch.randint(0, 2**31, (1,)).item())

    rng = np.random.default_rng(seed)

    # Sample correlation matrix if not provided
    if corr_matrix is None:
        corr_matrix = sample_correlation_matrix(
            n_variates,
            config.correlation_strength_range,
            rng,
        )

    # Sample SARIMA parameters (shared across variates)
    p = rng.integers(config.p_range[0], config.p_range[1] + 1)
    q = rng.integers(config.q_range[0], config.q_range[1] + 1)
    P = rng.integers(config.P_range[0], config.P_range[1] + 1)
    Q = rng.integers(config.Q_range[0], config.Q_range[1] + 1)
    s = rng.integers(config.s_range[0], config.s_range[1] + 1)
    d = rng.uniform(config.d_range[0], config.d_range[1])

    # Stability mixture
    if rng.random() < config.stability_mixture_prob:
        if rng.random() < 0.5:
            p = 0
        else:
            P = 0

    # Sample poles and get coefficients
    ar_poles = sample_poles(p, config.r_max, rng)
    ar_coeffs_full = poles_to_coeffs(ar_poles)
    ar_coeffs = -ar_coeffs_full[1:] if len(ar_coeffs_full) > 1 else np.array([])

    ma_poles = sample_poles(q, config.r_max, rng)
    ma_coeffs_full = poles_to_coeffs(ma_poles)
    ma_coeffs = -ma_coeffs_full[1:] if len(ma_coeffs_full) > 1 else np.array([])

    sar_poles = sample_poles(P, config.R_max, rng)
    sar_coeffs_full = poles_to_coeffs(sar_poles)
    sar_coeffs = -sar_coeffs_full[1:] if len(sar_coeffs_full) > 1 else np.array([])

    sma_poles = sample_poles(Q, config.R_max, rng)
    sma_coeffs_full = poles_to_coeffs(sma_poles)
    sma_coeffs = -sma_coeffs_full[1:] if len(sma_coeffs_full) > 1 else np.array([])

    # Generate multivariate series
    y = generate_multivariate_sarima(
        batch_size,
        n_variates,
        length,
        ar_coeffs,
        ma_coeffs,
        sar_coeffs,
        sma_coeffs,
        s,
        d,
        config.D,
        corr_matrix,
        rng,
    )

    return torch.from_numpy(y).float().to(device), corr_matrix


def generate_multivariate_sarsim0_batch(
    batch_size: int,
    n_variates: int,
    length: int,
    config: SarSimConfig,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate batch of multivariate SarSim0 series (full pipeline).

    Applies SARIMA + optional SARIMA-2 + Noisers to each variate.

    Args:
        batch_size: Number of samples
        n_variates: Number of variates per sample
        length: Length including burn-in
        config: SarSim0 configuration
        generator: PyTorch generator
        device: Output device

    Returns:
        Tensor of shape (batch_size, n_variates, length - burn_in)
    """
    # Generate base multivariate SARIMA
    y, _ = generate_multivariate_sarima_batch(
        batch_size, n_variates, length, config, generator, device
    )

    # Apply noisers to each variate independently
    # Reshape to (batch * n_variates, length) for noiser
    y_flat = y.reshape(-1, y.shape[-1])

    # Apply noiser
    y_noised, _ = apply_noiser_vectorized(y_flat, config, generator)

    # Reshape back
    y = y_noised.reshape(batch_size, n_variates, -1)

    # Remove burn-in
    y = y[:, :, config.burn_in :]

    return y
