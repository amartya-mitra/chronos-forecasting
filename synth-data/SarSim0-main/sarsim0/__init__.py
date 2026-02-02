"""
SarSim0 - SARIMA Simulator for Zero-Shot Forecasting

A PyTorch-based synthetic time series generator implementing the SarSim0 pipeline
as described in the paper. Generates diverse, realistic time series for training
foundation models.

Pipeline: y = N ∘ I ∘ S(ε)
- S: SARIMA base signal generator with pole-based stability
- I: SARIMA-2 bi-seasonal interaction layer
- N: Heavy-tailed noiser (Poisson, Gamma, Lognormal)

Example usage:
    from sarsim0 import SarSimConfig, create_dataloader, SarSim0Generator

    # Create dataloader for training
    config = SarSimConfig()
    dataloader = create_dataloader(batch_size=256, config=config, num_workers=4)

    for context, target in dataloader:
        # context: (batch_size, 4096)
        # target: (batch_size, 512)
        ...

    # Or use generator directly
    generator = SarSim0Generator(config=config, seed=42)
    context, target = generator.generate_batch(batch_size=256)
"""

from .config import SarSimConfig
from .sarima import generate_sarima_batch
from .sarima2 import (
    CompositionMode,
    apply_sarima2_vectorized,
    additive_compose,
    multiplicative_compose,
)
from .noisers import (
    NoiserType,
    apply_noiser_vectorized,
    poisson_noiser,
    gamma_noiser,
    lognormal_noiser,
)
from .pipeline import (
    generate_sarsim0_batch,
    SarSim0Dataset,
    SarSim0Generator,
    create_dataloader,
    generate_mixed_sarsim0_chronos2,
)
from .multivariate import (
    generate_multivariate_sarima_batch,
    generate_multivariate_sarsim0_batch,
    sample_correlation_matrix,
)
from .covariates import (
    generate_with_covariates,
    generate_mixed_with_covariates,
    generate_sarsim0_chronos2_with_covariates,
)

__version__ = "0.3.0"

__all__ = [
    # Config
    "SarSimConfig",
    # SARIMA
    "generate_sarima_batch",
    # SARIMA-2
    "CompositionMode",
    "apply_sarima2_vectorized",
    "additive_compose",
    "multiplicative_compose",
    # Noisers
    "NoiserType",
    "apply_noiser_vectorized",
    "poisson_noiser",
    "gamma_noiser",
    "lognormal_noiser",
    # Pipeline
    "generate_sarsim0_batch",
    "SarSim0Dataset",
    "SarSim0Generator",
    "create_dataloader",
    "generate_mixed_sarsim0_chronos2",
    # Multivariate
    "generate_multivariate_sarima_batch",
    "generate_multivariate_sarsim0_batch",
    "sample_correlation_matrix",
    # Covariates
    "generate_with_covariates",
    "generate_mixed_with_covariates",
    "generate_sarsim0_chronos2_with_covariates",
]
