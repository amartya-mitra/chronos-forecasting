"""Pytest configuration and fixtures for SarSim0 tests."""

import pytest
import torch
import numpy as np

from sarsim0 import SarSimConfig


@pytest.fixture
def config() -> SarSimConfig:
    """Default SarSim0 configuration."""
    return SarSimConfig()


@pytest.fixture
def small_config() -> SarSimConfig:
    """Smaller config for faster tests."""
    return SarSimConfig(
        series_length=1000,
        context_window=512,
        prediction_window=128,
        burn_in=100,
    )


@pytest.fixture
def generator() -> torch.Generator:
    """Seeded PyTorch generator for reproducibility."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    return gen


@pytest.fixture
def numpy_rng() -> np.random.Generator:
    """Seeded NumPy generator for reproducibility."""
    return np.random.default_rng(42)
