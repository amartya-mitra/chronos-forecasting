"""Tests for SARIMA-2 bi-seasonal composition module."""

import torch
import numpy as np

from sarsim0.sarima2 import (
    CompositionMode,
    normalize_to_range_batch,
    additive_compose_batch,
    multiplicative_compose_batch,
    apply_sarima2_vectorized,
    generate_sarima2_batch,
    normalize_to_range,
    additive_compose,
    multiplicative_compose,
)


class TestNormalization:
    """Tests for normalization functions."""

    def test_normalize_to_range_batch_shape(self):
        """Test output shape matches input."""
        x = np.random.randn(16, 100).astype(np.float64)
        out = normalize_to_range_batch(x, -1.0, 1.0)
        assert out.shape == x.shape

    def test_normalize_to_range_batch_bounds(self):
        """Test output is within target range."""
        x = np.random.randn(16, 100).astype(np.float64)
        out = normalize_to_range_batch(x, -1.0, 1.0)
        assert out.min() >= -1.0 - 1e-10
        assert out.max() <= 1.0 + 1e-10

    def test_normalize_to_range_batch_custom_bounds(self):
        """Test normalization to custom range."""
        x = np.random.randn(8, 50).astype(np.float64)
        out = normalize_to_range_batch(x, 0.0, 10.0)
        assert out.min() >= 0.0 - 1e-10
        assert out.max() <= 10.0 + 1e-10

    def test_normalize_constant_series(self):
        """Test normalization of constant series doesn't produce NaN."""
        x = np.ones((4, 100), dtype=np.float64) * 5.0
        out = normalize_to_range_batch(x, -1.0, 1.0)
        assert not np.isnan(out).any()

    def test_torch_normalize_to_range(self):
        """Test PyTorch normalization function."""
        x = torch.randn(8, 100)
        out = normalize_to_range(x, -1.0, 1.0)
        assert out.shape == x.shape
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5


class TestComposition:
    """Tests for composition functions."""

    def test_additive_compose_batch_shape(self):
        """Test additive composition output shape."""
        y_base = np.random.randn(16, 100).astype(np.float64)
        y_env = np.random.randn(16, 100).astype(np.float64)
        out = additive_compose_batch(y_base, y_env)
        assert out.shape == y_base.shape

    def test_additive_compose_batch_values(self):
        """Test additive composition is sum."""
        y_base = np.ones((4, 50), dtype=np.float64) * 2.0
        y_env = np.ones((4, 50), dtype=np.float64) * 3.0
        out = additive_compose_batch(y_base, y_env)
        np.testing.assert_allclose(out, 5.0)

    def test_multiplicative_compose_batch_shape(self):
        """Test multiplicative composition output shape."""
        y_base = np.random.randn(16, 100).astype(np.float64)
        y_env_norm = np.random.uniform(-1, 1, (16, 100)).astype(np.float64)
        omega = np.random.uniform(0, 1, 16).astype(np.float64)
        out = multiplicative_compose_batch(y_base, y_env_norm, omega)
        assert out.shape == y_base.shape

    def test_multiplicative_compose_omega_zero(self):
        """Test multiplicative with omega=0 returns base."""
        y_base = np.random.randn(8, 50).astype(np.float64)
        y_env_norm = np.random.uniform(-1, 1, (8, 50)).astype(np.float64)
        omega = np.zeros(8, dtype=np.float64)
        out = multiplicative_compose_batch(y_base, y_env_norm, omega)
        np.testing.assert_allclose(out, y_base)

    def test_torch_additive_compose(self):
        """Test PyTorch additive composition."""
        y_base = torch.randn(8, 100)
        y_env = torch.randn(8, 100)
        out = additive_compose(y_base, y_env)
        torch.testing.assert_close(out, y_base + y_env)

    def test_torch_multiplicative_compose(self):
        """Test PyTorch multiplicative composition."""
        y_base = torch.randn(8, 100)
        y_env = torch.randn(8, 100)
        omega = torch.zeros(8)
        out = multiplicative_compose(y_base, y_env, omega)
        # With omega=0, should return base
        torch.testing.assert_close(out, y_base)


class TestCompositionMode:
    """Tests for CompositionMode enum."""

    def test_enum_values(self):
        """Test enum has correct values."""
        assert CompositionMode.ADDITIVE.value == 0
        assert CompositionMode.MULTIPLICATIVE.value == 1


class TestApplySarima2:
    """Tests for apply_sarima2_vectorized."""

    def test_output_shape(self, config, generator):
        """Test output shape matches input."""
        y_base = torch.randn(32, 1000)
        y_out, modes = apply_sarima2_vectorized(y_base, config, generator)
        assert y_out.shape == y_base.shape
        assert modes.shape == (32,)

    def test_modes_binary(self, config, generator):
        """Test modes are 0 or 1."""
        y_base = torch.randn(64, 500)
        _, modes = apply_sarima2_vectorized(y_base, config, generator)
        assert ((modes == 0) | (modes == 1)).all()

    def test_no_nan_output(self, config, generator):
        """Test no NaN in output."""
        y_base = torch.randn(32, 1000)
        y_out, _ = apply_sarima2_vectorized(y_base, config, generator)
        assert not torch.isnan(y_out).any()


class TestGenerateSarima2Batch:
    """Tests for generate_sarima2_batch."""

    def test_output_shape(self, small_config, generator):
        """Test correct output shape."""
        batch_size, length = 16, 500
        y, _ = generate_sarima2_batch(batch_size, length, small_config, generator)
        assert y.shape == (batch_size, length)

    def test_output_dtype(self, small_config, generator):
        """Test output is float tensor."""
        y, _ = generate_sarima2_batch(8, 300, small_config, generator)
        assert y.dtype == torch.float32

    def test_no_nan_values(self, small_config, generator):
        """Test no NaN values in output."""
        y, _ = generate_sarima2_batch(16, 500, small_config, generator)
        assert not torch.isnan(y).any()

    def test_no_generator_provided(self, small_config):
        """Test function works without generator."""
        y, _ = generate_sarima2_batch(8, 300, small_config, generator=None)
        assert y.shape == (8, 300)
        assert not torch.isnan(y).any()
