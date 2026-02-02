"""Tests for noiser modules."""

import torch

from sarsim0.noisers import (
    NoiserType,
    log_uniform_sample,
    compute_rate,
    poisson_noiser,
    gamma_noiser,
    lognormal_noiser,
    apply_noiser_vectorized,
)


class TestNoiserType:
    """Tests for NoiserType enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert NoiserType.POISSON.value == 0
        assert NoiserType.GAMMA.value == 1
        assert NoiserType.LOGNORMAL.value == 2
        assert NoiserType.PASSTHROUGH.value == 3


class TestLogUniformSample:
    """Tests for log-uniform sampling."""

    def test_output_shape(self, generator):
        """Test output has correct shape."""
        samples = log_uniform_sample(1.0, 100.0, (32,), generator)
        assert samples.shape == (32,)

    def test_output_in_range(self, generator):
        """Test samples are within specified range."""
        low, high = 1.0, 100.0
        samples = log_uniform_sample(low, high, (1000,), generator)
        assert samples.min() >= low
        assert samples.max() <= high

    def test_multidimensional_shape(self, generator):
        """Test multidimensional output."""
        samples = log_uniform_sample(0.1, 10.0, (8, 16), generator)
        assert samples.shape == (8, 16)


class TestComputeRate:
    """Tests for rate computation."""

    def test_output_shape(self):
        """Test output shape matches input."""
        y = torch.randn(16, 100)
        lambda_0 = torch.ones(16) * 10.0
        rate = compute_rate(y, lambda_0)
        assert rate.shape == y.shape

    def test_rate_positive(self):
        """Test all rates are positive."""
        y = torch.randn(16, 100)
        lambda_0 = torch.ones(16) * 10.0
        rate = compute_rate(y, lambda_0)
        assert (rate > 0).all()

    def test_rate_scaled_by_lambda(self):
        """Test rates scale with lambda_0."""
        y = torch.randn(8, 50)
        rate1 = compute_rate(y, torch.ones(8) * 10.0)
        rate2 = compute_rate(y, torch.ones(8) * 100.0)
        # Higher lambda_0 should give higher rates
        assert rate2.mean() > rate1.mean()

    def test_handles_nan_input(self):
        """Test handles NaN in input gracefully."""
        y = torch.randn(8, 50)
        y[0, 10] = float("nan")
        lambda_0 = torch.ones(8) * 10.0
        rate = compute_rate(y, lambda_0)
        # Should not propagate NaN
        assert not torch.isnan(rate).all()


class TestPoissonNoiser:
    """Tests for Poisson noiser."""

    def test_output_shape(self, config, generator):
        """Test output shape matches input."""
        y = torch.randn(16, 100)
        out = poisson_noiser(y, config, generator)
        assert out.shape == y.shape

    def test_output_non_negative(self, config, generator):
        """Test Poisson output is non-negative."""
        y = torch.randn(16, 100)
        out = poisson_noiser(y, config, generator)
        assert (out >= 0).all()

    def test_output_integer_values(self, config, generator):
        """Test Poisson output contains integer-like values."""
        y = torch.randn(16, 100)
        out = poisson_noiser(y, config, generator)
        # Poisson samples should be close to integers
        assert torch.allclose(out, out.round(), atol=1e-5)


class TestGammaNoiser:
    """Tests for Gamma noiser."""

    def test_output_shape(self, config, generator):
        """Test output shape matches input."""
        y = torch.randn(16, 100)
        out = gamma_noiser(y, config, generator)
        assert out.shape == y.shape

    def test_output_positive(self, config, generator):
        """Test Gamma output is positive."""
        y = torch.randn(16, 100)
        out = gamma_noiser(y, config, generator)
        assert (out > 0).all()

    def test_no_nan_output(self, config, generator):
        """Test no NaN in output."""
        y = torch.randn(32, 200)
        out = gamma_noiser(y, config, generator)
        assert not torch.isnan(out).any()


class TestLognormalNoiser:
    """Tests for Lognormal noiser."""

    def test_output_shape(self, config, generator):
        """Test output shape matches input."""
        y = torch.randn(16, 100)
        out = lognormal_noiser(y, config, generator)
        assert out.shape == y.shape

    def test_output_positive(self, config, generator):
        """Test Lognormal output is positive."""
        y = torch.randn(16, 100)
        out = lognormal_noiser(y, config, generator)
        assert (out > 0).all()

    def test_no_nan_output(self, config, generator):
        """Test no NaN in output."""
        y = torch.randn(32, 200)
        out = lognormal_noiser(y, config, generator)
        assert not torch.isnan(out).any()


class TestApplyNoiserVectorized:
    """Tests for apply_noiser_vectorized."""

    def test_output_shape(self, config, generator):
        """Test output shape matches input."""
        y = torch.randn(32, 100)
        out, types = apply_noiser_vectorized(y, config, generator)
        assert out.shape == y.shape
        assert types.shape == (32,)

    def test_types_in_range(self, config, generator):
        """Test noiser types are valid."""
        y = torch.randn(64, 100)
        _, types = apply_noiser_vectorized(y, config, generator)
        assert (types >= 0).all()
        assert (types <= 3).all()

    def test_passthrough_preserves_input(self, config, generator):
        """Test passthrough noiser preserves input."""
        y = torch.randn(8, 100)
        # Force all passthrough
        types = torch.full((8,), NoiserType.PASSTHROUGH.value)
        out, _ = apply_noiser_vectorized(y, config, generator, noiser_types=types)
        torch.testing.assert_close(out, y)

    def test_specified_types_used(self, config, generator):
        """Test specified noiser types are used."""
        y = torch.randn(8, 100)
        types = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        out, returned_types = apply_noiser_vectorized(
            y, config, generator, noiser_types=types
        )
        torch.testing.assert_close(returned_types, types)

    def test_no_nan_output(self, config, generator):
        """Test no NaN values in output."""
        y = torch.randn(64, 200)
        out, _ = apply_noiser_vectorized(y, config, generator)
        assert not torch.isnan(out).any()

    def test_no_generator_works(self, config):
        """Test function works without generator."""
        y = torch.randn(8, 100)
        out, types = apply_noiser_vectorized(y, config, generator=None)
        assert out.shape == y.shape
