"""Tests for multivariate SARIMA generation."""

import torch
import numpy as np

from sarsim0.multivariate import (
    sample_correlation_matrix,
    generate_correlated_innovations,
    generate_multivariate_sarima_batch,
    generate_multivariate_sarsim0_batch,
)


class TestCorrelationMatrix:
    """Tests for correlation matrix sampling."""

    def test_shape(self, numpy_rng):
        """Test output shape."""
        for n in [2, 3, 5, 10]:
            corr = sample_correlation_matrix(n, (0.3, 0.9), numpy_rng)
            assert corr.shape == (n, n)

    def test_diagonal_ones(self, numpy_rng):
        """Test diagonal elements are 1."""
        corr = sample_correlation_matrix(5, (0.3, 0.9), numpy_rng)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_symmetric(self, numpy_rng):
        """Test matrix is symmetric."""
        corr = sample_correlation_matrix(5, (0.3, 0.9), numpy_rng)
        np.testing.assert_allclose(corr, corr.T)

    def test_positive_definite(self, numpy_rng):
        """Test matrix is positive definite."""
        corr = sample_correlation_matrix(5, (0.3, 0.9), numpy_rng)
        eigenvalues = np.linalg.eigvalsh(corr)
        assert np.all(eigenvalues > 0)

    def test_single_variate(self, numpy_rng):
        """Test single variate returns identity."""
        corr = sample_correlation_matrix(1, (0.3, 0.9), numpy_rng)
        np.testing.assert_allclose(corr, [[1.0]])

    def test_correlation_range(self, numpy_rng):
        """Test off-diagonal elements are in valid range."""
        corr = sample_correlation_matrix(5, (0.3, 0.9), numpy_rng)
        # All elements should be between -1 and 1
        assert np.all(np.abs(corr) <= 1.0 + 1e-10)


class TestCorrelatedInnovations:
    """Tests for correlated innovation generation."""

    def test_output_shape(self, numpy_rng):
        """Test output shape."""
        batch_size, n_variates, length = 16, 3, 100
        corr = sample_correlation_matrix(n_variates, (0.3, 0.9), numpy_rng)
        eps = generate_correlated_innovations(
            batch_size, n_variates, length, corr, numpy_rng
        )
        assert eps.shape == (batch_size, n_variates, length)

    def test_correlation_structure(self, numpy_rng):
        """Test that innovations have expected correlation structure."""
        n_variates = 3
        length = 10000  # Long for stable estimates
        corr = np.array(
            [
                [1.0, 0.5, 0.0],
                [0.5, 1.0, 0.3],
                [0.0, 0.3, 1.0],
            ]
        )
        eps = generate_correlated_innovations(1, n_variates, length, corr, numpy_rng)

        # Compute empirical correlation
        eps_2d = eps[0]  # (n_variates, length)
        empirical_corr = np.corrcoef(eps_2d)

        # Should be close to target (with some tolerance due to finite sample)
        np.testing.assert_allclose(empirical_corr, corr, atol=0.1)


class TestMultivariateSarimaBatch:
    """Tests for multivariate SARIMA batch generation."""

    def test_output_shape(self, config, generator):
        """Test output shape."""
        batch_size, n_variates, length = 8, 3, 500
        y, corr = generate_multivariate_sarima_batch(
            batch_size, n_variates, length, config, generator
        )
        assert y.shape == (batch_size, n_variates, length)
        assert corr.shape == (n_variates, n_variates)

    def test_no_nan_values(self, config, generator):
        """Test no NaN values."""
        y, _ = generate_multivariate_sarima_batch(16, 3, 1000, config, generator)
        assert not torch.isnan(y).any()

    def test_no_inf_values(self, config, generator):
        """Test no infinite values."""
        y, _ = generate_multivariate_sarima_batch(16, 3, 1000, config, generator)
        assert not torch.isinf(y).any()

    def test_values_bounded(self, config, generator):
        """Test values are within expected bounds."""
        y, _ = generate_multivariate_sarima_batch(16, 3, 1000, config, generator)
        assert y.abs().max() <= 1e6

    def test_variates_different(self, config, generator):
        """Test that different variates are not identical."""
        y, _ = generate_multivariate_sarima_batch(4, 3, 500, config, generator)
        # Variates should be different (correlated but not identical)
        assert not torch.allclose(y[0, 0], y[0, 1])

    def test_custom_correlation_matrix(self, config, generator):
        """Test with custom correlation matrix."""
        corr = np.array(
            [
                [1.0, 0.8],
                [0.8, 1.0],
            ]
        )
        y, returned_corr = generate_multivariate_sarima_batch(
            8, 2, 500, config, generator, corr_matrix=corr
        )
        np.testing.assert_allclose(returned_corr, corr)


class TestMultivariateSarsim0Batch:
    """Tests for full multivariate SarSim0 pipeline."""

    def test_output_shape(self, config, generator):
        """Test output shape includes burn-in removal."""
        batch_size, n_variates = 8, 3
        length = 1000
        y = generate_multivariate_sarsim0_batch(
            batch_size, n_variates, length, config, generator
        )
        expected_length = length - config.burn_in
        assert y.shape == (batch_size, n_variates, expected_length)

    def test_no_nan_values(self, config, generator):
        """Test no NaN values after full pipeline."""
        y = generate_multivariate_sarsim0_batch(16, 3, 1000, config, generator)
        assert not torch.isnan(y).any()

    def test_output_dtype(self, config, generator):
        """Test output is float tensor."""
        y = generate_multivariate_sarsim0_batch(8, 2, 500, config, generator)
        assert y.dtype == torch.float32


class TestGeneratorMultivariateMethod:
    """Tests for SarSim0Generator multivariate methods."""

    def test_generate_multivariate_series(self, small_config):
        """Test generate_multivariate_series method."""
        from sarsim0 import SarSim0Generator

        gen = SarSim0Generator(config=small_config, seed=42)
        y = gen.generate_multivariate_series(batch_size=4, n_variates=3, length=500)
        assert y.shape == (4, 3, 500)
        assert not torch.isnan(y).any()

    def test_generate_multivariate_random_variates(self, small_config):
        """Test with random number of variates."""
        from sarsim0 import SarSim0Generator

        gen = SarSim0Generator(config=small_config, seed=42)
        y = gen.generate_multivariate_series(batch_size=4, n_variates=None, length=500)
        assert y.ndim == 3
        assert y.shape[0] == 4
        assert y.shape[2] == 500


class TestMixedBatchGeneration:
    """Tests for mixed univariate/multivariate batch generation."""

    def test_mixed_batch_format(self, small_config):
        """Test mixed batch returns correct format."""
        from sarsim0 import SarSim0Generator

        small_config.multivariate_prob = 0.5
        gen = SarSim0Generator(config=small_config, seed=42)
        inputs = gen.generate_mixed_batch_chronos2(batch_size=20)

        assert len(inputs) == 20
        for inp in inputs:
            assert "target" in inp
            assert isinstance(inp["target"], np.ndarray)
            # Either 1D (univariate) or 2D (multivariate)
            assert inp["target"].ndim in [1, 2]

    def test_mixed_batch_has_both_types(self, small_config):
        """Test that mixed batch contains both univariate and multivariate."""
        from sarsim0 import SarSim0Generator

        small_config.multivariate_prob = 0.5
        gen = SarSim0Generator(config=small_config, seed=42)
        inputs = gen.generate_mixed_batch_chronos2(batch_size=100)

        univariate = sum(1 for inp in inputs if inp["target"].ndim == 1)
        multivariate = sum(1 for inp in inputs if inp["target"].ndim == 2)

        # With 50% probability, both should be well represented
        assert univariate > 0, "No univariate samples generated"
        assert multivariate > 0, "No multivariate samples generated"

    def test_mixed_batch_no_nan(self, small_config):
        """Test no NaN in mixed batch."""
        from sarsim0 import SarSim0Generator

        small_config.multivariate_prob = 0.5
        gen = SarSim0Generator(config=small_config, seed=42)
        inputs = gen.generate_mixed_batch_chronos2(batch_size=50)

        for inp in inputs:
            assert not np.isnan(inp["target"]).any()

    def test_convenience_function(self):
        """Test generate_mixed_sarsim0_chronos2 convenience function."""
        from sarsim0 import generate_mixed_sarsim0_chronos2

        inputs = generate_mixed_sarsim0_chronos2(
            num_series=20,
            context_length=256,
            prediction_length=32,
            seed=42,
        )

        assert len(inputs) == 20
        for inp in inputs:
            assert "target" in inp
            total_len = 256 + 32
            if inp["target"].ndim == 1:
                assert inp["target"].shape[0] == total_len
            else:
                assert inp["target"].shape[1] == total_len
