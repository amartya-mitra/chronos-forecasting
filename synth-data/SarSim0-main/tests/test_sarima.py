"""Tests for SARIMA generation module."""

import torch
import numpy as np

from sarsim0.sarima import (
    sample_poles,
    poles_to_coeffs,
    expand_seasonal_poly,
    build_sarima_polynomials,
    fractional_diff_coeffs,
    generate_sarima_batch,
)


class TestPoles:
    """Tests for pole sampling and conversion."""

    def test_sample_poles_empty(self, numpy_rng):
        """Test sampling zero poles."""
        poles = sample_poles(0, 0.9, numpy_rng)
        assert len(poles) == 0

    def test_sample_poles_count(self, numpy_rng):
        """Test correct number of poles sampled."""
        for n in [1, 5, 10]:
            poles = sample_poles(n, 0.9, numpy_rng)
            assert len(poles) == n

    def test_sample_poles_within_unit_circle(self, numpy_rng):
        """Test all poles are within specified radius."""
        r_max = 0.9
        poles = sample_poles(100, r_max, numpy_rng)
        magnitudes = np.abs(poles)
        assert np.all(magnitudes <= r_max)

    def test_poles_to_coeffs_empty(self):
        """Test conversion with no poles."""
        coeffs = poles_to_coeffs(np.array([]))
        assert len(coeffs) == 1
        assert coeffs[0] == 1.0

    def test_poles_to_coeffs_single_real(self):
        """Test conversion with single real pole."""
        pole = np.array([0.5])
        coeffs = poles_to_coeffs(pole)
        # (x - 0.5) = x - 0.5, so coeffs = [1, -0.5]
        assert len(coeffs) == 2
        np.testing.assert_allclose(coeffs, [1.0, -0.5])

    def test_poles_to_coeffs_conjugate_pair(self):
        """Test conversion with complex conjugate pair."""
        # Poles at 0.5 * exp(±i*pi/4)
        r, theta = 0.5, np.pi / 4
        poles = np.array([r * np.exp(1j * theta), r * np.exp(-1j * theta)])
        coeffs = poles_to_coeffs(poles)
        # Should produce real coefficients
        assert len(coeffs) == 3
        assert np.allclose(coeffs.imag, 0)


class TestSeasonalExpansion:
    """Tests for seasonal polynomial expansion."""

    def test_expand_no_season(self):
        """Test expansion with s=0 or s=1."""
        coeffs = np.array([1.0, -0.5])
        assert np.array_equal(expand_seasonal_poly(coeffs, 0), coeffs)
        assert np.array_equal(expand_seasonal_poly(coeffs, 1), coeffs)

    def test_expand_seasonal(self):
        """Test expansion with seasonal period."""
        coeffs = np.array([1.0, -0.5])
        expanded = expand_seasonal_poly(coeffs, 4)
        # [1, 0, 0, 0, -0.5]
        expected = np.array([1.0, 0.0, 0.0, 0.0, -0.5])
        np.testing.assert_allclose(expanded, expected)

    def test_expand_seasonal_multiple_coeffs(self):
        """Test expansion with multiple seasonal coefficients."""
        coeffs = np.array([1.0, -0.3, -0.2])
        expanded = expand_seasonal_poly(coeffs, 3)
        # [1, 0, 0, -0.3, 0, 0, -0.2]
        expected = np.array([1.0, 0.0, 0.0, -0.3, 0.0, 0.0, -0.2])
        np.testing.assert_allclose(expanded, expected)


class TestFractionalDiff:
    """Tests for fractional differencing coefficients."""

    def test_d_zero(self):
        """Test d=0 returns [1]."""
        coeffs = fractional_diff_coeffs(0.0)
        assert len(coeffs) == 1
        assert coeffs[0] == 1.0

    def test_coeffs_decay(self):
        """Test coefficients decay for d > 0."""
        coeffs = fractional_diff_coeffs(0.5)
        assert len(coeffs) > 1
        # Coefficients should generally decrease in magnitude
        assert abs(coeffs[-1]) < abs(coeffs[0])

    def test_negative_d_integration(self):
        """Test negative d for integration."""
        coeffs = fractional_diff_coeffs(-0.5)
        assert len(coeffs) > 1
        # First coeff is always 1
        assert coeffs[0] == 1.0


class TestBuildPolynomials:
    """Tests for building SARIMA polynomials."""

    def test_empty_coefficients(self):
        """Test with all empty coefficients."""
        ar_poly, ma_poly = build_sarima_polynomials(
            np.array([]), np.array([]), np.array([]), np.array([]), 0
        )
        np.testing.assert_allclose(ar_poly, [1.0])
        np.testing.assert_allclose(ma_poly, [1.0])

    def test_ar_only(self):
        """Test with only AR coefficients."""
        ar_coeffs = np.array([0.5, 0.3])
        ar_poly, ma_poly = build_sarima_polynomials(
            ar_coeffs, np.array([]), np.array([]), np.array([]), 0
        )
        # AR poly: 1 - 0.5*L - 0.3*L^2 = [1, -0.5, -0.3]
        np.testing.assert_allclose(ar_poly, [1.0, -0.5, -0.3])
        np.testing.assert_allclose(ma_poly, [1.0])

    def test_ma_only(self):
        """Test with only MA coefficients."""
        ma_coeffs = np.array([0.4, 0.2])
        ar_poly, ma_poly = build_sarima_polynomials(
            np.array([]), ma_coeffs, np.array([]), np.array([]), 0
        )
        np.testing.assert_allclose(ar_poly, [1.0])
        # MA poly: 1 + 0.4*L + 0.2*L^2 = [1, 0.4, 0.2]
        np.testing.assert_allclose(ma_poly, [1.0, 0.4, 0.2])


class TestGenerateSarimaBatch:
    """Tests for batch SARIMA generation."""

    def test_output_shape(self, config, generator):
        """Test correct output shape."""
        batch_size, length = 32, 1000
        y, _ = generate_sarima_batch(batch_size, length, config, generator)
        assert y.shape == (batch_size, length)

    def test_output_dtype(self, config, generator):
        """Test output is float tensor."""
        y, _ = generate_sarima_batch(16, 500, config, generator)
        assert y.dtype == torch.float32

    def test_no_nan_values(self, config, generator):
        """Test no NaN values in output."""
        y, _ = generate_sarima_batch(64, 1000, config, generator)
        assert not torch.isnan(y).any()

    def test_no_inf_values(self, config, generator):
        """Test no infinite values in output."""
        y, _ = generate_sarima_batch(64, 1000, config, generator)
        assert not torch.isinf(y).any()

    def test_values_bounded(self, config, generator):
        """Test values are within expected bounds."""
        y, _ = generate_sarima_batch(64, 1000, config, generator)
        assert y.abs().max() <= 1e6

    def test_different_seeds_different_output(self, config):
        """Test different seeds produce different outputs."""
        gen1 = torch.Generator().manual_seed(1)
        gen2 = torch.Generator().manual_seed(2)
        y1, _ = generate_sarima_batch(8, 100, config, gen1)
        y2, _ = generate_sarima_batch(8, 100, config, gen2)
        assert not torch.allclose(y1, y2)

    def test_batch_variation(self, config, generator):
        """Test series within batch are different."""
        y, _ = generate_sarima_batch(8, 500, config, generator)
        # Check that not all series are identical
        assert not torch.allclose(y[0], y[1])
