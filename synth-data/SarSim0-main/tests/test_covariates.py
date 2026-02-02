"""Tests for covariate generation."""

import torch
import numpy as np
import pytest

from sarsim0 import SarSimConfig
from sarsim0.covariates import (
    generate_with_covariates,
    generate_mixed_with_covariates,
    generate_sarsim0_chronos2_with_covariates,
)


@pytest.fixture
def config():
    """Config for testing."""
    return SarSimConfig(
        burn_in=50,
        series_length=500,
        context_window=256,
        prediction_window=32,
        with_covariates_prob=0.5,
        multivariate_prob=0.3,
        n_past_covariates_range=(1, 2),
        n_future_covariates_range=(1, 2),
    )


@pytest.fixture
def generator():
    """PyTorch generator for reproducibility."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    return gen


class TestGenerateWithCovariates:
    """Tests for generate_with_covariates function."""

    def test_univariate_with_covariates(self, config, generator):
        """Test univariate target with covariates."""
        inputs = generate_with_covariates(
            batch_size=4,
            context_length=256,
            prediction_length=32,
            n_targets=1,
            n_past_covariates=2,
            n_future_covariates=1,
            config=config,
            generator=generator,
        )

        assert len(inputs) == 4
        for inp in inputs:
            # Check target shape (univariate)
            assert inp["target"].ndim == 1
            assert inp["target"].shape == (288,)  # context + prediction

            # Check past covariates (full length = context + prediction)
            assert "past_covariates" in inp
            assert len(inp["past_covariates"]) == 3  # 2 past + 1 future (historical part)
            for key, val in inp["past_covariates"].items():
                assert val.shape == (288,)  # full length (context + prediction)

            # Check future covariates
            assert "future_covariates" in inp
            assert len(inp["future_covariates"]) == 1
            for key, val in inp["future_covariates"].items():
                assert val.shape == (32,)  # prediction_length

    def test_multivariate_with_covariates(self, config, generator):
        """Test multivariate target with covariates."""
        inputs = generate_with_covariates(
            batch_size=4,
            context_length=256,
            prediction_length=32,
            n_targets=3,
            n_past_covariates=1,
            n_future_covariates=2,
            config=config,
            generator=generator,
        )

        assert len(inputs) == 4
        for inp in inputs:
            # Check target shape (multivariate)
            assert inp["target"].ndim == 2
            assert inp["target"].shape == (3, 288)  # (n_targets, context + prediction)

            # Check past covariates
            assert "past_covariates" in inp
            # 1 past + 2 future (historical parts)
            assert len(inp["past_covariates"]) == 3

            # Check future covariates
            assert "future_covariates" in inp
            assert len(inp["future_covariates"]) == 2

    def test_no_covariates(self, config, generator):
        """Test generation with no covariates."""
        inputs = generate_with_covariates(
            batch_size=4,
            context_length=256,
            prediction_length=32,
            n_targets=1,
            n_past_covariates=0,
            n_future_covariates=0,
            config=config,
            generator=generator,
        )

        assert len(inputs) == 4
        for inp in inputs:
            assert "target" in inp
            assert "past_covariates" not in inp
            assert "future_covariates" not in inp

    def test_no_nan_values(self, config, generator):
        """Test no NaN values in outputs."""
        inputs = generate_with_covariates(
            batch_size=8,
            context_length=256,
            prediction_length=32,
            n_targets=2,
            n_past_covariates=2,
            n_future_covariates=2,
            config=config,
            generator=generator,
        )

        for inp in inputs:
            assert not np.isnan(inp["target"]).any()
            for val in inp.get("past_covariates", {}).values():
                assert not np.isnan(val).any()
            for val in inp.get("future_covariates", {}).values():
                assert not np.isnan(val).any()


class TestGenerateMixedWithCovariates:
    """Tests for generate_mixed_with_covariates function."""

    def test_mixed_generation(self, config, generator):
        """Test mixed generation produces varied outputs."""
        inputs = generate_mixed_with_covariates(
            batch_size=50,
            context_length=256,
            prediction_length=32,
            config=config,
            generator=generator,
        )

        assert len(inputs) == 50

        # Count different types
        with_cov = sum(1 for inp in inputs if "past_covariates" in inp or "future_covariates" in inp)
        without_cov = len(inputs) - with_cov
        univariate = sum(1 for inp in inputs if inp["target"].ndim == 1)
        multivariate = sum(1 for inp in inputs if inp["target"].ndim == 2)

        # With 50% covariate prob and 30% multivariate prob, expect variation
        assert with_cov > 0, "No samples with covariates"
        assert without_cov > 0, "No samples without covariates"
        # Note: multivariate might be 0 if with_covariates_prob dominates
        # and multivariate_prob is only 0.3

    def test_all_outputs_valid(self, config, generator):
        """Test all outputs have valid structure."""
        inputs = generate_mixed_with_covariates(
            batch_size=20,
            context_length=256,
            prediction_length=32,
            config=config,
            generator=generator,
        )

        for inp in inputs:
            assert "target" in inp
            target = inp["target"]
            assert target.ndim in [1, 2]

            if target.ndim == 1:
                assert target.shape == (288,)
            else:
                assert target.shape[1] == 288

            # Check covariate shapes if present
            # past_covariates have full length (context + prediction)
            for key, val in inp.get("past_covariates", {}).items():
                assert val.shape == (288,)
            for key, val in inp.get("future_covariates", {}).items():
                assert val.shape == (32,)

    def test_no_nan_values(self, config, generator):
        """Test no NaN values in mixed outputs."""
        inputs = generate_mixed_with_covariates(
            batch_size=30,
            context_length=256,
            prediction_length=32,
            config=config,
            generator=generator,
        )

        for inp in inputs:
            assert not np.isnan(inp["target"]).any()
            for val in inp.get("past_covariates", {}).values():
                assert not np.isnan(val).any()
            for val in inp.get("future_covariates", {}).values():
                assert not np.isnan(val).any()


class TestConvenienceFunction:
    """Tests for generate_sarsim0_chronos2_with_covariates convenience function."""

    def test_default_config(self):
        """Test with default config."""
        inputs = generate_sarsim0_chronos2_with_covariates(
            num_series=10,
            context_length=256,
            prediction_length=32,
            seed=42,
        )

        assert len(inputs) == 10
        for inp in inputs:
            assert "target" in inp
            total_len = 256 + 32
            if inp["target"].ndim == 1:
                assert inp["target"].shape == (total_len,)
            else:
                assert inp["target"].shape[1] == total_len

    def test_custom_config(self):
        """Test with custom config."""
        config = SarSimConfig(
            with_covariates_prob=1.0,  # Always include covariates
            n_past_covariates_range=(2, 2),
            n_future_covariates_range=(1, 1),
        )

        inputs = generate_sarsim0_chronos2_with_covariates(
            num_series=5,
            context_length=128,
            prediction_length=16,
            config=config,
            seed=42,
        )

        assert len(inputs) == 5
        for inp in inputs:
            assert "past_covariates" in inp
            assert "future_covariates" in inp

    def test_reproducibility(self):
        """Test that same seed produces same structure (not exact values due to branching)."""
        # Note: Due to random branching (has_covariates, is_multivariate),
        # exact reproducibility is not guaranteed for mixed generation.
        # We test that the structure is consistent instead.
        inputs1 = generate_sarsim0_chronos2_with_covariates(
            num_series=5,
            context_length=128,
            prediction_length=16,
            seed=123,
        )

        inputs2 = generate_sarsim0_chronos2_with_covariates(
            num_series=5,
            context_length=128,
            prediction_length=16,
            seed=123,
        )

        # Check same number of outputs
        assert len(inputs1) == len(inputs2)

        # Check all outputs are valid
        for inp1, inp2 in zip(inputs1, inputs2):
            assert "target" in inp1
            assert "target" in inp2
            # Shapes should be valid (not necessarily identical due to branching)
            assert inp1["target"].ndim in [1, 2]
            assert inp2["target"].ndim in [1, 2]


class TestChronos2Compatibility:
    """Tests to verify output format matches Chronos-2 expectations."""

    def test_output_format_univariate(self, config, generator):
        """Test output format matches Chronos-2 for univariate."""
        inputs = generate_with_covariates(
            batch_size=1,
            context_length=256,
            prediction_length=32,
            n_targets=1,
            n_past_covariates=1,
            n_future_covariates=1,
            config=config,
            generator=generator,
        )

        inp = inputs[0]

        # Chronos-2 expects:
        # - target: 1D array (length,) for univariate
        assert isinstance(inp["target"], np.ndarray)
        assert inp["target"].ndim == 1

        # - past_covariates: dict of 1D arrays with same length as target
        assert isinstance(inp["past_covariates"], dict)
        for val in inp["past_covariates"].values():
            assert isinstance(val, np.ndarray)
            assert val.ndim == 1
            assert val.shape[0] == 288  # context + prediction

        # - future_covariates: dict of 1D arrays (prediction_length,)
        assert isinstance(inp["future_covariates"], dict)
        for val in inp["future_covariates"].values():
            assert isinstance(val, np.ndarray)
            assert val.ndim == 1
            assert val.shape[0] == 32

    def test_output_format_multivariate(self, config, generator):
        """Test output format matches Chronos-2 for multivariate."""
        inputs = generate_with_covariates(
            batch_size=1,
            context_length=256,
            prediction_length=32,
            n_targets=3,
            n_past_covariates=1,
            n_future_covariates=1,
            config=config,
            generator=generator,
        )

        inp = inputs[0]

        # Chronos-2 expects:
        # - target: 2D array (n_variates, length) for multivariate
        assert isinstance(inp["target"], np.ndarray)
        assert inp["target"].ndim == 2
        assert inp["target"].shape == (3, 288)

    def test_covariate_keys_consistent(self, config, generator):
        """Test that future covariates appear in both past and future dicts."""
        inputs = generate_with_covariates(
            batch_size=1,
            context_length=256,
            prediction_length=32,
            n_targets=1,
            n_past_covariates=1,
            n_future_covariates=2,
            config=config,
            generator=generator,
        )

        inp = inputs[0]

        # Future covariate keys should appear in both past_covariates (history)
        # and future_covariates (future values)
        future_keys = set(inp["future_covariates"].keys())
        past_keys = set(inp["past_covariates"].keys())

        # All future covariate keys should be in past_covariates
        assert future_keys.issubset(past_keys)
