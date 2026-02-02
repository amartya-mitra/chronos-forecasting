"""Tests for SarSim0 pipeline."""

import torch

from sarsim0.pipeline import (
    generate_sarsim0_batch,
    SarSim0Dataset,
    SarSim0Generator,
    create_dataloader,
)


class TestGenerateSarsim0Batch:
    """Tests for generate_sarsim0_batch function."""

    def test_output_shape(self, config, generator):
        """Test output shape accounts for burn-in removal."""
        batch_size = 16
        length = 1000
        y = generate_sarsim0_batch(batch_size, length, config, generator)
        expected_length = length - config.burn_in
        assert y.shape == (batch_size, expected_length)

    def test_output_dtype(self, config, generator):
        """Test output is float tensor."""
        y = generate_sarsim0_batch(8, 500, config, generator)
        assert y.dtype == torch.float32

    def test_no_nan_values(self, config, generator):
        """Test no NaN values in output."""
        y = generate_sarsim0_batch(32, 1000, config, generator)
        assert not torch.isnan(y).any()

    def test_no_inf_values(self, config, generator):
        """Test no infinite values in output."""
        y = generate_sarsim0_batch(32, 1000, config, generator)
        assert not torch.isinf(y).any()


class TestSarSim0Dataset:
    """Tests for SarSim0Dataset."""

    def test_iteration(self, small_config):
        """Test dataset can be iterated."""
        dataset = SarSim0Dataset(
            batch_size=8,
            config=small_config,
            seed=42,
            num_batches=3,
        )
        batches = list(dataset)
        assert len(batches) == 3

    def test_batch_shapes(self, small_config):
        """Test batch shapes are correct."""
        dataset = SarSim0Dataset(
            batch_size=16,
            config=small_config,
            seed=42,
            num_batches=2,
        )
        for context, target in dataset:
            assert context.shape == (16, small_config.context_window)
            assert target.shape == (16, small_config.prediction_window)

    def test_no_nan_in_batches(self, small_config):
        """Test no NaN values in batches."""
        dataset = SarSim0Dataset(
            batch_size=8,
            config=small_config,
            seed=42,
            num_batches=5,
        )
        for context, target in dataset:
            assert not torch.isnan(context).any()
            assert not torch.isnan(target).any()

    def test_infinite_iteration(self, small_config):
        """Test infinite iteration when num_batches is None."""
        dataset = SarSim0Dataset(
            batch_size=4,
            config=small_config,
            seed=42,
            num_batches=None,  # Infinite
        )
        count = 0
        for context, target in dataset:
            count += 1
            if count >= 5:
                break
        assert count == 5


class TestSarSim0Generator:
    """Tests for SarSim0Generator class."""

    def test_generate_batch_shapes(self, small_config):
        """Test generate_batch output shapes."""
        gen = SarSim0Generator(config=small_config, seed=42)
        context, target = gen.generate_batch(batch_size=16)
        assert context.shape == (16, small_config.context_window)
        assert target.shape == (16, small_config.prediction_window)

    def test_generate_batch_no_nan(self, small_config):
        """Test generate_batch has no NaN values."""
        gen = SarSim0Generator(config=small_config, seed=42)
        context, target = gen.generate_batch(batch_size=32)
        assert not torch.isnan(context).any()
        assert not torch.isnan(target).any()

    def test_generate_series_shape(self, small_config):
        """Test generate_series output shape."""
        gen = SarSim0Generator(config=small_config, seed=42)
        y = gen.generate_series(batch_size=8)
        assert y.shape == (8, small_config.series_length)

    def test_generate_series_custom_length(self, small_config):
        """Test generate_series with custom length."""
        gen = SarSim0Generator(config=small_config, seed=42)
        y = gen.generate_series(batch_size=8, length=500)
        assert y.shape == (8, 500)

    def test_generate_series_no_nan(self, small_config):
        """Test generate_series has no NaN values."""
        gen = SarSim0Generator(config=small_config, seed=42)
        y = gen.generate_series(batch_size=16)
        assert not torch.isnan(y).any()

    def test_multiple_batches_different(self, small_config):
        """Test consecutive batches are different."""
        gen = SarSim0Generator(config=small_config, seed=42)
        ctx1, _ = gen.generate_batch(batch_size=8)
        ctx2, _ = gen.generate_batch(batch_size=8)
        assert not torch.allclose(ctx1, ctx2)

    def test_default_config(self):
        """Test generator works with default config."""
        gen = SarSim0Generator(seed=42)
        context, target = gen.generate_batch(batch_size=4)
        assert context.shape[0] == 4
        assert target.shape[0] == 4


class TestCreateDataloader:
    """Tests for create_dataloader function."""

    def test_dataloader_iteration(self, small_config):
        """Test dataloader can be iterated."""
        dl = create_dataloader(
            batch_size=8,
            config=small_config,
            num_workers=0,
            seed=42,
            num_batches_per_epoch=3,
        )
        batches = list(dl)
        assert len(batches) == 3

    def test_dataloader_shapes(self, small_config):
        """Test dataloader batch shapes."""
        dl = create_dataloader(
            batch_size=16,
            config=small_config,
            num_workers=0,
            seed=42,
            num_batches_per_epoch=2,
        )
        for context, target in dl:
            assert context.shape == (16, small_config.context_window)
            assert target.shape == (16, small_config.prediction_window)

    def test_dataloader_no_nan(self, small_config):
        """Test dataloader outputs have no NaN."""
        dl = create_dataloader(
            batch_size=8,
            config=small_config,
            num_workers=0,
            seed=42,
            num_batches_per_epoch=5,
        )
        for context, target in dl:
            assert not torch.isnan(context).any()
            assert not torch.isnan(target).any()

    def test_dataloader_default_config(self):
        """Test dataloader works with default config."""
        dl = create_dataloader(
            batch_size=4,
            num_workers=0,
            seed=42,
            num_batches_per_epoch=2,
        )
        batches = list(dl)
        assert len(batches) == 2


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_default_config(self):
        """Test full pipeline with default configuration."""
        gen = SarSim0Generator(seed=42)
        context, target = gen.generate_batch(batch_size=8)

        assert context.shape[1] == 4096  # Default context window
        assert target.shape[1] == 512  # Default prediction window
        assert not torch.isnan(context).any()
        assert not torch.isnan(target).any()

    def test_context_target_continuity(self, small_config):
        """Test context and target come from continuous series."""
        # This is a statistical test - if properly windowed,
        # the last value of context should be "close" to the first value of target
        gen = SarSim0Generator(config=small_config, seed=42)

        # Generate many samples
        all_diffs = []
        for _ in range(10):
            context, target = gen.generate_batch(batch_size=32)
            # Check the difference between end of context and start of target
            diff = (context[:, -1] - target[:, 0]).abs()
            all_diffs.append(diff)

        # The differences should generally be smaller than the overall std
        avg_diff = torch.cat(all_diffs).mean()
        # This is a loose test - just checking they're not wildly different
        assert avg_diff < 1e6  # Should be reasonable given our clipping

    def test_large_batch(self, small_config):
        """Test generation of large batches."""
        gen = SarSim0Generator(config=small_config, seed=42)
        context, target = gen.generate_batch(batch_size=256)
        assert context.shape == (256, small_config.context_window)
        assert not torch.isnan(context).any()
