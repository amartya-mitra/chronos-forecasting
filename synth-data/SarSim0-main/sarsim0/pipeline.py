"""
SarSim0 Pipeline - Full composition with PyTorch DataLoader.

Implements the full simulator pipeline: y₁:T = N ∘ I ∘ S(ε)

Provides:
- On-the-fly data generation for training
- PyTorch Dataset and DataLoader integration
- Fully vectorized batch generation
- Mixed univariate/multivariate generation for Chronos-2
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Optional, Tuple, List, Dict
import numpy as np

from .config import SarSimConfig
from .sarima import generate_sarima_batch
from .sarima2 import apply_sarima2_vectorized
from .noisers import apply_noiser_vectorized
from .multivariate import generate_multivariate_sarsim0_batch


def generate_sarsim0_batch(
    batch_size: int,
    length: int,
    config: SarSimConfig,
    generator: torch.Generator,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate a batch of SarSim0 series (vectorized).

    Pipeline: y = N ∘ I ∘ S(ε)

    Args:
        batch_size: Number of series
        length: Length including burn-in
        config: Configuration
        generator: PyTorch random generator
        device: Device for tensors

    Returns:
        Tensor of shape (batch_size, length - burn_in)
    """
    # Step 1: Generate SARIMA base signals
    y, _ = generate_sarima_batch(batch_size, length, config, generator, device)

    # Step 2: Optionally apply SARIMA-2 interaction
    use_sarima2 = (
        torch.rand(batch_size, generator=generator, device=device) < config.sarima2_prob
    )

    if use_sarima2.any():
        y_sarima2, _ = apply_sarima2_vectorized(
            y[use_sarima2], config, generator, device
        )
        y[use_sarima2] = y_sarima2

    # Step 3: Apply Noiser
    y, _ = apply_noiser_vectorized(y, config, generator)

    # Step 4: Remove burn-in
    y = y[:, config.burn_in :]

    return y


class SarSim0Dataset(IterableDataset):
    """
    Iterable PyTorch Dataset for on-the-fly SarSim0 generation.

    Generates batches of (context, target) pairs for training.
    Uses vectorized batch generation for efficiency.
    """

    def __init__(
        self,
        batch_size: int,
        config: Optional[SarSimConfig] = None,
        seed: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        num_batches: Optional[int] = None,
    ):
        """
        Initialize the dataset.

        Args:
            batch_size: Samples per batch
            config: SarSim0 configuration
            seed: Base random seed
            device: Device for tensor generation
            num_batches: Batches per epoch (None = infinite)
        """
        super().__init__()
        self.batch_size = batch_size
        self.config = config or SarSimConfig()
        self.seed = seed
        self.device = device
        self.num_batches = num_batches

    def __iter__(self):
        """Yield (context, target) batches."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id if self.seed else worker_info.id
        else:
            worker_seed = self.seed

        generator = torch.Generator(device="cpu")
        if worker_seed is not None:
            generator.manual_seed(worker_seed)

        total_length = self.config.series_length + self.config.burn_in
        batch_count = 0

        while self.num_batches is None or batch_count < self.num_batches:
            # Generate batch of full series
            y = generate_sarsim0_batch(
                self.batch_size,
                total_length,
                self.config,
                generator,
                self.device,
            )

            # Random window selection per series
            max_start = (
                self.config.series_length
                - self.config.context_window
                - self.config.prediction_window
            )
            starts = torch.randint(
                0, max(1, max_start + 1), (self.batch_size,), generator=generator
            )

            context = torch.zeros(
                self.batch_size,
                self.config.context_window,
                dtype=torch.float32,
                device=self.device,
            )
            target = torch.zeros(
                self.batch_size,
                self.config.prediction_window,
                dtype=torch.float32,
                device=self.device,
            )

            for i in range(self.batch_size):
                s = starts[i].item()
                context[i] = y[i, s : s + self.config.context_window]
                target[i] = y[
                    i,
                    s + self.config.context_window : s
                    + self.config.context_window
                    + self.config.prediction_window,
                ]

            yield context, target
            batch_count += 1


def create_dataloader(
    batch_size: int,
    config: Optional[SarSimConfig] = None,
    num_workers: int = 0,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    num_batches_per_epoch: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for SarSim0 training.

    Args:
        batch_size: Samples per batch
        config: SarSim0 configuration
        num_workers: Number of worker processes
        seed: Random seed for reproducibility
        device: Device for tensor generation
        num_batches_per_epoch: Batches per epoch (None = infinite)

    Returns:
        DataLoader yielding (context, target) batches
    """
    dataset = SarSim0Dataset(
        batch_size=batch_size,
        config=config,
        seed=seed,
        device=device,
        num_batches=num_batches_per_epoch,
    )

    # batch_size=1 since dataset yields full batches
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda x: (x[0][0], x[0][1]),
    )


class SarSim0Generator:
    """
    Stateful generator for on-the-fly data generation.

    Maintains internal random state for reproducibility.
    """

    def __init__(
        self,
        config: Optional[SarSimConfig] = None,
        seed: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize generator.

        Args:
            config: Configuration
            seed: Random seed
            device: Device for tensors
        """
        self.config = config or SarSimConfig()
        self.device = device
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)

    def generate_batch(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a training batch.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (context, target) tensors
        """
        total_length = self.config.series_length + self.config.burn_in

        # Generate series
        y = generate_sarsim0_batch(
            batch_size,
            total_length,
            self.config,
            self.generator,
            self.device,
        )

        # Window selection
        max_start = (
            self.config.series_length
            - self.config.context_window
            - self.config.prediction_window
        )
        starts = torch.randint(
            0, max(1, max_start + 1), (batch_size,), generator=self.generator
        )

        context = torch.zeros(
            batch_size,
            self.config.context_window,
            dtype=torch.float32,
            device=self.device,
        )
        target = torch.zeros(
            batch_size,
            self.config.prediction_window,
            dtype=torch.float32,
            device=self.device,
        )

        for i in range(batch_size):
            s = starts[i].item()
            context[i] = y[i, s : s + self.config.context_window]
            target[i] = y[
                i,
                s + self.config.context_window : s
                + self.config.context_window
                + self.config.prediction_window,
            ]

        return context, target

    def generate_series(
        self,
        batch_size: int,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate raw series without windowing.

        Args:
            batch_size: Number of series
            length: Series length (default: config.series_length)

        Returns:
            Tensor of shape (batch_size, length)
        """
        length = length or self.config.series_length
        total_length = length + self.config.burn_in

        return generate_sarsim0_batch(
            batch_size,
            total_length,
            self.config,
            self.generator,
            self.device,
        )

    def generate_multivariate_series(
        self,
        batch_size: int,
        n_variates: Optional[int] = None,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate multivariate series.

        Args:
            batch_size: Number of samples
            n_variates: Number of variates (random from config if None)
            length: Series length (default: config.series_length)

        Returns:
            Tensor of shape (batch_size, n_variates, length)
        """
        length = length or self.config.series_length
        total_length = length + self.config.burn_in

        if n_variates is None:
            seed = int(torch.randint(0, 2**31, (1,), generator=self.generator).item())
            rng = np.random.default_rng(seed)
            n_variates = rng.integers(
                self.config.n_variates_range[0], self.config.n_variates_range[1] + 1
            )

        return generate_multivariate_sarsim0_batch(
            batch_size,
            n_variates,
            total_length,
            self.config,
            self.generator,
            self.device,
        )

    def generate_mixed_batch_chronos2(
        self,
        batch_size: int,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate a mixed batch of univariate and multivariate series.

        Output format compatible with Chronos-2 fit() method.

        Args:
            batch_size: Number of samples

        Returns:
            List of dicts with "target" key, each target is either:
            - 1D array (univariate): shape (context_length + prediction_length,)
            - 2D array (multivariate): shape (n_variates, context_length + prediction_length,)
        """
        total_length = self.config.context_window + self.config.prediction_window
        full_length = total_length + self.config.burn_in

        inputs = []

        for _ in range(batch_size):
            # Decide univariate vs multivariate
            is_multivariate = (
                torch.rand(1, generator=self.generator).item()
                < self.config.multivariate_prob
            )

            if is_multivariate:
                # Sample number of variates
                seed = int(
                    torch.randint(0, 2**31, (1,), generator=self.generator).item()
                )
                rng = np.random.default_rng(seed)
                n_variates = rng.integers(
                    self.config.n_variates_range[0],
                    self.config.n_variates_range[1] + 1,
                )

                # Generate multivariate
                y = generate_multivariate_sarsim0_batch(
                    batch_size=1,
                    n_variates=n_variates,
                    length=full_length,
                    config=self.config,
                    generator=self.generator,
                    device=self.device,
                )
                # Shape: (1, n_variates, series_length) -> (n_variates, total_length)
                series_length = y.shape[-1]
                max_start = series_length - total_length
                start = (
                    int(
                        torch.randint(
                            0, max(1, max_start + 1), (1,), generator=self.generator
                        ).item()
                    )
                    if max_start > 0
                    else 0
                )
                target = y[0, :, start : start + total_length].numpy()
            else:
                # Generate univariate
                y = generate_sarsim0_batch(
                    batch_size=1,
                    length=full_length,
                    config=self.config,
                    generator=self.generator,
                    device=self.device,
                )
                # Shape: (1, series_length) -> (total_length,)
                series_length = y.shape[-1]
                max_start = series_length - total_length
                start = (
                    int(
                        torch.randint(
                            0, max(1, max_start + 1), (1,), generator=self.generator
                        ).item()
                    )
                    if max_start > 0
                    else 0
                )
                target = y[0, start : start + total_length].numpy()

            inputs.append({"target": target})

        return inputs


def generate_mixed_sarsim0_chronos2(
    num_series: int,
    context_length: int,
    prediction_length: int,
    config: Optional[SarSimConfig] = None,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate mixed univariate/multivariate data for Chronos-2 training.

    Convenience function that creates a generator and produces training data.

    Args:
        num_series: Number of series to generate
        context_length: Context window length
        prediction_length: Prediction horizon
        config: SarSim0 configuration (uses defaults if None)
        seed: Random seed

    Returns:
        List of dicts with "target" arrays, compatible with Chronos-2 fit()
    """
    if config is None:
        config = SarSimConfig()

    # Override window sizes
    config.context_window = context_length
    config.prediction_window = prediction_length

    generator = SarSim0Generator(config=config, seed=seed)
    return generator.generate_mixed_batch_chronos2(num_series)
