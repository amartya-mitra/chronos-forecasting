#!/usr/bin/env python3
"""
Evaluate fine-tuned Chronos-2 model on synthetic and real datasets.

Usage:
    python eval_chronos2.py --model_path ./chronos2-sarsim0-finetuned
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import Optional

from chronos import BaseChronosPipeline, Chronos2Pipeline

from sarsim0 import SarSimConfig, SarSim0Generator


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantile_predictions: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute forecasting metrics.

    Args:
        predictions: Point predictions (batch, horizon)
        targets: Ground truth (batch, horizon)
        quantile_predictions: Quantile predictions (batch, horizon, num_quantiles)

    Returns:
        Dict of metrics
    """
    # MAE
    mae = np.abs(predictions - targets).mean()

    # MSE
    mse = ((predictions - targets) ** 2).mean()

    # RMSE
    rmse = np.sqrt(mse)

    # MAPE (with epsilon for stability)
    eps = 1e-8
    mape = (np.abs(predictions - targets) / (np.abs(targets) + eps)).mean() * 100

    # sMAPE
    smape = (
        2
        * np.abs(predictions - targets)
        / (np.abs(predictions) + np.abs(targets) + eps)
    ).mean() * 100

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
    }

    # CRPS if quantiles provided
    if quantile_predictions is not None:
        # Simple CRPS approximation using quantiles
        crps = compute_crps_from_quantiles(quantile_predictions, targets)
        metrics["CRPS"] = crps

    return metrics


def compute_crps_from_quantiles(
    quantile_preds: np.ndarray,
    targets: np.ndarray,
    quantile_levels: list = [0.1, 0.5, 0.9],
) -> float:
    """Compute approximate CRPS from quantile predictions."""
    # Quantile loss
    total_loss = 0.0
    for i, q in enumerate(quantile_levels):
        error = targets - quantile_preds[..., i]
        loss = np.where(error >= 0, q * error, (q - 1) * error)
        total_loss += loss.mean()

    return total_loss / len(quantile_levels)


def evaluate_on_sarsim0(
    pipeline: Chronos2Pipeline,
    num_series: int = 100,
    context_length: int = 512,
    prediction_length: int = 64,
    seed: int = 12345,
) -> dict:
    """Evaluate model on SarSim0 synthetic data."""
    config = SarSimConfig(
        series_length=context_length + prediction_length + 200,
        context_window=context_length,
        prediction_window=prediction_length,
        burn_in=200,
    )

    generator = SarSim0Generator(config=config, seed=seed)
    series = generator.generate_series(
        batch_size=num_series, length=context_length + prediction_length
    )

    # Split into context and target
    context = series[:, :context_length].numpy()
    targets = series[:, context_length:].numpy()

    # Add variate dimension for Chronos-2
    context_3d = context[:, np.newaxis, :]

    # Predict
    quantiles, mean = pipeline.predict_quantiles(
        context_3d,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    # Extract predictions (remove variate dimension)
    predictions = np.array([m.squeeze(0).cpu().numpy() for m in mean])
    quantile_preds = np.array([q.squeeze(0).cpu().numpy() for q in quantiles])

    # Compute metrics
    metrics = compute_metrics(predictions, targets, quantile_preds)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chronos-2 model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model or HuggingFace model ID",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--num_series",
        type=int,
        default=100,
        help="Number of test series (default: 100)",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Context length (default: 512)",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=64,
        help="Prediction length (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for test data (default: 12345)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Chronos-2 Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device_map}")
    print(f"Test series: {args.num_series}")
    print(f"Context length: {args.context_length}")
    print(f"Prediction length: {args.prediction_length}")
    print()

    # Load model
    print("Loading model...")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_path,
        device_map=args.device_map,
    )
    print("Model loaded.")
    print()

    # Evaluate on SarSim0
    print("Evaluating on SarSim0 synthetic data...")
    metrics = evaluate_on_sarsim0(
        pipeline,
        num_series=args.num_series,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        seed=args.seed,
    )

    print()
    print("Results on SarSim0:")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print()

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
