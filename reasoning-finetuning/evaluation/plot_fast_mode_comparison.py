#!/usr/bin/env python3
"""
Generate performance comparison plot for fast-mode vs ground truth
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent  # repo root
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chronos import ChronosPipeline
import datasets

def main():
    model_path = "./output/reasoning-m4-hourly-50/"
    dataset_path = "reasoning-finetuning/data/m4-hourly-subset-50.arrow"
    output_dir = Path("reasoning-finetuning/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FAST-MODE PERFORMANCE COMPARISON")
    print("=" * 60)

    # Load model
    print(f"Loading model from: {model_path}")
    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    ds = datasets.load_dataset("arrow", data_files=dataset_path, split="train")

    # Generate forecasts for 5 samples
    num_samples_to_plot = 5
    prediction_length = 64

    print(f"\nGenerating forecasts for {num_samples_to_plot} samples...")

    fig, axes = plt.subplots(num_samples_to_plot, 1, figsize=(12, 3 * num_samples_to_plot))
    if num_samples_to_plot == 1:
        axes = [axes]

    for idx in range(num_samples_to_plot):
        sample = ds[idx]
        target = np.array(sample["target"])

        # Prepare context (512 timesteps)
        context_length = min(512, len(target) - prediction_length)
        context = target[-context_length-prediction_length:-prediction_length]
        ground_truth = target[-prediction_length:]

        # Generate forecast
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            forecast = pipeline.predict(
                context_tensor,
                prediction_length=prediction_length,
                num_samples=20,  # Generate 20 samples for uncertainty quantification
            )

        # Convert to numpy
        forecast_np = forecast[0].cpu().numpy() if isinstance(forecast[0], torch.Tensor) else forecast[0]

        # Calculate quantiles
        forecast_median = np.median(forecast_np, axis=0)
        forecast_mean = np.mean(forecast_np, axis=0)
        forecast_10 = np.percentile(forecast_np, 10, axis=0)
        forecast_90 = np.percentile(forecast_np, 90, axis=0)

        # Plot
        ax = axes[idx]

        # Context
        context_x = np.arange(len(context))
        ax.plot(context_x, context, label="Context", color="gray", alpha=0.6)

        # Ground truth
        forecast_x = np.arange(len(context), len(context) + prediction_length)
        ax.plot(forecast_x, ground_truth, label="Ground Truth", color="blue", linewidth=2)

        # Forecast
        ax.plot(forecast_x, forecast_median, label="Forecast (Median)", color="red", linewidth=2)
        ax.fill_between(forecast_x, forecast_10, forecast_90, alpha=0.3, color="red", label="80% Confidence Interval")

        # Styling
        ax.set_title(f"Sample {idx + 1}: Fast-Mode Forecast vs Ground Truth", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Add vertical line to separate context from forecast
        ax.axvline(x=len(context), color='black', linestyle='--', alpha=0.5)

        # Calculate and display metrics
        mae = np.mean(np.abs(ground_truth - forecast_median))
        mse = np.mean((ground_truth - forecast_median) ** 2)
        rmse = np.sqrt(mse)

        textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        print(f"  Sample {idx + 1}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "fast_mode_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {output_path}")

    # Create a summary plot with all samples
    print("\nGenerating summary metrics plot...")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    mae_values = []
    rmse_values = []

    for idx in range(min(10, len(ds))):
        sample = ds[idx]
        target = np.array(sample["target"])

        context_length = min(512, len(target) - prediction_length)
        context = target[-context_length-prediction_length:-prediction_length]
        ground_truth = target[-prediction_length:]

        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            forecast = pipeline.predict(
                context_tensor,
                prediction_length=prediction_length,
                num_samples=20,
            )

        forecast_np = forecast[0].cpu().numpy() if isinstance(forecast[0], torch.Tensor) else forecast[0]
        forecast_median = np.median(forecast_np, axis=0)

        mae = np.mean(np.abs(ground_truth - forecast_median))
        mse = np.mean((ground_truth - forecast_median) ** 2)
        rmse = np.sqrt(mse)

        mae_values.append(mae)
        rmse_values.append(rmse)

    # Plot MAE
    axes2[0].bar(range(len(mae_values)), mae_values, color='steelblue', alpha=0.7)
    axes2[0].set_title("Mean Absolute Error (MAE) by Sample", fontsize=12, fontweight="bold")
    axes2[0].set_xlabel("Sample Index")
    axes2[0].set_ylabel("MAE")
    axes2[0].grid(True, alpha=0.3, axis='y')

    # Plot RMSE
    axes2[1].bar(range(len(rmse_values)), rmse_values, color='coral', alpha=0.7)
    axes2[1].set_title("Root Mean Squared Error (RMSE) by Sample", fontsize=12, fontweight="bold")
    axes2[1].set_xlabel("Sample Index")
    axes2[1].set_ylabel("RMSE")
    axes2[1].grid(True, alpha=0.3, axis='y')

    # Add average lines
    axes2[0].axhline(y=np.mean(mae_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mae_values):.2f}')
    axes2[1].axhline(y=np.mean(rmse_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rmse_values):.2f}')

    axes2[0].legend()
    axes2[1].legend()

    plt.tight_layout()

    output_path2 = output_dir / "fast_mode_metrics_summary.png"
    plt.savefig(output_path2, dpi=150, bbox_inches="tight")
    print(f"✓ Summary plot saved to: {output_path2}")

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Average MAE across {len(mae_values)} samples: {np.mean(mae_values):.2f}")
    print(f"Average RMSE across {len(rmse_values)} samples: {np.mean(rmse_values):.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
