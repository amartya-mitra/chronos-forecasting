#!/usr/bin/env python3
"""
Simple example of fine-tuning Chronos-2 with SarSim0 data.

This is a minimal example showing the core workflow.
For production use, see train_chronos2.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from chronos import BaseChronosPipeline, Chronos2Pipeline

from sarsim0 import SarSimConfig, SarSim0Generator


def main():
    # Configuration
    context_length = 512
    prediction_length = 64
    num_train = 1000
    num_steps = 500
    batch_size = 32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Generate synthetic training data with SarSim0
    print("\n1. Generating training data with SarSim0...")
    config = SarSimConfig(
        series_length=context_length + prediction_length + 200,
        burn_in=200,
    )
    generator = SarSim0Generator(config=config, seed=42)
    series = generator.generate_series(
        batch_size=num_train, length=context_length + prediction_length
    )

    # Convert to Chronos-2 input format
    train_inputs = [{"target": series[i].numpy()} for i in range(num_train)]
    print(f"   Generated {len(train_inputs)} training series")
    print(f"   Series length: {len(train_inputs[0]['target'])}")

    # 2. Load pretrained Chronos-2
    print("\n2. Loading pretrained Chronos-2...")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
    )

    # 3. Fine-tune with LoRA (faster, fewer parameters)
    print(f"\n3. Fine-tuning with LoRA for {num_steps} steps...")
    finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=prediction_length,
        finetune_mode="lora",
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=1e-4,
        logging_steps=100,
    )

    # 4. Test the fine-tuned model
    print("\n4. Testing fine-tuned model...")
    test_generator = SarSim0Generator(config=config, seed=999)
    test_series = test_generator.generate_series(
        batch_size=10, length=context_length + prediction_length
    )

    # Prepare test input
    test_context = test_series[:, :context_length].numpy()
    test_target = test_series[:, context_length:].numpy()
    test_input_3d = test_context[:, np.newaxis, :]  # (batch, 1, context_length)

    # Predict
    quantiles, mean = finetuned_pipeline.predict_quantiles(
        test_input_3d,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    # Compute simple MAE
    predictions = np.array([m.squeeze(0).cpu().numpy() for m in mean])
    mae = np.abs(predictions - test_target).mean()

    print(f"   Test MAE: {mae:.4f}")
    print(f"   Predictions shape: {predictions.shape}")

    # 5. Save the model (optional)
    print("\n5. Model can be saved with:")
    print("   finetuned_pipeline.save_pretrained('path/to/save')")

    print("\nDone!")


if __name__ == "__main__":
    main()
