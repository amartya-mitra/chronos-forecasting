#!/usr/bin/env python3
"""
Analyze fast-mode output for special/non-numerical tokens
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chronos import ChronosPipeline
import datasets

def analyze_fast_mode_output(model_path: str, dataset_path: str, num_samples: int = 10):
    """
    Run fast-mode inference and analyze the output tokens.

    Args:
        model_path: Path to the trained model
        dataset_path: Path to the test dataset
        num_samples: Number of samples to test
    """
    print("=" * 60)
    print("FAST-MODE OUTPUT ANALYSIS")
    print("=" * 60)

    # Load the finetuned model
    print(f"Loading model from: {model_path}")
    pipeline = ChronosPipeline.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    ds = datasets.load_dataset("arrow", data_files=dataset_path, split="train")

    print(f"\nAnalyzing {num_samples} samples in fast mode...")
    print("=" * 60)

    special_token_counts = {}
    non_numerical_tokens = []
    output_lengths = []

    # Get tokenizer info
    tokenizer = pipeline.model.model.config
    n_tokens = getattr(tokenizer, 'n_tokens', 4096)
    n_special_tokens = getattr(tokenizer, 'n_special_tokens', 2)
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    eos_token_id = getattr(tokenizer, 'eos_token_id', 1)

    print(f"Tokenizer Configuration:")
    print(f"  n_tokens: {n_tokens}")
    print(f"  n_special_tokens: {n_special_tokens}")
    print(f"  pad_token_id: {pad_token_id}")
    print(f"  eos_token_id: {eos_token_id}")
    print(f"  fast_mode_token_id: {n_tokens} (if exists)")
    print(f"  reasoning_mode_token_id: {n_tokens + 1} (if exists)")
    print()

    for i in range(min(num_samples, len(ds))):
        sample = ds[i]
        target = np.array(sample["target"])

        # Use last 512 timesteps as context
        context_length = min(512, len(target) - 64)
        context = target[-context_length-64:-64]

        # Generate forecast
        try:
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

            # Get the raw token output
            with torch.no_grad():
                # Run model forward pass
                output = pipeline.model.model.generate(
                    pipeline.model.tokenizer.input_transform(context_tensor),
                    max_new_tokens=64,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                # Extract token IDs
                token_ids = output.sequences[0].cpu().numpy()

                # Analyze tokens
                output_length = len(token_ids)
                output_lengths.append(output_length)

                # Find special/non-numerical tokens
                for token_id in token_ids:
                    # Check if it's a special token
                    if token_id == pad_token_id:
                        special_token_counts['PAD'] = special_token_counts.get('PAD', 0) + 1
                    elif token_id == eos_token_id:
                        special_token_counts['EOS'] = special_token_counts.get('EOS', 0) + 1
                    elif token_id >= n_tokens:
                        # Control tokens (fast_mode, reasoning_mode)
                        token_name = f"CONTROL_{token_id}"
                        special_token_counts[token_name] = special_token_counts.get(token_name, 0) + 1
                        non_numerical_tokens.append((i, token_id, token_name))
                    elif token_id < 2:
                        # Other special tokens
                        token_name = f"SPECIAL_{token_id}"
                        special_token_counts[token_name] = special_token_counts.get(token_name, 0) + 1
                        non_numerical_tokens.append((i, token_id, token_name))

                if (i + 1) % 5 == 0:
                    print(f"Processed {i + 1}/{num_samples} samples...")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nTotal samples analyzed: {len(output_lengths)}")

    if output_lengths:
        print(f"\nOutput Length Statistics:")
        print(f"  Average length: {np.mean(output_lengths):.2f}")
        print(f"  Min length: {np.min(output_lengths)}")
        print(f"  Max length: {np.max(output_lengths)}")

    print(f"\nSpecial/Non-numerical Token Counts:")
    if special_token_counts:
        for token_name, count in sorted(special_token_counts.items()):
            print(f"  {token_name}: {count} occurrences")
    else:
        print("  None found!")

    if non_numerical_tokens:
        print(f"\nNon-numerical tokens found in outputs:")
        print(f"  Total occurrences: {len(non_numerical_tokens)}")
        print(f"  Unique tokens: {set(tok[2] for tok in non_numerical_tokens)}")
    else:
        print(f"\nNo non-numerical tokens found in fast-mode output!")

    print("\n" + "=" * 60)

    return {
        'special_token_counts': special_token_counts,
        'non_numerical_tokens': non_numerical_tokens,
        'output_lengths': output_lengths,
    }

if __name__ == "__main__":
    import fire

    # Default paths - use final checkpoint
    model_path = "./output/reasoning-m4-hourly-50/"
    dataset_path = "reasoning-finetuning/data/m4-hourly-subset-50.arrow"

    analyze_fast_mode_output(model_path, dataset_path, num_samples=10)
