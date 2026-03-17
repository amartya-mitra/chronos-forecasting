#!/usr/bin/env python3
"""
Check fast-mode output for special/non-numerical tokens
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent  # repo root
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chronos import ChronosPipeline
import datasets

def main():
    model_path = "./output/reasoning-m4-hourly-50/"
    dataset_path = "reasoning-finetuning/data/m4-hourly-subset-50.arrow"

    print("=" * 60)
    print("FAST-MODE TOKEN ANALYSIS")
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

    # Get model config
    model = pipeline.model
    config = model.model.config

    print("\nModel Tokenizer Configuration:")
    print(f"  Vocabulary size: {model.model.config.vocab_size}")
    print(f"  n_tokens: {config.n_tokens if hasattr(config, 'n_tokens') else 'N/A'}")
    print(f"  n_special_tokens: {config.n_special_tokens if hasattr(config, 'n_special_tokens') else 'N/A'}")
    print(f"  pad_token_id: {config.pad_token_id}")
    print(f"  eos_token_id: {config.eos_token_id}")

    # Determine special token ranges
    n_tokens = config.n_tokens if hasattr(config, 'n_tokens') else 4096
    n_special = config.n_special_tokens if hasattr(config, 'n_special_tokens') else 2

    print(f"\nToken ID Ranges:")
    print(f"  Numerical tokens: 0-{n_tokens-1}")
    print(f"  Special tokens (PAD, EOS): 0-{n_special-1}")
    print(f"  Control tokens (fast/reasoning): {n_tokens}+")

    print("\n" + "=" * 60)
    print("Running Fast-Mode Inference on 10 samples...")
    print("=" * 60)

    all_output_tokens = []
    special_token_stats = {
        'pad': 0,
        'eos': 0,
        'control': 0,
        'other_special': 0,
    }

    for i in range(10):
        sample = ds[i]
        target = np.array(sample["target"])

        # Prepare context (512 timesteps)
        context_length = min(512, len(target) - 64)
        context = target[-context_length-64:-64]

        # Generate forecast using pipeline
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        # Get token-level output
        with torch.no_grad():
            # Use pipeline's predict method to get forecasts
            forecast = pipeline.predict(
                context_tensor,
                prediction_length=64,
                num_samples=1,
            )

            # For token analysis, we need to go deeper
            # Transform context using the internal tokenizer
            tokenizer = pipeline.tokenizer
            input_ids, _, _ = tokenizer._input_transform(context_tensor)

            # Generate
            outputs = model.model.generate(
                input_ids,
                max_new_tokens=64,
                num_return_sequences=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

            # Get generated token IDs
            # For decoder-only models, sequences contains full output
            # For encoder-decoder models, sequences contains decoder output only
            full_sequence = outputs.sequences[0].cpu().numpy()

            # The generated tokens are in the decoder output
            # For T5, the first token is decoder_start_token_id (0), then the generated tokens
            if len(full_sequence) > 1:
                generated_tokens = full_sequence[1:]  # Skip decoder start token
            else:
                generated_tokens = full_sequence

            all_output_tokens.append(generated_tokens)

            # Analyze tokens
            for token_id in generated_tokens:
                if token_id == config.pad_token_id:
                    special_token_stats['pad'] += 1
                elif token_id == config.eos_token_id:
                    special_token_stats['eos'] += 1
                elif token_id >= n_tokens:
                    special_token_stats['control'] += 1
                elif token_id < n_special and token_id not in [config.pad_token_id, config.eos_token_id]:
                    special_token_stats['other_special'] += 1

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/10 samples")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Calculate output lengths
    output_lengths = [len(tokens) for tokens in all_output_tokens]
    print(f"\nOutput Length Statistics:")
    print(f"  Average: {np.mean(output_lengths):.1f} tokens")
    print(f"  Min: {np.min(output_lengths)} tokens")
    print(f"  Max: {np.max(output_lengths)} tokens")

    # Count unique special tokens
    all_tokens = np.concatenate(all_output_tokens)
    unique_tokens = np.unique(all_tokens)

    special_tokens_found = []
    for token_id in unique_tokens:
        if token_id == config.pad_token_id:
            special_tokens_found.append(('PAD', token_id))
        elif token_id == config.eos_token_id:
            special_tokens_found.append(('EOS', token_id))
        elif token_id >= n_tokens:
            special_tokens_found.append((f'CONTROL_{token_id}', token_id))
        elif token_id < n_special:
            special_tokens_found.append((f'SPECIAL_{token_id}', token_id))

    print(f"\nSpecial/Non-numerical Tokens Found:")
    if special_tokens_found:
        print(f"  Total unique special tokens: {len(special_tokens_found)}")
        for name, tid in special_tokens_found:
            count = np.sum(all_tokens == tid)
            print(f"    - {name} (ID={tid}): {count} occurrences")
    else:
        print("  None found!")

    print(f"\nSpecial Token Statistics:")
    print(f"  PAD tokens: {special_token_stats['pad']}")
    print(f"  EOS tokens: {special_token_stats['eos']}")
    print(f"  Control tokens (4096+): {special_token_stats['control']}")
    print(f"  Other special tokens: {special_token_stats['other_special']}")

    total_special = sum(special_token_stats.values())
    total_tokens = sum(output_lengths)
    if total_tokens > 0:
        print(f"\nTotal special tokens: {total_special} / {total_tokens} ({100*total_special/total_tokens:.1f}%)")
    else:
        print(f"\nTotal special tokens: {total_special} / {total_tokens}")

    print("\n" + "=" * 60)
    print("SUMMARY FOR USER")
    print("=" * 60)
    print(f"\n✓ Fast-mode output length: {int(np.mean(output_lengths))} tokens (average)")
    print(f"✓ Special/non-numerical tokens found: {len(special_tokens_found)}")
    if special_tokens_found:
        print(f"✓ Token types:")
        for name, tid in special_tokens_found:
            print(f"    - {name} (ID={tid})")
    print()

if __name__ == "__main__":
    main()
