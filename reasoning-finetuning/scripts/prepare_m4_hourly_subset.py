#!/usr/bin/env python3
"""
Extract 50 samples from GiftEval M4-Hourly dataset
"""

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
import datasets
from pathlib import Path

def extract_m4_hourly_subset(output_path: str, num_samples: int = 50):
    """
    Extract a subset of samples from M4-Hourly dataset

    Args:
        output_path: Path to save the subset
        num_samples: Number of samples to extract (default: 50)
    """
    repo_url_base = "hf://datasets/Salesforce/GiftEval@30841734ac5cfddbd0c3bad6d09d2b6b32becbb0/"
    m4_hourly_path = "m4_hourly/data-00000-of-00001.arrow"

    url = repo_url_base + m4_hourly_path

    print(f"Loading M4-Hourly dataset from: {m4_hourly_path}")

    # Load the dataset
    ds = datasets.load_dataset(
        "arrow",
        data_files=url,
        split="train"
    )

    print(f"Total samples in M4-Hourly: {len(ds)}")

    # Extract first num_samples
    arrow_dataset = []
    for i in range(min(num_samples, len(ds))):
        entry = ds[i]
        target = entry.get('target')
        start = entry.get('start')

        if start is None:
            start = np.datetime64("2000-01-01 00:00", "s")

        if isinstance(target, list):
            target = np.array(target)

        # Handle multivariate data (take first dimension)
        if target.ndim > 1:
            target = target[0, :]

        # Skip if all NaNs
        if np.isnan(target).all():
            continue

        arrow_dataset.append({"start": start, "target": target})

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_samples} series...")

    print(f"Writing {len(arrow_dataset)} series to {output_path}")
    ArrowWriter(compression="lz4").write_to_file(
        arrow_dataset,
        path=output_path,
    )

    print(f"✓ Done! Extracted {len(arrow_dataset)} samples from M4-Hourly")
    return len(arrow_dataset)

if __name__ == "__main__":
    output_dir = Path("reasoning-finetuning/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "m4-hourly-subset-50.arrow"
    extract_m4_hourly_subset(str(output_path), num_samples=50)
