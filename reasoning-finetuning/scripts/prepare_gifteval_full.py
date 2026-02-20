#!/usr/bin/env python3
"""
Full GiftEval Dataset Preparation
Loads all arrow files from Salesforce/GiftEval repository to create a comprehensive dataset.
"""

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
import datasets
from pathlib import Path
import os
import sys

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def convert_to_arrow(
    path: str,
    file_list: list,
    limit: int = 50000,
    compression: str = "lz4"
):
    """
    Store a given set of series into Arrow format at the specified path.
    """
    arrow_dataset = []
    count = 0
    
    repo_url_base = "hf://datasets/Salesforce/GiftEval@30841734ac5cfddbd0c3bad6d09d2b6b32becbb0/"

    print(f"Converting full dataset (limit={limit})...")
    
    for relative_path in file_list:
        if "cache-" in relative_path: # Skip cache files
            continue
            
        print(f"Loading: {relative_path}")
        url = repo_url_base + relative_path
        
        try:
            # Use specific arrow builder for each file
            ds = datasets.load_dataset(
                "arrow", 
                data_files=url,
                split="train", 
                streaming=True
            )
            
            for entry in ds:
                target = entry.get('target')
                start = entry.get('start')
                
                if start is None:
                     start = np.datetime64("2000-01-01 00:00", "s")
                
                if isinstance(target, list):
                    target = np.array(target)
                
                # Handle multivariate data (take first dimension)
                if target.ndim > 1:
                    target = target[0, :]
                
                # Check for NaNs and handle
                if np.isnan(target).all():
                    continue
                    
                arrow_dataset.append({"start": start, "target": target})
                count += 1
                
                if count % 1000 == 0:
                    print(f"Processed {count} series...")
                    
                if limit and count >= limit:
                    break
                    
            if limit and count >= limit:
                break
                
        except Exception as e:
            print(f"Error loading {relative_path}: {e}")
            continue

    print(f"Writing {len(arrow_dataset)} series to {path}")
    ArrowWriter(compression=compression).write_to_file(
        arrow_dataset,
        path=path,
    )
    return count

def main():
    arrow_files_path = Path("reasoning-finetuning/scripts/arrow_files.txt")
    if not arrow_files_path.exists():
        print("Error: arrow_files.txt not found")
        return
        
    with open(arrow_files_path, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]
        
    output_path = Path("reasoning-finetuning/data/gifteval-full.arrow")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 10k is a good balance of diversity vs. preparation time
    limit = 10000 
    
    total = convert_to_arrow(output_path, file_list, limit=limit)
    print(f"Done! Total series collected: {total}")

if __name__ == "__main__":
    main()
