import numpy as np
from gluonts.dataset.arrow import ArrowWriter
import datasets
from datasets import Features, Value, Sequence
from pathlib import Path
import os
import sys

def convert_to_arrow(
    path: str,
    dataset,
    compression: str = "lz4",
    limit: int = None
):
    """
    Store a given set of series into Arrow format at the specified path.
    """
    arrow_dataset = []
    
    print(f"Converting dataset (limit={limit})...")
    count = 0
    # Use explicit iterator with try-except if needed, but the dataset object handles it
    for i, entry in enumerate(dataset):
        try:
            target = entry.get('target')
            start = entry.get('start')
            
            # Fallback for start
            if start is None:
                 start = np.datetime64("2000-01-01 00:00", "s")
            
            # Ensure target is numpy array
            if isinstance(target, list):
                target = np.array(target)
                
            arrow_dataset.append({"start": start, "target": target})
            count += 1
            
            if count % 100 == 0:
                print(f"Processed {count} series")
                
            if limit and count >= limit:
                break
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            continue

    print(f"Writing {len(arrow_dataset)} series to {path}")
    ArrowWriter(compression=compression).write_to_file(
        arrow_dataset,
        path=path,
    )

def main():
    output_path = Path("reasoning-finetuning/data/gifteval-full.arrow")
    
    limit = 1000 
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        limit = None
        
    print(f"Running conversion (limit={limit})...")

    # Define minimal features currently failing
    # Based on error: 
    # item_id: string, start: timestamp[s], freq: string, target: list<float>, 
    # past_feat_dynamic_real: fixed_size_list<item: list<item: float>>[7]
    
    # We attempt to define a compatible schema. 
    # Note: 'past_feat_dynamic_real' is complex. 
    
    features = Features({
        "item_id": Value("string"),
        "start": Value("timestamp[s]"),
        "freq": Value("string"),
        "target": Sequence(Value("float64")), # Use float64 to be safe, or float32
        "past_feat_dynamic_real": Sequence(Sequence(Value("float64")), length=7),
    })

    print("Loading Salesforce/GiftEval...")
    try:
        # Load without explicit features to avoid casting errors
        ds = datasets.load_dataset(
            "Salesforce/GiftEval", 
            split="train", 
            streaming=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    convert_to_arrow(output_path, ds, limit=limit)
    print("Done!")

if __name__ == "__main__":
    main()
