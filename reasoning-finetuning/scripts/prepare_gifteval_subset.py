import numpy as np
from gluonts.dataset.arrow import ArrowWriter
import datasets
from pathlib import Path

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
    for i, entry in enumerate(dataset):
        try:
            target = entry.get('target')
            start = entry.get('start')
            
            if start is None:
                 start = np.datetime64("2000-01-01 00:00", "s")
            
            if isinstance(target, list):
                target = np.array(target)
                
            arrow_dataset.append({"start": start, "target": target})
            count += 1
            
            if count % 100 == 0:
                print(f"Processed {count} series")
                
            if limit and count >= limit:
                break
        except Exception:
            continue

    print(f"Writing {len(arrow_dataset)} series to {path}")
    ArrowWriter(compression=compression).write_to_file(
        arrow_dataset,
        path=path,
    )

def main():
    output_path = Path("reasoning-finetuning/data/gifteval-subset.arrow")
    limit = 2000 
    
    print(f"Running conversion for SUBSET (limit={limit})...")
    print("Using specific data file (M4 Hourly) to ensure schema consistency.")
    
    # Selecting one specific file/subset to avoid schema mixing issues
    # M4 Hourly is a standard benchmark
    url = "hf://datasets/Salesforce/GiftEval@30841734ac5cfddbd0c3bad6d09d2b6b32becbb0/m4_hourly/data-00000-of-00001.arrow"
    
    try:
        # Explicitly use "arrow" builder
        ds = datasets.load_dataset(
            "arrow", 
            data_files=url,
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
