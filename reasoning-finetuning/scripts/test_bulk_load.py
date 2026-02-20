import datasets
from pathlib import Path

def test_bulk_load():
    repo_url_base = "hf://datasets/Salesforce/GiftEval@30841734ac5cfddbd0c3bad6d09d2b6b32becbb0/"
    
    # Test with two different segments
    files = [
        "LOOP_SEATTLE/5T/data-00000-of-00001.arrow",
        "hospital/data-00000-of-00001.arrow"
    ]
    urls = [repo_url_base + f for f in files]
    
    print("Testing bulk load with arrow builder...")
    try:
        # Define target features
        features = datasets.Features({
            "start": datasets.Value("timestamp[s]"),
            "target": datasets.Sequence(datasets.Value("float32"))
        })
        
        ds = datasets.load_dataset(
            "arrow", 
            data_files=urls,
            split="train",
            features=features,
            streaming=True
        )
        
        print("Success! First entry keys:", next(iter(ds)).keys())
        return True
    except Exception as e:
        print(f"Bulk load failed: {e}")
        return False

if __name__ == "__main__":
    test_bulk_load()
