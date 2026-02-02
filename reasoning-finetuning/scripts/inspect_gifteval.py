import datasets

try:
    ds = datasets.load_dataset("Salesforce/GiftEval", split="train", streaming=True)
    print("Dataset loaded successfully.")
    
    # Get first example
    example = next(iter(ds))
    print("\nFirst example keys:", example.keys())
    print("\nFirst example content:")
    for k, v in example.items():
        print(f"{k}: {v} (Type: {type(v)})")

except Exception as e:
    print(f"Error loading dataset: {e}")
