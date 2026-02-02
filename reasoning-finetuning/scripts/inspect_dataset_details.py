import datasets

try:
    print("Inspecting Salesforce/GiftEval dataset...")
    ds_builder = datasets.load_dataset_builder("Salesforce/GiftEval")
    print(f"Description: {ds_builder.info.description}")
    print(f"Features: {ds_builder.info.features}")
    print(f"Splits: {ds_builder.info.splits}")
    
    # Check if there are specific configurations (subsets) typically exposed
    # Some benchmarks have a 'core' or 'all' config.
    print(f"Builder Configs: {ds_builder.BUILDER_CONFIGS}")
    
except Exception as e:
    print(f"Error inspecting dataset: {e}")
