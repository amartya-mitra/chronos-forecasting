
import json
import pandas as pd
from datasets import load_dataset

# Load a subset of the GiftEval dataset from Hugging Face
# We'll use the 'm4_hourly' configuration as an example
dataset = load_dataset("Salesforce/GiftEval", "m4_hourly", split="train")

# Define the output file name
output_file = "m4_hourly.jsonl"

with open(output_file, "w") as f:
    for entry in dataset:
        # Convert the start timestamp to a pandas Timestamp object
        start_timestamp = pd.to_datetime(entry["start"], unit="s")

        # Create the JSON object in the required format
        output_entry = {
            "start": str(start_timestamp),
            "target": entry["target"],
        }

        # Write the JSON object as a new line in the output file
        f.write(json.dumps(output_entry) + "\n")

print(f"Successfully converted and saved the data to {output_file}")
