#!/bin/bash
set -e

echo "--- Installing dependencies ---"
pip install datasets pandas

echo "--- Preparing data ---"
cd /Users/amartya/Library/CloudStorage/GoogleDrive-anakin.olliver@gmail.com/My Drive/Code/Chronos/chronos-forecasting
python scripts/prepare_data.py

echo "--- Starting training ---"
cd scripts/training
python train.py --config configs/chronos-t5-tiny.yaml --training_data_paths '["../m4_hourly.jsonl"]' --output_dir ../../output

echo "--- Finetuning finished ---"
