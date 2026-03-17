#!/bin/bash
#SBATCH --job-name=chronos_trend_v5
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_trend_v5_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_trend_v5_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
DATA_FILE="$REPO_ROOT/reasoning-finetuning/data/sarsim0-trend-test-v2.arrow"
BUILD_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/build_trend_test_dataset_v2.py"
TRAIN_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/fine_tune_trend_v5.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Started      : $(date)"
echo "========================================"

if [ ! -f "$DATA_FILE" ]; then
    echo ""
    echo "--- Dataset not found, building (70% fast / 30% trend) ---"
    /opt/conda/bin/python3.12 -u "$BUILD_SCRIPT"
else
    echo ""
    echo "--- Dataset already exists, skipping build ---"
fi

echo ""
echo "--- Fine-tune chronos-t5-small (v5 — decoder-mode + better embed init) ---"
/opt/conda/bin/python3.12 -u "$TRAIN_SCRIPT" --job-id "$SLURM_JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
