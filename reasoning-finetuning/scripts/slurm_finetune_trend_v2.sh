#!/bin/bash
#SBATCH --job-name=chronos_trend_v2
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_trend_v2_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_trend_v2_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
BUILD_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/build_trend_test_dataset_v2.py"
TRAIN_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/fine_tune_trend_v2.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Started      : $(date)"
echo "========================================"

echo ""
echo "--- Step 1: Build dataset (70% fast / 30% trend) ---"
/opt/conda/bin/python3.12 -u "$BUILD_SCRIPT"

echo ""
echo "--- Step 2: Fine-tune chronos-t5-small ---"
/opt/conda/bin/python3.12 -u "$TRAIN_SCRIPT" --job-id "$SLURM_JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
