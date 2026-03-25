#!/bin/bash
#SBATCH --job-name=chronos_ts
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_ts_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_ts_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
DATA_FILE="$REPO_ROOT/reasoning-finetuning/data/sarsim0-ts.arrow"
BUILD_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/build_ts_dataset.py"
TRAIN_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/fine_tune_st_ts.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Started      : $(date)"
echo "========================================"

echo ""
echo "--- Building TS dataset (trend|seasonal, 128 steps, per-component normalisation) ---"
/opt/conda/bin/python3.12 -u "$BUILD_SCRIPT"

echo ""
echo "--- Fine-tune chronos-t5-small (TS variant, 128-step output) ---"
/opt/conda/bin/python3.12 -u "$TRAIN_SCRIPT" --variant TS --job-id "$SLURM_JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
