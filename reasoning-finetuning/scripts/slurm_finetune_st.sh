#!/bin/bash
#SBATCH --job-name=chronos_st
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_st_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_st_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
DATA_FILE="$REPO_ROOT/reasoning-finetuning/data/sarsim0-st.arrow"
BUILD_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/build_st_dataset.py"
TRAIN_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/fine_tune_st_ts.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Started      : $(date)"
echo "========================================"

echo ""
echo "--- Building ST dataset (seasonal|trend, 128 steps, per-component normalisation) ---"
/opt/conda/bin/python3.12 -u "$BUILD_SCRIPT"

echo ""
echo "--- Fine-tune chronos-t5-small (ST variant, 128-step output) ---"
/opt/conda/bin/python3.12 -u "$TRAIN_SCRIPT" --variant ST --job-id "$SLURM_JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
