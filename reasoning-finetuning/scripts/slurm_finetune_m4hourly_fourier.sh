#!/bin/bash
#SBATCH --job-name=chronos_m4h_fourier
#SBATCH --partition=hpc-low
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_m4hourly_fourier_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_m4hourly_fourier_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
TRAIN_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/fine_tune_m4hourly_fourier.py"
LOGS_DIR="$REPO_ROOT/reasoning-finetuning/logs"
mkdir -p "$LOGS_DIR"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Script       : $TRAIN_SCRIPT"
echo "Started      : $(date)"
echo "========================================"

/mnt/home/amitra/.local/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    "$TRAIN_SCRIPT" \
    --job-id "$SLURM_JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
