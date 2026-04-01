#!/bin/bash
#SBATCH --job-name=chronos_sarsim0_joint
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_sarsim0_joint_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_sarsim0_joint_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
TRAIN_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/fine_tune_sarsim0_joint.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Started      : $(date)"
echo "========================================"

/mnt/home/amitra/.local/bin/torchrun \
    --nproc_per_node=2 \
    --master_port=29502 \
    "$TRAIN_SCRIPT" \
    --job-id "$SLURM_JOB_ID" \
    --fast-ratio 0.20

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
