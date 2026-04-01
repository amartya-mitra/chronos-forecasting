#!/bin/bash
#SBATCH --job-name=chronos_sarsim0_baseline
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/eval_pretrained_sarsim0_fast_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/eval_pretrained_sarsim0_fast_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/eval_pretrained_sarsim0_fast.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Started      : $(date)"
echo "========================================"

/opt/conda/bin/python3.12 -u "$SCRIPT"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
