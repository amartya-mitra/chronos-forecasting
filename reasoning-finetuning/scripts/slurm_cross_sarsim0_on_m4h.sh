#!/bin/bash
#SBATCH --job-name=chronos_cross_sarsim0_m4h
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/cross_sarsim0_on_m4h_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/cross_sarsim0_on_m4h_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/eval_cross_sarsim0_on_m4h.py"
JOB_ID="${1:-46391}"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Eval target  : finetune-sarsim0-joint-$JOB_ID  →  M4H"
echo "Started      : $(date)"
echo "========================================"

/opt/conda/bin/python3.12 -u "$SCRIPT" --job-id "$JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
