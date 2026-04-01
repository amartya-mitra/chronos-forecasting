#!/bin/bash
#SBATCH --job-name=chronos_fast_eval
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/eval_finetuned_fast_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/eval_finetuned_fast_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/eval_finetuned_fast.py"
JOB_ID="${1:-46008}"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Eval target  : finetune-m4h-joint-$JOB_ID"
echo "Started      : $(date)"
echo "========================================"

/opt/conda/bin/python3.12 -u "$SCRIPT" --job-id "$JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
