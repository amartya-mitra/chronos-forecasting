#!/bin/bash
#SBATCH --job-name=chronos_cross_m4h_sarsim0
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/cross_m4h_on_sarsim0_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/cross_m4h_on_sarsim0_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/eval_cross_m4h_on_sarsim0.py"
JOB_ID="${1:-46008}"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Eval target  : finetune-m4h-joint-$JOB_ID  →  SarSim0"
echo "Started      : $(date)"
echo "========================================"

/opt/conda/bin/python3.12 -u "$SCRIPT" --job-id "$JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
