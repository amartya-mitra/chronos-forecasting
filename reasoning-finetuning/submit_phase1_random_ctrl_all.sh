#!/bin/bash
#SBATCH --job-name=phase1-rand-ctrl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/phase1-rand-ctrl-%j.out
#SBATCH --error=reasoning-finetuning/logs/phase1-rand-ctrl-%j.err

echo "Node:   $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Date:   $(date)"

VENV=/mnt/home/amitra/chronos-env
PYTHON=$VENV/bin/python

echo "Python: $($PYTHON --version)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

cd /mnt/home/amitra/chronos-forecasting
mkdir -p reasoning-finetuning/logs results

$PYTHON -u reasoning-finetuning/phase1_random_ctrl_all.py

echo ""
echo "Done: $(date)"
