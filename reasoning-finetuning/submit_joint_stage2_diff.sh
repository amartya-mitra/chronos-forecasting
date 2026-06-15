#!/bin/bash
#SBATCH --job-name=joint-stage2-diff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/joint-stage2-diff-%j.out
#SBATCH --error=reasoning-finetuning/logs/joint-stage2-diff-%j.err

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

echo "=== PART C: Differentiation analysis ==="
$PYTHON reasoning-finetuning/diff_stage2.py

echo ""
echo "Done: $(date)"
