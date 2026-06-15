#!/bin/bash
#SBATCH --job-name=pilot-joint-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/pilot-joint-eval-%j.out
#SBATCH --error=reasoning-finetuning/logs/pilot-joint-eval-%j.err

echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

VENV=/mnt/home/amitra/chronos-env
PYTHON=$VENV/bin/python

echo "Python: $($PYTHON --version)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

cd /mnt/home/amitra/chronos-forecasting
mkdir -p results

$PYTHON reasoning-finetuning/eval_joint_pilot.py

echo ""
echo "Done: $(date)"
