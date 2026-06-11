#!/bin/bash
#SBATCH --job-name=sweep-m4weekly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/sweep-m4weekly-%j.out
#SBATCH --error=reasoning-finetuning/logs/sweep-m4weekly-%j.err

echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

VENV=/mnt/home/amitra/chronos-env
PYTHON=$VENV/bin/python

echo "Python: $($PYTHON --version)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

cd /mnt/home/amitra/chronos-forecasting

$PYTHON reasoning-finetuning/train_ddp.py \
    --dataset m4weekly \
    --steps 500

echo ""
echo "Done: $(date)"
