#!/bin/bash
#SBATCH --job-name=fetch-m4weekly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/fetch-m4weekly-%j.out
#SBATCH --error=reasoning-finetuning/logs/fetch-m4weekly-%j.err

echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

VENV=/mnt/home/amitra/chronos-env
PYTHON=$VENV/bin/python

echo "Python: $($PYTHON --version)"
echo ""

cd /mnt/home/amitra/chronos-forecasting

$PYTHON reasoning-finetuning/scripts/fetch_m4weekly.py

echo ""
echo "Done: $(date)"
