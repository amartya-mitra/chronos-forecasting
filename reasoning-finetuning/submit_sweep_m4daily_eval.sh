#!/bin/bash
#SBATCH --job-name=sweep-m4daily-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/sweep-m4daily-eval-%j.out
#SBATCH --error=reasoning-finetuning/logs/sweep-m4daily-eval-%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR

VENV=/mnt/home/amitra/chronos-env
PYTHON=$VENV/bin/python
echo "Python:   $($PYTHON --version 2>&1)"
echo "Torch:    $($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null)"

mkdir -p reasoning-finetuning/logs results

$PYTHON reasoning-finetuning/eval_m4daily.py

EXIT_CODE=$?
echo ""
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
