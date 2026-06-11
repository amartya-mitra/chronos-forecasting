#!/bin/bash
#SBATCH --job-name=sweep-m4daily
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/sweep-m4daily-%j.out
#SBATCH --error=reasoning-finetuning/logs/sweep-m4daily-%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR

VENV=/mnt/home/amitra/chronos-env
PYTHON=$VENV/bin/python
TORCHRUN=$VENV/bin/torchrun
echo "Python:   $($PYTHON --version 2>&1)"
echo "Torch:    $($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null)"

mkdir -p reasoning-finetuning/logs reasoning-finetuning/checkpoints results

# Pre-flight checks (P1–P6 + P_new + P_daily): exits non-zero on assertion failure
echo ""
echo "=== PRE-FLIGHT CHECKS ==="
$PYTHON reasoning-finetuning/scripts/preflight_m4daily.py
PREFLIGHT_EXIT=$?

if [ $PREFLIGHT_EXIT -ne 0 ]; then
    echo "ABORT: Pre-flight checks failed (exit code $PREFLIGHT_EXIT). Halting training."
    exit $PREFLIGHT_EXIT
fi

echo ""
echo "=== TRAINING ==="

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29504 \
    reasoning-finetuning/train_ddp.py \
    --dataset m4daily \
    --steps   2000

EXIT_CODE=$?
echo ""
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
