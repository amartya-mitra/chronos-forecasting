#!/bin/bash
#SBATCH --job-name=pilot-joint-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/pilot-joint-train-%j.out
#SBATCH --error=reasoning-finetuning/logs/pilot-joint-train-%j.err

echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

VENV=/mnt/home/amitra/chronos-env
TORCHRUN=$VENV/bin/torchrun

echo "Python: $($VENV/bin/python --version)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -2 | tr '\n' '  ')"
echo ""

cd /mnt/home/amitra/chronos-forecasting
mkdir -p reasoning-finetuning/logs

# Smoke test first (3 steps, single process) — catches import/shape errors before 30-min DDP run
echo "=== SMOKE TEST (3 steps) ==="
SMOKE_TEST=1 $VENV/bin/python reasoning-finetuning/joint_train.py
SMOKE_EXIT=$?
if [ $SMOKE_EXIT -ne 0 ]; then
    echo "SMOKE TEST FAILED (exit $SMOKE_EXIT) — aborting DDP run"
    exit $SMOKE_EXIT
fi
echo "=== SMOKE TEST PASSED — starting DDP run ==="
echo ""

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29503 \
    reasoning-finetuning/joint_train.py

echo ""
echo "Done: $(date)"
