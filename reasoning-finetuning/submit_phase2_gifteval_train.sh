#!/bin/bash
#SBATCH --job-name=phase2-gifteval-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/phase2-gifteval-train-%j.out
#SBATCH --error=reasoning-finetuning/logs/phase2-gifteval-train-%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR
source /opt/conda/etc/profile.d/conda.sh
conda activate base

mkdir -p reasoning-finetuning/logs reasoning-finetuning/checkpoints results

# Pre-flight checks (P1–P6): exits non-zero on assertion failure
echo ""
echo "=== PRE-FLIGHT CHECKS ==="
/opt/conda/bin/python3.12 reasoning-finetuning/scripts/preflight_gifteval.py
PREFLIGHT_EXIT=$?

if [ $PREFLIGHT_EXIT -ne 0 ]; then
    echo "ABORT: Pre-flight checks failed (exit code $PREFLIGHT_EXIT). Halting training."
    exit $PREFLIGHT_EXIT
fi

echo ""
echo "=== TRAINING ==="

/opt/conda/bin/python3.12 \
    reasoning-finetuning/train_ddp.py \
    --dataset gifteval \
    --steps   1000

EXIT_CODE=$?
echo ""
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
