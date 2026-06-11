#!/bin/bash
#SBATCH --job-name=sweep-m4monthly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/sweep-m4monthly-%j.out
#SBATCH --error=reasoning-finetuning/logs/sweep-m4monthly-%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR
source /opt/conda/etc/profile.d/conda.sh
conda activate base

mkdir -p reasoning-finetuning/logs reasoning-finetuning/checkpoints results

# Pre-flight checks (P1–P6 + P_new): exits non-zero on assertion failure
echo ""
echo "=== PRE-FLIGHT CHECKS ==="
/opt/conda/bin/python3.12 reasoning-finetuning/scripts/preflight_m4monthly.py
PREFLIGHT_EXIT=$?

if [ $PREFLIGHT_EXIT -ne 0 ]; then
    echo "ABORT: Pre-flight checks failed (exit code $PREFLIGHT_EXIT). Halting training."
    exit $PREFLIGHT_EXIT
fi

echo ""
echo "=== TRAINING ==="

torchrun \
    --nproc_per_node=2 \
    --master_port=29503 \
    reasoning-finetuning/train_ddp.py \
    --dataset m4monthly \
    --steps   2000

EXIT_CODE=$?
echo ""
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
