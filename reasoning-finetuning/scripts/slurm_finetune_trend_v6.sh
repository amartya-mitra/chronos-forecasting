#!/bin/bash
#SBATCH --job-name=chronos_trend_v6
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_trend_v6_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_trend_v6_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
DATA_FILE="$REPO_ROOT/reasoning-finetuning/data/sarsim0-contrastive-v1.arrow"
BUILD_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/build_contrastive_dataset.py"
TRAIN_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/fine_tune_trend_v6.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Started      : $(date)"
echo "========================================"

if [ ! -f "$DATA_FILE" ]; then
    echo ""
    echo "--- Dataset not found, building (30% fast / 70% trend, with wrong_target) ---"
    /opt/conda/bin/python3.12 -u "$BUILD_SCRIPT"
else
    echo ""
    echo "--- Dataset already exists, skipping build ---"
fi

echo ""
echo "--- Fine-tune chronos-t5-small (v6 — decoder-mode + contrastive loss) ---"
/opt/conda/bin/python3.12 -u "$TRAIN_SCRIPT" --job-id "$SLURM_JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
