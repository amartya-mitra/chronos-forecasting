#!/bin/bash
#SBATCH --job-name=chronos_tsf
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_tsf_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_tsf_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
BUILD_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/build_tsf_dataset.py"
TRAIN_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/fine_tune_st_ts.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Started      : $(date)"
echo "========================================"

echo ""
echo "--- Building TSF dataset (trend|seasonal|forecast, 192 steps, per-component normalisation) ---"
/opt/conda/bin/python3.12 -u "$BUILD_SCRIPT"

echo ""
echo "--- Fine-tune chronos-t5-small (TSF variant, 192-step output with SEP tokens) ---"
/opt/conda/bin/python3.12 -u "$TRAIN_SCRIPT" --variant TSF --job-id "$SLURM_JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
