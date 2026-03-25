#!/bin/bash
#SBATCH --job-name=chronos_m4m_tsf
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_m4monthly_tsf_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_m4monthly_tsf_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
BUILD_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/build_m4monthly_tsf_dataset.py"
TRAIN_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/fine_tune_st_ts.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU(s)       : $CUDA_VISIBLE_DEVICES"
echo "N GPUs       : $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "Started      : $(date)"
echo "========================================"

echo ""
echo "--- Building M4 Monthly TSF dataset (36-month context, 18-step horizon, STL period=12) ---"
/opt/conda/bin/python3.12 -u "$BUILD_SCRIPT"

echo ""
echo "--- Fine-tune chronos-t5-small (TSF_M4M variant, 54-step output with SEP tokens) ---"
/mnt/home/amitra/.local/bin/torchrun --nproc_per_node=4 --master_port=29500 \
    "$TRAIN_SCRIPT" --variant TSF_M4M --job-id "$SLURM_JOB_ID"

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
