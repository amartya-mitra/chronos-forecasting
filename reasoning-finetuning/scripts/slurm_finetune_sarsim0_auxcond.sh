#!/bin/bash
#SBATCH --job-name=chronos_sarsim0_auxcond
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_sarsim0_auxcond_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/finetune_sarsim0_auxcond_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
TRAIN_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/fine_tune_sarsim0_aux_cond.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Script       : $TRAIN_SCRIPT"
echo "Phase        : Phase 1 (freeze-backbone — diagnostic)"
echo "Started      : $(date)"
echo "========================================"

/mnt/home/amitra/.local/bin/torchrun \
    --nproc_per_node=2 \
    --master_port=29504 \
    "$TRAIN_SCRIPT" \
    --job-id "$SLURM_JOB_ID" \
    --freeze-backbone

echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
