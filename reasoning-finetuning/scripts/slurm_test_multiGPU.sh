#!/bin/bash
#SBATCH --job-name=test_ddp
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/test_ddp_%j.log
#SBATCH --error=/mnt/home/amitra/chronos-forecasting/reasoning-finetuning/logs/test_ddp_%j.err

set -e

REPO_ROOT="/mnt/home/amitra/chronos-forecasting"
TEST_SCRIPT="$REPO_ROOT/reasoning-finetuning/scripts/test_multiGPU.py"

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU(s)       : $CUDA_VISIBLE_DEVICES"
echo "Started      : $(date)"
echo "========================================"

echo ""
echo "--- Smoke test: DDP train→eval(generate)→train cycle ---"
/mnt/home/amitra/.local/bin/torchrun --nproc_per_node=4 --master_port=29501 "$TEST_SCRIPT"

echo ""
echo "========================================"
echo "Finished     : $(date)"
echo "========================================"
