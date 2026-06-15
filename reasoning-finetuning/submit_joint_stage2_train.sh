#!/bin/bash
#SBATCH --job-name=joint-stage2-5ds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/joint-stage2-%j.out
#SBATCH --error=reasoning-finetuning/logs/joint-stage2-%j.err

echo "Node:   $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Date:   $(date)"

VENV=/mnt/home/amitra/chronos-env
PYTHON=$VENV/bin/python
TORCHRUN=$VENV/bin/torchrun

echo "Python: $($PYTHON --version)"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -4 | tr '\n' '|')"
echo ""

cd /mnt/home/amitra/chronos-forecasting
mkdir -p reasoning-finetuning/logs results

$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29400 \
    reasoning-finetuning/joint_stage2_train.py

echo ""
echo "Done: $(date)"
