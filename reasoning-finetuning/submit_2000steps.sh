#!/bin/bash
#SBATCH --job-name=prefix-2k-fixC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/prefix_2k_fixC_%j.out
#SBATCH --error=reasoning-finetuning/logs/prefix_2k_fixC_%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR
source /opt/conda/etc/profile.d/conda.sh
conda activate base

mkdir -p reasoning-finetuning/logs

/mnt/home/amitra/.local/bin/torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    reasoning-finetuning/train_2000steps_ddp.py

EXIT_CODE=$?
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
