#!/bin/bash
#SBATCH --job-name=pflt-fixC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/preflight_fixC_%j.out
#SBATCH --error=reasoning-finetuning/logs/preflight_fixC_%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR
source /opt/conda/etc/profile.d/conda.sh
conda activate base

mkdir -p reasoning-finetuning/logs

/opt/conda/bin/python3.12 \
    reasoning-finetuning/scripts/preflight_fixC.py

EXIT_CODE=$?
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
