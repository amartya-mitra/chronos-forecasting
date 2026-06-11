#!/bin/bash
#SBATCH --job-name=prefix-val2k-both
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/prefix_val2k_both_%j.out
#SBATCH --error=reasoning-finetuning/logs/prefix_val2k_both_%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR
source /opt/conda/etc/profile.d/conda.sh
conda activate base

mkdir -p reasoning-finetuning/logs

/opt/conda/bin/python3.12 \
    reasoning-finetuning/scripts/validate_2000steps_both.py

EXIT_CODE=$?
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
