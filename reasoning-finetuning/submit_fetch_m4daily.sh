#!/bin/bash
#SBATCH --job-name=fetch-m4daily
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=hpc-mid
#SBATCH --output=reasoning-finetuning/logs/fetch-m4daily-%j.out
#SBATCH --error=reasoning-finetuning/logs/fetch-m4daily-%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR
source /opt/conda/etc/profile.d/conda.sh
conda activate base

mkdir -p reasoning-finetuning/logs reasoning-finetuning/data

/opt/conda/bin/python3.12 reasoning-finetuning/scripts/fetch_m4daily.py

EXIT_CODE=$?
echo ""
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
