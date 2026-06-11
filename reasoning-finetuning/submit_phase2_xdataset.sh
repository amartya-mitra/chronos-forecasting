#!/bin/bash
#SBATCH --job-name=phase2-xdataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/phase2-xdataset-%j.out
#SBATCH --error=reasoning-finetuning/logs/phase2-xdataset-%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR
source /opt/conda/etc/profile.d/conda.sh
conda activate base

mkdir -p reasoning-finetuning/logs results

/opt/conda/bin/python3.12 \
    reasoning-finetuning/eval_cross_dataset.py \
    --checkpoint reasoning-finetuning/checkpoints/run-2000steps-2000.pt \
    --dataset    reasoning-finetuning/data/m4hourly-tsf.arrow \
    --output     results/phase2_gifteval_eval.json

EXIT_CODE=$?
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
