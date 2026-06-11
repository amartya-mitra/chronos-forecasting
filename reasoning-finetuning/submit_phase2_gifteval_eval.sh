#!/bin/bash
#SBATCH --job-name=phase2-gifteval-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/phase2-gifteval-eval-%j.out
#SBATCH --error=reasoning-finetuning/logs/phase2-gifteval-eval-%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR
source /opt/conda/etc/profile.d/conda.sh
conda activate base

mkdir -p reasoning-finetuning/logs results

/opt/conda/bin/python3.12 \
    reasoning-finetuning/eval_gifteval_trained.py \
    --ckpt_sarsim0   reasoning-finetuning/checkpoints/run-2000steps-2000.pt \
    --ckpt_gifteval  reasoning-finetuning/checkpoints/gifteval-1000steps.pt \
    --dataset        reasoning-finetuning/data/m4hourly-tsf.arrow \
    --output         results/phase2_gifteval_trained_eval.json

EXIT_CODE=$?
echo ""
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
