#!/bin/bash
#SBATCH --job-name=prefix-1k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=hpc-high
#SBATCH --nodelist=slurm-h100-206-129
#SBATCH --output=reasoning-finetuning/logs/prefix_1k_%j.out
#SBATCH --error=reasoning-finetuning/logs/prefix_1k_%j.err

# ── Cluster notes ──────────────────────────────────────────────────────────────
# Nodes:  slurm-h100-206-* (H100, 8 GPUs each, 128 CPUs, ~2TB RAM)
# CUDA:   12.8 (via conda env — no module load required)
# Partition: hpc-high selected for highest scheduling priority.
# Node pinned to slurm-h100-206-129 (fully idle: 8/8 GPUs free, 0 CPUs alloc).
#   Other nodes are at 7–8/8 GPUs occupied — pinning avoids queueing.
# Using 4/8 GPUs; raise --gres=gpu:h100:8 and remove --nodelist to use full node.
# torchrun manages process spawning; ntasks-per-node=1 is correct.
# ──────────────────────────────────────────────────────────────────────────────

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Start time:  $(date)"
echo "Working dir: $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate base

echo "Python:   $(which python3)"
echo "PyTorch:  $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA:     $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "GPUs:     $(nvidia-smi --list-gpus | wc -l) available"
echo ""

# Ensure logs directory exists
mkdir -p reasoning-finetuning/logs

# ── Launch ─────────────────────────────────────────────────────────────────────
# torchrun spawns 4 processes (one per GPU), sets LOCAL_RANK automatically.
# --master_port: pick a free port; change if 29500 is in use on the node.

/mnt/home/amitra/.local/bin/torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    reasoning-finetuning/train_1000steps_ddp.py

EXIT_CODE=$?
echo ""
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
exit $EXIT_CODE
