#!/bin/bash
#SBATCH --job-name=diag-deep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/diag-deep-%j-%N.out
#SBATCH --error=reasoning-finetuning/logs/diag-deep-%j-%N.err

echo "Node: $SLURMD_NODENAME"

echo "=== ALL CONDA ENVS ==="
source /opt/conda/etc/profile.d/conda.sh
conda env list

echo ""
echo "=== CONDA BASE PACKAGES (torch-related) ==="
conda run -n base pip list 2>/dev/null | grep -i "torch\|datasets\|chronos\|transformers\|numpy\|pyarrow" || echo "pip list failed"

echo ""
echo "=== PIP USER PACKAGES ==="
/opt/conda/bin/python3.13 -m pip list --user 2>/dev/null | grep -i "torch\|datasets" | head -10

echo ""
echo "=== NFS / SHARED MOUNTS ==="
df -h | grep -v tmpfs | grep -v overlay

echo ""
echo "=== CHECK HOME DIR FOR VENVS ==="
find /mnt/home -name "pyvenv.cfg" 2>/dev/null | head -5

echo ""
echo "=== GPU INFO ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -4
