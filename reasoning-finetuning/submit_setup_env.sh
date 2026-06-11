#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/setup-env-%j.out
#SBATCH --error=reasoning-finetuning/logs/setup-env-%j.err

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"

VENV_DIR="/mnt/home/amitra/chronos-env"
CHRONOS_REPO="/mnt/home/amitra/chronos-forecasting"

# --- Check CUDA ---
echo ""
echo "=== CUDA INFO ==="
nvidia-smi 2>&1 | head -15
nvcc --version 2>/dev/null || echo "nvcc not in PATH"
ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -3

# --- Python 3.13 is available ---
PYTHON="/opt/conda/bin/python3.13"
echo ""
echo "=== PYTHON ==="
$PYTHON --version

# --- Create venv on NFS if not already present ---
echo ""
echo "=== CREATING VENV at $VENV_DIR ==="
if [ -f "$VENV_DIR/bin/python" ]; then
    echo "  Venv already exists, skipping creation"
else
    $PYTHON -m venv "$VENV_DIR"
    echo "  Venv created"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

echo "  Venv python: $($VENV_PY --version 2>&1)"

# --- Upgrade pip ---
echo ""
echo "=== UPGRADING PIP ==="
$VENV_PIP install --upgrade pip setuptools wheel 2>&1 | tail -3

# --- Detect CUDA version for torch install ---
CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "  Driver version: $CUDA_VER"

# Determine torch cu-index based on CUDA runtime
# H100s with recent drivers use CUDA 12.x
TORCH_URL="https://download.pytorch.org/whl/cu128"
echo "  Using torch index: $TORCH_URL"

# --- Install torch ---
echo ""
echo "=== INSTALLING TORCH ==="
$VENV_PIP install torch==2.7.0 --index-url $TORCH_URL 2>&1 | tail -5
echo ""
$VENV_PY -c "import torch; print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# --- Install ML packages ---
echo ""
echo "=== INSTALLING ML PACKAGES ==="
$VENV_PIP install \
    transformers==4.52.2 \
    accelerate \
    datasets==3.6.0 \
    pyarrow \
    numpy \
    scikit-learn \
    statsmodels \
    scipy \
    tqdm \
    2>&1 | tail -10

# --- Install chronos from local repo ---
echo ""
echo "=== INSTALLING CHRONOS ==="
$VENV_PIP install -e "$CHRONOS_REPO" 2>&1 | tail -5

# --- Final verification ---
echo ""
echo "=== VERIFICATION ==="
$VENV_PY - <<'PYEOF'
import torch
import transformers
import datasets
import pyarrow
import numpy as np
import statsmodels
import chronos

print(f"  torch:          {torch.__version__}  (cuda={torch.cuda.is_available()})")
print(f"  transformers:   {transformers.__version__}")
print(f"  datasets:       {datasets.__version__}")
print(f"  pyarrow:        {pyarrow.__version__}")
print(f"  numpy:          {np.__version__}")
print(f"  chronos:        OK")
if torch.cuda.is_available():
    print(f"  GPU:            {torch.cuda.get_device_name(0)}")
    t = torch.ones(10).cuda()
    print(f"  CUDA alloc:     OK  ({t.device})")
PYEOF

echo ""
echo "=== DONE: Venv at $VENV_DIR ==="
echo "End: $(date)"
