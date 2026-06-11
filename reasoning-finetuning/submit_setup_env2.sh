#!/bin/bash
#SBATCH --job-name=setup-env2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/setup-env2-%j.out
#SBATCH --error=reasoning-finetuning/logs/setup-env2-%j.err

echo "Node: $SLURMD_NODENAME"
VENV=/mnt/home/amitra/chronos-env
PIP=$VENV/bin/pip

echo "Installing missing packages ..."
$PIP install matplotlib 2>&1 | tail -5

# Also verify the full import chain used by preflight scripts
PYTHON=$VENV/bin/python
$PYTHON - <<'PYEOF'
import torch, transformers, datasets, pyarrow
import numpy as np, pandas as pd, scipy, statsmodels, matplotlib
from statsmodels.tsa.seasonal import STL, MSTL
from chronos import ChronosPipeline
import sys
sys.path.insert(0, "/mnt/home/amitra/chronos-forecasting/src")
sys.path.insert(0, "/mnt/home/amitra/chronos-forecasting/reasoning-finetuning")
from prefix_generator import PrefixGenerator
from prefix_injection import inject_prefix, remove_prefix_hooks
from scripts.step1_stl_decompose import detect_periods, stl_decompose
print("All imports OK")
print(f"  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
PYEOF

echo "Done."
