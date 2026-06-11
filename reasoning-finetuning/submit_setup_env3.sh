#!/bin/bash
#SBATCH --job-name=setup-env3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/setup-env3-%j.out
#SBATCH --error=reasoning-finetuning/logs/setup-env3-%j.err

echo "Node: $SLURMD_NODENAME"
VENV=/mnt/home/amitra/chronos-env
PIP=$VENV/bin/pip
PYTHON=$VENV/bin/python

echo "Upgrading transformers to 4.57.6 (matches login node working version) ..."
$PIP install "transformers==4.57.6" 2>&1 | tail -5

echo ""
echo "=== FULL IMPORT VERIFICATION ==="
$PYTHON - <<'PYEOF'
import sys
sys.path.insert(0, "/mnt/home/amitra/chronos-forecasting/src")
sys.path.insert(0, "/mnt/home/amitra/chronos-forecasting/reasoning-finetuning")

import torch
import transformers
import datasets
import pyarrow
import numpy as np
import matplotlib
from statsmodels.tsa.seasonal import STL, MSTL
from chronos import ChronosPipeline
from prefix_generator import PrefixGenerator
from prefix_injection import inject_prefix, remove_prefix_hooks
from scripts.step1_stl_decompose import detect_periods, stl_decompose

print("All imports OK")
print(f"  torch:          {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  transformers:   {transformers.__version__}")
print(f"  datasets:       {datasets.__version__}")
print(f"  pyarrow:        {pyarrow.__version__}")
if torch.cuda.is_available():
    print(f"  GPU:            {torch.cuda.get_device_name(0)}")
print("VERIFICATION PASSED")
PYEOF

echo ""
echo "Done: $(date)"
