#!/bin/bash
#SBATCH --job-name=diag-all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/diag-all-nodes-%j-%N.out
#SBATCH --error=reasoning-finetuning/logs/diag-all-nodes-%j-%N.err

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"

# Find any Python with torch available
GOOD_PY=""
for py in /opt/conda/bin/python3.12 /opt/conda/bin/python3.13 /opt/conda/bin/python3.11 /opt/conda/bin/python3 python3.12 python3.13 python3; do
    if [ -x "$(command -v $py 2>/dev/null || echo $py)" ]; then
        result=$($py -c "import torch; import datasets; print('OK')" 2>/dev/null)
        if [ "$result" = "OK" ]; then
            GOOD_PY=$py
            break
        fi
    fi
done

if [ -n "$GOOD_PY" ]; then
    ver=$($GOOD_PY --version 2>&1)
    torch_ver=$($GOOD_PY -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "STATUS: GOOD  py=$GOOD_PY  ver=$ver  torch=$torch_ver"
else
    echo "STATUS: BAD  — no python found with torch+datasets"
    echo "  Available python binaries:"
    find /opt/conda/bin /usr/bin -name "python3*" -type f 2>/dev/null | while read f; do
        echo "    $f: $($f --version 2>&1)"
    done
fi
