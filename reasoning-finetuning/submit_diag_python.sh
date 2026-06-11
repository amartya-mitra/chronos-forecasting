#!/bin/bash
#SBATCH --job-name=diag-python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --partition=hpc-high
#SBATCH --output=reasoning-finetuning/logs/diag-python-%j.out
#SBATCH --error=reasoning-finetuning/logs/diag-python-%j.err

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

echo "=== WHICH PYTHON BINARIES EXIST ==="
for p in python python3 python3.12 python3.11 python3.10; do
    loc=$(which $p 2>/dev/null)
    if [ -n "$loc" ]; then
        ver=$($loc --version 2>&1)
        echo "  $p -> $loc  ($ver)"
    else
        echo "  $p -> NOT FOUND"
    fi
done

echo ""
echo "=== CONDA ENVIRONMENTS ==="
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda info --envs 2>&1 || echo "conda info failed"
    echo ""
    echo "Trying 'conda activate base'..."
    conda activate base 2>&1
    echo "  After activate, python3.12 = $(which python3.12 2>/dev/null || echo NOT FOUND)"
    echo "  After activate, python3    = $(which python3    2>/dev/null || echo NOT FOUND)"
else
    echo "  /opt/conda/etc/profile.d/conda.sh NOT FOUND"
fi

echo ""
echo "=== FIND PYTHON3.12 IN COMMON PATHS ==="
for d in /opt/conda/bin /opt/conda/envs/base/bin /usr/bin /usr/local/bin /home/miniconda3/bin /root/miniconda3/bin; do
    if [ -f "$d/python3.12" ]; then
        echo "  FOUND: $d/python3.12"
        $d/python3.12 --version 2>&1
    fi
done

echo ""
echo "=== FIND python3.12 WITH find ==="
find /opt /usr /home -name "python3.12" -type f 2>/dev/null | head -10

echo ""
echo "=== MODULE SYSTEM ==="
module avail python 2>&1 | head -20 || echo "module command not found"

echo ""
echo "=== IMPORT TEST (datasets, torch) ==="
for py in $(find /opt /usr -name "python3*" -type f 2>/dev/null | head -5); do
    echo "Testing $py:"
    $py -c "import torch; import datasets; print('  torch OK, datasets OK')" 2>&1 | head -3
done

echo ""
echo "Done."
