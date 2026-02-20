#!/usr/bin/env python3
"""
Training Script for Reasoning Mode Finetuning

This is a wrapper that calls the main training script with proper paths.
"""

import sys
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Change to project root for relative path resolution
os.chdir(PROJECT_ROOT)

# Add PROJECT_ROOT and src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the main training function
from scripts.training.train_reasoning import train
from fire import Fire


def main(config_path: str = None):
    """
    Run reasoning mode training.
    
    Args:
        config_path: Path to YAML config file. Defaults to reasoning-finetuning/configs/default.yaml
    """
    if config_path is None:
        config_path = str(PROJECT_ROOT / "reasoning-finetuning" / "configs" / "default.yaml")
    
    # Convert to absolute path if relative
    config_path = str(Path(config_path).resolve())
    
    print("=" * 60)
    print("REASONING MODE FINETUNING")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)
    
    train(config_path)


if __name__ == "__main__":
    Fire(main)
